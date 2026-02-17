#!/usr/bin/env python3
"""
Expand training prompts via Claude CLI (uses Claude Max subscription).

Batches multiple descriptions per Claude call to minimize subprocess overhead.
Each call processes ~20 descriptions at once, generating 3 variants each.

LEAKAGE CONTROL: Only augments training examples. Reads split_manifest.json.

Usage:
    python scripts/expand_prompts_claude_cli.py --input-dir training_data/ -v
    python scripts/expand_prompts_claude_cli.py --input-dir training_data/ --batch-size 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

BATCH_PROMPT_TEMPLATE = """For each Apple Shortcut description below, generate 3 alternative ways a user might request it. Vary style (casual, specific, imperative).

Output format: For each description, output its number followed by 3 variants, like:
1:
variant a
variant b
variant c
2:
variant a
variant b
variant c

Descriptions:
{descriptions}

Output ONLY the numbered variants, nothing else."""

MIN_VARIANT_LENGTH = 10
DEFAULT_BATCH_SIZE = 20
DEFAULT_CONCURRENCY = 5


def _parse_batch_response(text: str, n_descriptions: int) -> list[list[str]]:
    """Parse batch response into per-description variant lists."""
    all_variants: list[list[str]] = [[] for _ in range(n_descriptions)]

    current_idx = -1
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check if this is a number header (e.g., "1:", "2:", "1.", "1)")
        stripped = line.rstrip(":.)").strip()
        if stripped.isdigit():
            idx = int(stripped) - 1
            if 0 <= idx < n_descriptions:
                current_idx = idx
            continue

        # Otherwise it's a variant line
        if current_idx >= 0:
            # Strip numbering from variant lines
            cleaned = line.lstrip("0123456789.)-:* ")
            if len(cleaned) >= MIN_VARIANT_LENGTH:
                all_variants[current_idx].append(cleaned)
                # Cap at 3 variants per description
                if len(all_variants[current_idx]) >= 3:
                    current_idx = -1  # Stop collecting for this one

    return all_variants


async def _expand_batch_cli(
    semaphore: asyncio.Semaphore,
    descriptions: list[str],
    model: str = "haiku",
) -> list[list[str]]:
    """Generate variants for a batch of descriptions via claude CLI."""
    # Build numbered descriptions
    numbered = "\n".join(f"{i+1}. {desc}" for i, desc in enumerate(descriptions))
    prompt = BATCH_PROMPT_TEMPLATE.format(descriptions=numbered)

    async with semaphore:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        proc = await asyncio.create_subprocess_exec(
            "claude", "-p", prompt, "--model", model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return [[] for _ in descriptions]

        text = stdout.decode().strip()
        return _parse_batch_response(text, len(descriptions))


async def run_expansion(
    input_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    concurrency: int = DEFAULT_CONCURRENCY,
    model: str = "haiku",
    verbose: bool = False,
) -> dict:
    """Run the full expansion pipeline using batched claude CLI calls."""
    stats = {
        "engine": f"claude-cli-batched ({model})",
        "train_originals": 0,
        "eval_originals": 0,
        "variants_generated": 0,
        "total_expanded": 0,
    }

    # Load manifest for leakage control
    manifest_path = os.path.join(input_dir, "split_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    eval_ids = {k for k, v in manifest["splits"].items() if v == "eval"}

    # Load training data
    train_path = os.path.join(input_dir, "shortcutdsl_train.jsonl")
    train_pairs = []
    with open(train_path) as f:
        for line in f:
            data = json.loads(line)
            if data["shortcut_id"] not in eval_ids:
                train_pairs.append(data)

    stats["train_originals"] = len(train_pairs)

    # Load eval data
    eval_path = os.path.join(input_dir, "shortcutdsl_eval.jsonl")
    eval_count = 0
    with open(eval_path) as f:
        for _ in f:
            eval_count += 1
    stats["eval_originals"] = eval_count

    print(f"  [1/3] Loaded {len(train_pairs)} train, {eval_count} eval examples")

    # Load cache
    cache_path = os.path.join(input_dir, "expansion_cache.jsonl")
    cached_variants: dict[str, list[str]] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    cached_variants[entry["shortcut_id"]] = entry["variants"]
        print(f"  Resuming: {len(cached_variants)} cached expansions found")

    # Find uncached pairs
    uncached_pairs = [p for p in train_pairs if p["shortcut_id"] not in cached_variants]
    print(f"  Need to expand: {len(uncached_pairs)} (skipping {len(cached_variants)} cached)")

    # Expand in batches
    n_batches = (len(uncached_pairs) + batch_size - 1) // batch_size
    print(f"  [2/3] Expanding ({n_batches} batches of ~{batch_size}, {concurrency} concurrent)...", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    t0 = time.monotonic()

    cache_f = open(cache_path, "a")

    # Process concurrent batches
    for mega_start in range(0, len(uncached_pairs), batch_size * concurrency):
        mega_end = min(mega_start + batch_size * concurrency, len(uncached_pairs))
        mega_batch = uncached_pairs[mega_start:mega_end]

        # Split into sub-batches for concurrent processing
        sub_batches = []
        for sb_start in range(0, len(mega_batch), batch_size):
            sb_end = min(sb_start + batch_size, len(mega_batch))
            sub_batches.append(mega_batch[sb_start:sb_end])

        # Launch concurrent CLI calls
        tasks = []
        for sub_batch in sub_batches:
            descriptions = [p["messages"][1]["content"] for p in sub_batch]
            tasks.append(_expand_batch_cli(semaphore, descriptions, model=model))

        results = await asyncio.gather(*tasks)

        # Process results
        for sub_batch, batch_variants in zip(sub_batches, results):
            for pair, variants in zip(sub_batch, batch_variants):
                sid = pair["shortcut_id"]
                cached_variants[sid] = variants
                cache_entry = {"shortcut_id": sid, "variants": variants}
                cache_f.write(json.dumps(cache_entry) + "\n")

        cache_f.flush()

        if verbose:
            done = mega_end
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            remaining = len(uncached_pairs) - done
            eta = remaining / rate if rate > 0 else 0
            total_variants = sum(len(v) for v in cached_variants.values())
            print(
                f"    {done}/{len(uncached_pairs)} expanded: "
                f"{total_variants} variants ({rate:.1f}/s, ETA {eta:.0f}s)",
                flush=True,
            )

    cache_f.close()

    # Build final output
    new_examples = []
    for pair in train_pairs:
        sid = pair["shortcut_id"]
        variants = cached_variants.get(sid, [])
        for variant in variants:
            new_examples.append({
                "shortcut_id": sid,
                "messages": [
                    pair["messages"][0],
                    {"role": "user", "content": variant},
                    pair["messages"][2],
                ],
            })

    stats["variants_generated"] = len(new_examples)

    # Combine originals + variants
    all_train = train_pairs + new_examples
    stats["total_expanded"] = len(all_train)

    # Write expanded training file
    print(f"  [3/3] Writing expanded JSONL...", end=" ", flush=True)
    expanded_path = os.path.join(input_dir, "shortcutdsl_train_expanded.jsonl")
    with open(expanded_path, "w") as f:
        for example in all_train:
            f.write(json.dumps(example) + "\n")

    print(f"done")
    print(f"\n  Output: {expanded_path} ({len(all_train)} examples)")
    print(f"  Eval:   {eval_path} ({eval_count} examples, unchanged)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        prog="expand_prompts_claude_cli",
        description="Expand training prompts via batched Claude CLI calls (uses Max subscription)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="training_data",
        help="Directory with shortcutdsl_train.jsonl and split_manifest.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Descriptions per Claude call (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Concurrent Claude CLI processes (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="haiku",
        help="Claude model to use (default: haiku)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show progress",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_dir = args.input_dir if os.path.isabs(args.input_dir) else str(project_root / args.input_dir)

    print(f"\nShortcutForge: Expanding training prompts (Claude CLI batched, {args.model})...\n")

    stats = asyncio.run(run_expansion(
        input_dir=input_dir,
        batch_size=args.batch_size,
        concurrency=args.concurrency,
        model=args.model,
        verbose=args.verbose,
    ))

    print(f"\n  --- Statistics ---")
    print(f"  Engine:              {stats['engine']}")
    print(f"  Train originals:     {stats['train_originals']}")
    print(f"  Variants generated:  {stats['variants_generated']}")
    print(f"  Total expanded:      {stats['total_expanded']}")
    print(f"  Eval (unchanged):    {stats['eval_originals']}")


if __name__ == "__main__":
    main()
