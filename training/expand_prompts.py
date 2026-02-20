#!/usr/bin/env python3
"""
Expand training prompts via paraphrasing.

For each (description, DSL) pair in the TRAINING set, generates 3 alternative
prompt phrasings. Each variant becomes a new training example with the same DSL target.

Supports two backends:
  --engine claude  — Claude Haiku API (default, needs ANTHROPIC_API_KEY)
  --engine local   — Local Llama 3.1 8B Instruct via MLX-LM (no API needed)

LEAKAGE CONTROL: Only augments training examples. Reads split_manifest.json
to ensure eval shortcut IDs are never augmented.

Usage:
    python scripts/expand_prompts.py --input-dir training_data/ --engine local -v
    python scripts/expand_prompts.py --input-dir training_data/ --engine claude -v
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


EXPANSION_PROMPT = """Given this Apple Shortcut description:
"{description}"

Generate 3 alternative ways a user might request this shortcut. Vary style:
1. Casual/conversational
2. Specific/detailed
3. Imperative/concise

Output exactly 3 lines, one per variant. Nothing else — no numbering, no labels."""

MIN_VARIANT_LENGTH = 10
DEFAULT_BATCH_SIZE = 50
DEFAULT_VARIANTS = 3


def _parse_variants(text: str) -> list[str]:
    """Parse variant lines from LLM response text."""
    variants = []
    for line in text.split("\n"):
        line = line.strip()
        # Strip common numbering patterns
        line = line.lstrip("0123456789.)-: ")
        if len(line) >= MIN_VARIANT_LENGTH:
            variants.append(line)
    return variants[:DEFAULT_VARIANTS]


# ── Claude Backend ────────────────────────────────────────────────────


async def _expand_one_claude(
    client,
    description: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 300,
) -> list[str]:
    """Generate variant prompts using Claude API."""
    prompt = EXPANSION_PROMPT.format(description=description)
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        return _parse_variants(text)
    except Exception as e:
        print(f"    [claude error] {e}", file=sys.stderr)
        return []


async def expand_batch_claude(
    pairs: list[dict],
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = "claude-haiku-4-5-20251001",
    verbose: bool = False,
) -> list[dict]:
    """Expand a batch of training pairs using Claude API."""
    import anthropic

    client = anthropic.AsyncAnthropic()

    new_examples = []
    total = len(pairs)

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = pairs[batch_start:batch_end]

        tasks = []
        for pair in batch:
            description = pair["messages"][1]["content"]
            tasks.append(_expand_one_claude(client, description, model=model))

        results = await asyncio.gather(*tasks)

        for pair, variants in zip(batch, results):
            for variant in variants:
                new_example = {
                    "shortcut_id": pair["shortcut_id"],
                    "messages": [
                        pair["messages"][0],
                        {"role": "user", "content": variant},
                        pair["messages"][2],
                    ],
                }
                new_examples.append(new_example)

        if verbose:
            expanded = sum(len(r) for r in results)
            print(
                f"    Batch {batch_start + 1}-{batch_end}/{total}: +{expanded} variants",
                flush=True,
            )

    return new_examples


# ── Local Backend ─────────────────────────────────────────────────────


def _expand_one_local(
    model,
    tokenizer,
    description: str,
    max_tokens: int = 300,
) -> list[str]:
    """Generate variant prompts using local MLX model."""
    import mlx_lm

    prompt = EXPANSION_PROMPT.format(description=description)

    # Format as Llama 3 chat
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

    try:
        text = mlx_lm.generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            verbose=False,
        )
        return _parse_variants(text)
    except Exception as e:
        print(f"    [local error] {e}", file=sys.stderr)
        return []


def expand_batch_local(
    pairs: list[dict],
    model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    verbose: bool = False,
) -> list[dict]:
    """Expand a batch of training pairs using local MLX model."""
    import mlx_lm

    print(f"  Loading model: {model_path}...", flush=True)
    model, tokenizer = mlx_lm.load(model_path)
    print("  Model loaded.", flush=True)

    new_examples = []
    total = len(pairs)
    t0 = time.monotonic()

    for i, pair in enumerate(pairs):
        description = pair["messages"][1]["content"]
        variants = _expand_one_local(model, tokenizer, description)

        for variant in variants:
            new_example = {
                "shortcut_id": pair["shortcut_id"],
                "messages": [
                    pair["messages"][0],
                    {"role": "user", "content": variant},
                    pair["messages"][2],
                ],
            }
            new_examples.append(new_example)

        if verbose and (i + 1) % 10 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(
                f"    {i + 1}/{total}: +{len(variants)} variants "
                f"({rate:.1f} examples/s, ETA {eta:.0f}s)",
                flush=True,
            )

    return new_examples


# ── Main Pipeline ─────────────────────────────────────────────────────


def run_expansion(
    input_dir: str,
    engine: str = "claude",
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = "",
    model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    verbose: bool = False,
) -> dict:
    """Run the full expansion pipeline.

    Returns stats dict.
    """
    stats = {
        "engine": engine,
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

    # Load eval data (pass through unchanged)
    eval_path = os.path.join(input_dir, "shortcutdsl_eval.jsonl")
    eval_pairs = []
    with open(eval_path) as f:
        for line in f:
            eval_pairs.append(json.loads(line))
    stats["eval_originals"] = len(eval_pairs)

    print(f"  [1/3] Loaded {len(train_pairs)} train, {len(eval_pairs)} eval examples")

    # Expand training prompts
    if engine == "local":
        print("  [2/3] Expanding training prompts (local MLX model)...", flush=True)
        new_examples = expand_batch_local(
            train_pairs, model_path=model_path, verbose=verbose
        )
    else:
        print(
            f"  [2/3] Expanding training prompts (Claude API, {batch_size} concurrent)...",
            flush=True,
        )
        claude_model = model or "claude-haiku-4-5-20251001"
        new_examples = asyncio.run(
            expand_batch_claude(
                train_pairs, batch_size=batch_size, model=claude_model, verbose=verbose
            )
        )

    stats["variants_generated"] = len(new_examples)

    # Combine originals + variants
    all_train = train_pairs + new_examples
    stats["total_expanded"] = len(all_train)

    # Write expanded training file
    print("  [3/3] Writing expanded JSONL...", end=" ", flush=True)
    expanded_path = os.path.join(input_dir, "shortcutdsl_train_expanded.jsonl")
    with open(expanded_path, "w") as f:
        for example in all_train:
            f.write(json.dumps(example) + "\n")

    print("done")
    print(f"\n  Output: {expanded_path} ({len(all_train)} examples)")
    print(f"  Eval:   {eval_path} ({len(eval_pairs)} examples, unchanged)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        prog="expand_prompts",
        description="Expand training prompts via paraphrasing (Claude API or local MLX model)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="training_data",
        help="Directory with shortcutdsl_train.jsonl and split_manifest.json",
    )
    parser.add_argument(
        "--engine",
        choices=["claude", "local"],
        default="local",
        help="Expansion engine: claude (API) or local (MLX model, default)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Concurrent API calls for Claude mode (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Claude model for expansion (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        help="Local MLX model path (default: mlx-community/Meta-Llama-3.1-8B-Instruct-4bit)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-batch/per-example progress",
    )

    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    input_dir = (
        args.input_dir
        if os.path.isabs(args.input_dir)
        else str(project_root / args.input_dir)
    )

    # Check API key for Claude mode
    if args.engine == "claude" and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: ANTHROPIC_API_KEY not set (required for --engine claude)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nShortcutForge: Expanding training prompts ({args.engine} engine)...\n")

    stats = run_expansion(
        input_dir=input_dir,
        engine=args.engine,
        batch_size=args.batch_size,
        model=args.model,
        model_path=args.model_path,
        verbose=args.verbose,
    )

    print("\n  --- Statistics ---")
    print(f"  Engine:              {stats['engine']}")
    print(f"  Train originals:     {stats['train_originals']}")
    print(f"  Variants generated:  {stats['variants_generated']}")
    print(f"  Total expanded:      {stats['total_expanded']}")
    print(f"  Eval (unchanged):    {stats['eval_originals']}")


if __name__ == "__main__":
    main()
