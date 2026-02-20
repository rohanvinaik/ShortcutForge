#!/usr/bin/env python3
"""
Build training data for ShortcutForge fine-tuning.

Converts Cassinelli .shortcut files → validated DSL → JSONL training pairs.

Pipeline:
  1. Load cassinelli_shortcuts_library.json (descriptions + names)
  2. Match library entries to downloaded/*.shortcut files via normalized name
  3. Convert each .shortcut → DSL via plist_to_dsl
  4. Validate each DSL through parse_dsl() AND validate_ir()
  5. Filter by token count (max_tokens budget)
  6. Deterministic train/eval split by shortcut ID
  7. Output JSONL files + split manifest

Usage:
    python scripts/build_training_data.py --output-dir training_data/ --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dsl_parser import parse_dsl
from dsl_validator import validate_ir
from plist_to_dsl import shortcut_file_to_dsl_safe

# ── Constants ─────────────────────────────────────────────────────────

SYSTEM_MESSAGE = (
    "You are ShortcutForge, an expert Apple Shortcuts DSL generator. "
    "Given a natural language description, output ONLY the DSL code. "
    'Start with SHORTCUT "Name" and end with ENDSHORTCUT. '
    "Include all necessary ACTION, SET, "
    "IF/ENDIF, MENU/ENDMENU, REPEAT/ENDREPEAT, and FOREACH/ENDFOREACH blocks. "
    "Do not include any explanation or commentary."
)

DEFAULT_MAX_TOKENS = 3800
DEFAULT_EVAL_SIZE = 100
MIN_DESCRIPTION_LENGTH = 10


# ── Normalization ─────────────────────────────────────────────────────


def _normalize_name(name: str) -> str:
    """Normalize a shortcut name for matching: lowercase, strip non-alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _normalize_filename(filename: str) -> str:
    """Normalize a .shortcut filename for matching."""
    stem = Path(filename).stem
    return _normalize_name(stem)


# ── Token Counting ────────────────────────────────────────────────────

_tokenizer = None


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base, close to Llama tokenizer)."""
    global _tokenizer
    if _tokenizer is None:
        try:
            import tiktoken

            _tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            # Fallback: rough estimate (4 chars per token)
            return len(text) // 4
    return len(_tokenizer.encode(text))


# ── Deterministic Split ───────────────────────────────────────────────


def _deterministic_split(
    shortcut_ids: list[str],
    eval_size: int,
    seed: str = "shortcutforge-v1",
) -> dict[str, str]:
    """Assign each shortcut ID to 'train' or 'eval' deterministically.

    Uses hash-based assignment for reproducibility without random state.
    Stratifies by DSL length bucket isn't needed here — we just pick
    eval_size IDs with the lowest hash values for a stable random subset.
    """

    def _hash_id(sid: str) -> str:
        return hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest()

    sorted_ids = sorted(shortcut_ids, key=_hash_id)
    split = {}
    for i, sid in enumerate(sorted_ids):
        split[sid] = "eval" if i < eval_size else "train"
    return split


# ── Main Pipeline ─────────────────────────────────────────────────────


def build_training_data(
    library_path: str,
    downloaded_dir: str,
    output_dir: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    eval_size: int = DEFAULT_EVAL_SIZE,
    validate: bool = True,
    verbose: bool = False,
) -> dict:
    """Build training data from Cassinelli shortcuts.

    Returns:
        Statistics dict with counts for each pipeline stage.
    """
    stats = {
        "library_entries": 0,
        "downloaded_files": 0,
        "matched": 0,
        "conversion_success": 0,
        "conversion_failed": 0,
        "parse_success": 0,
        "parse_failed": 0,
        "validate_success": 0,
        "validate_failed": 0,
        "validate_warnings": 0,
        "token_filtered": 0,
        "description_filtered": 0,
        "train_count": 0,
        "eval_count": 0,
    }

    # ── 1. Load library ──
    print("  [1/7] Loading library...", end=" ", flush=True)
    with open(library_path) as f:
        library = json.load(f)
    stats["library_entries"] = len(library)
    print(f"done ({len(library)} entries)")

    # ── 2. Index downloaded files ──
    print("  [2/7] Indexing downloaded files...", end=" ", flush=True)
    downloaded = Path(downloaded_dir)
    file_map: dict[str, Path] = {}  # normalized_name -> file path
    for fpath in downloaded.glob("*.shortcut"):
        norm = _normalize_filename(fpath.name)
        file_map[norm] = fpath
    stats["downloaded_files"] = len(file_map)
    print(f"done ({len(file_map)} files)")

    # ── 3. Match library entries to files ──
    print("  [3/7] Matching entries to files...", end=" ", flush=True)
    matched_pairs: list[dict] = []  # {slug, name, description, file_path}

    for entry in library:
        slug = entry.get("slug", "")
        name = entry.get("name", entry.get("title", ""))
        description = entry.get("description", "")

        # Try matching by slug, then by name
        norm_slug = _normalize_name(slug)
        norm_name = _normalize_name(name)

        file_path = file_map.get(norm_slug) or file_map.get(norm_name)
        if file_path is None:
            # Try with underscores replacing hyphens (filename convention)
            alt_slug = slug.replace("-", "_")
            norm_alt = _normalize_name(alt_slug)
            file_path = file_map.get(norm_alt)

        if file_path is not None:
            matched_pairs.append(
                {
                    "slug": slug,
                    "name": name,
                    "description": description,
                    "file_path": str(file_path),
                }
            )

    stats["matched"] = len(matched_pairs)
    match_rate = len(matched_pairs) / len(library) * 100 if library else 0
    print(f"done ({len(matched_pairs)} matched, {match_rate:.1f}%)")

    # ── 4. Convert to DSL + Parse + Validate ──
    print("  [4/7] Converting to DSL + validating...", flush=True)
    valid_pairs: list[dict] = []  # {slug, name, description, dsl_text}

    for i, pair in enumerate(matched_pairs):
        if verbose and (i + 1) % 100 == 0:
            print(f"         {i + 1}/{len(matched_pairs)}...", flush=True)

        # Convert .shortcut → DSL
        dsl_text, error = shortcut_file_to_dsl_safe(pair["file_path"])
        if dsl_text is None:
            stats["conversion_failed"] += 1
            if verbose:
                print(f"         SKIP (conversion): {pair['slug']}: {error}")
            continue
        stats["conversion_success"] += 1

        # Inject correct name (replace "Untitled" header if present)
        escaped_name = pair["name"].replace('"', '\\"')
        # Replace the first SHORTCUT "..." line with the correct name
        dsl_text = re.sub(
            r'^SHORTCUT\s+"[^"]*"',
            f'SHORTCUT "{escaped_name}"',
            dsl_text,
            count=1,
        )

        # Parse validation
        try:
            ir = parse_dsl(dsl_text)
            stats["parse_success"] += 1
        except Exception as e:
            stats["parse_failed"] += 1
            if verbose:
                err_msg = str(e)[:100]
                print(f"         SKIP (parse): {pair['slug']}: {err_msg}")
            continue

        # Semantic validation
        if validate:
            validation = validate_ir(ir)
            if validation.errors:
                stats["validate_failed"] += 1
                if verbose:
                    for err in validation.errors[:2]:
                        print(
                            f"         SKIP (validate): {pair['slug']}: {err.message[:80]}"
                        )
                continue
            stats["validate_success"] += 1
            if validation.warnings:
                stats["validate_warnings"] += 1
        else:
            stats["validate_success"] += 1

        # Description quality filter
        desc = pair["description"].strip()
        if len(desc) < MIN_DESCRIPTION_LENGTH:
            stats["description_filtered"] += 1
            if verbose:
                print(f"         SKIP (short desc): {pair['slug']}: {len(desc)} chars")
            continue
        # Skip descriptions that are just the name
        if _normalize_name(desc) == _normalize_name(pair["name"]):
            stats["description_filtered"] += 1
            if verbose:
                print(f"         SKIP (desc=name): {pair['slug']}")
            continue

        # Token count filter
        token_count = _count_tokens(dsl_text)
        if token_count > max_tokens:
            stats["token_filtered"] += 1
            if verbose:
                print(
                    f"         SKIP (tokens): {pair['slug']}: {token_count} tokens > {max_tokens}"
                )
            continue

        valid_pairs.append(
            {
                "slug": pair["slug"],
                "name": pair["name"],
                "description": desc,
                "dsl_text": dsl_text,
                "token_count": token_count,
            }
        )

    print(f"         done ({len(valid_pairs)} valid pairs)")

    # ── 5. Deterministic split ──
    print("  [5/7] Splitting train/eval...", end=" ", flush=True)
    actual_eval_size = min(eval_size, len(valid_pairs) // 5)  # Cap at 20%
    slugs = [p["slug"] for p in valid_pairs]
    split_map = _deterministic_split(slugs, actual_eval_size)

    train_pairs = [p for p in valid_pairs if split_map[p["slug"]] == "train"]
    eval_pairs = [p for p in valid_pairs if split_map[p["slug"]] == "eval"]
    stats["train_count"] = len(train_pairs)
    stats["eval_count"] = len(eval_pairs)
    print(f"done (train={len(train_pairs)}, eval={len(eval_pairs)})")

    # ── 6. Write JSONL files ──
    print("  [6/7] Writing JSONL files...", end=" ", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "shortcutdsl_train.jsonl")
    eval_path = os.path.join(output_dir, "shortcutdsl_eval.jsonl")
    manifest_path = os.path.join(output_dir, "split_manifest.json")

    def _pair_to_jsonl(pair: dict) -> dict:
        return {
            "shortcut_id": pair["slug"],
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": pair["description"]},
                {"role": "assistant", "content": pair["dsl_text"]},
            ],
        }

    with open(train_path, "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(_pair_to_jsonl(pair)) + "\n")

    with open(eval_path, "w") as f:
        for pair in eval_pairs:
            f.write(json.dumps(_pair_to_jsonl(pair)) + "\n")

    print("done")

    # ── 7. Write split manifest ──
    print("  [7/7] Writing split manifest...", end=" ", flush=True)
    manifest = {
        "version": 1,
        "seed": "shortcutforge-v1",
        "eval_size": actual_eval_size,
        "total_valid": len(valid_pairs),
        "splits": split_map,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print("done")

    # ── Summary ──
    print("\n  Output files:")
    print(f"    {train_path} ({len(train_pairs)} examples)")
    print(f"    {eval_path} ({len(eval_pairs)} examples)")
    print(f"    {manifest_path}")

    return stats


def print_stats(stats: dict):
    """Print a statistics summary."""
    print("\n  --- Statistics ---")
    print(f"  Library entries:       {stats['library_entries']}")
    print(f"  Downloaded files:      {stats['downloaded_files']}")
    print(
        f"  Matched:               {stats['matched']} ({stats['matched'] / max(stats['library_entries'], 1) * 100:.1f}%)"
    )
    print(f"  Conversion success:    {stats['conversion_success']}")
    print(f"  Conversion failed:     {stats['conversion_failed']}")
    print(f"  Parse success:         {stats['parse_success']}")
    print(f"  Parse failed:          {stats['parse_failed']}")
    print(f"  Validate success:      {stats['validate_success']}")
    print(f"  Validate failed:       {stats['validate_failed']}")
    print(f"  Validate with warns:   {stats['validate_warnings']}")
    print(f"  Description filtered:  {stats['description_filtered']}")
    print(f"  Token filtered:        {stats['token_filtered']}")
    print("  ──────────────────────")
    print(f"  Train examples:        {stats['train_count']}")
    print(f"  Eval examples:         {stats['eval_count']}")
    print(f"  Total valid:           {stats['train_count'] + stats['eval_count']}")


def main():
    parser = argparse.ArgumentParser(
        prog="build_training_data",
        description="Build training data for ShortcutForge fine-tuning",
    )
    parser.add_argument(
        "--library",
        type=str,
        default="references/cassinelli_shortcuts_library.json",
        help="Path to Cassinelli library JSON",
    )
    parser.add_argument(
        "--downloaded-dir",
        type=str,
        default="downloaded",
        help="Path to downloaded .shortcut files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="training_data",
        help="Output directory for JSONL files (default: training_data/)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max DSL token count (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=DEFAULT_EVAL_SIZE,
        help=f"Number of eval examples (default: {DEFAULT_EVAL_SIZE})",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip semantic validation (parse-only)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-file skip reasons",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    library = (
        args.library
        if os.path.isabs(args.library)
        else str(project_root / args.library)
    )
    downloaded = (
        args.downloaded_dir
        if os.path.isabs(args.downloaded_dir)
        else str(project_root / args.downloaded_dir)
    )
    output = (
        args.output_dir
        if os.path.isabs(args.output_dir)
        else str(project_root / args.output_dir)
    )

    print("\nShortcutForge: Building training data...\n")

    stats = build_training_data(
        library_path=library,
        downloaded_dir=downloaded,
        output_dir=output,
        max_tokens=args.max_tokens,
        eval_size=args.eval_size,
        validate=not args.no_validate,
        verbose=args.verbose,
    )

    print_stats(stats)


if __name__ == "__main__":
    main()
