#!/usr/bin/env python3
"""
Build hard negative bank from distillation logs and linter repair taxonomy.

Negative bank entries are contrastive pairs (positive, negative) with error
tags for margin loss training.

Usage:
    uv run python research/scripts/build_negative_bank.py -v
    uv run python research/scripts/build_negative_bank.py --dry-run
    uv run python research/scripts/build_negative_bank.py --fingerprint path/to/fingerprint.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _RESEARCH_ROOT.parent

# ── Imports from project ──────────────────────────────────────────
# Research root FIRST so `src.*` resolves to research/src/ not ShortcutForge/src/
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsl_linter import ActionResolver  # noqa: E402

from src.contracts import NegativeBankEntry  # noqa: E402
from src.negative_bank_builder import (  # noqa: E402
    MutationWeights,
    generate_distillation_negatives,
    generate_linter_negatives,
    generate_synthetic_negatives,
)

# ---------------------------------------------------------------------------
# Summary and save
# ---------------------------------------------------------------------------


def _summarize_and_save(
    entries: list[NegativeBankEntry],
    source_counts: dict[str, int],
    out: Path,
    min_triples: int,
    dry_run: bool,
) -> None:
    """Print summary, check gate, and optionally write output."""
    total = len(entries)
    print("\n=== Summary ===")
    print(f"  Distillation errors:  {source_counts['distillation']:>6}")
    print(f"  Synthetic mutations:  {source_counts['synthetic_mutation']:>6}")
    print(f"  Linter taxonomy:      {source_counts['linter_repair']:>6}")
    print(f"  Total:                {total:>6}")

    if total < min_triples:
        print(f"\n  WARNING: {total} entries < {min_triples} minimum. Gate NOT met.")
        sys.exit(1)
    else:
        print(f"\n  Gate: {total} >= {min_triples} minimum. PASSED.")

    if dry_run:
        print("\n  --dry-run: skipping write")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        print(f"\n  Wrote {total} entries to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build hard negative bank for margin loss training",
    )
    parser.add_argument(
        "--distillation-log",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "distillation_log.jsonl",
        help="Distillation log JSONL (raw->canonicalized pairs)",
    )
    parser.add_argument(
        "--typed-ir-train",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_train.jsonl",
        help="Typed IR training JSONL (for positive examples)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "hard_negative_bank.jsonl",
        help="Output negative bank JSONL",
    )
    parser.add_argument(
        "--synthetic-mutations-per-example",
        type=int,
        default=3,
        help="Number of synthetic negative mutations per example",
    )
    parser.add_argument(
        "--min-triples",
        type=int,
        default=3000,
        help="Minimum entries for Phase 0 acceptance",
    )
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=None,
        help="Path to fingerprint JSON for weighted mutation selection",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Count without writing output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    random.seed(args.seed)

    # Derive mutation weights from fingerprint if provided
    weights: MutationWeights | None = None
    if args.fingerprint:
        from src.behavioral_fingerprint import BehavioralFingerprint
        from src.negative_bank_builder import weights_from_fingerprint

        fp = BehavioralFingerprint.load(args.fingerprint)
        weights = weights_from_fingerprint(fp)

    resolver = ActionResolver()
    canonical_map: dict[str, str] = dict(resolver._canonical_map)

    # Generate entries from all three sources
    distillation_entries = generate_distillation_negatives(args.distillation_log)
    synthetic_entries = generate_synthetic_negatives(
        args.typed_ir_train,
        args.synthetic_mutations_per_example,
        args.verbose,
        weights=weights,
    )
    linter_entries = generate_linter_negatives(canonical_map)

    entries = distillation_entries + synthetic_entries + linter_entries
    source_counts = {
        "distillation": len(distillation_entries),
        "synthetic_mutation": len(synthetic_entries),
        "linter_repair": len(linter_entries),
    }

    _summarize_and_save(entries, source_counts, args.out, args.min_triples, args.dry_run)


if __name__ == "__main__":
    main()
