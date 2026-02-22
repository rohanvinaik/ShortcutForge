#!/usr/bin/env python3
"""Build Tier 1/2 decoder vocabularies from typed IR data.

Coverage measured against the frozen eval set.

Usage:
    uv run python research/scripts/build_decoder_vocab.py -v
    uv run python research/scripts/build_decoder_vocab.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research.src.contracts import CoverageReport
from research.src.data import load_typed_ir_jsonl

# Special tokens shared across all vocabularies
SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
FIRST_REAL_INDEX = len(SPECIAL_TOKENS)  # 4

# Minimum examples for an action to get its own vocab (otherwise use fallback)
MIN_ACTION_EXAMPLES = 5


def _build_tier1_vocab(train_examples):
    """Collect all unique tier1_tokens across training examples.

    Returns a dict of {token: index} starting with special tokens at 0-3
    and real tokens from index 4 onward.
    """
    unique_tokens = set()
    for ex in train_examples:
        unique_tokens.update(ex.tier1_tokens)

    vocab = dict(SPECIAL_TOKENS)
    for idx, token in enumerate(sorted(unique_tokens), start=FIRST_REAL_INDEX):
        vocab[token] = idx
    return vocab


def _build_tier2_vocabs(train_examples):
    """Build per-action tier2 vocabularies and a global fallback.

    For each unique action_name across all Tier2Blocks:
      - Collect all tokens from blocks with that action_name.
      - If the action has >= MIN_ACTION_EXAMPLES observed blocks, build
        a dedicated per-action vocabulary.
      - Otherwise, tokens feed into the global fallback vocab only.

    Returns:
        per_action_vocabs: dict of {action_name: {token: index}}
        global_fallback_vocab: dict of {token: index}
        action_example_counts: Counter of blocks per action_name
    """
    # Gather tokens per action and count examples
    action_tokens: dict[str, set[str]] = {}
    action_example_counts: Counter = Counter()
    all_tier2_tokens: set[str] = set()

    for ex in train_examples:
        for block in ex.tier2_blocks:
            action_example_counts[block.action_name] += 1
            if block.action_name not in action_tokens:
                action_tokens[block.action_name] = set()
            action_tokens[block.action_name].update(block.tokens)
            all_tier2_tokens.update(block.tokens)

    # Build per-action vocabs for actions with enough examples
    per_action_vocabs: dict[str, dict[str, int]] = {}
    for action_name, tokens in sorted(action_tokens.items()):
        if action_example_counts[action_name] >= MIN_ACTION_EXAMPLES:
            vocab = dict(SPECIAL_TOKENS)
            for idx, tok in enumerate(sorted(tokens), start=FIRST_REAL_INDEX):
                vocab[tok] = idx
            per_action_vocabs[action_name] = vocab

    # Build global fallback from ALL tier2 tokens (covers rare actions too)
    global_fallback = dict(SPECIAL_TOKENS)
    for idx, tok in enumerate(sorted(all_tier2_tokens), start=FIRST_REAL_INDEX):
        global_fallback[tok] = idx

    return per_action_vocabs, global_fallback, action_example_counts


def _measure_tier1_coverage(eval_examples, tier1_vocab):
    """Measure how many tier1_tokens in eval are covered by the vocab."""
    total = 0
    covered = 0
    uncovered_set: set[str] = set()

    for ex in eval_examples:
        for tok in ex.tier1_tokens:
            total += 1
            if tok in tier1_vocab:
                covered += 1
            else:
                uncovered_set.add(tok)

    pct = (covered / total * 100.0) if total > 0 else 100.0
    return CoverageReport(
        scope="tier1",
        dataset="eval",
        total_tokens_in_eval=total,
        covered=covered,
        uncovered=sorted(uncovered_set),
        coverage_pct=round(pct, 4),
    )


def _measure_tier2_coverage(eval_examples, per_action_vocabs, global_fallback):
    """Measure how many tier2 tokens in eval blocks are in their action's vocab."""
    total = 0
    covered = 0
    uncovered_set: set[str] = set()

    for ex in eval_examples:
        for block in ex.tier2_blocks:
            vocab = per_action_vocabs.get(block.action_name, global_fallback)
            for tok in block.tokens:
                total += 1
                if tok in vocab:
                    covered += 1
                else:
                    uncovered_set.add(tok)

    pct = (covered / total * 100.0) if total > 0 else 100.0
    return CoverageReport(
        scope="tier2",
        dataset="eval",
        total_tokens_in_eval=total,
        covered=covered,
        uncovered=sorted(uncovered_set),
        coverage_pct=round(pct, 4),
    )


# ---------------------------------------------------------------------------
# Extracted helpers for main()
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build Tier 1/2 decoder vocabularies from typed IR data",
    )
    parser.add_argument(
        "--typed-ir-train",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_train.jsonl",
        help="Typed IR training JSONL",
    )
    parser.add_argument(
        "--typed-ir-eval",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_eval.jsonl",
        help="Typed IR eval JSONL (for coverage)",
    )
    parser.add_argument(
        "--action-catalog",
        type=Path,
        default=PROJECT_ROOT / "references" / "action_catalog.json",
        help="Action catalog JSON (615 actions)",
    )
    parser.add_argument(
        "--param-schemas",
        type=Path,
        default=None,
        help="Optional param schema directory",
    )
    parser.add_argument(
        "--tier1-out",
        type=Path,
        default=PROJECT_ROOT / "references" / "tier1_vocab.json",
        help="Output Tier 1 vocabulary JSON",
    )
    parser.add_argument(
        "--tier2-dir",
        type=Path,
        default=PROJECT_ROOT / "references" / "tier2_vocab",
        help="Output directory for per-action Tier 2 JSONs",
    )
    parser.add_argument(
        "--coverage-out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "vocab_coverage.json",
        help="Output coverage report JSON (tier1 + tier2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Compute vocab without writing files"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def _validate_inputs(train_path: Path, eval_path: Path) -> tuple[list, list]:
    """Load and validate both data files, exit if not found.

    Returns (train_examples, eval_examples).
    """
    if not train_path.exists():
        print(
            f"ERROR: Training file not found: {train_path}\n"
            f"Run build_typed_ir_data.py first to generate typed IR training data:\n"
            f"  uv run python research/scripts/build_typed_ir_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading training data from {train_path} ...")
    train_examples = load_typed_ir_jsonl(train_path)
    print(f"  Loaded {len(train_examples)} training examples.")

    if not eval_path.exists():
        print(
            f"ERROR: Eval file not found: {eval_path}\n"
            f"Run build_typed_ir_data.py first to generate typed IR eval data:\n"
            f"  uv run python research/scripts/build_typed_ir_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading eval data from {eval_path} ...")
    eval_examples = load_typed_ir_jsonl(eval_path)
    print(f"  Loaded {len(eval_examples)} eval examples.")

    return train_examples, eval_examples


def _build_all_vocabs(train_examples) -> tuple[dict, dict, dict, Counter]:
    """Build tier1 vocab, per-action tier2 vocabs, and global fallback.

    Returns (tier1_vocab, per_action_vocabs, global_fallback, action_counts).
    Prints summary info.
    """
    print("\nBuilding Tier 1 vocabulary ...")
    tier1_vocab = _build_tier1_vocab(train_examples)
    print(
        f"  Tier 1 vocab size: {len(tier1_vocab)} "
        f"({len(tier1_vocab) - len(SPECIAL_TOKENS)} real + "
        f"{len(SPECIAL_TOKENS)} special)"
    )

    print("\nBuilding Tier 2 per-action vocabularies ...")
    per_action_vocabs, global_fallback, action_counts = _build_tier2_vocabs(
        train_examples,
    )

    total_actions_seen = len(action_counts)
    dedicated_count = len(per_action_vocabs)
    fallback_count = total_actions_seen - dedicated_count

    print(f"  Unique actions in training data: {total_actions_seen}")
    print(f"  Actions with dedicated vocab (>= {MIN_ACTION_EXAMPLES} examples): {dedicated_count}")
    print(f"  Actions using global fallback (< {MIN_ACTION_EXAMPLES} examples): {fallback_count}")
    print(
        f"  Global fallback vocab size: {len(global_fallback)} "
        f"({len(global_fallback) - len(SPECIAL_TOKENS)} real + "
        f"{len(SPECIAL_TOKENS)} special)"
    )

    return tier1_vocab, per_action_vocabs, global_fallback, action_counts


def _print_per_action_details(per_action_vocabs, action_counts) -> None:
    """Print per-action vocab sizes (verbose mode)."""
    print("\n  Per-action vocab sizes:")
    for action_name, vocab in sorted(per_action_vocabs.items()):
        n_examples = action_counts[action_name]
        print(f"    {action_name}: {len(vocab)} tokens ({n_examples} training blocks)")


def _measure_all_coverage(
    eval_examples, tier1_vocab, per_action_vocabs, global_fallback
) -> tuple[CoverageReport, CoverageReport]:
    """Measure tier1 and tier2 coverage on eval set, print results.

    Returns (tier1_coverage, tier2_coverage).
    """
    print("\nMeasuring coverage on eval set ...")
    tier1_coverage = _measure_tier1_coverage(eval_examples, tier1_vocab)
    tier2_coverage = _measure_tier2_coverage(
        eval_examples,
        per_action_vocabs,
        global_fallback,
    )

    print(
        f"\n  Tier 1 coverage: {tier1_coverage.coverage_pct:.2f}% "
        f"({tier1_coverage.covered}/{tier1_coverage.total_tokens_in_eval} tokens)"
    )
    if tier1_coverage.uncovered:
        print(
            f"    Uncovered tier1 tokens ({len(tier1_coverage.uncovered)}): "
            f"{tier1_coverage.uncovered[:20]}"
            f"{'...' if len(tier1_coverage.uncovered) > 20 else ''}"
        )

    print(
        f"  Tier 2 coverage: {tier2_coverage.coverage_pct:.2f}% "
        f"({tier2_coverage.covered}/{tier2_coverage.total_tokens_in_eval} tokens)"
    )
    if tier2_coverage.uncovered:
        print(
            f"    Uncovered tier2 tokens ({len(tier2_coverage.uncovered)}): "
            f"{tier2_coverage.uncovered[:20]}"
            f"{'...' if len(tier2_coverage.uncovered) > 20 else ''}"
        )

    return tier1_coverage, tier2_coverage


def _write_vocab_artifacts(
    args, tier1_vocab, per_action_vocabs, global_fallback, tier1_coverage, tier2_coverage
) -> None:
    """Handle all file I/O for vocabulary artifacts and coverage report."""
    # Tier 1 vocab
    args.tier1_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.tier1_out, "w") as f:
        json.dump(tier1_vocab, f, indent=2, ensure_ascii=False)
    print(f"\nWrote tier1 vocab -> {args.tier1_out}")

    # Tier 2 per-action vocabs
    args.tier2_dir.mkdir(parents=True, exist_ok=True)
    for action_name, vocab in sorted(per_action_vocabs.items()):
        out_path = args.tier2_dir / f"{action_name}.json"
        with open(out_path, "w") as f:
            json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(per_action_vocabs)} per-action tier2 vocabs -> {args.tier2_dir}/")

    # Global fallback
    fallback_path = args.tier2_dir / "_global_fallback.json"
    with open(fallback_path, "w") as f:
        json.dump(global_fallback, f, indent=2, ensure_ascii=False)
    print(f"Wrote global fallback vocab -> {fallback_path}")

    # Coverage report
    coverage_data = {
        "tier1": tier1_coverage.to_dict(),
        "tier2": tier2_coverage.to_dict(),
    }
    args.coverage_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.coverage_out, "w") as f:
        json.dump(coverage_data, f, indent=2, ensure_ascii=False)
    print(f"Wrote coverage report -> {args.coverage_out}")


def _enforce_gates(tier1_coverage: CoverageReport, tier2_coverage: CoverageReport) -> None:
    """Check coverage thresholds, sys.exit(1) if failed."""
    gate_pass = True
    if tier1_coverage.coverage_pct < 98.0:
        print(f"\nFAIL: Tier 1 coverage {tier1_coverage.coverage_pct:.2f}% < 98% threshold")
        gate_pass = False
    if tier2_coverage.coverage_pct < 95.0:
        print(f"\nFAIL: Tier 2 coverage {tier2_coverage.coverage_pct:.2f}% < 95% threshold")
        gate_pass = False

    if gate_pass:
        print("\nPASS: All coverage gates met.")
    else:
        sys.exit(1)


def main():
    args = _parse_args()
    train_examples, eval_examples = _validate_inputs(args.typed_ir_train, args.typed_ir_eval)
    tier1_vocab, per_action_vocabs, global_fallback, action_counts = _build_all_vocabs(
        train_examples
    )

    if args.verbose:
        _print_per_action_details(per_action_vocabs, action_counts)

    tier1_cov, tier2_cov = _measure_all_coverage(
        eval_examples, tier1_vocab, per_action_vocabs, global_fallback
    )

    if not args.dry_run:
        _write_vocab_artifacts(
            args, tier1_vocab, per_action_vocabs, global_fallback, tier1_cov, tier2_cov
        )
    else:
        print("\n--dry-run: skipping file writes.")

    _enforce_gates(tier1_cov, tier2_cov)


if __name__ == "__main__":
    main()
