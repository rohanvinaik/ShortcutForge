#!/usr/bin/env python3
"""
Build typed IR training/eval data from existing ShortcutDSL JSONL.

Decomposes each DSL example into three-tier representation (TypedIRExample):
Tier 1 structural tokens, Tier 2 parameter blocks, Tier 3 value slots.

Usage:
    uv run python research/scripts/build_typed_ir_data.py -v
    uv run python research/scripts/build_typed_ir_data.py --dry-run --max-examples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src dirs to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research"))

from src.contracts import TypedIRExample  # noqa: E402
from src.ir_decomposer import decompose_dsl_to_typed_ir  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _convert_records(
    records: list[dict[str, Any]],
    verbose: bool,
) -> tuple[list[TypedIRExample], list[str], float]:
    """Convert raw records to TypedIRExamples. Returns (results, failed_ids, elapsed)."""
    results: list[TypedIRExample] = []
    failed_ids: list[str] = []
    total = len(records)
    t0 = time.time()

    for i, record in enumerate(records):
        result = decompose_dsl_to_typed_ir(record, verbose=verbose)
        if result is not None:
            results.append(result)
        else:
            failed_ids.append(record.get("shortcut_id", f"record_{i}"))
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  ... {i + 1}/{total} processed ({rate:.0f} rec/s)")

    return results, failed_ids, time.time() - t0


def _write_results(
    results: list[TypedIRExample],
    output_path: Path | None,
    dry_run: bool,
    success: int,
) -> None:
    """Write converted results to JSONL."""
    if output_path and not dry_run and results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in results:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"  Written: {output_path} ({success} records)")
    elif dry_run:
        print(f"  [DRY RUN] Would write {success} records to {output_path}")


def _print_tier_stats(results: list[TypedIRExample]) -> None:
    """Print tier distribution summary."""
    if not results:
        return
    t1_lens = [len(r.tier1_tokens) for r in results]
    t2_counts = [len(r.tier2_blocks) for r in results]
    t3_counts = [len(r.tier3_slots) for r in results]
    n = len(results)
    print(
        f"\n  Tier1 token counts: min={min(t1_lens)}, max={max(t1_lens)}, mean={sum(t1_lens) / n:.1f}"
    )
    print(
        f"  Tier2 block counts: min={min(t2_counts)}, max={max(t2_counts)}, mean={sum(t2_counts) / n:.1f}"
    )
    print(
        f"  Tier3 slot counts:  min={min(t3_counts)}, max={max(t3_counts)}, mean={sum(t3_counts) / n:.1f}"
    )


def _process_file(
    input_path: Path,
    output_path: Path | None,
    label: str,
    max_examples: int | None,
    verbose: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Process a single JSONL file: load, convert, write.

    Returns stats dict for the conversion report.
    """
    if not input_path.exists():
        logger.warning("%s file not found: %s", label, input_path)
        return {
            "file": str(input_path),
            "label": label,
            "total": 0,
            "success": 0,
            "failed": 0,
            "failure_rate_pct": 0.0,
            "skipped_not_found": True,
        }

    print(f"\n{'=' * 60}")
    print(f"Processing {label}: {input_path}")
    print(f"{'=' * 60}")

    records = _load_jsonl(input_path, max_examples)
    total = len(records)
    print(f"  Loaded {total} records")

    results, failed_ids, elapsed = _convert_records(records, verbose)
    success = len(results)
    failed = len(failed_ids)
    failure_rate = (failed / total * 100) if total > 0 else 0.0

    print(
        f"\n  Results: {success}/{total} succeeded, {failed} failed ({failure_rate:.1f}% failure rate)"
    )
    print(f"  Time: {elapsed:.1f}s")
    if failed_ids and verbose:
        print(f"  Failed IDs (first 20): {failed_ids[:20]}")

    _write_results(results, output_path, dry_run, success)
    _print_tier_stats(results)

    return {
        "file": str(input_path),
        "label": label,
        "total": total,
        "success": success,
        "failed": failed,
        "failure_rate_pct": round(failure_rate, 2),
        "failed_ids": failed_ids[:50],
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build typed IR data from ShortcutDSL JSONL",
    )
    parser.add_argument(
        "--train-in",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "shortcutdsl_train_expanded.jsonl",
        help="Input training JSONL",
    )
    parser.add_argument(
        "--eval-in",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "shortcutdsl_eval.jsonl",
        help="Input eval JSONL (frozen, 100 examples)",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_train.jsonl",
        help="Output typed IR training JSONL",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_eval.jsonl",
        help="Output typed IR eval JSONL",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_conversion_report.json",
        help="Output conversion report JSON",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=5.0,
        help="Max acceptable failure rate percent (default: 5.0)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit to N examples (for testing)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing output",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("build_typed_ir_data.py")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  max_examples: {args.max_examples}")
    print(f"  fail_threshold: {args.fail_threshold}%")

    report: dict[str, Any] = {"splits": []}
    overall_total = 0
    overall_failed = 0

    # Process training data
    train_stats = _process_file(
        input_path=args.train_in,
        output_path=args.train_out,
        label="train",
        max_examples=args.max_examples,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    report["splits"].append(train_stats)
    overall_total += train_stats["total"]
    overall_failed += train_stats["failed"]

    # Process eval data
    eval_stats = _process_file(
        input_path=args.eval_in,
        output_path=args.eval_out,
        label="eval",
        max_examples=args.max_examples,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    report["splits"].append(eval_stats)
    overall_total += eval_stats["total"]
    overall_failed += eval_stats["failed"]

    # Overall stats
    overall_failure_rate = (overall_failed / overall_total * 100) if overall_total > 0 else 0.0
    report["overall"] = {
        "total": overall_total,
        "success": overall_total - overall_failed,
        "failed": overall_failed,
        "failure_rate_pct": round(overall_failure_rate, 2),
    }

    # Write conversion report
    if not args.dry_run:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nConversion report: {args.report_out}")
    else:
        print(f"\n[DRY RUN] Would write report to {args.report_out}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(
        f"OVERALL: {overall_total - overall_failed}/{overall_total} succeeded "
        f"({overall_failure_rate:.1f}% failure rate)"
    )
    print(f"{'=' * 60}")

    # Gate check
    if overall_failure_rate > args.fail_threshold:
        print(
            f"\nFAIL: Failure rate {overall_failure_rate:.1f}% exceeds "
            f"threshold {args.fail_threshold}%"
        )
        sys.exit(1)
    else:
        print(
            f"\nPASS: Failure rate {overall_failure_rate:.1f}% is within "
            f"threshold {args.fail_threshold}%"
        )


if __name__ == "__main__":
    main()
