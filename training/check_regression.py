#!/usr/bin/env python3
"""
No-regression gate script for ShortcutForge.

Compares current eval results against a frozen baseline snapshot
and asserts that no key metrics have regressed.

Usage:
    # Check against default baseline:
    python scripts/check_regression.py

    # Check specific files:
    python scripts/check_regression.py \\
        --baseline training_data/baseline_snapshot.json \\
        --results training_data/eval_results.json

    # Verbose output:
    python scripts/check_regression.py -v

Exit codes:
    0 = PASS (no regressions)
    1 = FAIL (regressions detected)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_regression(
    baseline_path: str,
    results_path: str,
    category_tolerance: int = 1,
    verbose: bool = False,
) -> bool:
    """Compare eval results against baseline snapshot.

    Args:
        baseline_path: Path to baseline_snapshot.json
        results_path: Path to eval_results.json (current run)
        category_tolerance: Max allowed increase per failure category
        verbose: Print detailed comparison

    Returns:
        True if no regressions, False if regressions found.
    """
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(results_path) as f:
        results = json.load(f)

    regressions: list[str] = []
    improvements: list[str] = []

    # --- Eval set mismatch check ---
    baseline_n = baseline.get("n_examples", 0)
    results_n = results.get("total", results.get("n_examples", 0))
    baseline_eval_file = baseline.get("eval_file", "")
    results_eval_file = results.get("eval_file", "")

    if baseline_n and results_n and baseline_n != results_n:
        print(f"  ⚠ WARNING: Eval set size mismatch — baseline has {baseline_n} examples, results have {results_n}.")
        print(f"    Baseline eval: {baseline_eval_file}")
        print(f"    Results eval:  {results_eval_file or '(not recorded)'}")
        print(f"    Regression check may not be meaningful with different eval sets.")
        print(f"    Re-run the frozen eval to get a valid comparison.\n")

    # --- Metric comparisons ---
    # Map from results field names to baseline metric names
    metric_checks = [
        ("parse_rate", "parse_rate", "Parse rate"),
        ("validate_rate", "validate_strict_rate", "Validate (strict) rate"),
        ("compile_rate", "compile_strict_rate", "Compile (strict) rate"),
    ]

    baseline_metrics = baseline.get("metrics", {})

    for results_key, baseline_key, label in metric_checks:
        current = results.get(results_key, 0)
        expected = baseline_metrics.get(baseline_key, 0)

        if verbose:
            delta = current - expected
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"  {label}: {current}% (baseline: {expected}%) {arrow} {delta:+.1f}")

        if current < expected:
            regressions.append(
                f"{label}: {current}% < baseline {expected}% (regression of {expected - current:.1f}%)"
            )
        elif current > expected:
            improvements.append(
                f"{label}: {current}% > baseline {expected}% (improvement of {current - expected:.1f}%)"
            )

    # --- Failure category checks ---
    baseline_categories = baseline.get("failure_categories", {})
    current_categories = results.get("failure_categories", {})

    if verbose:
        print(f"\n  Failure categories:")

    all_cats = set(list(baseline_categories.keys()) + list(current_categories.keys()))
    for cat in sorted(all_cats):
        baseline_count = baseline_categories.get(cat, 0)
        current_count = current_categories.get(cat, 0)

        if verbose:
            delta = current_count - baseline_count
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"    {cat}: {current_count} (baseline: {baseline_count}) {arrow} {delta:+d}")

        if current_count > baseline_count + category_tolerance:
            regressions.append(
                f"Failure category '{cat}': {current_count} > baseline {baseline_count} + tolerance {category_tolerance}"
            )

    # --- Scenario score checks ---
    baseline_scenarios = baseline.get("scenario_scores", {})
    current_scenarios = results.get("scenario_scores", {})

    if baseline_scenarios or current_scenarios:
        if verbose:
            print(f"\n  Scenario scores:")

        all_scenarios = set(list(baseline_scenarios.keys()) + list(current_scenarios.keys()))
        for scenario in sorted(all_scenarios):
            baseline_score = baseline_scenarios.get(scenario, 0.0)
            current_score = current_scenarios.get(scenario, 0.0)

            if verbose:
                delta = current_score - baseline_score
                arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
                print(f"    {scenario}: {current_score:.2f} (baseline: {baseline_score:.2f}) {arrow} {delta:+.2f}")

            if current_score < baseline_score - 0.05:  # 5% tolerance for scenario scores
                regressions.append(
                    f"Scenario '{scenario}': {current_score:.2f} < baseline {baseline_score:.2f} (regression of {baseline_score - current_score:.2f})"
                )
            elif current_score > baseline_score + 0.05:
                improvements.append(
                    f"Scenario '{scenario}': {current_score:.2f} > baseline {baseline_score:.2f} (improvement of {current_score - baseline_score:.2f})"
                )

    # --- Report ---
    print()
    if improvements:
        print("  Improvements:")
        for imp in improvements:
            print(f"    ✓ {imp}")

    if regressions:
        print("  REGRESSIONS DETECTED:")
        for reg in regressions:
            print(f"    ✗ {reg}")
        print(f"\n  RESULT: FAIL ({len(regressions)} regression(s))")
        return False
    else:
        print(f"  RESULT: PASS (no regressions)")
        return True


def main():
    parser = argparse.ArgumentParser(
        prog="check_regression",
        description="Check eval results against baseline snapshot for regressions",
    )

    project_root = Path(__file__).resolve().parent.parent

    parser.add_argument(
        "--baseline",
        type=str,
        default=str(project_root / "training_data" / "baseline_snapshot.json"),
        help="Path to baseline snapshot JSON",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(project_root / "training_data" / "eval_results.json"),
        help="Path to current eval results JSON",
    )
    parser.add_argument(
        "--category-tolerance",
        type=int,
        default=1,
        help="Max allowed increase per failure category (default: 1)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed comparison",
    )

    args = parser.parse_args()

    print(f"\nShortcutForge: Regression Check\n")
    print(f"  Baseline: {args.baseline}")
    print(f"  Results:  {args.results}")
    print()

    passed = check_regression(
        baseline_path=args.baseline,
        results_path=args.results,
        category_tolerance=args.category_tolerance,
        verbose=args.verbose,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
