#!/usr/bin/env python3
"""
End-to-end single experiment runner for Balanced Sashimi.

Orchestrates: data check -> train -> evaluate -> (optional) PAB compare.

Usage:
    python research/scripts/run_experiment.py \
        --config research/configs/base.yaml \
        --run-id exp-001 \
        [--compare exp-000] \
        [--dry-run] \
        [--device mps] \
        [--skip-data-check]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

# Ensure research/ root is on sys.path for absolute imports (from src.X)
_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

PROJECT_ROOT = _RESEARCH_ROOT.parent


def check_data_files(config: dict) -> list[str]:
    """Verify required data files exist. Returns list of missing paths."""
    data_cfg = config["data"]
    required = [
        data_cfg["typed_ir_train"],
        data_cfg["typed_ir_eval"],
        data_cfg["tier1_vocab"],
    ]
    missing = []
    for rel_path in required:
        full = PROJECT_ROOT / rel_path
        if not full.exists():
            missing.append(str(full))
    return missing


def run_training(config: dict, run_id: str, device: str, seed: int, dry_run: bool) -> dict:
    """Instantiate trainer and run training loop."""
    from src.trainer import BalancedSashimiTrainer

    trainer = BalancedSashimiTrainer(
        config=config,
        run_id=run_id,
        device=device,
        seed=seed,
    )
    return trainer.train(dry_run=dry_run)


def run_evaluation(
    config: dict,
    checkpoint_path: Path,
    eval_file: Path,
    output_json: Path,
) -> dict:
    """Evaluate a checkpoint against the eval set."""
    from src.evaluate import evaluate_checkpoint

    return evaluate_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        eval_file=eval_file,
        output_json=output_json,
    )


def run_comparison(
    run_dir: Path,
    current_run_id: str,
    compare_run_id: str,
) -> dict | None:
    """Compare PAB profiles between two runs."""
    from src.pab_comparison import compare_profiles, export_json, export_markdown
    from src.pab_profile import PABProfile

    current_profile_path = run_dir / f"{current_run_id}_pab_profile.json"
    # Look for comparison profile in sibling run directory
    compare_dir = run_dir.parent / compare_run_id
    compare_profile_path = compare_dir / f"{compare_run_id}_pab_profile.json"

    if not current_profile_path.exists():
        print(f"  No PAB profile found for {current_run_id}, skipping comparison")
        return None
    if not compare_profile_path.exists():
        print(f"  No PAB profile found for {compare_run_id}, skipping comparison")
        return None

    profiles = [
        PABProfile.load(current_profile_path),
        PABProfile.load(compare_profile_path),
    ]
    report = compare_profiles(profiles)

    # Save comparison artifacts
    report_json_path = run_dir / f"{current_run_id}_comparison.json"
    report_json_path.write_text(json.dumps(export_json(report), indent=2))

    report_md_path = run_dir / f"{current_run_id}_comparison.md"
    report_md_path.write_text(export_markdown(report))

    print(f"  Comparison saved: {report_json_path}")
    print(report.summary_table())
    return export_json(report)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run a complete Balanced Sashimi experiment",
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment YAML config")
    parser.add_argument("--run-id", type=str, required=True, help="Unique identifier for this run")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Target device (default: mps)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Run one batch and exit")
    parser.add_argument("--compare", type=str, default=None, help="Run ID to compare against")
    parser.add_argument("--skip-data-check", action="store_true", help="Skip data verification")
    return parser.parse_args()


def _write_summary(
    run_dir: Path,
    run_id: str,
    config_path: Path,
    train_result: dict,
    eval_result: dict | None,
    elapsed: float,
) -> None:
    """Write experiment summary JSON."""
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_id": run_id,
        "config": str(config_path),
        "training": train_result,
        "evaluation": eval_result,
        "elapsed_s": round(elapsed, 1),
    }
    summary_path = run_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")


def main():
    args = _parse_args()
    config = yaml.safe_load(open(args.config))
    start = time.time()

    print(f"=== Experiment: {args.run_id} ===")
    print(f"Config: {args.config}\nDevice: {args.device}\n")

    # Phase 1: Data check
    if not args.skip_data_check:
        print("[1/3] Checking data files...")
        missing = check_data_files(config)
        if missing:
            print("ERROR: Missing required data files:")
            for p in missing:
                print(f"  - {p}")
            sys.exit(1)
        print("  All data files present.")
    else:
        print("[1/3] Skipping data check.")
    print()

    # Phase 2: Training
    print("[2/3] Training...")
    train_result = run_training(config, args.run_id, args.device, args.seed, args.dry_run)
    print(f"  Status: {train_result['status']}")
    if train_result["status"] not in ("complete", "dry_run"):
        print(f"  Training ended early: {train_result}")
        sys.exit(1)
    print()

    # Phase 3: Evaluation
    eval_result = None
    if not args.dry_run and "checkpoint" in train_result:
        print("[3/3] Evaluating...")
        data_cfg = config["data"]
        run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / args.run_id
        eval_result = run_evaluation(
            config=config,
            checkpoint_path=Path(train_result["checkpoint"]),
            eval_file=PROJECT_ROOT / data_cfg["typed_ir_eval"],
            output_json=run_dir / f"{args.run_id}_eval_results.json",
        )
        print(f"  Tier1 exact match: {eval_result['tier1_exact_match_rate']:.2%}")
        print(f"  Roundtrip success: {eval_result['roundtrip_success_rate']:.2%}")
    else:
        print("[3/3] Skipping evaluation (dry run).")
    print()

    # Optional: PAB comparison
    if args.compare and not args.dry_run:
        print(f"[+] Comparing with {args.compare}...")
        run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / args.run_id
        run_comparison(run_dir, args.run_id, args.compare)
        print()

    elapsed = time.time() - start
    print(f"=== Experiment complete in {elapsed:.1f}s ===")
    run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / args.run_id
    _write_summary(run_dir, args.run_id, args.config, train_result, eval_result, elapsed)


if __name__ == "__main__":
    main()
