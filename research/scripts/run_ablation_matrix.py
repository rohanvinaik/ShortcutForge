#!/usr/bin/env python3
"""
Run ablation matrix — sequential experiment execution with PAB comparison.

Loads an ablation matrix config, merges each configuration with a base config,
executes training + evaluation for each, then runs cross-profile comparison.

Usage:
    python research/scripts/run_ablation_matrix.py \
        --matrix-config research/configs/ablation.yaml \
        [--dry-run] [--resume 4.1.3] [--device mps]
"""

from __future__ import annotations

import argparse
import copy
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


def load_matrix_config(path: Path) -> dict:
    """Load and validate ablation matrix YAML."""
    with open(path) as f:
        matrix = yaml.safe_load(f)

    if "configurations" not in matrix:
        raise ValueError("ablation.yaml must have a 'configurations' section")
    return matrix


def merge_config(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of base config.

    Nested dicts are merged recursively; scalar values are replaced.
    """
    result = copy.deepcopy(base)
    _deep_merge(result, overrides)
    return result


def _deep_merge(target: dict, source: dict) -> None:
    """Recursively merge source into target in-place."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def generate_run_configs(
    matrix: dict,
    base_config: dict,
) -> list[tuple[str, str, dict]]:
    """Produce (run_id, description, merged_config) tuples from the matrix.

    Each configuration inherits from base_config with its overrides applied.
    Execution-level overrides (device, seed, max_iterations) are also applied.
    """
    execution = matrix.get("execution", {})
    configs = []

    for run_id, run_def in matrix["configurations"].items():
        overrides = run_def.get("overrides", {})
        merged = merge_config(base_config, overrides)

        # Apply execution-level overrides
        if "device" in execution:
            merged.setdefault("training", {})["device"] = execution["device"]
        if "seed" in execution:
            merged.setdefault("training", {})["seed"] = execution["seed"]
        if "max_iterations" in execution:
            merged.setdefault("training", {})["max_iterations"] = execution["max_iterations"]

        description = run_def.get("description", "")
        configs.append((str(run_id), description, merged))

    return configs


def run_single_experiment(
    run_id: str,
    config: dict,
    device: str,
    dry_run: bool = False,
) -> dict:
    """Train and evaluate a single configuration. Returns result summary."""
    from src.evaluate import evaluate_checkpoint
    from src.trainer import BalancedSashimiTrainer

    seed = config.get("training", {}).get("seed", 42)
    trainer = BalancedSashimiTrainer(
        config=config,
        run_id=run_id,
        device=device,
        seed=seed,
    )
    train_result = trainer.train(dry_run=dry_run)

    eval_result = None
    if not dry_run and train_result.get("status") == "complete" and "checkpoint" in train_result:
        data_cfg = config["data"]
        run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / run_id
        eval_result = evaluate_checkpoint(
            config=config,
            checkpoint_path=Path(train_result["checkpoint"]),
            eval_file=PROJECT_ROOT / data_cfg["typed_ir_eval"],
            output_json=run_dir / f"{run_id}_eval_results.json",
        )

    return {
        "run_id": run_id,
        "training": train_result,
        "evaluation": eval_result,
    }


def run_comparison(results: list[dict], base_config: dict) -> dict | None:
    """Load all PAB profiles from completed runs and produce comparison report."""
    from src.pab_comparison import compare_profiles, export_json, export_markdown
    from src.pab_profile import PABProfile

    profiles = []
    for r in results:
        train = r.get("training", {})
        if train.get("status") != "complete":
            continue
        pab_path = train.get("pab_profile")
        if pab_path and Path(pab_path).exists():
            profiles.append(PABProfile.load(pab_path))

    if len(profiles) < 2:
        print("  Not enough PAB profiles for comparison (need >= 2)")
        return None

    report = compare_profiles(profiles)
    report_data = export_json(report)

    # Save comparison artifacts
    run_dir = PROJECT_ROOT / base_config["logging"]["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = run_dir / "ablation_comparison.json"
    comparison_json.write_text(json.dumps(report_data, indent=2))
    comparison_md = run_dir / "ablation_comparison.md"
    comparison_md.write_text(export_markdown(report))

    print(report.summary_table())
    print(f"  Comparison saved: {comparison_json}")
    return report_data


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ablation matrix runner."""
    parser = argparse.ArgumentParser(
        description="Run ablation matrix with sequential execution",
    )
    parser.add_argument(
        "--matrix-config",
        type=Path,
        required=True,
        help="Path to ablation matrix YAML config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Target device (default: mps)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a specific run ID (skip completed runs before it)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned runs without executing",
    )
    return parser.parse_args()


def _load_base_config(matrix: dict) -> dict:
    """Load the base YAML config referenced by the matrix config."""
    base_config_path = PROJECT_ROOT / "research" / matrix.get("base_config", "configs/base.yaml")
    with open(base_config_path) as f:
        return yaml.safe_load(f)


def _should_skip_run(
    run_id: str,
    skip_until: str | None,
    skipping: bool,
) -> tuple[bool, bool]:
    """Determine whether to skip a run based on resume state.

    Returns (should_skip, still_skipping).
    """
    if not skipping:
        return False, False
    if run_id == skip_until:
        return False, False
    return True, True


def _try_load_cached_result(
    run_id: str,
    config: dict,
    is_resuming: bool,
) -> dict | None:
    """Load a previously saved run summary if it exists and we're not resuming."""
    run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / run_id
    summary_path = run_dir / f"{run_id}_summary.json"
    if summary_path.exists() and not is_resuming:
        with open(summary_path) as f:
            return json.load(f)
    return None


def _execute_and_save(run_id: str, config: dict, device: str) -> dict:
    """Run a single experiment and persist its summary to disk."""
    result = run_single_experiment(run_id, config, device)
    run_dir = PROJECT_ROOT / config["logging"]["run_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / f"{run_id}_summary.json"
    summary_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def _execute_matrix(
    args: argparse.Namespace,
    run_configs: list[tuple[str, str, dict]],
    base_config: dict,
) -> None:
    """Execute all runs in the ablation matrix with optional resume support."""
    skipping = args.resume is not None
    results: list[dict] = []
    start = time.time()

    for i, (run_id, desc, config) in enumerate(run_configs, 1):
        should_skip, skipping = _should_skip_run(run_id, args.resume, skipping)
        if should_skip:
            print(f"[{i}/{len(run_configs)}] Skipping {run_id} (resuming from {args.resume})")
            continue

        cached = _try_load_cached_result(run_id, config, is_resuming=bool(args.resume))
        if cached is not None:
            print(f"[{i}/{len(run_configs)}] {run_id} already completed, skipping")
            results.append(cached)
            continue

        print(f"[{i}/{len(run_configs)}] Running {run_id}: {desc}")
        result = _execute_and_save(run_id, config, args.device)
        results.append(result)
        print(f"  Status: {result['training'].get('status', 'unknown')}")
        print()

    print("=== Cross-Profile Comparison ===")
    run_comparison(results, base_config)

    elapsed = time.time() - start
    print(f"\nAblation matrix complete in {elapsed:.1f}s")


def main():
    args = _parse_args()
    matrix = load_matrix_config(args.matrix_config)
    base_config = _load_base_config(matrix)
    run_configs = generate_run_configs(matrix, base_config)

    print(f"Ablation matrix: {len(run_configs)} configurations")
    print()

    if args.dry_run:
        for run_id, desc, _config in run_configs:
            print(f"  [{run_id}] {desc}")
        print()
        print("Dry run complete. No training executed.")
        return

    _execute_matrix(args, run_configs, base_config)


if __name__ == "__main__":
    main()
