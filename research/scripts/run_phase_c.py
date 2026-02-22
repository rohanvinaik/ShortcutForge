#!/usr/bin/env python3
"""Phase C: Within-architecture encoder comparison.

Uses top Phase B models as encoders in the Balanced Sashimi pipeline.
Runs an encoder x ablation matrix to test encoder-architecture interaction.

Usage:
    python research/scripts/run_phase_c.py [--dry-run] [--config path] [--device mps]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

PROJECT_ROOT = _RESEARCH_ROOT.parent


def _deep_merge(target: dict, source: dict) -> None:
    """Recursively merge source into target in-place."""
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def merge_config(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of base config."""
    result = copy.deepcopy(base)
    _deep_merge(result, overrides)
    return result


def load_phase_c_config(path: Path) -> dict:
    """Load and validate Phase C configuration."""
    with open(path) as f:
        config = yaml.safe_load(f)
    if "selected_encoders" not in config:
        raise ValueError("phase_c.yaml must have a 'selected_encoders' section")
    if "ablation_configs" not in config:
        raise ValueError("phase_c.yaml must have an 'ablation_configs' section")
    return config


def build_run_matrix(
    phase_config: dict,
    base_config: dict,
    encoder_filter: list[str] | None = None,
) -> list[tuple[str, str, str, dict]]:
    """Generate (run_id, encoder_id, ablation_id, merged_config) tuples.

    Returns the full encoder x ablation cross-product, optionally filtered
    to a subset of encoders.
    """
    execution = phase_config.get("execution", {})
    encoders = phase_config["selected_encoders"]
    ablations = phase_config["ablation_configs"]

    if encoder_filter:
        encoders = [e for e in encoders if e["model_id"] in encoder_filter]

    runs = []
    for encoder_entry in encoders:
        encoder_id = encoder_entry["model_id"]
        for ablation_id, ablation_def in ablations.items():
            overrides = ablation_def.get("overrides", {})
            merged = merge_config(base_config, overrides)

            # Apply execution-level overrides
            if "seed" in execution:
                merged.setdefault("training", {})["seed"] = execution["seed"]
            if "max_iterations" in execution:
                merged.setdefault("training", {})["max_iterations"] = execution["max_iterations"]

            run_id = f"{encoder_id}_{ablation_id}"
            runs.append((run_id, encoder_id, ablation_id, merged))

    return runs


def run_encoder_ablation(
    encoder_id: str,
    runs: list[tuple[str, str, str, dict]],
    device: str,
    output_dir: Path,
    dry_run: bool = False,
    resume: bool = False,
) -> list[dict]:
    """Run all ablation configs for a single encoder."""
    from src.encoder_adapter import build_encoder
    from src.model_registry import get_model_spec
    from src.trainer import BalancedSashimiTrainer

    # Build the external encoder once for all ablation configs
    model_spec = get_model_spec(encoder_id)
    print(f"\nLoading encoder: {encoder_id} ({model_spec.display_name})")
    external_encoder = build_encoder(model_spec, target_dim=384, device=device)

    results = []
    encoder_runs = [(rid, aid, cfg) for rid, eid, aid, cfg in runs if eid == encoder_id]

    for i, (run_id, ablation_id, config) in enumerate(encoder_runs, 1):
        # Check for existing results
        run_dir = output_dir / run_id
        summary_path = run_dir / f"{run_id}_summary.json"
        if resume and summary_path.exists():
            print(f"  [{i}/{len(encoder_runs)}] {run_id} already completed, skipping")
            with open(summary_path) as f:
                results.append(json.load(f))
            continue

        print(f"  [{i}/{len(encoder_runs)}] Running {run_id}")
        seed = config.get("training", {}).get("seed", 42)
        trainer = BalancedSashimiTrainer(
            config=config,
            run_id=run_id,
            device=device,
            seed=seed,
            encoder_override=external_encoder,
        )
        train_result = trainer.train(dry_run=dry_run)

        result = {
            "run_id": run_id,
            "encoder_id": encoder_id,
            "ablation_id": ablation_id,
            "training": train_result,
        }
        results.append(result)

        # Save per-run summary
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"    Status: {train_result.get('status', 'unknown')}")

    # Unload encoder to free memory
    external_encoder.unload()
    print(f"  Encoder {encoder_id} unloaded")

    return results


def run_cross_comparison(
    all_results: list[dict],
    output_dir: Path,
) -> dict | None:
    """Compare PAB profiles across all encoder x ablation runs."""
    from src.pab_comparison import compare_profiles, export_json, export_markdown
    from src.pab_profile import PABProfile

    profiles = []
    for r in all_results:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = output_dir / "phase_c_comparison.json"
    comparison_json.write_text(json.dumps(report_data, indent=2))
    comparison_md = output_dir / "phase_c_comparison.md"
    comparison_md.write_text(export_markdown(report))

    print(report.summary_table())
    print(f"  Comparison saved: {comparison_json}")
    return report_data


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase C."""
    parser = argparse.ArgumentParser(
        description="Phase C: encoder x ablation matrix",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_RESEARCH_ROOT / "configs" / "phase_c.yaml",
        help="Path to Phase C config (default: research/configs/phase_c.yaml)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu"],
        help="Target device (default: mps)",
    )
    parser.add_argument(
        "--encoders",
        type=str,
        nargs="*",
        default=None,
        help="Subset of encoder model_ids to run (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have saved results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing",
    )
    return parser.parse_args()


def _load_base_config(phase_config: dict) -> dict:
    """Load the base YAML config referenced by the phase config."""
    base_config_path = _RESEARCH_ROOT / phase_config.get("base_config", "configs/base.yaml")
    with open(base_config_path) as f:
        return yaml.safe_load(f)


def _print_dry_run(phase_config: dict, run_matrix: list[tuple[str, str, str, dict]]) -> None:
    """Print planned runs without executing."""
    current_encoder = None
    for run_id, encoder_id, ablation_id, _config in run_matrix:
        if encoder_id != current_encoder:
            current_encoder = encoder_id
            enc_entry = next(
                e for e in phase_config["selected_encoders"] if e["model_id"] == encoder_id
            )
            print(f"Encoder: {encoder_id} — {enc_entry.get('description', '')}")
        ablation_desc = phase_config["ablation_configs"][ablation_id].get("description", "")
        print(f"  [{run_id}] {ablation_desc}")
    print()
    print("Dry run complete. No training executed.")


def _unique_encoder_ids(run_matrix: list[tuple[str, str, str, dict]]) -> list[str]:
    """Extract unique encoder IDs from the run matrix, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for _, encoder_id, _, _ in run_matrix:
        if encoder_id not in seen:
            result.append(encoder_id)
            seen.add(encoder_id)
    return result


def _execute_runs(
    args: argparse.Namespace,
    phase_config: dict,
    run_matrix: list[tuple[str, str, str, dict]],
) -> None:
    """Execute the full encoder x ablation matrix and run cross-comparison."""
    execution = phase_config.get("execution", {})
    output_dir = PROJECT_ROOT / execution.get("output_dir", "research/results/phase_c")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    start = time.time()

    for encoder_id in _unique_encoder_ids(run_matrix):
        results = run_encoder_ablation(
            encoder_id=encoder_id,
            runs=run_matrix,
            device=args.device,
            output_dir=output_dir,
            dry_run=False,
            resume=args.resume,
        )
        all_results.extend(results)

    print("\n=== Phase C Cross-Comparison ===")
    run_cross_comparison(all_results, output_dir)

    elapsed = time.time() - start
    print(f"\nPhase C complete in {elapsed:.1f}s")


def main():
    args = _parse_args()
    phase_config = load_phase_c_config(args.config)
    base_config = _load_base_config(phase_config)
    run_matrix = build_run_matrix(phase_config, base_config, encoder_filter=args.encoders)

    print(
        f"Phase C matrix: {len(phase_config['selected_encoders'])} encoders"
        f" x {len(phase_config['ablation_configs'])} ablations"
        f" = {len(run_matrix)} runs"
    )
    print()

    if args.dry_run:
        _print_dry_run(phase_config, run_matrix)
        return

    _execute_runs(args, phase_config, run_matrix)


if __name__ == "__main__":
    main()
