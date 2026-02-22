#!/usr/bin/env python3
"""Phase B: PAB fine-tuning study across architectures.

LoRA fine-tunes selected models with PAB profiling to test
hypotheses H1-H6 about architecture-training dynamics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _RESEARCH_ROOT.parent

# Ensure research root is on sys.path for absolute imports
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from src.cross_model_comparison import (
    ArchitectureComparisonReport,
    compare_by_architecture,
    export_hypothesis_report,
)
from src.lora_trainer import (
    LoRATrainConfig,
    LoRATrainer,
    _CheckpointSchedule,
    _LoRAHyperparams,
)
from src.model_registry import get_model_spec, list_model_ids
from src.pab_profile import PABProfile

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load phase B YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_train_config(cfg: dict, model_id: str) -> LoRATrainConfig:
    """Build a LoRATrainConfig from the YAML config and model_id."""
    lora = cfg.get("lora", {})
    data = cfg.get("data", {})
    execution = cfg.get("execution", {})

    train_file = PROJECT_ROOT / data.get("train_file", "training_data/train.jsonl")
    eval_file = PROJECT_ROOT / data.get("eval_file", "training_data/eval.jsonl")
    output_dir = PROJECT_ROOT / execution.get("output_dir", "research/results/phase_b")

    return LoRATrainConfig(
        model_id=model_id,
        train_file=train_file,
        eval_file=eval_file,
        hparams=_LoRAHyperparams(
            rank=lora.get("rank", 16),
            alpha=lora.get("alpha", 32),
            lr=lora.get("lr", 2e-4),
            batch_size=lora.get("batch_size", 4),
            max_steps=lora.get("max_steps", 1000),
        ),
        schedule=_CheckpointSchedule(
            checkpoint_interval=lora.get("checkpoint_interval", 50),
            eval_interval=lora.get("eval_interval", 100),
        ),
        output_dir=output_dir,
        device=execution.get("device", "mps"),
    )


def run_single_model(
    model_id: str,
    cfg: dict,
    dry_run: bool = False,
) -> dict | None:
    """Train a single model with LoRA and PAB profiling."""
    try:
        spec = get_model_spec(model_id)
    except KeyError:
        logger.error("Model spec not found for %s, skipping", model_id)
        return None

    train_config = build_train_config(cfg, model_id)

    if dry_run:
        logger.info(
            "[DRY RUN] %s: rank=%d, alpha=%d, steps=%d, device=%s",
            model_id,
            train_config.hparams.rank,
            train_config.hparams.alpha,
            train_config.hparams.max_steps,
            train_config.device,
        )
        return {"model_id": model_id, "status": "dry_run"}

    logger.info("=" * 60)
    logger.info("Training model: %s (%s)", spec.display_name, spec.architecture_family)
    logger.info("=" * 60)

    trainer = LoRATrainer(train_config, spec)
    try:
        trainer.setup()
        result = trainer.train()
        result["status"] = "success"
        return result
    except Exception as exc:
        logger.error("Training failed for %s: %s", model_id, exc, exc_info=True)
        return {"model_id": model_id, "status": "failed", "error": str(exc)}


def load_phase_a_taxonomy(phase_a_dir: Path) -> dict | None:
    """Load Phase A taxonomy clusters if available."""
    clusters_path = phase_a_dir / "taxonomy_clusters.json"
    report_path = phase_a_dir / "taxonomy_report.json"

    if clusters_path.exists():
        with open(clusters_path) as f:
            return json.load(f)

    if not report_path.exists():
        logger.info("No Phase A taxonomy found at %s or %s", clusters_path, report_path)
        return None

    # Backward-compatible loader for taxonomy_report.json format.
    with open(report_path) as f:
        report = json.load(f)
    clusters = report.get("clusters", {})
    model_to_cluster: dict[str, int] = {}
    for cluster_id, members in clusters.items():
        try:
            cid = int(cluster_id)
        except (TypeError, ValueError):
            cid = abs(hash(str(cluster_id))) % 1000
        for model_id in members:
            model_to_cluster[str(model_id)] = cid
    return model_to_cluster


def run_comparison(
    cfg: dict,
    results: list[dict],
    phase_a_clusters: dict | None = None,
) -> ArchitectureComparisonReport | None:
    """Run cross-architecture comparison after all models are trained."""
    execution = cfg.get("execution", {})
    output_dir = PROJECT_ROOT / execution.get("output_dir", "research/results/phase_b")

    profiles = []
    model_specs = []
    for r in results:
        if r.get("status") != "success":
            continue
        model_id = r["model_id"]
        profile_path = output_dir / model_id / "pab_profile.json"
        if profile_path.exists():
            profiles.append(PABProfile.load(profile_path))
            spec = get_model_spec(model_id)
            if spec:
                model_specs.append(spec)

    if len(profiles) < 2:
        logger.warning("Need at least 2 successful profiles for comparison, got %d", len(profiles))
        return None

    report = compare_by_architecture(profiles, model_specs)

    # Override H6 with actual Phase A data if available
    if phase_a_clusters:
        from src.cross_model_comparison import check_h6_taxonomy_predicts_trainability

        report.hypothesis_results["H6"] = check_h6_taxonomy_predicts_trainability(
            phase_a_clusters,
            profiles,
        )

    # Save report
    report_md = export_hypothesis_report(report)
    report_path = output_dir / "hypothesis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info("Hypothesis report saved to %s", report_path)

    report_json = output_dir / "hypothesis_report.json"
    with open(report_json, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    return report


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Phase B."""
    parser = argparse.ArgumentParser(description="Phase B: PAB fine-tuning study")
    parser.add_argument(
        "--config",
        type=Path,
        default=_RESEARCH_ROOT / "configs" / "phase_b.yaml",
        help="Path to phase B config file",
    )
    parser.add_argument("--models", nargs="+", help="Override: train only these model IDs")
    parser.add_argument("--device", default=None, help="Override device (mps, cuda, cpu)")
    parser.add_argument(
        "--phase-a-dir",
        type=Path,
        default=None,
        help="Directory containing Phase A taxonomy output",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List models and configs without training"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def _select_models(model_ids: list[str]) -> list[str]:
    """Filter model IDs against the registry, logging skipped ones."""
    available = set(list_model_ids())
    selected = [m for m in model_ids if m in available]
    skipped = [m for m in model_ids if m not in available]

    if skipped:
        logger.warning("Models not in registry (skipped): %s", skipped)

    logger.info("Phase B: training %d models", len(selected))
    for mid in selected:
        spec = get_model_spec(mid)
        if spec:
            logger.info(
                "  - %s (%s, %.1fB params)", mid, spec.architecture_family, spec.param_count_b
            )

    if not selected:
        logger.error("No valid models selected. Available: %s", sorted(available))
        sys.exit(1)

    return selected


def _log_summary_and_save(
    cfg: dict, model_ids: list[str], selected: list[str], results: list[dict]
) -> None:
    """Log training summary and save the run manifest."""
    successes = [r for r in results if r.get("status") == "success"]
    failures = [r for r in results if r.get("status") == "failed"]
    logger.info("Training complete: %d success, %d failed", len(successes), len(failures))
    for f in failures:
        logger.info("  FAILED: %s — %s", f["model_id"], f.get("error", "unknown"))

    execution = cfg.get("execution", {})
    output_dir = PROJECT_ROOT / execution.get("output_dir", "research/results/phase_b")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"models_attempted": model_ids, "models_selected": selected, "results": results}
    with open(output_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    """Entry point for Phase B training study."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.device:
        cfg.setdefault("execution", {})["device"] = args.device

    model_ids = args.models or cfg.get("selected_models", [])
    selected = _select_models(model_ids)

    # Train each model
    results = []
    for model_id in selected:
        result = run_single_model(model_id, cfg, dry_run=args.dry_run)
        if result:
            results.append(result)

    # Cross-architecture comparison
    successes = [r for r in results if r.get("status") == "success"]
    if not args.dry_run and len(successes) >= 2:
        phase_a_clusters = None
        if args.phase_a_dir:
            phase_a_clusters = load_phase_a_taxonomy(args.phase_a_dir)
        run_comparison(cfg, results, phase_a_clusters)

    _log_summary_and_save(cfg, model_ids, selected, results)


if __name__ == "__main__":
    main()
