#!/usr/bin/env python3
"""CLI wrapper for PAB-integrated trajectory curation.

Usage:
    python research/scripts/run_trajectory_curator.py \
        --config research/configs/base.yaml \
        --data training_data/typed_ir_train.jsonl \
        --output training_data/difficulty_report.json \
        [--probe-steps 250] [--device mps]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))

from src.trajectory_curator import (  # noqa: E402
    build_difficulty_report,
    classify_examples,
    run_probe,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PAB-informed trajectory curation",
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML config")
    parser.add_argument("--data", type=Path, required=True, help="Training JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Difficulty report JSON")
    parser.add_argument("--probe-steps", type=int, default=250, help="Probe training steps")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cpu"])
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))

    from src.data import TypedIRDataset
    from src.trainer import BalancedSashimiTrainer

    probe_config = {**config}
    probe_config["training"] = {**config["training"], "max_iterations": args.probe_steps}

    trainer = BalancedSashimiTrainer(
        config=probe_config,
        run_id="trajectory-probe",
        device=args.device,
        seed=config.get("training", {}).get("seed", 42),
    )
    trainer.setup()

    dataset = TypedIRDataset(args.data)
    batch_size = config["training"].get("batch_size", 16)

    print(f"Running probe: {args.probe_steps} steps, {len(dataset)} examples")
    probe_result = run_probe(trainer, dataset, args.probe_steps, batch_size=batch_size)
    examples = classify_examples(probe_result, dataset)
    report = build_difficulty_report(examples, probe_result)

    report.save(args.output)
    print(f"Difficulty distribution: {report.counts}")


if __name__ == "__main__":
    main()
