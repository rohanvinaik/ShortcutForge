"""PAB-integrated trajectory curation — classify training examples by learnability.

Runs a short probe training with PAB enabled and tracks per-example loss
trajectories to classify examples as easy, hard-but-learnable, unlearnable,
or destabilizing.  Classification thresholds are derived from the probe's
PAB summary (stability_mean, stability_std) rather than hardcoded constants.

Used for curriculum design and data quality analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

try:
    from src.pab_metrics import compute_stability
    from src.pab_profile import PABProfile
except ImportError:  # pragma: no cover - fallback for direct module execution
    from research.src.pab_metrics import compute_stability
    from research.src.pab_profile import PABProfile

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ExampleDifficulty:
    """Classification result for a single training example."""

    index: int = 0
    prompt: str = ""
    classification: str = "unknown"  # easy | hard_but_learnable | unlearnable | destabilizing
    loss_trajectory: list[float] = field(default_factory=list)
    spearman_rho: float = 0.0
    mean_loss: float = 0.0
    final_loss: float = 0.0


@dataclass
class DifficultyReport:
    """Aggregate difficulty classification report."""

    total_examples: int = 0
    probe_steps: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    examples: list[ExampleDifficulty] = field(default_factory=list)
    pab_summary: dict[str, float | str | None] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_examples": self.total_examples,
            "probe_steps": self.probe_steps,
            "counts": self.counts,
            "pab_summary": self.pab_summary,
            "examples": [asdict(e) for e in self.examples],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> DifficultyReport:
        with open(path) as f:
            data = json.load(f)
        return cls(
            total_examples=data["total_examples"],
            probe_steps=data["probe_steps"],
            counts=data["counts"],
            pab_summary=data.get("pab_summary", {}),
            examples=[ExampleDifficulty(**e) for e in data["examples"]],
        )


class ProbeResult(NamedTuple):
    """Output of run_probe: per-example losses + PAB profile."""

    per_example_losses: dict[int, list[float]]
    pab_profile: PABProfile


# ---------------------------------------------------------------------------
# Probe execution
# ---------------------------------------------------------------------------


def _snapshot_per_example_losses(
    per_example_losses: dict[int, list[float]],
    losses: list[float],
    batch_start: int,
    batch_len: int,
) -> None:
    """Append loss to per-example trajectories for a batch."""
    for i in range(min(batch_len, len(losses))):
        idx = batch_start + i
        if idx in per_example_losses:
            per_example_losses[idx].append(float(losses[i]))


def run_probe(
    trainer,
    dataset,
    probe_steps: int = 250,
    checkpoint_interval: int = 50,
    batch_size: int = 16,
) -> ProbeResult:
    """Run a short probe training with PAB enabled, tracking per-example losses.

    Args:
        trainer: A BalancedSashimiTrainer (or compatible) with .train_step().
        dataset: An indexable dataset of training examples.
        probe_steps: Number of training steps to probe.
        checkpoint_interval: Steps between per-example loss snapshots.
        batch_size: Batch size for the probe DataLoader.

    Returns:
        ProbeResult with per-example loss trajectories and PAB profile.
    """
    from torch.utils.data import DataLoader

    try:
        from src.pab_tracker import CheckpointData, PABTracker
    except ImportError:  # pragma: no cover - fallback for direct module execution
        from research.src.pab_tracker import CheckpointData, PABTracker

    n_examples = len(dataset)
    per_example_losses: dict[int, list[float]] = {i: [] for i in range(n_examples)}

    tracker = PABTracker(
        experiment_id="trajectory-probe",
        checkpoint_interval=checkpoint_interval,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: b,
    )

    step = 0
    while step < probe_steps:
        for batch_idx, batch in enumerate(loader):
            if step >= probe_steps:
                break
            loss_dict = trainer.train_step(batch, return_per_example=True)
            step += 1

            if step % checkpoint_interval == 0:
                loss = float(loss_dict["L_total"])
                tracker.record(CheckpointData(step=step, train_loss=loss))
                per_example = loss_dict.get("per_example_loss")
                if not isinstance(per_example, list) or len(per_example) != len(batch):
                    # Fallback if trainer does not expose per-example losses.
                    per_example = [loss] * len(batch)
                _snapshot_per_example_losses(
                    per_example_losses,
                    per_example,
                    batch_idx * batch_size,
                    len(batch),
                )

    return ProbeResult(per_example_losses=per_example_losses, pab_profile=tracker.finalize())


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _spearman_rho(values: list[float]) -> float:
    """Compute Spearman rank correlation between values and their index."""
    if len(values) < 3:
        return 0.0
    n = len(values)
    x_ranks = np.arange(1, n + 1, dtype=float)
    y = np.array(values)
    order = y.argsort()
    y_ranks = np.empty_like(order, dtype=float)
    y_ranks[order] = np.arange(1, n + 1, dtype=float)
    d_sq = np.sum((x_ranks - y_ranks) ** 2)
    return float(1 - (6 * d_sq) / (n * (n**2 - 1)))


def _classify_single(
    losses: list[float],
    aggregate_mean_loss: float,
    stability_mean: float,
    stability_std: float,
    stability_regime: str,
) -> str:
    """Classify a single example using PAB-calibrated thresholds.

    Classification rules (all thresholds derived from probe dynamics):
    - easy: final loss < aggregate_mean × 0.5 AND monotonically decreasing
    - destabilizing: per-example stability > stability_mean + 2 × stability_std
    - hard_but_learnable: loss trend clearly negative (rho < -0.3)
    - unlearnable: loss trend flat/rising AND regime != "chaotic"
    """
    if not losses:
        return "unknown"

    final_loss = losses[-1]
    rho = _spearman_rho(losses)

    # Easy: low loss with monotonic decrease
    is_monotonic = all(losses[i] >= losses[i + 1] for i in range(len(losses) - 1))
    if final_loss < aggregate_mean_loss * 0.5 and is_monotonic:
        return "easy"

    # Destabilizing: per-example instability exceeds probe baseline
    if len(losses) >= 2:
        per_example_stab = [
            compute_stability(losses[i], losses[i + 1]) for i in range(len(losses) - 1)
        ]
        max_instability = max(per_example_stab)
        threshold = stability_mean + 2.0 * stability_std
        if threshold > 0 and max_instability > threshold:
            return "destabilizing"

    # Trend-based classification
    if rho < -0.3:
        return "hard_but_learnable"
    if rho > -0.1 and stability_regime != "chaotic":
        return "unlearnable"

    return "hard_but_learnable"


def classify_examples(
    probe_result: ProbeResult,
    dataset,
) -> list[ExampleDifficulty]:
    """Classify all examples using PAB-calibrated thresholds.

    Args:
        probe_result: Output of run_probe().
        dataset: The training dataset (for extracting prompts).

    Returns:
        List of ExampleDifficulty, one per example.
    """
    summary = probe_result.pab_profile.summary
    per_example_losses = probe_result.per_example_losses

    # Compute aggregate mean loss from all example trajectories
    all_final_losses = [losses[-1] for losses in per_example_losses.values() if losses]
    aggregate_mean_loss = float(np.mean(all_final_losses)) if all_final_losses else 1.0

    examples = []
    for idx in range(len(dataset)):
        losses = per_example_losses.get(idx, [])
        ex = dataset[idx]
        prompt = ex.prompt if hasattr(ex, "prompt") else str(idx)

        classification = _classify_single(
            losses=losses,
            aggregate_mean_loss=aggregate_mean_loss,
            stability_mean=summary.stability_mean,
            stability_std=summary.stability_std,
            stability_regime=summary.stability_regime,
        )
        rho = _spearman_rho(losses) if losses else 0.0

        examples.append(
            ExampleDifficulty(
                index=idx,
                prompt=prompt[:120],
                classification=classification,
                loss_trajectory=losses,
                spearman_rho=rho,
                mean_loss=float(np.mean(losses)) if losses else 0.0,
                final_loss=losses[-1] if losses else 0.0,
            )
        )

    return examples


def build_difficulty_report(
    examples: list[ExampleDifficulty],
    probe_result: ProbeResult,
) -> DifficultyReport:
    """Build a DifficultyReport from classified examples and probe results."""
    counts: dict[str, int] = {}
    for ex in examples:
        counts[ex.classification] = counts.get(ex.classification, 0) + 1

    summary = probe_result.pab_profile.summary
    pab_summary_dict: dict[str, float | str | None] = {
        "stability_mean": summary.stability_mean,
        "stability_std": summary.stability_std,
        "predictability_final": summary.predictability_final,
        "stability_regime": summary.stability_regime,
        "convergence_epoch": summary.convergence_epoch,
    }

    return DifficultyReport(
        total_examples=len(examples),
        probe_steps=len(probe_result.pab_profile.checkpoints) * 50,
        counts=counts,
        examples=examples,
        pab_summary=pab_summary_dict,
    )
