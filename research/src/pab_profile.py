"""Process-Aware Benchmarking (PAB) profile data model.

Defines the PABProfile artifact and its sub-dataclasses. This is the frozen
record saved alongside model checkpoints after training. Serialization uses
a flat JSON format matching the EXPERIMENT_RESULTS.md schema.

See ``pab_tracker.py`` for the accumulation logic that produces these profiles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sub-dataclasses (keeps each class under 10 attributes)
# ---------------------------------------------------------------------------


@dataclass
class PABCoreSeries:
    """Core PAB time series: stability, predictability, gen gap, repr evolution."""

    stability: list[float] = field(default_factory=list)
    predictability: list[float] = field(default_factory=list)
    generalization_gap: list[float] = field(default_factory=list)
    representation_evolution: list[float] = field(default_factory=list)


@dataclass
class PABTierSeries:
    """Tier-wise accuracy and ternary crystallization time series."""

    tier1_accuracy: list[float] = field(default_factory=list)
    tier2_accuracy: list[float] = field(default_factory=list)
    tier3_accuracy: list[float] = field(default_factory=list)
    ternary_crystallization: list[float] = field(default_factory=list)


@dataclass
class PABDomainData:
    """Per-domain and per-action progression data."""

    domain_progression: dict[str, list[float]] = field(default_factory=dict)
    domain_classification: dict[str, str] = field(default_factory=dict)
    action_progression: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class PABLossSeries:
    """Loss component trajectories."""

    loss_ce: list[float] = field(default_factory=list)
    loss_margin: list[float] = field(default_factory=list)
    loss_repair: list[float] = field(default_factory=list)
    loss_adaptive_weights: list[dict[str, float]] = field(default_factory=list)


@dataclass
class PABSummary:
    """Derived summary statistics, computed by finalize()."""

    stability_mean: float = 0.0
    stability_std: float = 0.0
    predictability_final: float = 0.0
    early_stop_epoch: int | None = None
    convergence_epoch: int | None = None
    stability_regime: str = "unknown"
    tier1_convergence_step: int | None = None
    tier2_convergence_step: int | None = None
    crystallization_rate: float = 0.0
    feature_importance_L: float = 0.0


# ---------------------------------------------------------------------------
# PAB Profile — the frozen artifact saved per training run
# ---------------------------------------------------------------------------


@dataclass
class PABProfile:
    """Process-Aware Benchmark profile for a training run.

    Contains time series (via sub-dataclasses) and derived summary statistics.
    Saved as JSON alongside model checkpoints.
    """

    experiment_id: str = ""
    config_hash: str = ""
    checkpoints: list[int] = field(default_factory=list)
    core: PABCoreSeries = field(default_factory=PABCoreSeries)
    tiers: PABTierSeries = field(default_factory=PABTierSeries)
    domains: PABDomainData = field(default_factory=PABDomainData)
    losses: PABLossSeries = field(default_factory=PABLossSeries)
    summary: PABSummary = field(default_factory=PABSummary)

    def save(self, path: str | Path) -> None:
        """Save profile to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_serialize_profile(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> PABProfile:
        """Load profile from JSON."""
        with open(path) as f:
            data = json.load(f)
        return _deserialize_profile(data)


def _serialize_profile(p: PABProfile) -> dict[str, Any]:
    """Serialize PABProfile to flat JSON dict (matches EXPERIMENT_RESULTS schema)."""
    return {
        "experiment_id": p.experiment_id,
        "config_hash": p.config_hash,
        "checkpoints": p.checkpoints,
        "stability": p.core.stability,
        "predictability": p.core.predictability,
        "generalization_gap": p.core.generalization_gap,
        "representation_evolution": p.core.representation_evolution,
        "tier1_accuracy": p.tiers.tier1_accuracy,
        "tier2_accuracy": p.tiers.tier2_accuracy,
        "tier3_accuracy": p.tiers.tier3_accuracy,
        "ternary_crystallization": p.tiers.ternary_crystallization,
        "domain_progression": p.domains.domain_progression,
        "domain_classification": p.domains.domain_classification,
        "action_progression": p.domains.action_progression,
        "loss_ce": p.losses.loss_ce,
        "loss_margin": p.losses.loss_margin,
        "loss_repair": p.losses.loss_repair,
        "loss_adaptive_weights": p.losses.loss_adaptive_weights,
        "summary": {
            "stability_mean": p.summary.stability_mean,
            "stability_std": p.summary.stability_std,
            "predictability_final": p.summary.predictability_final,
            "early_stop_epoch": p.summary.early_stop_epoch,
            "convergence_epoch": p.summary.convergence_epoch,
            "stability_regime": p.summary.stability_regime,
            "tier1_convergence_step": p.summary.tier1_convergence_step,
            "tier2_convergence_step": p.summary.tier2_convergence_step,
            "crystallization_rate": p.summary.crystallization_rate,
            "feature_importance_L": p.summary.feature_importance_L,
        },
    }


def _deserialize_profile(data: dict[str, Any]) -> PABProfile:
    """Deserialize flat JSON dict to PABProfile."""
    s = data.get("summary", {})
    return PABProfile(
        experiment_id=data.get("experiment_id", ""),
        config_hash=data.get("config_hash", ""),
        checkpoints=data.get("checkpoints", []),
        core=PABCoreSeries(
            stability=data.get("stability", []),
            predictability=data.get("predictability", []),
            generalization_gap=data.get("generalization_gap", []),
            representation_evolution=data.get("representation_evolution", []),
        ),
        tiers=PABTierSeries(
            tier1_accuracy=data.get("tier1_accuracy", []),
            tier2_accuracy=data.get("tier2_accuracy", []),
            tier3_accuracy=data.get("tier3_accuracy", []),
            ternary_crystallization=data.get("ternary_crystallization", []),
        ),
        domains=PABDomainData(
            domain_progression=data.get("domain_progression", {}),
            domain_classification=data.get("domain_classification", {}),
            action_progression=data.get("action_progression", {}),
        ),
        losses=PABLossSeries(
            loss_ce=data.get("loss_ce", []),
            loss_margin=data.get("loss_margin", []),
            loss_repair=data.get("loss_repair", []),
            loss_adaptive_weights=data.get("loss_adaptive_weights", []),
        ),
        summary=PABSummary(
            stability_mean=s.get("stability_mean", 0.0),
            stability_std=s.get("stability_std", 0.0),
            predictability_final=s.get("predictability_final", 0.0),
            early_stop_epoch=s.get("early_stop_epoch"),
            convergence_epoch=s.get("convergence_epoch"),
            stability_regime=s.get("stability_regime", "unknown"),
            tier1_convergence_step=s.get("tier1_convergence_step"),
            tier2_convergence_step=s.get("tier2_convergence_step"),
            crystallization_rate=s.get("crystallization_rate", 0.0),
            feature_importance_L=s.get("feature_importance_L", 0.0),
        ),
    )
