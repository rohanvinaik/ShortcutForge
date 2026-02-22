"""PAB training tracker with checkpoint accumulation.

Accumulates metrics during training and produces a ``PABProfile`` artifact.
Uses pure metric functions from ``pab_metrics`` for computation.

Usage:
    tracker = PABTracker(experiment_id="EXP-2.1", checkpoint_interval=50)
    for step in training_loop:
        ...
        if step % 50 == 0:
            data = CheckpointData(
                step=step,
                train_loss=loss,
                val_loss=val_loss,
                tier_accuracies={"tier1": t1, "tier2": t2, "tier3": t3},
            )
            tracker.record(data)
    profile = tracker.finalize()
    profile.save("research/models/EXP-2.1/pab_profile.json")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from src.pab_metrics import (
        classify_domain,
        compute_crystallization,
        compute_feature_importance,
        compute_generalization_gap,
        compute_predictability,
        compute_repr_evolution,
        compute_stability,
        find_tier_convergence,
        linear_slope,
        monotonic_trend,
    )
    from src.pab_profile import (
        PABCoreSeries,
        PABDomainData,
        PABLossSeries,
        PABProfile,
        PABSummary,
        PABTierSeries,
    )
except ImportError:  # pragma: no cover - fallback for direct module execution
    from research.src.pab_metrics import (
        classify_domain,
        compute_crystallization,
        compute_feature_importance,
        compute_generalization_gap,
        compute_predictability,
        compute_repr_evolution,
        compute_stability,
        find_tier_convergence,
        linear_slope,
        monotonic_trend,
    )
    from research.src.pab_profile import (
        PABCoreSeries,
        PABDomainData,
        PABLossSeries,
        PABProfile,
        PABSummary,
        PABTierSeries,
    )

_STABILITY_CONVERGED = 0.10
_CONVERGENCE_WINDOW = 5
_TIER1_THRESHOLD = 0.80
_TIER2_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# Input dataclass — bundles checkpoint metrics for record()
# ---------------------------------------------------------------------------


@dataclass
class CheckpointData:
    """Metrics collected at a single training checkpoint.

    Bundles all per-checkpoint measurements into one object so that
    ``PABTracker.record()`` takes a single argument instead of 10.
    """

    step: int = 0
    train_loss: float = 0.0
    val_loss: float | None = None
    loss_components: dict[str, float] | None = None
    adaptive_weights: dict[str, float] | None = None
    tier_accuracies: dict[str, float] | None = None
    bottleneck_embeddings: np.ndarray | None = field(default=None, repr=False)
    decoder_weight_signs: np.ndarray | None = field(default=None, repr=False)
    domain_accuracies: dict[str, float] | None = None
    action_accuracies: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# Internal accumulators
# ---------------------------------------------------------------------------


@dataclass
class _CoreAccum:
    """Accumulators for core PAB metrics."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_checkpoints: list[int] = field(default_factory=list)
    stability: list[float] = field(default_factory=list)
    predictability: list[float] = field(default_factory=list)
    gen_gap: list[float] = field(default_factory=list)
    repr_evolution: list[float] = field(default_factory=list)
    prev_bottleneck_mean: np.ndarray | None = field(default=None, repr=False)


@dataclass
class _TierAccum:
    """Accumulators for tier data."""

    tier1: list[float] = field(default_factory=list)
    tier2: list[float] = field(default_factory=list)
    tier3: list[float] = field(default_factory=list)
    crystallization: list[float] = field(default_factory=list)
    prev_weight_signs: np.ndarray | None = field(default=None, repr=False)
    weight_snapshots: list[np.ndarray] = field(default_factory=list)


@dataclass
class _AuxAccum:
    """Accumulators for domain/action/loss data."""

    domain_acc: dict[str, list[float]] = field(default_factory=dict)
    action_acc: dict[str, list[float]] = field(default_factory=dict)
    loss_ce: list[float] = field(default_factory=list)
    loss_margin: list[float] = field(default_factory=list)
    loss_repair: list[float] = field(default_factory=list)
    loss_weights: list[dict[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PAB Tracker
# ---------------------------------------------------------------------------


@dataclass
class PABTracker:
    """Accumulates PAB metrics during training and produces a PABProfile.

    Call ``record()`` at each checkpoint (e.g., every 50 training steps).
    Call ``finalize()`` after training to compute summary statistics.
    """

    experiment_id: str = ""
    config_hash: str = ""
    checkpoint_interval: int = 50
    _checkpoints: list[int] = field(default_factory=list)
    _core: _CoreAccum = field(default_factory=_CoreAccum)
    _tier: _TierAccum = field(default_factory=_TierAccum)
    _aux: _AuxAccum = field(default_factory=_AuxAccum)

    def record(self, data: CheckpointData) -> None:
        """Record metrics at a training checkpoint."""
        self._checkpoints.append(data.step)
        self._core.train_losses.append(data.train_loss)

        self._record_core_metrics(data)
        self._record_tier_metrics(data)
        self._record_aux_metrics(data)

    def _record_core_metrics(self, data: CheckpointData) -> None:
        """Compute and store stability, predictability, gen gap, repr evolution."""
        c = self._core

        if len(c.train_losses) >= 2:
            c.stability.append(compute_stability(c.train_losses[-2], data.train_loss))
        else:
            c.stability.append(0.0)

        c.predictability.append(compute_predictability(c.train_losses))

        if data.val_loss is not None:
            c.val_losses.append(data.val_loss)
            c.val_checkpoints.append(data.step)
            c.gen_gap.append(compute_generalization_gap(data.train_loss, data.val_loss))
        else:
            c.gen_gap.append(c.gen_gap[-1] if c.gen_gap else 0.0)

        if data.bottleneck_embeddings is not None:
            r_t, new_mean = compute_repr_evolution(
                data.bottleneck_embeddings, c.prev_bottleneck_mean
            )
            c.prev_bottleneck_mean = new_mean
            c.repr_evolution.append(r_t)
        else:
            c.repr_evolution.append(c.repr_evolution[-1] if c.repr_evolution else 0.0)

    def _record_tier_metrics(self, data: CheckpointData) -> None:
        """Store tier accuracies and compute crystallization."""
        t = self._tier
        tiers = data.tier_accuracies or {}
        t.tier1.append(tiers.get("tier1", 0.0))
        t.tier2.append(tiers.get("tier2", 0.0))
        t.tier3.append(tiers.get("tier3", 0.0))

        if data.decoder_weight_signs is not None:
            signs = np.array(data.decoder_weight_signs, dtype=np.float32)
            t.crystallization.append(compute_crystallization(signs, t.prev_weight_signs))
            t.prev_weight_signs = signs.copy()
            t.weight_snapshots.append(signs.copy())
        else:
            t.crystallization.append(t.crystallization[-1] if t.crystallization else 0.0)

    def _record_aux_metrics(self, data: CheckpointData) -> None:
        """Store loss components, domain/action accuracies."""
        a = self._aux
        components = data.loss_components or {}
        a.loss_ce.append(components.get("ce", 0.0))
        a.loss_margin.append(components.get("margin", 0.0))
        a.loss_repair.append(components.get("repair", 0.0))
        a.loss_weights.append(data.adaptive_weights or {})

        if data.domain_accuracies:
            for domain, acc in data.domain_accuracies.items():
                a.domain_acc.setdefault(domain, []).append(acc)
        if data.action_accuracies:
            for action, acc in data.action_accuracies.items():
                a.action_acc.setdefault(action, []).append(acc)

    def finalize(self) -> PABProfile:
        """Compute summary statistics and return the final PABProfile."""
        c, t, a = self._core, self._tier, self._aux

        return PABProfile(
            experiment_id=self.experiment_id,
            config_hash=self.config_hash,
            checkpoints=list(self._checkpoints),
            core=PABCoreSeries(
                stability=list(c.stability),
                predictability=list(c.predictability),
                generalization_gap=list(c.gen_gap),
                representation_evolution=list(c.repr_evolution),
            ),
            tiers=PABTierSeries(
                tier1_accuracy=list(t.tier1),
                tier2_accuracy=list(t.tier2),
                tier3_accuracy=list(t.tier3),
                ternary_crystallization=list(t.crystallization),
            ),
            domains=self._build_domain_data(),
            losses=PABLossSeries(
                loss_ce=list(a.loss_ce),
                loss_margin=list(a.loss_margin),
                loss_repair=list(a.loss_repair),
                loss_adaptive_weights=list(a.loss_weights),
            ),
            summary=self._compute_summary(),
        )

    def _build_domain_data(self) -> PABDomainData:
        """Build domain progression data with classification."""
        a = self._aux
        if not a.domain_acc:
            return PABDomainData(domain_progression=dict(a.domain_acc))
        n = max(len(v) for v in a.domain_acc.values())
        classification = {domain: classify_domain(accs, n) for domain, accs in a.domain_acc.items()}
        return PABDomainData(
            domain_progression=dict(a.domain_acc),
            domain_classification=classification,
            action_progression=dict(a.action_acc),
        )

    def _compute_summary(self) -> PABSummary:
        """Compute all derived summary statistics."""
        c, t = self._core, self._tier
        s = PABSummary()

        if c.stability:
            s.stability_mean = float(np.mean(c.stability))
            s.stability_std = float(np.std(c.stability))
        if c.predictability:
            s.predictability_final = c.predictability[-1]

        s.convergence_epoch = self._detect_convergence()
        s.early_stop_epoch = self._detect_early_stop()
        s.stability_regime = self._classify_regime()
        s.tier1_convergence_step = find_tier_convergence(
            t.tier1, _TIER1_THRESHOLD, self._checkpoints
        )
        s.tier2_convergence_step = find_tier_convergence(
            t.tier2, _TIER2_THRESHOLD, self._checkpoints
        )
        s.crystallization_rate = linear_slope(t.crystallization)
        s.feature_importance_L = compute_feature_importance(t.weight_snapshots)
        return s

    def should_early_exit(self, step: int) -> bool:
        """PAB-informed early exit decision.

        Returns True if the training trajectory indicates the run should be
        terminated early (bad process predicts bad outcomes).
        """
        if step < 200 or len(self._checkpoints) < 4:
            return False

        c, t = self._core, self._tier
        tier1_acc = t.tier1[-1] if t.tier1 else 0.0
        recent_stability = float(np.mean(c.stability[-5:])) if len(c.stability) >= 5 else 0.0
        recent_pred = c.predictability[-1] if c.predictability else 0.0
        tier1_trend = monotonic_trend(t.tier1[-10:])

        if tier1_acc < 0.60 and recent_stability > 0.30 and recent_pred > 0.10:
            return True
        if tier1_acc < 0.70 and recent_stability < 0.15 and tier1_trend > 0.5:
            return False
        if step > 400 and tier1_trend < 0.05 and tier1_acc < 0.75:
            return True
        if recent_stability > 0.50:
            return True
        return False

    def _detect_convergence(self) -> int | None:
        """Find step where stability < threshold for consecutive checks."""
        stab = self._core.stability
        if len(stab) < _CONVERGENCE_WINDOW:
            return None
        for i in range(len(stab) - _CONVERGENCE_WINDOW + 1):
            if all(s < _STABILITY_CONVERGED for s in stab[i : i + _CONVERGENCE_WINDOW]):
                return self._checkpoints[i]
        return None

    def _detect_early_stop(self) -> int | None:
        """Find first step where val loss increases."""
        c = self._core
        vl = c.val_losses
        if len(vl) < 2:
            return None
        for i in range(1, len(vl)):
            if vl[i] > vl[i - 1]:
                if i < len(c.val_checkpoints):
                    return c.val_checkpoints[i]
                return self._checkpoints[min(i, len(self._checkpoints) - 1)]
        return None

    def _classify_regime(self) -> str:
        """Classify training stability regime."""
        stab = self._core.stability
        if not stab:
            return "unknown"
        mean_s = float(np.mean(stab))
        if mean_s < 0.15:
            return "stable"
        if mean_s > 0.30:
            return "chaotic"
        if len(stab) >= 10:
            half = len(stab) // 2
            if float(np.mean(stab[:half])) > 0.25 and float(np.mean(stab[half:])) < 0.15:
                return "phase_transition"
        return "moderate"
