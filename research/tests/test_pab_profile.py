"""Tests for PAB profile data model, metrics, and tracker."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.pab_metrics import (
    classify_domain,
    compute_crystallization,
    compute_predictability,
    compute_repr_evolution,
    compute_stability,
    find_tier_convergence,
    is_oscillating,
    linear_slope,
    monotonic_trend,
)
from src.pab_profile import PABCoreSeries, PABProfile, PABSummary, PABTierSeries
from src.pab_tracker import CheckpointData, PABTracker

# ---------------------------------------------------------------------------
# Metric computation tests
# ---------------------------------------------------------------------------


class TestComputeStability:
    def test_no_change(self):
        assert compute_stability(1.0, 1.0) == pytest.approx(0.0, abs=1e-6)

    def test_decrease(self):
        s = compute_stability(2.0, 1.0)
        assert s == pytest.approx(0.5, abs=1e-6)

    def test_increase(self):
        s = compute_stability(1.0, 2.0)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_near_zero_prev(self):
        s = compute_stability(1e-10, 0.5)
        assert s > 0


class TestComputePredictability:
    def test_empty(self):
        assert compute_predictability([]) == 0.0

    def test_single(self):
        assert compute_predictability([1.0]) == 0.0

    def test_constant_losses(self):
        assert compute_predictability([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-6)

    def test_varying_losses(self):
        losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        p = compute_predictability(losses)
        assert p == pytest.approx(0.0, abs=1e-6)  # constant deltas -> zero variance

    def test_erratic_losses(self):
        losses = [1.0, 0.5, 0.9, 0.3, 0.8]
        p = compute_predictability(losses)
        assert p > 0  # high variance


class TestComputeReprEvolution:
    def test_first_step(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        r, mean = compute_repr_evolution(emb, None)
        assert r == 1.0
        assert mean.shape == (2,)

    def test_no_change(self):
        emb = np.array([[1.0, 0.0], [1.0, 0.0]])
        mean = np.array([1.0, 0.0])
        r, _ = compute_repr_evolution(emb, mean)
        assert r == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal(self):
        emb = np.array([[0.0, 1.0]])
        prev = np.array([1.0, 0.0])
        r, _ = compute_repr_evolution(emb, prev)
        assert r == pytest.approx(1.0, abs=1e-5)


class TestComputeCrystallization:
    def test_no_prev(self):
        signs = np.array([1, -1, 0, 1])
        assert compute_crystallization(signs, None) == 0.0

    def test_all_same(self):
        signs = np.array([1, -1, 0, 1])
        c = compute_crystallization(signs, signs.copy())
        assert c == pytest.approx(1.0, abs=1e-5)

    def test_all_different(self):
        a = np.array([1, -1, 0])
        b = np.array([-1, 1, 1])
        c = compute_crystallization(a, b)
        assert c == pytest.approx(0.0, abs=1e-5)


class TestClassifyDomain:
    def test_too_few(self):
        assert classify_domain([0.5, 0.6], 10) == "unknown"

    def test_early(self):
        accs = [0.3, 0.6, 0.85, 0.9, 0.92, 0.93, 0.95, 0.95, 0.96]
        assert classify_domain(accs, 9) == "early"

    def test_late(self):
        accs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]
        assert classify_domain(accs, 9) == "late"

    def test_unstable(self):
        accs = [0.1, 0.5, 0.2, 0.6, 0.3, 0.85, 0.2, 0.7, 0.3]
        assert classify_domain(accs, 9) == "unstable"


class TestHelpers:
    def test_is_oscillating_flat(self):
        assert not is_oscillating([1.0, 1.0, 1.0])

    def test_is_oscillating_yes(self):
        assert is_oscillating([1, 2, 1, 2, 1, 2])

    def test_monotonic_trend_increasing(self):
        assert monotonic_trend([1, 2, 3, 4]) == pytest.approx(1.0)

    def test_monotonic_trend_decreasing(self):
        assert monotonic_trend([4, 3, 2, 1]) == pytest.approx(0.0)

    def test_find_tier_convergence(self):
        assert find_tier_convergence([0.5, 0.6, 0.7, 0.85], 0.8) == 3

    def test_find_tier_convergence_none(self):
        assert find_tier_convergence([0.5, 0.6, 0.7], 0.8) is None

    def test_linear_slope_positive(self):
        s = linear_slope([1.0, 2.0, 3.0, 4.0])
        assert s == pytest.approx(1.0, abs=1e-5)

    def test_linear_slope_flat(self):
        assert linear_slope([5.0, 5.0, 5.0]) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# PAB Profile serialization tests
# ---------------------------------------------------------------------------


class TestPABProfileSerialization:
    def test_roundtrip(self, tmp_path):
        profile = PABProfile(
            experiment_id="TEST-1",
            config_hash="abc123",
            checkpoints=[0, 50, 100],
            core=PABCoreSeries(
                stability=[0.5, 0.3, 0.1],
                predictability=[0.1, 0.05, 0.02],
                generalization_gap=[0.2, 0.15, 0.1],
                representation_evolution=[1.0, 0.5, 0.2],
            ),
            tiers=PABTierSeries(
                tier1_accuracy=[0.3, 0.6, 0.8],
                tier2_accuracy=[0.1, 0.3, 0.5],
                tier3_accuracy=[0.0, 0.1, 0.2],
                ternary_crystallization=[0.0, 0.5, 0.8],
            ),
            summary=PABSummary(
                stability_mean=0.3,
                stability_regime="phase_transition",
            ),
        )

        path = tmp_path / "profile.json"
        profile.save(path)

        loaded = PABProfile.load(path)
        assert loaded.experiment_id == "TEST-1"
        assert loaded.core.stability == [0.5, 0.3, 0.1]
        assert loaded.tiers.tier1_accuracy == [0.3, 0.6, 0.8]
        assert loaded.summary.stability_mean == pytest.approx(0.3)
        assert loaded.summary.stability_regime == "phase_transition"

    def test_json_is_flat(self, tmp_path):
        profile = PABProfile(
            experiment_id="TEST-2",
            core=PABCoreSeries(stability=[0.1, 0.2]),
        )
        path = tmp_path / "flat.json"
        profile.save(path)

        with open(path) as f:
            data = json.load(f)

        # Top-level keys should be flat (not nested under 'core')
        assert "stability" in data
        assert "core" not in data
        assert "summary" in data
        assert isinstance(data["summary"], dict)


# ---------------------------------------------------------------------------
# PAB Tracker tests
# ---------------------------------------------------------------------------


class TestPABTracker:
    def _make_checkpoint(self, step: int, train_loss: float, **kwargs) -> CheckpointData:
        return CheckpointData(step=step, train_loss=train_loss, **kwargs)

    def test_basic_record_and_finalize(self):
        tracker = PABTracker(experiment_id="T-1", checkpoint_interval=50)

        for i in range(5):
            data = self._make_checkpoint(
                step=i * 50,
                train_loss=1.0 - i * 0.15,
                val_loss=1.1 - i * 0.14,
                tier_accuracies={"tier1": 0.2 + i * 0.15, "tier2": 0.1 + i * 0.1, "tier3": 0.0},
            )
            tracker.record(data)

        profile = tracker.finalize()
        assert profile.experiment_id == "T-1"
        assert len(profile.checkpoints) == 5
        assert len(profile.core.stability) == 5
        assert len(profile.tiers.tier1_accuracy) == 5
        assert profile.summary.stability_regime in ("stable", "moderate", "chaotic", "unknown")

    def test_domain_tracking(self):
        tracker = PABTracker(experiment_id="T-2")
        domains = {"health": 0.5, "api": 0.3}

        for i in range(3):
            data = self._make_checkpoint(
                step=i * 50,
                train_loss=0.5,
                domain_accuracies=domains,
            )
            tracker.record(data)

        profile = tracker.finalize()
        assert "health" in profile.domains.domain_progression
        assert "api" in profile.domains.domain_progression
        assert len(profile.domains.domain_progression["health"]) == 3

    def test_crystallization_tracking(self):
        tracker = PABTracker(experiment_id="T-3")

        signs_a = np.array([1, -1, 0, 1, -1])
        signs_b = np.array([1, -1, 0, 1, 1])  # one change

        tracker.record(self._make_checkpoint(0, 1.0, decoder_weight_signs=signs_a))
        tracker.record(self._make_checkpoint(50, 0.8, decoder_weight_signs=signs_b))

        profile = tracker.finalize()
        assert len(profile.tiers.ternary_crystallization) == 2
        assert profile.tiers.ternary_crystallization[0] == 0.0  # no prev
        assert profile.tiers.ternary_crystallization[1] == pytest.approx(0.8, abs=0.01)

    def test_early_exit_not_triggered_early(self):
        tracker = PABTracker(experiment_id="T-4")
        for i in range(3):
            tracker.record(self._make_checkpoint(i * 50, 1.0 - i * 0.1))
        assert not tracker.should_early_exit(100)

    def test_loss_component_tracking(self):
        tracker = PABTracker(experiment_id="T-5")
        data = self._make_checkpoint(
            0,
            1.0,
            loss_components={"ce": 0.5, "margin": 0.3, "repair": 0.2},
            adaptive_weights={"ce": 0.6, "margin": 0.3, "repair": 0.1},
        )
        tracker.record(data)
        profile = tracker.finalize()
        assert profile.losses.loss_ce == [0.5]
        assert profile.losses.loss_margin == [0.3]

    def test_summary_convergence(self):
        tracker = PABTracker(experiment_id="T-6")
        # Simulate training that converges: high stability -> low stability
        losses = [1.0, 0.8, 0.65, 0.55, 0.50, 0.48, 0.47, 0.465, 0.462, 0.461]
        for i, loss in enumerate(losses):
            tracker.record(self._make_checkpoint(i * 50, loss, val_loss=loss + 0.05))

        profile = tracker.finalize()
        assert profile.summary.stability_mean > 0
        # Should detect early stop (val loss always increasing relative to each other)
        # val_losses decrease, so early_stop may or may not trigger
        assert profile.summary.stability_regime in (
            "stable",
            "moderate",
            "chaotic",
            "phase_transition",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
