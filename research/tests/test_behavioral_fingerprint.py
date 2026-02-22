"""Tests for behavioral fingerprinting module."""

from __future__ import annotations

import numpy as np
import pytest

from src.behavioral_fingerprint import (
    BehavioralFingerprint,
    compute_action_distribution,
    compute_action_entropy,
    compute_discreteness,
    compute_variance_eigenvalues,
    fingerprint_stability,
)


class TestActionEntropy:
    def test_empty(self):
        assert compute_action_entropy([]) == 0.0

    def test_single_action(self):
        entropy = compute_action_entropy(["a", "a", "a"])
        assert entropy == pytest.approx(0.0, abs=1e-5)

    def test_uniform_two(self):
        entropy = compute_action_entropy(["a", "b"])
        assert entropy == pytest.approx(1.0, abs=0.01)

    def test_more_actions_higher_entropy(self):
        e2 = compute_action_entropy(["a", "b"])
        e4 = compute_action_entropy(["a", "b", "c", "d"])
        assert e4 > e2


class TestActionDistribution:
    def test_empty(self):
        assert compute_action_distribution([]) == {}

    def test_uniform(self):
        dist = compute_action_distribution(["a", "b"])
        assert dist["a"] == pytest.approx(0.5)
        assert dist["b"] == pytest.approx(0.5)

    def test_sorted_keys(self):
        dist = compute_action_distribution(["c", "a", "b", "a"])
        assert list(dist.keys()) == ["a", "b", "c"]


class TestVarianceEigenvalues:
    def test_too_few_rows(self):
        assert compute_variance_eigenvalues(np.array([[1, 2, 3]])) == []

    def test_returns_top_k(self):
        rng = np.random.default_rng(42)
        logits = rng.normal(size=(20, 50))
        eigs = compute_variance_eigenvalues(logits, top_k=5)
        assert len(eigs) == 5
        # Should be sorted descending
        assert all(eigs[i] >= eigs[i + 1] for i in range(len(eigs) - 1))


class TestDiscreteness:
    def test_empty(self):
        assert compute_discreteness(np.array([])) == 0.0

    def test_peaked(self):
        # Very peaked: one logit dominates
        logits = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        d = compute_discreteness(logits)
        assert d > 0.8  # Should be very discrete

    def test_uniform(self):
        # All logits equal -> uniform -> low discreteness
        logits = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        d = compute_discreteness(logits)
        assert d < 0.1  # Should be near zero


class TestFingerprint:
    def test_from_outputs(self):
        rng = np.random.default_rng(42)
        logits = rng.normal(size=(10, 20))
        preds = ["action_a"] * 6 + ["action_b"] * 4

        fp = BehavioralFingerprint.from_outputs(
            experiment_id="T-1",
            step=100,
            output_logits=logits,
            action_predictions=preds,
            probe_labels=[f"probe_{i}" for i in range(10)],
        )

        assert fp.experiment_id == "T-1"
        assert fp.step == 100
        assert fp.action_entropy > 0
        assert "action_a" in fp.action_distribution
        assert len(fp.variance_eigenvalues) > 0
        assert len(fp.probe_responses) == 10

    def test_roundtrip(self, tmp_path):
        fp = BehavioralFingerprint(
            experiment_id="T-2",
            step=200,
            action_entropy=1.5,
            action_distribution={"a": 0.6, "b": 0.4},
            variance_eigenvalues=[5.0, 2.0, 1.0],
            discreteness_score=0.7,
        )

        path = tmp_path / "fp.json"
        fp.save(path)
        loaded = BehavioralFingerprint.load(path)

        assert loaded.experiment_id == "T-2"
        assert loaded.step == 200
        assert loaded.action_entropy == pytest.approx(1.5)
        assert loaded.action_distribution == {"a": 0.6, "b": 0.4}


class TestFingerprintStability:
    def test_too_few(self):
        result = fingerprint_stability([BehavioralFingerprint()])
        assert result["entropy_variance"] == 0.0

    def test_stable_fingerprints(self):
        fps = [
            BehavioralFingerprint(
                step=i * 50,
                action_entropy=1.0,
                action_distribution={"a": 0.5, "b": 0.5},
                discreteness_score=0.5,
            )
            for i in range(5)
        ]
        result = fingerprint_stability(fps)
        assert result["entropy_variance"] == pytest.approx(0.0)
        assert result["distribution_drift"] == pytest.approx(0.0, abs=1e-5)

    def test_unstable_fingerprints(self):
        fps = [
            BehavioralFingerprint(
                step=0,
                action_entropy=0.5,
                action_distribution={"a": 0.9, "b": 0.1},
                discreteness_score=0.9,
            ),
            BehavioralFingerprint(
                step=50,
                action_entropy=2.0,
                action_distribution={"a": 0.3, "b": 0.3, "c": 0.4},
                discreteness_score=0.2,
            ),
        ]
        result = fingerprint_stability(fps)
        assert result["entropy_variance"] > 0
        assert result["distribution_drift"] > 0
        assert result["discreteness_variance"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
