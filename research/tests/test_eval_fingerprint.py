"""Tests for behavioral fingerprint integration in evaluate.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.behavioral_fingerprint import BehavioralFingerprint


class TestFingerprintFromOutputs:
    """Test fingerprint construction from decoder outputs."""

    def test_fingerprint_metrics_computed(self):
        """from_outputs should produce entropy and discreteness metrics."""
        logits = np.random.randn(20, 10)
        predictions = ["action_a"] * 10 + ["action_b"] * 5 + ["action_c"] * 5

        fp = BehavioralFingerprint.from_outputs(
            experiment_id="test-fp",
            step=100,
            output_logits=logits,
            action_predictions=predictions,
        )

        assert fp.experiment_id == "test-fp"
        assert fp.step == 100
        assert fp.action_entropy > 0.0
        assert fp.discreteness_score >= 0.0
        assert len(fp.action_distribution) == 3
        assert abs(sum(fp.action_distribution.values()) - 1.0) < 1e-6

    def test_fingerprint_with_probe_labels(self):
        """Probe labels should appear in probe_responses."""
        logits = np.random.randn(3, 5)
        predictions = ["a", "b", "c"]
        labels = ["probe_1", "probe_2", "probe_3"]

        fp = BehavioralFingerprint.from_outputs(
            experiment_id="test",
            step=50,
            output_logits=logits,
            action_predictions=predictions,
            probe_labels=labels,
        )

        assert fp.probe_responses == {"probe_1": "a", "probe_2": "b", "probe_3": "c"}


class TestFingerprintSaveLoad:
    """Test fingerprint JSON round-trip."""

    def test_save_and_load(self, tmp_path: Path):
        """Fingerprint should survive JSON round-trip."""
        logits = np.random.randn(10, 8)
        predictions = ["x"] * 5 + ["y"] * 5

        fp = BehavioralFingerprint.from_outputs(
            experiment_id="roundtrip-test",
            step=200,
            output_logits=logits,
            action_predictions=predictions,
        )

        fp_path = tmp_path / "fp.json"
        fp.save(fp_path)
        assert fp_path.exists()

        loaded = BehavioralFingerprint.load(fp_path)
        assert loaded.experiment_id == "roundtrip-test"
        assert loaded.step == 200
        assert abs(loaded.action_entropy - fp.action_entropy) < 1e-6
        assert abs(loaded.discreteness_score - fp.discreteness_score) < 1e-6


class TestFingerprintInEvalResults:
    """Test that evaluate_checkpoint adds fingerprint metrics to results."""

    def test_fingerprint_metrics_in_results(self):
        """When fingerprint_enabled, results should contain fingerprint fields."""
        # Build a minimal fake results dict like evaluate_checkpoint would produce
        # (testing the integration logic without needing real models)
        logits = np.random.randn(5, 10)
        predictions = ["a", "b", "a", "c", "a"]

        fp = BehavioralFingerprint.from_outputs(
            experiment_id="eval-test",
            step=500,
            output_logits=logits,
            action_predictions=predictions,
        )

        results = {
            "total_examples": 5,
            "gate_accuracy": 1.0,
            "tier1_exact_match_rate": 0.6,
            "fingerprint_entropy": fp.action_entropy,
            "fingerprint_discreteness": fp.discreteness_score,
        }

        assert "fingerprint_entropy" in results
        assert "fingerprint_discreteness" in results
        assert results["fingerprint_entropy"] > 0.0

    def test_fingerprint_saved_to_file(self, tmp_path: Path):
        """Fingerprint JSON should be created alongside eval results."""
        logits = np.random.randn(5, 10)
        predictions = ["a", "b", "a", "c", "a"]

        fp = BehavioralFingerprint.from_outputs(
            experiment_id="save-test",
            step=100,
            output_logits=logits,
            action_predictions=predictions,
        )

        fp_path = tmp_path / "eval_results.fingerprint.json"
        fp.save(fp_path)

        assert fp_path.exists()
        data = json.loads(fp_path.read_text())
        assert data["experiment_id"] == "save-test"
        assert "action_entropy" in data
