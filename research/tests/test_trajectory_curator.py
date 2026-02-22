"""Tests for PAB-integrated trajectory curator."""

from __future__ import annotations

import pytest

from src.pab_profile import PABProfile, PABSummary
from src.trajectory_curator import (
    DifficultyReport,
    ExampleDifficulty,
    ProbeResult,
    _classify_single,
    _spearman_rho,
    build_difficulty_report,
    classify_examples,
)

# ---------------------------------------------------------------------------
# _spearman_rho tests
# ---------------------------------------------------------------------------


class TestSpearmanRho:
    def test_perfect_increasing(self):
        rho = _spearman_rho([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rho == pytest.approx(1.0, abs=1e-6)

    def test_perfect_decreasing(self):
        rho = _spearman_rho([5.0, 4.0, 3.0, 2.0, 1.0])
        assert rho == pytest.approx(-1.0, abs=1e-6)

    def test_too_few_values(self):
        assert _spearman_rho([1.0]) == 0.0
        assert _spearman_rho([1.0, 2.0]) == 0.0

    def test_flat_sequence(self):
        # All same values → d_sq=0 → rho=1.0 (degenerate case)
        rho = _spearman_rho([3.0, 3.0, 3.0, 3.0])
        assert rho == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# _classify_single tests — PAB-calibrated classification
# ---------------------------------------------------------------------------


class TestClassifySingle:
    """Test PAB-calibrated classification logic."""

    def test_empty_losses_returns_unknown(self):
        result = _classify_single([], 1.0, 0.1, 0.05, "stable")
        assert result == "unknown"

    def test_easy_low_monotonic_loss(self):
        # Final loss < aggregate_mean × 0.5, monotonically decreasing
        losses = [0.5, 0.4, 0.3, 0.2, 0.1]
        result = _classify_single(losses, 1.0, 0.1, 0.05, "stable")
        assert result == "easy"

    def test_not_easy_if_not_monotonic(self):
        # Low final loss but NOT monotonically decreasing
        losses = [0.5, 0.6, 0.3, 0.2, 0.1]
        result = _classify_single(losses, 1.0, 0.1, 0.05, "stable")
        assert result != "easy"

    def test_destabilizing_high_instability(self):
        # Large per-example stability jump exceeds threshold
        losses = [1.0, 1.0, 1.0, 5.0, 1.0]
        result = _classify_single(losses, 2.0, 0.1, 0.05, "stable")
        assert result == "destabilizing"

    def test_hard_but_learnable_negative_trend(self):
        # Clear downward trend (rho < -0.3) but not low enough for easy
        losses = [2.0, 1.8, 1.5, 1.3, 1.1]
        result = _classify_single(losses, 2.0, 0.1, 0.05, "stable")
        assert result == "hard_but_learnable"

    def test_unlearnable_flat_trend(self):
        # Flat/rising trend (rho > -0.1) in non-chaotic regime
        losses = [1.0, 1.1, 1.0, 1.1, 1.0]
        result = _classify_single(losses, 2.0, 0.1, 0.05, "stable")
        assert result == "unlearnable"

    def test_chaotic_regime_not_unlearnable(self):
        # Flat trend but chaotic regime — should not be classified as unlearnable
        losses = [1.0, 1.1, 1.0, 1.1, 1.0]
        result = _classify_single(losses, 2.0, 0.1, 0.05, "chaotic")
        assert result != "unlearnable"

    def test_moderate_trend_gives_benefit(self):
        # Moderate trend (-0.3 < rho < -0.1) → hard_but_learnable
        losses = [2.0, 1.9, 1.85, 1.8, 1.75]
        result = _classify_single(losses, 2.0, 0.1, 0.05, "stable")
        assert result == "hard_but_learnable"


# ---------------------------------------------------------------------------
# classify_examples integration test
# ---------------------------------------------------------------------------


class _FakeExample:
    """Minimal stand-in for a dataset example."""

    def __init__(self, prompt: str):
        self.prompt = prompt


class _FakeDataset:
    """Minimal indexable dataset for testing."""

    def __init__(self, n: int):
        self._examples = [_FakeExample(f"prompt_{i}") for i in range(n)]

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]


class TestClassifyExamples:
    def _make_probe_result(self, per_example_losses, summary=None):
        if summary is None:
            summary = PABSummary(
                stability_mean=0.1,
                stability_std=0.05,
                stability_regime="stable",
            )
        profile = PABProfile(
            experiment_id="test",
            checkpoints=[50, 100, 150, 200, 250],
            summary=summary,
        )
        return ProbeResult(per_example_losses=per_example_losses, pab_profile=profile)

    def test_classifies_all_examples(self):
        dataset = _FakeDataset(3)
        losses = {
            0: [0.5, 0.4, 0.3, 0.2, 0.1],  # easy
            1: [2.0, 1.8, 1.5, 1.3, 1.1],  # hard_but_learnable
            2: [1.0, 1.1, 1.0, 1.1, 1.0],  # unlearnable
        }
        probe = self._make_probe_result(losses)
        examples = classify_examples(probe, dataset)

        assert len(examples) == 3
        assert examples[0].classification == "easy"
        assert examples[1].classification == "hard_but_learnable"
        assert examples[2].classification == "unlearnable"

    def test_empty_dataset(self):
        dataset = _FakeDataset(0)
        probe = self._make_probe_result({})
        examples = classify_examples(probe, dataset)
        assert examples == []

    def test_prompt_truncation(self):
        dataset = _FakeDataset(1)
        dataset._examples[0].prompt = "x" * 200
        losses = {0: [1.0, 0.9, 0.8]}
        probe = self._make_probe_result(losses)
        examples = classify_examples(probe, dataset)
        assert len(examples[0].prompt) <= 120


# ---------------------------------------------------------------------------
# build_difficulty_report tests
# ---------------------------------------------------------------------------


class TestBuildDifficultyReport:
    def test_counts_and_summary(self):
        examples = [
            ExampleDifficulty(index=0, classification="easy"),
            ExampleDifficulty(index=1, classification="easy"),
            ExampleDifficulty(index=2, classification="unlearnable"),
        ]
        summary = PABSummary(stability_mean=0.1, stability_std=0.05, stability_regime="stable")
        profile = PABProfile(
            experiment_id="test",
            checkpoints=[50, 100, 150, 200, 250],
            summary=summary,
        )
        probe = ProbeResult(per_example_losses={}, pab_profile=profile)

        report = build_difficulty_report(examples, probe)
        assert report.total_examples == 3
        assert report.counts == {"easy": 2, "unlearnable": 1}
        assert report.pab_summary["stability_mean"] == 0.1
        assert report.pab_summary["stability_regime"] == "stable"


# ---------------------------------------------------------------------------
# DifficultyReport serialization
# ---------------------------------------------------------------------------


class TestDifficultyReportSerialization:
    def test_save_and_load_roundtrip(self, tmp_path):
        report = DifficultyReport(
            total_examples=2,
            probe_steps=250,
            counts={"easy": 1, "hard_but_learnable": 1},
            pab_summary={"stability_mean": 0.1, "stability_regime": "stable"},
            examples=[
                ExampleDifficulty(
                    index=0,
                    prompt="test prompt",
                    classification="easy",
                    loss_trajectory=[0.5, 0.3, 0.1],
                    spearman_rho=-0.9,
                    mean_loss=0.3,
                    final_loss=0.1,
                ),
                ExampleDifficulty(
                    index=1,
                    prompt="another prompt",
                    classification="hard_but_learnable",
                    loss_trajectory=[1.0, 0.8, 0.6],
                    spearman_rho=-0.5,
                    mean_loss=0.8,
                    final_loss=0.6,
                ),
            ],
        )

        path = tmp_path / "report.json"
        report.save(path)

        loaded = DifficultyReport.load(path)
        assert loaded.total_examples == 2
        assert loaded.counts == {"easy": 1, "hard_but_learnable": 1}
        assert loaded.pab_summary["stability_mean"] == 0.1
        assert len(loaded.examples) == 2
        assert loaded.examples[0].classification == "easy"
        assert loaded.examples[1].final_loss == pytest.approx(0.6)
