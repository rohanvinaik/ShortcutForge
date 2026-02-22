"""Tests for PAB profile comparison and ranking."""

from __future__ import annotations

import pytest

from src.pab_comparison import (
    ComparisonReport,
    compare_profiles,
    trajectory_correlation,
)
from src.pab_profile import PABCoreSeries, PABProfile, PABSummary, PABTierSeries


def _make_profile(
    exp_id: str,
    stability_mean: float = 0.2,
    predictability: float = 0.05,
    tier1_accs: list[float] | None = None,
    tier2_accs: list[float] | None = None,
    crystallization_rate: float = 0.01,
    convergence_epoch: int | None = None,
    stability_regime: str = "moderate",
) -> PABProfile:
    """Build a minimal PABProfile for testing."""
    t1 = tier1_accs or [0.5, 0.7, 0.85]
    t2 = tier2_accs or [0.3, 0.5, 0.65]
    return PABProfile(
        experiment_id=exp_id,
        core=PABCoreSeries(stability=[stability_mean] * 3),
        tiers=PABTierSeries(
            tier1_accuracy=t1,
            tier2_accuracy=t2,
        ),
        summary=PABSummary(
            stability_mean=stability_mean,
            predictability_final=predictability,
            crystallization_rate=crystallization_rate,
            convergence_epoch=convergence_epoch,
            stability_regime=stability_regime,
        ),
    )


class TestCompareProfiles:
    def test_single_profile(self):
        report = compare_profiles([_make_profile("A")])
        assert report.profile_ids == ["A"]
        assert len(report.rankings) == 0

    def test_two_profiles_stability(self):
        stable = _make_profile("stable", stability_mean=0.05)
        chaotic = _make_profile("chaotic", stability_mean=0.40)
        report = compare_profiles([stable, chaotic])

        assert report.best_by["stability_mean"] == "stable"
        assert report.rankings["stability_mean"] == ["stable", "chaotic"]

    def test_two_profiles_accuracy(self):
        good = _make_profile("good", tier1_accs=[0.5, 0.8, 0.95])
        bad = _make_profile("bad", tier1_accs=[0.3, 0.4, 0.50])
        report = compare_profiles([good, bad])

        assert report.best_by["tier1_final_acc"] == "good"

    def test_three_profiles_ranking(self):
        a = _make_profile("A", stability_mean=0.30)
        b = _make_profile("B", stability_mean=0.10)
        c = _make_profile("C", stability_mean=0.20)
        report = compare_profiles([a, b, c])

        assert report.rankings["stability_mean"] == ["B", "C", "A"]

    def test_convergence_speed(self):
        fast = _make_profile("fast", convergence_epoch=100)
        slow = _make_profile("slow", convergence_epoch=400)
        report = compare_profiles([fast, slow])

        assert report.best_by["convergence_speed"] == "fast"

    def test_deltas_computed(self):
        a = _make_profile("A", stability_mean=0.10, tier1_accs=[0.9])
        b = _make_profile("B", stability_mean=0.30, tier1_accs=[0.7])
        report = compare_profiles([a, b])

        assert len(report.deltas) == 1
        delta = report.deltas[0]
        assert delta.id_a == "A"
        assert delta.id_b == "B"
        assert delta.stability_diff == pytest.approx(-0.20)
        assert delta.tier1_acc_diff == pytest.approx(0.20)

    def test_regime_match(self):
        a = _make_profile("A", stability_regime="stable")
        b = _make_profile("B", stability_regime="stable")
        report = compare_profiles([a, b])
        assert report.deltas[0].regime_match is True

    def test_regime_mismatch(self):
        a = _make_profile("A", stability_regime="stable")
        b = _make_profile("B", stability_regime="chaotic")
        report = compare_profiles([a, b])
        assert report.deltas[0].regime_match is False


class TestSummaryTable:
    def test_summary_not_empty(self):
        profiles = [_make_profile("A"), _make_profile("B")]
        report = compare_profiles(profiles)
        table = report.summary_table()
        assert "PAB Profile Comparison" in table
        assert "stability_mean" in table

    def test_empty_report(self):
        report = ComparisonReport()
        table = report.summary_table()
        assert "No profiles" in table


class TestTrajectoryCorrelation:
    def test_too_few_profiles(self):
        assert trajectory_correlation([_make_profile("A")]) == 0.0

    def test_perfect_positive(self):
        profiles = [
            _make_profile("A", stability_mean=0.1, tier1_accs=[0.9]),
            _make_profile("B", stability_mean=0.2, tier1_accs=[0.8]),
            _make_profile("C", stability_mean=0.3, tier1_accs=[0.7]),
        ]
        # Lower stability = higher accuracy -> negative correlation for stability
        # (stability_mean is "badness", lower = better learning, higher = worse)
        rho = trajectory_correlation(profiles, "stability")
        assert rho < 0  # Anti-correlated: low stability_mean -> high acc

    def test_unknown_metric(self):
        profiles = [_make_profile("A"), _make_profile("B"), _make_profile("C")]
        assert trajectory_correlation(profiles, "nonexistent") == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
