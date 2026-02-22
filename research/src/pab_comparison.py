"""PAB profile comparison for ablation analysis.

Overlays PAB profiles from multiple training runs to identify which
configurations produce better learning trajectories. Used in Phase 4
ablation analysis and Phase 5 PAB framework validation.

Usage:
    profiles = [PABProfile.load(p) for p in profile_paths]
    report = compare_profiles(profiles)
    print(report.summary_table())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.pab_profile import PABProfile


@dataclass
class ProfileDelta:
    """Pairwise difference between two PAB profiles."""

    id_a: str
    id_b: str
    stability_diff: float = 0.0
    predictability_diff: float = 0.0
    tier1_acc_diff: float = 0.0
    tier2_acc_diff: float = 0.0
    crystallization_diff: float = 0.0
    convergence_speed_diff: int | None = None
    regime_match: bool = False


@dataclass
class ComparisonReport:
    """Result of comparing multiple PAB profiles."""

    profile_ids: list[str] = field(default_factory=list)
    rankings: dict[str, list[str]] = field(default_factory=dict)
    deltas: list[ProfileDelta] = field(default_factory=list)
    best_by: dict[str, str] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Format a human-readable comparison table."""
        lines = ["PAB Profile Comparison", "=" * 60]

        if not self.profile_ids:
            return "\n".join(lines + ["No profiles to compare."])

        lines.append(f"Profiles compared: {len(self.profile_ids)}")
        lines.append("")

        # Best-by metrics
        lines.append("Best configuration by metric:")
        for metric, winner in self.best_by.items():
            lines.append(f"  {metric:30s} -> {winner}")

        # Rankings
        if self.rankings:
            lines.append("")
            lines.append("Rankings (best to worst):")
            for metric, ranked in self.rankings.items():
                lines.append(f"  {metric:30s}: {' > '.join(ranked)}")

        return "\n".join(lines)


def compare_profiles(profiles: list[PABProfile]) -> ComparisonReport:
    """Compare multiple PAB profiles and produce a ranking report."""
    if len(profiles) < 2:
        return ComparisonReport(
            profile_ids=[p.experiment_id for p in profiles],
        )

    report = ComparisonReport(
        profile_ids=[p.experiment_id for p in profiles],
    )

    # Rank by each key metric (lower is better for stability/predictability,
    # higher is better for accuracy/crystallization)
    _rank_profiles(profiles, report)
    _compute_deltas(profiles, report)

    return report


def _rank_profiles(profiles: list[PABProfile], report: ComparisonReport) -> None:
    """Rank profiles by each metric and determine winners."""
    ids = [p.experiment_id for p in profiles]

    # Lower is better
    lower_metrics = {
        "stability_mean": lambda p: p.summary.stability_mean,
        "predictability": lambda p: p.summary.predictability_final,
    }

    # Higher is better
    higher_metrics = {
        "tier1_final_acc": lambda p: p.tiers.tier1_accuracy[-1] if p.tiers.tier1_accuracy else 0,
        "tier2_final_acc": lambda p: p.tiers.tier2_accuracy[-1] if p.tiers.tier2_accuracy else 0,
        "crystallization_rate": lambda p: p.summary.crystallization_rate,
    }

    # Earlier is better (convergence step)
    def conv_step(p: PABProfile) -> float:
        return (
            p.summary.convergence_epoch if p.summary.convergence_epoch is not None else float("inf")
        )

    for name, extractor in lower_metrics.items():
        values = [extractor(p) for p in profiles]
        ranked_idx = sorted(range(len(values)), key=lambda i: values[i])
        report.rankings[name] = [ids[i] for i in ranked_idx]
        report.best_by[name] = ids[ranked_idx[0]]

    for name, extractor in higher_metrics.items():
        values = [extractor(p) for p in profiles]
        ranked_idx = sorted(range(len(values)), key=lambda i: -values[i])
        report.rankings[name] = [ids[i] for i in ranked_idx]
        report.best_by[name] = ids[ranked_idx[0]]

    conv_values = [conv_step(p) for p in profiles]
    ranked_idx = sorted(range(len(conv_values)), key=lambda i: conv_values[i])
    report.rankings["convergence_speed"] = [ids[i] for i in ranked_idx]
    report.best_by["convergence_speed"] = ids[ranked_idx[0]]


def _compute_deltas(profiles: list[PABProfile], report: ComparisonReport) -> None:
    """Compute pairwise deltas between all profile pairs."""
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            report.deltas.append(_pairwise_delta(profiles[i], profiles[j]))


def _pairwise_delta(a: PABProfile, b: PABProfile) -> ProfileDelta:
    """Compute the difference between two profiles."""
    t1a = a.tiers.tier1_accuracy[-1] if a.tiers.tier1_accuracy else 0
    t1b = b.tiers.tier1_accuracy[-1] if b.tiers.tier1_accuracy else 0
    t2a = a.tiers.tier2_accuracy[-1] if a.tiers.tier2_accuracy else 0
    t2b = b.tiers.tier2_accuracy[-1] if b.tiers.tier2_accuracy else 0

    conv_diff = None
    if a.summary.convergence_epoch is not None and b.summary.convergence_epoch is not None:
        conv_diff = a.summary.convergence_epoch - b.summary.convergence_epoch

    return ProfileDelta(
        id_a=a.experiment_id,
        id_b=b.experiment_id,
        stability_diff=a.summary.stability_mean - b.summary.stability_mean,
        predictability_diff=a.summary.predictability_final - b.summary.predictability_final,
        tier1_acc_diff=t1a - t1b,
        tier2_acc_diff=t2a - t2b,
        crystallization_diff=a.summary.crystallization_rate - b.summary.crystallization_rate,
        convergence_speed_diff=conv_diff,
        regime_match=a.summary.stability_regime == b.summary.stability_regime,
    )


def trajectory_correlation(profiles: list[PABProfile], metric: str = "stability") -> float:
    """Compute Spearman rank correlation between a trajectory metric and final accuracy.

    Tests PAB Claim 2 (H15): trajectory stability predicts generalization.
    """
    if len(profiles) < 3:
        return 0.0

    extractors = {
        "stability": lambda p: p.summary.stability_mean,
        "predictability": lambda p: p.summary.predictability_final,
        "crystallization": lambda p: p.summary.crystallization_rate,
    }

    extractor = extractors.get(metric)
    if not extractor:
        return 0.0

    metric_values = np.array([extractor(p) for p in profiles])
    acc_values = np.array(
        [p.tiers.tier1_accuracy[-1] if p.tiers.tier1_accuracy else 0 for p in profiles]
    )

    # Spearman rank correlation
    metric_ranks = _rank_array(metric_values)
    acc_ranks = _rank_array(acc_values)

    n = len(profiles)
    d_sq = np.sum((metric_ranks - acc_ranks) ** 2)
    rho = 1 - (6 * d_sq) / (n * (n**2 - 1))
    return float(rho)


def _rank_array(values: np.ndarray) -> np.ndarray:
    """Compute ranks (1-based) for an array of values."""
    order = values.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_json(report: ComparisonReport) -> dict:
    """Serialize a ComparisonReport to a JSON-compatible dict."""
    deltas_list = []
    for d in report.deltas:
        deltas_list.append(
            {
                "id_a": d.id_a,
                "id_b": d.id_b,
                "stability_diff": d.stability_diff,
                "predictability_diff": d.predictability_diff,
                "tier1_acc_diff": d.tier1_acc_diff,
                "tier2_acc_diff": d.tier2_acc_diff,
                "crystallization_diff": d.crystallization_diff,
                "convergence_speed_diff": d.convergence_speed_diff,
                "regime_match": d.regime_match,
            }
        )
    return {
        "profile_ids": report.profile_ids,
        "rankings": report.rankings,
        "best_by": report.best_by,
        "deltas": deltas_list,
    }


def export_markdown(report: ComparisonReport) -> str:
    """Render a ComparisonReport as formatted Markdown.

    Suitable for pasting into EXPERIMENT_RESULTS.md.
    """
    lines = [
        "## PAB Profile Comparison",
        "",
        f"**Profiles compared:** {len(report.profile_ids)}",
        f"**IDs:** {', '.join(report.profile_ids)}",
        "",
    ]

    # Best-by table
    if report.best_by:
        lines.append("### Best Configuration by Metric")
        lines.append("")
        lines.append("| Metric | Winner |")
        lines.append("|--------|--------|")
        for metric, winner in report.best_by.items():
            lines.append(f"| {metric} | {winner} |")
        lines.append("")

    # Rankings
    if report.rankings:
        lines.append("### Rankings (best to worst)")
        lines.append("")
        for metric, ranked in report.rankings.items():
            lines.append(f"- **{metric}**: {' > '.join(ranked)}")
        lines.append("")

    # Pairwise deltas
    if report.deltas:
        lines.append("### Pairwise Deltas")
        lines.append("")
        lines.append("| A | B | Stability | Predictability | Tier1 Acc | Crystallization |")
        lines.append("|---|---|-----------|----------------|-----------|-----------------|")
        for d in report.deltas:
            lines.append(
                f"| {d.id_a} | {d.id_b} "
                f"| {d.stability_diff:+.4f} "
                f"| {d.predictability_diff:+.4f} "
                f"| {d.tier1_acc_diff:+.4f} "
                f"| {d.crystallization_diff:+.4f} |"
            )
        lines.append("")

    return "\n".join(lines)
