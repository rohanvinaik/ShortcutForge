"""Cross-model PAB comparison with H1-H6 hypothesis testing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.pab_comparison import ComparisonReport, compare_profiles
from src.pab_profile import PABProfile


@dataclass
class ArchitectureComparisonReport:
    """Cross-architecture comparison with hypothesis test results."""

    model_specs: list[Any] = field(default_factory=list)
    profiles: list[PABProfile] = field(default_factory=list)
    hypothesis_results: dict[str, dict] = field(default_factory=dict)
    architecture_groups: dict[str, list[str]] = field(default_factory=dict)
    comparison: ComparisonReport = field(default_factory=ComparisonReport)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "model_ids": [p.experiment_id for p in self.profiles],
            "architecture_groups": self.architecture_groups,
            "hypothesis_results": self.hypothesis_results,
            "comparison": {
                "profile_ids": self.comparison.profile_ids,
                "rankings": self.comparison.rankings,
                "best_by": self.comparison.best_by,
            },
        }


def compare_by_architecture(
    profiles: list[PABProfile],
    model_specs: list[Any],
) -> ArchitectureComparisonReport:
    """Run cross-architecture comparison with H1-H6 hypothesis tests."""
    # Build architecture groups
    spec_map = {s.model_id: s for s in model_specs}
    arch_groups: dict[str, list[str]] = {}
    for p in profiles:
        spec = spec_map.get(p.experiment_id)
        family = spec.architecture_family if spec else "unknown"
        arch_groups.setdefault(family, []).append(p.experiment_id)

    comparison = compare_profiles(profiles)

    report = ArchitectureComparisonReport(
        model_specs=model_specs,
        profiles=profiles,
        architecture_groups=arch_groups,
        comparison=comparison,
    )

    # Run hypothesis tests
    report.hypothesis_results["H1"] = check_h1_architecture_signature(profiles, arch_groups)
    report.hypothesis_results["H2"] = check_h2_attention_mechanism(profiles, arch_groups)
    report.hypothesis_results["H3"] = check_h3_moe_routing(profiles, arch_groups)
    report.hypothesis_results["H4"] = check_h4_specialization_transfer(profiles, model_specs)
    report.hypothesis_results["H5"] = check_h5_scale_process_decoupling(profiles, model_specs)
    report.hypothesis_results["H6"] = check_h6_taxonomy_predicts_trainability(None, profiles)

    return report


_RECURRENT_FAMILIES = {"linear_rnn", "ssm", "griffin", "modern_lstm"}
_ATTENTION_FAMILIES = {"dense", "dense_gqa", "qwen3_gqa", "mistral_swa"}
_MOE_FAMILIES = {"qwen3_moe", "hybrid_ssm_moe"}


def _insufficient(msg: str, **extra: float) -> dict:
    """Standard return for insufficient data."""
    return {"statistic": 0.0, "p_value": 1.0, "interpretation": msg, "effect_size": 0.0, **extra}


def check_h1_architecture_signature(
    profiles: list[PABProfile],
    arch_groups: dict[str, list[str]],
) -> dict:
    """H1: ANOVA on stability_mean across architecture families."""
    profile_map = {p.experiment_id: p for p in profiles}
    groups = []
    for family, ids in arch_groups.items():
        vals = [profile_map[eid].summary.stability_mean for eid in ids if eid in profile_map]
        if vals:
            groups.append(vals)

    if len(groups) < 2:
        return _insufficient("insufficient groups for ANOVA")

    try:
        from scipy.stats import f_oneway

        stat, p_val = f_oneway(*groups)
        all_vals = [v for g in groups for v in g]
        grand_mean = np.mean(all_vals)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
        eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
        interp = "significant" if p_val < 0.05 else "not significant"
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "interpretation": f"Architecture effect {interp} (p={p_val:.4f})",
            "effect_size": eta_sq,
        }
    except ImportError:
        return _scipy_unavailable("H1: ANOVA on stability signatures")


def check_h2_attention_mechanism(
    profiles: list[PABProfile],
    arch_groups: dict[str, list[str]],
) -> dict:
    """H2: Mann-Whitney U on repr_evolution between recurrent and attention."""
    profile_map = {p.experiment_id: p for p in profiles}
    recurrent_vals = _gather_repr_evolution(profile_map, arch_groups, _RECURRENT_FAMILIES)
    attention_vals = _gather_repr_evolution(profile_map, arch_groups, _ATTENTION_FAMILIES)
    r_mean = float(np.mean(recurrent_vals)) if recurrent_vals else 0.0
    a_mean = float(np.mean(attention_vals)) if attention_vals else 0.0

    if not recurrent_vals or not attention_vals:
        return _insufficient(
            "insufficient data for recurrent vs attention",
            recurrent_mean=r_mean,
            attention_mean=a_mean,
        )
    try:
        from scipy.stats import mannwhitneyu

        stat, p_val = mannwhitneyu(recurrent_vals, attention_vals, alternative="two-sided")
        interp = "significant" if p_val < 0.05 else "not significant"
        return {
            "statistic": float(stat),
            "p_value": float(p_val),
            "interpretation": f"Attention mechanism effect {interp} (p={p_val:.4f})",
            "effect_size": abs(r_mean - a_mean),
            "recurrent_mean": r_mean,
            "attention_mean": a_mean,
        }
    except ImportError:
        return _scipy_unavailable("H2: Mann-Whitney U on repr evolution")


def check_h3_moe_routing(
    profiles: list[PABProfile],
    arch_groups: dict[str, list[str]],
) -> dict:
    """H3: Crystallization rate comparison between MoE and dense models."""
    profile_map = {p.experiment_id: p for p in profiles}

    moe_rates = []
    dense_rates = []
    for family, ids in arch_groups.items():
        for eid in ids:
            if eid not in profile_map:
                continue
            rate = profile_map[eid].summary.crystallization_rate
            if family in _MOE_FAMILIES:
                moe_rates.append(rate)
            else:
                dense_rates.append(rate)

    if not moe_rates or not dense_rates:
        return {
            "moe_mean_crystallization": float(np.mean(moe_rates)) if moe_rates else 0.0,
            "dense_mean_crystallization": float(np.mean(dense_rates)) if dense_rates else 0.0,
            "interpretation": "insufficient data for MoE vs dense comparison",
        }

    return {
        "moe_mean_crystallization": float(np.mean(moe_rates)),
        "dense_mean_crystallization": float(np.mean(dense_rates)),
        "difference": float(np.mean(moe_rates) - np.mean(dense_rates)),
        "interpretation": (
            "MoE crystallizes faster"
            if np.mean(moe_rates) > np.mean(dense_rates)
            else "Dense crystallizes faster"
        ),
    }


def check_h4_specialization_transfer(
    profiles: list[PABProfile],
    model_specs: list[Any],
) -> dict:
    """H4: Code-specialized vs general convergence_epoch comparison."""
    spec_map = {s.model_id: s for s in model_specs}

    code_convergence = []
    general_convergence = []

    for p in profiles:
        conv = p.summary.convergence_epoch
        if conv is None:
            continue
        spec = spec_map.get(p.experiment_id)
        if spec and "code" in spec.display_name.lower():
            code_convergence.append(conv)
        else:
            general_convergence.append(conv)

    if not code_convergence or not general_convergence:
        return {
            "code_mean_convergence": float(np.mean(code_convergence)) if code_convergence else 0.0,
            "general_mean_convergence": (
                float(np.mean(general_convergence)) if general_convergence else 0.0
            ),
            "interpretation": "insufficient data for specialization comparison",
        }

    return {
        "code_mean_convergence": float(np.mean(code_convergence)),
        "general_mean_convergence": float(np.mean(general_convergence)),
        "speedup_ratio": float(np.mean(general_convergence) / np.mean(code_convergence)),
        "interpretation": (
            "Code-specialized converges faster"
            if np.mean(code_convergence) < np.mean(general_convergence)
            else "General models converge faster or equal"
        ),
    }


def check_h5_scale_process_decoupling(
    profiles: list[PABProfile],
    model_specs: list[Any],
) -> dict:
    """H5: Pearson correlation between param_count_b and stability_mean."""
    spec_map = {s.model_id: s for s in model_specs}

    sizes = []
    stabilities = []
    for p in profiles:
        spec = spec_map.get(p.experiment_id)
        if spec is None:
            continue
        sizes.append(spec.param_count_b)
        stabilities.append(p.summary.stability_mean)

    if len(sizes) < 3:
        return {
            "r_squared": 0.0,
            "correlation": 0.0,
            "interpretation": "insufficient data for scale-stability regression",
            "n_models": len(sizes),
        }

    corr_matrix = np.corrcoef(sizes, stabilities)
    r = float(corr_matrix[0, 1])
    r_sq = r**2

    if r_sq < 0.25:
        interp = f"Weak relationship (R^2={r_sq:.3f}): scale does NOT predict stability"
    elif r_sq < 0.50:
        interp = f"Moderate relationship (R^2={r_sq:.3f}): some scale-stability coupling"
    else:
        interp = f"Strong relationship (R^2={r_sq:.3f}): scale predicts stability"

    return {
        "r_squared": r_sq,
        "correlation": r,
        "interpretation": interp,
        "n_models": len(sizes),
    }


def check_h6_taxonomy_predicts_trainability(
    phase_a_clusters: dict | None,
    profiles: list[PABProfile],
) -> dict:
    """H6: Correlate Phase A cluster assignments with training outcomes."""
    if phase_a_clusters is None:
        return {
            "correlation": 0.0,
            "interpretation": "Phase A clusters not available; run Phase A first",
        }

    # Map cluster assignments to a numeric score
    profile_map = {p.experiment_id: p for p in profiles}
    cluster_ids = []
    stability_vals = []

    for model_id, cluster in phase_a_clusters.items():
        if model_id in profile_map:
            cluster_ids.append(cluster if isinstance(cluster, (int, float)) else hash(cluster) % 10)
            stability_vals.append(profile_map[model_id].summary.stability_mean)

    if len(cluster_ids) < 3:
        return {
            "correlation": 0.0,
            "interpretation": "insufficient overlap between Phase A clusters and Phase B profiles",
        }

    corr_matrix = np.corrcoef(cluster_ids, stability_vals)
    r = float(corr_matrix[0, 1])

    return {
        "correlation": r,
        "interpretation": (
            f"Taxonomy-trainability correlation r={r:.3f}: "
            + ("predictive" if abs(r) > 0.5 else "weak predictive power")
        ),
        "n_models": len(cluster_ids),
    }


def export_hypothesis_report(report: ArchitectureComparisonReport) -> str:
    """Render hypothesis testing results as Markdown."""
    lines = [
        "# Cross-Architecture Hypothesis Testing Report",
        "",
        f"**Models tested:** {len(report.profiles)}",
        f"**Architecture families:** {len(report.architecture_groups)}",
        "",
        "## Architecture Groups",
        "",
    ]

    for family, ids in sorted(report.architecture_groups.items()):
        lines.append(f"- **{family}**: {', '.join(ids)}")
    lines.append("")

    # Hypothesis results table
    lines.append("## Hypothesis Results")
    lines.append("")
    lines.append("| Hypothesis | Key Metric | Interpretation |")
    lines.append("|------------|-----------|----------------|")

    hypothesis_labels = {
        "H1": "Architecture signature (ANOVA)",
        "H2": "Attention vs recurrent (repr evolution)",
        "H3": "MoE routing (crystallization)",
        "H4": "Specialization transfer (convergence)",
        "H5": "Scale-process decoupling",
        "H6": "Taxonomy predicts trainability",
    }

    for h_id in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        result = report.hypothesis_results.get(h_id, {})
        label = hypothesis_labels.get(h_id, h_id)
        interp = result.get("interpretation", "not tested")
        key_metric = ""
        if "p_value" in result:
            key_metric = f"p={result['p_value']:.4f}"
        elif "r_squared" in result:
            key_metric = f"R^2={result['r_squared']:.3f}"
        elif "correlation" in result:
            key_metric = f"r={result['correlation']:.3f}"
        elif "difference" in result:
            key_metric = f"diff={result['difference']:.4f}"
        lines.append(f"| {h_id}: {label} | {key_metric} | {interp} |")

    lines.append("")

    # Standard comparison summary
    if report.comparison.profile_ids:
        lines.append("## Standard PAB Comparison")
        lines.append("")
        lines.append(report.comparison.summary_table())

    return "\n".join(lines)


def _gather_repr_evolution(
    profile_map: dict[str, PABProfile],
    arch_groups: dict[str, list[str]],
    families: set[str],
) -> list[float]:
    """Gather mean representation evolution values for given architecture families."""
    vals = []
    for family in families:
        for eid in arch_groups.get(family, []):
            if eid not in profile_map:
                continue
            re_series = profile_map[eid].core.representation_evolution
            if re_series:
                vals.append(float(np.mean(re_series)))
    return vals


def _scipy_unavailable(context: str) -> dict:
    """Return a placeholder result when scipy is not installed."""
    return {
        "statistic": 0.0,
        "p_value": 1.0,
        "interpretation": f"scipy not available for {context}",
        "effect_size": 0.0,
    }
