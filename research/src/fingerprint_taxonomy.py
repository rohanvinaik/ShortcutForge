"""Clustering and HBT-style taxonomy analysis of behavioral fingerprints.

Clusters models by behavioral similarity to test whether architectural
families produce distinct behavioral signatures (Phase A core analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.behavioral_fingerprint import BehavioralFingerprint
from src.model_registry import ModelSpec

_EPS = 1e-8

# Fixed vector layout: entropy(1) + discreteness(1) + eigenvalues(5) + distribution(20) = 27
_EIGEN_SLOTS = 5
_DIST_SLOTS = 20
_VECTOR_LEN = 1 + 1 + _EIGEN_SLOTS + _DIST_SLOTS


def fingerprint_to_vector(fp: BehavioralFingerprint) -> np.ndarray:
    """Convert a BehavioralFingerprint to a fixed-length numeric vector.

    Layout (27 elements):
        [action_entropy, discreteness_score, top-5 eigenvalues, top-20 distribution values]
    """
    vec = np.zeros(_VECTOR_LEN, dtype=np.float64)
    vec[0] = fp.action_entropy
    vec[1] = fp.discreteness_score

    # Eigenvalues: pad with 0 if fewer than 5
    eigs = fp.variance_eigenvalues[:_EIGEN_SLOTS]
    for i, e in enumerate(eigs):
        vec[2 + i] = e

    # Distribution values sorted descending, pad/truncate to 20
    dist_vals = sorted(fp.action_distribution.values(), reverse=True)[:_DIST_SLOTS]
    for i, v in enumerate(dist_vals):
        vec[2 + _EIGEN_SLOTS + i] = v

    return vec


def _js_divergence(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two action distributions."""
    all_actions = set(dist_a) | set(dist_b)
    if not all_actions:
        return 0.0

    p = np.array([dist_a.get(a, 0.0) for a in sorted(all_actions)])
    q = np.array([dist_b.get(a, 0.0) for a in sorted(all_actions)])

    p = p / (np.sum(p) + _EPS)
    q = q / (np.sum(q) + _EPS)

    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2((p + _EPS) / (m + _EPS))))
    kl_qm = float(np.sum(q * np.log2((q + _EPS) / (m + _EPS))))
    return 0.5 * (kl_pm + kl_qm)


def compute_distance_matrix(
    fingerprints: list[BehavioralFingerprint], model_ids: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise distance matrix combining JS divergence and Euclidean distance.

    The distance between two models is a weighted combination of:
    - JS divergence on their action_distributions (weight 0.5)
    - Euclidean distance on scalar feature vectors (weight 0.5)

    Returns:
        (n x n distance matrix, ordered model_ids)
    """
    n = len(fingerprints)
    vectors = [fingerprint_to_vector(fp) for fp in fingerprints]
    dist = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            js = _js_divergence(
                fingerprints[i].action_distribution,
                fingerprints[j].action_distribution,
            )
            eucl = float(np.linalg.norm(vectors[i] - vectors[j]))
            # Normalize Euclidean by vector length for comparable scale
            eucl_norm = eucl / (np.sqrt(_VECTOR_LEN) + _EPS)
            combined = 0.5 * js + 0.5 * eucl_norm
            dist[i, j] = combined
            dist[j, i] = combined

    return dist, list(model_ids)


def cluster_fingerprints(
    distance_matrix: np.ndarray,
    model_ids: list[str],
    method: str = "agglomerative",
    n_clusters: int | None = None,
) -> dict:
    """Cluster fingerprints using agglomerative hierarchical clustering.

    If n_clusters is None, auto-selects k via silhouette score (k=2..min(8, n-1)).

    Returns dict with keys: labels, n_clusters, silhouette_score, model_ids.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    n = distance_matrix.shape[0]
    if n < 2:
        return {
            "labels": [0] * n,
            "n_clusters": 1,
            "silhouette_score": 0.0,
            "model_ids": list(model_ids),
        }

    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method="average")

    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        labels = [int(x) - 1 for x in labels]
        sil = silhouette_score(distance_matrix, labels, metric="precomputed") if n > 2 else 0.0
        return {
            "labels": labels,
            "n_clusters": n_clusters,
            "silhouette_score": float(sil),
            "model_ids": list(model_ids),
        }

    # Auto-select k
    max_k = min(8, n - 1)
    best_k = 2
    best_sil = -1.0
    best_labels = [0] * n

    for k in range(2, max_k + 1):
        candidate_labels = fcluster(Z, t=k, criterion="maxclust")
        candidate_labels = [int(x) - 1 for x in candidate_labels]
        if len(set(candidate_labels)) < 2:
            continue
        sil = silhouette_score(distance_matrix, candidate_labels, metric="precomputed")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = candidate_labels

    return {
        "labels": best_labels,
        "n_clusters": best_k,
        "silhouette_score": float(best_sil),
        "model_ids": list(model_ids),
    }


@dataclass
class TaxonomyReport:
    """Result of cross-architecture behavioral taxonomy analysis."""

    clusters: dict[int, list[str]] = field(default_factory=dict)
    architecture_correlation: float = 0.0
    size_correlation: float = 0.0
    silhouette_score: float = 0.0
    distance_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    model_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clusters": self.clusters,
            "architecture_correlation": self.architecture_correlation,
            "size_correlation": self.size_correlation,
            "silhouette_score": self.silhouette_score,
            "model_ids": self.model_ids,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Behavioral Taxonomy Report",
            "",
            f"**Silhouette Score:** {self.silhouette_score:.3f}",
            f"**Architecture Correlation (ARI):** {self.architecture_correlation:.3f}",
            f"**Size Correlation:** {self.size_correlation:.3f}",
            "",
            "## Clusters",
            "",
        ]
        for cid in sorted(self.clusters):
            members = ", ".join(self.clusters[cid])
            lines.append(f"- **Cluster {cid}:** {members}")
        return "\n".join(lines)


def build_taxonomy(
    fingerprints: list[BehavioralFingerprint],
    model_specs: list[ModelSpec],
) -> TaxonomyReport:
    """Build behavioral taxonomy from fingerprints and model specs.

    Clusters models by behavioral similarity, then measures how well
    the clusters align with architecture families and model sizes.
    """
    model_ids = [spec.model_id for spec in model_specs]
    distance_matrix, ordered_ids = compute_distance_matrix(fingerprints, model_ids)
    result = cluster_fingerprints(distance_matrix, ordered_ids)
    labels = result["labels"]

    clusters: dict[int, list[str]] = {}
    for mid, label in zip(ordered_ids, labels):
        clusters.setdefault(label, []).append(mid)

    spec_map = {s.model_id: s for s in model_specs}
    arch_corr = _architecture_correlation(spec_map, ordered_ids, labels)
    size_corr = _size_correlation(spec_map, ordered_ids, labels)

    return TaxonomyReport(
        clusters=clusters,
        architecture_correlation=arch_corr,
        size_correlation=size_corr,
        silhouette_score=result["silhouette_score"],
        distance_matrix=distance_matrix,
        model_ids=ordered_ids,
    )


def _architecture_correlation(
    spec_map: dict[str, ModelSpec], ordered_ids: list[str], labels: list[int]
) -> float:
    """ARI between cluster labels and architecture families."""
    from sklearn.metrics import adjusted_rand_score

    arch_labels = [spec_map[mid].architecture_family for mid in ordered_ids]
    unique_archs = sorted(set(arch_labels))
    arch_to_int = {a: i for i, a in enumerate(unique_archs)}
    arch_ints = [arch_to_int[a] for a in arch_labels]
    return float(adjusted_rand_score(arch_ints, labels))


def _size_correlation(
    spec_map: dict[str, ModelSpec], ordered_ids: list[str], labels: list[int]
) -> float:
    """Pearson correlation between cluster label and param count."""
    sizes = np.array([spec_map[mid].param_count_b for mid in ordered_ids])
    label_arr = np.array(labels, dtype=np.float64)
    if np.std(sizes) > 0 and np.std(label_arr) > 0:
        return float(np.corrcoef(sizes, label_arr)[0, 1])
    return 0.0
