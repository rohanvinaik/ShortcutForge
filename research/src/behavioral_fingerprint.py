"""Behavioral fingerprinting for deployment-side model evaluation.

Captures decoder behavioral signatures to connect training quality
(measured by PAB) to deployment reliability. Tests hypotheses H13-H14:
- H13: PAB trajectory stability predicts behavioral fingerprint stability
- H14: Ternary architectures produce more discrete behavioral signatures

Fingerprints are computed on a fixed set of diagnostic probes and stored
alongside PAB profiles for cross-analysis.

Usage:
    fp = BehavioralFingerprint.from_decoder(decoder, probe_set)
    fp.save("research/models/EXP-2.1/fingerprint_step500.json")

    # Compare fingerprints across training checkpoints
    stability = fingerprint_stability([fp_100, fp_200, fp_300])
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class BehavioralFingerprint:
    """Behavioral signature of a model at a training checkpoint.

    Captures how the decoder behaves on a fixed set of diagnostic probes,
    independent of absolute accuracy. Two models with identical accuracy
    can have very different behavioral fingerprints.
    """

    experiment_id: str = ""
    step: int = 0
    action_entropy: float = 0.0
    action_distribution: dict[str, float] = field(default_factory=dict)
    variance_eigenvalues: list[float] = field(default_factory=list)
    discreteness_score: float = 0.0
    probe_responses: dict[str, str] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Save fingerprint to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> BehavioralFingerprint:
        """Load fingerprint from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    def _to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "step": self.step,
            "action_entropy": self.action_entropy,
            "action_distribution": self.action_distribution,
            "variance_eigenvalues": self.variance_eigenvalues,
            "discreteness_score": self.discreteness_score,
            "probe_responses": self.probe_responses,
        }

    @classmethod
    def _from_dict(cls, data: dict) -> BehavioralFingerprint:
        return cls(
            experiment_id=data.get("experiment_id", ""),
            step=data.get("step", 0),
            action_entropy=data.get("action_entropy", 0.0),
            action_distribution=data.get("action_distribution", {}),
            variance_eigenvalues=data.get("variance_eigenvalues", []),
            discreteness_score=data.get("discreteness_score", 0.0),
            probe_responses=data.get("probe_responses", {}),
        )

    @classmethod
    def from_outputs(
        cls,
        experiment_id: str,
        step: int,
        output_logits: np.ndarray,
        action_predictions: list[str],
        probe_labels: list[str] | None = None,
    ) -> BehavioralFingerprint:
        """Build a fingerprint from decoder outputs on diagnostic probes.

        Args:
            experiment_id: Training run identifier.
            step: Training step at which this fingerprint was captured.
            output_logits: (n_probes, vocab_size) logit matrix from decoder.
            action_predictions: Predicted action for each probe.
            probe_labels: Optional names for each probe.
        """
        entropy = compute_action_entropy(action_predictions)
        distribution = compute_action_distribution(action_predictions)
        eigenvalues = compute_variance_eigenvalues(output_logits)
        discreteness = compute_discreteness(output_logits)

        responses = {}
        if probe_labels:
            for label, pred in zip(probe_labels, action_predictions):
                responses[label] = pred

        return cls(
            experiment_id=experiment_id,
            step=step,
            action_entropy=entropy,
            action_distribution=distribution,
            variance_eigenvalues=eigenvalues,
            discreteness_score=discreteness,
            probe_responses=responses,
        )

    @classmethod
    def from_text_only(
        cls,
        experiment_id: str,
        step: int,
        action_predictions: list[str],
        probe_labels: list[str] | None = None,
    ) -> BehavioralFingerprint:
        """Build partial fingerprint from text outputs only (no logits).

        Computes entropy + distribution from action predictions.
        Eigenvalues and discreteness are set to 0.0 since no logits are available.
        Use this fallback for models where logit extraction fails.
        """
        entropy = compute_action_entropy(action_predictions)
        distribution = compute_action_distribution(action_predictions)

        responses = {}
        if probe_labels:
            for label, pred in zip(probe_labels, action_predictions):
                responses[label] = pred

        return cls(
            experiment_id=experiment_id,
            step=step,
            action_entropy=entropy,
            action_distribution=distribution,
            variance_eigenvalues=[],
            discreteness_score=0.0,
            probe_responses=responses,
        )


_EPS = 1e-8


def compute_action_entropy(predictions: list[str]) -> float:
    """Shannon entropy of the action distribution over probes.

    Lower entropy = more concentrated action selection.
    Ternary decoders should produce lower entropy (H14 prediction).
    """
    if not predictions:
        return 0.0
    counts: dict[str, int] = {}
    for p in predictions:
        counts[p] = counts.get(p, 0) + 1

    total = len(predictions)
    probs = np.array([c / total for c in counts.values()])
    return -float(np.sum(probs * np.log2(probs + _EPS)))


def compute_action_distribution(predictions: list[str]) -> dict[str, float]:
    """Normalized frequency of each predicted action."""
    if not predictions:
        return {}
    counts: dict[str, int] = {}
    for p in predictions:
        counts[p] = counts.get(p, 0) + 1
    total = len(predictions)
    return {action: count / total for action, count in sorted(counts.items())}


def compute_variance_eigenvalues(logits: np.ndarray, top_k: int = 10) -> list[float]:
    """Top eigenvalues of the output variance matrix.

    High concentration in few eigenvalues = more structured behavior.
    The eigenvalue spectrum characterizes how the decoder varies across probes.
    """
    if logits.ndim != 2 or logits.shape[0] < 2:
        return []

    # Center the logits
    centered = logits - np.mean(logits, axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=True)

    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(cov)))[::-1]
    return [float(e) for e in eigenvalues[:top_k]]


def compute_discreteness(logits: np.ndarray) -> float:
    """Measure how discrete/peaked the output distribution is.

    Higher = more peaked distributions (decisive predictions).
    Ternary decoders should produce more discrete outputs (H14).

    Computed as: mean(max_prob - second_max_prob) across probes.
    """
    if logits.ndim != 2 or logits.shape[0] < 1:
        return 0.0

    # Softmax per probe
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + _EPS)

    gaps = []
    for row in probs:
        sorted_probs = np.sort(row)[::-1]
        if len(sorted_probs) >= 2:
            gaps.append(sorted_probs[0] - sorted_probs[1])
        else:
            gaps.append(sorted_probs[0] if len(sorted_probs) > 0 else 0.0)

    return float(np.mean(gaps))


def fingerprint_stability(fingerprints: list[BehavioralFingerprint]) -> dict[str, float]:
    """Measure stability of behavioral fingerprints across training checkpoints.

    Tests H13: PAB trajectory stability should predict fingerprint stability.

    Returns metrics measuring how much the fingerprint changes across checkpoints.
    """
    if len(fingerprints) < 2:
        return {"entropy_variance": 0.0, "distribution_drift": 0.0, "discreteness_variance": 0.0}

    entropies = [fp.action_entropy for fp in fingerprints]
    discreteness_vals = [fp.discreteness_score for fp in fingerprints]

    # Distribution drift: average JS divergence between consecutive checkpoints
    drifts = []
    for i in range(1, len(fingerprints)):
        drift = _distribution_drift(
            fingerprints[i - 1].action_distribution,
            fingerprints[i].action_distribution,
        )
        drifts.append(drift)

    return {
        "entropy_variance": float(np.var(entropies)),
        "distribution_drift": float(np.mean(drifts)) if drifts else 0.0,
        "discreteness_variance": float(np.var(discreteness_vals)),
    }


def _distribution_drift(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Jensen-Shannon divergence between two action distributions."""
    all_actions = set(dist_a) | set(dist_b)
    if not all_actions:
        return 0.0

    p = np.array([dist_a.get(a, 0.0) for a in sorted(all_actions)])
    q = np.array([dist_b.get(a, 0.0) for a in sorted(all_actions)])

    # Normalize
    p = p / (np.sum(p) + _EPS)
    q = q / (np.sum(q) + _EPS)

    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2((p + _EPS) / (m + _EPS))))
    kl_qm = float(np.sum(q * np.log2((q + _EPS) / (m + _EPS))))
    return 0.5 * (kl_pm + kl_qm)
