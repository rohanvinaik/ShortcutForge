"""Pure metric computation functions for PAB trajectory analysis.

Stateless functions used by ``PABTracker`` to compute stability,
predictability, representation evolution, crystallization, and
domain classification at each training checkpoint.

**Canonical PAB metrics** (stability, generalization efficiency, feature
importance consistency) are delegated to **PABKit** (Pal, 2026) when
available.  Architecture-specific extensions (crystallization, tier
convergence, domain classification) remain in this module.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# PABKit integration — canonical metric implementations (Pal, 2026)
# ---------------------------------------------------------------------------
try:
    from pab.metrics import (  # type: ignore[import-untyped]
        feature_importance_consistency as _pabkit_feature_importance,
    )
    from pab.metrics import (  # type: ignore[import-untyped]
        generalization_efficiency as _pabkit_gen_efficiency,
    )
    from pab.metrics import (  # type: ignore[import-untyped]
        learning_stability as _pabkit_stability,
    )

    _HAS_PABKIT = True
except ImportError:
    _HAS_PABKIT = False

_EPS = 1e-8
_PREDICTABILITY_WINDOW = 10


# ---------------------------------------------------------------------------
# Core PAB metrics (delegated to PABKit when available)
# ---------------------------------------------------------------------------


def compute_stability(prev_loss: float, curr_loss: float) -> float:
    """S(t) — learning trajectory stability (Pal, 2026, Eq. 2).

    Delegates to PABKit's ``learning_stability`` when available.
    """
    if _HAS_PABKIT:
        return _pabkit_stability(prev_loss, curr_loss)
    return abs(prev_loss - curr_loss) / (abs(prev_loss) + _EPS)


def compute_generalization_gap(train_loss: float, val_loss: float) -> float:
    """G(t) — instantaneous generalization efficiency (Pal, 2026, Eq. 3).

    Delegates to PABKit's ``generalization_efficiency`` when available.
    """
    if _HAS_PABKIT:
        return _pabkit_gen_efficiency(train_loss, val_loss)
    return val_loss - train_loss


def compute_feature_importance(weight_snapshots: list) -> float:
    """L — feature importance consistency across training (Pal, 2026, Eq. 7).

    For ternary architectures, decoder weight signs serve as the feature
    importance proxy: ternary commitment forces discrete sign stability,
    which directly bounds this metric.

    Delegates to PABKit's ``feature_importance_consistency``.
    Falls back to a local implementation if PABKit is not installed.
    """
    if len(weight_snapshots) < 2:
        return 0.0
    if _HAS_PABKIT:
        return _pabkit_feature_importance(weight_snapshots)
    # Local fallback: L = (1/n) * sum((w_t - w_bar)^2)
    arrays = [np.asarray(w).reshape(-1).astype(np.float64) for w in weight_snapshots]
    stacked = np.stack(arrays)
    w_bar = np.mean(stacked, axis=0)
    return float(np.mean((stacked - w_bar) ** 2))


def compute_predictability(train_losses: list[float]) -> float:
    """Var(delta-L) over a sliding window of loss deltas.

    Extension of PABKit's ``learning_curve_predictability`` (Pal, 2026,
    Eq. 5) with a sliding window for online tracking during training.
    """
    if len(train_losses) < 2:
        return 0.0
    start = max(1, len(train_losses) - _PREDICTABILITY_WINDOW)
    deltas = [train_losses[i] - train_losses[i - 1] for i in range(start, len(train_losses))]
    return float(np.var(deltas)) if len(deltas) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Balanced Sashimi-specific extensions (not in PABKit)
# ---------------------------------------------------------------------------


def compute_repr_evolution(
    embeddings: np.ndarray, prev_mean: np.ndarray | None
) -> tuple[float, np.ndarray]:
    """1 - cos_sim of normalized bottleneck means. Returns (r_t, new_mean)."""
    z_mean = np.mean(embeddings, axis=0)
    z_norm = z_mean / (np.linalg.norm(z_mean) + _EPS)
    if prev_mean is not None:
        prev_norm = prev_mean / (np.linalg.norm(prev_mean) + _EPS)
        r_t = 1.0 - float(np.dot(z_norm, prev_norm))
    else:
        r_t = 1.0
    return r_t, z_mean


def compute_crystallization(signs: np.ndarray, prev_signs: np.ndarray | None) -> float:
    """Fraction of decoder weights whose sign hasn't changed."""
    if prev_signs is None or len(signs) != len(prev_signs):
        return 0.0
    return float(np.sum(signs == prev_signs) / (len(signs) + _EPS))


def classify_domain(accs: list[float], n_total: int) -> str:
    """Classify a domain's learning trajectory as early/late/unstable."""
    if len(accs) < 3:
        return "unknown"
    third = max(1, n_total // 3)
    reached_early = any(a >= 0.80 for a in accs[:third])
    reached_ever = any(a >= 0.80 for a in accs)
    oscillating = is_oscillating(accs)

    if reached_early:
        return "early"
    if reached_ever:
        return "unstable" if oscillating else "late"
    return "unstable" if oscillating else "late"


def is_oscillating(values: list[float], min_reversals: int = 3) -> bool:
    """Check if a sequence oscillates (multiple direction reversals)."""
    if len(values) < 3:
        return False
    reversals = sum(
        1
        for i in range(2, len(values))
        if (values[i - 1] - values[i - 2]) * (values[i] - values[i - 1]) < 0
    )
    return reversals >= min_reversals


def monotonic_trend(values: list[float]) -> float:
    """Fraction of consecutive pairs where value increases (0 to 1)."""
    if len(values) < 2:
        return 0.0
    increases = sum(1 for i in range(1, len(values)) if values[i] > values[i - 1])
    return increases / (len(values) - 1)


def find_tier_convergence(
    accuracies: list[float],
    threshold: float,
    checkpoints: list[int] | None = None,
) -> int | None:
    """First training step where accuracy exceeds threshold.

    If ``checkpoints`` is omitted, returns the 0-based index for backward
    compatibility.
    """
    for i, acc in enumerate(accuracies):
        if acc >= threshold:
            if checkpoints and i < len(checkpoints):
                return int(checkpoints[i])
            return i
    return None


def linear_slope(values: list[float]) -> float:
    """Simple linear regression slope of a sequence."""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    x_mean, y_mean = np.mean(x), np.mean(y)
    denom = float(np.sum((x - x_mean) ** 2))
    if abs(denom) < _EPS:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean))) / denom
