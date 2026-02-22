"""Per-step safety checks and logging helpers for the trainer.

Extracted from ``trainer.py`` to keep the main module under the 400-line limit.
All functions are stateless — they operate on the pipeline/infra containers
passed in from :class:`BalancedSashimiTrainer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

try:
    from src.pab_tracker import CheckpointData
except ImportError:  # pragma: no cover - fallback for direct module execution
    from research.src.pab_tracker import CheckpointData

# ---------------------------------------------------------------------------
# Gradient health
# ---------------------------------------------------------------------------


def check_gradient_health(pipeline: Any) -> bool:
    """Check all gradients for NaN/Inf. Returns True if healthy.

    Args:
        pipeline: ``_Pipeline`` namedtuple with ``domain_gate``,
                  ``intent_extractor``, ``bridge``, ``decoder``.
    """
    import torch

    for name, module in [
        ("gate", pipeline.domain_gate),
        ("intent", pipeline.intent_extractor),
        ("bridge", pipeline.bridge),
        ("decoder", pipeline.decoder),
    ]:
        for pname, p in module.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print(f"  WARNING: NaN/Inf gradient in {name}.{pname}")
                return False
    return True


# ---------------------------------------------------------------------------
# Ternary distribution logging
# ---------------------------------------------------------------------------


def log_ternary_distribution(decoder: Any) -> dict[str, dict[str, float]]:
    """Log {-1, 0, +1} distribution per TernaryLinear layer.

    Args:
        decoder: The ``TernaryDecoder`` module.

    Returns:
        Dict mapping layer name to ``{neg_pct, zero_pct, pos_pct}``.
    """
    try:
        from src.ternary_decoder import TernaryLinear
    except ImportError:  # pragma: no cover - fallback for direct module execution
        from research.src.ternary_decoder import TernaryLinear

    dist: dict[str, dict[str, float]] = {}
    for name, module in decoder.named_modules():
        if isinstance(module, TernaryLinear):
            w = module.weight.data
            total = w.numel()
            neg = (w < -0.01).sum().item() / total * 100
            zero = ((w >= -0.01) & (w <= 0.01)).sum().item() / total * 100
            pos = (w > 0.01).sum().item() / total * 100
            dist[name] = {"neg_pct": neg, "zero_pct": zero, "pos_pct": pos}
    return dist


def collect_decoder_weight_signs(decoder: Any) -> np.ndarray | None:
    """Extract sign pattern from TernaryLinear layers for crystallization tracking.

    Args:
        decoder: The ``TernaryDecoder`` module.

    Returns:
        Flattened numpy array of signs, or None if no TernaryLinear layers.
    """
    try:
        from src.ternary_decoder import TernaryLinear
    except ImportError:  # pragma: no cover - fallback for direct module execution
        from research.src.ternary_decoder import TernaryLinear

    signs = []
    for module in decoder.modules():
        if isinstance(module, TernaryLinear):
            signs.append(np.sign(module.weight.data.detach().cpu().numpy()).flatten())
    if not signs:
        return None
    return np.concatenate(signs)


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------


def save_checkpoint(
    checkpoint_dir: Path,
    run_id: str,
    step: int,
    config: dict,
    pipeline: Any,
    infra: Any,
) -> Path:
    """Save model checkpoint with optimizer state.

    Args:
        checkpoint_dir: Directory to write the ``.pt`` file.
        run_id: Experiment identifier for the filename.
        step: Current training step.
        config: Full experiment config dict.
        pipeline: ``_Pipeline`` namedtuple.
        infra: ``_TrainInfra`` namedtuple.

    Returns:
        Path to the saved checkpoint file.
    """
    import torch

    path = checkpoint_dir / f"{run_id}_step{step}.pt"
    torch.save(
        {
            "step": step,
            "run_id": run_id,
            "config": config,
            "domain_gate": pipeline.domain_gate.state_dict(),
            "intent_extractor": pipeline.intent_extractor.state_dict(),
            "bridge": pipeline.bridge.state_dict(),
            "decoder": pipeline.decoder.state_dict(),
            "composite_loss": infra.composite_loss.state_dict(),
            "optimizer": infra.optimizer.state_dict(),
            "gate_optimizer": (
                infra.gate_optimizer.state_dict()
                if hasattr(infra, "gate_optimizer") and infra.gate_optimizer is not None
                else None
            ),
        },
        path,
    )
    return path


# ---------------------------------------------------------------------------
# Per-step check helpers (called from _run_step_checks)
# ---------------------------------------------------------------------------


def check_gradient_abort(
    step: int,
    safety: dict,
    pipeline: Any,
    save_fn: Any,
) -> dict | None:
    """Check gradients and return abort dict if NaN/Inf detected, else None.

    Args:
        step: Current training step.
        safety: Config ``safety`` section.
        pipeline: ``_Pipeline`` namedtuple.
        save_fn: Callable that saves a checkpoint (takes ``step``).
    """
    if step % safety["gradient_check_every_n_steps"] != 0:
        return None
    if check_gradient_health(pipeline):
        return None
    if not safety["nan_abort"]:
        return None
    if safety.get("nan_checkpoint_before_abort", True):
        save_fn(step)
    print(f"ABORT: NaN/Inf gradient at step {step}")
    return {"status": "nan_abort", "step": step}


def record_pab_checkpoint(
    step: int,
    loss_dict: dict,
    config: dict,
    infra: Any,
    last_bridge_features: np.ndarray | None,
    decoder: Any,
    val_loss: float | None = None,
    tier_accuracies: dict[str, float] | None = None,
    domain_accuracies: dict[str, float] | None = None,
    action_accuracies: dict[str, float] | None = None,
) -> dict | None:
    """Record a PAB checkpoint if due. Returns early-exit dict or None.

    Args:
        step: Current training step.
        loss_dict: Loss metrics from the current step.
        config: Full experiment config dict.
        infra: ``_TrainInfra`` namedtuple (must have ``pab_tracker``).
        last_bridge_features: Cached bridge features, or None.
        decoder: The ``TernaryDecoder`` module.
    """
    tracker = infra.pab_tracker
    if tracker is None:
        return None
    pab_cfg = config.get("pab", {})
    interval = pab_cfg.get("checkpoint_interval", 50)
    if step % interval != 0:
        return None
    data = CheckpointData(
        step=step,
        train_loss=loss_dict["L_total"],
        val_loss=val_loss,
        loss_components={
            "ce": loss_dict.get("L_ce", 0.0),
            "margin": loss_dict.get("L_margin", 0.0),
            "repair": loss_dict.get("L_repair", 0.0),
        },
        adaptive_weights={
            k: v for k, v in loss_dict.items() if k.startswith("w_") and isinstance(v, float)
        },
        tier_accuracies=tier_accuracies or {},
        bottleneck_embeddings=last_bridge_features,
        decoder_weight_signs=collect_decoder_weight_signs(decoder),
        domain_accuracies=domain_accuracies or {},
        action_accuracies=action_accuracies or {},
    )
    tracker.record(data)
    if pab_cfg.get("early_exit_enabled", False) and tracker.should_early_exit(step):
        print(f"  PAB early exit triggered at step {step}")
        return {"status": "pab_early_exit", "step": step}
    return None
