"""LoRA utility helpers: target module detection and data loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def detect_target_modules(model: Any) -> list[str]:
    """Detect appropriate LoRA target modules for the given model architecture."""
    import torch.nn as nn

    # Priority patterns by architecture family
    transformer_targets = {"q_proj", "v_proj"}
    ssm_targets = {"in_proj", "out_proj"}

    leaf_linear_names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short = name.rsplit(".", 1)[-1] if "." in name else name
            leaf_linear_names.append(short)

    leaf_set = set(leaf_linear_names)

    # Check transformer patterns first
    if transformer_targets & leaf_set:
        return sorted(transformer_targets & leaf_set)

    # Check SSM/Mamba patterns
    if ssm_targets & leaf_set:
        return sorted(ssm_targets & leaf_set)

    # Fallback: any linear layers with 'proj' or 'linear' in the name
    fallback = {n for n in leaf_set if "proj" in n or "linear" in n.lower()}
    if fallback:
        return sorted(fallback)

    # Last resort: all unique leaf Linear names
    if leaf_linear_names:
        return sorted(set(leaf_linear_names))[:4]

    return ["q_proj", "v_proj"]


def load_jsonl_data(path: Path) -> list[dict]:
    """Load JSONL chat-format training data."""
    data: list[dict] = []
    if not path.exists():
        logger.warning("Data file not found: %s", path)
        return data
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
