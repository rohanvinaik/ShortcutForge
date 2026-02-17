"""
Value filler — continuous module for Tier 3 free-text value generation.

Differentiable path: forward() returns tensor/logits.
Non-differentiable path: decode() returns list[Tier3Slot] for pipeline use.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.src.contracts import Tier3Slot


class ValueFiller(nn.Module):
    """Fills free-text value slots using encoder features and skeleton tokens.

    Two interfaces:
        forward(): Returns tensors/logits (differentiable training path).
        decode(): Returns list[Tier3Slot] (symbolic decode for pipeline).

    Args:
        encoder_dim: Encoder feature dimension (384).
        hidden_dim: Internal dimension.
    """

    def __init__(self, encoder_dim: int = 384, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        skeleton_token_ids: torch.Tensor,
        encoder_features: torch.Tensor,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Produce value predictions (differentiable path).

        Args:
            skeleton_token_ids: [batch, seq_len] Tier1+2 token indices.
            encoder_features: [batch, encoder_dim] from PromptEncoder.

        Returns:
            Tensor or dict of tensors (logits/predictions for loss computation).
        """
        raise NotImplementedError("Phase 3: ValueFiller forward")

    def decode(
        self,
        skeleton_tokens: list[str],
        encoder_features: torch.Tensor,
    ) -> list[Tier3Slot]:
        """Produce structured Tier3Slots (non-differentiable).

        Args:
            skeleton_tokens: Decoded Tier1+2 token strings.
            encoder_features: [batch, encoder_dim] from PromptEncoder.

        Returns:
            List of Tier3Slot with filled values.
        """
        raise NotImplementedError("Phase 3: ValueFiller decode")
