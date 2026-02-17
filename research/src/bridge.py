"""
Information bridge — bottleneck between continuous encoder and ternary decoder.

Receives tensor frame features from IntentExtractor.forward() (never
SemanticFrame). Compresses to a fixed-dim representation suitable for
the ternary decoder's discrete decision space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InformationBridge(nn.Module):
    """Bottleneck bridge from continuous to discrete space.

    Args:
        input_dim: Frame feature dimension from IntentExtractor.
        bridge_dim: Compressed representation dimension.
    """

    def __init__(self, input_dim: int = 256, bridge_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bridge_dim = bridge_dim
        self.projection = nn.Linear(input_dim, bridge_dim)
        self.norm = nn.LayerNorm(bridge_dim)

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        """Compress frame features through bottleneck.

        Args:
            frame_features: [batch, input_dim] from IntentExtractor.forward().

        Returns:
            [batch, bridge_dim] compressed representation.
        """
        return self.norm(self.projection(frame_features))
