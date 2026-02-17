"""
Ternary structural decoder — {-1, 0, +1} weight layers for discrete decisions.

Uses Straight-Through Estimator (STE) for training: fp32 shadow weights with
quantized forward pass. Gradients flow through quantization via STE.

Training memory is dominated by fp32 shadow weights + Adam states,
NOT by the ternary weights themselves.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quantize(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Quantize continuous weights to {-1, 0, +1}.

    Uses threshold-based quantization with per-channel scaling.
    STE: gradient of this function is identity (straight-through).

    Args:
        weights: Continuous fp32 weights.
        eps: Small constant for numerical stability.

    Returns:
        Ternary tensor with values in {-1, 0, +1}.
    """
    # Per-row threshold: 0.7 * mean(|w|)
    threshold = 0.7 * weights.abs().mean(dim=-1, keepdim=True)

    # Quantize: > threshold → +1, < -threshold → -1, else → 0
    quantized = torch.zeros_like(weights)
    quantized[weights > threshold] = 1.0
    quantized[weights < -threshold] = -1.0

    # STE: forward uses quantized, backward passes gradient through as identity
    return weights + (quantized - weights).detach()


class TernaryLinear(nn.Module):
    """Linear layer with ternary weights via STE.

    Maintains fp32 shadow weights for gradient computation.
    Forward pass uses quantized {-1, 0, +1} weights.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether to include bias term.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with ternary-quantized weights (STE backward).

        Args:
            x: [batch, in_features] input tensor.

        Returns:
            [batch, out_features] output tensor.
        """
        q_weight = ternary_quantize(self.weight)
        return F.linear(x, q_weight, self.bias)


class TernaryDecoder(nn.Module):
    """Multi-layer ternary decoder for Tier 1 and Tier 2 token prediction.

    Args:
        input_dim: Bridge output dimension.
        hidden_dim: Internal layer dimension.
        tier1_vocab_size: Number of Tier 1 structural tokens.
        tier2_vocab_size: Number of Tier 2 parameter tokens.
        num_layers: Number of TernaryLinear layers.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        tier1_vocab_size: int = 0,
        tier2_vocab_size: int = 0,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tier1_vocab_size = tier1_vocab_size
        self.tier2_vocab_size = tier2_vocab_size
        self.num_layers = num_layers

        # Build the shared hidden layer stack
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(TernaryLinear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.layers = nn.Sequential(*layers)

        # Prediction heads (only created if vocab_size > 0)
        self.tier1_head: TernaryLinear | None = None
        self.tier2_head: TernaryLinear | None = None

        if tier1_vocab_size > 0:
            self.tier1_head = TernaryLinear(hidden_dim, tier1_vocab_size)
        if tier2_vocab_size > 0:
            self.tier2_head = TernaryLinear(hidden_dim, tier2_vocab_size)

    def forward(self, bridge_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode bridge representation to tier token logits.

        Args:
            bridge_output: [batch, input_dim] from InformationBridge.

        Returns:
            Dict with keys:
                "tier1_logits": [batch, tier1_vocab_size]
                "tier2_logits": [batch, tier2_vocab_size]
        """
        hidden = self.layers(bridge_output)

        result: dict[str, torch.Tensor] = {}

        if self.tier1_head is not None:
            result["tier1_logits"] = self.tier1_head(hidden)

        if self.tier2_head is not None:
            result["tier2_logits"] = self.tier2_head(hidden)

        return result
