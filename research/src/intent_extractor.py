"""
Intent extractor — maps encoder embeddings to latent frame features.

Differentiable path: forward() returns a tensor for downstream modules.
Non-differentiable path: extract() returns SemanticFrame for logging/debug.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.src.contracts import SemanticFrame


class IntentExtractor(nn.Module):
    """Extracts structured intent from encoder embeddings.

    Two interfaces:
        forward(): Returns tensor (latent frame features). Differentiable.
        extract(): Returns SemanticFrame(s). Non-differentiable, for logging.

    Args:
        input_dim: Embedding dimension (384).
        frame_dim: Latent frame feature dimension.
    """

    def __init__(self, input_dim: int = 384, frame_dim: int = 256) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.frame_dim = frame_dim
        self.projection = nn.Linear(input_dim, frame_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Map encoder embeddings to latent frame features.

        Args:
            embeddings: [batch, input_dim] float32 tensor.

        Returns:
            [batch, frame_dim] latent frame features (differentiable).
        """
        return self.projection(embeddings)

    def extract(self, embeddings: torch.Tensor) -> list[SemanticFrame]:
        """Produce structured SemanticFrame(s) for logging/debug.

        Non-differentiable. Gradient flow stops here.

        Args:
            embeddings: [batch, input_dim] float32 tensor.

        Returns:
            List of SemanticFrame (one per batch element).
        """
        with torch.no_grad():
            features = self.forward(embeddings)
            frames = []
            for i in range(features.shape[0]):
                frames.append(
                    SemanticFrame(
                        domain="shortcuts",
                        primary_intent="unknown",
                        entities=[],
                        constraints=[],
                        estimated_complexity="simple",
                    )
                )
            return frames
