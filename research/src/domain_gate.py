"""
Domain gate — binary classifier for in-domain vs. out-of-domain prompts.

Takes encoder embeddings and produces a scalar confidence that the prompt
is within the Shortcuts domain. Trained separately with BCE loss + OOD data.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from research.src.contracts import GateDecision


class DomainGate(nn.Module):
    """Binary in-domain classifier on top of encoder embeddings.

    Args:
        input_dim: Embedding dimension (384 for all-MiniLM-L6-v2).
        hidden_dim: Hidden layer size.
    """

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Produce in-domain logits from encoder embeddings.

        Args:
            embeddings: [batch, input_dim] float32 tensor.

        Returns:
            [batch, 1] logits (pre-sigmoid).
        """
        return self.net(embeddings)

    def predict(
        self,
        embeddings: torch.Tensor,
        threshold: float = 0.5,
    ) -> list[GateDecision]:
        """Produce structured gate decisions (non-differentiable).

        Args:
            embeddings: [batch, input_dim] float32 tensor.
            threshold: Decision boundary for in_domain classification.

        Returns:
            List of GateDecision with in_domain bool and confidence float.
        """
        with torch.no_grad():
            logits = self.forward(embeddings)
            probs = torch.sigmoid(logits)
            decisions = []
            for p in probs.squeeze(-1).tolist():
                decisions.append(GateDecision(in_domain=p >= threshold, confidence=p))
            return decisions
