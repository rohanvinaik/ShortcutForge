"""
Composite loss with adaptive UW-SO weighting.

L_total = L_ce + L_margin + L_repair, with uncertainty-weighted
soft-optimal (UW-SO) adaptive balancing. Each component has a
learnable log-variance parameter (sigma) that the optimizer discovers.

Separate L_ood for domain gate training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    """Adaptive multi-task loss for Balanced Sashimi training.

    Components:
        L_ce: Cross-entropy over tier token predictions.
        L_margin: Contrastive margin loss from negative bank.
        L_repair: Linter-repair-weighted penalty.

    Each has a learnable log-sigma for UW-SO adaptive weighting.

    Args:
        initial_log_sigma: Initial value for log-variance parameters.
    """

    def __init__(self, initial_log_sigma: float = 0.0) -> None:
        super().__init__()
        self.log_sigma_ce = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_margin = nn.Parameter(torch.tensor(initial_log_sigma))
        self.log_sigma_repair = nn.Parameter(torch.tensor(initial_log_sigma))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        positive_features: torch.Tensor | None = None,
        negative_features: torch.Tensor | None = None,
        repair_weights: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute composite loss with adaptive weighting.

        Args:
            logits: Model predictions.
            targets: Ground truth labels.
            positive_features: Features for positive examples (margin loss).
            negative_features: Features for negative examples (margin loss).
            repair_weights: Per-example weights from linter repair profile.

        Returns:
            Dict with keys: L_total, L_ce, L_margin, L_repair,
            sigma_ce, sigma_margin, sigma_repair.
        """
        # L_ce: standard cross-entropy
        L_ce = F.cross_entropy(logits, targets)

        # L_margin: contrastive margin loss (if positive/negative features provided)
        if positive_features is not None and negative_features is not None:
            margin = 0.5
            distance = 1.0 - F.cosine_similarity(
                positive_features, negative_features, dim=-1
            )
            L_margin = F.relu(margin - distance).mean()
        else:
            L_margin = torch.tensor(0.0, device=logits.device)

        # L_repair: weighted cross-entropy (if repair_weights provided)
        if repair_weights is not None:
            per_sample = F.cross_entropy(logits, targets, reduction="none")
            L_repair = (per_sample * repair_weights).mean()
        else:
            L_repair = torch.tensor(0.0, device=logits.device)

        # UW-SO: precision_i = exp(-log_sigma_i)
        # L_total = sum(precision_i * L_i + log_sigma_i)
        precision_ce = torch.exp(-self.log_sigma_ce)
        precision_margin = torch.exp(-self.log_sigma_margin)
        precision_repair = torch.exp(-self.log_sigma_repair)

        L_total = (
            precision_ce * L_ce + self.log_sigma_ce
            + precision_margin * L_margin + self.log_sigma_margin
            + precision_repair * L_repair + self.log_sigma_repair
        )

        return {
            "L_total": L_total,
            "L_ce": L_ce,
            "L_margin": L_margin,
            "L_repair": L_repair,
            "sigma_ce": torch.exp(self.log_sigma_ce),
            "sigma_margin": torch.exp(self.log_sigma_margin),
            "sigma_repair": torch.exp(self.log_sigma_repair),
        }


class OODLoss(nn.Module):
    """Binary cross-entropy loss for domain gate training.

    Separate from CompositeLoss because it's trained on different data
    (OOD prompt set) and may use different schedule.
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BCE loss for domain gate.

        Args:
            logits: [batch, 1] pre-sigmoid logits from DomainGate.
            labels: [batch, 1] binary labels (1=in_domain, 0=ood).

        Returns:
            Scalar BCE loss.
        """
        return F.binary_cross_entropy_with_logits(logits, labels)
