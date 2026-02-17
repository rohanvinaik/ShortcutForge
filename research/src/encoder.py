"""
Prompt encoder — frozen sentence-transformer producing semantic embeddings.

Wraps all-MiniLM-L6-v2 (22M params, encoder-only) as a frozen feature
extractor. Output: [batch, 384] float32 embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PromptEncoder(nn.Module):
    """Frozen sentence-transformer encoder for prompt embeddings.

    Produces 384-dim embeddings from natural language prompts.
    Weights are frozen during training (no gradient flow).

    Args:
        model_name: HuggingFace model identifier.
        device: Target device ("mps", "cpu").
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._model = None  # Lazy load

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, prompts: list[str]) -> torch.Tensor:
        """Encode prompts to [batch, 384] embeddings.

        Returns:
            Float32 tensor on self.device, shape [len(prompts), 384].
        """
        if self._model is None:
            self._load_model()
        with torch.no_grad():
            embeddings = self._model.encode(prompts, convert_to_tensor=True)
        return embeddings.to(self.device)

    def forward(self, prompts: list[str]) -> torch.Tensor:
        """Alias for encode(). Frozen — no gradients."""
        return self.encode(prompts)
