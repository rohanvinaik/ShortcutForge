"""External encoder adapter for Balanced Sashimi pipeline.

Wraps any HuggingFace model to produce 384-dim embeddings compatible
with the Balanced Sashimi pipeline. Used in Phase C to swap the default
sentence-transformer encoder with different model architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from src.model_adapter import ModelAdapter, ModelSpec


@dataclass
class ExternalEncoder:
    """Wraps an arbitrary HF model as a fixed-dim encoder for the pipeline.

    Produces embeddings of shape (batch, target_dim) from raw text prompts,
    matching the PromptEncoder.encode() interface expected by the trainer.
    """

    model_spec: ModelSpec
    target_dim: int = 384
    _adapter: ModelAdapter | None = field(default=None, repr=False)
    _projection: torch.nn.Linear | None = field(default=None, repr=False)
    _hidden_dim: int | None = field(default=None, repr=False)
    _device: str = field(default="cpu", repr=False)

    def setup(self, device: str = "mps") -> None:
        """Load the model and create the projection layer."""
        import torch.nn as nn

        from src.model_adapter import load_adapter

        self._device = device
        self._adapter = load_adapter(self.model_spec)
        self._adapter.load()

        # Probe hidden dimension with a dummy forward pass
        hidden = self._extract_hidden_states(["hello"])
        self._hidden_dim = hidden.shape[-1]

        # Build projection: hidden_dim -> target_dim
        self._projection = nn.Linear(self._hidden_dim, self.target_dim, bias=False)
        nn.init.xavier_uniform_(self._projection.weight)
        self._projection = self._projection.to(device)

    def encode(self, prompts: list[str]) -> torch.Tensor:
        """Encode prompts to (batch, target_dim) embeddings.

        Matches the PromptEncoder.encode() interface used by the trainer.
        """
        import torch

        hidden = self._extract_hidden_states(prompts)
        hidden = hidden.to(self._device)
        with torch.no_grad():
            projected = self._projection(hidden)
        return projected

    def _extract_hidden_states(self, prompts: list[str]) -> torch.Tensor:
        """Run model forward pass and mean-pool the last hidden state.

        Returns tensor of shape (batch, hidden_dim).
        """
        import torch

        model = self._adapter._model
        tokenizer = self._adapter._tokenizer

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Last hidden state: (batch, seq_len, hidden_dim)
        last_hidden = outputs.hidden_states[-1]

        # Mean-pool across sequence length, respecting attention mask
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.detach().cpu()

    def unload(self) -> None:
        """Release model memory."""
        if self._adapter is not None:
            self._adapter.unload()
            self._adapter = None
        self._projection = None
        self._hidden_dim = None

    def forward(self, prompts: list[str]) -> torch.Tensor:
        """Alias for encode(), providing nn.Module-style interface."""
        return self.encode(prompts)


def build_encoder(
    model_spec: ModelSpec,
    target_dim: int = 384,
    device: str = "mps",
) -> ExternalEncoder:
    """Factory: create an ExternalEncoder, set it up, and return it."""
    encoder = ExternalEncoder(model_spec=model_spec, target_dim=target_dim)
    encoder.setup(device=device)
    return encoder
