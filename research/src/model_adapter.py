"""Unified model loading and inference abstraction.

Provides a consistent interface for loading, generating, and extracting
logits from diverse HuggingFace models across architecture families.
"""

from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from src.model_registry import ModelSpec

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Base class for model loading and inference."""

    def __init__(self, model_spec: ModelSpec) -> None:
        self.model_spec = model_spec
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load model and tokenizer into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Release model and tokenizer from memory."""

    @abstractmethod
    def generate(self, prompts: list[str], max_tokens: int = 256) -> list[str]:
        """Generate text completions for a batch of prompts."""

    @abstractmethod
    def generate_with_logits(
        self, prompts: list[str], max_tokens: int = 256
    ) -> tuple[list[str], np.ndarray]:
        """Generate text and return first-token logits.

        Returns:
            Tuple of (generated_texts, logits_array) where logits_array has
            shape (n_prompts, vocab_size) from the first generated token.
        """

    def get_metadata(self) -> ModelSpec:
        """Return the model specification."""
        return self.model_spec

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class HFModelAdapter(ModelAdapter):
    """Standard HuggingFace model adapter using AutoModelForCausalLM."""

    def __init__(self, model_spec: ModelSpec) -> None:
        super().__init__(model_spec)
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = self.model_spec.local_path or self.model_spec.hf_repo_or_path
        log.info("Loading model %s from %s", self.model_spec.model_id, repo)

        self._tokenizer = AutoTokenizer.from_pretrained(repo)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self._model.eval()
        self._loaded = True
        log.info("Model %s loaded", self.model_spec.model_id)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False

        try:
            import torch

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        log.info("Model %s unloaded", self.model_spec.model_id)

    def generate(self, prompts: list[str], max_tokens: int = 256) -> list[str]:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"Model {self.model_spec.model_id} not loaded. Call load() first.")

        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        # Decode only the generated portion (exclude input tokens)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        return self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def generate_with_logits(
        self, prompts: list[str], max_tokens: int = 256
    ) -> tuple[list[str], np.ndarray]:
        import torch

        if self._model is None or self._tokenizer is None:
            raise RuntimeError(f"Model {self.model_spec.model_id} not loaded. Call load() first.")

        inputs = self._tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode generated text
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs.sequences[:, input_len:]
        texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Extract first-token logits: scores[0] has shape (batch, vocab_size)
        if outputs.scores:
            first_token_logits = outputs.scores[0].cpu().float().numpy()
        else:
            vocab_size = self._model.config.vocab_size
            first_token_logits = np.zeros((len(prompts), vocab_size))

        return texts, first_token_logits


class ExoticHFAdapter(HFModelAdapter):
    """Adapter for models requiring trust_remote_code=True."""

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = self.model_spec.local_path or self.model_spec.hf_repo_or_path
        log.info("Loading exotic model %s from %s", self.model_spec.model_id, repo)

        self._tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            repo,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True
        log.info("Exotic model %s loaded", self.model_spec.model_id)


def load_adapter(model_spec: ModelSpec) -> ModelAdapter:
    """Factory function returning the appropriate adapter for a model spec."""
    if model_spec.trust_remote_code:
        return ExoticHFAdapter(model_spec)
    return HFModelAdapter(model_spec)
