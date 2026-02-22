"""Model registry for cross-architecture study.

Static catalog of 16 models across 13 architecture families. Supports
filtering by architecture family, availability checks, and YAML override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ARCHITECTURE_FAMILIES = [
    "dense",
    "dense_gqa",
    "qwen3_gqa",
    "mistral_swa",
    "mpt_alibi",
    "qwen3_moe",
    "linear_rnn",
    "ssm",
    "griffin",
    "modern_lstm",
    "native_ternary",
    "long_convolution",
    "hybrid_ssm_moe",
]


@dataclass
class ModelSpec:
    """Specification for a single model in the registry.

    Rarely-used fields (chat_template, memory_gb_estimate) live in ``extra``.
    """

    model_id: str
    display_name: str
    hf_repo_or_path: str
    architecture_family: str
    attention_type: str
    param_count_b: float
    local_path: str | None = None
    needs_download: bool = False
    trust_remote_code: bool = False
    enabled: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def memory_gb_estimate(self) -> float:
        """Estimated memory in GB (stored in extra)."""
        return self.extra.get("memory_gb_estimate", 0.0)


def _m(gb: float) -> dict:
    """Shorthand for extra dict with memory estimate."""
    return {"memory_gb_estimate": gb}


def _build_registry() -> dict[str, ModelSpec]:
    """Build the static model registry."""
    models = [
        # --- Models already on disk ---
        ModelSpec(
            "rwkv7",
            "RWKV7-World",
            "RWKV/v7-Goose-1.6B-Pile",
            "linear_rnn",
            "linear_attention",
            1.6,
            trust_remote_code=True,
            extra=_m(4),
        ),
        ModelSpec(
            "cerebrum-7b",
            "Cerebrum-1.0-7b",
            "AetherResearch/Cerebrum-1.0-7b",
            "mistral_swa",
            "sliding_window",
            7.0,
            extra=_m(16),
        ),
        ModelSpec(
            "mpt-7b",
            "MPT-7B-StoryWriter",
            "mosaicml/mpt-7b-storywriter",
            "mpt_alibi",
            "alibi",
            7.0,
            trust_remote_code=True,
            extra=_m(16),
        ),
        ModelSpec(
            "qwen3-8b",
            "Qwen3-8B-Drama",
            "Orion-zhen/Qwen3-8B-Drama",
            "qwen3_gqa",
            "gqa",
            8.0,
            extra=_m(18),
        ),
        ModelSpec(
            "nomos-1", "Nomos-1", "Nomos-AI/Nomos-1", "qwen3_moe", "gqa_moe", 14.0, extra=_m(30)
        ),
        ModelSpec("phi-2", "Phi-2", "microsoft/phi-2", "dense", "dense", 2.7, extra=_m(6)),
        ModelSpec(
            "qwen2.5-0.5b",
            "Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "dense_gqa",
            "gqa",
            0.5,
            extra=_m(2),
        ),
        ModelSpec(
            "mistral-7b",
            "Mistral-7B-Instruct-v0.3",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistral_swa",
            "sliding_window",
            7.0,
            extra=_m(16),
        ),
        ModelSpec(
            "llama-1b",
            "Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "dense_gqa",
            "gqa",
            1.0,
            extra=_m(3),
        ),
        ModelSpec(
            "qwen3-8b-base", "Qwen3-8B", "Qwen/Qwen3-8B", "qwen3_gqa", "gqa", 8.0, extra=_m(18)
        ),
        # --- Models to download ---
        ModelSpec(
            "mamba-2.8b",
            "Mamba-2.8B",
            "state-spaces/mamba-2.8b-hf",
            "ssm",
            "state_space",
            2.8,
            needs_download=True,
            trust_remote_code=True,
            extra=_m(7),
        ),
        ModelSpec(
            "recurrentgemma-2b",
            "RecurrentGemma-2B",
            "google/recurrentgemma-2b",
            "griffin",
            "linear_recurrence",
            2.0,
            needs_download=True,
            extra=_m(5),
        ),
        ModelSpec(
            "xlstm-7b",
            "xLSTM-7B",
            "NX-AI/xLSTM-7b",
            "modern_lstm",
            "exponential_gating",
            7.0,
            needs_download=True,
            trust_remote_code=True,
            extra=_m(16),
        ),
        ModelSpec(
            "bitnet-2b",
            "BitNet-b1.58-2B",
            "microsoft/bitnet-b1.58-2B-4T",
            "native_ternary",
            "ternary",
            2.0,
            needs_download=True,
            trust_remote_code=True,
            extra=_m(3),
        ),
        ModelSpec(
            "stripedhyena-7b",
            "StripedHyena-Hessian-7B",
            "togethercomputer/StripedHyena-Hessian-7B",
            "long_convolution",
            "hyena",
            7.0,
            needs_download=True,
            trust_remote_code=True,
            extra=_m(16),
        ),
        ModelSpec(
            "jamba-tiny",
            "Jamba-tiny-dev",
            "ai21labs/Jamba-tiny-dev",
            "hybrid_ssm_moe",
            "hybrid",
            0.319,
            needs_download=True,
            trust_remote_code=True,
            extra=_m(2),
        ),
    ]
    return {m.model_id: m for m in models}


REGISTRY: dict[str, ModelSpec] = _build_registry()


def get_model_spec(model_id: str) -> ModelSpec:
    """Get a model spec by ID. Raises KeyError if not found."""
    if model_id not in REGISTRY:
        raise KeyError(f"Unknown model_id: {model_id!r}. Available: {sorted(REGISTRY)}")
    return REGISTRY[model_id]


def get_available() -> list[ModelSpec]:
    """Return models that don't need download or have a local_path set."""
    return [
        spec
        for spec in REGISTRY.values()
        if spec.enabled and (not spec.needs_download or spec.local_path is not None)
    ]


def list_model_ids(enabled_only: bool = True) -> list[str]:
    """Return sorted model IDs from the registry.

    Args:
        enabled_only: If True, include only specs with enabled=True.
    """
    if enabled_only:
        return sorted(mid for mid, spec in REGISTRY.items() if spec.enabled)
    return sorted(REGISTRY.keys())


def get_by_family(family: str) -> list[ModelSpec]:
    """Filter registry by architecture_family."""
    return [spec for spec in REGISTRY.values() if spec.architecture_family == family]


def load_yaml_overrides(yaml_path: Path) -> None:
    """Load YAML overrides and update REGISTRY entries.

    Supports setting local_path, needs_download, enabled, and other fields
    on a per-model basis.
    """
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f) or {}

    models_section = data.get("models", {})
    for model_id, overrides in models_section.items():
        if model_id not in REGISTRY:
            continue
        spec = REGISTRY[model_id]
        if not isinstance(overrides, dict):
            continue
        for key, value in overrides.items():
            if hasattr(spec, key):
                setattr(spec, key, value)
