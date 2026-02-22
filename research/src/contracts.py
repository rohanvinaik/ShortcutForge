"""
Balanced Sashimi data contracts — single typed source of truth.

All persisted artifacts (JSONL, JSON) serialize/deserialize through these
dataclasses. Field names are stable across the pipeline.

Conventions:
    - Persisted forms use human-readable strings (e.g., Tier2Block.tokens).
    - Model internals consume integer indices (e.g., Tier2Block.token_ids).
    - to_dict()/from_dict() handle stable JSON serialization for each type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Semantic analysis types
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A named entity extracted from the user prompt."""

    name: str
    entity_type: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "entity_type": self.entity_type, "value": self.value}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> Entity:
        return cls(name=d["name"], entity_type=d["entity_type"], value=d["value"])


@dataclass
class Constraint:
    """A structural or behavioral constraint on the shortcut."""

    kind: str
    payload: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": self.payload}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Constraint:
        return cls(kind=d["kind"], payload=d["payload"])


@dataclass
class SemanticFrame:
    """Structured intent extraction output (non-differentiable view).

    Produced by IntentExtractor.extract(), never by forward().
    Used for logging, debugging, and human inspection.
    """

    domain: str
    primary_intent: str
    entities: list[Entity]
    constraints: list[Constraint]
    estimated_complexity: str  # "simple", "moderate", "complex"

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "primary_intent": self.primary_intent,
            "entities": [e.to_dict() for e in self.entities],
            "constraints": [c.to_dict() for c in self.constraints],
            "estimated_complexity": self.estimated_complexity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticFrame:
        return cls(
            domain=d["domain"],
            primary_intent=d["primary_intent"],
            entities=[Entity.from_dict(e) for e in d.get("entities", [])],
            constraints=[Constraint.from_dict(c) for c in d.get("constraints", [])],
            estimated_complexity=d.get("estimated_complexity", "simple"),
        )


# ---------------------------------------------------------------------------
# Three-tier output types
# ---------------------------------------------------------------------------


@dataclass
class Tier2Block:
    """A single action's typed parameter block (Tier 2 decoding output).

    Persisted form uses human-readable token strings for artifact readability
    and deterministic lowering. Model internals consume integer indices via
    token_ids during training/inference.

    Attributes:
        action_index: Position of this action in the shortcut.
        action_name: Canonical action identifier (e.g., "is.workflow.actions.showalert").
        tokens: Decoded canonical token strings (persisted form).
        token_ids: Integer vocab indices for training (optional, not persisted).
    """

    action_index: int
    action_name: str
    tokens: list[str]
    token_ids: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "action_index": self.action_index,
            "action_name": self.action_name,
            "tokens": self.tokens,
        }
        # token_ids intentionally omitted from persistence
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tier2Block:
        return cls(
            action_index=d["action_index"],
            action_name=d["action_name"],
            tokens=d["tokens"],
        )


@dataclass
class Tier3Slot:
    """A free-text value slot filled by the ValueFiller (Tier 3 output).

    Produced by ValueFiller.decode(), not by forward().
    """

    slot_id: str
    value_kind: str  # "string", "number", "boolean", "enum"
    value: str | int | float | bool
    source_param: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "slot_id": self.slot_id,
            "value_kind": self.value_kind,
            "value": self.value,
        }
        if self.source_param is not None:
            d["source_param"] = self.source_param
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Tier3Slot:
        return cls(
            slot_id=d["slot_id"],
            value_kind=d["value_kind"],
            value=d["value"],
            source_param=d.get("source_param"),
        )


# ---------------------------------------------------------------------------
# Training data types
# ---------------------------------------------------------------------------


@dataclass
class TypedIRExample:
    """A single training/eval example with three-tier decomposition.

    This is the canonical persisted format for typed_ir_train.jsonl
    and typed_ir_eval.jsonl.
    """

    shortcut_id: str
    system_prompt: str
    prompt: str
    dsl: str
    shortcut_name: str
    tier1_tokens: list[str]
    tier2_blocks: list[Tier2Block]
    tier3_slots: list[Tier3Slot]
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shortcut_id": self.shortcut_id,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt,
            "dsl": self.dsl,
            "shortcut_name": self.shortcut_name,
            "tier1_tokens": self.tier1_tokens,
            "tier2_blocks": [b.to_dict() for b in self.tier2_blocks],
            "tier3_slots": [s.to_dict() for s in self.tier3_slots],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TypedIRExample:
        return cls(
            shortcut_id=d["shortcut_id"],
            system_prompt=d.get("system_prompt", ""),
            prompt=d["prompt"],
            dsl=d["dsl"],
            shortcut_name=d.get("shortcut_name", ""),
            tier1_tokens=d["tier1_tokens"],
            tier2_blocks=[Tier2Block.from_dict(b) for b in d.get("tier2_blocks", [])],
            tier3_slots=[Tier3Slot.from_dict(s) for s in d.get("tier3_slots", [])],
            metadata=d.get("metadata", {}),
        )


@dataclass
class NegativeBankEntry:
    """A contrastive pair for margin loss training.

    Seeded from the linter repair taxonomy (215 hallucination aliases,
    8 repair categories) and distillation error logs.
    """

    prompt: str
    shortcut_id: str
    positive: TypedIRExample
    negative: TypedIRExample | None
    error_tags: list[str]
    source: str  # "distillation", "synthetic_mutation", "linter_repair"
    lint_changes: list[dict[str, str | float]]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "prompt": self.prompt,
            "shortcut_id": self.shortcut_id,
            "positive": self.positive.to_dict(),
            "error_tags": self.error_tags,
            "source": self.source,
            "lint_changes": self.lint_changes,
        }
        if self.negative is not None:
            d["negative"] = self.negative.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NegativeBankEntry:
        return cls(
            prompt=d["prompt"],
            shortcut_id=d["shortcut_id"],
            positive=TypedIRExample.from_dict(d["positive"]),
            negative=TypedIRExample.from_dict(d["negative"]) if d.get("negative") else None,
            error_tags=d.get("error_tags", []),
            source=d.get("source", "unknown"),
            lint_changes=d.get("lint_changes", []),
        )


# ---------------------------------------------------------------------------
# Domain gate types
# ---------------------------------------------------------------------------


@dataclass
class OODPrompt:
    """An out-of-domain prompt for domain gate training/evaluation."""

    prompt: str
    label: str  # "in_domain" or "ood"
    category: str  # e.g., "general_chat", "code_generation", "creative_writing"
    source: str  # "seed_file", "synthetic", "manual"

    def to_dict(self) -> dict[str, str]:
        return {
            "prompt": self.prompt,
            "label": self.label,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> OODPrompt:
        return cls(
            prompt=d["prompt"],
            label=d["label"],
            category=d.get("category", "unknown"),
            source=d.get("source", "unknown"),
        )


@dataclass
class GateDecision:
    """Domain gate classification result."""

    in_domain: bool
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"in_domain": self.in_domain, "confidence": self.confidence}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GateDecision:
        return cls(in_domain=d["in_domain"], confidence=d["confidence"])


# ---------------------------------------------------------------------------
# Coverage metrics (Phase 0 acceptance)
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    """Vocab coverage metric used by Phase 0 acceptance tests.

    Schema:
        scope: "tier1" or "tier2"
        dataset: "eval" (always measured against frozen eval set)
        total_tokens_in_eval: count of tokens observed in eval data
        covered: count of tokens present in vocabulary
        uncovered: de-duplicated, sorted list of missing tokens
        coverage_pct: covered / total_tokens_in_eval * 100
    """

    scope: str  # "tier1" or "tier2"
    dataset: str  # "eval"
    total_tokens_in_eval: int
    covered: int
    uncovered: list[str]
    coverage_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "dataset": self.dataset,
            "total_tokens_in_eval": self.total_tokens_in_eval,
            "covered": self.covered,
            "uncovered": self.uncovered,
            "coverage_pct": self.coverage_pct,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CoverageReport:
        return cls(
            scope=d["scope"],
            dataset=d["dataset"],
            total_tokens_in_eval=d["total_tokens_in_eval"],
            covered=d["covered"],
            uncovered=sorted(d.get("uncovered", [])),
            coverage_pct=d["coverage_pct"],
        )
