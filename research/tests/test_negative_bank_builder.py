"""Tests for the negative_bank_builder module."""

from __future__ import annotations

import copy
import random

import pytest

from src.behavioral_fingerprint import BehavioralFingerprint
from src.contracts import NegativeBankEntry, Tier2Block, TypedIRExample
from src.mutations import (
    action_category,
    mutate_action_swap,
    mutate_hallucinated_name,
    mutate_param_drop,
)
from src.negative_bank_builder import (
    MutationWeights,
    _parse_distillation_record,
    weights_from_fingerprint,
)

# ---------------------------------------------------------------------------
# MutationWeights
# ---------------------------------------------------------------------------


class TestMutationWeights:
    def test_defaults(self):
        w = MutationWeights()
        assert w.action_swap == 1.0
        assert w.param_drop == 1.0
        assert w.hallucinated_name == 1.0

    def test_custom_values(self):
        w = MutationWeights(action_swap=2.0, param_drop=0.5, hallucinated_name=3.0)
        assert w.action_swap == 2.0
        assert w.param_drop == 0.5
        assert w.hallucinated_name == 3.0

    def test_is_namedtuple(self):
        w = MutationWeights()
        assert isinstance(w, tuple)
        assert len(w) == 3


# ---------------------------------------------------------------------------
# weights_from_fingerprint
# ---------------------------------------------------------------------------


class TestWeightsFromFingerprint:
    def test_high_entropy_boosts_hallucinated_name(self):
        fp = BehavioralFingerprint(
            action_entropy=2.5,
            action_distribution={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            discreteness_score=0.8,
        )
        weights = weights_from_fingerprint(fp)
        assert weights.hallucinated_name > 1.0
        # hallucinated_name = 1.0 + 2.5 = 3.5
        assert weights.hallucinated_name == pytest.approx(3.5)

    def test_low_discreteness_boosts_param_drop(self):
        fp = BehavioralFingerprint(
            action_entropy=0.5,
            action_distribution={"a": 0.9, "b": 0.1},
            discreteness_score=0.2,
        )
        weights = weights_from_fingerprint(fp)
        assert weights.param_drop > 1.0
        # param_drop = 1.0 + max(0, 1.0 - 0.2) = 1.8
        assert weights.param_drop == pytest.approx(1.8)

    def test_concentrated_distribution_boosts_action_swap(self):
        # Few actions (2) in distribution -> high action_swap weight
        fp = BehavioralFingerprint(
            action_entropy=0.5,
            action_distribution={"a": 0.9, "b": 0.1},
            discreteness_score=0.9,
        )
        weights = weights_from_fingerprint(fp)
        # action_swap = 1.0 + max(0, 1.0 - 2/10) = 1.0 + 0.8 = 1.8
        assert weights.action_swap == pytest.approx(1.8)

    def test_many_actions_low_action_swap(self):
        # Many actions (15) -> action_swap stays low
        dist = {f"action_{i}": 1 / 15 for i in range(15)}
        fp = BehavioralFingerprint(
            action_entropy=3.0,
            action_distribution=dist,
            discreteness_score=0.5,
        )
        weights = weights_from_fingerprint(fp)
        # action_swap = 1.0 + max(0, 1.0 - 15/10) = 1.0 + max(0, -0.5) = 1.0
        assert weights.action_swap == pytest.approx(1.0)

    def test_high_discreteness_low_param_drop(self):
        fp = BehavioralFingerprint(
            action_entropy=1.0,
            action_distribution={"a": 0.5, "b": 0.5},
            discreteness_score=1.5,
        )
        weights = weights_from_fingerprint(fp)
        # param_drop = 1.0 + max(0, 1.0 - 1.5) = 1.0
        assert weights.param_drop == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# action_category
# ---------------------------------------------------------------------------


class TestActionCategory:
    def test_workflow_prefix(self):
        assert action_category("is.workflow.actions.showalert") == "is.workflow.actions"

    def test_dotted_name(self):
        assert action_category("com.apple.mobiletimer") == "com.apple"

    def test_short_name(self):
        assert action_category("alert") == "short"


# ---------------------------------------------------------------------------
# Mutation functions
# ---------------------------------------------------------------------------


def _make_example(
    action_name: str = "is.workflow.actions.showalert",
    tokens: list[str] | None = None,
) -> TypedIRExample:
    """Helper to build a minimal TypedIRExample for mutation tests."""
    if tokens is None:
        tokens = ["PARAM", "title", "PARAM", "message"]
    return TypedIRExample(
        shortcut_id="test_1",
        system_prompt="",
        prompt="Test prompt",
        dsl='SHORTCUT "Test"\nACTION is.workflow.actions.showalert\nENDSHORTCUT\n',
        shortcut_name="Test",
        tier1_tokens=["SHORTCUT", "ACTION", "ENDSHORTCUT"],
        tier2_blocks=[
            Tier2Block(action_index=0, action_name=action_name, tokens=tokens),
        ],
        tier3_slots=[],
        metadata={},
    )


class TestMutateActionSwap:
    def test_swaps_to_different_action(self):
        random.seed(42)
        ex = _make_example()
        block = ex.tier2_blocks[0]
        neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]
        all_names = [
            "is.workflow.actions.showalert",
            "is.workflow.actions.gettext",
            "is.workflow.actions.setbrightness",
        ]
        categories = {
            "is.workflow.actions": all_names,
        }

        result = mutate_action_swap(block, neg_blocks, 0, all_names, categories)
        assert result is not None
        mutated_blocks, lint_change = result
        assert mutated_blocks[0].action_name != "is.workflow.actions.showalert"
        assert lint_change["kind"] == "action_swap"
        assert lint_change["original"] == "is.workflow.actions.showalert"

    def test_returns_none_when_no_candidates(self):
        ex = _make_example(action_name="unique_action")
        block = ex.tier2_blocks[0]
        neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]
        # Only one action available, same as original
        result = mutate_action_swap(
            block, neg_blocks, 0, ["unique_action"], {"short": ["unique_action"]}
        )
        assert result is None


class TestMutateParamDrop:
    def test_reduces_token_count(self):
        random.seed(42)
        ex = _make_example(tokens=["PARAM", "title", "PARAM", "message"])
        block = ex.tier2_blocks[0]
        neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]

        result = mutate_param_drop(block, neg_blocks, 0)
        assert result is not None
        mutated_blocks, lint_change = result
        assert len(mutated_blocks[0].tokens) < len(block.tokens)
        assert lint_change["kind"] == "param_drop"

    def test_returns_none_for_single_token(self):
        ex = _make_example(tokens=["X"])
        block = ex.tier2_blocks[0]
        neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]

        result = mutate_param_drop(block, neg_blocks, 0)
        assert result is None

    def test_drops_non_param_token(self):
        random.seed(42)
        ex = _make_example(tokens=["FOO", "BAR", "BAZ"])
        block = ex.tier2_blocks[0]
        neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]

        result = mutate_param_drop(block, neg_blocks, 0)
        assert result is not None
        mutated_blocks, lint_change = result
        assert len(mutated_blocks[0].tokens) == 2
        assert lint_change["replacement"] == "<removed>"


class TestMutateHallucinatedName:
    def test_changes_action_name(self):
        random.seed(42)
        ex = _make_example()
        block = ex.tier2_blocks[0]
        neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]
        hallucination_aliases = list(
            __import__(
                "dsl_linter", fromlist=["ActionResolver"]
            ).ActionResolver.HALLUCINATION_ALIASES.keys()
        )

        result = mutate_hallucinated_name(block, neg_blocks, 0, hallucination_aliases)
        mutated_blocks, lint_change = result
        assert mutated_blocks[0].action_name != "is.workflow.actions.showalert"
        assert lint_change["kind"] == "hallucinated_name"
        assert lint_change["original"] == "is.workflow.actions.showalert"


# ---------------------------------------------------------------------------
# _parse_distillation_record
# ---------------------------------------------------------------------------


class TestParseDistillationRecord:
    def test_returns_none_for_identical_raw_canonicalized(self):
        record = {
            "shortcut_id": "test_1",
            "prompt": "Test prompt",
            "raw_output": "same text",
            "canonicalized_output": "same text",
            "lint_changes": [],
        }
        result = _parse_distillation_record(record, 1)
        assert result is None

    def test_returns_entry_for_actual_correction(self):
        record = {
            "shortcut_id": "test_2",
            "prompt": "Test prompt",
            "raw_output": "bad output",
            "canonicalized_output": "good output",
            "lint_changes": [
                {
                    "kind": "action",
                    "original": "bad_action",
                    "replacement": "good_action",
                    "confidence": 0.95,
                }
            ],
        }
        result = _parse_distillation_record(record, 2)
        assert result is not None
        assert isinstance(result, NegativeBankEntry)
        assert result.source == "distillation"
        assert result.positive.dsl == "good output"
        assert result.negative is not None
        assert result.negative.dsl == "bad output"
        assert "action" in result.error_tags

    def test_returns_entry_for_failure_category(self):
        record = {
            "shortcut_id": "test_3",
            "prompt": "Test prompt",
            "raw_output": "same text",
            "canonicalized_output": "same text",
            "failure_category": "parse_error",
            "lint_changes": [],
        }
        result = _parse_distillation_record(record, 3)
        assert result is not None
        assert "parse_error" in result.error_tags

    def test_default_shortcut_id(self):
        record = {
            "prompt": "Test",
            "raw_output": "bad",
            "canonicalized_output": "good",
            "lint_changes": [],
        }
        result = _parse_distillation_record(record, 42)
        assert result is not None
        assert result.shortcut_id == "distill_42"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
