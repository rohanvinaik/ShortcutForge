#!/usr/bin/env python3
"""Tests for research.src.ir_decomposer -- IR decomposition and value serialization."""

from __future__ import annotations

import pytest

lark = pytest.importorskip("lark")

from src.ir_decomposer import (  # noqa: E402
    _serialize_ir_value,
    decompose_dsl_to_typed_ir,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_record(
    shortcut_id: str = "test1",
    dsl: str = 'SHORTCUT "Test"\nACTION is.workflow.actions.timer duration=5\nENDSHORTCUT\n',
) -> dict:
    return {
        "shortcut_id": shortcut_id,
        "messages": [
            {"role": "system", "content": "You are a shortcut generator."},
            {"role": "user", "content": "Set a timer"},
            {"role": "assistant", "content": dsl},
        ],
    }


# ---------------------------------------------------------------------------
# _serialize_ir_value tests
# ---------------------------------------------------------------------------


class TestSerializeIRValue:
    """Test value serialization for various IR value types."""

    def test_string_value(self):
        from dsl_ir import StringValue

        text, kind = _serialize_ir_value(StringValue("hello"))
        assert text == "hello"
        assert kind == "string"

    def test_number_value(self):
        from dsl_ir import NumberValue

        text, kind = _serialize_ir_value(NumberValue(42))
        assert text == "42"
        assert kind == "number"

    def test_number_value_float(self):
        from dsl_ir import NumberValue

        text, kind = _serialize_ir_value(NumberValue(3.14))
        assert text == "3.14"
        assert kind == "number"

    def test_bool_value_true(self):
        from dsl_ir import BoolValue

        text, kind = _serialize_ir_value(BoolValue(True))
        assert text == "true"
        assert kind == "boolean"

    def test_bool_value_false(self):
        from dsl_ir import BoolValue

        text, kind = _serialize_ir_value(BoolValue(False))
        assert text == "false"
        assert kind == "boolean"

    def test_var_ref(self):
        from dsl_ir import VarRef

        text, kind = _serialize_ir_value(VarRef("myVar"))
        assert text == "$myVar"
        assert kind == "var_ref"

    def test_handle_ref(self):
        from dsl_ir import HandleRef

        text, kind = _serialize_ir_value(HandleRef("CurrentDate"))
        assert text == "@CurrentDate"
        assert kind == "handle_ref"

    def test_list_literal(self):
        from dsl_ir import ListLiteral, StringValue

        val = ListLiteral([StringValue("a"), StringValue("b")])
        text, kind = _serialize_ir_value(val)
        assert kind == "list"
        assert text == "[a, b]"

    def test_dict_literal(self):
        from dsl_ir import DictLiteral, StringValue

        val = DictLiteral([("key", StringValue("val"))])
        text, kind = _serialize_ir_value(val)
        assert kind == "dict"
        assert '"key": val' in text

    def test_unknown_type_fallback(self):
        """Unknown types should return str(value) with 'unknown' kind."""

        class FakeIRValue:
            def __str__(self):
                return "fake"

        text, kind = _serialize_ir_value(FakeIRValue())
        assert kind == "unknown"
        assert text == "fake"


# ---------------------------------------------------------------------------
# decompose_dsl_to_typed_ir tests
# ---------------------------------------------------------------------------


class TestDecomposeDslToTypedIR:
    """Test the top-level decomposition function."""

    def test_valid_record_produces_typed_ir(self):
        record = _make_valid_record()
        result = decompose_dsl_to_typed_ir(record)
        assert result is not None
        assert result.shortcut_id == "test1"
        assert result.shortcut_name == "Test"
        assert "SHORTCUT" in result.tier1_tokens
        assert "ENDSHORTCUT" in result.tier1_tokens
        assert result.tier1_tokens[0] == "SHORTCUT"
        assert result.tier1_tokens[-1] == "ENDSHORTCUT"

    def test_valid_record_has_tier2_blocks(self):
        record = _make_valid_record()
        result = decompose_dsl_to_typed_ir(record)
        assert result is not None
        assert len(result.tier2_blocks) >= 1
        # Linter may canonicalize action names (e.g. timer -> settimer)
        assert result.tier2_blocks[0].action_name.startswith("is.workflow.actions.")

    def test_valid_record_has_tier3_slots(self):
        record = _make_valid_record()
        result = decompose_dsl_to_typed_ir(record)
        assert result is not None
        assert len(result.tier3_slots) >= 1
        assert result.tier3_slots[0].source_param == "duration"

    def test_valid_record_metadata(self):
        record = _make_valid_record()
        result = decompose_dsl_to_typed_ir(record)
        assert result is not None
        assert "action_count" in result.metadata
        assert "tier1_len" in result.metadata
        assert result.metadata["action_count"] >= 1

    def test_too_few_messages_returns_none(self):
        record = {
            "shortcut_id": "bad1",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
            ],
        }
        result = decompose_dsl_to_typed_ir(record, verbose=True)
        assert result is None

    def test_empty_dsl_returns_none(self):
        record = {
            "shortcut_id": "bad2",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "   "},
            ],
        }
        result = decompose_dsl_to_typed_ir(record, verbose=True)
        assert result is None

    def test_unparseable_dsl_returns_none(self):
        record = {
            "shortcut_id": "bad3",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "THIS IS NOT VALID DSL AT ALL"},
            ],
        }
        result = decompose_dsl_to_typed_ir(record, verbose=True)
        assert result is None

    def test_no_messages_returns_none(self):
        record = {"shortcut_id": "bad4", "messages": []}
        result = decompose_dsl_to_typed_ir(record)
        assert result is None

    def test_missing_shortcut_id_uses_default(self):
        record = {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Set a timer"},
                {
                    "role": "assistant",
                    "content": 'SHORTCUT "Test"\nACTION is.workflow.actions.timer duration=5\nENDSHORTCUT\n',
                },
            ],
        }
        result = decompose_dsl_to_typed_ir(record)
        assert result is not None
        assert result.shortcut_id == "unknown"

    def test_prompt_preserved(self):
        record = _make_valid_record()
        result = decompose_dsl_to_typed_ir(record)
        assert result is not None
        assert result.prompt == "Set a timer"
        assert result.system_prompt == "You are a shortcut generator."
