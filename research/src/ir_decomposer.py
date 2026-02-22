"""IR decomposition — three-tier typed IR from parsed ShortcutDSL.

Lints, parses, and walks IR to produce Tier 1/2/3 representations.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Allow importing from the main ShortcutForge src/ directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_COMPILER_SRC = _PROJECT_ROOT / "src"
if str(_COMPILER_SRC) not in sys.path:
    sys.path.insert(0, str(_COMPILER_SRC))

from dsl_ir import (  # noqa: E402
    ActionStatement,
    BoolValue,
    Comment,
    DictLiteral,
    ForeachBlock,
    HandleRef,
    HeadersLiteral,
    IfBlock,
    InterpolatedString,
    IRValue,
    ListLiteral,
    MenuBlock,
    NumberValue,
    QuantityLiteral,
    RepeatBlock,
    SetVariable,
    ShortcutIR,
    Statement,
    StringValue,
    VarRef,
    iter_child_blocks,
)
from dsl_linter import lint_dsl  # noqa: E402
from dsl_parser import parse_dsl  # noqa: E402

from src.contracts import Tier2Block, Tier3Slot, TypedIRExample  # noqa: E402

logger = logging.getLogger(__name__)


# Value serialization


def _serialize_string(value: StringValue) -> tuple[str, str]:
    return value.value, "string"


def _serialize_number(value: NumberValue) -> tuple[str, str]:
    return str(value.value), "number"


def _serialize_bool(value: BoolValue) -> tuple[str, str]:
    return "true" if value.value else "false", "boolean"


def _serialize_var_ref(value: VarRef) -> tuple[str, str]:
    return f"${value.name}", "var_ref"


def _serialize_handle_ref(value: HandleRef) -> tuple[str, str]:
    return f"@{value.kind}", "handle_ref"


def _serialize_interpolated(value: InterpolatedString) -> tuple[str, str]:
    """Reconstruct backtick string from parts."""
    parts_str = []
    for part in value.parts:
        if isinstance(part, StringValue):
            parts_str.append(part.value)
        elif isinstance(part, VarRef):
            parts_str.append(f"{{{part.name}}}")
        elif isinstance(part, HandleRef):
            parts_str.append(f"{{{part.kind}}}")
        else:
            parts_str.append(str(part))
    return "`" + "".join(parts_str) + "`", "interpolated"


def _serialize_dict(value: DictLiteral) -> tuple[str, str]:
    entries = []
    for key, val in value.entries:
        val_str, _ = _serialize_ir_value(val)
        entries.append(f'"{key}": {val_str}')
    return "{" + ", ".join(entries) + "}", "dict"


def _serialize_list(value: ListLiteral) -> tuple[str, str]:
    items = []
    for item in value.items:
        item_str, _ = _serialize_ir_value(item)
        items.append(item_str)
    return "[" + ", ".join(items) + "]", "list"


def _serialize_quantity(value: QuantityLiteral) -> tuple[str, str]:
    if isinstance(value.magnitude, (VarRef, HandleRef)):
        mag_str, _ = _serialize_ir_value(value.magnitude)
    else:
        mag_str = str(value.magnitude)
    return f'QTY({mag_str}, "{value.unit}")', "quantity"


def _serialize_headers(value: HeadersLiteral) -> tuple[str, str]:
    entries = []
    for key, val in value.entries:
        val_str, _ = _serialize_ir_value(val)
        entries.append(f'"{key}": {val_str}')
    return "HEADERS {" + ", ".join(entries) + "}", "headers"


_SERIALIZE_DISPATCH: dict[type, Any] = {
    StringValue: _serialize_string,
    NumberValue: _serialize_number,
    BoolValue: _serialize_bool,
    VarRef: _serialize_var_ref,
    HandleRef: _serialize_handle_ref,
    InterpolatedString: _serialize_interpolated,
    DictLiteral: _serialize_dict,
    ListLiteral: _serialize_list,
    QuantityLiteral: _serialize_quantity,
    HeadersLiteral: _serialize_headers,
}


def _serialize_ir_value(value: IRValue) -> tuple[str, str]:
    """Convert an IR value to (string_repr, value_kind).

    Returns a human-readable string representation and a kind tag
    used by Tier3Slot.
    """
    handler = _SERIALIZE_DISPATCH.get(type(value))
    if handler is not None:
        return handler(value)
    return str(value), "unknown"


# ---------------------------------------------------------------------------
# IR walking -- three-tier decomposition
# ---------------------------------------------------------------------------


# Mapping from context_label (from iter_child_blocks) to (open_token, close_token).
# IfBlock emits open/close at the statement level since it has ELSE handling;
# the child bodies produced by iter_child_blocks use "if_then" and "if_else".
_BLOCK_OPEN_CLOSE: dict[type, tuple[str, str]] = {
    RepeatBlock: ("REPEAT", "ENDREPEAT"),
    ForeachBlock: ("FOREACH", "ENDFOREACH"),
    MenuBlock: ("MENU", "ENDMENU"),
}


def _walk_action(
    stmt: ActionStatement,
    tier1: list[str],
    tier2_blocks: list[Tier2Block],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
    action_idx: int,
) -> tuple[int, int]:
    """Emit tier1/tier2/tier3 data for an ActionStatement."""
    tier1.append("ACTION")
    param_tokens: list[str] = []
    for param_name in stmt.params:
        tier1.append("PARAM")
        tier1.append(param_name)
        param_tokens.append("PARAM")
        param_tokens.append(param_name)

    tier2_blocks.append(
        Tier2Block(
            action_index=action_idx,
            action_name=stmt.action_name,
            tokens=param_tokens,
        )
    )

    for param_name, param_value in stmt.params.items():
        value_str, value_kind = _serialize_ir_value(param_value)
        tier3_slots.append(
            Tier3Slot(
                slot_id=f"s{slot_counter}",
                value_kind=value_kind,
                value=value_str,
                source_param=param_name,
            )
        )
        slot_counter += 1

    action_idx += 1
    return slot_counter, action_idx


def _walk_set_variable(
    stmt: SetVariable,
    tier1: list[str],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
) -> int:
    """Emit tier1/tier3 data for a SetVariable statement."""
    tier1.append("SET")
    value_str, value_kind = _serialize_ir_value(stmt.value)
    tier3_slots.append(
        Tier3Slot(
            slot_id=f"s{slot_counter}",
            value_kind=value_kind,
            value=value_str,
            source_param=stmt.var_name,
        )
    )
    return slot_counter + 1


def _walk_control_block(
    stmt: Statement,
    tier1: list[str],
    tier2_blocks: list[Tier2Block],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
    action_idx: int,
) -> tuple[int, int]:
    """Emit tier1 tokens and recurse into child blocks for control-flow statements.

    Handles IfBlock (with ELSE), RepeatBlock, ForeachBlock, and MenuBlock
    by using iter_child_blocks for recursion and emitting block-specific tokens.
    """
    if isinstance(stmt, IfBlock):
        tier1.append("IF")
        for body, ctx, _is_loop in iter_child_blocks(stmt):
            if ctx == "if_else":
                tier1.append("ELSE")
            slot_counter, action_idx = _walk_statements(
                body, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )
        tier1.append("ENDIF")
    else:
        open_tok, close_tok = _BLOCK_OPEN_CLOSE[type(stmt)]
        tier1.append(open_tok)
        for body, _ctx, _is_loop in iter_child_blocks(stmt):
            slot_counter, action_idx = _walk_statements(
                body, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )
        tier1.append(close_tok)

    return slot_counter, action_idx


def _walk_statements(
    stmts: list[Statement],
    tier1: list[str],
    tier2_blocks: list[Tier2Block],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
    action_idx: int,
) -> tuple[int, int]:
    """Recursive walk over Statement list, populating all three tiers.

    Returns updated (slot_counter, action_idx).
    """
    for stmt in stmts:
        if isinstance(stmt, Comment):
            continue
        elif isinstance(stmt, ActionStatement):
            slot_counter, action_idx = _walk_action(
                stmt, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )
        elif isinstance(stmt, SetVariable):
            slot_counter = _walk_set_variable(stmt, tier1, tier3_slots, slot_counter)
        elif isinstance(stmt, (IfBlock, RepeatBlock, ForeachBlock, MenuBlock)):
            slot_counter, action_idx = _walk_control_block(
                stmt, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )

    return slot_counter, action_idx


# ---------------------------------------------------------------------------
# Per-record conversion
# ---------------------------------------------------------------------------


def _extract_record_parts(
    raw_record: dict[str, Any],
    verbose: bool,
) -> tuple[str, str, str, str] | None:
    """Extract shortcut_id, system_prompt, user_prompt, dsl_text from a record.

    Returns None if the record is invalid (too few messages or empty DSL).
    """
    shortcut_id = raw_record.get("shortcut_id", "unknown")
    messages = raw_record.get("messages", [])

    if len(messages) < 3:
        if verbose:
            logger.warning("[%s] Fewer than 3 messages, skipping", shortcut_id)
        return None

    dsl_text = messages[2].get("content", "")
    if not dsl_text.strip():
        if verbose:
            logger.warning("[%s] Empty DSL text, skipping", shortcut_id)
        return None

    return (
        shortcut_id,
        messages[0].get("content", ""),
        messages[1].get("content", ""),
        dsl_text,
    )


def _canonicalize_and_parse(
    dsl_text: str,
    shortcut_id: str,
    verbose: bool,
) -> tuple[ShortcutIR, bool] | None:
    """Lint and parse DSL text to IR. Returns (ir, lint_applied) or None."""
    canonicalized = dsl_text
    lint_applied = False
    try:
        lint_result = lint_dsl(dsl_text)
        canonicalized = lint_result.text
        lint_applied = lint_result.was_modified
    except Exception as e:
        if verbose:
            logger.warning("[%s] Lint failed (%s), using raw DSL", shortcut_id, e)

    try:
        return parse_dsl(canonicalized), lint_applied
    except Exception as e:
        if lint_applied:
            try:
                return parse_dsl(dsl_text), lint_applied
            except Exception:
                pass
        if verbose:
            logger.warning("[%s] Parse failed: %s", shortcut_id, e)
        return None


def decompose_dsl_to_typed_ir(
    raw_record: dict[str, Any],
    verbose: bool = False,
) -> TypedIRExample | None:
    """Convert a single raw training record to TypedIRExample.

    Returns None on failure (with logging if verbose).
    """
    parts = _extract_record_parts(raw_record, verbose)
    if parts is None:
        return None
    shortcut_id, system_prompt, user_prompt, dsl_text = parts

    parsed = _canonicalize_and_parse(dsl_text, shortcut_id, verbose)
    if parsed is None:
        return None
    ir, lint_applied = parsed

    # Walk IR to produce three tiers
    tier1_tokens: list[str] = ["SHORTCUT"]
    tier2_blocks: list[Tier2Block] = []
    tier3_slots: list[Tier3Slot] = []

    _walk_statements(
        ir.statements,
        tier1_tokens,
        tier2_blocks,
        tier3_slots,
        slot_counter=0,
        action_idx=0,
    )
    tier1_tokens.append("ENDSHORTCUT")

    return TypedIRExample(
        shortcut_id=shortcut_id,
        system_prompt=system_prompt,
        prompt=user_prompt,
        dsl=dsl_text,
        shortcut_name=ir.name,
        tier1_tokens=tier1_tokens,
        tier2_blocks=tier2_blocks,
        tier3_slots=tier3_slots,
        metadata={
            "action_count": ir.action_count(),
            "tier1_len": len(tier1_tokens),
            "tier2_count": len(tier2_blocks),
            "tier3_count": len(tier3_slots),
            "lint_applied": lint_applied,
            "statement_count": len(ir.statements),
        },
    )
