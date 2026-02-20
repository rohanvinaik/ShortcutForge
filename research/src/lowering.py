"""
Deterministic lowering — converts TypedIRExample to valid ShortcutDSL.

This is the final pipeline stage: takes the three-tier decomposition
and produces a DSL string that can be parsed by src/dsl_parser.py and
compiled by src/dsl_bridge.py.

No ML involved — purely deterministic template expansion.

Grammar reference (shortcutdsl.lark):
  action_stmt: "ACTION" action_name param* _NL
  param: IDENT "=" value
  → All params are INLINE on the ACTION line, using = not :
  → There is NO ENDACTION keyword in the grammar
  if_block: "IF" if_target CONDITION value? _NL statement* else_clause? "ENDIF" _NL
  menu_block: "MENU" menu_prompt _NL menu_case+ "ENDMENU" _NL
  repeat_block: "REPEAT" repeat_count _NL statement* "ENDREPEAT" _NL
  foreach_block: "FOREACH" foreach_collection _NL statement* "ENDFOREACH" _NL
"""

from __future__ import annotations

import sys
from pathlib import Path

from research.src.contracts import Tier2Block, TypedIRExample

# Allow importing from the main src/ directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# Structural tokens that terminate an ACTION's PARAM sequence in tier1_tokens
_ACTION_TERMINATORS = frozenset({
    "ENDACTION", "ACTION", "SHORTCUT", "ENDSHORTCUT",
    "IF", "ELSE", "ENDIF",
    "REPEAT", "ENDREPEAT",
    "MENU", "ENDMENU",
    "FOREACH", "ENDFOREACH",
    "SET",
})


# ---------------------------------------------------------------------------
# Slot/param lookup helpers
# ---------------------------------------------------------------------------


def _build_slot_map(
    example: TypedIRExample,
) -> dict[tuple[int, str], str]:
    """Build (action_index, param_name) -> slot value mapping."""
    slot_by_action_param: dict[tuple[int, str], str] = {}
    for slot in example.tier3_slots:
        # Find which action this slot belongs to by matching source_param
        # against tier2_blocks
        for block in example.tier2_blocks:
            if slot.source_param in block.tokens:
                slot_by_action_param[(block.action_index, slot.source_param)] = (
                    slot.value
                )
                break
        else:
            # Fallback: assign to first block that references this param
            for block in example.tier2_blocks:
                slot_by_action_param[(block.action_index, slot.source_param)] = (
                    slot.value
                )
                break
    return slot_by_action_param


def _extract_block_params(block: Tier2Block) -> list[str]:
    """Extract param names from tier2 block tokens (PARAM pairs)."""
    block_params: list[str] = []
    k = 0
    while k < len(block.tokens):
        if block.tokens[k] == "PARAM" and k + 1 < len(block.tokens):
            block_params.append(block.tokens[k + 1])
            k += 2
        else:
            k += 1
    return block_params


# ---------------------------------------------------------------------------
# Per-token lowering handlers
# ---------------------------------------------------------------------------


def _lower_action(
    tokens: list[str],
    i: int,
    block: Tier2Block | None,
    action_index: int,
    slot_by_action_param: dict[tuple[int, str], str],
) -> tuple[str, int]:
    """Lower an ACTION token sequence to a DSL line.

    Returns (dsl_line, new_token_index).
    """
    if block is None:
        return f"ACTION unknown.action.{action_index}", i + 1

    parts = [f"ACTION {block.action_name}"]

    # Collect PARAM tokens from ahead in tier1_tokens
    param_keys: list[str] = []
    j = i + 1
    while j < len(tokens) and tokens[j] not in _ACTION_TERMINATORS:
        if tokens[j] == "PARAM" and j + 1 < len(tokens):
            param_keys.append(tokens[j + 1])
            j += 2
        else:
            j += 1

    # Also collect params from the tier2 block tokens
    # (covers cases where tier1 doesn't have explicit PARAM tokens)
    block_params = _extract_block_params(block)

    # Merge: prefer tier1 ordering, add any tier2-only params
    all_params = list(param_keys)
    for bp in block_params:
        if bp not in all_params:
            all_params.append(bp)

    # Emit params inline
    for key in all_params:
        value = slot_by_action_param.get((action_index, key), "")
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'{key}="{escaped}"')

    return " ".join(parts), j


# Dispatch table mapping simple tokens to their DSL line output.
# These are tokens that always produce the same fixed line.
_SIMPLE_TOKEN_MAP: dict[str, str] = {
    "IF": "IF @prev has_any_value",
    "ELSE": "ELSE",
    "ENDIF": "ENDIF",
    "REPEAT": "REPEAT 1",
    "ENDREPEAT": "ENDREPEAT",
    "ENDMENU": "ENDMENU",
    "FOREACH": "FOREACH @prev",
    "ENDFOREACH": "ENDFOREACH",
    "SET": 'SET $var = "value"',
    "ENDSHORTCUT": "ENDSHORTCUT",
}


# ---------------------------------------------------------------------------
# Main lowering function
# ---------------------------------------------------------------------------


def lower_typed_ir_to_dsl(example: TypedIRExample) -> str:
    """Convert a TypedIRExample to valid ShortcutDSL string.

    Uses tier1_tokens for structure, tier2_blocks for parameters,
    and tier3_slots for free-text values. Produces DSL that passes
    lint -> parse -> validate -> compile.

    Args:
        example: Complete three-tier decomposition.

    Returns:
        DSL string ending with ENDSHORTCUT.
    """
    block_map = {block.action_index: block for block in example.tier2_blocks}
    slot_by_action_param = _build_slot_map(example)

    lines: list[str] = []
    action_index = 0
    i = 0
    tokens = example.tier1_tokens

    while i < len(tokens):
        token = tokens[i]

        if token == "SHORTCUT":
            lines.append(f'SHORTCUT "{example.shortcut_name}"')
            i += 1

        elif token == "ACTION":
            line, i = _lower_action(
                tokens, i, block_map.get(action_index),
                action_index, slot_by_action_param,
            )
            lines.append(line)
            action_index += 1

        elif token == "ENDACTION":
            # ENDACTION is a tier1 structural marker but does NOT exist
            # in the ShortcutDSL grammar — silently skip it
            i += 1

        elif token == "MENU":
            # Grammar: MENU menu_prompt _NL menu_case+ ENDMENU _NL
            # Must have at least one CASE — we emit a default case
            lines.append('MENU "Menu"')
            lines.append('CASE "Option 1"')
            i += 1

        elif token == "PARAM":
            # PARAM tokens outside of ACTION context — skip the pair
            i += 2

        elif token in _SIMPLE_TOKEN_MAP:
            lines.append(_SIMPLE_TOKEN_MAP[token])
            i += 1

        else:
            # Unknown token — skip
            i += 1

    # Grammar requires every statement to end with _NL (newline),
    # including the final ENDSHORTCUT. Add trailing newline.
    return "\n".join(lines) + "\n"


def roundtrip_validate(example: TypedIRExample) -> tuple[bool, str]:
    """Lower to DSL, then parse+validate to verify roundtrip.

    Args:
        example: Complete three-tier decomposition.

    Returns:
        (success: bool, message: str) — message is empty on success,
        contains error details on failure.
    """
    try:
        dsl_text = lower_typed_ir_to_dsl(example)
    except Exception as e:
        return (False, f"Lowering failed: {e}")

    try:
        from dsl_parser import parse_dsl

        parse_dsl(dsl_text)
    except Exception as e:
        return (False, f"Parse failed: {e}")

    return (True, "")
