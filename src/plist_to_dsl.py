"""
Reverse Compiler: plist → ShortcutDSL.

Reads a .shortcut file (binary plist) and produces DSL text that represents it.
This is used for:
  1. Training data generation (Cassinelli corpus → DSL examples)
  2. Round-trip validation (plist → DSL → IR → compile → compare)
  3. Human-readable shortcut inspection

The reverse compiler handles:
  - Flat action list → nested control flow blocks (IF/MENU/REPEAT/FOREACH)
  - UUID-based GroupingIdentifier → block nesting
  - WFTextTokenString → interpolated strings
  - WFTextTokenAttachment → @prev / $var references
  - WFQuantityFieldValue → QTY() literals
  - build_dict_items → dict literals
  - build_list → list literals
"""

from __future__ import annotations

import json
import plistlib
import re
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "references" / "action_catalog.json"

# System params to skip (compiler internals, not user-visible)
SKIP_PARAMS = {
    "UUID",
    "GroupingIdentifier",
    "WFControlFlowMode",
    "WFMenuItems",  # Derived from CASE labels
    "WFMenuItemTitle",  # Emitted by CASE
    "WFMenuItemAttributedTitle",  # Rich text version
    "WFMenuPrompt",  # Emitted by MENU
    "WFCondition",  # Emitted by IF condition
    "WFInput",  # Handled separately for control flow
    "AppIntentDescriptor",  # Pass through as JSON
    "WFEnumeration",  # Captured in IF condition compare
    "WFLinkEnumeration",  # Captured in IF condition compare
    "WFNumberValue",  # Captured in IF condition compare
    "WFConditionalActionString",  # Captured in IF condition compare
}

# Prefix-based skip (e.g., Show-* are UI visibility flags, not functional)
SKIP_PARAM_PREFIXES = ("Show-",)

# Control flow identifiers
CF_CONDITIONAL = "is.workflow.actions.conditional"
CF_MENU = "is.workflow.actions.choosefrommenu"
CF_REPEAT_COUNT = "is.workflow.actions.repeat.count"
CF_REPEAT_EACH = "is.workflow.actions.repeat.each"

CONTROL_FLOW_IDS = {CF_CONDITIONAL, CF_MENU, CF_REPEAT_COUNT, CF_REPEAT_EACH}


def _load_canonical_map() -> dict[str, str]:
    """Load reverse map: full identifier → short name."""
    with CATALOG_PATH.open() as f:
        catalog = json.load(f)
    cmap = catalog.get("_meta", {}).get("canonical_map", {})
    # Build reverse: full → short
    reverse = {}
    for short, full in cmap.items():
        # Prefer the shortest name
        if full not in reverse or len(short) < len(reverse[full]):
            reverse[full] = short
    return reverse


_REVERSE_MAP: dict[str, str] | None = None


def _get_reverse_map() -> dict[str, str]:
    global _REVERSE_MAP
    if _REVERSE_MAP is None:
        _REVERSE_MAP = _load_canonical_map()
    return _REVERSE_MAP


def _short_name(identifier: str) -> str:
    """Convert a full action identifier to its shortest name."""
    rmap = _get_reverse_map()
    if identifier in rmap:
        return rmap[identifier]
    # Strip common prefix
    if identifier.startswith("is.workflow.actions."):
        return identifier[len("is.workflow.actions.") :]
    return identifier


def _format_ref_type(value: dict, output_uuids: dict, var_tracker: dict) -> str | None:
    """Resolve a plist reference dict to a DSL ref string, or None if not a ref type."""
    ref_type = value.get("Type", "")
    if ref_type == "ActionOutput":
        return _format_action_output_ref(value, output_uuids)
    if ref_type == "Variable":
        return _format_variable_ref(value, var_tracker)
    if ref_type == "ExtensionInput":
        return "@input"
    if ref_type == "CurrentDate":
        return "@date"
    if ref_type in ("DeviceDetails", "Ask", "Clipboard", "AskEachTime"):
        return f"@{_sanitize_ident(ref_type)}"
    return None


def _escape_string(s: Any) -> str:
    """Escape a string for DSL output. Handles non-string values gracefully."""
    if not isinstance(s, str):
        s = str(s)
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def _format_value(value: Any, output_uuids: dict[str, str], var_tracker: dict) -> str:
    """Format a plist value as DSL text.

    Args:
        value: The plist value to format.
        output_uuids: Map of OutputUUID → action short info for @prev resolution.
        var_tracker: Tracks variable definitions for $var references.
    """
    if isinstance(value, str):
        return f'"{_escape_string(value)}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        # Emit as integer if possible
        if isinstance(value, float) and value == int(value):
            return str(int(value))
        return str(value)
    elif isinstance(value, bytes):
        # Binary data (NSData) — can't represent in DSL, use placeholder
        return f'"<binary:{len(value)}bytes>"'
    elif isinstance(value, dict):
        return _format_dict_value(value, output_uuids, var_tracker)
    elif isinstance(value, list):
        return _format_list_value(value, output_uuids, var_tracker)
    else:
        return f'"{_escape_string(str(value))}"'


def _format_dict_value(value: dict, output_uuids: dict, var_tracker: dict) -> str:
    """Format a dict value, detecting special types."""
    if not isinstance(value, dict):
        return f'"{_escape_string(str(value))}"'

    ser_dispatch = {
        "WFTextTokenAttachment": _format_attachment,
        "WFTextTokenString": _format_token_string,
        "WFQuantityFieldValue": _format_quantity,
    }
    ser_type = value.get("WFSerializationType", "")
    handler = ser_dispatch.get(ser_type)
    if handler is not None:
        return handler(value, output_uuids, var_tracker)

    # Check if it's a reference dict
    ref_str = _format_ref_type(value, output_uuids, var_tracker)
    if ref_str is not None:
        return ref_str

    # Plain dict — emit as dict literal
    entries = []
    for k, v in value.items():
        try:
            formatted = _format_value(v, output_uuids, var_tracker)
            entries.append(f'"{_escape_string(str(k))}": {formatted}')
        except Exception:
            entries.append(f'"{_escape_string(str(k))}": "<complex>"')
    return "{" + ", ".join(entries) + "}"


def _format_attachment(value: dict, output_uuids: dict, var_tracker: dict) -> str:
    """Format a WFTextTokenAttachment value."""
    inner = value.get("Value", {})
    if not isinstance(inner, dict):
        return f'"{_escape_string(str(inner))}"'
    ref_str = _format_ref_type(inner, output_uuids, var_tracker)
    if ref_str is not None:
        return ref_str
    return _format_dict_value(inner, output_uuids, var_tracker)


def _format_token_string(value: dict, output_uuids: dict, var_tracker: dict) -> str:
    """Format a WFTextTokenString as an interpolated string."""
    inner = value.get("Value", {})
    if not isinstance(inner, dict):
        # Value is a raw string or other primitive
        return f'"{_escape_string(str(inner))}"'
    raw_string = inner.get("string", "")
    attachments = inner.get("attachmentsByRange", {})

    if not attachments:
        # No variable references — just a plain string
        return f'"{_escape_string(raw_string)}"'

    # Parse attachments by position
    positions = {}
    for range_key, att in attachments.items():
        match = re.match(r"\{(\d+),\s*\d+\}", range_key)
        if match:
            pos = int(match.group(1))
            positions[pos] = att

    # Build interpolated string
    parts = _build_interp_parts(raw_string, positions, output_uuids, var_tracker)
    return _assemble_interp_string(parts)


def _resolve_attachment_ref(att: dict, output_uuids: dict, var_tracker: dict) -> str:
    """Resolve a single token-string attachment to a DSL ref string."""
    ref_str = _format_ref_type(att, output_uuids, var_tracker)
    if ref_str is not None:
        return ref_str
    var_name = att.get("VariableName", "")
    if var_name:
        return f"${var_name.replace(' ', '_')}"
    return "@prev"


def _build_interp_parts(
    raw_string: str, positions: dict, output_uuids: dict, var_tracker: dict
) -> list[str]:
    """Build list of text and {ref} parts from a token string."""
    parts: list[str] = []
    for char_idx, ch in enumerate(raw_string):
        if ch == "\ufffc" and char_idx in positions:
            ref_str = _resolve_attachment_ref(positions[char_idx], output_uuids, var_tracker)
            parts.append("{" + ref_str + "}")
        elif parts and not parts[-1].startswith("{"):
            parts[-1] += ch
        else:
            parts.append(ch)
    return parts


def _assemble_interp_string(parts: list[str]) -> str:
    """Assemble parts into a backtick-interpolated or quoted string."""
    has_conflict = any(
        not p.startswith("{") and ("`" in p or "{" in p or "}" in p) for p in parts
    )
    if has_conflict:
        flat = [p[1:-1] if (p.startswith("{") and p.endswith("}")) else p for p in parts]
        return f'"{_escape_string("".join(flat))}"'
    return "`" + "".join(parts) + "`"


def _format_quantity(
    value: dict, output_uuids: dict | None = None, var_tracker: dict | None = None
) -> str:
    """Format a WFQuantityFieldValue as QTY(magnitude, "unit")."""
    inner = value.get("Value", {})
    magnitude = inner.get("Magnitude", 0)
    unit = inner.get("Unit", "")

    # Magnitude may be a dict (ActionOutput reference, Variable, etc.)
    if isinstance(magnitude, dict):
        ref_str = _format_ref_type(magnitude, output_uuids or {}, var_tracker or {})
        if ref_str is not None:
            mag_str = ref_str
        elif output_uuids:
            mag_str = _format_value(magnitude, output_uuids, var_tracker or {})
        else:
            mag_str = "0"
        return f'QTY({mag_str}, "{unit}")'

    # Magnitude is a string or number
    try:
        mag = float(magnitude) if "." in str(magnitude) else int(magnitude)
    except (ValueError, TypeError):
        mag = magnitude
    return f'QTY({mag}, "{unit}")'


def _format_action_output_ref(att: dict, output_uuids: dict) -> str:
    """Format an ActionOutput reference."""
    uuid = att.get("OutputUUID", "")
    if uuid in output_uuids:
        info = output_uuids[uuid]
        # If it's the immediately preceding action, use @prev
        if info.get("is_prev"):
            return "@prev"
        # Otherwise use a named handle
        name = info.get("name", "result")
        return f"@{_sanitize_ident(name)}"
    return "@prev"  # Default fallback


def _format_variable_ref(att: dict, _var_tracker: dict | None = None) -> str:
    """Format a Variable reference."""
    # Check both locations for variable name
    var_name = att.get("VariableName", "")
    if not var_name:
        inner_var = att.get("Variable", {})
        if isinstance(inner_var, dict):
            inner_val = inner_var.get("Value", {})
            var_name = inner_val.get("VariableName", "")

    if var_name:
        clean = _sanitize_ident(var_name)
        return f"${clean}"
    return "@prev"


def _format_list_value(value: list, output_uuids: dict, var_tracker: dict) -> str:
    """Format a list value.

    Detects WFItems-style dict lists and converts to list literals.
    """
    # Check if it's a WFItems list (dict items with WFKey/WFValue structure)
    if value and isinstance(value[0], dict) and "WFKey" in value[0]:
        return _format_wf_items_as_dict(value, output_uuids, var_tracker)

    # Check if it's a WFItems list (with WFItemType and WFValue)
    if (
        value
        and isinstance(value[0], dict)
        and "WFValue" in value[0]
        and "WFKey" not in value[0]
    ):
        items = []
        for item in value:
            if not isinstance(item, dict):
                # Mixed list: plain string among dict items
                items.append(_format_value(item, output_uuids, var_tracker))
                continue
            wf_val = item.get("WFValue", {})
            if isinstance(wf_val, dict) and "Value" in wf_val:
                items.append(
                    _format_value(
                        wf_val.get("Value", {}).get("string", ""),
                        output_uuids,
                        var_tracker,
                    )
                )
            else:
                items.append(_format_value(wf_val, output_uuids, var_tracker))
        return "[" + ", ".join(items) + "]"

    # Plain list
    items = [_format_value(item, output_uuids, var_tracker) for item in value]
    return "[" + ", ".join(items) + "]"


def _format_wf_items_as_dict(items: list, output_uuids: dict, var_tracker: dict) -> str:
    """Format WFItems (dict entries) as a dict literal."""
    entries = []
    for item in items:
        key_val = item.get("WFKey", {})
        val_val = item.get("WFValue", {})

        # Extract key string
        if isinstance(key_val, dict):
            key_inner = key_val.get("Value", {})
            key_str = (
                key_inner.get("string", "")
                if isinstance(key_inner, dict)
                else str(key_inner)
            )
        else:
            key_str = str(key_val)

        # Format value
        formatted_val = _format_value(val_val, output_uuids, var_tracker)
        entries.append(f'"{_escape_string(key_str)}": {formatted_val}')

    return "{" + ", ".join(entries) + "}"


def _sanitize_ident(name: str) -> str:
    """Convert a variable/handle name to a valid DSL identifier."""
    # Replace spaces with underscores, remove non-alphanumeric
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    # Ensure starts with letter/underscore
    if clean and clean[0].isdigit():
        clean = "_" + clean
    return clean or "unnamed"


# ============================================================
# Main Reverse Compiler
# ============================================================


def plist_to_dsl(plist_data: dict) -> str:
    """Convert a plist dict to DSL text.

    Args:
        plist_data: The parsed plist dict from a .shortcut file.

    Returns:
        DSL text string.
    """
    actions = plist_data.get("WFWorkflowActions", [])
    name = plist_data.get("WFWorkflowName", "Untitled")

    # If name not in plist, try to get it from the icon
    if name == "Untitled":
        # Some plists store it elsewhere
        for key in ("name", "Name"):
            if key in plist_data:
                name = plist_data[key]
                break

    # Pre-process: build OutputUUID → action info map
    output_uuids = _build_output_uuid_map(actions)
    var_tracker = {}

    # Parse into a tree of blocks
    lines = [f'SHORTCUT "{_escape_string(name)}"']
    _emit_actions(
        actions, 0, len(actions), lines, output_uuids, var_tracker, prev_uuid=None
    )
    lines.append("ENDSHORTCUT")

    return "\n".join(lines) + "\n"


def _build_output_uuid_map(actions: list) -> dict[str, dict]:
    """Build a map of OutputUUID → {name, index, is_prev} for @prev resolution."""
    result = {}
    for i, action in enumerate(actions):
        params = action.get("WFWorkflowActionParameters", {})
        uuid = params.get("UUID", "")
        custom_name = params.get("CustomOutputName", "")
        ident = action.get("WFWorkflowActionIdentifier", "")
        short = _short_name(ident)

        if uuid:
            result[uuid] = {
                "name": custom_name or short,
                "index": i,
                "identifier": ident,
                "is_prev": False,  # Set dynamically during emission
            }

    return result


def _emit_actions(
    actions: list,
    start: int,
    end: int,
    lines: list[str],
    output_uuids: dict,
    var_tracker: dict,
    prev_uuid: str | None,
) -> None:
    """Emit DSL lines for a range of actions, handling control flow nesting."""
    i = start
    while i < end:
        action = actions[i]
        ident = action.get("WFWorkflowActionIdentifier", "")
        params = action.get("WFWorkflowActionParameters", {})
        mode = params.get("WFControlFlowMode", -1)
        group_id = params.get("GroupingIdentifier", "")

        # Update @prev tracking
        if prev_uuid:
            for uid, info in output_uuids.items():
                info["is_prev"] = uid == prev_uuid

        if ident == CF_CONDITIONAL and mode == 0:
            # IF block — find matching ENDIF
            i = _emit_if_block(
                actions,
                i,
                end,
                lines,
                output_uuids,
                var_tracker,
                group_id,
                params,
                prev_uuid,
            )
        elif ident == CF_MENU and mode == 0:
            # MENU block
            i = _emit_menu_block(
                actions,
                i,
                end,
                lines,
                output_uuids,
                var_tracker,
                group_id,
                params,
                prev_uuid,
            )
        elif ident == CF_REPEAT_COUNT and mode == 0:
            # REPEAT block
            i = _emit_repeat_block(
                actions,
                i,
                end,
                lines,
                output_uuids,
                var_tracker,
                group_id,
                params,
                prev_uuid,
            )
        elif ident == CF_REPEAT_EACH and mode == 0:
            # FOREACH block
            i = _emit_foreach_block(
                actions,
                i,
                end,
                lines,
                output_uuids,
                var_tracker,
                group_id,
                params,
                prev_uuid,
            )
        elif ident in CONTROL_FLOW_IDS and mode in (1, 2):
            # Skip — these are handled by their parent block
            i += 1
        else:
            # Regular action
            prev_uuid = _emit_regular_action(
                action, ident, params, lines, output_uuids, var_tracker
            )
            i += 1


def _find_group_end(
    actions: list, start: int, end: int, group_id: str, _ident: str = ""
) -> int:
    """Find the index of the end marker (mode=2) for a given GroupingIdentifier."""
    depth = 0
    for i in range(start, end):
        action = actions[i]
        params = action.get("WFWorkflowActionParameters", {})
        a_group = params.get("GroupingIdentifier", "")
        a_mode = params.get("WFControlFlowMode", -1)

        if a_group == group_id:
            if a_mode == 0:
                depth += 1
            elif a_mode == 2:
                depth -= 1
                if depth == 0:
                    return i
    return end  # Not found — return end


def _find_else_marker(
    actions: list, start: int, end_idx: int, group_id: str
) -> int | None:
    """Find the ELSE marker (mode=1) for an IF block within the given range."""
    for i in range(start, end_idx):
        action = actions[i]
        params = action.get("WFWorkflowActionParameters", {})
        if (
            params.get("GroupingIdentifier") == group_id
            and params.get("WFControlFlowMode") == 1
            and action.get("WFWorkflowActionIdentifier") == CF_CONDITIONAL
        ):
            return i
    return None


def _emit_if_block(
    actions, start, end, lines, output_uuids, var_tracker, group_id, params, prev_uuid
) -> int:
    """Emit an IF/ELSE/ENDIF block. Returns the index after the ENDIF."""
    # Determine condition
    cond_int = params.get("WFCondition", 100)
    condition = _condition_name(cond_int)

    # Determine input reference
    wf_input = params.get("WFInput", {})
    input_ref = _format_conditional_input(wf_input, output_uuids, var_tracker)

    # Determine compare value
    compare_str = ""
    for key in ("WFNumberValue", "WFConditionalActionString", "WFDate"):
        if key in params:
            val = params[key]
            compare_str = " " + _format_value(val, output_uuids, var_tracker)
            break

    # Also check for WFEnumeration / WFLinkEnumeration (used in some conditions)
    if not compare_str:
        for enum_key in ("WFEnumeration", "WFLinkEnumeration"):
            if enum_key in params:
                val = params[enum_key]
                if isinstance(val, str):
                    compare_str = f' "{_escape_string(val)}"'
                else:
                    compare_str = " " + _format_value(val, output_uuids, var_tracker)
                break

    # Find block boundaries
    end_idx = _find_group_end(actions, start, end, group_id, CF_CONDITIONAL)
    else_idx = _find_else_marker(actions, start + 1, end_idx, group_id)

    lines.append(f"IF {input_ref} {condition}{compare_str}")

    if else_idx is not None:
        # Then body
        _emit_actions(
            actions, start + 1, else_idx, lines, output_uuids, var_tracker, prev_uuid
        )
        lines.append("ELSE")
        # Else body
        _emit_actions(
            actions, else_idx + 1, end_idx, lines, output_uuids, var_tracker, prev_uuid
        )
    else:
        # No else
        _emit_actions(
            actions, start + 1, end_idx, lines, output_uuids, var_tracker, prev_uuid
        )

    lines.append("ENDIF")
    return end_idx + 1


def _emit_menu_block(
    actions, start, end, lines, output_uuids, var_tracker, group_id, params, prev_uuid
) -> int:
    """Emit a MENU/CASE/ENDMENU block."""
    raw_prompt = params.get("WFMenuPrompt", "Choose")
    if isinstance(raw_prompt, dict):
        # Token string prompt — extract plain text
        prompt_str = _format_value(raw_prompt, output_uuids, var_tracker)
    else:
        prompt_str = f'"{_escape_string(raw_prompt)}"'
    end_idx = _find_group_end(actions, start, end, group_id, CF_MENU)

    lines.append(f"MENU {prompt_str}")

    # Find all CASE markers (mode=1)
    case_indices = []
    for i in range(start + 1, end_idx):
        action = actions[i]
        a_params = action.get("WFWorkflowActionParameters", {})
        if (
            a_params.get("GroupingIdentifier") == group_id
            and a_params.get("WFControlFlowMode") == 1
            and action.get("WFWorkflowActionIdentifier") == CF_MENU
        ):
            title = a_params.get("WFMenuItemTitle", "")
            if not title:
                # Try attributed title
                attr = a_params.get("WFMenuItemAttributedTitle", {})
                if isinstance(attr, dict):
                    title = attr.get("string", f"Case {len(case_indices) + 1}")
                else:
                    title = f"Case {len(case_indices) + 1}"
            case_indices.append((i, title))

    # Emit each case
    for ci, (case_start, title) in enumerate(case_indices):
        case_end = case_indices[ci + 1][0] if ci + 1 < len(case_indices) else end_idx
        lines.append(f'CASE "{_escape_string(title)}"')
        _emit_actions(
            actions,
            case_start + 1,
            case_end,
            lines,
            output_uuids,
            var_tracker,
            prev_uuid,
        )

    lines.append("ENDMENU")
    return end_idx + 1


def _emit_repeat_block(
    actions, start, end, lines, output_uuids, var_tracker, group_id, params, prev_uuid
) -> int:
    """Emit a REPEAT/ENDREPEAT block."""
    count = params.get("WFRepeatCount", 1)
    count_str = _format_value(count, output_uuids, var_tracker)

    end_idx = _find_group_end(actions, start, end, group_id, CF_REPEAT_COUNT)

    lines.append(f"REPEAT {count_str}")
    _emit_actions(
        actions, start + 1, end_idx, lines, output_uuids, var_tracker, prev_uuid
    )
    lines.append("ENDREPEAT")
    return end_idx + 1


def _emit_foreach_block(
    actions, start, end, lines, output_uuids, var_tracker, group_id, params, prev_uuid
) -> int:
    """Emit a FOREACH/ENDFOREACH block."""
    wf_input = params.get("WFInput", {})
    input_ref = _format_value(wf_input, output_uuids, var_tracker)

    end_idx = _find_group_end(actions, start, end, group_id, CF_REPEAT_EACH)

    lines.append(f"FOREACH {input_ref}")
    _emit_actions(
        actions, start + 1, end_idx, lines, output_uuids, var_tracker, prev_uuid
    )
    lines.append("ENDFOREACH")
    return end_idx + 1


def _emit_regular_action(
    _action: dict,
    ident: str,
    params: dict,
    lines: list,
    output_uuids: dict,
    var_tracker: dict,
) -> str | None:
    """Emit a regular (non-control-flow) action. Returns its UUID for @prev tracking."""
    short = _short_name(ident)
    uuid = params.get("UUID", "")

    # Special case: setvariable → SET $var = value
    if ident == "is.workflow.actions.setvariable":
        var_name = params.get("WFVariableName", "")
        wf_input = params.get("WFInput", {})
        if var_name:
            clean_name = _sanitize_ident(var_name)
            val_str = _format_value(wf_input, output_uuids, var_tracker)
            lines.append(f"SET ${clean_name} = {val_str}")
            var_tracker[var_name] = clean_name
            return uuid
        # Fall through to regular action if no var name

    # Build param string
    param_parts = []
    for key, value in params.items():
        if key in SKIP_PARAMS:
            continue
        if any(key.startswith(p) for p in SKIP_PARAM_PREFIXES):
            continue
        if key == "CustomOutputName":
            continue  # Not a user-facing param in DSL

        val_str = _format_value(value, output_uuids, var_tracker)
        param_parts.append(f"{key}={val_str}")

    param_str = " ".join(param_parts)
    if param_str:
        lines.append(f"ACTION {short} {param_str}")
    else:
        lines.append(f"ACTION {short}")

    return uuid


def _format_conditional_input(
    wf_input: dict, output_uuids: dict, var_tracker: dict
) -> str:
    """Format the WFInput of a conditional (has extra Variable wrapping)."""
    # WFInput for conditionals: {Type: Variable, Variable: {Value: {...}, WFSerializationType: ...}}
    if isinstance(wf_input, dict):
        ref_type = wf_input.get("Type", "")
        if ref_type == "Variable":
            inner = wf_input.get("Variable", {})
            if isinstance(inner, dict):
                return _format_value(inner, output_uuids, var_tracker)

        # Direct ActionOutput ref (some shortcuts)
        if ref_type == "ActionOutput":
            return _format_action_output_ref(wf_input, output_uuids)

        # Token string wrapping
        ser_type = wf_input.get("WFSerializationType", "")
        if ser_type:
            return _format_value(wf_input, output_uuids, var_tracker)

    return "@prev"


def _condition_name(cond_int: int) -> str:
    """Convert condition integer to DSL condition name."""
    REVERSE_CONDITIONS = {
        0: "equals_number",
        2: "is_greater_than",
        3: "is_less_than",
        4: "equals_string",
        5: "not_equal_string",
        99: "contains",
        999: "does_not_contain",
        100: "has_any_value",
        101: "does_not_have_any_value",
        1002: "is_before",
    }
    return REVERSE_CONDITIONS.get(cond_int, f"condition_{cond_int}")


# ============================================================
# File-level API
# ============================================================


def shortcut_file_to_dsl(filepath: str) -> str:
    """Convert a .shortcut file to DSL text.

    Args:
        filepath: Path to a .shortcut file.

    Returns:
        DSL text string.
    """
    with open(filepath, "rb") as f:
        plist_data = plistlib.load(f)
    return plist_to_dsl(plist_data)


def shortcut_file_to_dsl_safe(filepath: str) -> tuple[str | None, str | None]:
    """Convert a .shortcut file to DSL text, returning (dsl, error).

    Returns:
        (dsl_text, None) on success, (None, error_message) on failure.
    """
    try:
        dsl = shortcut_file_to_dsl(filepath)
        return dsl, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plist_to_dsl.py <shortcut_file> [--validate]")
        sys.exit(1)

    filepath = sys.argv[1]
    validate = "--validate" in sys.argv

    dsl = shortcut_file_to_dsl(filepath)
    print(dsl)

    if validate:
        from dsl_parser import parse_dsl
        from dsl_validator import validate_ir

        print("\n--- Validation ---")
        try:
            ir = parse_dsl(dsl)
            print(f"Parsed: {ir}")
            result = validate_ir(ir)
            print(f"Validated: {result}")
        except Exception as e:
            print(f"Error: {e}")
