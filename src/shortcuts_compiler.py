"""
Apple Shortcuts Compiler
========================
Generates valid .shortcut (plist) files from a catalog-driven Python API.
The LLM writes readable Python; this compiler handles all plist mechanics,
UUID wiring, variable references, string interpolation, and control flow.

All actions present in action_catalog.json are available via actions.make().

Usage:
    from shortcuts_compiler import Shortcut, actions

    s = Shortcut("My Shortcut")
    url = s.add(actions.make("url", WFURLActionURL="https://example.com"))
    resp = s.add(actions.make("downloadurl"))
    s.add(actions.make("notification", WFNotificationActionTitle="Done!"))
    s.save("my_shortcut.shortcut")

    # Control flow uses context managers on Shortcut:
    with s.if_block(resp, condition="has_any_value"):
        s.add(actions.make("showresult", Text="Got a response"))

    with s.menu_block("Pick one", ["A", "B"]) as cases:
        cases["A"]()
        s.add(actions.make("comment", WFCommentActionText="Chose A"))
        cases["B"]()
        s.add(actions.make("comment", WFCommentActionText="Chose B"))
"""

import os
import plistlib
import subprocess
import uuid
from contextlib import contextmanager
from typing import Any, Optional, Union

# =============================================================================
# CONSTANTS
# =============================================================================

# Condition codes derived from real shortcut corpus data.
# Apple uses type-dependent semantics:
#   0-3: numeric comparisons (compare_value → WFNumberValue)
#   4-5: string/enum equality (compare_value → WFConditionalActionString)
#   99, 999: string containment (compare_value → WFConditionalActionString)
#   100, 101: existence checks (no compare_value needed)
#   1002: date comparison (compare_value → WFDate)
CONDITION_MAP = {
    # Numeric comparisons — use WFNumberValue
    "equals_number": 0,  # is (number)
    "is_greater_than": 2,  # is greater than
    "is_less_than": 3,  # is less than
    # String comparisons — use WFConditionalActionString
    "equals_string": 4,  # is (string)
    "not_equal_string": 5,  # is not (string)
    "contains": 99,  # contains
    "does_not_contain": 999,  # does not contain
    # Existence checks — no compare_value
    "has_any_value": 100,  # has any value
    "does_not_have_any_value": 101,  # does not have any value
    # Date comparison — use WFDate
    "is_before": 1002,  # is before (date)
    # Aliases
    "==": 0,
    "!=": 5,
    ">": 2,
    "<": 3,
    "has_value": 100,
    "is_empty": 101,
}

# Which comparison parameter to use for each condition code.
# This is a COMPILER RULE derived from real shortcut data — not a guess.
_CONDITION_PARAM = {
    0: "WFNumberValue",  # equals_number
    2: "WFNumberValue",  # is_greater_than
    3: "WFNumberValue",  # is_less_than
    4: "WFConditionalActionString",  # equals_string
    5: "WFConditionalActionString",  # not_equal_string
    99: "WFConditionalActionString",  # contains
    999: "WFConditionalActionString",  # does_not_contain
    100: None,  # has_any_value (no comparison)
    101: None,  # does_not_have_any_value (no comparison)
    1002: "WFDate",  # is_before
}


def _set_compare_value(if_params: dict, cond_int: int, compare_value) -> None:
    """
    Set the comparison value on a conditional action's parameters.
    This is the SINGLE place where compare_value → plist param mapping happens.

    Rules (derived from real shortcut conditionals):
    - The condition code determines WHICH param key to use (_CONDITION_PARAM).
    - The value type determines HOW to encode it:
      - ActionHandle → WFTextTokenAttachment (variable reference)
      - int/float → raw value (for WFNumberValue) or str (for WFConditionalActionString)
      - str → raw string
      - dict → pass through (already wrapped)
      - None → skip (existence checks don't need a comparison value)
    """
    if compare_value is None:
        return

    param_key = _CONDITION_PARAM.get(cond_int)
    if param_key is None:
        # Existence checks (100, 101) — compare_value should not be set.
        # If the caller passes one anyway, ignore it silently.
        return

    # Encode the value based on its Python type
    if isinstance(compare_value, ActionHandle):
        # Dynamic comparison: wrap as WFTextTokenAttachment
        if_params[param_key] = wrap_token_attachment(compare_value.ref())
    elif isinstance(compare_value, dict):
        # Already a wrapped plist structure — pass through
        if_params[param_key] = compare_value
    elif param_key == "WFNumberValue":
        # Numeric params: Apple stores these as int, float, OR string
        # Real data shows str is most common (23/41), int second (13/41)
        if isinstance(compare_value, (int, float)):
            if_params[param_key] = compare_value
        else:
            if_params[param_key] = str(compare_value)
    else:
        # String params (WFConditionalActionString, WFDate, etc.)
        if_params[param_key] = str(compare_value)


# Default envelope values derived from observed real shortcuts.
DEFAULT_CLIENT_VERSION = "2302.0.4"
DEFAULT_MIN_VERSION = 900
DEFAULT_ICON_GLYPH = 59653
DEFAULT_ICON_COLOR = 4274264319

# Friendly name → class name mapping for input_types parameter
INPUT_TYPES = {
    "string": "WFStringContentItem",
    "image": "WFImageContentItem",
    "url": "WFURLContentItem",
    "file": "WFGenericFileContentItem",
    "rich_text": "WFRichTextContentItem",
    "pdf": "WFPDFContentItem",
    "date": "WFDateContentItem",
    "location": "WFLocationContentItem",
    "contact": "WFContactContentItem",
    "email": "WFEmailAddressContentItem",
    "phone": "WFPhoneNumberContentItem",
    "number": "WFNumberContentItem",
    "dictionary": "WFDictionaryContentItem",
    "boolean": "WFBooleanContentItem",
}

# Workflow types
WORKFLOW_TYPES = {
    "widget": "NCWidget",
    "watch": "WatchKit",
    "menu": "Sleep",
    "automation": "AutomationTrigger",
    "quick_action": "ReceivesOnScreenContent",
}


# =============================================================================
# VARIABLE REFERENCE BUILDERS
# =============================================================================


def _gen_uuid() -> str:
    """Generate an uppercase UUID string."""
    return str(uuid.uuid4()).upper()


def ref_action_output(output_uuid: str, output_name: str = "Result") -> dict:
    """Reference another action's output by UUID."""
    return {
        "OutputName": output_name,
        "OutputUUID": output_uuid,
        "Type": "ActionOutput",
    }


def ref_variable(name: str) -> dict:
    """Reference a named variable (from Set Variable)."""
    return {
        "Type": "Variable",
        "VariableName": name,
    }


def ref_extension_input() -> dict:
    """Reference the shortcut's input (what was passed to it)."""
    return {"Type": "ExtensionInput"}


def ref_current_date() -> dict:
    """Reference the current date/time."""
    return {"Type": "CurrentDate"}


# =============================================================================
# WRAPPING HELPERS
# =============================================================================


def wrap_token_attachment(reference: dict) -> dict:
    """
    Wrap a reference as WFTextTokenAttachment (single-value parameter).
    Used for: Set Variable WFInput, Get Dictionary Value WFInput, etc.
    """
    return {
        "Value": reference,
        "WFSerializationType": "WFTextTokenAttachment",
    }


def wrap_token_string(text_before: str, attachment: dict, text_after: str = "") -> dict:
    """
    Wrap a reference inside a string with variable interpolation.
    Automatically computes the attachment position.
    """
    full_string = text_before + "\ufffc" + text_after
    position = len(text_before)
    return {
        "Value": {
            "string": full_string,
            "attachmentsByRange": {
                f"{{{position}, 1}}": attachment,
            },
        },
        "WFSerializationType": "WFTextTokenString",
    }


def wrap_token_string_multi(template: str, attachments: list[dict]) -> dict:
    """
    Wrap multiple variable references in a single interpolated string.
    Use \\ufffc as placeholder in template for each attachment (in order).

    Example:
        wrap_token_string_multi(
            "Hello \\ufffc, your score is \\ufffc",
            [ref_variable("Name"), ref_action_output(some_uuid)]
        )
    """
    positions = []
    idx = 0
    for char_idx, ch in enumerate(template):
        if ch == "\ufffc":
            positions.append(char_idx)
            idx += 1

    if len(positions) != len(attachments):
        raise ValueError(
            f"Template has {len(positions)} placeholders but {len(attachments)} attachments provided"
        )

    attachments_by_range = {}
    for pos, attachment in zip(positions, attachments):
        attachments_by_range[f"{{{pos}, 1}}"] = attachment

    return {
        "Value": {
            "string": template,
            "attachmentsByRange": attachments_by_range,
        },
        "WFSerializationType": "WFTextTokenString",
    }


def wrap_conditional_input(reference: dict) -> dict:
    """
    Wrap a reference for use in a conditional's WFInput.
    Conditionals need an EXTRA wrapping layer compared to most actions.
    This is the #1 source of silent failures — the compiler handles it
    so the LLM never has to think about it.
    """
    return {
        "Type": "Variable",
        "Variable": wrap_token_attachment(reference),
    }


def wrap_quantity_field(magnitude: Union[dict, float, int], unit: str) -> dict:
    """Wrap a quantity value for HealthKit logging."""
    mag = magnitude if isinstance(magnitude, dict) else magnitude
    return {
        "Value": {
            "Magnitude": mag,
            "Unit": unit,
        },
        "WFSerializationType": "WFQuantityFieldValue",
    }


# =============================================================================
# ACTION HANDLE — returned by Shortcut.add() for wiring
# =============================================================================


class ActionHandle:
    """
    Returned when an action is added to a shortcut.
    Use this to wire outputs to downstream actions.
    """

    def __init__(self, uuid: str, output_name: str, action_identifier: str):
        self.uuid = uuid
        self.output_name = output_name
        self.action_identifier = action_identifier

    def ref(self, output_name: Optional[str] = None) -> dict:
        """Get a reference dict pointing to this action's output."""
        return ref_action_output(self.uuid, output_name or self.output_name)

    def as_input(self) -> dict:
        """Get this action's output wrapped as a WFTextTokenAttachment (for most params)."""
        return wrap_token_attachment(self.ref())

    def as_conditional_input(self) -> dict:
        """Get this action's output wrapped for conditional WFInput (extra layer)."""
        return wrap_conditional_input(self.ref())

    def in_string(self, text_before: str, text_after: str = "") -> dict:
        """Embed this action's output in an interpolated string."""
        return wrap_token_string(text_before, self.ref(), text_after)


# =============================================================================
# SHORTCUT CLASS — the main entry point
# =============================================================================


class Shortcut:
    """
    Build an Apple Shortcut programmatically.

    Usage:
        s = Shortcut("My Shortcut")
        s.add(actions.make("comment", WFCommentActionText="This is my shortcut"))
        url = s.add(actions.make("url", WFURLActionURL="https://example.com"))
        s.save("output.shortcut")
    """

    def __init__(
        self,
        name: str = "Generated Shortcut",
        input_types: Optional[list[str]] = None,
        icon_glyph: int = DEFAULT_ICON_GLYPH,
        icon_color: int = DEFAULT_ICON_COLOR,
        workflow_types: Optional[list[str]] = None,
    ):
        self.name = name
        self.actions: list[dict] = []
        # Apple Shortcuts uses a FLAT action list — control flow nesting is expressed
        # via GroupingIdentifier pairing, not structural nesting in the plist. This stack
        # always has exactly one element (self.actions). All control flow context managers
        # append to self._action_stack[-1] which is always self.actions. The stack is
        # retained as infrastructure for potential future block-level tracking.
        self._action_stack: list[list[dict]] = [self.actions]

        # Input types
        input_classes = []
        if input_types:
            for t in input_types:
                if t in INPUT_TYPES:
                    input_classes.append(INPUT_TYPES[t])
                else:
                    input_classes.append(t)

        # Workflow types
        wf_types = ["NCWidget", "WatchKit"]
        if workflow_types:
            wf_types = []
            for t in workflow_types:
                if t in WORKFLOW_TYPES:
                    wf_types.append(WORKFLOW_TYPES[t])
                else:
                    wf_types.append(t)

        self.envelope = {
            "WFWorkflowActions": self.actions,
            "WFWorkflowClientVersion": DEFAULT_CLIENT_VERSION,
            "WFWorkflowIcon": {
                "WFWorkflowIconGlyphNumber": icon_glyph,
                "WFWorkflowIconStartColor": icon_color,
            },
            "WFWorkflowImportQuestions": [],
            "WFWorkflowInputContentItemClasses": input_classes,
            "WFWorkflowMinimumClientVersion": DEFAULT_MIN_VERSION,
            "WFWorkflowMinimumClientVersionString": str(DEFAULT_MIN_VERSION),
            "WFWorkflowTypes": wf_types,
        }

    def add(self, action: dict) -> ActionHandle:
        """
        Add an action to the shortcut. Returns an ActionHandle for wiring.
        """
        # Ensure UUID exists
        params = action.setdefault("WFWorkflowActionParameters", {})
        action_uuid = params.get("UUID")
        if not action_uuid:
            action_uuid = _gen_uuid()
            params["UUID"] = action_uuid

        # Get output name
        output_name = params.get("CustomOutputName", "Result")
        identifier = action.get("WFWorkflowActionIdentifier", "")

        # Infer output name from action type using the catalog
        if output_name == "Result":
            output_name = _get_output_name(identifier)

        # Add to current scope (top of stack for nesting)
        self._action_stack[-1].append(action)

        return ActionHandle(action_uuid, output_name, identifier)

    @contextmanager
    def if_block(
        self,
        input_ref: Union[ActionHandle, dict, str],
        condition: str = "has_any_value",
        compare_value: Optional[Any] = None,
    ):
        """
        Context manager for If/EndIf blocks.

        Usage:
            with s.if_block(some_handle, condition="has_any_value"):
                s.add(actions.make("showresult", Text="It had a value!"))

        The compiler auto-handles GroupingIdentifier pairing and
        the conditional WFInput wrapping (the extra Variable layer).
        """
        group_id = _gen_uuid()
        cond_int = CONDITION_MAP.get(condition)
        if cond_int is None:
            raise ValueError(
                f"Unknown condition '{condition}'. Valid: {list(CONDITION_MAP.keys())}"
            )

        # Build WFInput with correct conditional wrapping
        if isinstance(input_ref, ActionHandle):
            wf_input = wrap_conditional_input(input_ref.ref())
        elif isinstance(input_ref, str):
            wf_input = wrap_conditional_input(ref_variable(input_ref))
        elif isinstance(input_ref, dict):
            wf_input = wrap_conditional_input(input_ref)
        else:
            raise TypeError(
                f"input_ref must be ActionHandle, dict, or str, got {type(input_ref)}"
            )

        # If-start action
        if_params = {
            "GroupingIdentifier": group_id,
            "WFControlFlowMode": 0,
            "WFCondition": cond_int,
            "WFInput": wf_input,
        }
        _set_compare_value(if_params, cond_int, compare_value)

        if_action = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
            "WFWorkflowActionParameters": if_params,
        }
        self._action_stack[-1].append(if_action)

        # Yield for body actions (they go into the same list, between if-start and if-end)
        yield

        # End-If action
        end_action = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": group_id,
                "WFControlFlowMode": 2,
            },
        }
        self._action_stack[-1].append(end_action)

    @contextmanager
    def if_else_block(
        self,
        input_ref: Union[ActionHandle, dict, str],
        condition: str = "has_any_value",
        compare_value: Optional[Any] = None,
    ):
        """
        Context manager for If/Otherwise/EndIf blocks.
        Yields a callable `otherwise()` that you call to mark the else branch.

        Usage:
            with s.if_else_block(handle, condition="has_any_value") as otherwise:
                s.add(actions.make("comment", WFCommentActionText="If branch"))
                otherwise()
                s.add(actions.make("comment", WFCommentActionText="Else branch"))
        """
        group_id = _gen_uuid()
        cond_int = CONDITION_MAP.get(condition)
        if cond_int is None:
            raise ValueError(f"Unknown condition '{condition}'.")

        if isinstance(input_ref, ActionHandle):
            wf_input = wrap_conditional_input(input_ref.ref())
        elif isinstance(input_ref, str):
            wf_input = wrap_conditional_input(ref_variable(input_ref))
        elif isinstance(input_ref, dict):
            wf_input = wrap_conditional_input(input_ref)
        else:
            raise TypeError("input_ref must be ActionHandle, dict, or str")

        if_params = {
            "GroupingIdentifier": group_id,
            "WFControlFlowMode": 0,
            "WFCondition": cond_int,
            "WFInput": wf_input,
        }
        _set_compare_value(if_params, cond_int, compare_value)

        if_action = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
            "WFWorkflowActionParameters": if_params,
        }
        self._action_stack[-1].append(if_action)

        def otherwise():
            otherwise_action = {
                "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
                "WFWorkflowActionParameters": {
                    "GroupingIdentifier": group_id,
                    "WFControlFlowMode": 1,
                },
            }
            self._action_stack[-1].append(otherwise_action)

        yield otherwise

        end_action = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": group_id,
                "WFControlFlowMode": 2,
            },
        }
        self._action_stack[-1].append(end_action)

    @contextmanager
    def menu_block(self, prompt: Union[str, dict], options: list[str]):
        """
        Context manager for Choose from Menu blocks.
        Yields a dict of callables — call each case before adding its actions.

        Usage:
            with s.menu_block("Pick one:", ["A", "B", "C"]) as cases:
                cases["A"]()
                s.add(actions.make("comment", WFCommentActionText="Chose A"))
                cases["B"]()
                s.add(actions.make("comment", WFCommentActionText="Chose B"))
                cases["C"]()
                s.add(actions.make("comment", WFCommentActionText="Chose C"))

        The compiler handles GroupingIdentifier pairing, WFControlFlowMode,
        and WFMenuItems structure automatically.
        """
        group_id = _gen_uuid()

        # Menu start action (WFControlFlowMode: 0)
        menu_start = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.choosefrommenu",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": group_id,
                "WFControlFlowMode": 0,
                "WFMenuPrompt": prompt,
                "WFMenuItems": options,
            },
        }
        self._action_stack[-1].append(menu_start)

        # Build case callables
        case_dict = {}
        for option in options:

            def _make_case(opt_title):
                def _case():
                    case_action = {
                        "WFWorkflowActionIdentifier": "is.workflow.actions.choosefrommenu",
                        "WFWorkflowActionParameters": {
                            "GroupingIdentifier": group_id,
                            "WFControlFlowMode": 1,
                            "WFMenuItemTitle": opt_title,
                        },
                    }
                    self._action_stack[-1].append(case_action)

                return _case

            case_dict[option] = _make_case(option)

        yield case_dict

        # Menu end action (WFControlFlowMode: 2)
        menu_end = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.choosefrommenu",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": group_id,
                "WFControlFlowMode": 2,
            },
        }
        self._action_stack[-1].append(menu_end)

    @contextmanager
    def repeat_block(self, count: Union[int, ActionHandle, dict]):
        """
        Context manager for Repeat N times blocks.

        Usage:
            with s.repeat_block(5):
                s.add(actions.make("comment", WFCommentActionText="Loop iteration"))
        """
        group_id = _gen_uuid()

        repeat_params: dict[str, Any] = {
            "GroupingIdentifier": group_id,
            "WFControlFlowMode": 0,
        }

        if isinstance(count, int):
            repeat_params["WFRepeatCount"] = count
        elif isinstance(count, ActionHandle):
            repeat_params["WFRepeatCount"] = wrap_token_attachment(count.ref())
        elif isinstance(count, dict):
            repeat_params["WFRepeatCount"] = wrap_token_attachment(count)

        repeat_start = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.repeat.count",
            "WFWorkflowActionParameters": repeat_params,
        }
        self._action_stack[-1].append(repeat_start)

        yield

        repeat_end = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.repeat.count",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": group_id,
                "WFControlFlowMode": 2,
            },
        }
        self._action_stack[-1].append(repeat_end)

    @contextmanager
    def repeat_each_block(self, input_ref: Union[ActionHandle, dict, str]):
        """
        Context manager for Repeat with Each (for-each loop).

        Usage:
            items = s.add(actions.make("list", WFItems=actions.build_list(["a", "b", "c"])))
            with s.repeat_each_block(items):
                s.add(actions.make("comment", WFCommentActionText="Processing item"))
        """
        group_id = _gen_uuid()

        if isinstance(input_ref, ActionHandle):
            wf_input = wrap_token_attachment(input_ref.ref())
        elif isinstance(input_ref, str):
            wf_input = wrap_token_attachment(ref_variable(input_ref))
        elif isinstance(input_ref, dict):
            wf_input = wrap_token_attachment(input_ref)
        else:
            raise TypeError("input_ref must be ActionHandle, dict, or str")

        repeat_start = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.repeat.each",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": group_id,
                "WFControlFlowMode": 0,
                "WFInput": wf_input,
            },
        }
        self._action_stack[-1].append(repeat_start)

        yield

        repeat_end = {
            "WFWorkflowActionIdentifier": "is.workflow.actions.repeat.each",
            "WFWorkflowActionParameters": {
                "GroupingIdentifier": group_id,
                "WFControlFlowMode": 2,
            },
        }
        self._action_stack[-1].append(repeat_end)

    def build(self) -> dict:
        """Build and return the complete plist dict."""
        return self.envelope

    def save(self, filepath: str) -> str:
        """
        Save as unsigned .shortcut file. Returns the absolute filepath.
        Written as binary plist (standard Apple format).
        """
        data = self.build()
        filepath = os.path.abspath(filepath)
        with open(filepath, "wb") as f:
            plistlib.dump(data, f, fmt=plistlib.FMT_BINARY)
        return filepath

    def sign(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        mode: str = "anyone",
    ) -> str:
        """
        Sign the shortcut using macOS `shortcuts sign` CLI.
        Returns the output path.

        mode: "anyone" or "people-who-know-me"

        REQUIRES: macOS 12+ with the `shortcuts` CLI tool available.
        This will NOT work on Linux/Windows — use save() + manual signing instead.
        """
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_signed{ext}"

        cmd = ["shortcuts", "sign", "-i", input_path, "-o", output_path, "-m", mode]
        result = subprocess.run(cmd, capture_output=True, text=True)

        # The sign command often emits ObjC runtime warnings that are harmless.
        # Check if the output file was actually created.
        if os.path.exists(output_path):
            return output_path
        else:
            raise RuntimeError(
                f"Signing failed. stderr: {result.stderr}\nstdout: {result.stdout}"
            )

    def save_and_sign(self, filepath: str, mode: str = "anyone") -> str:
        """Save and sign in one step. Returns the signed file path."""
        unsigned = self.save(filepath)
        return self.sign(unsigned, mode=mode)

    def deliver(
        self,
        name: Optional[str] = None,
        output_dir: str = ".",
        sign: bool = True,
        mode: str = "anyone",
        auto_import: bool = False,
        open_file: bool = False,
    ) -> dict:
        """
        All-in-one delivery method. Saves, signs, and optionally imports.

        Primary use case: LLM running on macOS via Claude Code / desktop app.
        The compiler detects the platform and takes the best available path.

        Args:
            name: Filename (defaults to shortcut name). Extension auto-added.
            output_dir: Where to save the file.
            sign: Attempt to sign with macOS `shortcuts sign` CLI.
            mode: "anyone" or "people-who-know-me" (signing mode).
            auto_import: If True AND on macOS, silently import into Shortcuts app
                         via `shortcuts import`. The shortcut just appears in the app.
            open_file: If True AND on macOS, open the file with `open` command,
                       which triggers the native Shortcuts import dialog.
                       Ignored if auto_import is True.

        Returns:
            {
                "unsigned": "/path/to/file.shortcut",
                "signed": "/path/to/file_signed.shortcut" or None,
                "imported": True/False,
                "method": "imported" | "signed" | "unsigned_macos" | "unsigned_manual",
                "instructions": "Human-readable next steps"
            }
        """
        filename = name or self.name.replace(" ", "_")
        if not filename.endswith(".shortcut"):
            filename += ".shortcut"
        unsigned_path = os.path.join(output_dir, filename)

        # Validate first
        warnings = self.validate()
        if any("will crash" in w.lower() for w in warnings):
            raise ValueError(
                "Shortcut has structural errors that will crash on import:\n"
                + "\n".join(f"  - {w}" for w in warnings)
            )

        # Save unsigned
        self.save(unsigned_path)
        abs_unsigned = os.path.abspath(unsigned_path)
        on_macos = _is_macos()

        result = {
            "unsigned": abs_unsigned,
            "signed": None,
            "imported": False,
            "method": None,
            "instructions": None,
        }

        # Step 1: Try signing (macOS only, harmless no-op elsewhere)
        signed_path = None
        if sign and on_macos and _has_shortcuts_cli():
            try:
                signed_path = self.sign(unsigned_path, mode=mode)
                result["signed"] = os.path.abspath(signed_path)
            except RuntimeError:
                pass  # Signing failed, fall through to unsigned paths

        # Step 2: Try auto-import (macOS only)
        best_file = signed_path or abs_unsigned
        if auto_import and on_macos and _has_shortcuts_cli():
            try:
                success = import_shortcut(best_file)
                if success:
                    result["imported"] = True
                    result["method"] = "imported"
                    shortcut_display = self.name or filename.replace(".shortcut", "")
                    result["instructions"] = (
                        f'"{shortcut_display}" has been imported into the Shortcuts app.\n'
                        f"Open the Shortcuts app to find and run it."
                    )
                    # Clean up the unsigned file if we also have a signed one
                    return result
            except (FileNotFoundError, RuntimeError):
                pass

        # Step 3: Try open (macOS native import dialog) — only with signed file
        if open_file and not result["imported"] and on_macos and signed_path:
            try:
                subprocess.run(["open", signed_path], capture_output=True, timeout=5)
                result["method"] = "signed"
                result["instructions"] = (
                    f"Opened {os.path.basename(signed_path)} — the Shortcuts import dialog should appear.\n"
                    f"Click 'Add Shortcut' to import it."
                )
                return result
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Step 4: Return file with instructions
        if signed_path:
            result["method"] = "signed"
            result["instructions"] = (
                f"Signed shortcut: {signed_path}\n"
                f"Double-click to import on this Mac, or AirDrop to iOS.\n"
                f"Prerequisite: Settings → Shortcuts → Private Sharing → ON"
            )
        elif on_macos:
            # Signing failed but we're on macOS — give signing instructions
            result["method"] = "unsigned_needs_signing"
            signed_name = abs_unsigned.replace(".shortcut", "_signed.shortcut")
            result["instructions"] = (
                f"Shortcut saved (unsigned): {abs_unsigned}\n"
                f"⚠️  Signing is REQUIRED for import. Unsigned files cannot be imported.\n"
                f"Sign it with:\n"
                f'  shortcuts sign -m anyone -i "{abs_unsigned}" -o "{signed_name}"\n'
                f"Then double-click the signed file or run:\n"
                f'  shortcuts import "{signed_name}"'
            )
        else:
            result["method"] = "unsigned_manual"
            result["instructions"] = _unsigned_import_instructions(abs_unsigned)

        return result

    def validate(self) -> list[str]:
        """
        Validate the shortcut structure for common issues before saving.
        Returns a list of warnings (empty = clean).

        Checks:
        1. Unclosed/orphan control flow blocks (GroupingIdentifier pairing)
        2. Mode 0 must precede Mode 2 for same GroupingIdentifier
        3. Mode 1 markers must be between their group's Mode 0 and Mode 2
        4. Menu case count must match WFMenuItems count
        5. No interleaved (non-nested) control flow blocks
        6. UUID collisions
        7. Empty menus
        """
        warnings = []
        actions_list = self.envelope.get("WFWorkflowActions", [])

        if not actions_list:
            warnings.append("Shortcut has no actions.")
            return warnings

        # === Pass 1: Collect all control flow markers ===
        group_starts: dict[str, int] = {}
        group_ends: dict[str, int] = {}
        group_middles: dict[str, list[int]] = {}

        for i, action in enumerate(actions_list):
            params = action.get("WFWorkflowActionParameters", {})
            gid = params.get("GroupingIdentifier")
            cfm = params.get("WFControlFlowMode")
            if gid is not None and cfm is not None:
                if cfm == 0:
                    group_starts[gid] = i
                elif cfm == 1:
                    group_middles.setdefault(gid, []).append(i)
                elif cfm == 2:
                    group_ends[gid] = i

        # === Check 1: Unclosed / orphan blocks ===
        unclosed = set(group_starts.keys()) - set(group_ends.keys())
        if unclosed:
            warnings.append(
                f"Unclosed control flow blocks ({len(unclosed)} groups). "
                f"This will crash on import."
            )

        unopened = set(group_ends.keys()) - set(group_starts.keys())
        if unopened:
            warnings.append(
                f"End blocks without matching start ({len(unopened)} groups)."
            )

        # === Check 2: Mode 0 must precede Mode 2 ===
        for gid in group_starts:
            if gid in group_ends:
                if group_starts[gid] > group_ends[gid]:
                    warnings.append(
                        f"Control flow block has end (index {group_ends[gid]}) "
                        f"before start (index {group_starts[gid]}). This will crash."
                    )

        # === Check 3: Mode 1 markers within their group's range ===
        for gid, mid_indices in group_middles.items():
            if gid not in group_starts:
                warnings.append("Mode 1 marker for group without a start block.")
                continue
            start_idx = group_starts[gid]
            end_idx = group_ends.get(gid)
            if end_idx is None:
                continue  # Already flagged as unclosed
            for mid_idx in mid_indices:
                if mid_idx < start_idx or mid_idx > end_idx:
                    warnings.append(
                        f"Mode 1 marker at index {mid_idx} is outside its "
                        f"block range ({start_idx}-{end_idx})."
                    )

        # === Check 4: Menu case count matches WFMenuItems ===
        for i, action in enumerate(actions_list):
            aid = action.get("WFWorkflowActionIdentifier", "")
            params = action.get("WFWorkflowActionParameters", {})
            if "choosefrommenu" in aid and params.get("WFControlFlowMode") == 0:
                gid = params.get("GroupingIdentifier")
                menu_items = params.get("WFMenuItems", [])
                if not menu_items:
                    warnings.append(f"Action {i}: Menu has no items.")
                elif gid:
                    case_count = len(group_middles.get(gid, []))
                    if case_count != len(menu_items):
                        warnings.append(
                            f"Action {i}: Menu has {len(menu_items)} items but "
                            f"{case_count} case markers."
                        )

        # === Check 5: No interleaved blocks ===
        block_ranges = []
        for gid in group_starts:
            if gid in group_ends:
                block_ranges.append((group_starts[gid], group_ends[gid], gid))
        block_ranges.sort()

        for i_idx, (s1, e1, _g1) in enumerate(block_ranges):
            for s2, e2, _g2 in block_ranges[i_idx + 1 :]:
                if s1 < s2 < e1 < e2:
                    warnings.append(
                        f"Interleaved control flow: blocks at "
                        f"({s1}-{e1}) and ({s2}-{e2}) overlap. "
                        f"Blocks must be properly nested."
                    )

        # === Check 6: UUID collisions ===
        uuids = []
        for action in actions_list:
            uid = action.get("WFWorkflowActionParameters", {}).get("UUID")
            if uid:
                uuids.append(uid)
        if len(uuids) != len(set(uuids)):
            warnings.append(
                "Duplicate UUIDs detected. This will cause variable wiring failures."
            )

        return warnings


def _unsigned_import_instructions(filepath: str) -> str:
    """Generate comprehensive import instructions for an unsigned shortcut."""
    signed_path = filepath.replace(".shortcut", "_signed.shortcut")
    return f"""Unsigned shortcut saved to: {filepath}

⚠️  SIGNING IS REQUIRED — unsigned .shortcut files cannot be imported on macOS or iOS.

=== IMPORT OPTIONS (pick one) ===

OPTION A — Sign on macOS, then import (recommended):
  1. Open Terminal on your Mac
  2. Run: shortcuts sign -m anyone -i "{filepath}" -o "{signed_path}"
  3. Then import: shortcuts import "{signed_path}"
     OR double-click the signed file in Finder
  4. The signed file can also be AirDropped to iOS or shared via iCloud
  Requirements: macOS 12+

OPTION B — One-time signing server setup (for repeated use):
  1. On your Mac, create a script that watches a folder and auto-signs:
     #!/bin/bash
     # auto_sign.sh — watches a folder and signs any new .shortcut files
     WATCH_DIR="$HOME/Desktop/sign_inbox"
     DONE_DIR="$HOME/Desktop/sign_outbox"
     mkdir -p "$WATCH_DIR" "$DONE_DIR"
     fswatch -0 "$WATCH_DIR" | while read -d "" event; do
       if [[ "$event" == *.shortcut ]]; then
         base=$(basename "$event" .shortcut)
         shortcuts sign -i "$event" -o "$DONE_DIR/${{base}}_signed.shortcut" -m anyone
         rm "$event"
         echo "Signed: $base"
       fi
     done
  2. Install fswatch: brew install fswatch
  3. Run: ./auto_sign.sh (leave running in background)
  4. Drop unsigned .shortcut files into ~/Desktop/sign_inbox/
  5. Signed files appear in ~/Desktop/sign_outbox/

=== PREREQUISITES (one-time setup) ===
• macOS: Settings → Shortcuts → Private Sharing → ON
• iOS: Settings → Shortcuts → Private Sharing → ON
• Optional: Settings → Shortcuts → Allow Running Scripts → ON
"""


# =============================================================================
# STANDALONE SIGNING UTILITIES
# =============================================================================


def sign_shortcut(
    input_path: str, output_path: Optional[str] = None, mode: str = "anyone"
) -> str:
    """
    Standalone function to sign any .shortcut file.
    Works outside the Shortcut class — useful for signing pre-existing files.

    Args:
        input_path: Path to unsigned .shortcut file
        output_path: Path for signed output (default: input_signed.shortcut)
        mode: "anyone" or "people-who-know-me"

    Returns: Path to the signed file.
    Raises: RuntimeError if signing fails, FileNotFoundError if shortcuts CLI missing.
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_signed{ext}"

    # Check if shortcuts CLI exists
    if not _has_shortcuts_cli():
        raise FileNotFoundError(
            "The 'shortcuts' CLI is not available. "
            "It requires macOS 12+ (Monterey or later). "
            "See the unsigned import instructions for alternative methods."
        )

    cmd = ["shortcuts", "sign", "-i", input_path, "-o", output_path, "-m", mode]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(output_path):
        return output_path
    else:
        raise RuntimeError(
            f"Signing failed.\nCommand: {' '.join(cmd)}\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )


def batch_sign(
    input_dir: str, output_dir: Optional[str] = None, mode: str = "anyone"
) -> list[dict]:
    """
    Sign all .shortcut files in a directory.

    Returns list of {"input": path, "output": path, "status": "ok"|"error", "error": msg}
    """
    if output_dir is None:
        output_dir = input_dir

    os.makedirs(output_dir, exist_ok=True)
    results = []

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".shortcut"):
            continue
        input_path = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        output_path = os.path.join(output_dir, f"{base}_signed.shortcut")

        try:
            sign_shortcut(input_path, output_path, mode)
            results.append({"input": input_path, "output": output_path, "status": "ok"})
        except Exception as e:
            results.append(
                {
                    "input": input_path,
                    "output": None,
                    "status": "error",
                    "error": str(e),
                }
            )

    return results


def import_shortcut(filepath: str) -> bool:
    """
    Import a shortcut into the Shortcuts app (macOS only).
    The file MUST be signed — unsigned .shortcut files are rejected
    by macOS with "Importing unsigned shortcut files is not supported."
    Use sign() or save_and_sign() before calling this.

    Returns True if import succeeded.
    """
    if not _has_shortcuts_cli():
        raise FileNotFoundError("The 'shortcuts' CLI requires macOS 12+.")

    cmd = ["shortcuts", "import", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _is_macos() -> bool:
    """Check if we're running on macOS."""
    import platform

    return platform.system() == "Darwin"


def _has_shortcuts_cli() -> bool:
    """Check if the macOS shortcuts CLI is available."""
    try:
        result = subprocess.run(["which", "shortcuts"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def generate_signing_script(
    watch_dir: str = "~/Desktop/sign_inbox",
    output_dir: str = "~/Desktop/sign_outbox",
    mode: str = "anyone",
) -> str:
    """
    Generate a bash script for a persistent auto-signing watcher.
    Run this once on your Mac for hands-free signing.

    Returns: The script content as a string.
    """
    return f"""#!/bin/bash
# Auto-Sign Shortcuts — watches a folder and signs new .shortcut files
# Generated by shortcuts_compiler.py
#
# SETUP:
#   1. brew install fswatch (one-time)
#   2. chmod +x auto_sign.sh
#   3. ./auto_sign.sh &  (run in background)
#   4. Drop .shortcut files into {watch_dir}
#   5. Signed files appear in {output_dir}

WATCH_DIR="{watch_dir}"
DONE_DIR="{output_dir}"
MODE="{mode}"

mkdir -p "$WATCH_DIR" "$DONE_DIR"

echo "Watching $WATCH_DIR for .shortcut files..."
echo "Signed files will appear in $DONE_DIR"
echo "Press Ctrl+C to stop."

fswatch -0 "$WATCH_DIR" | while read -d "" event; do
  if [[ "$event" == *.shortcut ]] && [[ -f "$event" ]]; then
    base=$(basename "$event" .shortcut)
    outfile="$DONE_DIR/${{base}}_signed.shortcut"
    echo "[$(date +%H:%M:%S)] Signing: $base..."
    if shortcuts sign -i "$event" -o "$outfile" -m "$MODE" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] ✓ Signed: $outfile"
      rm "$event"
    else
      echo "[$(date +%H:%M:%S)] ✗ Failed to sign: $base"
    fi
  fi
done
"""


def generate_setup_script() -> str:
    """
    Generate a one-time macOS setup script that configures everything needed
    for seamless shortcut generation and signing.

    Returns: The script content as a string.
    """
    return """#!/bin/bash
# Apple Shortcuts Compiler — One-Time macOS Setup
# Run this once to configure your Mac for seamless shortcut generation.

set -e
echo "=== Apple Shortcuts Compiler Setup ==="
echo ""

# 1. Check macOS version
sw_vers_major=$(sw_vers -productVersion | cut -d. -f1)
if [ "$sw_vers_major" -lt 12 ]; then
  echo "ERROR: macOS 12 (Monterey) or later required for shortcuts CLI."
  echo "Your version: $(sw_vers -productVersion)"
  exit 1
fi
echo "✓ macOS $(sw_vers -productVersion) detected"

# 2. Check shortcuts CLI
if ! command -v shortcuts &> /dev/null; then
  echo "ERROR: 'shortcuts' CLI not found. It should come with macOS 12+."
  echo "Try: xcode-select --install"
  exit 1
fi
echo "✓ shortcuts CLI available"

# 3. Test signing capability
TMPFILE=$(mktemp /tmp/test_shortcut.XXXXXX.shortcut)
python3 -c "
import plistlib
data = {
    'WFWorkflowActions': [],
    'WFWorkflowClientRelease': '4.0',
    'WFWorkflowClientVersion': '2302.0.4',
    'WFWorkflowIcon': {'WFWorkflowIconGlyphNumber': 59653, 'WFWorkflowIconStartColor': 4274264319},
    'WFWorkflowImportQuestions': [],
    'WFWorkflowInputContentItemClasses': [],
    'WFWorkflowMinimumClientVersion': 900,
    'WFWorkflowMinimumClientVersionString': '900',
    'WFWorkflowTypes': ['NCWidget', 'WatchKit'],
}
with open('$TMPFILE', 'wb') as f:
    plistlib.dump(data, f)
" 2>/dev/null

SIGNED_TMP="${TMPFILE%.shortcut}_signed.shortcut"
if shortcuts sign -i "$TMPFILE" -o "$SIGNED_TMP" -m anyone 2>/dev/null; then
  echo "✓ Shortcut signing works"
  rm -f "$TMPFILE" "$SIGNED_TMP"
else
  echo "WARNING: Signing test failed. You may need to:"
  echo "  1. Open Shortcuts app at least once"
  echo "  2. Sign into iCloud"
  echo "  3. Enable Private Sharing in Shortcuts settings"
  rm -f "$TMPFILE" "$SIGNED_TMP"
fi

# 4. Create convenience directories
SIGN_INBOX="$HOME/Desktop/sign_inbox"
SIGN_OUTBOX="$HOME/Desktop/sign_outbox"
mkdir -p "$SIGN_INBOX" "$SIGN_OUTBOX"
echo "✓ Created signing folders:"
echo "  Drop unsigned → $SIGN_INBOX"
echo "  Get signed   ← $SIGN_OUTBOX"

# 5. Check for fswatch (optional, for auto-signing)
if command -v fswatch &> /dev/null; then
  echo "✓ fswatch available (auto-signing supported)"
elif command -v brew &> /dev/null; then
  echo "  fswatch not found. Install for auto-signing: brew install fswatch"
else
  echo "  fswatch not found. Install Homebrew first: https://brew.sh"
fi

# 6. Create quick-sign alias
SHELL_RC="$HOME/.zshrc"
if [ -f "$HOME/.bashrc" ] && [ ! -f "$HOME/.zshrc" ]; then
  SHELL_RC="$HOME/.bashrc"
fi

if ! grep -q "sign-shortcut" "$SHELL_RC" 2>/dev/null; then
  echo "" >> "$SHELL_RC"
  echo '# Apple Shortcuts quick-sign alias' >> "$SHELL_RC"
  echo 'sign-shortcut() { shortcuts sign -i "$1" -o "${1%.shortcut}_signed.shortcut" -m "${2:-anyone}"; }' >> "$SHELL_RC"
  echo "✓ Added 'sign-shortcut' alias to $SHELL_RC"
  echo "  Usage: sign-shortcut myfile.shortcut [anyone|people-who-know-me]"
else
  echo "✓ sign-shortcut alias already exists"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Quick reference:"
echo "  sign-shortcut my.shortcut          → Sign for anyone"
echo "  sign-shortcut my.shortcut people-who-know-me  → Sign for contacts only"
echo "  Drop files in ~/Desktop/sign_inbox → Use auto_sign.sh for batch"
echo ""
echo "Prerequisites on each device:"
echo "  macOS: Shortcuts → Settings → Private Sharing → ON"
echo "  iOS:   Settings → Shortcuts → Private Sharing → ON"
"""


# =============================================================================
# ACTIONS MODULE — catalog-driven action system
# =============================================================================

# Load action catalog once at module level
_ACTION_CATALOG = None
_CANONICAL_MAP = None
_PARAM_SCHEMAS = None
_REFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "references")
_CATALOG_PATH = os.path.join(_REFS_DIR, "action_catalog.json")
_SCHEMA_PATH = os.path.join(_REFS_DIR, "param_schemas.json")


def _load_catalog():
    """Load the action catalog and parameter schemas from disk. Called once on first use."""
    global _ACTION_CATALOG, _CANONICAL_MAP, _PARAM_SCHEMAS
    import json

    catalog_path = os.path.normpath(_CATALOG_PATH)
    if os.path.exists(catalog_path):
        with open(catalog_path) as f:
            data = json.load(f)
        _ACTION_CATALOG = data.get("actions", {})
        _CANONICAL_MAP = data.get("_meta", {}).get("canonical_map", {})
    else:
        _ACTION_CATALOG = {}
        _CANONICAL_MAP = {}

    schema_path = os.path.normpath(_SCHEMA_PATH)
    if os.path.exists(schema_path):
        with open(schema_path) as f:
            _PARAM_SCHEMAS = json.load(f)
    else:
        _PARAM_SCHEMAS = {}


def _resolve_identifier(name: str) -> str:
    """
    Resolve any action name to its canonical identifier.

    Accepts:
      - Full identifier: "is.workflow.actions.gettext" → pass through
      - Short name: "gettext" → "is.workflow.actions.gettext"
      - Third-party: "com.apple.Pages.TSA..." → pass through
      - Dotted short: "text.split" → "is.workflow.actions.text.split"

    Uses the canonical_map from action_catalog.json for known mappings.
    Falls back to "is.workflow.actions." prefix for unknown short names.
    """
    if _CANONICAL_MAP is None:
        _load_catalog()

    # Already a full is.workflow.actions.* identifier? Pass through.
    if name.startswith("is.workflow.actions."):
        return name

    # Check canonical map (handles known short names and aliases)
    if name in _CANONICAL_MAP:
        return _CANONICAL_MAP[name]

    # Try adding is.workflow.actions. prefix (handles dotted short names
    # like "properties.weather.conditions" → "is.workflow.actions.properties.weather.conditions")
    prefixed = f"is.workflow.actions.{name}"
    if _ACTION_CATALOG and prefixed in _ACTION_CATALOG:
        return prefixed

    # For dotted names with 3+ segments that aren't in our catalog,
    # treat as fully-qualified third-party identifiers (reverse-domain format).
    # Covers: com.apple.*, ai.perplexity.*, codes.rambo.*, fm.overcast.*,
    # org.joinmastodon.*, fyi.lunar.*, etc.
    if "." in name:
        parts = name.split(".")
        if len(parts) >= 3:
            return name

    # Unknown action — this is a compile error.
    catalog_size = len(_ACTION_CATALOG or {})
    canonical_size = len(_CANONICAL_MAP or {})
    raise ValueError(
        f"Unknown action '{name}'. Not found in current catalog ({catalog_size} entries) "
        f"or canonical map ({canonical_size} entries). "
        f"Use actions.info() to look up valid action names, or check references/action_catalog.json."
    )


def _wrap_params(identifier: str, params: dict) -> dict:
    """
    Wrap parameter values according to the schema rules derived from
    245+ real shortcuts.

    For each parameter, the schema tells us:
    - Whether it accepts variables at all
    - HOW to wrap ActionHandle values:
        'attachment'    → WFTextTokenAttachment (direct variable ref)
        'token_string'  → WFTextTokenString with embedded variable
        'variable_ref'  → Variable ref dict (Type: Variable, Variable: ...)
        None            → This param doesn't accept variables; if an
                          ActionHandle is passed, raise an error.

    For plain values (str, int, float, bool, list, dict), pass through.
    """
    if _PARAM_SCHEMAS is None:
        _load_catalog()

    action_schema = _PARAM_SCHEMAS.get(identifier)
    has_schema = action_schema is not None
    if action_schema is None:
        action_schema = {}
    wrapped = {}

    for key, value in params.items():
        if isinstance(value, ActionHandle):
            param_info = action_schema.get(key) if has_schema else None
            if param_info:
                wrap_mode = param_info.get("handle_wrap")
                if wrap_mode == "attachment":
                    wrapped[key] = wrap_token_attachment(value.ref())
                elif wrap_mode == "token_string":
                    # Embed variable in a token string: "\ufffc" with attachment
                    wrapped[key] = {
                        "Value": {
                            "string": "\ufffc",
                            "attachmentsByRange": {"{0, 1}": value.ref()},
                        },
                        "WFSerializationType": "WFTextTokenString",
                    }
                elif wrap_mode == "variable_ref":
                    wrapped[key] = wrap_conditional_input(value.ref())
                else:
                    # handle_wrap is None or unrecognized — default to attachment.
                    # WFTextTokenAttachment is the most common wrapping mode (~70%
                    # of variable-accepting params use it). This is a safe fallback
                    # that produces valid shortcuts even when schema is incomplete.
                    wrapped[key] = wrap_token_attachment(value.ref())
            else:
                # No schema for this action or no wrapping rule for this param.
                # Default to WFTextTokenAttachment — it's correct for the majority
                # of cases and produces valid plist output. Better than hard-failing
                # when the model generates valid DSL referencing variables.
                wrapped[key] = wrap_token_attachment(value.ref())
        else:
            wrapped[key] = value

    return wrapped


def _get_output_name(identifier: str) -> str:
    """
    Derive a human-readable output name for an action from the catalog.
    Falls back to "Result" if the action isn't cataloged.
    """
    if _ACTION_CATALOG is None:
        _load_catalog()
    entry = _ACTION_CATALOG.get(identifier)
    if entry:
        return entry.get("name", "Result")
    return "Result"


class actions:
    """
    Catalog-driven action builder. Every action in action_catalog.json
    is accessible through actions.make().

    Primary API:
        actions.make("identifier_or_short_name", ParamName=value, ...)

    The system auto-resolves short names ("gettext") to canonical identifiers
    ("is.workflow.actions.gettext") and auto-wraps ActionHandle values.

    For complex parameter construction (dictionaries, headers, token strings),
    use the builder helpers: actions.build_dict_items(), actions.build_list(),
    actions.build_token_string().
    """

    # ── Core API ──────────────────────────────────────────────────

    # Parameters injected by the compiler (add(), control flow methods).
    # These are always valid and never come from user code via make().
    _COMPILER_PARAMS = frozenset(
        {
            "UUID",
            "CustomOutputName",
            "GroupingIdentifier",
            "WFControlFlowMode",
        }
    )

    @staticmethod
    def make(name: str, _validate_params: bool = True, **params) -> dict:
        """
        Create ANY action by name. This is the primary API.
        Validates both the action name and all parameter names against the catalog.

        Raises ValueError if:
          - The action name is not in the catalog
          - Any parameter name is not in the catalog's observed_params for that action
            (unless _validate_params=False)

        Args:
            name: Action identifier — accepts any of:
                  - Short name: "gettext", "text.split", "randomnumber"
                  - Full identifier: "is.workflow.actions.gettext"
                  - Third-party: "com.apple.Pages.TSADocumentCreateIntent"
            _validate_params: If False, skip parameter name validation against the
                  catalog. Used by the compiler bridge for IR that has already been
                  semantically validated. Action name resolution still runs.
            **params: Action parameters (Apple's WF* parameter names).
                      ActionHandle values are auto-wrapped.

        Examples:
            actions.make("comment", WFCommentActionText="Hello")
            actions.make("randomnumber", WFRandomNumberMinimum=1, WFRandomNumberMaximum=6)
            actions.make("text.split", WFTextSeparator="Custom", WFTextCustomSeparator=",")
            actions.make("setvariable", WFVariableName="MyVar", WFInput=some_handle)
        """
        # Resolve name → canonical identifier (raises ValueError if unknown)
        identifier = _resolve_identifier(name)

        # Validate parameter names against the catalog
        if _ACTION_CATALOG is None:
            _load_catalog()

        if _validate_params:
            catalog_entry = _ACTION_CATALOG.get(identifier)
            if catalog_entry:
                known_params = set(catalog_entry.get("observed_params", {}).keys())
                # Add compiler-injected params that are always valid
                known_params |= actions._COMPILER_PARAMS
                for param_name in params:
                    if param_name not in known_params:
                        raise ValueError(
                            f"Unknown parameter '{param_name}' for action '{name}' "
                            f"({identifier}). "
                            f"Known parameters: {sorted(known_params - actions._COMPILER_PARAMS)}. "
                            f"Use actions.info('{name}') to see valid parameters."
                        )

        # Wrap ActionHandle values according to schema rules
        wrapped_params = _wrap_params(identifier, params)
        return {
            "WFWorkflowActionIdentifier": identifier,
            "WFWorkflowActionParameters": wrapped_params,
        }

    @staticmethod
    def info(name: str) -> Optional[dict]:
        """
        Look up an action in the catalog. Returns its metadata:
        identifier, name, description, parameters_doc, observed_params, etc.
        Returns None if not found.
        """
        if _ACTION_CATALOG is None:
            _load_catalog()
        identifier = _resolve_identifier(name)
        return _ACTION_CATALOG.get(identifier)

    # ── Builder Helpers ───────────────────────────────────────────
    # These construct complex parameter values that Apple's plist
    # format requires. Use them as parameter values in make().

    @staticmethod
    def build_dict_items(d: dict[str, Any]) -> list[dict]:
        """
        Build a WFItems-style list for Dictionary actions.
        Converts a Python dict to Apple's field-item format.

        Usage:
            actions.make("dictionary",
                WFItems={"Value": {"WFDictionaryFieldValueItems":
                    actions.build_dict_items({"key": "value", "num": 42})
                }})
        """
        items = []
        for key, val in d.items():
            item_type = 0  # String
            # IMPORTANT: bool must be checked before int because Python's bool subclasses int.
            # isinstance(True, int) is True, so checking int first would misclassify booleans.
            if isinstance(val, bool):
                item_type = 6  # Boolean
            elif isinstance(val, (int, float)):
                item_type = 3  # Number
            elif isinstance(val, dict):
                item_type = 1  # Dictionary
            elif isinstance(val, list):
                item_type = 2  # Array

            # Build the value field — nested dicts get recursive serialization
            if item_type == 1 and isinstance(val, dict):
                # Recursive: nested dict → WFDictionaryFieldValueItems
                val_field = {
                    "Value": {
                        "WFDictionaryFieldValueItems": actions.build_dict_items(val)
                    },
                    "WFSerializationType": "WFDictionaryFieldValue",
                }
            elif item_type == 6:
                val_field = {
                    "Value": val,
                    "WFSerializationType": "WFNumberSubstitutableState",
                }
            else:
                val_field = {
                    "Value": {"string": str(val)},
                    "WFSerializationType": "WFTextTokenString",
                }

            item = {
                "WFItemType": item_type,
                "WFKey": {
                    "Value": {"string": str(key)},
                    "WFSerializationType": "WFTextTokenString",
                },
                "WFValue": val_field,
            }
            items.append(item)
        return items

    @staticmethod
    def build_list(items: list[str]) -> list[dict]:
        """
        Build a WFItems list for List actions.
        Converts Python strings to Apple's text token format.

        Usage:
            actions.make("list", WFItems=actions.build_list(["a", "b", "c"]))
        """
        return [
            {
                "WFItemType": 0,
                "WFValue": {
                    "Value": {"string": s},
                    "WFSerializationType": "WFTextTokenString",
                },
            }
            for s in items
        ]

    @staticmethod
    def build_headers(headers: dict[str, str]) -> dict:
        """
        Build HTTP headers dict for download_url actions.

        Usage:
            actions.make("downloadurl",
                WFHTTPMethod="POST",
                WFHTTPHeaders=actions.build_headers({"Authorization": "Bearer xyz"}))
        """
        items = []
        for key, val in headers.items():
            items.append(
                {
                    "WFItemType": 0,
                    "WFKey": {
                        "Value": {"string": key},
                        "WFSerializationType": "WFTextTokenString",
                    },
                    "WFValue": {
                        "Value": {"string": val},
                        "WFSerializationType": "WFTextTokenString",
                    },
                }
            )
        return {"Value": {"WFDictionaryFieldValueItems": items}}

    @staticmethod
    def build_quantity(magnitude: Union[int, float], unit: str) -> dict:
        """
        Build a WFQuantityFieldValue for duration/measurement params.

        Usage:
            actions.make("adjustdate",
                WFDuration=actions.build_quantity(7, "days"))
        """
        return {
            "Value": {"Unit": unit, "Magnitude": magnitude},
            "WFSerializationType": "WFQuantityFieldValue",
        }

    @staticmethod
    def build_token_string(text: str, attachment: Optional[dict] = None) -> dict:
        """
        Build a WFTextTokenString — a string with an optional embedded variable.
        If no attachment, returns a plain token string.

        Usage:
            actions.make("gettext",
                WFTextActionText=actions.build_token_string("Hello world"))
        """
        result = {"Value": {"string": text}, "WFSerializationType": "WFTextTokenString"}
        if attachment:
            pos = text.find("\ufffc")
            if pos >= 0:
                result["Value"]["attachmentsByRange"] = {f"{{{pos}, 1}}": attachment}
        return result
