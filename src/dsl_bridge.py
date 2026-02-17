"""
ShortcutDSL Compiler Bridge: IR → shortcuts_compiler API calls.

Takes a validated ShortcutIR and produces a compiled Shortcut object
by translating each IR statement into the appropriate compiler API calls.

Pipeline: DSL text → (parser) → ShortcutIR → (validator) → validated IR → (bridge) → Shortcut
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Union

from dsl_ir import (
    ShortcutIR,
    ActionStatement,
    SetVariable,
    IfBlock,
    MenuBlock,
    RepeatBlock,
    ForeachBlock,
    Comment,
    StringValue,
    NumberValue,
    BoolValue,
    VarRef,
    HandleRef,
    InterpolatedString,
    DictLiteral,
    ListLiteral,
    QuantityLiteral,
    HeadersLiteral,
    Statement,
    IRValue,
)

from shortcuts_compiler import (
    Shortcut,
    actions,
    ActionHandle,
    ref_variable,
    ref_extension_input,
    ref_current_date,
    wrap_token_attachment,
    wrap_token_string,
    wrap_token_string_multi,
    wrap_conditional_input,
)


class CompilerBridge:
    """Translates a ShortcutIR into a compiled Shortcut object."""

    def __init__(self):
        # Track ActionHandles by position for @prev resolution
        self._handle_stack: list[ActionHandle | None] = []
        # Track named variables (from SET) → their ActionHandle
        self._variables: dict[str, ActionHandle] = {}
        # Track named handles (from ACTION) → their ActionHandle
        # Keyed by action short name so @someName resolves to ActionOutput refs
        self._named_handles: dict[str, ActionHandle] = {}
        # The Shortcut being built
        self._shortcut: Shortcut | None = None

    def compile(self, ir: ShortcutIR) -> Shortcut:
        """Compile a ShortcutIR into a Shortcut object.

        Args:
            ir: The validated ShortcutIR.

        Returns:
            A Shortcut ready for .save(), .deliver(), etc.
        """
        self._shortcut = Shortcut(ir.name)
        self._handle_stack = []
        self._variables = {}
        self._named_handles = {}

        self._compile_statements(ir.statements)

        return self._shortcut

    def _compile_statements(self, stmts: list[Statement]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ActionStatement):
                self._compile_action(stmt)
            elif isinstance(stmt, SetVariable):
                self._compile_set_variable(stmt)
            elif isinstance(stmt, IfBlock):
                self._compile_if_block(stmt)
            elif isinstance(stmt, MenuBlock):
                self._compile_menu_block(stmt)
            elif isinstance(stmt, RepeatBlock):
                self._compile_repeat_block(stmt)
            elif isinstance(stmt, ForeachBlock):
                self._compile_foreach_block(stmt)
            elif isinstance(stmt, Comment):
                pass  # Comments don't compile to actions

    def _compile_action(self, stmt: ActionStatement) -> None:
        """Compile an ACTION statement."""
        params = {}
        for key, value in stmt.params.items():
            params[key] = self._resolve_value(value)

        # Route through actions.make() for identifier resolution and _wrap_params.
        # Skip param-name validation (_validate_params=False) because the IR has
        # already been semantically validated — unknown params are warnings, not errors.
        # Handle 'name' key collision: actions.make() takes 'name' as first positional arg.
        if "name" in params:
            name_val = params.pop("name")
            action_dict = actions.make(stmt.action_name, _validate_params=False, **params)
            action_dict["WFWorkflowActionParameters"]["name"] = name_val
        else:
            action_dict = actions.make(stmt.action_name, _validate_params=False, **params)
        handle = self._shortcut.add(action_dict)
        self._handle_stack.append(handle)
        # Register by action short name for named handle resolution (@someName)
        # The decompiler names non-@prev handles after the action's short name,
        # so we mirror that here. Later refs to @someName look up this dict first
        # and produce ActionOutput refs (not Variable refs).
        short = stmt.action_name.rsplit(".", 1)[-1] if "." in stmt.action_name else stmt.action_name
        self._named_handles[short] = handle

    def _compile_set_variable(self, stmt: SetVariable) -> None:
        """Compile a SET $var = value statement into a setvariable action."""
        resolved_value = self._resolve_value(stmt.value)

        action_dict = actions.make(
            "setvariable",
            WFVariableName=stmt.var_name,
            WFInput=resolved_value,
        )
        handle = self._shortcut.add(action_dict)
        self._handle_stack.append(handle)
        self._variables[stmt.var_name] = handle

    def _compile_if_block(self, stmt: IfBlock) -> None:
        """Compile an IF/ELSE/ENDIF block."""
        target_ref = self._resolve_ref(stmt.target)

        compare_value = None
        if stmt.compare_value is not None:
            compare_value = self._resolve_value_raw(stmt.compare_value)

        if stmt.else_body is not None:
            with self._shortcut.if_else_block(
                target_ref, condition=stmt.condition, compare_value=compare_value
            ) as otherwise:
                self._compile_statements(stmt.then_body)
                otherwise()
                self._compile_statements(stmt.else_body)
        else:
            with self._shortcut.if_block(
                target_ref, condition=stmt.condition, compare_value=compare_value
            ):
                self._compile_statements(stmt.then_body)

    def _compile_menu_block(self, stmt: MenuBlock) -> None:
        """Compile a MENU/CASE/ENDMENU block."""
        options = [case.label for case in stmt.cases]

        # Resolve the prompt: plain string or interpolated token string
        if isinstance(stmt.prompt, InterpolatedString):
            prompt_value = self._resolve_interpolated(stmt.prompt)
        else:
            prompt_value = stmt.prompt

        with self._shortcut.menu_block(prompt_value, options) as cases:
            for case in stmt.cases:
                cases[case.label]()
                self._compile_statements(case.body)

    def _compile_repeat_block(self, stmt: RepeatBlock) -> None:
        """Compile a REPEAT/ENDREPEAT block."""
        count = self._resolve_repeat_count(stmt.count)

        with self._shortcut.repeat_block(count):
            self._compile_statements(stmt.body)

    def _compile_foreach_block(self, stmt: ForeachBlock) -> None:
        """Compile a FOREACH/ENDFOREACH block."""
        collection_ref = self._resolve_ref(stmt.collection)

        with self._shortcut.repeat_each_block(collection_ref):
            self._compile_statements(stmt.body)

    # ================================================================
    # Value Resolution
    # ================================================================

    def _resolve_value(self, value: IRValue) -> Any:
        """Resolve an IR value to a compiler-compatible value.

        For ActionHandle-wrappable values, returns the handle itself
        so the compiler's auto-wrapping can apply.
        """
        if isinstance(value, StringValue):
            return value.value
        elif isinstance(value, NumberValue):
            return value.value
        elif isinstance(value, BoolValue):
            return value.value
        elif isinstance(value, VarRef):
            # If we have an ActionHandle for this variable, use it
            if value.name in self._variables:
                return self._variables[value.name]
            # Otherwise, build a variable reference dict
            return ref_variable(value.name)
        elif isinstance(value, HandleRef):
            return self._resolve_handle(value)
        elif isinstance(value, InterpolatedString):
            return self._resolve_interpolated(value)
        elif isinstance(value, DictLiteral):
            return self._resolve_dict(value)
        elif isinstance(value, ListLiteral):
            return self._resolve_list(value)
        elif isinstance(value, QuantityLiteral):
            mag = value.magnitude
            if isinstance(mag, (VarRef, HandleRef)):
                mag = self._resolve_value(mag)
            return actions.build_quantity(mag, value.unit)
        elif isinstance(value, HeadersLiteral):
            return self._resolve_headers(value)
        else:
            raise ValueError(f"Unknown IR value type: {type(value)}")

    def _resolve_value_raw(self, value: IRValue) -> Any:
        """Resolve an IR value to a raw Python value (for compare_value in conditionals).
        Does NOT return ActionHandles — returns the primitive or wrapped dict.
        """
        if isinstance(value, StringValue):
            return value.value
        elif isinstance(value, NumberValue):
            return value.value
        elif isinstance(value, BoolValue):
            return value.value
        elif isinstance(value, VarRef):
            if value.name in self._variables:
                return self._variables[value.name]
            return ref_variable(value.name)
        elif isinstance(value, HandleRef):
            return self._resolve_handle(value)
        else:
            return self._resolve_value(value)

    def _resolve_ref(self, ref: VarRef | HandleRef) -> Any:
        """Resolve a variable or handle reference for control flow input."""
        if isinstance(ref, VarRef):
            if ref.name in self._variables:
                return self._variables[ref.name]
            return ref.name  # Pass as string — compiler wraps it
        elif isinstance(ref, HandleRef):
            return self._resolve_handle(ref)
        return ref

    def _resolve_handle(self, ref: HandleRef) -> Any:
        """Resolve a handle reference to an ActionHandle or dict."""
        if ref.kind == "prev":
            if self._handle_stack:
                return self._handle_stack[-1]
            raise ValueError("@prev used but no preceding action handle")
        elif ref.kind == "item":
            # In a repeat_each, @item refers to the current item
            # The compiler doesn't have a built-in for this — use magic variable
            return ref_variable("Repeat Item")
        elif ref.kind == "index":
            return ref_variable("Repeat Index")
        elif ref.kind == "input":
            return ref_extension_input()
        elif ref.kind == "date":
            return ref_current_date()
        else:
            # Named handle — look up as ActionOutput first, fall back to Variable
            if ref.kind in self._named_handles:
                return self._named_handles[ref.kind]
            # Could also be a variable set with SET
            if ref.kind in self._variables:
                return self._variables[ref.kind]
            # Fall back to variable reference (may be runtime-defined)
            return ref_variable(ref.kind)

    def _resolve_interpolated(self, interp: InterpolatedString) -> Any:
        """Resolve an interpolated string to a token string or plain string."""
        # Check if there are any embedded references
        refs = [p for p in interp.parts if isinstance(p, (VarRef, HandleRef))]
        if not refs:
            # No references — just concatenate strings
            return "".join(
                p.value if isinstance(p, StringValue) else str(p)
                for p in interp.parts
            )

        if len(refs) == 1:
            # Single reference — use wrap_token_string
            before_parts = []
            after_parts = []
            found_ref = False
            attachment = None

            for part in interp.parts:
                if isinstance(part, (VarRef, HandleRef)):
                    found_ref = True
                    resolved = self._resolve_value(part)
                    if isinstance(resolved, ActionHandle):
                        attachment = resolved.ref()
                    elif isinstance(resolved, dict):
                        attachment = resolved
                    else:
                        attachment = ref_variable(str(resolved))
                elif isinstance(part, StringValue):
                    if found_ref:
                        after_parts.append(part.value)
                    else:
                        before_parts.append(part.value)

            before = "".join(before_parts)
            after = "".join(after_parts)
            return wrap_token_string(before, attachment, after)
        else:
            # Multiple references — use wrap_token_string_multi
            template = ""
            attachments = []
            for part in interp.parts:
                if isinstance(part, StringValue):
                    template += part.value
                elif isinstance(part, (VarRef, HandleRef)):
                    template += "\ufffc"
                    resolved = self._resolve_value(part)
                    if isinstance(resolved, ActionHandle):
                        attachments.append(resolved.ref())
                    elif isinstance(resolved, dict):
                        attachments.append(resolved)
                    else:
                        attachments.append(ref_variable(str(resolved)))

            return wrap_token_string_multi(template, attachments)

    def _resolve_dict(self, d: DictLiteral) -> list[dict]:
        """Resolve a dict literal to build_dict_items format.

        Handles recursive nesting: {"key": {"subkey": "val"}} is serialized
        as nested WFDictionaryFieldValueItems — the full horror show of Apple's
        plist dictionary format, handled cleanly.
        """
        py_dict = {}
        for key, value in d.entries:
            if isinstance(value, DictLiteral):
                # Recursive: nested DictLiteral → nested Python dict
                py_dict[key] = self._resolve_dict_to_python(value)
            elif isinstance(value, ListLiteral):
                py_dict[key] = self._resolve_list_to_python(value)
            else:
                py_dict[key] = self._resolve_value_raw(value)
        return actions.build_dict_items(py_dict)

    def _resolve_dict_to_python(self, d: DictLiteral) -> dict:
        """Recursively resolve a DictLiteral to a plain Python dict."""
        result = {}
        for key, value in d.entries:
            if isinstance(value, DictLiteral):
                result[key] = self._resolve_dict_to_python(value)
            elif isinstance(value, ListLiteral):
                result[key] = self._resolve_list_to_python(value)
            else:
                result[key] = self._resolve_value_raw(value)
        return result

    def _resolve_list_to_python(self, lst: ListLiteral) -> list:
        """Recursively resolve a ListLiteral to a plain Python list."""
        result = []
        for item in lst.items:
            if isinstance(item, DictLiteral):
                result.append(self._resolve_dict_to_python(item))
            elif isinstance(item, ListLiteral):
                result.append(self._resolve_list_to_python(item))
            elif isinstance(item, StringValue):
                result.append(item.value)
            else:
                result.append(self._resolve_value_raw(item))
        return result

    def _resolve_list(self, lst: ListLiteral) -> list[dict]:
        """Resolve a list literal to build_list format."""
        py_list = []
        for item in lst.items:
            if isinstance(item, StringValue):
                py_list.append(item.value)
            else:
                py_list.append(str(self._resolve_value_raw(item)))
        return actions.build_list(py_list)

    def _resolve_headers(self, h: HeadersLiteral) -> dict:
        """Resolve a headers literal to build_headers format."""
        py_dict = {}
        for key, value in h.entries:
            if isinstance(value, StringValue):
                py_dict[key] = value.value
            else:
                py_dict[key] = str(self._resolve_value_raw(value))
        return actions.build_headers(py_dict)

    def _resolve_repeat_count(self, count: IRValue) -> Any:
        """Resolve a repeat count to int or ActionHandle."""
        if isinstance(count, NumberValue):
            return int(count.value)
        elif isinstance(count, VarRef):
            if count.name in self._variables:
                return self._variables[count.name]
            return ref_variable(count.name)
        elif isinstance(count, HandleRef):
            return self._resolve_handle(count)
        return int(self._resolve_value_raw(count))


# ============================================================
# Public API
# ============================================================

def compile_ir(ir: ShortcutIR) -> Shortcut:
    """Compile a ShortcutIR into a Shortcut object.

    This is the main entry point for the compiler bridge.
    The IR should be validated first via dsl_validator.validate_ir().

    Args:
        ir: A validated ShortcutIR.

    Returns:
        A Shortcut object ready for .save(), .deliver(), etc.
    """
    bridge = CompilerBridge()
    return bridge.compile(ir)
