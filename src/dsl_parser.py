"""
ShortcutDSL Parser: DSL text → ShortcutIR.

Uses Lark to parse DSL text according to shortcutdsl.lark,
then transforms the parse tree into typed IR dataclasses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lark import Lark, Transformer, Tree, v_args

from dsl_ir import (
    ActionStatement,
    BoolValue,
    Comment,
    DictLiteral,
    ForeachBlock,
    HandleRef,
    HeadersLiteral,
    IfBlock,
    InterpolatedString,
    ListLiteral,
    MenuBlock,
    MenuCase,
    NumberValue,
    QuantityLiteral,
    RepeatBlock,
    SetVariable,
    ShortcutIR,
    StringValue,
    VarRef,
)

GRAMMAR_PATH = (
    Path(__file__).resolve().parent.parent / "references" / "shortcutdsl.lark"
)


def _unquote(s: str) -> str:
    """Remove surrounding quotes and unescape."""
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    return (
        s.replace('\\"', '"')
        .replace("\\\\", "\\")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
    )


def _parse_number(s: str) -> int | float:
    """Parse a number string to int or float."""
    if "." in s:
        return float(s)
    return int(s)


@v_args(inline=True)
class DSLTransformer(Transformer):
    """Transform Lark parse tree → ShortcutIR."""

    def start(self, header, *statements):
        name = header
        stmts = [s for s in statements if s is not None]
        return ShortcutIR(name=name, statements=stmts)

    def header(self, name):
        return _unquote(str(name))

    # --- Values ---

    def string_value(self, token):
        return StringValue(_unquote(str(token)))

    def number_value(self, token):
        return NumberValue(_parse_number(str(token)))

    def bool_value(self, token):
        return BoolValue(str(token) == "true")

    def var_ref_value(self, ref):
        return ref

    def handle_ref_value(self, ref):
        return ref

    def interp_value(self, interp):
        return interp

    # --- References ---

    def var_ref(self, name):
        return VarRef(str(name))

    def handle_prev(self):
        return HandleRef("prev")

    def handle_item(self):
        return HandleRef("item")

    def handle_index(self):
        return HandleRef("index")

    def handle_input(self):
        return HandleRef("input")

    def handle_date(self):
        return HandleRef("date")

    def handle_named(self, name):
        return HandleRef(str(name))

    # --- Interpolated Strings ---

    def interpolated(self, *parts):
        return InterpolatedString(tuple(parts))

    def interp_text(self, token):
        return StringValue(str(token))

    def interp_var(self, ref):
        return ref

    def interp_handle(self, ref):
        return ref

    # --- Structured Literals ---

    def dict_literal(self, *entries):
        return DictLiteral(tuple(entries))

    def dict_entry(self, key, val):
        return (_unquote(str(key)), val)

    def list_literal(self, *items):
        return ListLiteral(tuple(items))

    def quantity_literal(self, magnitude, unit):
        # magnitude can be a NumberValue, HandleRef, or VarRef (already transformed)
        if isinstance(magnitude, NumberValue):
            mag = magnitude.value
        elif isinstance(magnitude, (HandleRef, VarRef)):
            mag = magnitude  # Store the reference as-is
        else:
            mag = _parse_number(str(magnitude))
        return QuantityLiteral(mag, _unquote(str(unit)))

    def headers_literal(self, dict_lit):
        return HeadersLiteral(dict_lit.entries)

    # --- Actions ---

    def action_name(self, name):
        return str(name)

    def param(self, name, val):
        return (str(name), val)

    def action_stmt(self, name, *params):
        param_dict = {}
        for item in params:
            if isinstance(item, tuple) and len(item) == 2:
                param_dict[item[0]] = item[1]
        return ActionStatement(action_name=name, params=param_dict)

    # --- Variables ---

    def set_stmt(self, ref, val):
        return SetVariable(var_name=ref.name, value=val)

    # --- Control Flow ---

    def if_block(self, *args):
        # args: target, condition, [compare_value], *then_stmts, [else_clause]
        parts = list(args)
        target = parts[0]
        condition = str(parts[1])

        # Find the boundary: compare_value is optional, then body stmts, then optional else
        idx = 2
        compare_value = None
        then_body = []
        else_body = None

        # Check if next item is a compare value (IRValue, not Statement, not list)
        if idx < len(parts) and _is_ir_value(parts[idx]):
            compare_value = parts[idx]
            idx += 1

        # Remaining items are then_body statements and possibly an else_body list
        for i in range(idx, len(parts)):
            item = parts[i]
            if isinstance(item, list):
                # This is the else clause body
                else_body = item
            elif item is not None:
                then_body.append(item)

        return IfBlock(
            target=target,
            condition=condition,
            compare_value=compare_value,
            then_body=then_body,
            else_body=else_body,
        )

    def else_clause(self, *stmts):
        return [s for s in stmts if s is not None]

    def menu_block(self, prompt, *cases):
        # prompt can be a QUOTED_STRING Token or an InterpolatedString
        if isinstance(prompt, InterpolatedString):
            prompt_val = prompt  # Preserve InterpolatedString for bridge serialization
        else:
            prompt_val = _unquote(str(prompt))
        return MenuBlock(
            prompt=prompt_val,
            cases=[c for c in cases if isinstance(c, MenuCase)],
        )

    def menu_case(self, label, *stmts):
        return MenuCase(
            label=_unquote(str(label)),
            body=[s for s in stmts if s is not None],
        )

    def repeat_block(self, count, *stmts):
        return RepeatBlock(
            count=count,
            body=[s for s in stmts if s is not None],
        )

    def repeat_number(self, token):
        return NumberValue(_parse_number(str(token)))

    def foreach_block(self, collection, *stmts):
        return ForeachBlock(
            collection=collection,
            body=[s for s in stmts if s is not None],
        )

    # --- Comments ---

    def comment(self, text):
        return Comment(text=str(text).lstrip("#").strip())

    # _NL terminals are auto-discarded by Lark (underscore prefix)


def _is_ir_value(obj: Any) -> bool:
    """Check if object is an IR value (not a statement or list)."""
    return isinstance(
        obj,
        (
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
        ),
    )


# ============================================================
# Public API
# ============================================================

_parser: Lark | None = None


def get_parser() -> Lark:
    """Get or create the Lark parser (cached)."""
    global _parser
    if _parser is None:
        grammar_text = GRAMMAR_PATH.read_text()
        _parser = Lark(
            grammar_text,
            parser="lalr",
            transformer=None,  # We apply transformer separately for line numbers
            propagate_positions=True,
        )
    return _parser


def parse_dsl(text: str) -> ShortcutIR:
    """Parse DSL text into a ShortcutIR.

    Args:
        text: The DSL source text.

    Returns:
        A ShortcutIR representing the parsed shortcut.

    Raises:
        lark.exceptions.LarkError: If the DSL text has syntax errors.
    """
    parser = get_parser()
    tree = parser.parse(text)

    # Extract line numbers from parse tree before transformation
    line_map = _extract_line_numbers(tree)

    transformer = DSLTransformer()
    ir = transformer.transform(tree)

    # Assign line numbers to IR nodes
    _assign_line_numbers(ir.statements, line_map, [0])

    return ir


# Statement-level tree rule names that carry line numbers
_STMT_RULES = frozenset(
    {
        "action_stmt",
        "set_stmt",
        "if_block",
        "menu_block",
        "repeat_block",
        "foreach_block",
        "comment",
    }
)


def _extract_line_numbers(tree: Tree) -> list[int]:
    """Walk the parse tree and extract line numbers for statement nodes in order."""
    lines: list[int] = []
    _walk_for_lines(tree, lines)
    return lines


def _walk_for_lines(node: Any, lines: list[int]) -> None:
    """Recursively walk tree, collecting line numbers for statement nodes."""
    if not isinstance(node, Tree):
        return
    if node.data in _STMT_RULES:
        line = getattr(node.meta, "line", 0) if hasattr(node, "meta") else 0
        lines.append(line)
    for child in node.children:
        _walk_for_lines(child, lines)


def _assign_line_numbers(
    stmts: list,
    line_map: list[int],
    idx: list[int],
) -> None:
    """Walk IR statements and assign line numbers from the pre-extracted map."""
    from dsl_ir import iter_child_blocks

    for stmt in stmts:
        line = line_map[idx[0]] if idx[0] < len(line_map) else 0
        stmt.line_number = line
        idx[0] += 1
        for body, _ctx, _is_loop in iter_child_blocks(stmt):
            _assign_line_numbers(body, line_map, idx)
