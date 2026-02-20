"""
ShortcutDSL Intermediate Representation (IR).

Typed dataclass layer between the DSL parser and the compiler bridge.
Captures the full structure of a shortcut in a form that can be:
  - Validated against action_catalog.json + param_schemas.json
  - Compiled to shortcuts_compiler API calls
  - Inspected, diffed, and serialized for debugging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

# ============================================================
# IR Value Types
# ============================================================


@dataclass(frozen=True)
class StringValue:
    """A literal quoted string."""

    value: str

    def __repr__(self) -> str:
        return f'StringValue("{self.value}")'


@dataclass(frozen=True)
class NumberValue:
    """A literal number (int or float)."""

    value: Union[int, float]

    def __repr__(self) -> str:
        return f"NumberValue({self.value})"


@dataclass(frozen=True)
class BoolValue:
    """A literal boolean."""

    value: bool

    def __repr__(self) -> str:
        return f"BoolValue({self.value})"


@dataclass(frozen=True)
class VarRef:
    """Reference to a named variable: $varname."""

    name: str

    def __repr__(self) -> str:
        return f"VarRef(${self.name})"


@dataclass(frozen=True)
class HandleRef:
    """Reference to an action handle: @prev, @item, @index, @input, @date, @named."""

    kind: str  # "prev", "item", "index", "input", "date", or a custom name

    def __repr__(self) -> str:
        return f"HandleRef(@{self.kind})"


@dataclass(frozen=True)
class InterpolatedString:
    """A backtick-delimited string with embedded variable/handle references.
    Parts is a list of StringValue, VarRef, or HandleRef."""

    parts: tuple[Union[StringValue, VarRef, HandleRef], ...]

    def __repr__(self) -> str:
        return f"InterpolatedString({list(self.parts)})"


@dataclass(frozen=True)
class DictLiteral:
    """A dictionary literal: {"key": value, ...}."""

    entries: tuple[tuple[str, IRValue], ...]

    def __repr__(self) -> str:
        items = ", ".join(f'"{k}": {v}' for k, v in self.entries)
        return f"DictLiteral({{{items}}})"


@dataclass(frozen=True)
class ListLiteral:
    """A list literal: [value, ...]."""

    items: tuple[IRValue, ...]

    def __repr__(self) -> str:
        return f"ListLiteral([{', '.join(str(i) for i in self.items)}])"


@dataclass(frozen=True)
class QuantityLiteral:
    """A quantity literal: QTY(7, "days") or QTY(@prev, "m")."""

    magnitude: Union[int, float, "VarRef", "HandleRef"]
    unit: str

    def __repr__(self) -> str:
        return f'QuantityLiteral({self.magnitude}, "{self.unit}")'


@dataclass(frozen=True)
class HeadersLiteral:
    """HTTP headers literal: HEADERS {"Key": "Value"}."""

    entries: tuple[tuple[str, IRValue], ...]

    def __repr__(self) -> str:
        items = ", ".join(f'"{k}": {v}' for k, v in self.entries)
        return f"HeadersLiteral({{{items}}})"


# Union type for all IR values
IRValue = Union[
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
]


# ============================================================
# IR Statement Types
# ============================================================


@dataclass
class ActionStatement:
    """A single action: ACTION name param=value param=value."""

    action_name: str
    params: dict[str, IRValue]
    line_number: int = 0

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"ActionStatement({self.action_name}, {{{params_str}}})"


@dataclass
class SetVariable:
    """Variable assignment: SET $name = value."""

    var_name: str
    value: IRValue
    line_number: int = 0

    def __repr__(self) -> str:
        return f"SetVariable(${self.var_name} = {self.value})"


@dataclass
class IfBlock:
    """Conditional block: IF target condition [compare_value] ... ELSE ... ENDIF."""

    target: Union[VarRef, HandleRef]
    condition: str
    compare_value: IRValue | None
    then_body: list[Statement]
    else_body: list[Statement] | None
    line_number: int = 0

    def __repr__(self) -> str:
        cmp = f" {self.compare_value}" if self.compare_value else ""
        els = f", else={len(self.else_body)} stmts" if self.else_body else ""
        return f"IfBlock({self.target} {self.condition}{cmp}, then={len(self.then_body)} stmts{els})"


@dataclass
class MenuCase:
    """A single menu case: CASE "label" statements..."""

    label: str
    body: list[Statement]

    def __repr__(self) -> str:
        return f'MenuCase("{self.label}", {len(self.body)} stmts)'


@dataclass
class MenuBlock:
    """Menu block: MENU "prompt" CASE ... CASE ... ENDMENU."""

    prompt: Union[str, "InterpolatedString"]
    cases: list[MenuCase]
    line_number: int = 0

    def __repr__(self) -> str:
        return f'MenuBlock("{self.prompt}", {len(self.cases)} cases)'


@dataclass
class RepeatBlock:
    """Repeat block: REPEAT count ... ENDREPEAT."""

    count: IRValue
    body: list[Statement]
    line_number: int = 0

    def __repr__(self) -> str:
        return f"RepeatBlock({self.count}, {len(self.body)} stmts)"


@dataclass
class ForeachBlock:
    """For-each block: FOREACH collection ... ENDFOREACH."""

    collection: Union[VarRef, HandleRef]
    body: list[Statement]
    line_number: int = 0

    def __repr__(self) -> str:
        return f"ForeachBlock({self.collection}, {len(self.body)} stmts)"


@dataclass
class Comment:
    """A comment line: # text."""

    text: str
    line_number: int = 0


# Union type for all statements
Statement = Union[
    ActionStatement,
    SetVariable,
    IfBlock,
    MenuBlock,
    RepeatBlock,
    ForeachBlock,
    Comment,
]


# ============================================================
# Control-flow child-block dispatch
# ============================================================


def iter_child_blocks(
    stmt: Statement,
) -> list[tuple[list["Statement"], str, bool]]:
    """Return (body, context_label, is_loop) for every child block of a control-flow statement.

    Centralizes IfBlock/MenuBlock/RepeatBlock/ForeachBlock dispatch so that
    every tree-walker in the codebase can iterate child bodies without
    duplicating a 4-way isinstance chain.
    """
    results: list[tuple[list[Statement], str, bool]] = []
    if isinstance(stmt, IfBlock):
        results.append((stmt.then_body, "if_then", False))
        if stmt.else_body:
            results.append((stmt.else_body, "if_else", False))
    elif isinstance(stmt, MenuBlock):
        for case in stmt.cases:
            results.append((case.body, "menu_case", False))
    elif isinstance(stmt, RepeatBlock):
        results.append((stmt.body, "repeat", True))
    elif isinstance(stmt, ForeachBlock):
        results.append((stmt.body, "foreach", True))
    return results


# ============================================================
# Top-Level IR
# ============================================================


@dataclass
class ShortcutIR:
    """Complete intermediate representation of a shortcut."""

    name: str
    statements: list[Statement] = field(default_factory=list)

    def action_count(self) -> int:
        """Count total actions (recursive through control flow)."""
        return _count_actions(self.statements)

    def all_actions(self) -> list[ActionStatement]:
        """Flatten all actions (recursive through control flow)."""
        return _collect_actions(self.statements)

    def defined_variables(self) -> set[str]:
        """Return all variable names defined via SET or setvariable actions."""
        return _collect_defined_vars(self.statements)

    def __repr__(self) -> str:
        return f'ShortcutIR("{self.name}", {len(self.statements)} stmts, {self.action_count()} actions)'


# ============================================================
# Helpers
# ============================================================


def _count_actions(stmts: list[Statement]) -> int:
    count = 0
    for s in stmts:
        if isinstance(s, ActionStatement):
            count += 1
        elif isinstance(s, SetVariable):
            count += 1  # SET compiles to setvariable action
        for body, _ctx, _is_loop in iter_child_blocks(s):
            count += _count_actions(body)
    return count


def _collect_actions(stmts: list[Statement]) -> list[ActionStatement]:
    result = []
    for s in stmts:
        if isinstance(s, ActionStatement):
            result.append(s)
        for body, _ctx, _is_loop in iter_child_blocks(s):
            result.extend(_collect_actions(body))
    return result


def _collect_defined_vars(stmts: list[Statement]) -> set[str]:
    names = set()
    for s in stmts:
        if isinstance(s, SetVariable):
            names.add(s.var_name)
        elif isinstance(s, ActionStatement):
            # setvariable actions also define variables
            if s.action_name in ("setvariable", "is.workflow.actions.setvariable"):
                name_val = s.params.get("WFVariableName")
                if isinstance(name_val, StringValue):
                    names.add(name_val.value)
        for body, _ctx, _is_loop in iter_child_blocks(s):
            names |= _collect_defined_vars(body)
    return names
