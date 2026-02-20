"""
ShortcutForge Contract Validator — Logical Error Detection.

Validates ShortcutIR sequences against 13 rules across 4 categories:
  1. API Contract Rules (4)   — network request patterns
  2. Mapping Completeness (3) — data structure usage
  3. Runtime Risk Rules (3)   — performance and correctness
  4. Data Flow Rules (3)      — variable scoping and reachability

Returns a ContractReport with ContractFinding items.
Non-blocking: findings are warnings/info/errors for review.

Usage:
    from contract_validator import ContractValidator
    validator = ContractValidator()
    report = validator.validate(ir)
    for f in report.findings:
        print(f)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dsl_ir import (
    ActionStatement,
    Comment,
    DictLiteral,
    ForeachBlock,
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

__version__ = "1.0"


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class ContractFinding:
    """A single finding from contract validation."""

    severity: str  # "error", "warning", "info"
    category: str  # "api", "map", "risk", "flow"
    rule_id: str  # e.g., "api.missing_error_check"
    message: str
    line: int  # line number where issue was found
    suggestion: str = ""

    def __repr__(self) -> str:
        loc = f" (line {self.line})" if self.line else ""
        sug = f" -> {self.suggestion}" if self.suggestion else ""
        return f"[{self.severity}] {self.category}/{self.rule_id}{loc}: {self.message}{sug}"


@dataclass
class ContractReport:
    """Complete contract validation report."""

    findings: list[ContractFinding] = field(default_factory=list)
    rules_checked: int = 13

    @property
    def errors(self) -> list[ContractFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[ContractFinding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def has_errors(self) -> bool:
        return any(f.severity == "error" for f in self.findings)

    def summary(self) -> str:
        errs = len(self.errors)
        warns = len(self.warnings)
        infos = sum(1 for f in self.findings if f.severity == "info")
        return (
            f"ContractReport: {errs} errors, {warns} warnings, "
            f"{infos} info ({self.rules_checked} rules checked)"
        )


# ── Helpers ──────────────────────────────────────────────────────


def _get_string_value(value: IRValue) -> str | None:
    """Extract a plain string from an IRValue, if possible."""
    if isinstance(value, StringValue):
        return value.value
    if isinstance(value, InterpolatedString):
        parts = []
        for p in value.parts:
            if isinstance(p, StringValue):
                parts.append(p.value)
            else:
                parts.append("{...}")
        return "".join(parts)
    return None


def _extract_var_refs_from_value(value: IRValue) -> set[str]:
    """Extract all $variable names referenced in an IRValue."""
    refs: set[str] = set()
    if isinstance(value, VarRef):
        refs.add(value.name)
    elif isinstance(value, InterpolatedString):
        for part in value.parts:
            if isinstance(part, VarRef):
                refs.add(part.name)
    elif isinstance(value, DictLiteral):
        for _, v in value.entries:
            refs |= _extract_var_refs_from_value(v)
    elif isinstance(value, ListLiteral):
        for item in value.items:
            refs |= _extract_var_refs_from_value(item)
    elif isinstance(value, HeadersLiteral):
        for _, v in value.entries:
            refs |= _extract_var_refs_from_value(v)
    elif isinstance(value, QuantityLiteral):
        if isinstance(value.magnitude, VarRef):
            refs.add(value.magnitude.name)
    return refs


def _extract_var_refs_from_stmt(stmt: Statement) -> set[str]:
    """Extract all $variable names used (referenced) in a statement."""
    refs: set[str] = set()
    if isinstance(stmt, ActionStatement):
        for v in stmt.params.values():
            refs |= _extract_var_refs_from_value(v)
    elif isinstance(stmt, SetVariable):
        refs |= _extract_var_refs_from_value(stmt.value)
    elif isinstance(stmt, IfBlock):
        if isinstance(stmt.target, VarRef):
            refs.add(stmt.target.name)
        if stmt.compare_value:
            refs |= _extract_var_refs_from_value(stmt.compare_value)
    elif isinstance(stmt, ForeachBlock):
        if isinstance(stmt.collection, VarRef):
            refs.add(stmt.collection.name)
    elif isinstance(stmt, RepeatBlock):
        if isinstance(stmt.count, VarRef):
            refs.add(stmt.count.name)
        elif hasattr(stmt.count, "__class__") and isinstance(
            stmt.count, IRValue.__args__ if hasattr(IRValue, "__args__") else ()
        ):
            refs |= _extract_var_refs_from_value(stmt.count)
    return refs


def _value_contains_string(value: IRValue, needle: str) -> bool:
    """Check if an IRValue contains a specific substring."""
    s = _get_string_value(value)
    return s is not None and needle in s


# ── Exit actions ─────────────────────────────────────────────────

_EXIT_ACTIONS = frozenset({"exit", "exitshortcut", "nothing", "stop"})


_iter_child_blocks = iter_child_blocks  # local alias for brevity


# ── Flatten Helper ───────────────────────────────────────────────

FlatEntry = tuple[Statement, int, str]  # (statement, depth, parent_type)


def _flatten_statements(
    stmts: list[Statement],
    depth: int = 0,
    parent_type: str = "top",
) -> list[FlatEntry]:
    """Walk the IR tree recursively and return flattened (statement, depth, parent_type) tuples."""
    result: list[FlatEntry] = []
    for stmt in stmts:
        result.append((stmt, depth, parent_type))
        if isinstance(stmt, IfBlock):
            result.extend(_flatten_statements(stmt.then_body, depth + 1, "if_then"))
            if stmt.else_body:
                result.extend(_flatten_statements(stmt.else_body, depth + 1, "if_else"))
        elif isinstance(stmt, MenuBlock):
            for case in stmt.cases:
                result.extend(_flatten_statements(case.body, depth + 1, "menu_case"))
        elif isinstance(stmt, RepeatBlock):
            result.extend(_flatten_statements(stmt.body, depth + 1, "repeat"))
        elif isinstance(stmt, ForeachBlock):
            result.extend(_flatten_statements(stmt.body, depth + 1, "foreach"))
    return result


# ── Contract Validator ───────────────────────────────────────────


class ContractValidator:
    """Validate ShortcutIR against 13 contract rules.

    Usage:
        validator = ContractValidator()
        report = validator.validate(ir)
    """

    def validate(self, ir: ShortcutIR) -> ContractReport:
        """Run all 13 contract rules and return a ContractReport."""
        findings: list[ContractFinding] = []
        flat = _flatten_statements(ir.statements)
        findings.extend(self._check_api_contracts(flat, ir.statements))
        findings.extend(self._check_mapping_completeness(flat, ir.statements))
        findings.extend(self._check_runtime_risks(flat, ir.statements))
        findings.extend(self._check_data_flow(flat, ir.statements))
        return ContractReport(findings=findings)

    # ── Category 1: API Contract Rules ───────────────────────────

    def _check_api_contracts(
        self, flat: list[FlatEntry], stmts: list[Statement]
    ) -> list[ContractFinding]:
        """Rules 1-4: API contract checks."""
        findings: list[ContractFinding] = []
        try:
            findings.extend(self._rule_api_missing_error_check(flat))
        except Exception:
            pass
        try:
            findings.extend(self._rule_api_missing_content_type(flat))
        except Exception:
            pass
        try:
            findings.extend(self._rule_api_json_parse_after_fetch(flat))
        except Exception:
            pass
        try:
            findings.extend(self._rule_api_url_not_variable(flat))
        except Exception:
            pass
        return findings

    def _rule_api_missing_error_check(
        self, flat: list[FlatEntry]
    ) -> list[ContractFinding]:
        """Rule 1: downloadurl without IF within next 3 statements."""
        findings: list[ContractFinding] = []
        for i, (stmt, depth, parent) in enumerate(flat):
            if not isinstance(stmt, ActionStatement):
                continue
            if stmt.action_name.lower() != "downloadurl":
                continue
            # Look ahead up to 3 statements in the flat list
            has_if = False
            for j in range(i + 1, min(i + 4, len(flat))):
                next_stmt = flat[j][0]
                if isinstance(next_stmt, IfBlock):
                    has_if = True
                    break
            if not has_if:
                findings.append(
                    ContractFinding(
                        severity="warning",
                        category="api",
                        rule_id="api.missing_error_check",
                        message="Network request without error handling",
                        line=stmt.line_number,
                        suggestion="Add an IF block after downloadurl to check for errors",
                    )
                )
        return findings

    def _rule_api_missing_content_type(
        self, flat: list[FlatEntry]
    ) -> list[ContractFinding]:
        """Rule 2: POST/PUT/PATCH with body type but no headers."""
        findings: list[ContractFinding] = []
        for stmt, depth, parent in flat:
            if not isinstance(stmt, ActionStatement):
                continue
            if stmt.action_name.lower() != "downloadurl":
                continue
            method_val = stmt.params.get("WFHTTPMethod")
            if not method_val:
                continue
            method_str = _get_string_value(method_val)
            if not method_str or method_str.upper() not in ("POST", "PUT", "PATCH"):
                continue
            body_type = stmt.params.get("WFHTTPBodyType")
            if not body_type:
                continue
            headers = stmt.params.get("WFHTTPHeaders")
            if not headers:
                findings.append(
                    ContractFinding(
                        severity="warning",
                        category="api",
                        rule_id="api.missing_content_type",
                        message="POST/PUT request without Content-Type header",
                        line=stmt.line_number,
                        suggestion="Add WFHTTPHeaders with Content-Type for POST/PUT requests",
                    )
                )
        return findings

    def _rule_api_json_parse_after_fetch(
        self, flat: list[FlatEntry]
    ) -> list[ContractFinding]:
        """Rule 3: downloadurl not followed by detect.dictionary or getvalueforkey."""
        findings: list[ContractFinding] = []
        for i, (stmt, depth, parent) in enumerate(flat):
            if not isinstance(stmt, ActionStatement):
                continue
            if stmt.action_name.lower() != "downloadurl":
                continue
            has_parse = False
            for j in range(i + 1, min(i + 4, len(flat))):
                next_stmt = flat[j][0]
                if isinstance(next_stmt, ActionStatement):
                    next_name = next_stmt.action_name.lower()
                    if next_name in ("detect.dictionary", "getvalueforkey"):
                        has_parse = True
                        break
            if not has_parse:
                findings.append(
                    ContractFinding(
                        severity="info",
                        category="api",
                        rule_id="api.json_parse_after_fetch",
                        message="Raw response used without JSON parsing",
                        line=stmt.line_number,
                        suggestion="Add detect.dictionary or getvalueforkey after downloadurl",
                    )
                )
        return findings

    def _rule_api_url_not_variable(
        self, flat: list[FlatEntry]
    ) -> list[ContractFinding]:
        """Rule 4: downloadurl or url action with hardcoded URL."""
        findings: list[ContractFinding] = []
        for stmt, depth, parent in flat:
            if not isinstance(stmt, ActionStatement):
                continue
            action_lower = stmt.action_name.lower()
            if action_lower not in ("downloadurl", "url"):
                continue
            # Check for hardcoded URL in params
            for param_name, param_val in stmt.params.items():
                if _value_contains_string(param_val, "http"):
                    findings.append(
                        ContractFinding(
                            severity="info",
                            category="api",
                            rule_id="api.url_not_variable",
                            message="Consider using variable for URL",
                            line=stmt.line_number,
                            suggestion="Store URL in a variable for easier maintenance",
                        )
                    )
                    break  # One finding per action
        return findings

    # ── Category 2: Mapping Completeness Rules ───────────────────

    def _check_mapping_completeness(
        self, flat: list[FlatEntry], stmts: list[Statement]
    ) -> list[ContractFinding]:
        """Rules 5-7: Mapping completeness checks."""
        findings: list[ContractFinding] = []
        try:
            findings.extend(self._rule_foreach_empty_body(stmts))
        except Exception:
            pass
        try:
            findings.extend(self._rule_dictionary_unused_keys(flat))
        except Exception:
            pass
        try:
            findings.extend(self._rule_set_never_used(flat, stmts))
        except Exception:
            pass
        return findings

    def _rule_foreach_empty_body(self, stmts: list[Statement]) -> list[ContractFinding]:
        """Rule 5: FOREACH with 0 or 1 body statements."""
        findings: list[ContractFinding] = []

        def walk(statements: list[Statement]) -> None:
            for stmt in statements:
                if isinstance(stmt, ForeachBlock):
                    non_comment_body = [
                        s for s in stmt.body if not isinstance(s, Comment)
                    ]
                    if len(non_comment_body) <= 1:
                        findings.append(
                            ContractFinding(
                                severity="warning",
                                category="map",
                                rule_id="map.foreach_empty_body",
                                message="FOREACH with trivial body",
                                line=stmt.line_number,
                                suggestion="FOREACH body should contain meaningful operations",
                            )
                        )
                for body, _ctx, _is_loop in _iter_child_blocks(stmt):
                    walk(body)

        walk(stmts)
        return findings

    def _rule_dictionary_unused_keys(
        self, flat: list[FlatEntry]
    ) -> list[ContractFinding]:
        """Rule 6: Dictionary keys set but never accessed."""
        findings: list[ContractFinding] = []
        keys_set: dict[str, int] = {}  # key_name -> line_number
        keys_accessed: set[str] = set()

        for stmt, depth, parent in flat:
            if not isinstance(stmt, ActionStatement):
                continue
            action_lower = stmt.action_name.lower()
            if action_lower == "setvalueforkey":
                key_val = stmt.params.get("WFDictionaryKey")
                if key_val:
                    key_str = _get_string_value(key_val)
                    if key_str and key_str not in keys_set:
                        keys_set[key_str] = stmt.line_number
            elif action_lower == "getvalueforkey":
                key_val = stmt.params.get("WFDictionaryKey")
                if key_val:
                    key_str = _get_string_value(key_val)
                    if key_str:
                        keys_accessed.add(key_str)

        unused = set(keys_set.keys()) - keys_accessed
        if unused:
            # Report at the line of the first unused key
            first_line = min(keys_set[k] for k in unused)
            findings.append(
                ContractFinding(
                    severity="info",
                    category="map",
                    rule_id="map.dictionary_unused_keys",
                    message=f"Dictionary has unused keys: {', '.join(sorted(unused))}",
                    line=first_line,
                    suggestion="Remove unused keys or add getvalueforkey calls",
                )
            )
        return findings

    def _rule_set_never_used(
        self, flat: list[FlatEntry], stmts: list[Statement]
    ) -> list[ContractFinding]:
        """Rule 7: SET $VarName but variable never referenced later."""
        findings: list[ContractFinding] = []

        # Collect all SET statements
        set_vars: dict[str, int] = {}  # var_name -> line_number
        for stmt, depth, parent in flat:
            if isinstance(stmt, SetVariable):
                var_name = stmt.var_name
                # Skip internal/macro-generated variables
                if var_name.startswith("__"):
                    continue
                if var_name not in set_vars:
                    set_vars[var_name] = stmt.line_number

        if not set_vars:
            return findings

        # Collect all variable references across the full tree
        all_refs: set[str] = set()
        self._collect_all_refs(stmts, all_refs)

        for var_name, line in set_vars.items():
            if var_name not in all_refs:
                findings.append(
                    ContractFinding(
                        severity="warning",
                        category="map",
                        rule_id="map.set_never_used",
                        message=f"Variable '${var_name}' set but never used",
                        line=line,
                        suggestion=f"Remove SET ${var_name} or use ${var_name} in a subsequent action",
                    )
                )
        return findings

    def _collect_all_refs(self, stmts: list[Statement], refs: set[str]) -> None:
        """Walk the full statement tree collecting all variable references."""
        for stmt in stmts:
            # Collect refs from the statement itself
            refs |= _extract_var_refs_from_stmt(stmt)
            # Recurse into child blocks
            for body, _ctx, _is_loop in _iter_child_blocks(stmt):
                self._collect_all_refs(body, refs)

    # ── Category 3: Runtime Risk Rules ───────────────────────────

    def _check_runtime_risks(
        self, flat: list[FlatEntry], stmts: list[Statement]
    ) -> list[ContractFinding]:
        """Rules 8-10: Runtime risk checks."""
        findings: list[ContractFinding] = []
        try:
            findings.extend(self._rule_infinite_repeat(stmts))
        except Exception:
            pass
        try:
            findings.extend(self._rule_nested_network(stmts))
        except Exception:
            pass
        try:
            findings.extend(self._rule_menu_duplicate_labels(stmts))
        except Exception:
            pass
        return findings

    def _rule_infinite_repeat(self, stmts: list[Statement]) -> list[ContractFinding]:
        """Rule 8: REPEAT with count > 1000."""
        findings: list[ContractFinding] = []

        def walk(statements: list[Statement]) -> None:
            for stmt in statements:
                if isinstance(stmt, RepeatBlock) and isinstance(stmt.count, NumberValue):
                    if stmt.count.value > 1000:
                        findings.append(
                            ContractFinding(
                                severity="warning",
                                category="risk",
                                rule_id="risk.infinite_repeat",
                                message="Repeat count > 1000 may cause performance issues",
                                line=stmt.line_number,
                                suggestion="Consider reducing repeat count or using a different approach",
                            )
                        )
                for body, _ctx, _is_loop in _iter_child_blocks(stmt):
                    walk(body)

        walk(stmts)
        return findings

    def _rule_nested_network(self, stmts: list[Statement]) -> list[ContractFinding]:
        """Rule 9: downloadurl inside a FOREACH or REPEAT."""
        findings: list[ContractFinding] = []

        def walk(statements: list[Statement], in_loop: bool) -> None:
            for stmt in statements:
                if isinstance(stmt, ActionStatement):
                    if stmt.action_name.lower() == "downloadurl" and in_loop:
                        findings.append(
                            ContractFinding(
                                severity="warning",
                                category="risk",
                                rule_id="risk.nested_network",
                                message="Network request in loop -- consider batching",
                                line=stmt.line_number,
                                suggestion="Move network requests outside the loop or batch them",
                            )
                        )
                for body, _ctx, is_loop_child in _iter_child_blocks(stmt):
                    walk(body, in_loop or is_loop_child)

        walk(stmts, False)
        return findings

    def _rule_menu_duplicate_labels(
        self, stmts: list[Statement]
    ) -> list[ContractFinding]:
        """Rule 10: MENU with duplicate case labels."""
        findings: list[ContractFinding] = []

        def walk(statements: list[Statement]) -> None:
            for stmt in statements:
                if isinstance(stmt, MenuBlock):
                    labels_seen: list[str] = []
                    for case in stmt.cases:
                        if case.label in labels_seen:
                            findings.append(
                                ContractFinding(
                                    severity="error",
                                    category="risk",
                                    rule_id="risk.menu_duplicate_labels",
                                    message=f"Duplicate menu labels: '{case.label}'",
                                    line=stmt.line_number,
                                    suggestion="Use unique labels for each menu case",
                                )
                            )
                        else:
                            labels_seen.append(case.label)
                for body, _ctx, _is_loop in _iter_child_blocks(stmt):
                    walk(body)

        walk(stmts)
        return findings

    # ── Category 4: Data Flow Rules ──────────────────────────────

    def _check_data_flow(
        self, flat: list[FlatEntry], stmts: list[Statement]
    ) -> list[ContractFinding]:
        """Rules 11-13: Data flow checks."""
        findings: list[ContractFinding] = []
        try:
            findings.extend(self._rule_use_before_set(stmts))
        except Exception:
            pass
        try:
            findings.extend(self._rule_shadow_in_loop(stmts))
        except Exception:
            pass
        try:
            findings.extend(self._rule_dead_code_after_exit(stmts))
        except Exception:
            pass
        return findings

    # Magic variables that are always available
    _MAGIC_VARS = frozenset({"prev", "item", "input", "shortcutinput", "index", "date"})

    def _rule_use_before_set(self, stmts: list[Statement]) -> list[ContractFinding]:
        """Rule 11: Variable $X used before any SET $X."""
        findings: list[ContractFinding] = []
        defined: set[str] = set()

        def walk(statements: list[Statement], local_defined: set[str]) -> None:
            for stmt in statements:
                # Check uses first
                refs = _extract_var_refs_from_stmt(stmt)
                for var in refs:
                    if var not in local_defined and var.lower() not in self._MAGIC_VARS:
                        findings.append(
                            ContractFinding(
                                severity="error",
                                category="flow",
                                rule_id="flow.use_before_set",
                                message=f"Variable '${var}' used before being set",
                                line=getattr(stmt, "line_number", 0),
                                suggestion=f"Add SET ${var} = ... before this usage",
                            )
                        )
                        # Add to defined to avoid repeat findings for same var
                        local_defined.add(var)

                # Track definitions
                if isinstance(stmt, SetVariable):
                    local_defined.add(stmt.var_name)

                # Recurse into control flow (copy scope for each branch)
                for body, _ctx, _is_loop in _iter_child_blocks(stmt):
                    walk(body, set(local_defined))

        walk(stmts, defined)
        return findings

    def _rule_shadow_in_loop(self, stmts: list[Statement]) -> list[ContractFinding]:
        """Rule 12: SET $X inside FOREACH/REPEAT where $X already SET in outer scope."""
        findings: list[ContractFinding] = []

        def walk(
            statements: list[Statement], outer_vars: set[str], in_loop: bool
        ) -> None:
            for stmt in statements:
                if isinstance(stmt, SetVariable):
                    if in_loop and stmt.var_name in outer_vars:
                        findings.append(
                            ContractFinding(
                                severity="warning",
                                category="flow",
                                rule_id="flow.shadow_in_loop",
                                message=f"Variable '${stmt.var_name}' in loop shadows outer scope variable",
                                line=stmt.line_number,
                                suggestion="Use a different variable name inside the loop",
                            )
                        )
                    outer_vars.add(stmt.var_name)

                for body, _ctx, is_loop_child in _iter_child_blocks(stmt):
                    walk(body, set(outer_vars), in_loop or is_loop_child)

        walk(stmts, set(), False)
        return findings

    def _rule_dead_code_after_exit(
        self, stmts: list[Statement]
    ) -> list[ContractFinding]:
        """Rule 13: Statements after exitshortcut/stop in the same block."""
        findings: list[ContractFinding] = []

        def walk(statements: list[Statement]) -> None:
            found_exit = False
            exit_line = 0
            for stmt in statements:
                if found_exit and not isinstance(stmt, Comment):
                    findings.append(
                        ContractFinding(
                            severity="warning",
                            category="flow",
                            rule_id="flow.dead_code_after_exit",
                            message="Unreachable code after exit",
                            line=getattr(stmt, "line_number", 0),
                            suggestion=f"Remove unreachable code or the exit at line {exit_line}",
                        )
                    )
                    break  # Only report once per block

                if isinstance(stmt, ActionStatement):
                    if stmt.action_name.lower() in _EXIT_ACTIONS:
                        found_exit = True
                        exit_line = stmt.line_number

                for body, _ctx, _is_loop in _iter_child_blocks(stmt):
                    walk(body)

        walk(stmts)
        return findings


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dsl_linter import lint_dsl
    from dsl_parser import parse_dsl

    if len(sys.argv) < 2:
        print("Usage: python contract_validator.py <dsl_file>")
        sys.exit(1)

    dsl_path = Path(sys.argv[1])
    if not dsl_path.exists():
        print(f"Error: File not found: {dsl_path}")
        sys.exit(1)

    text = dsl_path.read_text()
    lint_result = lint_dsl(text)
    ir = parse_dsl(lint_result.text)

    validator = ContractValidator()
    report = validator.validate(ir)

    print(f"\nContract Validation Report for: {dsl_path.name}")
    print(f"  {report.summary()}")
    print()

    if report.findings:
        for f in report.findings:
            print(f"  {f}")
    else:
        print("  No findings.")
