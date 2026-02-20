"""
ShortcutForge Simulation Harness — Static Analysis Engine.

Seven analyses on ShortcutIR:
  1. Variable flow — set-before-use, branch coverage, loop scoping, unused vars
  2. Loop bound checking — repeat counts >1000
  3. Menu case completeness — duplicate labels, empty cases
  4. Dead code — statements after unconditional exits
  5. API endpoint validation — URL format, HTTP method consistency
  6. Type flow — track expected types through action pipeline
  7. Contract validation — 13 rules for API contracts, mapping, runtime risk, data flow

Returns SimulationReport with SimulationFinding items.
Non-blocking: findings are warnings/info, not hard errors.

Run: python3 scripts/simulation_harness.py [dsl_file]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from dsl_ir import (
    ActionStatement,
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
)

try:
    from contract_validator import ContractValidator

    _HAS_CONTRACT_VALIDATOR = True
except ImportError:
    _HAS_CONTRACT_VALIDATOR = False

__version__ = "1.0"


# ── Data Classes ─────────────────────────────────────────────────


class Severity(str, Enum):
    """Severity levels for simulation findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class FindingCategory(str, Enum):
    """Categories for simulation findings."""

    VARIABLE_FLOW = "variable_flow"
    LOOP_BOUND = "loop_bound"
    MENU_COMPLETENESS = "menu_completeness"
    DEAD_CODE = "dead_code"
    API_VALIDATION = "api_validation"
    TYPE_FLOW = "type_flow"
    CONTRACT = "contract"


@dataclass
class SimulationFinding:
    """A single finding from static analysis."""

    severity: Severity
    category: FindingCategory
    message: str
    line_number: int = 0
    suggestion: str = ""

    def __repr__(self) -> str:
        loc = f" (line {self.line_number})" if self.line_number else ""
        sug = f" → {self.suggestion}" if self.suggestion else ""
        return (
            f"[{self.severity.value}] {self.category.value}{loc}: {self.message}{sug}"
        )


@dataclass
class SimulationReport:
    """Complete simulation report."""

    findings: list[SimulationFinding] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.INFO)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0

    def findings_by_category(self, cat: FindingCategory) -> list[SimulationFinding]:
        return [f for f in self.findings if f.category == cat]

    def summary(self) -> str:
        return (
            f"SimulationReport: {self.error_count} errors, "
            f"{self.warning_count} warnings, {self.info_count} info"
        )


# ── Exit actions (terminate execution) ──────────────────────────

_EXIT_ACTIONS = frozenset({"exit", "nothing"})


# ── Helpers ──────────────────────────────────────────────────────


def _extract_var_refs(value: IRValue) -> set[str]:
    """Extract all variable names referenced in an IRValue."""
    refs: set[str] = set()
    if isinstance(value, VarRef):
        refs.add(value.name)
    elif isinstance(value, InterpolatedString):
        for part in value.parts:
            if isinstance(part, VarRef):
                refs.add(part.name)
    elif isinstance(value, DictLiteral):
        for _, v in value.entries:
            refs |= _extract_var_refs(v)
    elif isinstance(value, ListLiteral):
        for item in value.items:
            refs |= _extract_var_refs(item)
    elif isinstance(value, HeadersLiteral):
        for _, v in value.entries:
            refs |= _extract_var_refs(v)
    elif isinstance(value, QuantityLiteral):
        if isinstance(value.magnitude, VarRef):
            refs.add(value.magnitude.name)
    return refs


def _extract_all_var_refs_from_stmt(stmt: Statement) -> set[str]:
    """Extract all variable names referenced (used) in a statement."""
    refs: set[str] = set()
    if isinstance(stmt, ActionStatement):
        for v in stmt.params.values():
            refs |= _extract_var_refs(v)
    elif isinstance(stmt, SetVariable):
        refs |= _extract_var_refs(stmt.value)
    elif isinstance(stmt, IfBlock):
        if isinstance(stmt.target, VarRef):
            refs.add(stmt.target.name)
        if stmt.compare_value:
            refs |= _extract_var_refs(stmt.compare_value)
    elif isinstance(stmt, ForeachBlock):
        if isinstance(stmt.collection, VarRef):
            refs.add(stmt.collection.name)
    elif isinstance(stmt, RepeatBlock):
        refs |= _extract_var_refs(stmt.count)
    return refs


def _get_string_value(value: IRValue) -> str | None:
    """Extract string from StringValue or InterpolatedString (literal parts only)."""
    if isinstance(value, StringValue):
        return value.value
    if isinstance(value, InterpolatedString):
        parts = []
        for p in value.parts:
            if isinstance(p, StringValue):
                parts.append(p.value)
            else:
                parts.append("{...}")  # placeholder for dynamic parts
        return "".join(parts)
    return None


# ── Analysis 1: Variable Flow ───────────────────────────────────


def _analyze_variable_flow(ir: ShortcutIR, findings: list[SimulationFinding]) -> None:
    """Check variable set-before-use, branch coverage, loop scoping, unused vars.

    Tracks which variables are defined in each scope level.
    Detects:
      - Use of variable before it's defined (set-before-use)
      - Variables defined only in one branch of IF (branch coverage warning)
      - Variables defined in loop body used after loop (loop scoping warning)
      - Variables defined but never referenced (unused vars)
    """
    # Collect all definitions and all uses globally
    all_defs: dict[str, int] = {}  # var_name -> first def line
    all_uses: dict[str, int] = {}  # var_name -> first use line

    # Implicit variables always available
    always_available = {"Repeat_Item", "Repeat_Index", "ShortcutInput"}

    def walk_for_defs_and_uses(
        stmts: list[Statement],
        defined: set[str],
        scope_label: str = "top",
    ) -> set[str]:
        """Walk statements, tracking defined vars. Returns newly defined vars in this scope."""
        newly_defined: set[str] = set()

        for stmt in stmts:
            # Check uses first (before this stmt defines anything)
            used = _extract_all_var_refs_from_stmt(stmt)
            for var in used:
                if (
                    var not in defined
                    and var not in always_available
                    and var not in newly_defined
                ):
                    # Use before def
                    line = getattr(stmt, "line_number", 0)
                    findings.append(
                        SimulationFinding(
                            severity=Severity.WARNING,
                            category=FindingCategory.VARIABLE_FLOW,
                            message=f"Variable ${var} used before definition",
                            line_number=line,
                            suggestion=f"Add SET ${var} = ... before this point",
                        )
                    )
                if var not in all_uses:
                    all_uses[var] = getattr(stmt, "line_number", 0)

            # Track definitions
            if isinstance(stmt, SetVariable):
                defined.add(stmt.var_name)
                newly_defined.add(stmt.var_name)
                if stmt.var_name not in all_defs:
                    all_defs[stmt.var_name] = stmt.line_number
            elif isinstance(stmt, ActionStatement):
                if stmt.action_name in (
                    "setvariable",
                    "is.workflow.actions.setvariable",
                ):
                    name_val = stmt.params.get("WFVariableName")
                    if isinstance(name_val, StringValue):
                        defined.add(name_val.value)
                        newly_defined.add(name_val.value)
                        if name_val.value not in all_defs:
                            all_defs[name_val.value] = stmt.line_number

            # Recurse into control flow
            if isinstance(stmt, IfBlock):
                then_defs = walk_for_defs_and_uses(
                    stmt.then_body, set(defined) | newly_defined, "if-then"
                )
                else_defs: set[str] = set()
                if stmt.else_body:
                    else_defs = walk_for_defs_and_uses(
                        stmt.else_body, set(defined) | newly_defined, "if-else"
                    )

                # Vars defined in BOTH branches are safe
                both = then_defs & else_defs
                # Vars defined in only one branch: warning
                only_then = then_defs - else_defs
                only_else = else_defs - then_defs

                for var in only_then:
                    findings.append(
                        SimulationFinding(
                            severity=Severity.INFO,
                            category=FindingCategory.VARIABLE_FLOW,
                            message=f"Variable ${var} only defined in IF-then branch, not in ELSE",
                            line_number=stmt.line_number,
                            suggestion=f"Ensure ${var} is set in both branches if used after IF",
                        )
                    )
                for var in only_else:
                    findings.append(
                        SimulationFinding(
                            severity=Severity.INFO,
                            category=FindingCategory.VARIABLE_FLOW,
                            message=f"Variable ${var} only defined in ELSE branch, not in IF-then",
                            line_number=stmt.line_number,
                            suggestion=f"Ensure ${var} is set in both branches if used after IF",
                        )
                    )

                # Only both-branch defs are considered defined after the IF
                defined.update(both)
                newly_defined.update(both)

            elif isinstance(stmt, MenuBlock):
                case_defs_list = []
                for case in stmt.cases:
                    case_new = walk_for_defs_and_uses(
                        case.body, set(defined) | newly_defined, f"menu-{case.label}"
                    )
                    case_defs_list.append(case_new)
                # Only vars defined in ALL cases are safe after menu
                if case_defs_list:
                    all_cases = case_defs_list[0]
                    for cd in case_defs_list[1:]:
                        all_cases = all_cases & cd
                    defined.update(all_cases)
                    newly_defined.update(all_cases)

            elif isinstance(stmt, (RepeatBlock, ForeachBlock)):
                loop_defs = walk_for_defs_and_uses(
                    stmt.body, set(defined) | newly_defined, "loop"
                )
                # Loop-scoped vars: warn but still add (they will be defined at runtime)
                for var in loop_defs:
                    if var not in defined and var not in newly_defined:
                        findings.append(
                            SimulationFinding(
                                severity=Severity.INFO,
                                category=FindingCategory.VARIABLE_FLOW,
                                message=f"Variable ${var} defined inside loop body",
                                line_number=stmt.line_number,
                                suggestion=f"Value of ${var} after loop depends on last iteration",
                            )
                        )
                defined.update(loop_defs)
                newly_defined.update(loop_defs)

        return newly_defined

    walk_for_defs_and_uses(ir.statements, set())

    # Check for unused vars
    for var, def_line in all_defs.items():
        if var not in all_uses:
            findings.append(
                SimulationFinding(
                    severity=Severity.INFO,
                    category=FindingCategory.VARIABLE_FLOW,
                    message=f"Variable ${var} is defined but never used",
                    line_number=def_line,
                    suggestion="Remove unused variable or verify it's needed",
                )
            )


# ── Analysis 2: Loop Bound Checking ─────────────────────────────

_MAX_REASONABLE_REPEAT = 1000


def _analyze_loop_bounds(ir: ShortcutIR, findings: list[SimulationFinding]) -> None:
    """Flag repeat counts > 1000 as suspicious."""

    def walk(stmts: list[Statement]) -> None:
        for stmt in stmts:
            if isinstance(stmt, RepeatBlock):
                if isinstance(stmt.count, NumberValue):
                    if stmt.count.value > _MAX_REASONABLE_REPEAT:
                        findings.append(
                            SimulationFinding(
                                severity=Severity.WARNING,
                                category=FindingCategory.LOOP_BOUND,
                                message=f"REPEAT count {stmt.count.value} exceeds {_MAX_REASONABLE_REPEAT}",
                                line_number=stmt.line_number,
                                suggestion="Large repeat counts may cause performance issues or timeouts",
                            )
                        )
                    elif stmt.count.value <= 0:
                        findings.append(
                            SimulationFinding(
                                severity=Severity.WARNING,
                                category=FindingCategory.LOOP_BOUND,
                                message=f"REPEAT count {stmt.count.value} is zero or negative",
                                line_number=stmt.line_number,
                                suggestion="Loop body will never execute",
                            )
                        )
                walk(stmt.body)
            elif isinstance(stmt, ForeachBlock):
                walk(stmt.body)
            elif isinstance(stmt, IfBlock):
                walk(stmt.then_body)
                if stmt.else_body:
                    walk(stmt.else_body)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    walk(case.body)

    walk(ir.statements)


# ── Analysis 3: Menu Case Completeness ──────────────────────────


def _analyze_menu_completeness(
    ir: ShortcutIR, findings: list[SimulationFinding]
) -> None:
    """Check for duplicate menu labels and empty menu cases."""

    def walk(stmts: list[Statement]) -> None:
        for stmt in stmts:
            if isinstance(stmt, MenuBlock):
                # Check for duplicate labels
                labels: list[str] = []
                for case in stmt.cases:
                    label_lower = case.label.lower()
                    if label_lower in [lb.lower() for lb in labels]:
                        findings.append(
                            SimulationFinding(
                                severity=Severity.WARNING,
                                category=FindingCategory.MENU_COMPLETENESS,
                                message=f'Duplicate menu label "{case.label}"',
                                line_number=stmt.line_number,
                                suggestion="Use unique labels for each menu case",
                            )
                        )
                    labels.append(case.label)

                    # Check for empty cases (no statements, or only comments)
                    non_comment = [s for s in case.body if not isinstance(s, Comment)]
                    if not non_comment:
                        findings.append(
                            SimulationFinding(
                                severity=Severity.INFO,
                                category=FindingCategory.MENU_COMPLETENESS,
                                message=f'Empty menu case "{case.label}"',
                                line_number=stmt.line_number,
                                suggestion="Add actions to this menu case or remove it",
                            )
                        )

                # Check for single-case menu (usually a mistake)
                if len(stmt.cases) == 1:
                    findings.append(
                        SimulationFinding(
                            severity=Severity.INFO,
                            category=FindingCategory.MENU_COMPLETENESS,
                            message="Menu has only one case",
                            line_number=stmt.line_number,
                            suggestion="Consider using a simple alert instead of a single-option menu",
                        )
                    )

                # Recurse into cases
                for case in stmt.cases:
                    walk(case.body)

            elif isinstance(stmt, IfBlock):
                walk(stmt.then_body)
                if stmt.else_body:
                    walk(stmt.else_body)
            elif isinstance(stmt, RepeatBlock):
                walk(stmt.body)
            elif isinstance(stmt, ForeachBlock):
                walk(stmt.body)

    walk(ir.statements)


# ── Analysis 4: Dead Code Detection ─────────────────────────────


def _analyze_dead_code(ir: ShortcutIR, findings: list[SimulationFinding]) -> None:
    """Detect statements after unconditional exit actions."""

    def walk(stmts: list[Statement]) -> None:
        found_exit = False
        exit_line = 0

        for stmt in stmts:
            if found_exit and not isinstance(stmt, Comment):
                findings.append(
                    SimulationFinding(
                        severity=Severity.WARNING,
                        category=FindingCategory.DEAD_CODE,
                        message="Unreachable code after exit action",
                        line_number=getattr(stmt, "line_number", 0),
                        suggestion=f"This code won't execute (exit at line {exit_line})",
                    )
                )
                # Only report once per block
                break

            if isinstance(stmt, ActionStatement):
                if stmt.action_name.lower() in _EXIT_ACTIONS:
                    found_exit = True
                    exit_line = stmt.line_number

            # Recurse into sub-blocks
            if isinstance(stmt, IfBlock):
                walk(stmt.then_body)
                if stmt.else_body:
                    walk(stmt.else_body)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    walk(case.body)
            elif isinstance(stmt, RepeatBlock):
                walk(stmt.body)
            elif isinstance(stmt, ForeachBlock):
                walk(stmt.body)

    walk(ir.statements)


# ── Analysis 5: API Endpoint Validation ─────────────────────────

# Actions that take URLs
_URL_ACTIONS = frozenset(
    {
        "downloadurl",
        "getcontentsofurl",
        "openurl",
        "openxcallbackurl",
        "is.workflow.actions.downloadurl",
        "is.workflow.actions.url.expand",
        "is.workflow.actions.openurl",
    }
)

# HTTP methods
_VALID_HTTP_METHODS = frozenset(
    {
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
        "HEAD",
        "OPTIONS",
    }
)

# URL pattern (very basic)
_URL_PATTERN = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)


def _analyze_api_endpoints(ir: ShortcutIR, findings: list[SimulationFinding]) -> None:
    """Check URL format and HTTP method consistency."""

    def walk(stmts: list[Statement]) -> None:
        for stmt in stmts:
            if isinstance(stmt, ActionStatement):
                action_lower = stmt.action_name.lower()

                if action_lower in _URL_ACTIONS or "url" in action_lower:
                    # Check URL parameter
                    url_val = stmt.params.get("WFURL") or stmt.params.get("WFInput")
                    if url_val:
                        url_str = _get_string_value(url_val)
                        if url_str and "{...}" not in url_str:
                            # Pure static URL — validate format
                            if not _URL_PATTERN.match(url_str):
                                findings.append(
                                    SimulationFinding(
                                        severity=Severity.WARNING,
                                        category=FindingCategory.API_VALIDATION,
                                        message=f'URL "{url_str[:60]}" may be malformed',
                                        line_number=stmt.line_number,
                                        suggestion="Ensure URL starts with http:// or https://",
                                    )
                                )

                    # Check HTTP method
                    method_val = stmt.params.get("WFHTTPMethod")
                    if method_val:
                        method_str = _get_string_value(method_val)
                        if method_str and method_str.upper() not in _VALID_HTTP_METHODS:
                            findings.append(
                                SimulationFinding(
                                    severity=Severity.WARNING,
                                    category=FindingCategory.API_VALIDATION,
                                    message=f'Unknown HTTP method "{method_str}"',
                                    line_number=stmt.line_number,
                                    suggestion=f"Valid methods: {', '.join(sorted(_VALID_HTTP_METHODS))}",
                                )
                            )

                    # Check: POST/PUT without body is suspicious
                    method_val = stmt.params.get("WFHTTPMethod")
                    if method_val:
                        method_str = _get_string_value(method_val)
                        if method_str and method_str.upper() in ("POST", "PUT"):
                            body_val = stmt.params.get(
                                "WFHTTPBodyType"
                            ) or stmt.params.get("WFRequestVariable")
                            if not body_val:
                                findings.append(
                                    SimulationFinding(
                                        severity=Severity.INFO,
                                        category=FindingCategory.API_VALIDATION,
                                        message=f"{method_str.upper()} request without body specification",
                                        line_number=stmt.line_number,
                                        suggestion="POST/PUT typically include a request body",
                                    )
                                )

            # Recurse
            if isinstance(stmt, IfBlock):
                walk(stmt.then_body)
                if stmt.else_body:
                    walk(stmt.else_body)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    walk(case.body)
            elif isinstance(stmt, RepeatBlock):
                walk(stmt.body)
            elif isinstance(stmt, ForeachBlock):
                walk(stmt.body)

    walk(ir.statements)


# ── Analysis 6: Type Flow ───────────────────────────────────────

# Actions that produce known output types
_ACTION_OUTPUT_TYPES: dict[str, str] = {
    "ask": "text",
    "ask.input": "text",
    "askforinput": "text",
    "choosefromlist": "item",
    "choosefrommenu": "menu_selection",
    "downloadurl": "data",
    "getcontentsofurl": "data",
    "getclipboard": "text",
    "getcurrentdate": "date",
    "getdevicedetails": "text",
    "getbatterylevel": "number",
    "number": "number",
    "number.random": "number",
    "text": "text",
    "list": "list",
    "dictionary": "dictionary",
    "detect.dictionary": "dictionary",
    "detect.list": "list",
    "detect.number": "number",
    "detect.text": "text",
    "detect.url": "url",
    "detect.date": "date",
    "count": "number",
    "count.items": "number",
    "count.characters": "number",
    "math.calculate": "number",
    "getvalueforkey": "any",
    "setvalueforkey": "dictionary",
    "url": "url",
    "geturl": "url",
}

# Actions that expect specific input types
_ACTION_INPUT_EXPECTATIONS: dict[str, str] = {
    "detect.dictionary": "data",
    "detect.list": "data",
    "detect.number": "text",
    "detect.text": "data",
    "detect.url": "text",
    "detect.date": "text",
    "getvalueforkey": "dictionary",
    "setvalueforkey": "dictionary",
    "count": "list",
    "count.items": "list",
    "count.characters": "text",
    "math.calculate": "number",
}


def _analyze_type_flow(ir: ShortcutIR, findings: list[SimulationFinding]) -> None:
    """Track output types through the pipeline and flag mismatches.

    Uses @prev chain to propagate types between adjacent actions.
    """

    def walk(stmts: list[Statement], prev_type: str | None = None) -> str | None:
        """Walk statements tracking the type of @prev. Returns final prev type."""
        current_type = prev_type

        for stmt in stmts:
            if isinstance(stmt, ActionStatement):
                action_lower = stmt.action_name.lower()

                # Check if this action uses @prev and has type expectations
                uses_prev = False
                for v in stmt.params.values():
                    if isinstance(v, HandleRef) and v.kind == "prev":
                        uses_prev = True
                        break
                    if isinstance(v, InterpolatedString):
                        for p in v.parts:
                            if isinstance(p, HandleRef) and p.kind == "prev":
                                uses_prev = True
                                break

                # Also: some actions implicitly use @prev as input (when WFInput=@prev)
                input_val = stmt.params.get("WFInput")
                if isinstance(input_val, HandleRef) and input_val.kind == "prev":
                    uses_prev = True

                if (
                    uses_prev
                    and current_type
                    and action_lower in _ACTION_INPUT_EXPECTATIONS
                ):
                    expected = _ACTION_INPUT_EXPECTATIONS[action_lower]
                    if (
                        current_type != "any"
                        and expected != "any"
                        and current_type != expected
                    ):
                        findings.append(
                            SimulationFinding(
                                severity=Severity.INFO,
                                category=FindingCategory.TYPE_FLOW,
                                message=(
                                    f"Action '{stmt.action_name}' expects {expected} input "
                                    f"but previous output is {current_type}"
                                ),
                                line_number=stmt.line_number,
                                suggestion=f"Consider adding a conversion action (detect.{expected})",
                            )
                        )

                # Update current type based on what this action produces
                if action_lower in _ACTION_OUTPUT_TYPES:
                    current_type = _ACTION_OUTPUT_TYPES[action_lower]
                else:
                    # Unknown output type — reset to None
                    current_type = None

            elif isinstance(stmt, SetVariable):
                # SET doesn't change @prev
                pass

            elif isinstance(stmt, IfBlock):
                # @prev type after IF is ambiguous (depends on branch taken)
                walk(stmt.then_body, current_type)
                if stmt.else_body:
                    walk(stmt.else_body, current_type)
                current_type = None  # ambiguous after branching

            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    walk(case.body, current_type)
                current_type = None  # ambiguous

            elif isinstance(stmt, (RepeatBlock, ForeachBlock)):
                walk(stmt.body, current_type)
                current_type = None  # ambiguous after loop

        return current_type

    walk(ir.statements)


# ── Analysis 7: Contract Validation ─────────────────────────────

_SEVERITY_MAP = {
    "error": Severity.ERROR,
    "warning": Severity.WARNING,
    "info": Severity.INFO,
}


def _analyze_contracts(ir: ShortcutIR, findings: list[SimulationFinding]) -> None:
    """Run contract validation rules (13 rules across 4 categories).

    Delegates to ContractValidator and converts ContractFinding items
    to SimulationFinding items with FindingCategory.CONTRACT.
    """
    if not _HAS_CONTRACT_VALIDATOR:
        return

    cv = ContractValidator()
    report = cv.validate(ir)

    for f in report.findings:
        severity = _SEVERITY_MAP.get(f.severity, Severity.INFO)
        findings.append(
            SimulationFinding(
                severity=severity,
                category=FindingCategory.CONTRACT,
                message=f"[{f.rule_id}] {f.message}",
                line_number=f.line,
                suggestion=f.suggestion,
            )
        )


# ── Main Harness ────────────────────────────────────────────────


class SimulationHarness:
    """Run all static analyses on a ShortcutIR.

    Usage:
        harness = SimulationHarness()
        report = harness.analyze(ir)
        for finding in report.findings:
            print(finding)
    """

    def analyze(self, ir: ShortcutIR) -> SimulationReport:
        """Run all 7 analyses and return combined report."""
        findings: list[SimulationFinding] = []

        _analyze_variable_flow(ir, findings)
        _analyze_loop_bounds(ir, findings)
        _analyze_menu_completeness(ir, findings)
        _analyze_dead_code(ir, findings)
        _analyze_api_endpoints(ir, findings)
        _analyze_type_flow(ir, findings)
        _analyze_contracts(ir, findings)

        return SimulationReport(findings=findings)

    def analyze_selective(
        self,
        ir: ShortcutIR,
        categories: set[FindingCategory] | None = None,
    ) -> SimulationReport:
        """Run selected analyses only."""
        if categories is None:
            return self.analyze(ir)

        findings: list[SimulationFinding] = []

        if FindingCategory.VARIABLE_FLOW in categories:
            _analyze_variable_flow(ir, findings)
        if FindingCategory.LOOP_BOUND in categories:
            _analyze_loop_bounds(ir, findings)
        if FindingCategory.MENU_COMPLETENESS in categories:
            _analyze_menu_completeness(ir, findings)
        if FindingCategory.DEAD_CODE in categories:
            _analyze_dead_code(ir, findings)
        if FindingCategory.API_VALIDATION in categories:
            _analyze_api_endpoints(ir, findings)
        if FindingCategory.TYPE_FLOW in categories:
            _analyze_type_flow(ir, findings)
        if FindingCategory.CONTRACT in categories:
            _analyze_contracts(ir, findings)

        return SimulationReport(findings=findings)


# ── Convenience ──────────────────────────────────────────────────

_harness: SimulationHarness | None = None


def get_harness() -> SimulationHarness:
    """Get the global simulation harness (singleton)."""
    global _harness
    if _harness is None:
        _harness = SimulationHarness()
    return _harness


def simulate(ir: ShortcutIR) -> SimulationReport:
    """Run simulation on IR (convenience function)."""
    return get_harness().analyze(ir)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    _SCRIPT_DIR = Path(__file__).resolve().parent
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))

    from dsl_linter import lint_dsl
    from dsl_parser import parse_dsl

    if len(sys.argv) < 2:
        print("Usage: python simulation_harness.py <dsl_file>")
        sys.exit(1)

    dsl_path = Path(sys.argv[1])
    if not dsl_path.exists():
        print(f"Error: File not found: {dsl_path}")
        sys.exit(1)

    text = dsl_path.read_text()
    lint_result = lint_dsl(text)
    ir = parse_dsl(lint_result.text)

    harness = SimulationHarness()
    report = harness.analyze(ir)

    print(f"\nSimulation Report for: {dsl_path.name}")
    print(f"  {report.summary()}")
    print()

    if report.findings:
        for f in report.findings:
            print(f"  {f}")
    else:
        print("  No findings.")
