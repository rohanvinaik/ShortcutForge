"""
ShortcutDSL Semantic Validator.

Validates a ShortcutIR against:
  - action_catalog.json: are action names valid? are param names observed?
  - param_schemas.json: do wrapping modes make sense?
  - Variable references: are variables defined before use?
  - Handle references: is @prev valid here? (has preceding action)

Supports two modes:
  - strict=True (default): Unknown actions with known vendor prefix → error.
    Measures true model quality (only validates fully-known actions).
  - strict=False: Unknown actions with known vendor prefix → warning.
    Measures user utility (may work at runtime despite missing schema).

Tiered resolution:
  - Tier 1: Direct resolution via resolve_action() → pass
  - Tier 2: Canonical_map / hallucination alias → warning (alias_rewrite)
  - Tier 3: Known vendor prefix → error (strict) or warning (permissive)
  - Tier 4: Unknown vendor / no dots → error always

Returns structured errors that can be fed back to the generator for retry.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

BASE_DIR = Path(__file__).resolve().parent.parent
CATALOG_PATH = BASE_DIR / "references" / "action_catalog.json"
SCHEMAS_PATH = BASE_DIR / "references" / "param_schemas.json"


@dataclass
class ValidationError:
    """A structured validation error."""
    severity: str  # "error" or "warning"
    category: str  # "unknown_action", "unknown_param", "undefined_variable", etc.
    message: str
    action_name: str = ""
    param_name: str = ""
    line_number: int = 0

    def __str__(self) -> str:
        loc = f" (line {self.line_number})" if self.line_number else ""
        return f"[{self.severity}] {self.category}: {self.message}{loc}"


@dataclass
class ValidationResult:
    """Result of validating a ShortcutIR."""
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def all_issues(self) -> list[ValidationError]:
        return self.errors + self.warnings

    def summary(self) -> str:
        if self.is_valid and not self.warnings:
            return "Valid (no errors, no warnings)"
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        return ", ".join(parts)

    def __str__(self) -> str:
        lines = [self.summary()]
        for e in self.errors:
            lines.append(f"  {e}")
        for w in self.warnings:
            lines.append(f"  {w}")
        return "\n".join(lines)


_VENDOR_PREFIX_RE = re.compile(r"^(is\.workflow|com\.[a-z0-9-]+)\.")

# Manual overrides for non-standard TLD patterns not in catalog
_MANUAL_VENDOR_PREFIXES = frozenset({
    "company.thebrowser.",
    "codes.rambo.",
    "dk.simonbs.",
    "fm.overcast.",
    "fyi.lunar.",
    "net.shinyfrog.",
    "ai.perplexity.",
})


def _derive_vendor_prefixes(actions: dict) -> frozenset:
    """Auto-derive known vendor prefixes from catalog action identifiers.

    Extracts patterns like 'is.workflow.' and 'com.apple.' from the full
    action identifiers. Also includes manual overrides for non-standard
    TLD patterns (e.g., 'codes.rambo.*') that don't match the regex.
    """
    prefixes = set()
    for identifier in actions:
        m = _VENDOR_PREFIX_RE.match(identifier)
        if m:
            prefixes.add(m.group(1) + ".")
    prefixes.update(_MANUAL_VENDOR_PREFIXES)
    return frozenset(prefixes)


def _value_has_handle(value: IRValue) -> bool:
    """Recursively check if a value contains any HandleRef."""
    if isinstance(value, HandleRef):
        return True
    if isinstance(value, VarRef):
        return False
    if isinstance(value, (StringValue, NumberValue, BoolValue)):
        return False
    if isinstance(value, InterpolatedString):
        return any(isinstance(p, HandleRef) for p in value.parts)
    if isinstance(value, DictLiteral):
        return any(_value_has_handle(v) for _, v in value.entries)
    if isinstance(value, ListLiteral):
        return any(_value_has_handle(item) for item in value.items)
    if isinstance(value, HeadersLiteral):
        return any(_value_has_handle(v) for _, v in value.entries)
    if isinstance(value, QuantityLiteral):
        return isinstance(value.magnitude, HandleRef)
    return False


class SemanticValidator:
    """Validates ShortcutIR against the action catalog and param schemas.

    Supports strict and permissive modes:
      - strict=True (default): Unknown actions with known vendor prefix → error
      - strict=False: Unknown actions with known vendor prefix → warning
    """

    def __init__(
        self,
        catalog_path: Path = CATALOG_PATH,
        schemas_path: Path = SCHEMAS_PATH,
    ):
        with catalog_path.open() as f:
            catalog_data = json.load(f)

        self._canonical_map: dict[str, str] = catalog_data.get("_meta", {}).get("canonical_map", {})
        self._actions: dict[str, dict] = catalog_data.get("actions", {})

        # Auto-derive known vendor prefixes from catalog
        self._known_vendor_prefixes = _derive_vendor_prefixes(self._actions)

        # Build reverse map: full identifier -> action data
        self._action_lookup: dict[str, dict] = {}
        for identifier, action in self._actions.items():
            self._action_lookup[identifier] = action

        # Also allow lookup by short name
        for short, full in self._canonical_map.items():
            if full in self._action_lookup:
                self._action_lookup[short] = self._action_lookup[full]

        with schemas_path.open() as f:
            self._schemas: dict[str, dict] = json.load(f)

    def resolve_action(self, name: str) -> Optional[str]:
        """Resolve an action name to its full identifier. Returns None if unknown."""
        # Direct match on full identifier
        if name in self._actions:
            return name

        # Check canonical map
        if name in self._canonical_map:
            return self._canonical_map[name]

        # Try with is.workflow.actions. prefix
        prefixed = f"is.workflow.actions.{name}"
        if prefixed in self._actions:
            return prefixed

        return None

    def get_observed_params(self, identifier: str) -> set[str]:
        """Get the set of observed parameter names for an action."""
        action = self._actions.get(identifier, {})
        return set(action.get("observed_params", {}).keys())

    def get_param_schema(self, identifier: str, param_name: str) -> Optional[dict]:
        """Get the wrapping schema for a specific param of an action."""
        action_schema = self._schemas.get(identifier, {})
        return action_schema.get(param_name)

    def _has_known_vendor_prefix(self, name: str) -> bool:
        """Check if an action name starts with a known vendor prefix."""
        for prefix in self._known_vendor_prefixes:
            if name.startswith(prefix):
                return True
        return False

    def _resolve_action_tiered(
        self,
        name: str,
        strict: bool,
    ) -> tuple[str, Optional[str], Optional[str]]:
        """Resolve an action name using tiered resolution.

        Returns (tier, resolved_identifier_or_None, error_message_or_None):
          - tier: "t1_direct", "t2_alias", "t3_vendor_prefix", "t4_unknown"
          - resolved: the canonical identifier if resolved, None if failed
          - message: error/warning message if relevant
        """
        # Tier 1: Direct resolution
        identifier = self.resolve_action(name)
        if identifier is not None:
            return "t1_direct", identifier, None

        # Tier 2: Canonical map / hallucination alias
        # (Already checked in resolve_action via canonical_map)

        # Tier 3: Known vendor prefix → the action is from a known vendor
        # but not in our catalog (could be a valid third-party intent)
        if self._has_known_vendor_prefix(name):
            msg = (
                f"Action '{name}' has a known vendor prefix but is not in the catalog. "
                f"It may be a valid third-party intent not yet cataloged."
            )
            if strict:
                return "t3_vendor_prefix", None, msg
            else:
                return "t3_vendor_prefix", name, msg

        # Tier 4: Unknown — no known vendor prefix, likely hallucinated
        msg = f"Unknown action: '{name}'"
        return "t4_unknown", None, msg

    def validate(self, ir: ShortcutIR, strict: bool = True) -> ValidationResult:
        """Validate a ShortcutIR. Returns ValidationResult with errors and warnings.

        Args:
            ir: The ShortcutIR to validate.
            strict: If True (default), unknown actions with known vendor prefix
                are treated as errors. If False, they become warnings.
        """
        result = ValidationResult()
        ctx = _ValidationContext()

        self._validate_statements(ir.statements, result, ctx, strict)

        return result

    def _validate_statements(
        self,
        stmts: list[Statement],
        result: ValidationResult,
        ctx: _ValidationContext,
        strict: bool = True,
    ) -> None:
        for stmt in stmts:
            if isinstance(stmt, ActionStatement):
                self._validate_action(stmt, result, ctx, strict)
                ctx.has_preceding_action = True
            elif isinstance(stmt, SetVariable):
                self._validate_set_variable(stmt, result, ctx)
                ctx.has_preceding_action = True  # SET compiles to setvariable action
            elif isinstance(stmt, IfBlock):
                self._validate_if_block(stmt, result, ctx, strict)
            elif isinstance(stmt, MenuBlock):
                self._validate_menu_block(stmt, result, ctx, strict)
            elif isinstance(stmt, RepeatBlock):
                self._validate_repeat_block(stmt, result, ctx, strict)
            elif isinstance(stmt, ForeachBlock):
                self._validate_foreach_block(stmt, result, ctx, strict)
            elif isinstance(stmt, Comment):
                pass  # Comments are always valid

    def _validate_action(
        self,
        stmt: ActionStatement,
        result: ValidationResult,
        ctx: _ValidationContext,
        strict: bool = True,
    ) -> None:
        # 1. Tiered action resolution
        tier, identifier, msg = self._resolve_action_tiered(stmt.action_name, strict)

        if tier == "t4_unknown":
            # Always an error
            result.errors.append(ValidationError(
                severity="error",
                category="unknown_action",
                message=msg or f"Unknown action: '{stmt.action_name}'",
                action_name=stmt.action_name,
                line_number=stmt.line_number,
            ))
            return

        if tier == "t3_vendor_prefix":
            if strict:
                # Error in strict mode
                result.errors.append(ValidationError(
                    severity="error",
                    category="unknown_action",
                    message=msg or f"Unknown action: '{stmt.action_name}'",
                    action_name=stmt.action_name,
                    line_number=stmt.line_number,
                ))
                return
            else:
                # Warning in permissive mode — pass through
                result.warnings.append(ValidationError(
                    severity="warning",
                    category="vendor_prefix_unknown",
                    message=msg or f"Action '{stmt.action_name}' has known vendor prefix but is not in catalog",
                    action_name=stmt.action_name,
                    line_number=stmt.line_number,
                ))
                # Check for compiler risk: unknown action with handle params
                for param_name, param_value in stmt.params.items():
                    if _value_has_handle(param_value):
                        result.warnings.append(ValidationError(
                            severity="warning",
                            category="compiler_risk",
                            message=(
                                f"Action '{stmt.action_name}' has no schema; "
                                f"param '{param_name}' uses handle ref. Compiler defaults "
                                f"to WFTextTokenAttachment wrapping — may fail at runtime."
                            ),
                            action_name=stmt.action_name,
                            param_name=param_name,
                            line_number=stmt.line_number,
                        ))
                # Validate value refs even for unresolved actions (permissive mode)
                for param_name, param_value in stmt.params.items():
                    self._validate_value_refs(param_value, result, ctx, stmt.action_name, param_name, stmt.line_number)
                return

        if tier == "t2_alias" and identifier:
            result.warnings.append(ValidationError(
                severity="warning",
                category="alias_rewrite",
                message=f"Action '{stmt.action_name}' resolved via alias to '{identifier}'",
                action_name=stmt.action_name,
                line_number=stmt.line_number,
            ))

        # Tier 1 or Tier 2 resolved — validate params normally
        if identifier is None:
            return  # shouldn't happen for t1/t2 but guard

        # 2. Check parameter names
        observed = self.get_observed_params(identifier)
        for param_name, param_value in stmt.params.items():
            if observed and param_name not in observed:
                # Warn, don't error — the catalog may not have seen every param
                result.warnings.append(ValidationError(
                    severity="warning",
                    category="unknown_param",
                    message=f"Parameter '{param_name}' not observed for action '{stmt.action_name}' (known: {sorted(observed)[:5]}...)",
                    action_name=stmt.action_name,
                    param_name=param_name,
                    line_number=stmt.line_number,
                ))

            # 3. Check value references
            self._validate_value_refs(param_value, result, ctx, stmt.action_name, param_name, stmt.line_number)

    def _validate_set_variable(
        self,
        stmt: SetVariable,
        result: ValidationResult,
        ctx: _ValidationContext,
    ) -> None:
        # Check the value references
        self._validate_value_refs(stmt.value, result, ctx, "SET", "", stmt.line_number)

        # Register the variable as defined
        ctx.defined_variables.add(stmt.var_name)

    def _validate_if_block(
        self,
        stmt: IfBlock,
        result: ValidationResult,
        ctx: _ValidationContext,
        strict: bool = True,
    ) -> None:
        # Validate target reference
        self._validate_ref(stmt.target, result, ctx, "IF", stmt.line_number)

        # Validate compare value if present
        if stmt.compare_value:
            self._validate_value_refs(stmt.compare_value, result, ctx, "IF", "", stmt.line_number)

        # Scope semantics: In Apple Shortcuts, SET creates global variables
        # that persist after the block. So variables defined in IF branches
        # are accessible in ELSE and after ENDIF. We model this by:
        # 1. IF and ELSE get independent child contexts (no cross-branch leaking)
        # 2. Variables defined in EITHER branch merge back to parent scope
        #    (since Shortcuts variables are globally scoped)

        # Validate then body
        then_ctx = ctx.child()
        self._validate_statements(stmt.then_body, result, then_ctx, strict)

        # Validate else body
        if stmt.else_body:
            else_ctx = ctx.child()
            self._validate_statements(stmt.else_body, result, else_ctx, strict)
            # Merge: variables from either branch become available after ENDIF
            ctx.defined_variables |= then_ctx.defined_variables
            ctx.defined_variables |= else_ctx.defined_variables
        else:
            ctx.defined_variables |= then_ctx.defined_variables

        ctx.has_preceding_action = True  # IF block itself contains actions

    def _validate_menu_block(
        self,
        stmt: MenuBlock,
        result: ValidationResult,
        ctx: _ValidationContext,
        strict: bool = True,
    ) -> None:
        if len(stmt.cases) == 0:
            result.errors.append(ValidationError(
                severity="error",
                category="empty_menu",
                message="Menu block has no cases",
                line_number=stmt.line_number,
            ))

        # Each case gets an independent scope; variables merge back after ENDMENU
        for case in stmt.cases:
            case_ctx = ctx.child()
            self._validate_statements(case.body, result, case_ctx, strict)
            ctx.defined_variables |= case_ctx.defined_variables

        ctx.has_preceding_action = True  # Menu action itself is an action

    def _validate_repeat_block(
        self,
        stmt: RepeatBlock,
        result: ValidationResult,
        ctx: _ValidationContext,
        strict: bool = True,
    ) -> None:
        # Validate count reference
        if isinstance(stmt.count, (VarRef, HandleRef)):
            self._validate_ref(stmt.count, result, ctx, "REPEAT", stmt.line_number)

        # Validate body — Repeat_Index and Repeat_Item are implicitly defined
        body_ctx = ctx.child()
        body_ctx.in_repeat = True
        body_ctx.defined_variables.update(_IMPLICIT_LOOP_VARS)
        self._validate_statements(stmt.body, result, body_ctx, strict)

        ctx.has_preceding_action = True  # Repeat block is an action

    def _validate_foreach_block(
        self,
        stmt: ForeachBlock,
        result: ValidationResult,
        ctx: _ValidationContext,
        strict: bool = True,
    ) -> None:
        # Validate collection reference
        self._validate_ref(stmt.collection, result, ctx, "FOREACH", stmt.line_number)

        # Validate body — @item and @index are valid inside
        # Repeat_Item, Repeat_Index also implicitly defined
        body_ctx = ctx.child()
        body_ctx.in_foreach = True
        body_ctx.defined_variables.update(_IMPLICIT_LOOP_VARS)
        self._validate_statements(stmt.body, result, body_ctx, strict)

        ctx.has_preceding_action = True

    def _validate_value_refs(
        self,
        value: IRValue,
        result: ValidationResult,
        ctx: _ValidationContext,
        action_name: str,
        param_name: str,
        line_number: int,
    ) -> None:
        """Validate references within a value."""
        if isinstance(value, VarRef):
            self._validate_ref(value, result, ctx, action_name, line_number)
        elif isinstance(value, HandleRef):
            self._validate_ref(value, result, ctx, action_name, line_number)
        elif isinstance(value, InterpolatedString):
            for part in value.parts:
                if isinstance(part, (VarRef, HandleRef)):
                    self._validate_ref(part, result, ctx, action_name, line_number)
        elif isinstance(value, DictLiteral):
            for _key, val in value.entries:
                self._validate_value_refs(val, result, ctx, action_name, param_name, line_number)
        elif isinstance(value, ListLiteral):
            for item in value.items:
                self._validate_value_refs(item, result, ctx, action_name, param_name, line_number)
        elif isinstance(value, HeadersLiteral):
            for _key, val in value.entries:
                self._validate_value_refs(val, result, ctx, action_name, param_name, line_number)
        # StringValue, NumberValue, BoolValue, QuantityLiteral — no refs to validate

    def _validate_ref(
        self,
        ref: VarRef | HandleRef,
        result: ValidationResult,
        ctx: _ValidationContext,
        action_name: str,
        line_number: int,
    ) -> None:
        """Validate a single reference."""
        if isinstance(ref, VarRef):
            if ref.name not in ctx.defined_variables:
                # Downgrade to warning (not error) for variables that could be
                # implicitly defined by the Shortcuts runtime (menu selections,
                # Shortcut Input, etc.)
                result.warnings.append(ValidationError(
                    severity="warning",
                    category="undefined_variable",
                    message=f"Variable '${ref.name}' used before definition",
                    action_name=action_name,
                    line_number=line_number,
                ))
        elif isinstance(ref, HandleRef):
            if ref.kind == "prev":
                if not ctx.has_preceding_action:
                    result.errors.append(ValidationError(
                        severity="error",
                        category="invalid_handle",
                        message="@prev used without a preceding action",
                        action_name=action_name,
                        line_number=line_number,
                    ))
            elif ref.kind == "item":
                if not ctx.in_foreach:
                    result.warnings.append(ValidationError(
                        severity="warning",
                        category="handle_context",
                        message="@item used outside FOREACH block",
                        action_name=action_name,
                        line_number=line_number,
                    ))
            elif ref.kind == "index":
                if not ctx.in_foreach and not ctx.in_repeat:
                    result.warnings.append(ValidationError(
                        severity="warning",
                        category="handle_context",
                        message="@index used outside loop block",
                        action_name=action_name,
                        line_number=line_number,
                    ))
            elif ref.kind in ("input", "date"):
                pass  # Always valid
            # Named handles are custom — always allowed


# Implicitly-defined variables inside loop/menu blocks
_IMPLICIT_LOOP_VARS = {
    "Repeat_Item", "Repeat_Index",
    # Common variants from the reverse compiler
    "Repeat Item", "Repeat Index",
}


@dataclass
class _ValidationContext:
    """Tracks validation state through nested scopes."""
    defined_variables: set[str] = field(default_factory=set)
    has_preceding_action: bool = False
    in_foreach: bool = False
    in_repeat: bool = False

    def child(self) -> _ValidationContext:
        """Create a child context that inherits defined variables."""
        return _ValidationContext(
            defined_variables=set(self.defined_variables),  # Copy
            has_preceding_action=self.has_preceding_action,
            in_foreach=self.in_foreach,
            in_repeat=self.in_repeat,
        )


# ============================================================
# Domain Validation Layer (Phase 2)
# ============================================================


class DomainValidationLayer:
    """Domain-aware validation rules that catch logical errors before runtime.

    Runs after core SemanticValidator. Uses domain profile data to check:
      1. HealthKit sample type ↔ unit compatibility (Caffeine must use mg, not mcg)
      2. HealthKit value range sanity warnings
      3. Missing error handling after downloadurl
      4. Unreachable code detection (statements after unconditional exits)

    Domain data is loaded from the domain profile JSON (e.g., health_logger.json).
    """

    def __init__(self, domain_data: dict | None = None):
        self._domain_data = domain_data or {}

    def validate(
        self,
        ir: ShortcutIR,
        result: ValidationResult,
    ) -> ValidationResult:
        """Run domain-specific validation rules on a ShortcutIR.

        Appends findings as warnings (non-blocking) to the existing
        ValidationResult from the core validator.
        """
        # Always run generic rules
        self._check_missing_error_handling(ir, result)
        self._check_unreachable_code(ir, result)

        # Run HealthKit rules only if domain data is available
        if "hk_sample_types" in self._domain_data:
            self._check_hk_unit_compatibility(ir, result)
            self._check_hk_value_ranges(ir, result)

        return result

    # ── Rule 1: HealthKit Sample Type ↔ Unit Compatibility ──────────

    def _check_hk_unit_compatibility(
        self,
        ir: ShortcutIR,
        result: ValidationResult,
    ) -> None:
        """Check that HealthKit sample types use compatible units."""
        hk_types = self._domain_data.get("hk_sample_types", {})

        for stmt in self._walk_actions(ir.statements):
            if not self._is_action(stmt, "health.quantity.log"):
                continue

            sample_type = self._get_string_param(stmt, "WFQuantitySampleType")
            # Check if a unit is specified explicitly (WFQuantitySampleUnit or similar)
            # HealthKit actions typically don't require explicit unit params since
            # the unit is implicit from the sample type, but if one is present, validate it.
            if sample_type and sample_type in hk_types:
                expected_unit = hk_types[sample_type].get("unit", "")
                # Check for unit in params (some DSL patterns set it explicitly)
                explicit_unit = self._get_string_param(stmt, "WFQuantitySampleUnit")
                if explicit_unit and expected_unit and explicit_unit != expected_unit:
                    result.warnings.append(ValidationError(
                        severity="warning",
                        category="hk_unit_mismatch",
                        message=(
                            f"HealthKit sample '{sample_type}' expects unit "
                            f"'{expected_unit}', but got '{explicit_unit}'"
                        ),
                        action_name=stmt.action_name,
                        line_number=stmt.line_number,
                    ))

    # ── Rule 2: HealthKit Value Range Sanity ─────────────────────────

    def _check_hk_value_ranges(
        self,
        ir: ShortcutIR,
        result: ValidationResult,
    ) -> None:
        """Warn on HealthKit values outside typical ranges."""
        value_ranges = self._domain_data.get("value_ranges", {})

        for stmt in self._walk_actions(ir.statements):
            if not self._is_action(stmt, "health.quantity.log"):
                continue

            sample_type = self._get_string_param(stmt, "WFQuantitySampleType")
            if sample_type not in value_ranges:
                continue

            quantity = self._get_number_param(stmt, "WFQuantitySampleQuantity")
            if quantity is None:
                continue

            range_info = value_ranges[sample_type]
            min_val = range_info.get("min", 0)
            max_val = range_info.get("max", float("inf"))

            if quantity < min_val or quantity > max_val:
                result.warnings.append(ValidationError(
                    severity="warning",
                    category="hk_value_range",
                    message=(
                        f"HealthKit value {quantity} for '{sample_type}' "
                        f"is outside typical range [{min_val}, {max_val}]"
                    ),
                    action_name=stmt.action_name,
                    line_number=stmt.line_number,
                ))

    # ── Rule 3: Missing Error Handling After downloadurl ─────────────

    def _check_missing_error_handling(
        self,
        ir: ShortcutIR,
        result: ValidationResult,
    ) -> None:
        """Warn when downloadurl is not followed by an IF block for error checking."""
        self._check_error_handling_in_stmts(ir.statements, result)

    def _check_error_handling_in_stmts(
        self,
        stmts: list[Statement],
        result: ValidationResult,
    ) -> None:
        """Recursively check for error handling after downloadurl in statement lists."""
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, ActionStatement) and self._is_action(stmt, "downloadurl"):
                # Check if any of the next 3 statements is an IF block
                has_if = False
                for j in range(i + 1, min(i + 4, len(stmts))):
                    if isinstance(stmts[j], IfBlock):
                        has_if = True
                        break
                    # Also accept SET + IF pattern
                    if isinstance(stmts[j], SetVariable):
                        continue
                    if isinstance(stmts[j], ActionStatement):
                        # Actions like detect.dictionary are fine between downloadurl and IF
                        if self._is_action(stmts[j], "detect.dictionary"):
                            continue
                        break

                if not has_if:
                    result.warnings.append(ValidationError(
                        severity="warning",
                        category="missing_error_handling",
                        message=(
                            "downloadurl not followed by error handling (IF block). "
                            "Network requests can fail — consider checking the response."
                        ),
                        action_name=stmt.action_name,
                        line_number=stmt.line_number,
                    ))

            # Recurse into nested blocks
            if isinstance(stmt, IfBlock):
                self._check_error_handling_in_stmts(stmt.then_body, result)
                if stmt.else_body:
                    self._check_error_handling_in_stmts(stmt.else_body, result)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    self._check_error_handling_in_stmts(case.body, result)
            elif isinstance(stmt, RepeatBlock):
                self._check_error_handling_in_stmts(stmt.body, result)
            elif isinstance(stmt, ForeachBlock):
                self._check_error_handling_in_stmts(stmt.body, result)

    # ── Rule 4: Unreachable Code Detection ───────────────────────────

    def _check_unreachable_code(
        self,
        ir: ShortcutIR,
        result: ValidationResult,
    ) -> None:
        """Detect statements after unconditional exits (showresult with no follow-up)."""
        self._check_unreachable_in_stmts(ir.statements, result)

    def _check_unreachable_in_stmts(
        self,
        stmts: list[Statement],
        result: ValidationResult,
    ) -> None:
        """Check for unreachable code in a statement list."""
        # Exit-like actions that typically end execution
        _EXIT_ACTIONS = {"exit", "nothing"}  # note: showresult doesn't exit

        for i, stmt in enumerate(stmts):
            if isinstance(stmt, ActionStatement):
                action_lower = stmt.action_name.lower()
                if action_lower in _EXIT_ACTIONS and i < len(stmts) - 1:
                    # Check if remaining statements are all comments
                    remaining = stmts[i + 1:]
                    non_comment = [s for s in remaining if not isinstance(s, Comment)]
                    if non_comment:
                        next_stmt = non_comment[0]
                        line_num = getattr(next_stmt, 'line_number', 0)
                        result.warnings.append(ValidationError(
                            severity="warning",
                            category="unreachable_code",
                            message=(
                                f"Statement after '{stmt.action_name}' may be unreachable"
                            ),
                            action_name=getattr(next_stmt, 'action_name', ''),
                            line_number=line_num,
                        ))

            # Recurse into nested blocks
            if isinstance(stmt, IfBlock):
                self._check_unreachable_in_stmts(stmt.then_body, result)
                if stmt.else_body:
                    self._check_unreachable_in_stmts(stmt.else_body, result)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    self._check_unreachable_in_stmts(case.body, result)
            elif isinstance(stmt, RepeatBlock):
                self._check_unreachable_in_stmts(stmt.body, result)
            elif isinstance(stmt, ForeachBlock):
                self._check_unreachable_in_stmts(stmt.body, result)

    # ── Helpers ───────────────────────────────────────────────────────

    def _walk_actions(self, stmts: list[Statement]) -> list[ActionStatement]:
        """Recursively collect all ActionStatements from nested blocks."""
        actions: list[ActionStatement] = []
        for stmt in stmts:
            if isinstance(stmt, ActionStatement):
                actions.append(stmt)
            elif isinstance(stmt, IfBlock):
                actions.extend(self._walk_actions(stmt.then_body))
                if stmt.else_body:
                    actions.extend(self._walk_actions(stmt.else_body))
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    actions.extend(self._walk_actions(case.body))
            elif isinstance(stmt, RepeatBlock):
                actions.extend(self._walk_actions(stmt.body))
            elif isinstance(stmt, ForeachBlock):
                actions.extend(self._walk_actions(stmt.body))
        return actions

    @staticmethod
    def _is_action(stmt: ActionStatement, short_name: str) -> bool:
        """Check if an action statement matches a short action name."""
        action = stmt.action_name.lower()
        return (
            action == short_name.lower()
            or action == f"is.workflow.actions.{short_name.lower()}"
        )

    @staticmethod
    def _get_string_param(stmt: ActionStatement, param_name: str) -> str:
        """Get a string parameter value from an action statement."""
        value = stmt.params.get(param_name)
        if isinstance(value, StringValue):
            return value.value
        return ""

    @staticmethod
    def _get_number_param(stmt: ActionStatement, param_name: str) -> float | None:
        """Get a numeric parameter value from an action statement."""
        value = stmt.params.get(param_name)
        if isinstance(value, NumberValue):
            return value.value
        return None


# ============================================================
# Public API
# ============================================================

_validator: SemanticValidator | None = None


def get_validator() -> SemanticValidator:
    """Get or create the semantic validator (cached)."""
    global _validator
    if _validator is None:
        _validator = SemanticValidator()
    return _validator


def validate_ir(
    ir: ShortcutIR,
    strict: bool = True,
    domain_profile: str = "general",
) -> ValidationResult:
    """Validate a ShortcutIR against the action catalog and param schemas.

    Args:
        ir: The ShortcutIR to validate.
        strict: If True (default), unknown actions with known vendor prefix
            are treated as errors. If False, they become warnings.
        domain_profile: Domain profile ID for domain-aware validation rules.
            Loads validation data from the domain profile if available.

    Returns:
        ValidationResult with errors and warnings.
    """
    result = get_validator().validate(ir, strict=strict)

    # Run domain-aware validation layer (Phase 2)
    domain_data: dict = {}
    if domain_profile != "general":
        try:
            from domain_profile import DomainProfileManager
            mgr = DomainProfileManager()
            domain_data = mgr.get_validation_data(domain_profile)
        except ImportError:
            pass  # domain_profile module not available

    domain_layer = DomainValidationLayer(domain_data)
    domain_layer.validate(ir, result)

    return result
