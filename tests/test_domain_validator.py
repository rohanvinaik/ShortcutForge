#!/usr/bin/env python3
"""Tests for DomainValidationLayer in dsl_validator.py — Phase 2.

Tests cover:
  - HealthKit unit compatibility checking
  - HealthKit value range warnings
  - Missing error handling after downloadurl
  - Unreachable code detection
  - Domain-aware validate_ir() integration
"""

import sys
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dsl_ir import (
    ActionStatement,
    Comment,
    HandleRef,
    IfBlock,
    NumberValue,
    SetVariable,
    ShortcutIR,
    StringValue,
    VarRef,
)
from dsl_validator import (
    DomainValidationLayer,
    ValidationResult,
    validate_ir,
)

# ── Test Harness ─────────────────────────────────────────────────────

_pass = 0
_fail = 0
_results: list[tuple[str, bool, str]] = []


def run_test(name: str, fn):
    global _pass, _fail
    try:
        fn()
        _pass += 1
        _results.append((name, True, ""))
        print(f"  PASS: {name}")
    except Exception as e:
        _fail += 1
        _results.append((name, False, str(e)))
        print(f"  FAIL: {name} -- {e}")


# ── Helpers ──────────────────────────────────────────────────────────


def _make_ir(statements):
    """Create a ShortcutIR from a list of statements."""
    return ShortcutIR(name="Test", statements=statements)


def _make_action(name, params=None, line=1):
    """Create an ActionStatement."""
    return ActionStatement(action_name=name, params=params or {}, line_number=line)


def _health_domain_data():
    """Return health domain data for testing."""
    return {
        "hk_sample_types": {
            "Caffeine": {"unit": "mg", "category": "dietary"},
            "Vitamin D": {"unit": "mcg", "category": "dietary"},
            "Protein": {"unit": "g", "category": "dietary"},
        },
        "unit_compatibility": {
            "mg": ["Caffeine"],
            "mcg": ["Vitamin D"],
            "g": ["Protein"],
        },
        "value_ranges": {
            "Caffeine": {"min": 0, "max": 1000, "typical": [50, 200]},
            "Vitamin D": {"min": 0, "max": 250, "typical": [10, 100]},
            "Protein": {"min": 0, "max": 500, "typical": [20, 100]},
        },
    }


# ── HealthKit Unit Compatibility Tests ──────────────────────────────


def test_hk_unit_match_ok():
    """No warning when HealthKit unit matches expected."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Caffeine"),
                    "WFQuantitySampleQuantity": NumberValue(100),
                    "WFQuantitySampleUnit": StringValue("mg"),
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer(_health_domain_data())
    layer.validate(ir, result)
    unit_warnings = [w for w in result.warnings if w.category == "hk_unit_mismatch"]
    assert len(unit_warnings) == 0, f"Expected no unit mismatch, got {unit_warnings}"


run_test("hk_unit_match_ok", test_hk_unit_match_ok)


def test_hk_unit_mismatch():
    """Warning when HealthKit unit doesn't match expected."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Caffeine"),
                    "WFQuantitySampleQuantity": NumberValue(100),
                    "WFQuantitySampleUnit": StringValue("mcg"),  # Should be mg!
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer(_health_domain_data())
    layer.validate(ir, result)
    unit_warnings = [w for w in result.warnings if w.category == "hk_unit_mismatch"]
    assert len(unit_warnings) == 1, (
        f"Expected 1 unit mismatch warning, got {len(unit_warnings)}"
    )
    assert "mg" in unit_warnings[0].message
    assert "mcg" in unit_warnings[0].message


run_test("hk_unit_mismatch", test_hk_unit_mismatch)


def test_hk_unit_no_explicit_unit():
    """No warning when no explicit unit is specified (implicit from type)."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Caffeine"),
                    "WFQuantitySampleQuantity": NumberValue(100),
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer(_health_domain_data())
    layer.validate(ir, result)
    unit_warnings = [w for w in result.warnings if w.category == "hk_unit_mismatch"]
    assert len(unit_warnings) == 0


run_test("hk_unit_no_explicit_unit", test_hk_unit_no_explicit_unit)


# ── HealthKit Value Range Tests ─────────────────────────────────────


def test_hk_value_in_range():
    """No warning when value is within typical range."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Caffeine"),
                    "WFQuantitySampleQuantity": NumberValue(200),
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer(_health_domain_data())
    layer.validate(ir, result)
    range_warnings = [w for w in result.warnings if w.category == "hk_value_range"]
    assert len(range_warnings) == 0


run_test("hk_value_in_range", test_hk_value_in_range)


def test_hk_value_above_max():
    """Warning when value exceeds maximum."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Caffeine"),
                    "WFQuantitySampleQuantity": NumberValue(5000),  # Max is 1000
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer(_health_domain_data())
    layer.validate(ir, result)
    range_warnings = [w for w in result.warnings if w.category == "hk_value_range"]
    assert len(range_warnings) == 1
    assert "5000" in range_warnings[0].message


run_test("hk_value_above_max", test_hk_value_above_max)


def test_hk_value_negative():
    """Warning when value is negative."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Protein"),
                    "WFQuantitySampleQuantity": NumberValue(-10),  # Min is 0
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer(_health_domain_data())
    layer.validate(ir, result)
    range_warnings = [w for w in result.warnings if w.category == "hk_value_range"]
    assert len(range_warnings) == 1


run_test("hk_value_negative", test_hk_value_negative)


def test_hk_value_dynamic_skipped():
    """No warning when value is a variable (can't check at static time)."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Caffeine"),
                    "WFQuantitySampleQuantity": VarRef("Amount"),
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer(_health_domain_data())
    layer.validate(ir, result)
    range_warnings = [w for w in result.warnings if w.category == "hk_value_range"]
    assert len(range_warnings) == 0


run_test("hk_value_dynamic_skipped", test_hk_value_dynamic_skipped)


# ── Missing Error Handling Tests ─────────────────────────────────────


def test_downloadurl_with_if_ok():
    """No warning when downloadurl is followed by IF."""
    ir = _make_ir(
        [
            _make_action("downloadurl", {"WFHTTPMethod": StringValue("GET")}),
            IfBlock(
                target=HandleRef("prev"),
                condition="has_any_value",
                compare_value=None,
                then_body=[_make_action("showresult", {"Text": StringValue("ok")})],
                else_body=None,
                line_number=2,
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) == 0


run_test("downloadurl_with_if_ok", test_downloadurl_with_if_ok)


def test_downloadurl_with_set_then_if_ok():
    """No warning for downloadurl → SET → IF pattern."""
    ir = _make_ir(
        [
            _make_action("downloadurl", {"WFHTTPMethod": StringValue("GET")}),
            SetVariable(var_name="Response", value=HandleRef("prev"), line_number=2),
            IfBlock(
                target=VarRef("Response"),
                condition="has_any_value",
                compare_value=None,
                then_body=[_make_action("showresult", {"Text": StringValue("ok")})],
                else_body=None,
                line_number=3,
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) == 0


run_test("downloadurl_with_set_then_if_ok", test_downloadurl_with_set_then_if_ok)


def test_downloadurl_no_error_handling():
    """Warning when downloadurl has no following IF block."""
    ir = _make_ir(
        [
            _make_action("downloadurl", {"WFHTTPMethod": StringValue("GET")}, line=1),
            _make_action("showresult", {"Text": StringValue("done")}, line=2),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) == 1
    assert "downloadurl" in err_warnings[0].message.lower()


run_test("downloadurl_no_error_handling", test_downloadurl_no_error_handling)


def test_downloadurl_detect_dict_then_if_ok():
    """No warning for downloadurl → detect.dictionary → IF pattern."""
    ir = _make_ir(
        [
            _make_action("downloadurl", {"WFHTTPMethod": StringValue("GET")}),
            _make_action("detect.dictionary"),
            IfBlock(
                target=HandleRef("prev"),
                condition="has_any_value",
                compare_value=None,
                then_body=[_make_action("showresult", {"Text": StringValue("ok")})],
                else_body=None,
                line_number=3,
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) == 0


run_test("downloadurl_detect_dict_then_if_ok", test_downloadurl_detect_dict_then_if_ok)


def test_downloadurl_in_nested_block():
    """Warning fires for downloadurl inside nested blocks too."""
    ir = _make_ir(
        [
            IfBlock(
                target=VarRef("Flag"),
                condition="has_any_value",
                compare_value=None,
                then_body=[
                    _make_action(
                        "downloadurl", {"WFHTTPMethod": StringValue("GET")}, line=3
                    ),
                    _make_action("showresult", {"Text": StringValue("done")}, line=4),
                ],
                else_body=None,
                line_number=2,
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) == 1


run_test("downloadurl_in_nested_block", test_downloadurl_in_nested_block)


# ── Unreachable Code Tests ──────────────────────────────────────────


def test_code_after_exit():
    """Warning for statements after exit action."""
    ir = _make_ir(
        [
            _make_action("exit", {}, line=1),
            _make_action("showresult", {"Text": StringValue("unreachable")}, line=2),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    unreachable = [w for w in result.warnings if w.category == "unreachable_code"]
    assert len(unreachable) == 1


run_test("code_after_exit", test_code_after_exit)


def test_code_after_exit_comments_ok():
    """No warning when only comments follow exit."""
    ir = _make_ir(
        [
            _make_action("exit", {}, line=1),
            Comment(text="This is a comment", line_number=2),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    unreachable = [w for w in result.warnings if w.category == "unreachable_code"]
    assert len(unreachable) == 0


run_test("code_after_exit_comments_ok", test_code_after_exit_comments_ok)


def test_no_unreachable_normal_flow():
    """No warning for normal control flow."""
    ir = _make_ir(
        [
            _make_action(
                "comment", {"WFCommentActionText": StringValue("step 1")}, line=1
            ),
            _make_action("showresult", {"Text": StringValue("done")}, line=2),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()
    layer.validate(ir, result)
    unreachable = [w for w in result.warnings if w.category == "unreachable_code"]
    assert len(unreachable) == 0


run_test("no_unreachable_normal_flow", test_no_unreachable_normal_flow)


# ── No Domain Data Tests ────────────────────────────────────────────


def test_no_domain_data_no_hk_warnings():
    """Without domain data, no HK-specific warnings fire."""
    ir = _make_ir(
        [
            _make_action(
                "health.quantity.log",
                {
                    "WFQuantitySampleType": StringValue("Caffeine"),
                    "WFQuantitySampleQuantity": NumberValue(99999),
                    "WFQuantitySampleUnit": StringValue("wrong"),
                },
            ),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()  # No domain data
    layer.validate(ir, result)
    hk_warnings = [
        w
        for w in result.warnings
        if w.category in ("hk_unit_mismatch", "hk_value_range")
    ]
    assert len(hk_warnings) == 0


run_test("no_domain_data_no_hk_warnings", test_no_domain_data_no_hk_warnings)


def test_no_domain_data_still_checks_error_handling():
    """Generic rules (error handling, unreachable) fire without domain data."""
    ir = _make_ir(
        [
            _make_action("downloadurl", {"WFHTTPMethod": StringValue("GET")}, line=1),
            _make_action("showresult", {"Text": StringValue("done")}, line=2),
        ]
    )
    result = ValidationResult()
    layer = DomainValidationLayer()  # No domain data
    layer.validate(ir, result)
    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) == 1


run_test(
    "no_domain_data_still_checks_error_handling",
    test_no_domain_data_still_checks_error_handling,
)


# ── Integration with validate_ir() ──────────────────────────────────


def test_validate_ir_with_domain():
    """validate_ir() with domain_profile runs domain-aware validation."""
    from dsl_parser import parse_dsl

    dsl = """SHORTCUT "Health Test"

ACTION url WFURLActionURL="https://api.com"
ACTION downloadurl WFHTTPMethod="GET"
ACTION showresult Text="done"
ENDSHORTCUT
"""
    ir = parse_dsl(dsl)
    result = validate_ir(ir, strict=False, domain_profile="health_logger")

    # Should have missing_error_handling warning from domain layer
    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) >= 1, (
        f"Expected missing_error_handling warning, got categories: {[w.category for w in result.warnings]}"
    )


run_test("validate_ir_with_domain", test_validate_ir_with_domain)


def test_validate_ir_general_still_checks():
    """validate_ir() with general profile still runs generic domain checks."""
    from dsl_parser import parse_dsl

    dsl = """SHORTCUT "General Test"

ACTION url WFURLActionURL="https://api.com"
ACTION downloadurl WFHTTPMethod="GET"
ACTION showresult Text="done"
ENDSHORTCUT
"""
    ir = parse_dsl(dsl)
    result = validate_ir(ir, strict=False, domain_profile="general")

    err_warnings = [
        w for w in result.warnings if w.category == "missing_error_handling"
    ]
    assert len(err_warnings) >= 1


run_test("validate_ir_general_still_checks", test_validate_ir_general_still_checks)


# ── Report ───────────────────────────────────────────────────────────

print()
print("=" * 50)
print(f"Results: {_pass} passed, {_fail} failed, {_pass + _fail} total")
if _fail > 0:
    print(f"FAILURES: {_fail}")
    for name, passed, err in _results:
        if not passed:
            print(f"  {name}: {err}")
else:
    print("All tests passed.")
