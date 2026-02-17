"""
Regression tests for dsl_validator.py.
Tests tiered resolution, strict/permissive modes, compiler risk, and auto-derived prefixes.

Run: python3 scripts/test_dsl_validator.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
from dsl_validator import validate_ir, get_validator, _derive_vendor_prefixes
from dsl_parser import parse_dsl

passed = 0
failed = 0

def run_test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS: {name}")
    except Exception as e:
        failed += 1
        print(f"  FAIL: {name} -- {e}")


# ============================================================
# Auto-derived vendor prefixes
# ============================================================

def test_auto_derived_prefixes():
    v = get_validator()
    prefixes = v._known_vendor_prefixes
    assert "is.workflow." in prefixes, "is.workflow. should be auto-derived"
    assert "com.apple." in prefixes, "com.apple. should be auto-derived"
    assert "com.culturedcode." in prefixes, "com.culturedcode. should be auto-derived"
    assert len(prefixes) >= 25, f"Expected >= 25 prefixes, got {len(prefixes)}"

def test_manual_override_prefixes():
    v = get_validator()
    prefixes = v._known_vendor_prefixes
    assert "codes.rambo." in prefixes, "codes.rambo. should be in manual overrides"
    assert "dk.simonbs." in prefixes, "dk.simonbs. should be in manual overrides"

for name, fn in [
    ("auto_derived_prefixes", test_auto_derived_prefixes),
    ("manual_override_prefixes", test_manual_override_prefixes),
]:
    run_test(name, fn)


# ============================================================
# Tier 1: Known action — passes in both modes
# ============================================================

def test_tier1_known_action():
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="https://example.com"\nENDSHORTCUT\n'
    ir = parse_dsl(dsl)
    strict = validate_ir(ir, strict=True)
    perm = validate_ir(ir, strict=False)
    assert strict.is_valid, f"Strict should pass for known action: {strict}"
    assert perm.is_valid, f"Permissive should pass for known action: {perm}"

run_test("tier1_known_action", test_tier1_known_action)


# ============================================================
# Tier 3: Known vendor prefix — error strict, warning permissive
# ============================================================

def test_tier3_known_vendor_strict_error():
    dsl = 'SHORTCUT "Test"\nACTION com.corsair.ihc.ToggleLightIntent\nENDSHORTCUT\n'
    ir = parse_dsl(dsl)
    strict = validate_ir(ir, strict=True)
    assert not strict.is_valid, "Tier 3 should be error in strict mode"
    assert any(e.category == "unknown_action" for e in strict.errors)

def test_tier3_known_vendor_permissive_warning():
    dsl = 'SHORTCUT "Test"\nACTION com.corsair.ihc.ToggleLightIntent\nENDSHORTCUT\n'
    ir = parse_dsl(dsl)
    perm = validate_ir(ir, strict=False)
    assert perm.is_valid, "Tier 3 should be warning (not error) in permissive mode"
    assert any(w.category == "vendor_prefix_unknown" for w in perm.warnings)

for name, fn in [
    ("tier3_known_vendor_strict_error", test_tier3_known_vendor_strict_error),
    ("tier3_known_vendor_permissive_warning", test_tier3_known_vendor_permissive_warning),
]:
    run_test(name, fn)


# ============================================================
# Tier 3 + handle param: compiler_risk warning
# ============================================================

def test_tier3_compiler_risk_warning():
    dsl = 'SHORTCUT "Test"\nACTION openurl URL="https://example.com"\nACTION com.corsair.ihc.ToggleLightIntent Input=@prev\nENDSHORTCUT\n'
    ir = parse_dsl(dsl)
    perm = validate_ir(ir, strict=False)
    risk_warns = [w for w in perm.warnings if w.category == "compiler_risk"]
    assert len(risk_warns) >= 1, f"Expected compiler_risk warning, got categories: {[w.category for w in perm.warnings]}"
    assert "WFTextTokenAttachment" in risk_warns[0].message

run_test("tier3_compiler_risk_warning", test_tier3_compiler_risk_warning)


# ============================================================
# Tier 4: Unknown / no vendor prefix — error in both modes
# ============================================================

def test_tier4_unknown_error_both():
    dsl = 'SHORTCUT "Test"\nACTION notification Message="hello"\nACTION completelyFakeAction\nENDSHORTCUT\n'
    ir = parse_dsl(dsl)
    strict = validate_ir(ir, strict=True)
    perm = validate_ir(ir, strict=False)
    assert not strict.is_valid, "Tier 4 should be error in strict mode"
    assert not perm.is_valid, "Tier 4 should be error in permissive mode too"

def test_tier4_short_name_no_dots():
    """Short name without dots and no vendor prefix — Tier 4."""
    dsl = 'SHORTCUT "Test"\nACTION madeUpAction\nENDSHORTCUT\n'
    ir = parse_dsl(dsl)
    strict = validate_ir(ir, strict=True)
    perm = validate_ir(ir, strict=False)
    assert not strict.is_valid
    assert not perm.is_valid

for name, fn in [
    ("tier4_unknown_error_both", test_tier4_unknown_error_both),
    ("tier4_short_name_no_dots", test_tier4_short_name_no_dots),
]:
    run_test(name, fn)


# ============================================================
# Backward compatibility: default strict=True
# ============================================================

def test_default_strict():
    """validate_ir() without strict param should default to strict=True."""
    dsl = 'SHORTCUT "Test"\nACTION com.corsair.ihc.ToggleLightIntent\nENDSHORTCUT\n'
    ir = parse_dsl(dsl)
    result = validate_ir(ir)  # no strict param
    assert not result.is_valid, "Default should be strict (Tier 3 = error)"

run_test("default_strict", test_default_strict)


# ============================================================
# Results
# ============================================================

print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
if failed == 0:
    print("All tests passed.")
else:
    print(f"FAILURES: {failed}")
    sys.exit(1)
