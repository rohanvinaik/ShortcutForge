"""
Tests for creative_scoring.py -- Phase 3 creative scoring system.

Tests cover:
  - Scoring simple and complex shortcuts parsed from DSL
  - All 5 creative modes exist in CREATIVE_MODES
  - Mode weights sum to approximately 1.0
  - pragmatic mode favors error_handling
  - expressive mode favors ui_richness
  - Empty shortcuts score low
  - Score returns 5 dimensions
  - Convenience function score_shortcut()
  - Total score is bounded 0.0 to 1.0

Run: python3 scripts/test_creative_scoring.py
"""

import sys
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from creative_scoring import CREATIVE_MODES, CreativityScorer, score_shortcut
from dsl_ir import (
    ShortcutIR,
)
from dsl_parser import parse_dsl

# -- Test Harness ----------------------------------------------------------

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


# -- DSL text fixtures -----------------------------------------------------

_SIMPLE_DSL = (
    'SHORTCUT "Simple"\n'
    'ACTION showresult Text="Hello"\n'
    'ACTION notification Body="Done"\n'
    'ACTION comment Text="fin"\n'
    "ENDSHORTCUT\n"
)

_COMPLEX_DSL = (
    'SHORTCUT "Complex"\n'
    'ACTION ask Question="Name?"\n'
    "SET $name = @prev\n"
    "IF @prev has_any_value\n"
    '  ACTION showresult Text="Hi"\n'
    '  MENU "Pick"\n'
    '    CASE "A"\n'
    '      ACTION openurl URL="https://a.com"\n'
    '    CASE "B"\n'
    '      ACTION openurl URL="https://b.com"\n'
    "  ENDMENU\n"
    "ELSE\n"
    '  ACTION alert Title="Error" Message="No name"\n'
    "ENDIF\n"
    'ACTION notification Body="Complete"\n'
    "ENDSHORTCUT\n"
)


# -- Score simple vs complex shortcuts -------------------------------------


def test_score_simple_shortcut():
    """Simple 3-action shortcut gets a score > 0."""
    ir = parse_dsl(_SIMPLE_DSL)
    scorer = CreativityScorer()
    result = scorer.score(ir)
    assert result.total > 0, f"Expected score > 0, got {result.total}"


run_test("score_simple_shortcut", test_score_simple_shortcut)


def test_score_complex_shortcut():
    """Complex shortcut with IF, MENU, SET scores higher than simple."""
    ir_simple = parse_dsl(_SIMPLE_DSL)
    ir_complex = parse_dsl(_COMPLEX_DSL)
    scorer = CreativityScorer()
    score_simple = scorer.score(ir_simple)
    score_complex = scorer.score(ir_complex)
    assert score_complex.total > score_simple.total, (
        f"Complex ({score_complex.total:.3f}) should score higher "
        f"than simple ({score_simple.total:.3f})"
    )


run_test("score_complex_shortcut", test_score_complex_shortcut)


# -- Mode existence and weight validation ----------------------------------


def test_modes_exist():
    """All 5 creative modes exist in CREATIVE_MODES."""
    expected = {"pragmatic", "expressive", "playful", "automation_dense", "power_user"}
    actual = set(CREATIVE_MODES.keys())
    assert actual == expected, f"Expected modes {expected}, got {actual}"


run_test("modes_exist", test_modes_exist)


def test_mode_weights_sum_to_one():
    """Each mode's weights sum to approximately 1.0."""
    for mode_name, weights in CREATIVE_MODES.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, (
            f"Mode {mode_name} weights sum to {total}, expected ~1.0"
        )


run_test("mode_weights_sum_to_one", test_mode_weights_sum_to_one)


# -- Mode-specific weight checks ------------------------------------------


def test_pragmatic_favors_error_handling():
    """pragmatic mode gives error_handling weight >= 0.3."""
    weights = CREATIVE_MODES["pragmatic"]
    assert weights["error_handling"] >= 0.3, (
        f"pragmatic error_handling weight is {weights['error_handling']}, expected >= 0.3"
    )


run_test("pragmatic_favors_error_handling", test_pragmatic_favors_error_handling)


def test_expressive_favors_ui():
    """expressive mode gives ui_richness weight >= 0.25."""
    weights = CREATIVE_MODES["expressive"]
    assert weights["ui_richness"] >= 0.25, (
        f"expressive ui_richness weight is {weights['ui_richness']}, expected >= 0.25"
    )


run_test("expressive_favors_ui", test_expressive_favors_ui)


# -- Edge cases ------------------------------------------------------------


def test_score_empty_shortcut():
    """Empty shortcut (no statements) gets a low score."""
    ir = ShortcutIR(name="Empty", statements=[])
    scorer = CreativityScorer()
    result = scorer.score(ir)
    assert result.total < 0.5, f"Empty shortcut should score low, got {result.total}"


run_test("score_empty_shortcut", test_score_empty_shortcut)


# -- Dimension structure ---------------------------------------------------


def test_score_returns_dimensions():
    """Score result has 5 dimensions."""
    ir = parse_dsl(_SIMPLE_DSL)
    scorer = CreativityScorer()
    result = scorer.score(ir)
    assert len(result.dimensions) == 5, (
        f"Expected 5 dimensions, got {len(result.dimensions)}"
    )
    dim_names = {d.name for d in result.dimensions}
    expected_names = {
        "action_diversity",
        "ui_richness",
        "error_handling",
        "variable_reuse",
        "workflow_complexity",
    }
    assert dim_names == expected_names, (
        f"Expected dimension names {expected_names}, got {dim_names}"
    )


run_test("score_returns_dimensions", test_score_returns_dimensions)


# -- Convenience function --------------------------------------------------


def test_convenience_function():
    """score_shortcut() convenience function works."""
    ir = parse_dsl(_SIMPLE_DSL)
    result = score_shortcut(ir)
    assert result.total >= 0.0, f"Expected total >= 0, got {result.total}"
    assert result.mode == "pragmatic", (
        f"Expected default mode pragmatic, got {result.mode}"
    )


run_test("convenience_function", test_convenience_function)


# -- Bounds check ----------------------------------------------------------


def test_score_total_bounded():
    """Total score is bounded between 0.0 and 1.0."""
    ir = parse_dsl(_COMPLEX_DSL)
    scorer = CreativityScorer()
    for mode in CREATIVE_MODES:
        result = scorer.score(ir, mode=mode)
        assert 0.0 <= result.total <= 1.0, (
            f"Mode {mode}: total {result.total} out of bounds [0.0, 1.0]"
        )


run_test("score_total_bounded", test_score_total_bounded)


# -- Report ----------------------------------------------------------------

print()
print("=" * 50)
print(f"Results: {_pass} passed, {_fail} failed, {_pass + _fail} total")
if _fail > 0:
    print(f"FAILURES: {_fail}")
    for name, passed, err in _results:
        if not passed:
            print(f"  {name}: {err}")
    sys.exit(1)
else:
    print("All tests passed.")
