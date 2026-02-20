"""
Tests for simulation_harness.py — Phase 4 static analysis engine.

Tests cover all 6 analysis categories:
  1. Variable flow (set-before-use, branch coverage, loop scoping, unused)
  2. Loop bound checking (>1000, zero/negative)
  3. Menu case completeness (duplicates, empty cases, single case)
  4. Dead code (statements after exit)
  5. API endpoint validation (URL format, HTTP method, POST without body)
  6. Type flow (type mismatches through @prev chain)

Also tests:
  - SimulationReport properties
  - Selective analysis
  - Convenience function
  - Clean IR produces no findings

Run: python3 scripts/test_simulation_harness.py
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
    InterpolatedString,
    MenuBlock,
    MenuCase,
    NumberValue,
    RepeatBlock,
    SetVariable,
    ShortcutIR,
    StringValue,
    VarRef,
)
from simulation_harness import (
    FindingCategory,
    Severity,
    SimulationFinding,
    SimulationHarness,
    SimulationReport,
    simulate,
)

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


# -- Helpers ---------------------------------------------------------------


def _make_ir(name: str, stmts: list) -> ShortcutIR:
    return ShortcutIR(name=name, statements=stmts)


def _count_category(report: SimulationReport, cat: FindingCategory) -> int:
    return len(report.findings_by_category(cat))


# ==========================================================================
# Analysis 1: Variable Flow
# ==========================================================================


def test_var_use_before_def():
    """Variable used before it's defined → warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement("showresult", {"WFText": VarRef("x")}, line_number=1),
        ],
    )
    report = SimulationHarness().analyze(ir)
    flow = report.findings_by_category(FindingCategory.VARIABLE_FLOW)
    msgs = [f.message for f in flow]
    assert any("$x" in m and "before definition" in m for m in msgs), (
        f"Expected use-before-def for $x, got: {msgs}"
    )


run_test("var_use_before_def", test_var_use_before_def)


def test_var_defined_then_used():
    """Variable defined before use → no warning."""
    ir = _make_ir(
        "test",
        [
            SetVariable("x", StringValue("hello"), line_number=1),
            ActionStatement("showresult", {"WFText": VarRef("x")}, line_number=2),
        ],
    )
    report = SimulationHarness().analyze(ir)
    flow = report.findings_by_category(FindingCategory.VARIABLE_FLOW)
    use_before_def = [f for f in flow if "before definition" in f.message]
    assert len(use_before_def) == 0, (
        f"Expected no use-before-def, got: {[f.message for f in use_before_def]}"
    )


run_test("var_defined_then_used", test_var_defined_then_used)


def test_var_branch_only_then():
    """Variable defined only in IF-then branch → info."""
    ir = _make_ir(
        "test",
        [
            IfBlock(
                target=VarRef("cond"),
                condition="equals",
                compare_value=StringValue("yes"),
                then_body=[SetVariable("x", StringValue("1"), line_number=3)],
                else_body=[],
                line_number=2,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    flow = report.findings_by_category(FindingCategory.VARIABLE_FLOW)
    branch_msgs = [f for f in flow if "only defined in IF-then" in f.message]
    assert len(branch_msgs) >= 1, (
        f"Expected branch-only warning, got: {[f.message for f in flow]}"
    )


run_test("var_branch_only_then", test_var_branch_only_then)


def test_var_unused():
    """Variable defined but never used → info."""
    ir = _make_ir(
        "test",
        [
            SetVariable("unused_var", StringValue("val"), line_number=1),
            ActionStatement(
                "showresult", {"WFText": StringValue("done")}, line_number=2
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    flow = report.findings_by_category(FindingCategory.VARIABLE_FLOW)
    unused = [f for f in flow if "never used" in f.message]
    assert len(unused) >= 1, (
        f"Expected unused warning for $unused_var, got: {[f.message for f in flow]}"
    )


run_test("var_unused", test_var_unused)


def test_var_loop_scoped():
    """Variable defined inside loop → info about loop scoping."""
    ir = _make_ir(
        "test",
        [
            RepeatBlock(
                count=NumberValue(5),
                body=[SetVariable("loop_var", StringValue("iter"), line_number=3)],
                line_number=2,
            ),
            ActionStatement(
                "showresult", {"WFText": VarRef("loop_var")}, line_number=5
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    flow = report.findings_by_category(FindingCategory.VARIABLE_FLOW)
    loop_msgs = [f for f in flow if "inside loop" in f.message]
    assert len(loop_msgs) >= 1, (
        f"Expected loop-scoped info, got: {[f.message for f in flow]}"
    )


run_test("var_loop_scoped", test_var_loop_scoped)


# ==========================================================================
# Analysis 2: Loop Bound Checking
# ==========================================================================


def test_loop_large_count():
    """REPEAT count > 1000 → warning."""
    ir = _make_ir(
        "test",
        [
            RepeatBlock(
                count=NumberValue(5000),
                body=[
                    ActionStatement(
                        "showresult", {"WFText": StringValue("hi")}, line_number=2
                    ),
                ],
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    loop = report.findings_by_category(FindingCategory.LOOP_BOUND)
    assert len(loop) == 1, f"Expected 1 loop bound warning, got {len(loop)}"
    assert "5000" in loop[0].message


run_test("loop_large_count", test_loop_large_count)


def test_loop_zero_count():
    """REPEAT count of 0 → warning."""
    ir = _make_ir(
        "test",
        [
            RepeatBlock(
                count=NumberValue(0),
                body=[
                    ActionStatement(
                        "showresult", {"WFText": StringValue("hi")}, line_number=2
                    ),
                ],
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    loop = report.findings_by_category(FindingCategory.LOOP_BOUND)
    assert len(loop) == 1, f"Expected 1 loop bound warning, got {len(loop)}"
    assert "zero or negative" in loop[0].message


run_test("loop_zero_count", test_loop_zero_count)


def test_loop_normal_count():
    """REPEAT count of 10 → no warning."""
    ir = _make_ir(
        "test",
        [
            RepeatBlock(
                count=NumberValue(10),
                body=[
                    ActionStatement(
                        "showresult", {"WFText": StringValue("hi")}, line_number=2
                    ),
                ],
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    loop = report.findings_by_category(FindingCategory.LOOP_BOUND)
    assert len(loop) == 0, f"Expected no loop bound warnings, got {len(loop)}"


run_test("loop_normal_count", test_loop_normal_count)


# ==========================================================================
# Analysis 3: Menu Case Completeness
# ==========================================================================


def test_menu_duplicate_labels():
    """Duplicate menu labels → warning."""
    ir = _make_ir(
        "test",
        [
            MenuBlock(
                prompt="Pick",
                cases=[
                    MenuCase(
                        "Option A",
                        [
                            ActionStatement(
                                "showresult",
                                {"WFText": StringValue("a")},
                                line_number=3,
                            )
                        ],
                    ),
                    MenuCase(
                        "Option A",
                        [
                            ActionStatement(
                                "showresult",
                                {"WFText": StringValue("b")},
                                line_number=5,
                            )
                        ],
                    ),
                ],
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    menu = report.findings_by_category(FindingCategory.MENU_COMPLETENESS)
    dup = [f for f in menu if "Duplicate" in f.message]
    assert len(dup) >= 1, (
        f"Expected duplicate label warning, got: {[f.message for f in menu]}"
    )


run_test("menu_duplicate_labels", test_menu_duplicate_labels)


def test_menu_empty_case():
    """Empty menu case → info."""
    ir = _make_ir(
        "test",
        [
            MenuBlock(
                prompt="Pick",
                cases=[
                    MenuCase(
                        "Option A",
                        [
                            ActionStatement(
                                "showresult",
                                {"WFText": StringValue("a")},
                                line_number=3,
                            )
                        ],
                    ),
                    MenuCase("Option B", []),
                ],
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    menu = report.findings_by_category(FindingCategory.MENU_COMPLETENESS)
    empty = [f for f in menu if "Empty" in f.message]
    assert len(empty) >= 1, (
        f"Expected empty case info, got: {[f.message for f in menu]}"
    )


run_test("menu_empty_case", test_menu_empty_case)


def test_menu_single_case():
    """Menu with only one case → info."""
    ir = _make_ir(
        "test",
        [
            MenuBlock(
                prompt="Pick",
                cases=[
                    MenuCase(
                        "Only Option",
                        [
                            ActionStatement(
                                "showresult",
                                {"WFText": StringValue("a")},
                                line_number=3,
                            )
                        ],
                    ),
                ],
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    menu = report.findings_by_category(FindingCategory.MENU_COMPLETENESS)
    single = [f for f in menu if "only one case" in f.message]
    assert len(single) >= 1, (
        f"Expected single-case info, got: {[f.message for f in menu]}"
    )


run_test("menu_single_case", test_menu_single_case)


# ==========================================================================
# Analysis 4: Dead Code
# ==========================================================================


def test_dead_code_after_exit():
    """Statement after exit action → warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement("exit", {}, line_number=1),
            ActionStatement(
                "showresult", {"WFText": StringValue("unreachable")}, line_number=2
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    dead = report.findings_by_category(FindingCategory.DEAD_CODE)
    assert len(dead) >= 1, f"Expected dead code warning, got {len(dead)}"


run_test("dead_code_after_exit", test_dead_code_after_exit)


def test_dead_code_after_nothing():
    """Statement after 'nothing' action → warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement("nothing", {}, line_number=1),
            ActionStatement(
                "showresult", {"WFText": StringValue("unreachable")}, line_number=2
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    dead = report.findings_by_category(FindingCategory.DEAD_CODE)
    assert len(dead) >= 1, f"Expected dead code warning, got {len(dead)}"


run_test("dead_code_after_nothing", test_dead_code_after_nothing)


def test_no_dead_code_normal():
    """Normal flow without exit → no dead code warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement(
                "showresult", {"WFText": StringValue("one")}, line_number=1
            ),
            ActionStatement(
                "showresult", {"WFText": StringValue("two")}, line_number=2
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    dead = report.findings_by_category(FindingCategory.DEAD_CODE)
    assert len(dead) == 0, f"Expected no dead code, got {len(dead)}"


run_test("no_dead_code_normal", test_no_dead_code_normal)


def test_dead_code_comments_ok():
    """Comments after exit are OK → no warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement("exit", {}, line_number=1),
            Comment("this is a comment", line_number=2),
        ],
    )
    report = SimulationHarness().analyze(ir)
    dead = report.findings_by_category(FindingCategory.DEAD_CODE)
    assert len(dead) == 0, (
        f"Expected no dead code warning for comments, got {len(dead)}"
    )


run_test("dead_code_comments_ok", test_dead_code_comments_ok)


# ==========================================================================
# Analysis 5: API Endpoint Validation
# ==========================================================================


def test_api_malformed_url():
    """URL without protocol → warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement(
                "downloadurl",
                {
                    "WFURL": StringValue("example.com/api"),
                },
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    api = report.findings_by_category(FindingCategory.API_VALIDATION)
    url_warns = [f for f in api if "malformed" in f.message]
    assert len(url_warns) >= 1, (
        f"Expected malformed URL warning, got: {[f.message for f in api]}"
    )


run_test("api_malformed_url", test_api_malformed_url)


def test_api_valid_url():
    """Valid URL → no warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement(
                "downloadurl",
                {
                    "WFURL": StringValue("https://api.example.com/data"),
                },
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    api = report.findings_by_category(FindingCategory.API_VALIDATION)
    url_warns = [f for f in api if "malformed" in f.message]
    assert len(url_warns) == 0, (
        f"Expected no URL warnings, got: {[f.message for f in api]}"
    )


run_test("api_valid_url", test_api_valid_url)


def test_api_invalid_method():
    """Invalid HTTP method → warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement(
                "downloadurl",
                {
                    "WFURL": StringValue("https://api.example.com/data"),
                    "WFHTTPMethod": StringValue("FETCH"),
                },
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    api = report.findings_by_category(FindingCategory.API_VALIDATION)
    method_warns = [f for f in api if "Unknown HTTP method" in f.message]
    assert len(method_warns) >= 1, (
        f"Expected invalid method warning, got: {[f.message for f in api]}"
    )


run_test("api_invalid_method", test_api_invalid_method)


def test_api_post_without_body():
    """POST without body specification → info."""
    ir = _make_ir(
        "test",
        [
            ActionStatement(
                "downloadurl",
                {
                    "WFURL": StringValue("https://api.example.com/data"),
                    "WFHTTPMethod": StringValue("POST"),
                },
                line_number=1,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    api = report.findings_by_category(FindingCategory.API_VALIDATION)
    body_warns = [f for f in api if "without body" in f.message]
    assert len(body_warns) >= 1, (
        f"Expected POST-without-body info, got: {[f.message for f in api]}"
    )


run_test("api_post_without_body", test_api_post_without_body)


# ==========================================================================
# Analysis 6: Type Flow
# ==========================================================================


def test_type_mismatch_prev():
    """text output → detect.dictionary (expects data) → info."""
    ir = _make_ir(
        "test",
        [
            ActionStatement("ask", {}, line_number=1),  # outputs text
            ActionStatement(
                "detect.dictionary",
                {
                    "WFInput": HandleRef("prev"),
                },
                line_number=2,
            ),  # expects data
        ],
    )
    report = SimulationHarness().analyze(ir)
    tf = report.findings_by_category(FindingCategory.TYPE_FLOW)
    assert len(tf) >= 1, f"Expected type mismatch, got {len(tf)}"
    assert "text" in tf[0].message and "data" in tf[0].message


run_test("type_mismatch_prev", test_type_mismatch_prev)


def test_type_match_ok():
    """downloadurl (data) → detect.dictionary (expects data) → no type warning."""
    ir = _make_ir(
        "test",
        [
            ActionStatement(
                "downloadurl",
                {
                    "WFURL": StringValue("https://api.example.com"),
                },
                line_number=1,
            ),  # outputs data
            ActionStatement(
                "detect.dictionary",
                {
                    "WFInput": HandleRef("prev"),
                },
                line_number=2,
            ),  # expects data
        ],
    )
    report = SimulationHarness().analyze(ir)
    tf = report.findings_by_category(FindingCategory.TYPE_FLOW)
    assert len(tf) == 0, f"Expected no type warnings, got: {[f.message for f in tf]}"


run_test("type_match_ok", test_type_match_ok)


# ==========================================================================
# Report & Convenience
# ==========================================================================


def test_report_properties():
    """SimulationReport correctly counts errors/warnings/info."""
    report = SimulationReport(
        findings=[
            SimulationFinding(Severity.ERROR, FindingCategory.VARIABLE_FLOW, "e1"),
            SimulationFinding(Severity.ERROR, FindingCategory.VARIABLE_FLOW, "e2"),
            SimulationFinding(Severity.WARNING, FindingCategory.LOOP_BOUND, "w1"),
            SimulationFinding(Severity.INFO, FindingCategory.DEAD_CODE, "i1"),
        ]
    )
    assert report.error_count == 2
    assert report.warning_count == 1
    assert report.info_count == 1
    assert report.has_errors is True


run_test("report_properties", test_report_properties)


def test_selective_analysis():
    """analyze_selective only runs requested categories."""
    # IR that would trigger loop_bound AND dead_code
    ir = _make_ir(
        "test",
        [
            RepeatBlock(
                count=NumberValue(5000),
                body=[
                    ActionStatement(
                        "showresult", {"WFText": StringValue("hi")}, line_number=2
                    ),
                ],
                line_number=1,
            ),
            ActionStatement("exit", {}, line_number=4),
            ActionStatement(
                "showresult", {"WFText": StringValue("dead")}, line_number=5
            ),
        ],
    )
    harness = SimulationHarness()

    # Only loop bounds
    report = harness.analyze_selective(ir, {FindingCategory.LOOP_BOUND})
    loop = report.findings_by_category(FindingCategory.LOOP_BOUND)
    dead = report.findings_by_category(FindingCategory.DEAD_CODE)
    assert len(loop) >= 1, "Should find loop bound issue"
    assert len(dead) == 0, "Should NOT find dead code when not requested"


run_test("selective_analysis", test_selective_analysis)


def test_convenience_function():
    """simulate() convenience function works."""
    ir = _make_ir(
        "test",
        [
            ActionStatement("showresult", {"WFText": StringValue("hi")}, line_number=1),
        ],
    )
    report = simulate(ir)
    assert isinstance(report, SimulationReport)


run_test("convenience_function", test_convenience_function)


def test_clean_ir_minimal_findings():
    """Clean, well-formed IR produces no errors or warnings."""
    ir = _make_ir(
        "Clean Shortcut",
        [
            SetVariable("name", StringValue("World"), line_number=1),
            ActionStatement(
                "showresult",
                {
                    "WFText": InterpolatedString(
                        (StringValue("Hello, "), VarRef("name"))
                    ),
                },
                line_number=2,
            ),
        ],
    )
    report = SimulationHarness().analyze(ir)
    errors_warnings = [
        f for f in report.findings if f.severity in (Severity.ERROR, Severity.WARNING)
    ]
    assert len(errors_warnings) == 0, (
        f"Expected no errors/warnings for clean IR, got: {[f.message for f in errors_warnings]}"
    )


run_test("clean_ir_minimal_findings", test_clean_ir_minimal_findings)


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
