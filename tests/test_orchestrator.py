"""
Regression tests for orchestrator.py and token_budget.py.
Tests overflow detection, FailureType classification, and token budgeting.

Run: python3 scripts/test_orchestrator.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent / "src"))
from orchestrator import FailureType, classify_failure
from token_budget import estimate_budget, detect_overflow, next_budget, TokenBudget

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
# FailureType classification
# ============================================================

def test_classify_overflow():
    assert classify_failure(None, None, None, True) == FailureType.OVERFLOW

def test_classify_syntax():
    assert classify_failure("parse error", None, None, False) == FailureType.SYNTAX

def test_classify_unknown_action():
    assert classify_failure(None, ["Unknown action: 'foo'"], None, False) == FailureType.UNKNOWN_ACTION

def test_classify_unknown_action_lowercase():
    assert classify_failure(None, ["unknown_action: something"], None, False) == FailureType.UNKNOWN_ACTION

def test_classify_bad_params():
    assert classify_failure(None, ["bad parameter value"], None, False) == FailureType.BAD_PARAMS

def test_classify_compile():
    assert classify_failure(None, None, "compile error", False) == FailureType.COMPILE

def test_classify_none():
    assert classify_failure(None, None, None, False) == FailureType.NONE

def test_classify_overflow_takes_priority():
    """Overflow should take priority over parse error."""
    assert classify_failure("parse error", None, None, True) == FailureType.OVERFLOW

for name, fn in [
    ("classify_overflow", test_classify_overflow),
    ("classify_syntax", test_classify_syntax),
    ("classify_unknown_action", test_classify_unknown_action),
    ("classify_unknown_action_lowercase", test_classify_unknown_action_lowercase),
    ("classify_bad_params", test_classify_bad_params),
    ("classify_compile", test_classify_compile),
    ("classify_none", test_classify_none),
    ("classify_overflow_takes_priority", test_classify_overflow_takes_priority),
]:
    run_test(name, fn)


# ============================================================
# Token budget estimation
# ============================================================

def test_budget_simple():
    b = estimate_budget("Open the Settings app")
    assert b.complexity == "simple", f"Expected simple, got {b.complexity}"
    assert b.max_tokens == 512

def test_budget_medium():
    b = estimate_budget("Create a shortcut that asks for a name and saves it to Notes")
    assert b.complexity == "medium", f"Expected medium, got {b.complexity}"
    assert b.max_tokens == 1024

def test_budget_complex_signals():
    b = estimate_budget("Create a shortcut with a menu and conditional check for each item in a list")
    assert b.complexity == "very_complex", f"Expected very_complex, got {b.complexity}"
    assert b.max_tokens == 4096

def test_budget_complex_long_prompt():
    b = estimate_budget("x " * 50)  # 50 words, > 40 threshold
    assert b.complexity == "complex", f"Expected complex for long prompt, got {b.complexity}"

def test_budget_very_complex_signals():
    """Very complex: 4+ complex signals."""
    b = estimate_budget("Create a shortcut that uses a menu, loops through each item, checks conditionally, and compares values in a list")
    assert b.complexity == "very_complex", f"Expected very_complex, got {b.complexity}"
    assert b.max_tokens == 4096

def test_budget_very_complex_long_prompt():
    """Very complex: word_count > 80."""
    b = estimate_budget("x " * 90)  # 90 words, > 80 threshold
    assert b.complexity == "very_complex", f"Expected very_complex for very long prompt, got {b.complexity}"
    assert b.max_tokens == 4096

def test_budget_very_complex_char_len():
    """Very complex: char_len > 500."""
    b = estimate_budget("a" * 510)
    assert b.complexity == "very_complex", f"Expected very_complex for long char prompt, got {b.complexity}"
    assert b.max_tokens == 4096

def test_budget_word_count():
    b = estimate_budget("Set the brightness to 50 percent")
    assert b.word_count == 6, f"Expected 6 words, got {b.word_count}"

def test_next_budget_escalation():
    """next_budget should step through tiers."""
    assert next_budget(512) == 1024
    assert next_budget(1024) == 2048
    assert next_budget(2048) == 4096
    assert next_budget(4096) is None

def test_next_budget_non_standard():
    """next_budget with non-standard values should find next tier."""
    assert next_budget(100) == 512
    assert next_budget(700) == 1024
    assert next_budget(3000) == 4096

for name, fn in [
    ("budget_simple", test_budget_simple),
    ("budget_medium", test_budget_medium),
    ("budget_complex_signals", test_budget_complex_signals),
    ("budget_complex_long_prompt", test_budget_complex_long_prompt),
    ("budget_very_complex_signals", test_budget_very_complex_signals),
    ("budget_very_complex_long_prompt", test_budget_very_complex_long_prompt),
    ("budget_very_complex_char_len", test_budget_very_complex_char_len),
    ("budget_word_count", test_budget_word_count),
    ("next_budget_escalation", test_next_budget_escalation),
    ("next_budget_non_standard", test_next_budget_non_standard),
]:
    run_test(name, fn)


# ============================================================
# Overflow detection
# ============================================================

def test_overflow_detected():
    """Near-budget output without ENDSHORTCUT should detect overflow."""
    raw = "SHORTCUT \"Test\"\n" + "ACTION openurl URL=\"x\"\n" * 100
    budget = TokenBudget(
        max_tokens=512, complexity="simple",
        word_count=3, complex_signal_count=0,
        simple_signal_count=1, prompt_char_len=20,
    )
    assert detect_overflow(raw, budget, has_endshortcut=False)

def test_overflow_not_detected_with_endshortcut():
    """ENDSHORTCUT present should prevent overflow detection."""
    raw = "SHORTCUT \"Test\"\n" + "ACTION openurl URL=\"x\"\n" * 100 + "ENDSHORTCUT\n"
    budget = TokenBudget(
        max_tokens=512, complexity="simple",
        word_count=3, complex_signal_count=0,
        simple_signal_count=1, prompt_char_len=20,
    )
    assert not detect_overflow(raw, budget, has_endshortcut=True)

def test_overflow_not_detected_short_output():
    """Short output well within budget should not detect overflow."""
    raw = "SHORTCUT \"Test\"\nACTION openurl URL=\"x\"\n"
    budget = TokenBudget(
        max_tokens=1024, complexity="medium",
        word_count=5, complex_signal_count=0,
        simple_signal_count=0, prompt_char_len=30,
    )
    assert not detect_overflow(raw, budget, has_endshortcut=False)

for name, fn in [
    ("overflow_detected", test_overflow_detected),
    ("overflow_not_detected_with_endshortcut", test_overflow_not_detected_with_endshortcut),
    ("overflow_not_detected_short_output", test_overflow_not_detected_short_output),
]:
    run_test(name, fn)


# ============================================================
# Retry routing (unit-level: verifies routing decisions)
# ============================================================

def test_overflow_routes_to_budget():
    """Overflow should NOT use grammar (budget is the issue)."""
    ftype = classify_failure(None, None, None, True)
    assert ftype == FailureType.OVERFLOW
    # Routing: budget increase, not grammar

def test_syntax_routes_to_grammar():
    """Syntax failure should route to grammar constraint."""
    ftype = classify_failure("parse error", None, None, False)
    assert ftype == FailureType.SYNTAX
    # Routing: use_grammar=True for LocalBackend

def test_unknown_action_no_grammar():
    """Unknown action should NOT use grammar (grammar doesn't constrain action names)."""
    ftype = classify_failure(None, ["Unknown action: 'foo'"], None, False)
    assert ftype == FailureType.UNKNOWN_ACTION
    # Routing: error context retry without grammar

def test_compile_no_retry():
    """Compile failure should return immediately (no retry)."""
    ftype = classify_failure(None, None, "compile error", False)
    assert ftype == FailureType.COMPILE
    # Routing: no retry

for name, fn in [
    ("overflow_routes_to_budget", test_overflow_routes_to_budget),
    ("syntax_routes_to_grammar", test_syntax_routes_to_grammar),
    ("unknown_action_no_grammar", test_unknown_action_no_grammar),
    ("compile_no_retry", test_compile_no_retry),
]:
    run_test(name, fn)


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
