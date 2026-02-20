"""
Tests for architecture_reasoner.py -- Phase 3 architecture reasoning.

Tests cover:
  - shortcut_only decisions for simple/device prompts
  - shortcut_plus_blueprint decisions for server/database/webhook prompts
  - Multi-signal hybrid detection with confidence thresholds
  - Blueprint generation for hybrid decisions
  - Blueprint returns None for shortcut_only
  - Convenience function analyze_architecture()
  - is_hybrid property on ArchitectureDecision

Run: python3 scripts/test_architecture_reasoner.py
"""

import sys
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from architecture_reasoner import (
    ArchitectureDecision,
    ArchitectureReasoner,
    BlueprintDoc,
    analyze_architecture,
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


# -- Shortcut-only decisions -----------------------------------------------


def test_shortcut_only_simple():
    """Simple prompt with no hybrid signals -> shortcut_only."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze("Set a timer for 5 minutes")
    assert decision.strategy == "shortcut_only", (
        f"Expected shortcut_only, got {decision.strategy}"
    )


run_test("shortcut_only_simple", test_shortcut_only_simple)


def test_shortcut_only_device():
    """Device control prompt -> shortcut_only."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze("Toggle bluetooth and set brightness")
    assert decision.strategy == "shortcut_only", (
        f"Expected shortcut_only, got {decision.strategy}"
    )


run_test("shortcut_only_device", test_shortcut_only_device)


# -- Hybrid (shortcut_plus_blueprint) decisions ----------------------------


def test_hybrid_server():
    """Prompt mentioning server/deploy -> shortcut_plus_blueprint."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze("Deploy a server to handle API requests")
    assert decision.strategy == "shortcut_plus_blueprint", (
        f"Expected shortcut_plus_blueprint, got {decision.strategy}"
    )


run_test("hybrid_server", test_hybrid_server)


def test_hybrid_database():
    """Prompt mentioning permanent cloud storage -> shortcut_plus_blueprint."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze("Store data permanently in a cloud database")
    assert decision.strategy == "shortcut_plus_blueprint", (
        f"Expected shortcut_plus_blueprint, got {decision.strategy}"
    )


run_test("hybrid_database", test_hybrid_database)


def test_hybrid_webhook():
    """Prompt mentioning webhook/callback -> shortcut_plus_blueprint."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze(
        "Set up a webhook endpoint with a callback url to receive from external services"
    )
    assert decision.strategy == "shortcut_plus_blueprint", (
        f"Expected shortcut_plus_blueprint, got {decision.strategy}"
    )


run_test("hybrid_webhook", test_hybrid_webhook)


def test_hybrid_multi_signal():
    """Multiple hybrid categories -> shortcut_plus_blueprint with high confidence."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze("Deploy a backend server with database and webhook")
    assert decision.strategy == "shortcut_plus_blueprint", (
        f"Expected shortcut_plus_blueprint, got {decision.strategy}"
    )
    assert decision.confidence >= 0.8, (
        f"Expected confidence >= 0.8 for multi-signal, got {decision.confidence}"
    )


run_test("hybrid_multi_signal", test_hybrid_multi_signal)


# -- Blueprint generation --------------------------------------------------


def test_blueprint_generation():
    """Hybrid decision -> generate_blueprint returns BlueprintDoc with components."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze("Deploy a backend server with database")
    assert decision.is_hybrid, "Expected hybrid decision for blueprint test"

    bp = reasoner.generate_blueprint(decision, "Deploy a backend server with database")
    assert bp is not None, "Expected a BlueprintDoc for hybrid decision"
    assert isinstance(bp, BlueprintDoc), f"Expected BlueprintDoc, got {type(bp)}"
    assert len(bp.components) > 0, "Blueprint should have at least one component"
    assert bp.title, "Blueprint should have a title"


run_test("blueprint_generation", test_blueprint_generation)


def test_blueprint_none_for_shortcut_only():
    """shortcut_only decision -> generate_blueprint returns None."""
    reasoner = ArchitectureReasoner()
    decision = reasoner.analyze("Set a timer for 5 minutes")
    assert not decision.is_hybrid, "Expected shortcut_only decision"

    bp = reasoner.generate_blueprint(decision, "Set a timer for 5 minutes")
    assert bp is None, f"Expected None for shortcut_only blueprint, got {bp}"


run_test("blueprint_none_for_shortcut_only", test_blueprint_none_for_shortcut_only)


# -- Convenience function --------------------------------------------------


def test_convenience_function():
    """analyze_architecture() convenience function works."""
    decision = analyze_architecture("Set a timer for 5 minutes")
    assert isinstance(decision, ArchitectureDecision), (
        f"Expected ArchitectureDecision, got {type(decision)}"
    )
    assert decision.strategy == "shortcut_only"


run_test("convenience_function", test_convenience_function)


# -- Property tests --------------------------------------------------------


def test_is_hybrid_property():
    """is_hybrid property returns correct values for both strategies."""
    shortcut_only = ArchitectureDecision(
        strategy="shortcut_only",
        reason="Test",
    )
    hybrid = ArchitectureDecision(
        strategy="shortcut_plus_blueprint",
        reason="Test",
    )
    assert shortcut_only.is_hybrid is False, (
        "is_hybrid should be False for shortcut_only"
    )
    assert hybrid.is_hybrid is True, (
        "is_hybrid should be True for shortcut_plus_blueprint"
    )


run_test("is_hybrid_property", test_is_hybrid_property)


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
