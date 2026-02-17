#!/usr/bin/env python3
"""Tests for domain_profile.py — Phase 1 domain profile system.

Tests cover:
  - Profile loading from JSON
  - Profile selection via keyword scoring
  - Prompt context injection
  - Relevant actions formatting
  - Validation data extraction
  - Edge cases (no profiles, unknown profile, general fallback)
"""

import sys
import json
import tempfile
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from domain_profile import (
    DomainProfile,
    DomainProfileManager,
    get_domain_profile_manager,
    select_domain_profile,
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


# ── Profile Loading Tests ──────────────────────────────────────────

def test_profiles_load():
    """Domain profiles load from references/domain_profiles/."""
    mgr = DomainProfileManager()
    profiles = mgr.list_profiles()
    assert len(profiles) >= 3, f"Expected ≥3 profiles, got {len(profiles)}: {profiles}"

run_test("profiles_load", test_profiles_load)


def test_health_profile_exists():
    """health_logger profile exists with correct data."""
    mgr = DomainProfileManager()
    profile = mgr.get_profile("health_logger")
    assert profile is not None, "health_logger profile not found"
    assert profile.profile_id == "health_logger"
    assert len(profile.keywords) >= 10
    assert profile.has_context

run_test("health_profile_exists", test_health_profile_exists)


def test_api_profile_exists():
    """api_workflow profile exists with correct data."""
    mgr = DomainProfileManager()
    profile = mgr.get_profile("api_workflow")
    assert profile is not None, "api_workflow profile not found"
    assert profile.profile_id == "api_workflow"
    assert len(profile.keywords) >= 10
    assert profile.has_context

run_test("api_profile_exists", test_api_profile_exists)


def test_general_profile_exists():
    """general profile exists (fallback)."""
    mgr = DomainProfileManager()
    profile = mgr.get_profile("general")
    assert profile is not None, "general profile not found"
    assert profile.profile_id == "general"

run_test("general_profile_exists", test_general_profile_exists)


def test_profile_custom_dir():
    """Profiles can be loaded from a custom directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test profile
        profile_path = Path(tmpdir) / "test_domain.json"
        profile_path.write_text(json.dumps({
            "_meta": {"profile_id": "test_domain"},
            "keywords": ["test", "demo"],
            "prompt_context": "You are testing.",
            "relevant_actions": ["comment"],
        }))

        mgr = DomainProfileManager(profiles_dir=Path(tmpdir))
        profiles = mgr.list_profiles()
        assert "test_domain" in profiles
        profile = mgr.get_profile("test_domain")
        assert profile is not None
        assert profile.prompt_context == "You are testing."

run_test("profile_custom_dir", test_profile_custom_dir)


def test_profile_empty_dir():
    """Empty directory returns no profiles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = DomainProfileManager(profiles_dir=Path(tmpdir))
        profiles = mgr.list_profiles()
        assert len(profiles) == 0

run_test("profile_empty_dir", test_profile_empty_dir)


# ── Profile Selection Tests ──────────────────────────────────────────

def test_select_health_profile():
    """Health-related prompts select health_logger profile."""
    mgr = DomainProfileManager()
    profile = mgr.select_profile("Log my caffeine intake and vitamin supplements to Apple Health")
    assert profile.profile_id == "health_logger", f"Expected health_logger, got {profile.profile_id}"

run_test("select_health_profile", test_select_health_profile)


def test_select_api_profile():
    """API-related prompts select api_workflow profile."""
    mgr = DomainProfileManager()
    profile = mgr.select_profile("Fetch JSON from the REST API endpoint and parse the response")
    assert profile.profile_id == "api_workflow", f"Expected api_workflow, got {profile.profile_id}"

run_test("select_api_profile", test_select_api_profile)


def test_select_general_fallback():
    """Generic prompts fall back to general profile."""
    mgr = DomainProfileManager()
    profile = mgr.select_profile("Set a timer for 5 minutes")
    assert profile.profile_id == "general", f"Expected general, got {profile.profile_id}"

run_test("select_general_fallback", test_select_general_fallback)


def test_select_minimum_score_threshold():
    """Single keyword match doesn't trigger non-general profile (requires ≥2)."""
    mgr = DomainProfileManager()
    profile = mgr.select_profile("Show me something random")
    assert profile.profile_id == "general", f"Expected general, got {profile.profile_id}"

run_test("select_minimum_score_threshold", test_select_minimum_score_threshold)


def test_select_phrase_match_bonus():
    """Multi-word keywords get a bonus score (3 vs 1)."""
    mgr = DomainProfileManager()
    # "log health" is a phrase keyword in health_logger — counts as 3 points
    profile = mgr.select_profile("I want to log health data")
    assert profile.profile_id == "health_logger", f"Expected health_logger, got {profile.profile_id}"

run_test("select_phrase_match_bonus", test_select_phrase_match_bonus)


def test_select_empty_prompt():
    """Empty prompt returns general profile."""
    mgr = DomainProfileManager()
    profile = mgr.select_profile("")
    assert profile.profile_id == "general"

run_test("select_empty_prompt", test_select_empty_prompt)


# ── Prompt Context Tests ─────────────────────────────────────────────

def test_health_context_content():
    """Health profile context mentions HealthKit."""
    mgr = DomainProfileManager()
    context = mgr.get_prompt_context("health_logger")
    assert "HealthKit" in context or "health" in context.lower()

run_test("health_context_content", test_health_context_content)


def test_api_context_content():
    """API profile context mentions error handling or HTTP."""
    mgr = DomainProfileManager()
    context = mgr.get_prompt_context("api_workflow")
    assert "error" in context.lower() or "http" in context.lower()

run_test("api_context_content", test_api_context_content)


def test_unknown_profile_context():
    """Unknown profile returns empty context."""
    mgr = DomainProfileManager()
    context = mgr.get_prompt_context("nonexistent_profile")
    assert context == ""

run_test("unknown_profile_context", test_unknown_profile_context)


# ── Relevant Actions Tests ───────────────────────────────────────────

def test_health_relevant_actions():
    """Health profile has health.quantity.log as relevant action."""
    mgr = DomainProfileManager()
    profile = mgr.get_profile("health_logger")
    assert profile is not None
    assert "health.quantity.log" in profile.relevant_actions

run_test("health_relevant_actions", test_health_relevant_actions)


def test_format_relevant_actions():
    """format_relevant_actions() returns formatted text."""
    profile = DomainProfile(
        profile_id="test",
        relevant_actions=["action_a", "action_b", "action_c"],
    )
    formatted = profile.format_relevant_actions()
    assert "action_a" in formatted
    assert "action_b" in formatted
    assert "action_c" in formatted

run_test("format_relevant_actions", test_format_relevant_actions)


def test_format_relevant_actions_empty():
    """format_relevant_actions() returns empty string when no actions."""
    profile = DomainProfile(profile_id="test")
    formatted = profile.format_relevant_actions()
    assert formatted == ""

run_test("format_relevant_actions_empty", test_format_relevant_actions_empty)


# ── Validation Data Tests ────────────────────────────────────────────

def test_health_validation_data():
    """Health profile provides HK sample types for validation."""
    mgr = DomainProfileManager()
    data = mgr.get_validation_data("health_logger")
    assert "hk_sample_types" in data
    assert "Caffeine" in data["hk_sample_types"]
    assert data["hk_sample_types"]["Caffeine"]["unit"] == "mg"

run_test("health_validation_data", test_health_validation_data)


def test_health_unit_compatibility():
    """Health profile provides unit compatibility data."""
    mgr = DomainProfileManager()
    data = mgr.get_validation_data("health_logger")
    assert "unit_compatibility" in data
    assert "mg" in data["unit_compatibility"]
    assert "Caffeine" in data["unit_compatibility"]["mg"]

run_test("health_unit_compatibility", test_health_unit_compatibility)


def test_api_validation_data():
    """API profile provides HTTP patterns for validation."""
    mgr = DomainProfileManager()
    data = mgr.get_validation_data("api_workflow")
    assert "http_patterns" in data
    assert "GET" in data["http_patterns"]

run_test("api_validation_data", test_api_validation_data)


def test_unknown_validation_data():
    """Unknown profile returns empty validation data."""
    mgr = DomainProfileManager()
    data = mgr.get_validation_data("nonexistent")
    assert data == {}

run_test("unknown_validation_data", test_unknown_validation_data)


# ── Convenience Function Tests ───────────────────────────────────────

def test_convenience_select():
    """select_domain_profile() convenience function works."""
    profile = select_domain_profile("Log caffeine and vitamins to health")
    assert profile.profile_id == "health_logger"

run_test("convenience_select", test_convenience_select)


def test_singleton_manager():
    """get_domain_profile_manager() returns singleton."""
    mgr1 = get_domain_profile_manager()
    mgr2 = get_domain_profile_manager()
    assert mgr1 is mgr2

run_test("singleton_manager", test_singleton_manager)


# ── has_context Property Tests ──────────────────────────────────────

def test_has_context_true():
    """has_context returns True when prompt_context is non-empty."""
    profile = DomainProfile(profile_id="test", prompt_context="Some guidance.")
    assert profile.has_context is True

run_test("has_context_true", test_has_context_true)


def test_has_context_false():
    """has_context returns False when prompt_context is empty or whitespace."""
    profile = DomainProfile(profile_id="test", prompt_context="")
    assert profile.has_context is False
    profile2 = DomainProfile(profile_id="test2", prompt_context="   ")
    assert profile2.has_context is False

run_test("has_context_false", test_has_context_false)


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
