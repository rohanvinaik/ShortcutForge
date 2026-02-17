"""
Tests for distillation_curator.py — Phase 6 training data curation.

Tests cover:
  - Quality gate: parse/validate/compile filtering
  - Creativity threshold filtering
  - Prompt deduplication by word overlap similarity
  - Scenario balancing
  - CurationStats correctness
  - CuratedEntry.to_training_dict format
  - File-based curation
  - Convenience function

Run: python3 scripts/test_distillation_curator.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
_TRAINING_DIR = _SCRIPT_DIR.parent / "training"
sys.path.insert(0, str(_TRAINING_DIR))

from distillation_curator import (
    DistillationCurator,
    CuratedEntry,
    CurationStats,
    curate_distillation_log,
    _word_overlap_similarity,
    _normalize_prompt,
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

def _make_entry(
    prompt: str = "Test prompt",
    output: str = 'SHORTCUT "Test"\nENDSHORTCUT',
    parsed: bool = True,
    validated_permissive: bool = True,
    compiled_permissive: bool = True,
    creativity_score: float | None = None,
    scenario_profile: str = "default",
    domain_profile: str = "general",
    architecture_decision: str = "shortcut_only",
    budget_complexity: str = "medium",
    shortcut_id: str = "test-001",
) -> dict:
    return {
        "prompt": prompt,
        "canonicalized_output": output,
        "parsed": parsed,
        "validated_permissive": validated_permissive,
        "validated_strict": validated_permissive,
        "compiled_permissive": compiled_permissive,
        "compiled_strict": compiled_permissive,
        "creativity_score": creativity_score,
        "scenario_profile": scenario_profile,
        "domain_profile": domain_profile,
        "architecture_decision": architecture_decision,
        "budget_complexity": budget_complexity,
        "shortcut_id": shortcut_id,
        "provenance": {"model_id": "test-model"},
    }


# ==========================================================================
# Quality Gate Tests
# ==========================================================================

def test_quality_passes_good_entry():
    """Entry that parses+validates+compiles passes quality gate."""
    curator = DistillationCurator()
    entries = [_make_entry()]
    curated, stats = curator.curate(entries)

    assert len(curated) == 1, f"Expected 1 curated entry, got {len(curated)}"
    assert stats.quality_passed == 1
    assert stats.quality_failed == 0

run_test("quality_passes_good_entry", test_quality_passes_good_entry)


def test_quality_rejects_unparsed():
    """Entry that fails parse is rejected."""
    curator = DistillationCurator()
    entries = [_make_entry(parsed=False)]
    curated, stats = curator.curate(entries)

    assert len(curated) == 0
    assert stats.failed_parse == 1

run_test("quality_rejects_unparsed", test_quality_rejects_unparsed)


def test_quality_rejects_unvalidated():
    """Entry that fails validation is rejected."""
    curator = DistillationCurator()
    entries = [_make_entry(validated_permissive=False)]
    curated, stats = curator.curate(entries)

    assert len(curated) == 0
    assert stats.failed_validate == 1

run_test("quality_rejects_unvalidated", test_quality_rejects_unvalidated)


def test_quality_rejects_uncompiled():
    """Entry that fails compile is rejected."""
    curator = DistillationCurator()
    entries = [_make_entry(compiled_permissive=False)]
    curated, stats = curator.curate(entries)

    assert len(curated) == 0
    assert stats.failed_compile == 1

run_test("quality_rejects_uncompiled", test_quality_rejects_uncompiled)


def test_quality_rejects_low_creativity():
    """Entry below creativity threshold is rejected."""
    curator = DistillationCurator(creativity_threshold=0.5)
    entries = [_make_entry(creativity_score=0.3)]
    curated, stats = curator.curate(entries)

    assert len(curated) == 0
    assert stats.failed_creativity == 1

run_test("quality_rejects_low_creativity", test_quality_rejects_low_creativity)


def test_quality_passes_no_creativity_score():
    """Entry without creativity score passes (score is optional)."""
    curator = DistillationCurator(creativity_threshold=0.5)
    entries = [_make_entry(creativity_score=None)]
    curated, stats = curator.curate(entries)

    assert len(curated) == 1

run_test("quality_passes_no_creativity_score", test_quality_passes_no_creativity_score)


# ==========================================================================
# Deduplication Tests
# ==========================================================================

def test_dedup_removes_identical():
    """Identical prompts are deduplicated."""
    curator = DistillationCurator()
    entries = [
        _make_entry(prompt="Set a timer for 5 minutes"),
        _make_entry(prompt="Set a timer for 5 minutes"),
    ]
    curated, stats = curator.curate(entries)

    assert len(curated) == 1
    assert stats.dedup_removed == 1

run_test("dedup_removes_identical", test_dedup_removes_identical)


def test_dedup_removes_similar():
    """Very similar prompts are deduplicated."""
    curator = DistillationCurator(similarity_threshold=0.7)
    entries = [
        _make_entry(prompt="Set a timer for 5 minutes"),
        _make_entry(prompt="Set a timer for five minutes"),
    ]
    curated, stats = curator.curate(entries)

    assert len(curated) == 1
    assert stats.dedup_removed == 1

run_test("dedup_removes_similar", test_dedup_removes_similar)


def test_dedup_keeps_different():
    """Different prompts are kept."""
    curator = DistillationCurator()
    entries = [
        _make_entry(prompt="Set a timer for 5 minutes"),
        _make_entry(prompt="Toggle bluetooth and set brightness to maximum"),
    ]
    curated, stats = curator.curate(entries)

    assert len(curated) == 2
    assert stats.dedup_removed == 0

run_test("dedup_keeps_different", test_dedup_keeps_different)


# ==========================================================================
# Similarity Function Tests
# ==========================================================================

def test_similarity_identical():
    """Identical prompts have similarity 1.0."""
    sim = _word_overlap_similarity("Set a timer", "Set a timer")
    assert sim == 1.0, f"Expected 1.0, got {sim}"

run_test("similarity_identical", test_similarity_identical)


def test_similarity_different():
    """Completely different prompts have low similarity."""
    sim = _word_overlap_similarity(
        "Set a timer for 5 minutes",
        "Deploy a backend server with database"
    )
    assert sim < 0.3, f"Expected < 0.3, got {sim}"

run_test("similarity_different", test_similarity_different)


def test_normalize_prompt():
    """Prompt normalization removes case, whitespace, punctuation."""
    norm = _normalize_prompt("  Set a Timer!  For 5 Minutes. ")
    assert norm == "set a timer for 5 minutes", f"Got: {norm}"

run_test("normalize_prompt", test_normalize_prompt)


# ==========================================================================
# Scenario Balancing Tests
# ==========================================================================

def test_scenario_balancing():
    """max_per_scenario caps entries per scenario."""
    curator = DistillationCurator(max_per_scenario=2)
    entries = [
        _make_entry(prompt=f"Health task {i}", scenario_profile="health_tracking")
        for i in range(5)
    ]
    curated, stats = curator.curate(entries)

    assert len(curated) == 2, f"Expected 2 after capping, got {len(curated)}"

run_test("scenario_balancing", test_scenario_balancing)


# ==========================================================================
# Output Format Tests
# ==========================================================================

def test_curated_entry_to_dict():
    """CuratedEntry.to_training_dict produces expected format."""
    entry = CuratedEntry(
        prompt="Test prompt",
        canonicalized_output="SHORTCUT ...",
        parsed=True,
        validated=True,
        compiled=True,
        scenario_profile="health_tracking",
        domain_profile="health_logger",
        creativity_score=0.75,
    )
    d = entry.to_training_dict()

    assert d["prompt"] == "Test prompt"
    assert d["completion"] == "SHORTCUT ..."
    assert d["metadata"]["scenario_profile"] == "health_tracking"
    assert d["metadata"]["domain_profile"] == "health_logger"
    assert d["metadata"]["creativity_score"] == 0.75

run_test("curated_entry_to_dict", test_curated_entry_to_dict)


def test_stats_summary():
    """CurationStats.summary produces readable output."""
    stats = CurationStats(
        input_count=10,
        quality_passed=8,
        quality_failed=2,
        dedup_removed=1,
        output_count=7,
        failed_parse=1,
        failed_validate=1,
    )
    summary = stats.summary()
    assert "10" in summary
    assert "8" in summary
    assert "7" in summary

run_test("stats_summary", test_stats_summary)


# ==========================================================================
# File-Based Curation Tests
# ==========================================================================

def test_curate_file():
    """File-based curation reads JSONL and writes curated output."""
    entries = [
        _make_entry(prompt=f"Task {i}", shortcut_id=f"id-{i}")
        for i in range(5)
    ]
    # Add a failing entry
    entries.append(_make_entry(prompt="Failing task", parsed=False, shortcut_id="id-fail"))

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "distillation_log.jsonl"
        output_path = Path(tmpdir) / "curated_training.jsonl"

        # Write input
        with open(input_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        # Curate
        curator = DistillationCurator()
        stats = curator.curate_file(input_path, output_path)

        assert stats.input_count == 6
        assert stats.quality_passed == 5
        assert stats.quality_failed == 1
        assert stats.output_count == 5

        # Check output file
        assert output_path.exists()
        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 5

        # Parse first line
        first = json.loads(lines[0])
        assert "prompt" in first
        assert "completion" in first
        assert "metadata" in first

run_test("curate_file", test_curate_file)


def test_convenience_function():
    """curate_distillation_log convenience function works."""
    entries = [_make_entry(prompt=f"Prompt {i}") for i in range(3)]

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "distillation_log.jsonl"
        output_path = Path(tmpdir) / "curated.jsonl"

        with open(input_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        stats = curate_distillation_log(input_path, output_path)
        assert stats.input_count == 3
        assert stats.output_count == 3

run_test("convenience_function", test_convenience_function)


# ==========================================================================
# Scenario Distribution Test
# ==========================================================================

def test_scenario_distribution():
    """Stats correctly tracks scenario distribution."""
    curator = DistillationCurator()
    entries = [
        _make_entry(prompt="Health thing 1", scenario_profile="health_tracking"),
        _make_entry(prompt="API call 2", scenario_profile="api_integration"),
        _make_entry(prompt="Health thing 3", scenario_profile="health_tracking"),
    ]
    curated, stats = curator.curate(entries)

    assert stats.scenario_distribution.get("health_tracking") == 2
    assert stats.scenario_distribution.get("api_integration") == 1

run_test("scenario_distribution", test_scenario_distribution)


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
