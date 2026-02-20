#!/usr/bin/env python3
"""
Comprehensive tests for all 8 scenario packs.

Validates rubric schemas, reference DSL files, scoring weights,
prompt variants, and end-to-end lint/parse/validate/compile/score pipelines
for every scenario pack under references/scenario_packs/.
"""

import json
import sys
import unittest
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(_SRC_DIR))
_TRAINING_DIR = _SCRIPT_DIR.parent / "training"
sys.path.insert(0, str(_TRAINING_DIR))

from evaluate_scenario import load_scenario_pack, score_dsl_text

from dsl_bridge import compile_ir
from dsl_linter import lint_dsl
from dsl_parser import parse_dsl
from dsl_validator import validate_ir

# ── Constants ────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
SCENARIO_PACKS_DIR = BASE_DIR / "references" / "scenario_packs"

ALL_PACK_NAMES = [
    "health_logger",
    "share_sheet_text_cleaner",
    "api_pagination_fetcher",
    "file_router",
    "calendar_triage",
    "clipboard_utility",
    "media_metadata_pipeline",
    "morning_routine",
]

VALID_COMPLEXITY_TIERS = {"simple", "medium", "complex", "very_complex"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def _load_rubric(pack_name: str) -> dict:
    """Load and return the rubric.json for a given scenario pack."""
    rubric_path = SCENARIO_PACKS_DIR / pack_name / "rubric.json"
    with open(rubric_path) as f:
        return json.load(f)


def _load_reference_dsl(pack_name: str) -> str:
    """Load and return the reference.dsl text for a given scenario pack."""
    dsl_path = SCENARIO_PACKS_DIR / pack_name / "reference.dsl"
    return dsl_path.read_text()


# ── 1. TestAllPacksExist ─────────────────────────────────────────────


class TestAllPacksExist(unittest.TestCase):
    """Verify all 8 scenario pack directories exist and contain both files."""

    def test_all_packs_exist_with_required_files(self):
        for pack_name in ALL_PACK_NAMES:
            pack_dir = SCENARIO_PACKS_DIR / pack_name
            self.assertTrue(
                pack_dir.is_dir(),
                f"Scenario pack directory missing: {pack_dir}",
            )
            rubric_path = pack_dir / "rubric.json"
            self.assertTrue(
                rubric_path.is_file(),
                f"rubric.json missing in {pack_name}",
            )
            reference_path = pack_dir / "reference.dsl"
            self.assertTrue(
                reference_path.is_file(),
                f"reference.dsl missing in {pack_name}",
            )


# ── 2. TestRubricSchema ─────────────────────────────────────────────


class TestRubricSchema(unittest.TestCase):
    """Verify each rubric has the required top-level fields with correct types."""

    def test_rubric_schema(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                rubric = _load_rubric(pack_name)

                # scenario_id: non-empty string
                self.assertIn("scenario_id", rubric)
                self.assertIsInstance(rubric["scenario_id"], str)
                self.assertTrue(len(rubric["scenario_id"]) > 0)

                # scenario_name: non-empty string
                self.assertIn("scenario_name", rubric)
                self.assertIsInstance(rubric["scenario_name"], str)
                self.assertTrue(len(rubric["scenario_name"]) > 0)

                # complexity_tier: valid value
                self.assertIn("complexity_tier", rubric)
                self.assertIn(
                    rubric["complexity_tier"],
                    VALID_COMPLEXITY_TIERS,
                    f"{pack_name}: complexity_tier '{rubric['complexity_tier']}' "
                    f"not in {VALID_COMPLEXITY_TIERS}",
                )

                # required_actions: non-empty list
                self.assertIn("required_actions", rubric)
                self.assertIsInstance(rubric["required_actions"], list)
                self.assertTrue(len(rubric["required_actions"]) > 0)

                # required_constructs: non-empty list
                self.assertIn("required_constructs", rubric)
                self.assertIsInstance(rubric["required_constructs"], list)
                self.assertTrue(len(rubric["required_constructs"]) > 0)

                # scoring: non-empty dict
                self.assertIn("scoring", rubric)
                self.assertIsInstance(rubric["scoring"], dict)
                self.assertTrue(len(rubric["scoring"]) > 0)

                # prompt_variants: list with 3+ entries
                self.assertIn("prompt_variants", rubric)
                self.assertIsInstance(rubric["prompt_variants"], list)
                self.assertGreaterEqual(
                    len(rubric["prompt_variants"]),
                    3,
                    f"{pack_name}: expected >= 3 prompt_variants, "
                    f"got {len(rubric['prompt_variants'])}",
                )


# ── 3. TestScoringWeights ────────────────────────────────────────────


class TestScoringWeights(unittest.TestCase):
    """Verify scoring dimension weights sum to 1.0 for each pack."""

    def test_scoring_weights_sum_to_one(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                rubric = _load_rubric(pack_name)
                scoring = rubric["scoring"]

                total_weight = sum(dim["weight"] for dim in scoring.values())
                self.assertAlmostEqual(
                    total_weight,
                    1.0,
                    places=2,
                    msg=f"{pack_name}: scoring weights sum to {total_weight}, "
                    f"expected 1.0",
                )


# ── 4. TestPromptVariants ────────────────────────────────────────────


class TestPromptVariants(unittest.TestCase):
    """Verify each prompt variant has required fields with valid values."""

    def test_prompt_variant_fields(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                rubric = _load_rubric(pack_name)
                variants = rubric["prompt_variants"]

                for i, variant in enumerate(variants):
                    with self.subTest(variant_index=i):
                        self.assertIn(
                            "id",
                            variant,
                            f"{pack_name} variant {i}: missing 'id'",
                        )
                        self.assertIn(
                            "difficulty",
                            variant,
                            f"{pack_name} variant {i}: missing 'difficulty'",
                        )
                        self.assertIn(
                            "prompt",
                            variant,
                            f"{pack_name} variant {i}: missing 'prompt'",
                        )

                        self.assertIn(
                            variant["difficulty"],
                            VALID_DIFFICULTIES,
                            f"{pack_name} variant {i}: difficulty "
                            f"'{variant['difficulty']}' not in "
                            f"{VALID_DIFFICULTIES}",
                        )


# ── 5. TestReferenceDslParses ────────────────────────────────────────


class TestReferenceDslParses(unittest.TestCase):
    """Verify each reference.dsl parses without errors after linting."""

    def test_reference_dsl_parses(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                dsl_text = _load_reference_dsl(pack_name)
                lint_result = lint_dsl(dsl_text)
                try:
                    ir = parse_dsl(lint_result.text)
                except Exception as e:
                    self.fail(f"{pack_name}: reference.dsl failed to parse: {e}")
                self.assertIsNotNone(ir)
                self.assertTrue(
                    hasattr(ir, "name"),
                    f"{pack_name}: parsed IR missing 'name' attribute",
                )


# ── 6. TestReferenceDslValidates ─────────────────────────────────────


class TestReferenceDslValidates(unittest.TestCase):
    """Verify each reference.dsl passes validation (strict=False)."""

    def test_reference_dsl_validates(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                dsl_text = _load_reference_dsl(pack_name)
                lint_result = lint_dsl(dsl_text)
                ir = parse_dsl(lint_result.text)
                result = validate_ir(ir, strict=False)
                self.assertTrue(
                    result.is_valid,
                    f"{pack_name}: reference.dsl validation errors: "
                    f"{[str(e) for e in result.errors]}",
                )


# ── 7. TestReferenceDslCompiles ──────────────────────────────────────


class TestReferenceDslCompiles(unittest.TestCase):
    """Verify each reference.dsl compiles without errors."""

    def test_reference_dsl_compiles(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                dsl_text = _load_reference_dsl(pack_name)
                lint_result = lint_dsl(dsl_text)
                ir = parse_dsl(lint_result.text)
                try:
                    shortcut = compile_ir(ir)
                except Exception as e:
                    self.fail(f"{pack_name}: reference.dsl failed to compile: {e}")
                self.assertIsNotNone(shortcut)


# ── 8. TestReferenceDslActionCount ───────────────────────────────────


class TestReferenceDslActionCount(unittest.TestCase):
    """Verify each reference DSL has a reasonable minimum of actions (>= 5)."""

    def _count_actions(self, statements) -> int:
        """Recursively count ActionStatement nodes in IR statements."""
        from dsl_ir import (
            ActionStatement,
            ForeachBlock,
            IfBlock,
            MenuBlock,
            RepeatBlock,
        )

        count = 0
        for stmt in statements:
            if isinstance(stmt, ActionStatement):
                count += 1
            elif isinstance(stmt, IfBlock):
                count += self._count_actions(stmt.then_body)
                if stmt.else_body:
                    count += self._count_actions(stmt.else_body)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    count += self._count_actions(case.body)
            elif isinstance(stmt, RepeatBlock):
                count += self._count_actions(stmt.body)
            elif isinstance(stmt, ForeachBlock):
                count += self._count_actions(stmt.body)
        return count

    def test_reference_dsl_action_count(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                dsl_text = _load_reference_dsl(pack_name)
                lint_result = lint_dsl(dsl_text)
                ir = parse_dsl(lint_result.text)
                action_count = self._count_actions(ir.statements)
                self.assertGreaterEqual(
                    action_count,
                    5,
                    f"{pack_name}: reference.dsl has only {action_count} "
                    f"actions, expected >= 5",
                )


# ── 9. TestScoreReferenceDsl ─────────────────────────────────────────


class TestScoreReferenceDsl(unittest.TestCase):
    """Score each reference DSL against its own rubric and verify quality."""

    def test_score_reference_dsl(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                pack = load_scenario_pack(SCENARIO_PACKS_DIR / pack_name)
                rubric = pack["rubric"]
                dsl_text = pack["reference_dsl"]

                result = score_dsl_text(dsl_text, rubric, "reference")

                self.assertTrue(
                    result.parsed,
                    f"{pack_name}: reference DSL should parse "
                    f"(errors: {result.errors})",
                )
                self.assertTrue(
                    result.validated,
                    f"{pack_name}: reference DSL should validate "
                    f"(errors: {result.errors})",
                )
                self.assertGreater(
                    result.total_score,
                    0.3,
                    f"{pack_name}: reference DSL score {result.total_score:.3f} "
                    f"is too low (expected > 0.3)",
                )


# ── 10. TestScoringCriteria ──────────────────────────────────────────


class TestScoringCriteria(unittest.TestCase):
    """Verify each criterion has 'description' and 'points' fields."""

    def test_scoring_criteria_fields(self):
        for pack_name in ALL_PACK_NAMES:
            with self.subTest(pack=pack_name):
                rubric = _load_rubric(pack_name)
                scoring = rubric["scoring"]

                for dim_name, dim in scoring.items():
                    self.assertIn(
                        "criteria",
                        dim,
                        f"{pack_name}/{dim_name}: missing 'criteria'",
                    )
                    criteria = dim["criteria"]
                    self.assertIsInstance(criteria, dict)
                    self.assertTrue(
                        len(criteria) > 0,
                        f"{pack_name}/{dim_name}: criteria is empty",
                    )

                    for crit_name, crit in criteria.items():
                        with self.subTest(dimension=dim_name, criterion=crit_name):
                            self.assertIn(
                                "description",
                                crit,
                                f"{pack_name}/{dim_name}/{crit_name}: "
                                f"missing 'description'",
                            )
                            self.assertIn(
                                "points",
                                crit,
                                f"{pack_name}/{dim_name}/{crit_name}: missing 'points'",
                            )
                            points = crit["points"]
                            self.assertGreaterEqual(
                                points,
                                0,
                                f"{pack_name}/{dim_name}/{crit_name}: "
                                f"points={points} < 0",
                            )
                            self.assertLessEqual(
                                points,
                                1,
                                f"{pack_name}/{dim_name}/{crit_name}: "
                                f"points={points} > 1",
                            )


# ── 11. TestScenarioDiscovery ────────────────────────────────────────


class TestScenarioDiscovery(unittest.TestCase):
    """Verify scenario pack auto-discovery finds at least 8 packs."""

    def test_scenario_discovery(self):
        pack_dirs = sorted(
            p
            for p in SCENARIO_PACKS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        self.assertGreaterEqual(
            len(pack_dirs),
            8,
            f"Expected >= 8 scenario packs, found {len(pack_dirs)}: "
            f"{[p.name for p in pack_dirs]}",
        )


if __name__ == "__main__":
    unittest.main()
