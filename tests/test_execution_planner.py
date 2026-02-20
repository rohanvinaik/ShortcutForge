"""
Tests for ExecutionPlanner.

Verifies archetype classification, step building, domain/budget/creative
mode suggestions, plan formatting, and edge cases.
"""

import sys
import unittest
from pathlib import Path

# Ensure scripts/ is on the import path
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from execution_planner import (
    ExecutionPlan,
    ExecutionPlanner,
    format_plan_context,
)


class TestExecutionPlanner(unittest.TestCase):
    """Tests for the ExecutionPlanner class."""

    @classmethod
    def setUpClass(cls):
        cls.planner = ExecutionPlanner()

    # ── 1. test_simple_automation ─────────────────────────────────

    def test_simple_automation(self):
        """'Turn on bluetooth' -> automation or system_control, actions include bluetooth.set"""
        plan = self.planner.plan("Turn on bluetooth")
        self.assertIn(plan.archetype, ("automation", "system_control"))
        all_actions = []
        for step in plan.steps:
            all_actions.extend(step.candidate_actions)
        self.assertIn("bluetooth.set", all_actions)

    # ── 2. test_api_integration ───────────────────────────────────

    def test_api_integration(self):
        """'Download JSON from an API endpoint' -> api_integration, includes downloadurl"""
        plan = self.planner.plan("Download JSON from an API endpoint")
        self.assertEqual(plan.archetype, "api_integration")
        all_actions = []
        for step in plan.steps:
            all_actions.extend(step.candidate_actions)
        self.assertIn("downloadurl", all_actions)

    # ── 3. test_data_transform ────────────────────────────────────

    def test_data_transform(self):
        """'Split text and replace commas' -> data_transform, includes splittext and replacetext"""
        plan = self.planner.plan("Split text and replace commas")
        self.assertEqual(plan.archetype, "data_transform")
        all_actions = []
        for step in plan.steps:
            all_actions.extend(step.candidate_actions)
        self.assertIn("splittext", all_actions)
        self.assertIn("replacetext", all_actions)

    # ── 4. test_interactive_app ───────────────────────────────────

    def test_interactive_app(self):
        """'Show a menu and ask the user to choose' -> interactive_app"""
        plan = self.planner.plan("Show a menu and ask the user to choose")
        self.assertEqual(plan.archetype, "interactive_app")

    # ── 5. test_system_control ────────────────────────────────────

    def test_system_control(self):
        """'Set brightness and volume' -> system_control"""
        plan = self.planner.plan("Set brightness and volume")
        self.assertEqual(plan.archetype, "system_control")

    # ── 6. test_hybrid_archetype ──────────────────────────────────

    def test_hybrid_archetype(self):
        """Prompt spanning two domains -> hybrid (api + interactive)"""
        plan = self.planner.plan(
            "Fetch JSON from API endpoint and ask user to choose from menu"
        )
        self.assertEqual(plan.archetype, "hybrid")

    # ── 7. test_plan_step_count ───────────────────────────────────

    def test_plan_step_count(self):
        """Multi-step prompt produces correct number of steps."""
        plan = self.planner.plan(
            "Download JSON from API, filter results, save to file, and show notification"
        )
        # Should have multiple steps (at least 3)
        self.assertGreaterEqual(len(plan.steps), 3)

    # ── 8. test_suggested_domain_health ───────────────────────────

    def test_suggested_domain_health(self):
        """'Log caffeine to health' -> suggested_domain: health_logger"""
        plan = self.planner.plan("Log caffeine to health")
        self.assertEqual(plan.suggested_domain, "health_logger")

    # ── 9. test_suggested_domain_api ──────────────────────────────

    def test_suggested_domain_api(self):
        """'Fetch JSON from API' -> suggested_domain: api_workflow"""
        plan = self.planner.plan("Fetch JSON from API")
        self.assertEqual(plan.suggested_domain, "api_workflow")

    # ── 10. test_suggested_budget_simple ──────────────────────────

    def test_suggested_budget_simple(self):
        """Simple single-action prompt -> budget: simple"""
        plan = self.planner.plan("Turn on bluetooth")
        self.assertEqual(plan.suggested_budget, "simple")

    # ── 11. test_suggested_budget_complex ─────────────────────────

    def test_suggested_budget_complex(self):
        """Multi-step complex prompt -> budget: complex or very_complex"""
        plan = self.planner.plan(
            "Download JSON from API, split text, replace commas, "
            "filter results, save to file, and show notification"
        )
        self.assertIn(plan.suggested_budget, ("complex", "very_complex"))

    # ── 12. test_suggested_creative_mode ──────────────────────────

    def test_suggested_creative_mode(self):
        """Different archetypes produce correct creative modes."""
        # data_transform -> pragmatic
        plan_dt = self.planner.plan("Split text and replace commas")
        self.assertEqual(plan_dt.suggested_creative_mode, "pragmatic")

        # interactive_app -> expressive
        plan_ia = self.planner.plan("Show a menu and ask the user to choose")
        self.assertEqual(plan_ia.suggested_creative_mode, "expressive")

        # system_control -> automation_dense
        plan_sc = self.planner.plan("Set brightness and volume")
        self.assertEqual(plan_sc.suggested_creative_mode, "automation_dense")

        # hybrid -> power_user
        plan_hy = self.planner.plan(
            "Fetch JSON from API endpoint and ask user to choose from menu"
        )
        self.assertEqual(plan_hy.suggested_creative_mode, "power_user")

    # ── 13. test_format_plan_context ──────────────────────────────

    def test_format_plan_context(self):
        """format_plan_context produces non-empty string with expected structure."""
        plan = self.planner.plan("Download weather data and show a notification")
        context = format_plan_context(plan)
        self.assertIsInstance(context, str)
        self.assertGreater(len(context), 0)
        self.assertIn("## Execution Plan", context)
        self.assertIn("Archetype:", context)
        self.assertIn("Steps:", context)
        self.assertIn("Suggested domain:", context)
        self.assertIn("Suggested budget:", context)
        self.assertIn("Suggested creative mode:", context)

    # ── 14. test_empty_prompt ─────────────────────────────────────

    def test_empty_prompt(self):
        """Empty prompt doesn't crash, returns default plan."""
        plan = self.planner.plan("")
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(plan.archetype, "general")
        self.assertEqual(plan.confidence, 0.0)
        self.assertEqual(plan.suggested_domain, "general")
        self.assertEqual(plan.suggested_budget, "simple")

        # Also test None-like empty strings
        plan2 = self.planner.plan("   ")
        self.assertIsInstance(plan2, ExecutionPlan)
        self.assertEqual(plan2.archetype, "general")

    # ── 15. test_unknown_verbs ────────────────────────────────────

    def test_unknown_verbs(self):
        """'Flibbertigibbet the glorb' -> still produces a valid plan (general archetype)."""
        plan = self.planner.plan("Flibbertigibbet the glorb")
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertEqual(plan.archetype, "general")
        self.assertEqual(plan.confidence, 0.0)
        self.assertEqual(plan.suggested_domain, "general")
        # Should still have at least one step (generic fallback)
        self.assertGreaterEqual(len(plan.steps), 1)


if __name__ == "__main__":
    unittest.main()
