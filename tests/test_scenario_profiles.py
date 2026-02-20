"""
Tests for scenario_profiles.py -- Phase 3 scenario profile system.

Tests cover:
  - All 9 scenarios (4 original + 5 new) are registered
  - Each scenario has required fields (scenario_id, domain_profile, creative_mode, keywords, description)
  - Keyword matching works for each new scenario
  - Default scenario is returned for non-matching prompts
  - ScenarioProfileManager lists all scenarios
  - Version is "1.0"

Run: python3 scripts/test_scenario_profiles.py -v
"""

import sys
import unittest
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from scenario_profiles import (
    ScenarioProfile,
    ScenarioProfileManager,
    __version__,
    select_scenario,
)

ALL_SCENARIO_IDS = [
    "health_tracking",
    "api_integration",
    "mini_app",
    "quick_automation",
    "file_processing",
    "calendar_management",
    "clipboard_tools",
    "media_workflow",
    "daily_briefing",
]

REQUIRED_FIELDS = [
    "scenario_id",
    "domain_profile",
    "creative_mode",
    "keywords",
    "description",
]


class TestVersion(unittest.TestCase):
    """Module version is '1.0'."""

    def test_version_is_1_0(self):
        self.assertEqual(__version__, "1.0")


class TestAllScenariosRegistered(unittest.TestCase):
    """All 9 scenarios (4 original + 5 new) are registered."""

    def setUp(self):
        self.mgr = ScenarioProfileManager()

    def test_total_scenario_count(self):
        scenarios = self.mgr.list_scenarios()
        self.assertEqual(
            len(scenarios),
            9,
            f"Expected 9 scenarios, got {len(scenarios)}: {scenarios}",
        )

    def test_all_scenario_ids_present(self):
        scenarios = self.mgr.list_scenarios()
        for sid in ALL_SCENARIO_IDS:
            with self.subTest(scenario_id=sid):
                self.assertIn(sid, scenarios)


class TestRequiredFields(unittest.TestCase):
    """Each scenario has required fields: scenario_id, domain_profile, creative_mode, keywords, description."""

    def setUp(self):
        self.mgr = ScenarioProfileManager()

    def test_each_scenario_has_required_fields(self):
        for sid in ALL_SCENARIO_IDS:
            with self.subTest(scenario_id=sid):
                scenario = self.mgr.get_scenario(sid)
                self.assertIsNotNone(scenario, f"Scenario '{sid}' not found")
                for field_name in REQUIRED_FIELDS:
                    with self.subTest(field=field_name):
                        value = getattr(scenario, field_name, None)
                        self.assertIsNotNone(
                            value, f"Field '{field_name}' is None for scenario '{sid}'"
                        )
                        if field_name == "keywords":
                            self.assertIsInstance(value, list)
                            self.assertGreater(
                                len(value), 0, f"keywords list is empty for '{sid}'"
                            )
                        else:
                            self.assertIsInstance(value, str)
                            self.assertGreater(
                                len(value),
                                0,
                                f"Field '{field_name}' is empty for '{sid}'",
                            )


class TestOriginalScenarios(unittest.TestCase):
    """Original 4 scenarios have correct configuration."""

    def setUp(self):
        self.mgr = ScenarioProfileManager()

    def test_health_tracking(self):
        s = self.mgr.get_scenario("health_tracking")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "health_logger")
        self.assertEqual(s.creative_mode, "pragmatic")
        self.assertEqual(s.budget_override, "complex")

    def test_api_integration(self):
        s = self.mgr.get_scenario("api_integration")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "api_workflow")
        self.assertEqual(s.creative_mode, "pragmatic")
        self.assertEqual(s.budget_override, "complex")

    def test_mini_app(self):
        s = self.mgr.get_scenario("mini_app")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "general")
        self.assertEqual(s.creative_mode, "expressive")
        self.assertEqual(s.budget_override, "very_complex")

    def test_quick_automation(self):
        s = self.mgr.get_scenario("quick_automation")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "general")
        self.assertEqual(s.creative_mode, "pragmatic")
        self.assertEqual(s.budget_override, "simple")


class TestNewScenarioConfigs(unittest.TestCase):
    """New 5 scenarios have correct configuration."""

    def setUp(self):
        self.mgr = ScenarioProfileManager()

    def test_file_processing(self):
        s = self.mgr.get_scenario("file_processing")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "file_operations")
        self.assertEqual(s.creative_mode, "pragmatic")
        self.assertEqual(s.budget_override, "complex")
        self.assertEqual(s.description, "File management and routing workflows")

    def test_calendar_management(self):
        s = self.mgr.get_scenario("calendar_management")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "scheduling")
        self.assertEqual(s.creative_mode, "pragmatic")
        self.assertEqual(s.budget_override, "complex")
        self.assertEqual(
            s.description, "Calendar event management and triage workflows"
        )

    def test_clipboard_tools(self):
        s = self.mgr.get_scenario("clipboard_tools")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "general")
        self.assertEqual(s.creative_mode, "pragmatic")
        self.assertEqual(s.budget_override, "medium")
        self.assertEqual(
            s.description, "Clipboard management and text cleaning utilities"
        )

    def test_media_workflow(self):
        s = self.mgr.get_scenario("media_workflow")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "media_processing")
        self.assertEqual(s.creative_mode, "pragmatic")
        self.assertEqual(s.budget_override, "complex")
        self.assertEqual(
            s.description, "Media processing and metadata extraction workflows"
        )

    def test_daily_briefing(self):
        s = self.mgr.get_scenario("daily_briefing")
        self.assertIsNotNone(s)
        self.assertEqual(s.domain_profile, "general")
        self.assertEqual(s.creative_mode, "expressive")
        self.assertEqual(s.budget_override, "very_complex")
        self.assertEqual(
            s.description, "Multi-source daily briefing and morning routine workflows"
        )


class TestKeywordMatching(unittest.TestCase):
    """Keyword matching works for each new scenario."""

    def setUp(self):
        self.mgr = ScenarioProfileManager()

    def test_file_processing_match(self):
        scenario = self.mgr.select_scenario("organize my files by type and rename them")
        self.assertEqual(scenario.scenario_id, "file_processing")

    def test_calendar_management_match(self):
        scenario = self.mgr.select_scenario(
            "show me today's calendar events grouped by category"
        )
        self.assertEqual(scenario.scenario_id, "calendar_management")

    def test_clipboard_tools_match(self):
        scenario = self.mgr.select_scenario("clean my clipboard and save to notes")
        self.assertEqual(scenario.scenario_id, "clipboard_tools")

    def test_media_workflow_match(self):
        scenario = self.mgr.select_scenario(
            "batch resize my photos and extract metadata"
        )
        self.assertEqual(scenario.scenario_id, "media_workflow")

    def test_daily_briefing_match(self):
        scenario = self.mgr.select_scenario(
            "create a morning briefing with weather and calendar"
        )
        self.assertEqual(scenario.scenario_id, "daily_briefing")

    # Original scenario matching still works
    def test_health_tracking_match(self):
        scenario = self.mgr.select_scenario("Log caffeine and vitamins to health")
        self.assertEqual(scenario.scenario_id, "health_tracking")

    def test_api_integration_match(self):
        scenario = self.mgr.select_scenario("Fetch JSON from REST API endpoint")
        self.assertEqual(scenario.scenario_id, "api_integration")

    def test_mini_app_match(self):
        scenario = self.mgr.select_scenario("Build an interactive quiz app")
        self.assertEqual(scenario.scenario_id, "mini_app")


class TestDefaultFallback(unittest.TestCase):
    """Default scenario is returned for non-matching prompts."""

    def setUp(self):
        self.mgr = ScenarioProfileManager()

    def test_generic_prompt_returns_default(self):
        scenario = self.mgr.select_scenario("Set a 5 minute timer")
        self.assertEqual(scenario.scenario_id, "default")

    def test_nonsense_prompt_returns_default(self):
        scenario = self.mgr.select_scenario("xyzzy foobarbaz")
        self.assertEqual(scenario.scenario_id, "default")

    def test_default_has_general_domain(self):
        scenario = self.mgr.select_scenario("something unrelated")
        self.assertEqual(scenario.domain_profile, "general")
        self.assertEqual(scenario.creative_mode, "pragmatic")


class TestScenarioProfileManagerListAll(unittest.TestCase):
    """ScenarioProfileManager lists all scenarios."""

    def test_list_scenarios_returns_all_ids(self):
        mgr = ScenarioProfileManager()
        listed = mgr.list_scenarios()
        self.assertEqual(sorted(listed), sorted(ALL_SCENARIO_IDS))

    def test_get_scenario_returns_none_for_unknown(self):
        mgr = ScenarioProfileManager()
        self.assertIsNone(mgr.get_scenario("nonexistent_scenario"))


class TestConvenienceFunction(unittest.TestCase):
    """select_scenario() convenience function works."""

    def test_convenience_returns_scenario_profile(self):
        scenario = select_scenario("Log caffeine and vitamins to health")
        self.assertIsInstance(scenario, ScenarioProfile)
        self.assertEqual(scenario.scenario_id, "health_tracking")


if __name__ == "__main__":
    unittest.main()
