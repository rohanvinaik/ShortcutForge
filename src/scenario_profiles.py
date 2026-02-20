"""
ShortcutForge Scenario Profile Manager.

Bundles domain_profile + creative_mode + budget_override + system_prompt_addendum
into named scenarios for pipeline customization.

Profiles:
  - health_tracking: HealthKit workflows, pragmatic mode, complex budget
  - api_integration: REST API workflows, pragmatic mode, complex budget
  - mini_app: Interactive mini-apps, expressive mode, very complex budget
  - quick_automation: Simple device automations, pragmatic mode, simple budget
  - file_processing: File management and routing workflows
  - calendar_management: Calendar event management and triage workflows
  - clipboard_tools: Clipboard management and text cleaning utilities
  - media_workflow: Media processing and metadata extraction workflows
  - daily_briefing: Multi-source daily briefing and morning routine workflows

Selection is based on prompt keyword matching (same approach as DomainProfileManager).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__version__ = "1.0"


# ── Data Classes ─────────────────────────────────────────────────


@dataclass
class ScenarioProfile:
    """A scenario profile that bundles pipeline configuration."""

    scenario_id: str
    domain_profile: str = "general"
    creative_mode: str = "pragmatic"
    budget_override: str | None = None  # "simple", "medium", "complex", "very_complex"
    system_prompt_addendum: str = ""
    keywords: list[str] = field(default_factory=list)
    description: str = ""

    def __repr__(self) -> str:
        return (
            f"ScenarioProfile({self.scenario_id}, "
            f"domain={self.domain_profile}, "
            f"mode={self.creative_mode}, "
            f"budget={self.budget_override or 'auto'})"
        )


# ── Built-in Scenarios ──────────────────────────────────────────

_BUILTIN_SCENARIOS: dict[str, ScenarioProfile] = {
    "health_tracking": ScenarioProfile(
        scenario_id="health_tracking",
        domain_profile="health_logger",
        creative_mode="pragmatic",
        budget_override="complex",
        system_prompt_addendum=(
            "This is a health-tracking shortcut. Ensure all HealthKit sample types "
            "use the correct identifiers and units. Log each nutrient individually "
            "with proper error checking. Provide user feedback after logging."
        ),
        keywords=[
            "health",
            "healthkit",
            "nutrient",
            "vitamin",
            "mineral",
            "caffeine",
            "supplement",
            "workout",
            "fitness",
            "calorie",
            "protein",
            "log health",
            "apple health",
            "health sample",
        ],
        description="HealthKit health logging workflows",
    ),
    "api_integration": ScenarioProfile(
        scenario_id="api_integration",
        domain_profile="api_workflow",
        creative_mode="pragmatic",
        budget_override="complex",
        system_prompt_addendum=(
            "This is an API integration shortcut. Always include error handling "
            "after network requests. Parse JSON responses with detect.dictionary. "
            "Use appropriate HTTP methods and headers. Show clear feedback."
        ),
        keywords=[
            "api",
            "http",
            "json",
            "fetch",
            "server",
            "webhook",
            "rest",
            "endpoint",
            "request",
            "response",
            "download",
            "post data",
            "get request",
        ],
        description="REST API integration workflows",
    ),
    "mini_app": ScenarioProfile(
        scenario_id="mini_app",
        domain_profile="general",
        creative_mode="expressive",
        budget_override="very_complex",
        system_prompt_addendum=(
            "This shortcut should function like a mini-app with rich user interaction. "
            "Use MENU blocks for navigation, proper variable management for state, "
            "and multiple UI actions (alerts, prompts, showresult) for feedback. "
            "Include error handling and a polished user experience."
        ),
        keywords=[
            "app",
            "interactive",
            "quiz",
            "game",
            "tracker",
            "manager",
            "organizer",
            "planner",
            "dashboard",
            "mini app",
            "rich ui",
            "multiple screens",
        ],
        description="Interactive mini-app style shortcuts",
    ),
    "quick_automation": ScenarioProfile(
        scenario_id="quick_automation",
        domain_profile="general",
        creative_mode="pragmatic",
        budget_override="simple",
        system_prompt_addendum="",  # No addendum needed for simple automation
        keywords=[
            "toggle",
            "turn on",
            "turn off",
            "set brightness",
            "set volume",
            "open app",
            "launch",
            "quick",
            "simple",
            "fast",
            "one tap",
        ],
        description="Simple device automations and toggles",
    ),
    "file_processing": ScenarioProfile(
        scenario_id="file_processing",
        domain_profile="file_operations",
        creative_mode="pragmatic",
        budget_override="complex",
        system_prompt_addendum=(
            "This shortcut handles file operations. Use properties.files to get file "
            "details. Use MENU or IF chains to route files by type. Handle common "
            "formats (images, PDFs, text) with appropriate actions."
        ),
        keywords=[
            "file",
            "folder",
            "document",
            "pdf",
            "rename",
            "organize",
            "sort files",
            "file type",
            "extension",
            "move files",
            "import",
            "export",
            "save file",
            "select file",
        ],
        description="File management and routing workflows",
    ),
    "calendar_management": ScenarioProfile(
        scenario_id="calendar_management",
        domain_profile="scheduling",
        creative_mode="pragmatic",
        budget_override="complex",
        system_prompt_addendum=(
            "This is a calendar management shortcut. Use getcalendarevents or "
            "filter.calendarevents to query events. Use properties.calendarevents "
            "to extract event details. Group or categorize events as needed."
        ),
        keywords=[
            "calendar",
            "event",
            "events",
            "schedule",
            "meeting",
            "appointment",
            "agenda",
            "today's events",
            "upcoming",
            "calendar triage",
            "event summary",
        ],
        description="Calendar event management and triage workflows",
    ),
    "clipboard_tools": ScenarioProfile(
        scenario_id="clipboard_tools",
        domain_profile="general",
        creative_mode="pragmatic",
        budget_override="medium",
        system_prompt_addendum=(
            "This shortcut works with clipboard content. Use getclipboard and "
            "setclipboard for clipboard operations. Use MENU for multiple clipboard "
            "actions. Consider using Notes for clipboard history."
        ),
        keywords=[
            "clipboard",
            "copy",
            "paste",
            "clean",
            "clipboard history",
            "clean clipboard",
            "clipboard manager",
            "text clean",
            "clipboard utility",
        ],
        description="Clipboard management and text cleaning utilities",
    ),
    "media_workflow": ScenarioProfile(
        scenario_id="media_workflow",
        domain_profile="media_processing",
        creative_mode="pragmatic",
        budget_override="complex",
        system_prompt_addendum=(
            "This shortcut processes media files. Use getimagedetail for image "
            "metadata. Use image.resize for resizing. Use FOREACH to process "
            "multiple items. Generate summary reports of media properties."
        ),
        keywords=[
            "photo",
            "image",
            "video",
            "media",
            "resize",
            "metadata",
            "exif",
            "thumbnail",
            "batch process",
            "photo info",
            "image detail",
        ],
        description="Media processing and metadata extraction workflows",
    ),
    "daily_briefing": ScenarioProfile(
        scenario_id="daily_briefing",
        domain_profile="general",
        creative_mode="expressive",
        budget_override="very_complex",
        system_prompt_addendum=(
            "This is a comprehensive daily briefing shortcut. Gather information "
            "from multiple sources: weather, calendar, news, travel time. Compile "
            "everything into a spoken or displayed briefing. Make it time-aware."
        ),
        keywords=[
            "morning",
            "briefing",
            "daily",
            "routine",
            "morning routine",
            "daily summary",
            "weather check",
            "commute",
            "news",
            "daily briefing",
        ],
        description="Multi-source daily briefing and morning routine workflows",
    ),
}


# ── Scenario Profile Manager ────────────────────────────────────


class ScenarioProfileManager:
    """Manages scenario profiles for pipeline customization.

    Selects the best-matching scenario based on prompt keywords.
    Falls back to a default scenario with no customization.
    """

    def __init__(self, extra_scenarios: dict[str, ScenarioProfile] | None = None):
        self._scenarios = dict(_BUILTIN_SCENARIOS)
        if extra_scenarios:
            self._scenarios.update(extra_scenarios)

    def list_scenarios(self) -> list[str]:
        """List all available scenario IDs."""
        return list(self._scenarios.keys())

    def get_scenario(self, scenario_id: str) -> ScenarioProfile | None:
        """Get a specific scenario by ID."""
        return self._scenarios.get(scenario_id)

    def select_scenario(self, prompt: str) -> ScenarioProfile:
        """Select the best-matching scenario for a prompt.

        Uses keyword scoring (same approach as DomainProfileManager).
        Falls back to a default scenario if no strong match.

        Args:
            prompt: The user's natural language prompt.

        Returns:
            The best-matching ScenarioProfile.
        """
        prompt_lower = prompt.lower()
        prompt_words = set(re.findall(r"\w+", prompt_lower))

        best_id = None
        best_score = 0

        for sid, scenario in self._scenarios.items():
            score = 0
            for keyword in scenario.keywords:
                kw_lower = keyword.lower()
                if " " in kw_lower:
                    if kw_lower in prompt_lower:
                        score += 3
                else:
                    if kw_lower in prompt_words:
                        score += 1

            if score > best_score:
                best_score = score
                best_id = sid

        # Require minimum 2 score to avoid false positives
        if best_score < 2 or best_id is None:
            return self._default_scenario()

        return self._scenarios[best_id]

    @staticmethod
    def _default_scenario() -> ScenarioProfile:
        """Return the default scenario (no customization)."""
        return ScenarioProfile(
            scenario_id="default",
            domain_profile="general",
            creative_mode="pragmatic",
            description="Default scenario (no specialized configuration)",
        )


# ── Convenience ──────────────────────────────────────────────────

_manager: ScenarioProfileManager | None = None


def get_scenario_manager() -> ScenarioProfileManager:
    """Get the global scenario profile manager (singleton)."""
    global _manager
    if _manager is None:
        _manager = ScenarioProfileManager()
    return _manager


def select_scenario(prompt: str) -> ScenarioProfile:
    """Select the best scenario for a prompt (convenience function)."""
    return get_scenario_manager().select_scenario(prompt)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    mgr = ScenarioProfileManager()
    scenarios = mgr.list_scenarios()
    print(f"Available scenarios ({len(scenarios)}): {', '.join(scenarios)}\n")

    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        scenario = mgr.select_scenario(prompt)
        print(f"Prompt: {prompt!r}")
        print(f"Selected: {scenario}")
        if scenario.system_prompt_addendum:
            print(f"Addendum: {scenario.system_prompt_addendum[:100]}...")
    else:
        for sid in scenarios:
            s = mgr.get_scenario(sid)
            if s:
                print(f"  {sid}: {s.description}")
                print(
                    f"    Domain: {s.domain_profile}, Mode: {s.creative_mode}, Budget: {s.budget_override}"
                )
                print(f"    Keywords: {s.keywords[:5]}...")
