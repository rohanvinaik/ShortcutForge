"""
ExecutionPlanner for ShortcutForge.

Produces a structured execution plan from a natural language prompt before
DSL generation begins. The plan influences domain selection, budget, and
creative mode, and gets injected into the generation prompt as structured context.

The planner works by:
  1. Tokenizing the prompt into words
  2. Matching words against a verb-to-action mapping
  3. Matching against composition patterns from the corpus
  4. Classifying the prompt archetype
  5. Building discrete plan steps
  6. Suggesting domain, budget, and creative mode

Usage:
    from execution_planner import ExecutionPlanner

    planner = ExecutionPlanner()
    plan = planner.plan("Download weather data and show a notification")
    print(format_plan_context(plan))

CLI:
    python scripts/execution_planner.py "Create a shortcut that ..."
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__version__ = "1.0"

# ── Paths ──────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_REFS_DIR = _SCRIPT_DIR.parent / "references"
_ACTION_CATALOG_PATH = _REFS_DIR / "action_catalog.json"
_COMPOSITION_PATTERNS_PATH = _REFS_DIR / "composition_patterns.json"


# ── Verb-to-Action Mapping ────────────────────────────────────────────

_VERB_TO_ACTIONS: dict[str, list[str]] = {
    # Text processing
    "split": ["splittext"],
    "replace": ["replacetext"],
    "uppercase": ["changecase"],
    "lowercase": ["changecase"],
    "trim": ["changecase"],
    "count": ["count"],
    "combine": ["text"],
    "format": ["text"],
    # Networking
    "download": ["downloadurl"],
    "fetch": ["downloadurl"],
    "request": ["downloadurl"],
    "upload": ["downloadurl"],
    "api": ["downloadurl", "url"],
    # Media
    "resize": ["resizeimage"],
    "crop": ["cropimage"],
    "photo": ["selectphotos", "takephoto"],
    "screenshot": ["takescreenshot"],
    "gif": ["makegif"],
    "video": ["trimvideo"],
    # Communication
    "message": ["sendmessage"],
    "email": ["sendemail"],
    "text": ["sendmessage"],
    "call": ["phone"],
    "share": ["share"],
    # Calendar
    "event": ["addnewevent"],
    "reminder": ["addnewreminder"],
    "calendar": ["getcalendarevents"],
    "schedule": ["addnewevent"],
    # Files
    "save": ["savefile"],
    "open": ["openin", "openurl"],
    "file": ["getfile", "savefile"],
    "rename": ["rename"],
    "folder": ["createfolder"],
    # Health
    "health": ["health.quantity.log"],
    "caffeine": ["health.quantity.log"],
    "nutrient": ["health.quantity.log"],
    "workout": ["health.workout.log"],
    # System
    "bluetooth": ["bluetooth.set"],
    "wifi": ["wifi.set"],
    "brightness": ["setbrightness"],
    "volume": ["setvolume"],
    "timer": ["starttimer"],
    "alarm": ["createalarm"],
    # UI
    "menu": ["choosefromlist"],
    "ask": ["askforinput"],
    "input": ["askforinput"],
    "alert": ["alert"],
    "notify": ["notification"],
    "notification": ["notification"],
    "speak": ["speaktext"],
    "clipboard": ["getclipboard", "setclipboard"],
    "copy": ["setclipboard"],
    "paste": ["getclipboard"],
    # Data
    "dictionary": ["detect.dictionary"],
    "json": ["detect.dictionary", "downloadurl"],
    "list": ["choosefromlist"],
    "filter": ["filter.files"],
    "sort": ["filter.files"],
    "random": ["number.random"],
}


# ── Archetype Signal Keywords ─────────────────────────────────────────

_ARCHETYPE_SIGNALS: dict[str, set[str]] = {
    "data_transform": {
        "split",
        "replace",
        "format",
        "convert",
        "transform",
        "parse",
        "extract",
        "clean",
        "text",
        "uppercase",
        "lowercase",
        "trim",
        "combine",
        "regex",
    },
    "automation": {
        "set",
        "toggle",
        "enable",
        "disable",
        "turn",
        "start",
        "stop",
        "timer",
        "alarm",
        "automate",
        "routine",
        "schedule",
        "batch",
        "trigger",
    },
    "interactive_app": {
        "menu",
        "ask",
        "choose",
        "input",
        "prompt",
        "select",
        "pick",
        "quiz",
        "game",
        "dialog",
        "question",
    },
    "api_integration": {
        "api",
        "fetch",
        "download",
        "url",
        "json",
        "webhook",
        "server",
        "endpoint",
        "http",
        "rest",
        "request",
    },
    "system_control": {
        "bluetooth",
        "wifi",
        "volume",
        "brightness",
        "airplane",
        "dnd",
        "focus",
        "wallpaper",
        "flashlight",
        "cellular",
        "appearance",
        "dark",
        "light",
    },
}


# ── Domain Mapping ────────────────────────────────────────────────────

_ACTION_TO_DOMAIN: dict[str, str] = {
    "health.quantity.log": "health_logger",
    "health.workout.log": "health_logger",
    "filter.health.quantity": "health_logger",
    "downloadurl": "api_workflow",
    "url": "api_workflow",
    "detect.dictionary": "api_workflow",
    "sendmessage": "messaging",
    "sendemail": "messaging",
    "resizeimage": "media_processing",
    "cropimage": "media_processing",
    "makegif": "media_processing",
    "trimvideo": "media_processing",
    "selectphotos": "media_processing",
    "takephoto": "media_processing",
    "takescreenshot": "media_processing",
    "getfile": "file_operations",
    "savefile": "file_operations",
    "createfolder": "file_operations",
    "filter.files": "file_operations",
    "addnewevent": "scheduling",
    "addnewreminder": "scheduling",
    "getcalendarevents": "scheduling",
    "notification": "notifications",
    "alert": "notifications",
}

# ── Creative Mode Mapping ─────────────────────────────────────────────

_ARCHETYPE_TO_CREATIVE: dict[str, str] = {
    "data_transform": "pragmatic",
    "automation": "automation_dense",
    "interactive_app": "expressive",
    "api_integration": "pragmatic",
    "system_control": "automation_dense",
    "hybrid": "power_user",
    "general": "pragmatic",
}


# ── Data Classes ──────────────────────────────────────────────────────


@dataclass
class PlanStep:
    """A single step in the execution plan."""

    description: str  # Human-readable step description
    candidate_actions: list[str] = field(default_factory=list)
    pattern_match: str | None = None  # Matched composition pattern name
    estimated_complexity: str = "simple"  # simple/medium/complex


@dataclass
class ExecutionPlan:
    """Complete execution plan for a prompt."""

    archetype: str  # data_transform, automation, interactive_app, api_integration, system_control, hybrid, general
    steps: list[PlanStep] = field(default_factory=list)
    suggested_domain: str = "general"
    suggested_budget: str = "medium"
    suggested_creative_mode: str = "pragmatic"
    confidence: float = 0.0  # 0-1, how confident the planner is
    raw_prompt: str = ""


# ── ExecutionPlanner ──────────────────────────────────────────────────


class ExecutionPlanner:
    """Produces structured execution plans from natural language prompts.

    Lazily loads the action catalog and composition patterns on first call
    to plan(). Plans are used to inform domain selection, token budget,
    and creative mode before DSL generation begins.
    """

    def __init__(self):
        self._catalog: dict[str, Any] | None = None
        self._patterns: dict[str, Any] | None = None
        self._canonical_map: dict[str, str] = {}
        self._idiom_names: list[str] = []
        self._loaded = False

    # ── Lazy Loading ──────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load action catalog and composition patterns on first use."""
        if self._loaded:
            return

        # Load action catalog
        try:
            with open(_ACTION_CATALOG_PATH) as f:
                self._catalog = json.load(f)
            self._canonical_map = self._catalog.get("_meta", {}).get(
                "canonical_map", {}
            )
        except (FileNotFoundError, json.JSONDecodeError):
            self._catalog = {}
            self._canonical_map = {}

        # Load composition patterns
        try:
            with open(_COMPOSITION_PATTERNS_PATH) as f:
                self._patterns = json.load(f)
            idioms = self._patterns.get("idioms", {})
            self._idiom_names = list(idioms.keys())
        except (FileNotFoundError, json.JSONDecodeError):
            self._patterns = {}
            self._idiom_names = []

        self._loaded = True

    # ── Public API ────────────────────────────────────────────────

    def plan(self, prompt: str) -> ExecutionPlan:
        """Generate an execution plan for a natural language prompt.

        Args:
            prompt: The user's description of the shortcut to create.

        Returns:
            An ExecutionPlan with archetype, steps, and suggestions.
        """
        self._ensure_loaded()

        if not prompt or not prompt.strip():
            return ExecutionPlan(
                archetype="general",
                steps=[],
                suggested_domain="general",
                suggested_budget="simple",
                suggested_creative_mode="pragmatic",
                confidence=0.0,
                raw_prompt=prompt or "",
            )

        chunks = self._tokenize_prompt(prompt)
        action_matches = self._match_actions(chunks)
        pattern_matches = self._match_patterns(chunks, action_matches)
        archetype = self._classify_archetype(chunks, action_matches)
        steps = self._build_steps(chunks, action_matches, pattern_matches)
        domain = self._suggest_domain(chunks, action_matches, archetype)
        budget = self._suggest_budget(steps, archetype)
        creative = self._suggest_creative_mode(archetype, steps)
        confidence = self._compute_confidence(action_matches, pattern_matches)

        return ExecutionPlan(
            archetype=archetype,
            steps=steps,
            suggested_domain=domain,
            suggested_budget=budget,
            suggested_creative_mode=creative,
            confidence=confidence,
            raw_prompt=prompt,
        )

    # ── Prompt Tokenization ───────────────────────────────────────

    def _tokenize_prompt(self, prompt: str) -> list[str]:
        """Tokenize prompt into lowercase words.

        Returns a list of words extracted from the prompt, lowercased
        and stripped of punctuation.
        """
        lowered = prompt.lower()
        words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", lowered)
        return words

    # ── Action Matching ───────────────────────────────────────────

    def _match_actions(self, chunks: list[str]) -> dict[str, list[str]]:
        """Match tokenized words against verb-to-action mapping.

        Returns a dict mapping each matched word to its candidate
        action IDs.
        """
        matches: dict[str, list[str]] = {}
        for word in chunks:
            if word in _VERB_TO_ACTIONS:
                matches[word] = _VERB_TO_ACTIONS[word]
        return matches

    # ── Pattern Matching ──────────────────────────────────────────

    def _match_patterns(
        self,
        chunks: list[str],
        action_matches: dict[str, list[str]],
    ) -> list[str]:
        """Match against composition pattern idioms.

        Checks if the prompt words or matched actions correspond to
        known idioms from composition_patterns.json.

        Returns a list of matched idiom names.
        """
        matched_idioms: list[str] = []
        all_actions = set()
        for actions in action_matches.values():
            all_actions.update(actions)

        # Map idiom names to keyword heuristics
        _idiom_keywords: dict[str, set[str]] = {
            "api_fetch_chain": {"api", "fetch", "download", "url", "json"},
            "text_split_pipeline": {"split", "text", "replace", "process"},
            "device_settings_batch": {
                "bluetooth",
                "wifi",
                "volume",
                "brightness",
                "set",
            },
            "clipboard_roundtrip": {"clipboard", "copy", "paste"},
            "health_data": {"health", "caffeine", "workout", "nutrient", "log"},
            "multi_menu": {"menu", "choose", "select", "pick"},
            "conditional_chain": {"if", "check", "compare", "conditional"},
            "repeat_each": {"each", "every", "iterate", "loop", "repeat"},
            "dictionary_routing": {"dictionary", "json", "route", "lookup"},
            "date_arithmetic": {"date", "time", "schedule", "offset", "duration"},
            "large_menu": {"menu", "options", "launcher", "hub"},
            "sub_shortcut": {"shortcut", "delegate", "run"},
            "smart_input_handler": {"input", "clipboard", "fallback"},
        }

        chunk_set = set(chunks)
        for idiom_name in self._idiom_names:
            keywords = _idiom_keywords.get(idiom_name, set())
            if not keywords:
                continue
            overlap = chunk_set & keywords
            if len(overlap) >= 2:
                matched_idioms.append(idiom_name)

        return matched_idioms

    # ── Archetype Classification ──────────────────────────────────

    def _classify_archetype(
        self,
        chunks: list[str],
        action_matches: dict[str, list[str]],
    ) -> str:
        """Classify the prompt into an archetype based on keyword signals.

        Scores each archetype by counting overlapping keywords from the
        prompt. If two or more archetypes tie or score within 1 of each
        other, classifies as 'hybrid'. Returns 'general' if no archetype
        scores above 0.
        """
        chunk_set = set(chunks)
        scores: dict[str, int] = {}

        for archetype, signals in _ARCHETYPE_SIGNALS.items():
            score = len(chunk_set & signals)
            if score > 0:
                scores[archetype] = score

        if not scores:
            return "general"

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = sorted_scores[0]

        # Check for hybrid: 2+ archetypes from different domains with
        # meaningful overlap.  Both archetypes must score >= 2 to indicate
        # genuine multi-domain intent, not incidental keyword overlap.
        if len(sorted_scores) >= 2:
            second_name, second_score = sorted_scores[1]
            if second_score >= 2 and best_score >= 2:
                return "hybrid"

        return best_name

    # ── Step Building ─────────────────────────────────────────────

    def _build_steps(
        self,
        chunks: list[str],
        action_matches: dict[str, list[str]],
        pattern_matches: list[str],
    ) -> list[PlanStep]:
        """Build discrete plan steps from matched actions and patterns.

        Groups consecutive matched words into logical steps. Each step
        gets candidate actions, an optional pattern match, and a
        complexity estimate.
        """
        if not action_matches:
            # No recognized actions — return a single generic step
            if chunks:
                return [
                    PlanStep(
                        description=" ".join(chunks),
                        candidate_actions=[],
                        pattern_match=None,
                        estimated_complexity="simple",
                    )
                ]
            return []

        steps: list[PlanStep] = []
        seen_words: set[str] = set()

        # Walk through chunks in order, grouping consecutive matches
        current_group_words: list[str] = []
        current_group_actions: list[str] = []

        def _flush_group() -> None:
            """Flush the current group into a PlanStep."""
            if not current_group_words:
                return

            # Deduplicate actions while preserving order
            deduped: list[str] = []
            seen: set[str] = set()
            for a in current_group_actions:
                if a not in seen:
                    deduped.append(a)
                    seen.add(a)

            # Check for pattern match
            matched_pattern: str | None = None
            group_set = set(current_group_words)
            for pm in pattern_matches:
                # Simple heuristic: if any group word overlaps with idiom keywords
                _idiom_keyword_sets: dict[str, set[str]] = {
                    "api_fetch_chain": {"api", "fetch", "download", "url", "json"},
                    "text_split_pipeline": {"split", "text", "replace"},
                    "device_settings_batch": {
                        "bluetooth",
                        "wifi",
                        "volume",
                        "brightness",
                    },
                    "health_data": {"health", "caffeine", "workout", "nutrient"},
                    "clipboard_roundtrip": {"clipboard", "copy", "paste"},
                }
                pm_keywords = _idiom_keyword_sets.get(pm, set())
                if group_set & pm_keywords:
                    matched_pattern = pm
                    break

            # Complexity based on action count
            action_count = len(deduped)
            if action_count <= 1:
                complexity = "simple"
            elif action_count <= 3:
                complexity = "medium"
            else:
                complexity = "complex"

            description = " ".join(current_group_words)
            steps.append(
                PlanStep(
                    description=description,
                    candidate_actions=deduped,
                    pattern_match=matched_pattern,
                    estimated_complexity=complexity,
                )
            )

        for word in chunks:
            if word in action_matches and word not in seen_words:
                current_group_words.append(word)
                current_group_actions.extend(action_matches[word])
                seen_words.add(word)
            else:
                # Non-matched word breaks the group
                if current_group_words:
                    # Check if this is a connective word (and, then, to, etc.)
                    if word in {
                        "and",
                        "then",
                        "to",
                        "the",
                        "a",
                        "an",
                        "with",
                        "from",
                        "for",
                        "it",
                        "on",
                        "in",
                    }:
                        continue  # Skip connectives, keep group open
                    _flush_group()
                    current_group_words = []
                    current_group_actions = []

        # Flush remaining group
        _flush_group()

        return (
            steps
            if steps
            else [
                PlanStep(
                    description=" ".join(chunks),
                    candidate_actions=[],
                    pattern_match=None,
                    estimated_complexity="simple",
                )
            ]
        )

    # ── Domain Suggestion ─────────────────────────────────────────

    def _suggest_domain(
        self,
        chunks: list[str],
        action_matches: dict[str, list[str]],
        archetype: str,
    ) -> str:
        """Suggest the most appropriate domain profile.

        Counts domain votes from matched actions. The domain with the
        most votes wins. Falls back to 'general'.
        """
        domain_votes: dict[str, int] = {}

        for word, actions in action_matches.items():
            for action_id in actions:
                domain = _ACTION_TO_DOMAIN.get(action_id)
                if domain:
                    domain_votes[domain] = domain_votes.get(domain, 0) + 1

        if not domain_votes:
            return "general"

        best_domain = max(domain_votes, key=lambda d: domain_votes[d])
        return best_domain

    # ── Budget Suggestion ─────────────────────────────────────────

    def _suggest_budget(self, steps: list[PlanStep], archetype: str) -> str:
        """Suggest a complexity tier for token budgeting.

        Based on total step count and individual step complexity:
          - 1-2 steps, all simple  -> "simple"
          - 3-5 steps              -> "medium"
          - 5-8 steps or has complex steps -> "complex"
          - 8+ steps or hybrid     -> "very_complex"
        """
        step_count = len(steps)
        has_complex = any(s.estimated_complexity == "complex" for s in steps)
        has_medium = any(s.estimated_complexity == "medium" for s in steps)

        if step_count >= 8 or archetype == "hybrid":
            return "very_complex"
        if step_count >= 5 or has_complex:
            return "complex"
        if step_count >= 3 or has_medium:
            return "medium"
        return "simple"

    # ── Creative Mode Suggestion ──────────────────────────────────

    def _suggest_creative_mode(self, archetype: str, steps: list[PlanStep]) -> str:
        """Suggest a creative mode based on archetype.

        Maps archetype to creative mode via _ARCHETYPE_TO_CREATIVE.
        Falls back to 'pragmatic' for unknown archetypes.
        """
        return _ARCHETYPE_TO_CREATIVE.get(archetype, "pragmatic")

    # ── Confidence Computation ────────────────────────────────────

    def _compute_confidence(
        self,
        action_matches: dict[str, list[str]],
        pattern_matches: list[str],
    ) -> float:
        """Compute confidence score (0-1) for the plan.

        Higher confidence when more actions are matched and patterns
        are recognized. The score is:
          base = min(matched_action_words / 3.0, 0.6)
          pattern_bonus = min(len(pattern_matches) * 0.2, 0.4)
          total = min(base + pattern_bonus, 1.0)
        """
        action_count = len(action_matches)
        if action_count == 0:
            return 0.0

        base = min(action_count / 3.0, 0.6)
        pattern_bonus = min(len(pattern_matches) * 0.2, 0.4)
        return round(min(base + pattern_bonus, 1.0), 2)


# ── Plan Context Formatting ───────────────────────────────────────────


def format_plan_context(plan: ExecutionPlan) -> str:
    """Format an execution plan as structured text for prompt injection.

    Produces a human-readable block that can be prepended to the
    generation prompt to provide the model with structured context
    about the planned shortcut.

    Args:
        plan: The execution plan to format.

    Returns:
        A multi-line string with the plan details.
    """
    lines: list[str] = []
    lines.append("## Execution Plan")
    lines.append(f"Archetype: {plan.archetype}")
    lines.append(f"Confidence: {plan.confidence}")

    if plan.steps:
        lines.append("Steps:")
        for i, step in enumerate(plan.steps, 1):
            actions_str = (
                ", ".join(step.candidate_actions)
                if step.candidate_actions
                else "(none)"
            )
            pattern_str = (
                f" [pattern: {step.pattern_match}]" if step.pattern_match else ""
            )
            lines.append(
                f"  {i}. {step.description} "
                f"-> likely actions: {actions_str}{pattern_str}"
            )
    else:
        lines.append("Steps: (none identified)")

    lines.append(f"Suggested domain: {plan.suggested_domain}")
    lines.append(f"Suggested budget: {plan.suggested_budget}")
    lines.append(f"Suggested creative mode: {plan.suggested_creative_mode}")

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/execution_planner.py "<prompt>"')
        print(
            'Example: python scripts/execution_planner.py "Download weather data and show a notification"'
        )
        sys.exit(1)

    prompt = " ".join(sys.argv[1:])
    planner = ExecutionPlanner()
    plan = planner.plan(prompt)

    print(format_plan_context(plan))
    print()
    print(f"Raw prompt: {plan.raw_prompt!r}")
    print(f"Step count: {len(plan.steps)}")
    for i, step in enumerate(plan.steps, 1):
        print(f"  Step {i}: {step.description}")
        print(f"    Actions: {step.candidate_actions}")
        print(f"    Pattern: {step.pattern_match}")
        print(f"    Complexity: {step.estimated_complexity}")
