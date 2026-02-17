#!/usr/bin/env python3
"""
Scenario-Parity Benchmark Evaluator for ShortcutForge.

Evaluates generated shortcuts against scenario packs with rubric-based scoring.
Each scenario pack defines:
  - A reference DSL (ground truth)
  - A scoring rubric with weighted dimensions
  - Multiple prompt variants at different difficulty levels

Usage:
    # Evaluate all prompt variants against the health_logger scenario:
    python scripts/evaluate_scenario.py \\
        --scenario references/scenario_packs/health_logger/ \\
        --model-path mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \\
        --adapter-path models/baseline-v1-mlx

    # Evaluate a single prompt variant:
    python scripts/evaluate_scenario.py \\
        --scenario references/scenario_packs/health_logger/ \\
        --variant 0

    # Score an existing DSL file against the rubric (no generation):
    python scripts/evaluate_scenario.py \\
        --scenario references/scenario_packs/health_logger/ \\
        --score-dsl path/to/generated.dsl

    # Score the reference DSL (sanity check):
    python scripts/evaluate_scenario.py \\
        --scenario references/scenario_packs/health_logger/ \\
        --score-reference
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# ── Data Classes ───────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score for a single rubric dimension."""
    name: str
    weight: float
    score: float  # 0.0–1.0
    criteria_scores: dict[str, float] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)

    @property
    def weighted_score(self) -> float:
        return self.weight * self.score


@dataclass
class ScenarioScore:
    """Complete scoring result for one generation."""
    scenario_id: str
    prompt_variant_id: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    total_score: float = 0.0
    parsed: bool = False
    validated: bool = False
    compiled: bool = False
    action_count: int = 0
    errors: list[str] = field(default_factory=list)

    def compute_total(self):
        """Compute weighted total from dimensions."""
        self.total_score = sum(d.weighted_score for d in self.dimensions)


@dataclass
class ScenarioResult:
    """Complete evaluation result for a scenario (all prompt variants)."""
    scenario_id: str
    scenario_name: str
    variant_scores: list[ScenarioScore] = field(default_factory=list)
    reference_score: Optional[ScenarioScore] = None

    @property
    def average_score(self) -> float:
        if not self.variant_scores:
            return 0.0
        return sum(s.total_score for s in self.variant_scores) / len(self.variant_scores)


# ── Scenario Pack Loading ──────────────────────────────────────────

def load_scenario_pack(scenario_dir: str | Path) -> dict:
    """Load a scenario pack from a directory.

    Expected structure:
        scenario_dir/
            rubric.json
            reference.dsl
            nutrient_map.json (optional)
    """
    scenario_dir = Path(scenario_dir)
    if not scenario_dir.is_dir():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    rubric_path = scenario_dir / "rubric.json"
    reference_path = scenario_dir / "reference.dsl"

    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric not found: {rubric_path}")
    if not reference_path.exists():
        raise FileNotFoundError(f"Reference DSL not found: {reference_path}")

    with open(rubric_path) as f:
        rubric = json.load(f)

    reference_dsl = reference_path.read_text()

    # Optional files
    nutrient_map = None
    nutrient_map_path = scenario_dir / "nutrient_map.json"
    if nutrient_map_path.exists():
        with open(nutrient_map_path) as f:
            nutrient_map = json.load(f)

    return {
        "rubric": rubric,
        "reference_dsl": reference_dsl,
        "nutrient_map": nutrient_map,
        "scenario_dir": scenario_dir,
    }


# ── IR Analysis Utilities ─────────────────────────────────────────

def _collect_all_actions(ir) -> list:
    """Recursively collect all ActionStatement nodes from IR."""
    from dsl_ir import (
        ActionStatement, IfBlock, MenuBlock, RepeatBlock, ForeachBlock,
    )

    actions = []

    def _walk(statements):
        for stmt in statements:
            if isinstance(stmt, ActionStatement):
                actions.append(stmt)
            elif isinstance(stmt, IfBlock):
                _walk(stmt.then_body)
                if stmt.else_body:
                    _walk(stmt.else_body)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    _walk(case.body)
            elif isinstance(stmt, RepeatBlock):
                _walk(stmt.body)
            elif isinstance(stmt, ForeachBlock):
                _walk(stmt.body)

    _walk(ir.statements)
    return actions


def _collect_all_constructs(ir) -> dict[str, int]:
    """Count control flow constructs in IR."""
    from dsl_ir import (
        IfBlock, MenuBlock, RepeatBlock, ForeachBlock, SetVariable,
    )

    counts: dict[str, int] = {
        "IF": 0,
        "MENU": 0,
        "REPEAT": 0,
        "FOREACH": 0,
        "SET": 0,
    }

    def _walk(statements):
        for stmt in statements:
            if isinstance(stmt, SetVariable):
                counts["SET"] += 1
            elif isinstance(stmt, IfBlock):
                counts["IF"] += 1
                _walk(stmt.then_body)
                if stmt.else_body:
                    _walk(stmt.else_body)
            elif isinstance(stmt, MenuBlock):
                counts["MENU"] += 1
                for case in stmt.cases:
                    _walk(case.body)
            elif isinstance(stmt, RepeatBlock):
                counts["REPEAT"] += 1
                _walk(stmt.body)
            elif isinstance(stmt, ForeachBlock):
                counts["FOREACH"] += 1
                _walk(stmt.body)

    _walk(ir.statements)
    return counts


def _has_action_after_action(ir, before_action: str, after_action_or_construct: str) -> bool:
    """Check if an IF/action appears within a few statements after a specific action."""
    from dsl_ir import ActionStatement, IfBlock

    def _walk(statements):
        for i, stmt in enumerate(statements):
            if isinstance(stmt, ActionStatement) and stmt.action_name == before_action:
                # Check next 3 statements for the target
                for j in range(i + 1, min(i + 4, len(statements))):
                    next_stmt = statements[j]
                    if after_action_or_construct == "IF" and isinstance(next_stmt, IfBlock):
                        return True
                    if isinstance(next_stmt, ActionStatement) and next_stmt.action_name == after_action_or_construct:
                        return True
            # Recurse into blocks
            if isinstance(stmt, IfBlock):
                if _walk(stmt.then_body):
                    return True
                if stmt.else_body and _walk(stmt.else_body):
                    return True
        return False

    return _walk(ir.statements)


# ── Criterion Registry ─────────────────────────────────────────────

_CRITERION_REGISTRY: dict[str, Callable] = {}


def criterion(name: str):
    """Decorator that registers a scoring criterion function.

    The decorated function must accept
    (ir, actions, action_set, constructs, parsed, validated, compiled)
    and return bool.
    """
    def decorator(fn: Callable) -> Callable:
        _CRITERION_REGISTRY[name] = fn
        return fn
    return decorator


# ── Structural criteria ───────────────────────────────────────────

@criterion("parses")
def _crit_parses(ir, actions, action_set, constructs, parsed, validated, compiled):
    return parsed


@criterion("validates")
def _crit_validates(ir, actions, action_set, constructs, parsed, validated, compiled):
    return validated


@criterion("compiles")
def _crit_compiles(ir, actions, action_set, constructs, parsed, validated, compiled):
    return compiled


@criterion("has_foreach")
def _crit_has_foreach(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("FOREACH", 0) > 0


@criterion("has_conditional")
def _crit_has_conditional(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("IF", 0) > 0


@criterion("has_variables")
def _crit_has_variables(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("SET", 0) > 0


@criterion("has_sufficient_actions")
def _crit_has_sufficient_actions(ir, actions, action_set, constructs, parsed, validated, compiled):
    return len(actions) >= 8


@criterion("has_menu")
def _crit_has_menu(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("MENU", 0) > 0


@criterion("has_repeat")
def _crit_has_repeat(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("REPEAT", 0) > 0


# ── Action coverage criteria ─────────────────────────────────────

@criterion("uses_downloadurl")
def _crit_uses_downloadurl(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "downloadurl" in action_set


@criterion("uses_detect_dictionary")
def _crit_uses_detect_dictionary(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "detect.dictionary" in action_set


@criterion("uses_getvalueforkey")
def _crit_uses_getvalueforkey(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "getvalueforkey" in action_set


@criterion("uses_health_quantity_log")
def _crit_uses_health_quantity_log(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "health.quantity.log" in action_set


@criterion("health_log_count_gte_3")
def _crit_health_log_count_gte_3(ir, actions, action_set, constructs, parsed, validated, compiled):
    return sum(1 for a in actions if a.action_name == "health.quantity.log") >= 3


@criterion("health_log_count_gte_10")
def _crit_health_log_count_gte_10(ir, actions, action_set, constructs, parsed, validated, compiled):
    return sum(1 for a in actions if a.action_name == "health.quantity.log") >= 10


@criterion("uses_url_action")
def _crit_uses_url_action(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "url" in action_set or any(
        "URL" in str(a.params) for a in actions
        if a.action_name in ("downloadurl", "gettext")
    )


# ── Error handling criteria ──────────────────────────────────────

@criterion("checks_network_response")
def _crit_checks_network_response(ir, actions, action_set, constructs, parsed, validated, compiled):
    return _has_action_after_action(ir, "downloadurl", "IF")


@criterion("checks_empty_data")
def _crit_checks_empty_data(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("IF", 0) >= 2


@criterion("user_feedback_on_error")
def _crit_user_feedback_on_error(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "alert" in action_set or "notification" in action_set


@criterion("input_validation")
def _crit_input_validation(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("IF", 0) >= 1 and (
        "ask" in action_set or
        any("input" in str(a.params).lower() for a in actions[:5])
    )


# ── UX criteria ──────────────────────────────────────────────────

@criterion("shows_progress")
def _crit_shows_progress(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "notification" in action_set or (
        "showresult" in action_set and constructs.get("FOREACH", 0) > 0
    )


@criterion("shows_summary")
def _crit_shows_summary(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "showresult" in action_set or "alert" in action_set


@criterion("uses_notification_or_alert")
def _crit_uses_notification_or_alert(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "notification" in action_set or "alert" in action_set or "showresult" in action_set


@criterion("accepts_input")
def _crit_accepts_input(ir, actions, action_set, constructs, parsed, validated, compiled):
    # Check for @input references in the IR, or askforinput, or file.select, or ask action
    if "askforinput" in action_set or "ask" in action_set or "file.select" in action_set:
        return True
    # Check for @input handle references in IF blocks or params
    from dsl_ir import HandleRef, IfBlock
    def _check_input(stmts):
        for stmt in stmts:
            if isinstance(stmt, IfBlock):
                if isinstance(stmt.target, HandleRef) and stmt.target.kind == "input":
                    return True
                if _check_input(stmt.then_body):
                    return True
                if stmt.else_body and _check_input(stmt.else_body):
                    return True
        return False
    if _check_input(ir.statements):
        return True
    # Fallback: check for @input in text or action params
    for a in actions:
        for v in a.params.values():
            if "@input" in str(v).lower():
                return True
    return constructs.get("IF", 0) >= 1


@criterion("clean_naming")
def _crit_clean_naming(ir, actions, action_set, constructs, parsed, validated, compiled):
    return len(ir.name) > 3 and ir.name != "Untitled"


# ── New criteria: data flow & platform ───────────────────────────

@criterion("data_flow_completeness")
def _crit_data_flow_completeness(ir, actions, action_set, constructs, parsed, validated, compiled):
    """Check that variables that are SET are actually referenced later.
    Returns True if >60% of SET variables are referenced somewhere.
    """
    from dsl_ir import SetVariable, VarRef, InterpolatedString, IfBlock, MenuBlock, RepeatBlock, ForeachBlock

    # Collect all SET variable names
    set_vars = set()
    def _collect_sets(stmts):
        for s in stmts:
            if isinstance(s, SetVariable):
                set_vars.add(s.var_name)
            elif isinstance(s, IfBlock):
                _collect_sets(s.then_body)
                if s.else_body:
                    _collect_sets(s.else_body)
            elif isinstance(s, MenuBlock):
                for case in s.cases:
                    _collect_sets(case.body)
            elif isinstance(s, RepeatBlock):
                _collect_sets(s.body)
            elif isinstance(s, ForeachBlock):
                _collect_sets(s.body)
    _collect_sets(ir.statements)

    if not set_vars:
        return True  # No variables to check

    # Collect all referenced variable names
    referenced = set()
    def _collect_refs_from_value(val):
        if isinstance(val, VarRef):
            referenced.add(val.name)
        elif isinstance(val, InterpolatedString):
            for part in val.parts:
                if isinstance(part, VarRef):
                    referenced.add(part.name)

    def _collect_refs(stmts):
        from dsl_ir import ActionStatement
        for s in stmts:
            if isinstance(s, ActionStatement):
                for v in s.params.values():
                    _collect_refs_from_value(v)
            elif isinstance(s, SetVariable):
                _collect_refs_from_value(s.value)
            elif isinstance(s, IfBlock):
                if isinstance(s.target, VarRef):
                    referenced.add(s.target.name)
                if s.compare_value:
                    _collect_refs_from_value(s.compare_value)
                _collect_refs(s.then_body)
                if s.else_body:
                    _collect_refs(s.else_body)
            elif isinstance(s, MenuBlock):
                for case in s.cases:
                    _collect_refs(case.body)
            elif isinstance(s, RepeatBlock):
                _collect_refs(s.body)
            elif isinstance(s, ForeachBlock):
                _collect_refs(s.body)
    _collect_refs(ir.statements)

    used_count = len(set_vars & referenced)
    return (used_count / len(set_vars)) > 0.6


@criterion("platform_appropriateness")
def _crit_platform_appropriateness(ir, actions, action_set, constructs, parsed, validated, compiled):
    """Placeholder for future platform validation checks. Returns True always for now."""
    return True


# ── Generic construct criteria ───────────────────────────────────

@criterion("has_menu_or_if_chain")
def _crit_has_menu_or_if_chain(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("MENU", 0) > 0 or constructs.get("IF", 0) >= 2


@criterion("has_input_handling")
def _crit_has_input_handling(ir, actions, action_set, constructs, parsed, validated, compiled):
    if "askforinput" in action_set or "ask" in action_set:
        return True
    # Check IF in first 5 statements
    from dsl_ir import IfBlock
    for stmt in ir.statements[:5]:
        if isinstance(stmt, IfBlock):
            return True
    return False


# ── File router criteria ─────────────────────────────────────────

@criterion("fallback_selection")
def _crit_fallback_selection(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "file.select" in action_set


@criterion("shows_file_info")
def _crit_shows_file_info(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "properties.files" in action_set and (
        "showresult" in action_set or "alert" in action_set
    )


@criterion("shows_result_per_path")
def _crit_shows_result_per_path(ir, actions, action_set, constructs, parsed, validated, compiled):
    sr_count = sum(1 for a in actions if a.action_name == "showresult")
    alert_count = sum(1 for a in actions if a.action_name == "alert")
    return sr_count >= 2 or (sr_count + alert_count) >= 2


@criterion("detects_file_type")
def _crit_detects_file_type(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "properties.files" in action_set


@criterion("handles_images")
def _crit_handles_images(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "image.resize" in action_set


@criterion("handles_pdfs")
def _crit_handles_pdfs(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "previewdocument" in action_set


@criterion("handles_text")
def _crit_handles_text(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "gettext" in action_set


@criterion("handles_fallback")
def _crit_handles_fallback(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "openin" in action_set


# ── API pagination criteria ──────────────────────────────────────

@criterion("has_pagination_loop")
def _crit_has_pagination_loop(ir, actions, action_set, constructs, parsed, validated, compiled):
    has_loop = constructs.get("REPEAT", 0) > 0 or constructs.get("FOREACH", 0) > 0
    return has_loop and "downloadurl" in action_set


@criterion("accumulates_results")
def _crit_accumulates_results(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "appendvariable" in action_set


@criterion("uses_url_construction")
def _crit_uses_url_construction(ir, actions, action_set, constructs, parsed, validated, compiled):
    if "url" in action_set:
        return True
    # Check for gettext with URL content
    for a in actions:
        if a.action_name == "gettext":
            for v in a.params.values():
                if "url" in str(v).lower() or "http" in str(v).lower():
                    return True
    return False


@criterion("has_stop_condition")
def _crit_has_stop_condition(ir, actions, action_set, constructs, parsed, validated, compiled):
    """Check if there is an IF inside a REPEAT or FOREACH."""
    from dsl_ir import IfBlock, RepeatBlock, ForeachBlock

    def _has_if_in_loop(stmts):
        for stmt in stmts:
            if isinstance(stmt, (RepeatBlock, ForeachBlock)):
                for inner in stmt.body:
                    if isinstance(inner, IfBlock):
                        return True
                if _has_if_in_loop(stmt.body):
                    return True
            elif isinstance(stmt, IfBlock):
                if _has_if_in_loop(stmt.then_body):
                    return True
                if stmt.else_body and _has_if_in_loop(stmt.else_body):
                    return True
        return False

    return _has_if_in_loop(ir.statements)


# ── Calendar triage criteria ─────────────────────────────────────

@criterion("categorizes_events")
def _crit_categorizes_events(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("FOREACH", 0) > 0 and constructs.get("IF", 0) > 0


@criterion("shows_grouped_output")
def _crit_shows_grouped_output(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "showresult" in action_set and (
        constructs.get("FOREACH", 0) > 0 or constructs.get("MENU", 0) > 0
    )


# ── Clipboard utility criteria ──────────────────────────────────

@criterion("has_clipboard_operations")
def _crit_has_clipboard_operations(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "getclipboard" in action_set or "setclipboard" in action_set


@criterion("has_note_operations")
def _crit_has_note_operations(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "appendtonote" in action_set or "findnotes" in action_set


@criterion("has_menu_navigation")
def _crit_has_menu_navigation(ir, actions, action_set, constructs, parsed, validated, compiled):
    """MENU > 0 with >= 2 cases."""
    from dsl_ir import MenuBlock

    def _find_menus(stmts):
        for stmt in stmts:
            if isinstance(stmt, MenuBlock):
                if len(stmt.cases) >= 2:
                    return True
                for case in stmt.cases:
                    if _find_menus(case.body):
                        return True
        return False

    return _find_menus(ir.statements)


# ── Media metadata pipeline criteria ─────────────────────────────

@criterion("extracts_metadata")
def _crit_extracts_metadata(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "getimagedetail" in action_set or "properties.files" in action_set


@criterion("generates_report")
def _crit_generates_report(ir, actions, action_set, constructs, parsed, validated, compiled):
    has_text = "gettext" in action_set or "text" in action_set
    has_output = "showresult" in action_set or "sendemail" in action_set or "setclipboard" in action_set
    return has_text and has_output


@criterion("handles_multiple_items")
def _crit_handles_multiple_items(ir, actions, action_set, constructs, parsed, validated, compiled):
    return constructs.get("FOREACH", 0) > 0 or constructs.get("REPEAT", 0) > 0


# ── Morning routine criteria ────────────────────────────────────

@criterion("has_weather_check")
def _crit_has_weather_check(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "weather.currentconditions" in action_set


@criterion("has_calendar_preview")
def _crit_has_calendar_preview(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "getcalendarevents" in action_set or "filter.calendarevents" in action_set


@criterion("has_news_feed")
def _crit_has_news_feed(ir, actions, action_set, constructs, parsed, validated, compiled):
    if "getrssfeed" in action_set:
        return True
    # Check downloadurl with RSS-like content
    if "downloadurl" in action_set:
        for a in actions:
            if a.action_name == "downloadurl":
                if "rss" in str(a.params).lower():
                    return True
    return False


@criterion("has_commute_time")
def _crit_has_commute_time(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "gettraveltime" in action_set


@criterion("has_briefing_output")
def _crit_has_briefing_output(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "speaktext" in action_set or "showresult" in action_set


@criterion("is_time_aware")
def _crit_is_time_aware(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "date" in action_set or "format.date" in action_set


# ── Share sheet text cleaner criteria ────────────────────────────

@criterion("has_text_processing")
def _crit_has_text_processing(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "gettext" in action_set or "splittext" in action_set or "replacetext" in action_set


@criterion("handles_share_input")
def _crit_handles_share_input(ir, actions, action_set, constructs, parsed, validated, compiled):
    if "extensioninput" in action_set:
        return True
    # Check for @input references
    from dsl_ir import HandleRef, IfBlock
    def _check(stmts):
        for stmt in stmts:
            if isinstance(stmt, IfBlock):
                if isinstance(stmt.target, HandleRef) and stmt.target.kind == "input":
                    return True
        return False
    return _check(ir.statements)


@criterion("cleans_text")
def _crit_cleans_text(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "replacetext" in action_set


@criterion("copies_to_clipboard")
def _crit_copies_to_clipboard(ir, actions, action_set, constructs, parsed, validated, compiled):
    return "setclipboard" in action_set


# ── Scoring Engine ─────────────────────────────────────────────────

class ScenarioScorer:
    """Scores a ShortcutIR against a scenario rubric."""

    def __init__(self, rubric: dict):
        self.rubric = rubric
        self.scenario_id = rubric.get("scenario_id", "unknown")
        self.required_actions = rubric.get("required_actions", [])
        self.desired_actions = rubric.get("desired_actions", [])
        self.required_constructs = rubric.get("required_constructs", [])

    def score(
        self,
        ir,
        *,
        parsed: bool = True,
        validated: bool = False,
        compiled: bool = False,
        prompt_variant_id: str = "",
    ) -> ScenarioScore:
        """Score a parsed IR against the rubric."""
        result = ScenarioScore(
            scenario_id=self.scenario_id,
            prompt_variant_id=prompt_variant_id,
            parsed=parsed,
            validated=validated,
            compiled=compiled,
        )

        # Collect IR analysis
        actions = _collect_all_actions(ir)
        action_names = [a.action_name for a in actions]
        action_set = set(action_names)
        constructs = _collect_all_constructs(ir)
        result.action_count = len(actions)

        # Score each dimension
        scoring = self.rubric.get("scoring", {})

        for dim_name, dim_config in scoring.items():
            dim_score = self._score_dimension(
                dim_name, dim_config, ir, actions, action_set, constructs,
                parsed, validated, compiled,
            )
            result.dimensions.append(dim_score)

        result.compute_total()
        return result

    def _score_dimension(
        self,
        name: str,
        config: dict,
        ir,
        actions: list,
        action_set: set[str],
        constructs: dict[str, int],
        parsed: bool,
        validated: bool,
        compiled: bool,
    ) -> DimensionScore:
        """Score a single rubric dimension."""
        weight = config.get("weight", 0.25)
        criteria = config.get("criteria", {})
        criteria_scores: dict[str, float] = {}
        details: list[str] = []

        for crit_name, crit_config in criteria.items():
            points = crit_config.get("points", 0.0)
            met = self._check_criterion(
                crit_name, ir, actions, action_set, constructs,
                parsed, validated, compiled,
            )
            criteria_scores[crit_name] = points if met else 0.0
            status = "✓" if met else "✗"
            details.append(f"{status} {crit_name}: {crit_config.get('description', '')}")

        total_points = sum(criteria_scores.values())
        max_points = sum(c.get("points", 0.0) for c in criteria.values())

        return DimensionScore(
            name=name,
            weight=weight,
            score=total_points / max_points if max_points > 0 else 0.0,
            criteria_scores=criteria_scores,
            details=details,
        )

    def _check_criterion(
        self,
        criterion_name: str,
        ir,
        actions: list,
        action_set: set[str],
        constructs: dict[str, int],
        parsed: bool,
        validated: bool,
        compiled: bool,
    ) -> bool:
        """Check if a specific criterion is met.

        Lookup order:
        1. Check the criterion registry for an exact match.
        2. If the criterion starts with ``uses_``, check whether the
           action name (after stripping the prefix) is in *action_set*.
        3. Otherwise return False (unknown criterion).
        """
        # 1. Registry lookup
        fn = _CRITERION_REGISTRY.get(criterion_name)
        if fn is not None:
            return fn(ir, actions, action_set, constructs, parsed, validated, compiled)

        # 2. Generic uses_XYZ fallback
        if criterion_name.startswith("uses_"):
            action_name = criterion_name[len("uses_"):]
            # Support dotted action names encoded with underscores
            # e.g. uses_detect_dictionary -> detect.dictionary already registered,
            # but for truly generic ones like uses_count -> check "count"
            return action_name in action_set

        # 3. Unknown criterion
        return False


# ── Pipeline Integration ───────────────────────────────────────────

def score_dsl_text(
    dsl_text: str,
    rubric: dict,
    prompt_variant_id: str = "",
) -> ScenarioScore:
    """Score a DSL text against a rubric. Runs through lint→parse→validate→compile."""
    from dsl_linter import lint_dsl
    from dsl_parser import parse_dsl
    from dsl_validator import validate_ir
    from dsl_bridge import compile_ir

    scorer = ScenarioScorer(rubric)

    # Lint
    lint_result = lint_dsl(dsl_text)
    dsl_text = lint_result.text

    # Parse
    try:
        ir = parse_dsl(dsl_text)
        parsed = True
    except Exception as e:
        return ScenarioScore(
            scenario_id=rubric.get("scenario_id", "unknown"),
            prompt_variant_id=prompt_variant_id,
            parsed=False,
            errors=[f"Parse error: {e}"],
        )

    # Validate (permissive)
    validation = validate_ir(ir, strict=False)
    validated = not validation.errors

    # Compile
    compiled = False
    try:
        if validated:
            compile_ir(ir)
            compiled = True
    except Exception as e:
        pass  # Don't block scoring on compile errors

    return scorer.score(
        ir,
        parsed=parsed,
        validated=validated,
        compiled=compiled,
        prompt_variant_id=prompt_variant_id,
    )


def evaluate_scenario(
    scenario_dir: str | Path,
    backend=None,
    model: str = "",
    variant_idx: int | None = None,
    verbose: bool = False,
) -> ScenarioResult:
    """Evaluate a scenario pack by generating shortcuts and scoring them.

    Args:
        scenario_dir: Path to scenario pack directory
        backend: GeneratorBackend for LLM generation (None = score reference only)
        model: Model name (for Claude backend)
        variant_idx: Specific variant to evaluate (None = all)
        verbose: Print detailed output

    Returns:
        ScenarioResult with scores for each prompt variant
    """
    pack = load_scenario_pack(scenario_dir)
    rubric = pack["rubric"]
    prompt_variants = rubric.get("prompt_variants", [])

    result = ScenarioResult(
        scenario_id=rubric.get("scenario_id", "unknown"),
        scenario_name=rubric.get("scenario_name", "Unknown Scenario"),
    )

    # Score the reference DSL (sanity check)
    if verbose:
        print(f"\n  Scoring reference DSL...", end=" ", flush=True)
    ref_score = score_dsl_text(pack["reference_dsl"], rubric, prompt_variant_id="reference")
    result.reference_score = ref_score
    if verbose:
        print(f"done (score: {ref_score.total_score:.2f})")

    # If no backend, we're done (just reference scoring)
    if backend is None:
        return result

    # Generate and score each prompt variant
    from orchestrator import Orchestrator

    orch = Orchestrator(backend=backend)

    variants = prompt_variants if variant_idx is None else [prompt_variants[variant_idx]]

    for i, variant in enumerate(variants):
        variant_id = variant.get("id", str(i))
        prompt = variant["prompt"]
        difficulty = variant.get("difficulty", "unknown")

        if verbose:
            print(f"\n  Variant '{variant_id}' (difficulty: {difficulty})...", flush=True)
            print(f"    Prompt: {prompt[:80]}...", flush=True)

        t0 = time.monotonic()

        gen_result = orch.generate(
            prompt,
            model=model,
            max_retries=3,
            output_dir="/tmp/scenario_eval",
            sign=False,
            auto_import=False,
        )

        elapsed = time.monotonic() - t0

        if gen_result.dsl_text:
            vscore = score_dsl_text(gen_result.dsl_text, rubric, prompt_variant_id=variant_id)
        else:
            vscore = ScenarioScore(
                scenario_id=rubric.get("scenario_id", "unknown"),
                prompt_variant_id=variant_id,
                errors=gen_result.errors,
            )

        result.variant_scores.append(vscore)

        if verbose:
            print(f"    Score: {vscore.total_score:.2f} ({elapsed:.1f}s)")
            if vscore.errors:
                for e in vscore.errors:
                    print(f"    Error: {e}")
            for dim in vscore.dimensions:
                print(f"    {dim.name}: {dim.score:.2f} (weight: {dim.weight})")
                for detail in dim.details:
                    print(f"      {detail}")

    return result


# ── Report Generation ──────────────────────────────────────────────

def print_report(result: ScenarioResult):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print(f"Scenario Evaluation: {result.scenario_name}")
    print(f"{'='*60}")

    if result.reference_score:
        ref = result.reference_score
        print(f"\n  Reference DSL: {ref.total_score:.2f}")
        print(f"    Parsed: {'✓' if ref.parsed else '✗'}  "
              f"Validated: {'✓' if ref.validated else '✗'}  "
              f"Compiled: {'✓' if ref.compiled else '✗'}  "
              f"Actions: {ref.action_count}")
        for dim in ref.dimensions:
            print(f"    {dim.name}: {dim.score:.2f} (weight: {dim.weight}, "
                  f"weighted: {dim.weighted_score:.2f})")

    if result.variant_scores:
        print(f"\n  Generated Variants:")
        for vscore in result.variant_scores:
            print(f"\n  [{vscore.prompt_variant_id}] "
                  f"Score: {vscore.total_score:.2f}")
            print(f"    Parsed: {'✓' if vscore.parsed else '✗'}  "
                  f"Validated: {'✓' if vscore.validated else '✗'}  "
                  f"Compiled: {'✓' if vscore.compiled else '✗'}  "
                  f"Actions: {vscore.action_count}")
            for dim in vscore.dimensions:
                print(f"    {dim.name}: {dim.score:.2f} (weight: {dim.weight}, "
                      f"weighted: {dim.weighted_score:.2f})")
            if vscore.errors:
                for e in vscore.errors[:3]:
                    print(f"    Error: {e}")

        print(f"\n  Average Score: {result.average_score:.2f}")
    else:
        print(f"\n  No variants evaluated (no backend provided)")

    print(f"\n{'='*60}")


def export_result(result: ScenarioResult, output_path: str | Path):
    """Export evaluation result as JSON."""
    data = {
        "scenario_id": result.scenario_id,
        "scenario_name": result.scenario_name,
        "average_score": result.average_score,
        "reference_score": {
            "total": result.reference_score.total_score if result.reference_score else None,
            "parsed": result.reference_score.parsed if result.reference_score else None,
            "validated": result.reference_score.validated if result.reference_score else None,
            "compiled": result.reference_score.compiled if result.reference_score else None,
        } if result.reference_score else None,
        "variant_scores": [
            {
                "variant_id": vs.prompt_variant_id,
                "total": vs.total_score,
                "parsed": vs.parsed,
                "validated": vs.validated,
                "compiled": vs.compiled,
                "action_count": vs.action_count,
                "dimensions": {
                    d.name: {
                        "score": d.score,
                        "weight": d.weight,
                        "weighted": d.weighted_score,
                        "criteria": d.criteria_scores,
                    }
                    for d in vs.dimensions
                },
                "errors": vs.errors,
            }
            for vs in result.variant_scores
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_scenario",
        description="Evaluate ShortcutForge against scenario benchmarks",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Path to scenario pack directory",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Discover and score all scenario packs under references/scenario_packs/",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=None,
        help="Specific prompt variant index to evaluate (default: all)",
    )
    parser.add_argument(
        "--score-reference",
        action="store_true",
        help="Only score the reference DSL (no generation)",
    )
    parser.add_argument(
        "--score-dsl",
        type=str,
        default=None,
        help="Score an existing DSL file against the rubric (no generation)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local MLX model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["claude", "local"],
        default="local",
        help="Generation engine (default: local)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model name (when --engine=claude)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Export results to JSON file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    args = parser.parse_args()

    # ── --all-scenarios mode ──────────────────────────────────────
    if args.all_scenarios:
        # Discover scenario pack directories
        project_root = _SCRIPT_DIR.parent
        packs_root = project_root / "references" / "scenario_packs"
        if not packs_root.is_dir():
            print(f"Error: scenario packs directory not found: {packs_root}", file=sys.stderr)
            sys.exit(1)

        pack_dirs = sorted(
            d for d in packs_root.iterdir()
            if d.is_dir() and (d / "rubric.json").exists()
        )

        if not pack_dirs:
            print("No scenario packs found.", file=sys.stderr)
            sys.exit(1)

        print(f"\nShortcutForge: All-Scenarios Evaluation\n")
        print(f"  Found {len(pack_dirs)} scenario pack(s) under {packs_root}\n")

        rows: list[tuple[str, str, float, bool, bool, bool, int]] = []

        for pack_dir in pack_dirs:
            try:
                pack = load_scenario_pack(pack_dir)
                ref_score = score_dsl_text(
                    pack["reference_dsl"], pack["rubric"],
                    prompt_variant_id="reference",
                )
                scenario_name = pack["rubric"].get("scenario_name", pack_dir.name)
                rows.append((
                    pack_dir.name,
                    scenario_name,
                    ref_score.total_score,
                    ref_score.parsed,
                    ref_score.validated,
                    ref_score.compiled,
                    ref_score.action_count,
                ))
            except Exception as e:
                rows.append((pack_dir.name, f"ERROR: {e}", 0.0, False, False, False, 0))

        # Print summary table
        hdr = f"  {'Pack':<30s} {'Score':>6s}  {'P':>1s} {'V':>1s} {'C':>1s}  {'Acts':>4s}  Name"
        print(hdr)
        print(f"  {'-'*len(hdr.strip())}")
        for pack_id, name, score, p, v, c, acts in rows:
            p_s = "Y" if p else "-"
            v_s = "Y" if v else "-"
            c_s = "Y" if c else "-"
            print(f"  {pack_id:<30s} {score:>6.2f}  {p_s:>1s} {v_s:>1s} {c_s:>1s}  {acts:>4d}  {name}")

        avg = sum(r[2] for r in rows) / len(rows) if rows else 0.0
        print(f"\n  Average reference score: {avg:.2f}")
        print()
        return

    # ── Single-scenario mode ─────────────────────────────────────
    if not args.scenario:
        print("Error: --scenario is required (or use --all-scenarios)", file=sys.stderr)
        sys.exit(1)

    print(f"\nShortcutForge: Scenario Evaluation\n")
    print(f"  Scenario: {args.scenario}")

    # Score an existing DSL file
    if args.score_dsl:
        pack = load_scenario_pack(args.scenario)
        dsl_text = Path(args.score_dsl).read_text()
        score = score_dsl_text(dsl_text, pack["rubric"], prompt_variant_id="external")
        result = ScenarioResult(
            scenario_id=pack["rubric"].get("scenario_id", "unknown"),
            scenario_name=pack["rubric"].get("scenario_name", "Unknown"),
            variant_scores=[score],
        )
        print_report(result)
        if args.output:
            export_result(result, args.output)
        return

    # Score reference only
    if args.score_reference or (not args.model_path and args.engine == "local"):
        result = evaluate_scenario(
            args.scenario,
            backend=None,
            verbose=args.verbose,
        )
        print_report(result)
        if args.output:
            export_result(result, args.output)
        return

    # Full evaluation with generation
    from orchestrator import LocalBackend, ClaudeBackend

    backend = None
    if args.engine == "local":
        if not args.model_path:
            print("Error: --model-path required for --engine=local", file=sys.stderr)
            sys.exit(1)
        backend = LocalBackend(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
        )
    else:
        try:
            backend = ClaudeBackend()
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    result = evaluate_scenario(
        args.scenario,
        backend=backend,
        model=args.model,
        variant_idx=args.variant,
        verbose=args.verbose,
    )

    print_report(result)

    if args.output:
        export_result(result, args.output)
        print(f"\n  Results exported to: {args.output}")


if __name__ == "__main__":
    main()
