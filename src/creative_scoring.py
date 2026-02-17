"""
ShortcutForge Creative Scorer.

Heuristic scoring of generated shortcuts across 5 dimensions:
  1. action_diversity: unique actions / total actions
  2. ui_richness: usage of menus, alerts, prompts, showresult
  3. error_handling: IF blocks after data-producing actions
  4. variable_reuse: variables referenced more than once
  5. workflow_complexity: nesting depth, control flow diversity

Mode-specific weights allow tuning for different generation styles:
  - pragmatic (default): balanced, favors correctness
  - expressive: rewards UI richness and diversity
  - playful: rewards creativity and complexity
  - automation_dense: rewards variable reuse and flow control
  - power_user: rewards all dimensions equally

Scoring is purely heuristic (no LLM-as-judge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dsl_ir import (
    ShortcutIR,
    ActionStatement,
    SetVariable,
    IfBlock,
    MenuBlock,
    RepeatBlock,
    ForeachBlock,
    Comment,
    VarRef,
    HandleRef,
    InterpolatedString,
    Statement,
)

__version__ = "1.0"


# ── Mode Weights ─────────────────────────────────────────────────

CREATIVE_MODES: dict[str, dict[str, float]] = {
    "pragmatic": {
        "action_diversity": 0.15,
        "ui_richness": 0.15,
        "error_handling": 0.35,
        "variable_reuse": 0.15,
        "workflow_complexity": 0.20,
    },
    "expressive": {
        "action_diversity": 0.25,
        "ui_richness": 0.30,
        "error_handling": 0.15,
        "variable_reuse": 0.10,
        "workflow_complexity": 0.20,
    },
    "playful": {
        "action_diversity": 0.25,
        "ui_richness": 0.25,
        "error_handling": 0.10,
        "variable_reuse": 0.15,
        "workflow_complexity": 0.25,
    },
    "automation_dense": {
        "action_diversity": 0.10,
        "ui_richness": 0.10,
        "error_handling": 0.25,
        "variable_reuse": 0.30,
        "workflow_complexity": 0.25,
    },
    "power_user": {
        "action_diversity": 0.20,
        "ui_richness": 0.20,
        "error_handling": 0.20,
        "variable_reuse": 0.20,
        "workflow_complexity": 0.20,
    },
}


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score for a single creativity dimension."""
    name: str
    raw_score: float  # 0.0 to 1.0
    weight: float
    weighted_score: float

    def __repr__(self) -> str:
        return f"{self.name}: {self.raw_score:.2f} (w={self.weight:.2f}, ws={self.weighted_score:.2f})"


@dataclass
class CreativityScore:
    """Full creativity score with per-dimension breakdown."""
    total: float  # Weighted sum (0.0 to 1.0)
    mode: str
    dimensions: list[DimensionScore] = field(default_factory=list)
    action_count: int = 0

    def __repr__(self) -> str:
        dim_strs = ", ".join(f"{d.name}={d.raw_score:.2f}" for d in self.dimensions)
        return f"CreativityScore({self.total:.2f}, mode={self.mode}, {dim_strs})"


# ── UI Actions ───────────────────────────────────────────────────

_UI_ACTIONS = frozenset({
    "choosefrommenu", "ask", "alert", "showresult", "notification",
    "speaktext", "vibrate",
    # Also short names that map to UI
    "menu", "prompt",
})

_DATA_PRODUCING_ACTIONS = frozenset({
    "downloadurl", "getcontentsofurl",
    "documentpicker.open", "selectphoto",
    "getclipboard", "ask",
})


# ── Scorer ───────────────────────────────────────────────────────

class CreativityScorer:
    """Scores a ShortcutIR across 5 creativity dimensions."""

    def score(self, ir: ShortcutIR, mode: str = "pragmatic") -> CreativityScore:
        """Score a ShortcutIR using the specified creative mode.

        Args:
            ir: The ShortcutIR to score.
            mode: Creative mode for weight distribution.

        Returns:
            CreativityScore with per-dimension breakdown and weighted total.
        """
        if mode not in CREATIVE_MODES:
            mode = "pragmatic"

        weights = CREATIVE_MODES[mode]

        # Collect all actions and statements
        all_actions = self._collect_actions(ir.statements)
        all_statements = self._collect_all_statements(ir.statements)

        # Score each dimension
        d1 = self._score_action_diversity(all_actions)
        d2 = self._score_ui_richness(all_actions, ir.statements)
        d3 = self._score_error_handling(ir.statements)
        d4 = self._score_variable_reuse(ir.statements)
        d5 = self._score_workflow_complexity(ir.statements)

        raw_scores = {
            "action_diversity": d1,
            "ui_richness": d2,
            "error_handling": d3,
            "variable_reuse": d4,
            "workflow_complexity": d5,
        }

        dimensions = []
        total = 0.0
        for dim_name, raw in raw_scores.items():
            w = weights[dim_name]
            ws = raw * w
            total += ws
            dimensions.append(DimensionScore(
                name=dim_name,
                raw_score=raw,
                weight=w,
                weighted_score=ws,
            ))

        return CreativityScore(
            total=min(total, 1.0),
            mode=mode,
            dimensions=dimensions,
            action_count=len(all_actions),
        )

    # ── Dimension 1: Action Diversity ────────────────────────────

    def _score_action_diversity(self, actions: list[ActionStatement]) -> float:
        """Ratio of unique action names to total actions.

        High diversity = many different actions used.
        Score: 0.0 (all same) to 1.0 (all unique).
        """
        if not actions:
            return 0.0

        unique = len(set(a.action_name.lower() for a in actions))
        total = len(actions)

        if total <= 1:
            return 1.0

        # Raw ratio with diminishing returns after ~50% diversity
        ratio = unique / total
        # Bonus for absolute variety: 10+ unique actions is impressive
        variety_bonus = min(unique / 10, 0.3)

        return min(ratio + variety_bonus, 1.0)

    # ── Dimension 2: UI Richness ─────────────────────────────────

    def _score_ui_richness(
        self,
        actions: list[ActionStatement],
        stmts: list[Statement],
    ) -> float:
        """Score based on user-facing UI elements.

        Checks for: menus, alerts, prompts, showresult, notifications.
        Also checks for MENU blocks in control flow.
        """
        if not actions:
            return 0.0

        ui_count = 0
        for a in actions:
            name = a.action_name.lower()
            if name in _UI_ACTIONS:
                ui_count += 1
            # Also count common UI patterns
            elif any(ui in name for ui in ("show", "alert", "ask", "notify", "speak")):
                ui_count += 1

        # Check for MENU blocks (structural UI)
        menu_count = self._count_block_type(stmts, MenuBlock)
        ui_count += menu_count * 2  # Menus are worth more

        # Score: 0 UI = 0.0, 1-2 UI = 0.3-0.5, 3+ = 0.7-1.0
        if ui_count == 0:
            return 0.0
        elif ui_count <= 2:
            return 0.3 + (ui_count - 1) * 0.2
        else:
            return min(0.6 + (ui_count - 2) * 0.1, 1.0)

    # ── Dimension 3: Error Handling ──────────────────────────────

    def _score_error_handling(self, stmts: list[Statement]) -> float:
        """Score based on IF blocks following data-producing actions.

        Checks for error handling patterns after downloads, file access, etc.
        """
        data_action_count = 0
        handled_count = 0

        self._count_error_handling(stmts, data_action_count=0, handled_count=0,
                                   result={"data": 0, "handled": 0})

        result: dict[str, int] = {"data": 0, "handled": 0}
        self._count_error_handling_flat(stmts, result)

        data_action_count = result["data"]
        handled_count = result["handled"]

        if data_action_count == 0:
            # No data-producing actions — give partial credit
            # (shortcut may not need error handling)
            has_any_if = self._count_block_type(stmts, IfBlock) > 0
            return 0.5 if has_any_if else 0.3

        return min(handled_count / max(data_action_count, 1), 1.0)

    def _count_error_handling_flat(
        self,
        stmts: list[Statement],
        result: dict[str, int],
    ) -> None:
        """Count data-producing actions and how many have error handling."""
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, ActionStatement):
                name = stmt.action_name.lower()
                if name in _DATA_PRODUCING_ACTIONS:
                    result["data"] += 1
                    # Check if followed by IF within 3 statements
                    for j in range(i + 1, min(i + 4, len(stmts))):
                        if isinstance(stmts[j], IfBlock):
                            result["handled"] += 1
                            break

            # Recurse
            if isinstance(stmt, IfBlock):
                self._count_error_handling_flat(stmt.then_body, result)
                if stmt.else_body:
                    self._count_error_handling_flat(stmt.else_body, result)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    self._count_error_handling_flat(case.body, result)
            elif isinstance(stmt, RepeatBlock):
                self._count_error_handling_flat(stmt.body, result)
            elif isinstance(stmt, ForeachBlock):
                self._count_error_handling_flat(stmt.body, result)

    def _count_error_handling(self, stmts: list, **kwargs) -> None:
        """Stub — replaced by _count_error_handling_flat."""
        pass

    # ── Dimension 4: Variable Reuse ──────────────────────────────

    def _score_variable_reuse(self, stmts: list[Statement]) -> float:
        """Score based on variables that are referenced more than once.

        Measures how well the shortcut uses intermediate variables
        rather than passing everything through @prev.
        """
        # Count SET definitions
        defined_vars: set[str] = set()
        # Count variable references
        var_ref_counts: dict[str, int] = {}

        self._collect_var_usage(stmts, defined_vars, var_ref_counts)

        if not defined_vars:
            return 0.0

        # Count vars referenced more than once
        reused = sum(1 for v, c in var_ref_counts.items() if c > 1 and v in defined_vars)

        if len(defined_vars) == 0:
            return 0.0

        reuse_ratio = reused / len(defined_vars)
        # Bonus for absolute count
        count_bonus = min(len(defined_vars) / 8, 0.2)

        return min(reuse_ratio + count_bonus, 1.0)

    def _collect_var_usage(
        self,
        stmts: list[Statement],
        defined: set[str],
        ref_counts: dict[str, int],
    ) -> None:
        """Recursively collect variable definitions and reference counts."""
        for stmt in stmts:
            if isinstance(stmt, SetVariable):
                defined.add(stmt.var_name)
                self._count_var_refs_in_value(stmt.value, ref_counts)
            elif isinstance(stmt, ActionStatement):
                for pv in stmt.params.values():
                    self._count_var_refs_in_value(pv, ref_counts)
            elif isinstance(stmt, IfBlock):
                if isinstance(stmt.target, VarRef):
                    ref_counts[stmt.target.name] = ref_counts.get(stmt.target.name, 0) + 1
                self._collect_var_usage(stmt.then_body, defined, ref_counts)
                if stmt.else_body:
                    self._collect_var_usage(stmt.else_body, defined, ref_counts)
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    self._collect_var_usage(case.body, defined, ref_counts)
            elif isinstance(stmt, RepeatBlock):
                self._collect_var_usage(stmt.body, defined, ref_counts)
            elif isinstance(stmt, ForeachBlock):
                self._collect_var_usage(stmt.body, defined, ref_counts)

    def _count_var_refs_in_value(self, value, ref_counts: dict[str, int]) -> None:
        """Count VarRef references in a value."""
        if isinstance(value, VarRef):
            ref_counts[value.name] = ref_counts.get(value.name, 0) + 1
        elif isinstance(value, InterpolatedString):
            for part in value.parts:
                if isinstance(part, VarRef):
                    ref_counts[part.name] = ref_counts.get(part.name, 0) + 1

    # ── Dimension 5: Workflow Complexity ─────────────────────────

    def _score_workflow_complexity(self, stmts: list[Statement]) -> float:
        """Score based on nesting depth and control flow diversity.

        Higher scores for deeper nesting and more diverse control flow.
        """
        # Max nesting depth
        max_depth = self._max_nesting_depth(stmts, 0)

        # Control flow diversity: how many types of blocks are used
        block_types: set[str] = set()
        self._collect_block_types(stmts, block_types)

        # Score from depth: 0=0.0, 1=0.3, 2=0.5, 3+=0.7
        if max_depth == 0:
            depth_score = 0.0
        elif max_depth == 1:
            depth_score = 0.3
        elif max_depth == 2:
            depth_score = 0.5
        else:
            depth_score = min(0.5 + (max_depth - 2) * 0.1, 0.8)

        # Score from diversity: 0=0.0, 1=0.1, 2=0.2, 3+=0.3
        diversity_score = min(len(block_types) * 0.1, 0.3)

        return min(depth_score + diversity_score, 1.0)

    def _max_nesting_depth(self, stmts: list[Statement], current: int) -> int:
        """Find maximum nesting depth in statements."""
        max_d = current
        for stmt in stmts:
            if isinstance(stmt, IfBlock):
                max_d = max(max_d, self._max_nesting_depth(stmt.then_body, current + 1))
                if stmt.else_body:
                    max_d = max(max_d, self._max_nesting_depth(stmt.else_body, current + 1))
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    max_d = max(max_d, self._max_nesting_depth(case.body, current + 1))
            elif isinstance(stmt, RepeatBlock):
                max_d = max(max_d, self._max_nesting_depth(stmt.body, current + 1))
            elif isinstance(stmt, ForeachBlock):
                max_d = max(max_d, self._max_nesting_depth(stmt.body, current + 1))
        return max_d

    def _collect_block_types(self, stmts: list[Statement], types: set[str]) -> None:
        """Collect all block types used in statements."""
        for stmt in stmts:
            if isinstance(stmt, IfBlock):
                types.add("if")
                self._collect_block_types(stmt.then_body, types)
                if stmt.else_body:
                    self._collect_block_types(stmt.else_body, types)
            elif isinstance(stmt, MenuBlock):
                types.add("menu")
                for case in stmt.cases:
                    self._collect_block_types(case.body, types)
            elif isinstance(stmt, RepeatBlock):
                types.add("repeat")
                self._collect_block_types(stmt.body, types)
            elif isinstance(stmt, ForeachBlock):
                types.add("foreach")
                self._collect_block_types(stmt.body, types)

    # ── Utilities ────────────────────────────────────────────────

    def _collect_actions(self, stmts: list[Statement]) -> list[ActionStatement]:
        """Recursively collect all ActionStatements."""
        actions: list[ActionStatement] = []
        for stmt in stmts:
            if isinstance(stmt, ActionStatement):
                actions.append(stmt)
            elif isinstance(stmt, IfBlock):
                actions.extend(self._collect_actions(stmt.then_body))
                if stmt.else_body:
                    actions.extend(self._collect_actions(stmt.else_body))
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    actions.extend(self._collect_actions(case.body))
            elif isinstance(stmt, RepeatBlock):
                actions.extend(self._collect_actions(stmt.body))
            elif isinstance(stmt, ForeachBlock):
                actions.extend(self._collect_actions(stmt.body))
        return actions

    def _collect_all_statements(self, stmts: list[Statement]) -> list[Statement]:
        """Recursively collect all statements (flat)."""
        result: list[Statement] = []
        for stmt in stmts:
            result.append(stmt)
            if isinstance(stmt, IfBlock):
                result.extend(self._collect_all_statements(stmt.then_body))
                if stmt.else_body:
                    result.extend(self._collect_all_statements(stmt.else_body))
            elif isinstance(stmt, MenuBlock):
                for case in stmt.cases:
                    result.extend(self._collect_all_statements(case.body))
            elif isinstance(stmt, RepeatBlock):
                result.extend(self._collect_all_statements(stmt.body))
            elif isinstance(stmt, ForeachBlock):
                result.extend(self._collect_all_statements(stmt.body))
        return result

    @staticmethod
    def _count_block_type(stmts: list[Statement], block_type: type) -> int:
        """Count blocks of a specific type (non-recursive)."""
        return sum(1 for s in stmts if isinstance(s, block_type))


# ── Convenience ──────────────────────────────────────────────────

def score_shortcut(ir: ShortcutIR, mode: str = "pragmatic") -> CreativityScore:
    """Convenience function to score a ShortcutIR."""
    return CreativityScorer().score(ir, mode)
