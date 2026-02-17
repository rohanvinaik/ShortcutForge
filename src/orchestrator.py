"""
ShortcutForge Orchestrator: Natural language → working Apple Shortcut.

Single source of truth for the generation pipeline.
Both CLI and web server call this module.

Pipeline:
  prompt → Generator Backend (Claude or Local) → DSL text
        → parse → validate → compile → deliver
  with automatic retry on parse/validation errors (max 3 attempts).

Generator backends:
  ClaudeBackend  — Anthropic Claude API (default)
  LocalBackend   — Fine-tuned MLX model + Outlines grammar constraint
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Protocol

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from generate_prompt import build_system_prompt, build_user_message, build_retry_message, build_snippet_context, build_plan_context
from dsl_parser import parse_dsl
from dsl_linter import lint_dsl
from dsl_validator import validate_ir
from dsl_bridge import compile_ir
from token_budget import estimate_budget, detect_overflow, next_budget
from domain_profile import DomainProfileManager, DomainProfile
from architecture_reasoner import ArchitectureReasoner, ArchitectureDecision
from scenario_profiles import ScenarioProfileManager, ScenarioProfile
from simulation_harness import SimulationHarness, SimulationReport, Severity, FindingCategory

# Optional: ExecutionPlanner (degrades gracefully if not available)
try:
    from execution_planner import ExecutionPlanner
    _HAS_EXECUTION_PLANNER = True
except ImportError:
    _HAS_EXECUTION_PLANNER = False

# ── Failure Classification ─────────────────────────────────────────────


class FailureType(Enum):
    """Classification of generation pipeline failures for retry routing."""
    OVERFLOW = auto()       # Output near budget, missing ENDSHORTCUT
    SYNTAX = auto()         # Parse error (bad grammar/structure)
    UNKNOWN_ACTION = auto() # Validation: unknown action
    BAD_PARAMS = auto()     # Validation: wrong params / other validation error
    COMPILE = auto()        # Compilation error
    NONE = auto()           # No failure


def classify_failure(
    parse_error: str | None,
    validation_errors: list[str] | None,
    compile_error: str | None,
    overflow_detected: bool,
) -> FailureType:
    """Classify a pipeline failure for retry routing.

    Args:
        parse_error: Parse error message, or None if parsing succeeded.
        validation_errors: List of validation error messages, or None/empty.
        compile_error: Compile error message, or None if compilation succeeded.
        overflow_detected: Whether overflow was detected pre-parse.

    Returns:
        FailureType for retry routing decisions.
    """
    if overflow_detected:
        return FailureType.OVERFLOW
    if parse_error:
        return FailureType.SYNTAX
    if validation_errors:
        for err in validation_errors:
            if "unknown_action" in err.lower() or "Unknown action" in err:
                return FailureType.UNKNOWN_ACTION
        return FailureType.BAD_PARAMS
    if compile_error:
        return FailureType.COMPILE
    return FailureType.NONE


# ── Data Structures ────────────────────────────────────────────────────


@dataclass
class StageResult:
    """Status of a single pipeline stage."""

    stage: str  # "generating", "parsing", "validating", "compiling", "delivering"
    status: str  # "pending", "running", "success", "failed", "skipped"
    message: str = ""
    duration_ms: int | None = None


@dataclass
class GenerationResult:
    """Result of the full generation pipeline."""

    success: bool = False
    dsl_text: str | None = None
    shortcut_path: str | None = None
    signed_path: str | None = None
    imported: bool = False
    name: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    stages: list[StageResult] = field(default_factory=list)
    timed_out: bool = False
    timeout_retries: int = 0
    snippets_injected: int = 0
    # Phase 3 additions
    scenario_profile: str = "default"
    architecture_decision: str = "shortcut_only"
    blueprint_doc: str | None = None
    creativity_score: float | None = None
    candidates_generated: int = 0
    candidates_valid: int = 0
    # Phase D: Contract validation findings
    contract_findings: list[str] = field(default_factory=list)
    # Phase E: Execution planner
    execution_plan: dict | None = None


# ── DSL Extraction ─────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```\w*\n?(.*?)```", re.DOTALL)
_SHORTCUT_RE = re.compile(r"(SHORTCUT\s+\".*)", re.DOTALL)


def _extract_dsl(text: str) -> str:
    """Extract DSL text from LLM response.

    Handles:
      - Raw DSL (starts with SHORTCUT)
      - Markdown-fenced DSL (```...```)
      - Preamble text before SHORTCUT
    """
    text = text.strip()

    # If it starts with SHORTCUT, it's raw DSL
    if text.startswith("SHORTCUT"):
        return text

    # Try extracting from markdown fences
    fences = _FENCE_RE.findall(text)
    for fenced in fences:
        fenced = fenced.strip()
        if fenced.startswith("SHORTCUT"):
            return fenced

    # Try finding SHORTCUT anywhere in the text
    match = _SHORTCUT_RE.search(text)
    if match:
        return match.group(1).strip()

    # Last resort: return as-is (will fail at parse)
    return text


def _ensure_trailing_newline(dsl: str) -> str:
    """Ensure DSL text ends with a newline (grammar requires it)."""
    if not dsl.endswith("\n"):
        return dsl + "\n"
    return dsl


# ── Generator Backends ─────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_RETRIES = 3


class GeneratorBackend(Protocol):
    """Protocol for pluggable generation backends."""

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 4096,
    ) -> str: ...

    @property
    def engine_name(self) -> str: ...


class ClaudeBackend:
    """Generation backend using Anthropic Claude API."""

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key to ClaudeBackend()."
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=key)

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
    ) -> str:
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    @property
    def engine_name(self) -> str:
        return "claude"


class LocalBackend:
    """Generation backend using local fine-tuned MLX model.

    Supports dual-mode operation:
      - Default: use_grammar=False (fast unconstrained generation with timeout)
      - On retry for syntax failures: use_grammar=True (grammar constraint)

    Grammar-constrained generator is lazy-loaded only when first needed.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        use_grammar: bool = False,
        never_grammar: bool = False,
        timeout_s: float = 90,
        chat_template: str = "llama3",
    ):
        self._model_path = model_path
        self._adapter_path = adapter_path
        self._never_grammar = never_grammar
        self._timeout_s = timeout_s
        self._chat_template = chat_template

        # Primary generator (normally unconstrained in production entry points)
        from inference import LocalDSLGenerator
        self._generator = LocalDSLGenerator(
            model_path=model_path,
            adapter_path=adapter_path,
            use_grammar=use_grammar,
            timeout_s=timeout_s,
            chat_template=chat_template,
        )
        self._grammar_generator = None  # Lazy-loaded

    def _get_grammar_generator(self):
        """Lazy-load grammar-constrained generator."""
        if self._grammar_generator is None:
            from inference import LocalDSLGenerator
            self._grammar_generator = LocalDSLGenerator(
                model_path=self._model_path,
                adapter_path=self._adapter_path,
                use_grammar=True,
                timeout_s=self._timeout_s,
                chat_template=self._chat_template,
            )
        return self._grammar_generator

    def generate(
        self,
        system_prompt: str,
        messages: list[dict],
        model: str = "",
        max_tokens: int = 4096,
        use_grammar: bool = False,
        timeout_s: float | None = None,
    ) -> str:
        effective_timeout = timeout_s if timeout_s is not None else self._timeout_s
        if use_grammar and not self._never_grammar:
            gen = self._get_grammar_generator()
        else:
            gen = self._generator
        return gen.generate(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            timeout_s=effective_timeout,
        )

    def generate_with_meta(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
        timeout_s: float | None = None,
    ):
        """Generate and return metadata (timeout status, timing).

        Returns (text, GenerationMeta) tuple.
        """
        from inference import GenerationMeta
        effective_timeout = timeout_s if timeout_s is not None else self._timeout_s
        result = self._generator.generate_with_meta(
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            timeout_s=effective_timeout,
        )
        if isinstance(result, tuple):
            return result
        # Fallback for grammar mode
        return result, GenerationMeta()

    @property
    def engine_name(self) -> str:
        return "local"


# ── Orchestrator ───────────────────────────────────────────────────────


class Orchestrator:
    """Orchestrates LLM generation → parse → validate → compile → deliver.

    Uses a pluggable GeneratorBackend for the LLM step. Defaults to Claude.

    Optional distillation_logger callback receives a dict after each lint stage
    for capturing provenance-enriched generation data for future model training.
    """

    def __init__(
        self,
        backend: GeneratorBackend | None = None,
        api_key: str | None = None,
        distillation_logger: Callable[[dict], None] | None = None,
    ):
        if backend is not None:
            self._backend = backend
        else:
            self._backend = ClaudeBackend(api_key=api_key)

        self._system_prompt = build_system_prompt()
        self._distillation_logger = distillation_logger
        self._domain_manager = DomainProfileManager()
        self._architecture_reasoner = ArchitectureReasoner()
        self._scenario_manager = ScenarioProfileManager()
        self._simulation_harness = SimulationHarness()
        self._snippet_registry_path = _SCRIPT_DIR.parent / "references" / "snippet_registry.json"

        # Phase E: Execution planner (optional, degrades gracefully)
        self._execution_planner = None
        if _HAS_EXECUTION_PLANNER:
            try:
                self._execution_planner = ExecutionPlanner()
            except Exception:
                pass  # Non-critical, continue without planner

    def generate(
        self,
        prompt: str,
        *,
        model: str = DEFAULT_MODEL,
        max_retries: int = MAX_RETRIES,
        output_dir: str = "./output",
        sign: bool = True,
        auto_import: bool = False,
        on_stage_update: Callable[[StageResult], None] | None = None,
        candidate_count: int = 1,
        creative_mode: str = "pragmatic",
        implementation_strategy: str = "auto",
    ) -> GenerationResult:
        """Generate a Shortcut from a natural language prompt.

        Args:
            prompt: Natural language description of the shortcut.
            model: Anthropic model name.
            max_retries: Max retry attempts on parse/validation errors.
            output_dir: Where to save the .shortcut file.
            sign: Whether to sign the shortcut.
            auto_import: Whether to import into Shortcuts.app.
            on_stage_update: Callback for real-time stage status.
            candidate_count: Number of candidates to generate (Phase 3).
            creative_mode: Creative scoring mode (Phase 3).
            implementation_strategy: "auto", "shortcut_only", or "shortcut_plus_blueprint".

        Returns:
            GenerationResult with paths, DSL text, errors, and stage info.
        """
        result = GenerationResult()

        def _stage(stage: str, status: str, message: str = "", duration_ms: int | None = None):
            sr = StageResult(stage=stage, status=status, message=message, duration_ms=duration_ms)
            result.stages.append(sr)
            if on_stage_update:
                on_stage_update(sr)

        # ── Phase E: Execution planning ──
        # Run the planner before generation to inform domain, budget,
        # and creative mode defaults. Plan suggestions have the lowest
        # priority: explicit user params > scenario overrides > plan > defaults
        plan = None
        plan_context = ""
        if self._execution_planner is not None:
            try:
                plan = self._execution_planner.plan(prompt)
                plan_context = build_plan_context(plan)
                result.execution_plan = {
                    "archetype": plan.archetype,
                    "steps": len(plan.steps),
                    "suggested_domain": plan.suggested_domain,
                }
            except Exception:
                plan = None  # Non-critical, continue without plan

        # ── Phase 3: Architecture reasoning ──
        arch_decision = self._architecture_reasoner.analyze(prompt)

        # Override strategy if explicitly specified
        if implementation_strategy == "shortcut_only":
            arch_decision = type(arch_decision)(
                strategy="shortcut_only",
                reason="Forced by implementation_strategy parameter",
            )
        elif implementation_strategy == "shortcut_plus_blueprint":
            arch_decision = type(arch_decision)(
                strategy="shortcut_plus_blueprint",
                reason="Forced by implementation_strategy parameter",
                blueprint_scope=arch_decision.blueprint_scope,
                hybrid_signals=arch_decision.hybrid_signals,
            )

        # Set result field AFTER any override so it reflects the actual strategy used
        result.architecture_decision = arch_decision.strategy

        # Generate blueprint doc for hybrid architectures
        if arch_decision.is_hybrid:
            bp = self._architecture_reasoner.generate_blueprint(arch_decision, prompt)
            if bp:
                result.blueprint_doc = f"{bp.title}\n\nComponents:\n" + \
                    "\n".join(f"  - {c}" for c in bp.components) + \
                    f"\n\n{bp.integration_notes}"

        # ── Phase 3: Scenario profile selection ──
        scenario = self._scenario_manager.select_scenario(prompt)
        result.scenario_profile = scenario.scenario_id

        # Apply plan suggestions (lowest priority, only if user didn't
        # explicitly set params and before scenario overrides)
        if plan is not None and creative_mode == "pragmatic":
            # Plan suggestion for creative mode (overridden by scenario below)
            if plan.suggested_creative_mode and plan.suggested_creative_mode != "pragmatic":
                creative_mode = plan.suggested_creative_mode

        # Apply scenario overrides (higher priority than plan)
        if scenario.creative_mode and creative_mode == "pragmatic":
            creative_mode = scenario.creative_mode

        # Select domain profile for prompt-aware context injection
        # Priority: scenario domain > plan suggested domain > auto-detect
        if scenario.domain_profile != "general":
            domain_profile = self._domain_manager.get_profile(scenario.domain_profile) or \
                self._domain_manager.select_profile(prompt)
        elif plan is not None and plan.suggested_domain != "general":
            domain_profile = self._domain_manager.get_profile(plan.suggested_domain) or \
                self._domain_manager.select_profile(prompt)
        else:
            domain_profile = self._domain_manager.select_profile(prompt)
        domain_context = domain_profile.prompt_context if domain_profile.has_context else ""
        domain_actions = domain_profile.format_relevant_actions()

        # Add scenario-specific system prompt addendum
        if scenario.system_prompt_addendum:
            domain_context = (domain_context + "\n\n" + scenario.system_prompt_addendum).strip()

        # Append execution plan context to domain_context
        if plan_context:
            domain_context = (domain_context + "\n\n" + plan_context).strip()

        # Build snippet context for retrieval-augmented generation
        snippet_ctx = build_snippet_context(prompt, registry_path=self._snippet_registry_path)
        if snippet_ctx:
            # Count how many snippets were injected (each starts with "### Pattern")
            result.snippets_injected = snippet_ctx.count("### Pattern")

        # Build initial messages with domain context and snippets
        user_message = build_user_message(
            prompt,
            domain_context=domain_context,
            domain_actions=domain_actions,
            include_snippets=bool(snippet_ctx),
            snippet_context=snippet_ctx,
        )
        messages = [{"role": "user", "content": user_message}]

        # Dynamic token budgeting
        # Priority: explicit user params > scenario override > plan suggestion > defaults
        budget = estimate_budget(prompt)
        _BUDGET_MAP = {"simple": 512, "medium": 1024, "complex": 2048, "very_complex": 4096}

        # Apply plan budget suggestion (lowest priority, only increases budget)
        if plan is not None and plan.suggested_budget:
            plan_tokens = _BUDGET_MAP.get(plan.suggested_budget)
            if plan_tokens is not None and plan_tokens > budget.max_tokens:
                budget = type(budget)(
                    max_tokens=plan_tokens,
                    complexity=plan.suggested_budget,
                    word_count=budget.word_count,
                    complex_signal_count=budget.complex_signal_count,
                    simple_signal_count=budget.simple_signal_count,
                    prompt_char_len=budget.prompt_char_len,
                )

        # Apply scenario budget_override (higher priority than plan)
        if scenario.budget_override:
            override_tokens = _BUDGET_MAP.get(scenario.budget_override)
            if override_tokens is not None and override_tokens > budget.max_tokens:
                budget = type(budget)(
                    max_tokens=override_tokens,
                    complexity=scenario.budget_override,
                    word_count=budget.word_count,
                    complex_signal_count=budget.complex_signal_count,
                    simple_signal_count=budget.simple_signal_count,
                    prompt_char_len=budget.prompt_char_len,
                )
        effective_max_tokens = budget.max_tokens

        dsl_text = None
        last_errors: list[str] = []
        use_grammar_next = False  # Grammar constraint flag for retry routing

        for attempt in range(1, max_retries + 1):
            result.attempts = attempt

            # ── Stage: Generating ──
            retry_note = f" (retry {attempt}/{max_retries})" if attempt > 1 else ""
            engine = self._backend.engine_name
            grammar_note = " +grammar" if use_grammar_next else ""
            budget_note = f", {budget.complexity} budget={effective_max_tokens}"
            _stage("generating", "running", f"Calling {engine}{retry_note}{budget_note}{grammar_note}...")

            t0 = time.monotonic()
            gen_timed_out = False
            try:
                # Use generate_with_meta for LocalBackend to get timeout info
                if isinstance(self._backend, LocalBackend) and hasattr(self._backend, 'generate_with_meta'):
                    if use_grammar_next:
                        # Grammar-constrained retry: use generate() with use_grammar=True
                        raw_text = self._backend.generate(
                            system_prompt=self._system_prompt,
                            messages=messages,
                            max_tokens=effective_max_tokens,
                            use_grammar=True,
                        )
                        gen_timed_out = False  # Grammar mode doesn't support timeout
                    else:
                        raw_text, gen_meta = self._backend.generate_with_meta(
                            system_prompt=self._system_prompt,
                            messages=messages,
                            max_tokens=effective_max_tokens,
                        )
                        gen_timed_out = gen_meta.timed_out
                    if gen_timed_out:
                        result.timed_out = True
                        result.timeout_retries += 1
                else:
                    raw_text = self._backend.generate(
                        system_prompt=self._system_prompt,
                        messages=messages,
                        model=model,
                        max_tokens=effective_max_tokens,
                    )
                # Reset grammar flag after use
                use_grammar_next = False
            except Exception as e:
                _stage("generating", "failed", f"Generation error: {e}")
                result.errors = [f"Generation error: {e}"]
                return result

            elapsed = int((time.monotonic() - t0) * 1000)
            dsl_text = _ensure_trailing_newline(_extract_dsl(raw_text))
            timeout_note = " [TIMED OUT]" if gen_timed_out else ""
            _stage("generating", "success",
                   f"Generated ({elapsed}ms, budget={effective_max_tokens}){timeout_note}", elapsed)

            # ── Pre-check: Timeout-triggered escalation ──
            if gen_timed_out:
                escalated = next_budget(effective_max_tokens)
                if escalated is not None:
                    old_budget = effective_max_tokens
                    effective_max_tokens = escalated
                    _stage("generating", "running",
                           f"Timeout detected, escalating budget {old_budget}→{effective_max_tokens}...")
                    # Don't regenerate here — let the overflow check below
                    # or the retry loop handle it. The timed-out output may
                    # still be usable if ENDSHORTCUT was reached.

            # ── Pre-check: Overflow detection with progressive escalation ──
            has_endshortcut = "ENDSHORTCUT" in dsl_text
            if detect_overflow(dsl_text, budget, has_endshortcut):
                escalated = next_budget(effective_max_tokens)
                if escalated is not None:
                    old_budget = effective_max_tokens
                    effective_max_tokens = escalated
                    _stage("generating", "running",
                           f"Overflow detected ({old_budget} tokens), escalating to {effective_max_tokens}...")

                    t0 = time.monotonic()
                    try:
                        raw_text = self._backend.generate(
                            system_prompt=self._system_prompt,
                            messages=messages,
                            model=model,
                            max_tokens=effective_max_tokens,
                        )
                    except Exception as e:
                        _stage("generating", "failed", f"Generation error (budget escalation): {e}")
                        result.errors = [f"Generation error: {e}"]
                        return result

                    elapsed = int((time.monotonic() - t0) * 1000)
                    dsl_text = _ensure_trailing_newline(_extract_dsl(raw_text))
                    _stage("generating", "success",
                           f"Budget escalation generated ({elapsed}ms, budget={effective_max_tokens})", elapsed)

            # ── Stage: Linting ──
            raw_dsl_text = dsl_text  # pre-lint for distillation
            lint_result = lint_dsl(dsl_text)
            if lint_result.was_modified:
                lint_summary = ", ".join(
                    f"{c.original!r}→{c.replacement!r}" for c in lint_result.changes[:5]
                )
                if len(lint_result.changes) > 5:
                    lint_summary += f", +{len(lint_result.changes) - 5} more"
                _stage("linting", "success", f"Repaired {len(lint_result.changes)} issue(s): {lint_summary}")
                dsl_text = lint_result.text
            else:
                _stage("linting", "success", "No repairs needed")

            result.dsl_text = dsl_text

            # ── Distillation logging (after lint, before parse) ──
            if self._distillation_logger:
                self._distillation_logger({
                    "prompt": prompt,
                    "raw_output": raw_dsl_text,
                    "canonicalized_output": dsl_text,
                    "lint_changes": [
                        {"kind": c.kind, "original": c.original,
                         "replacement": c.replacement, "confidence": c.confidence,
                         "reason": c.reason}
                        for c in lint_result.changes
                    ],
                    "was_modified": lint_result.was_modified,
                    "attempt": attempt,
                    "engine": engine,
                    "token_budget": effective_max_tokens,
                    "budget_complexity": budget.complexity,
                    "domain_profile": domain_profile.profile_id,
                    "scenario_profile": scenario.scenario_id,
                    "architecture_decision": arch_decision.strategy,
                })

            # ── Stage: Parsing ──
            parse_error = None
            ir = None
            _stage("parsing", "running", "Parsing DSL...")
            try:
                ir = parse_dsl(dsl_text)
                result.name = ir.name
                _stage("parsing", "success", f"Parsed: \"{ir.name}\" ({ir.action_count()} actions)")
            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."
                parse_error = error_msg
                _stage("parsing", "failed", f"Parse error: {error_msg}")
                last_errors = [f"Parse error: {error_msg}"]

            # ── Stage: Validating ──
            validation_errors: list[str] = []
            if ir is not None:
                _stage("validating", "running", "Validating against action catalog...")
                validation = validate_ir(ir, domain_profile=domain_profile.profile_id)
                result.warnings = [str(w) for w in validation.warnings]

                if validation.errors:
                    validation_errors = [
                        f"Line {e.line_number}: [{e.category}] {e.message}"
                        for e in validation.errors
                    ]
                    _stage("validating", "failed", f"{len(validation.errors)} error(s)")
                    last_errors = validation_errors
                else:
                    warn_note = f" ({len(validation.warnings)} warning(s))" if validation.warnings else ""
                    _stage("validating", "success", f"Valid{warn_note}")

            # ── Stage: Simulation (static analysis) ──
            if ir is not None and not validation_errors:
                try:
                    sim_report = self._simulation_harness.analyze(ir)
                    if sim_report.findings:
                        sim_warnings = [
                            f"[{f.category.value}] {f.message}"
                            + (f" (line {f.line_number})" if f.line_number else "")
                            for f in sim_report.findings
                            if f.severity in (Severity.WARNING, Severity.ERROR)
                        ]
                        result.warnings.extend(sim_warnings)
                        # Extract contract-category findings
                        result.contract_findings = [
                            f"[{f.category.value}] {f.message}"
                            + (f" (line {f.line_number})" if f.line_number else "")
                            for f in sim_report.findings
                            if f.category == FindingCategory.CONTRACT
                        ]
                        if sim_warnings:
                            _stage("simulation", "success",
                                   f"{len(sim_warnings)} finding(s)")
                        else:
                            _stage("simulation", "success", "Clean")
                    else:
                        _stage("simulation", "success", "Clean")
                except Exception:
                    _stage("simulation", "success", "Skipped")

            # ── Stage: Compiling ──
            compile_error = None
            shortcut = None
            if ir is not None and not validation_errors:
                _stage("compiling", "running", "Compiling to .shortcut...")
                try:
                    shortcut = compile_ir(ir)
                    _stage("compiling", "success", f"Compiled ({len(shortcut.actions)} actions)")
                except Exception as e:
                    compile_error = str(e)
                    _stage("compiling", "failed", f"Compile error: {e}")
                    result.errors = [f"Compile error: {e}"]

            # ── Failure-type-routed retry ──
            ftype = classify_failure(
                parse_error=parse_error,
                validation_errors=validation_errors if validation_errors else None,
                compile_error=compile_error,
                overflow_detected=False,  # overflow handled in pre-check above
            )

            if ftype == FailureType.NONE:
                # Success — score with CreativityScorer (Phase 3)
                best_ir = ir
                best_shortcut = shortcut
                best_dsl = dsl_text
                best_score: float | None = None

                if ir is not None:
                    try:
                        from creative_scoring import CreativityScorer
                        scorer = CreativityScorer()
                        cs = scorer.score(ir, mode=creative_mode)
                        best_score = cs.total
                        result.creativity_score = cs.total
                    except Exception:
                        pass  # Non-critical, don't fail pipeline

                result.candidates_generated = 1
                result.candidates_valid = 1

                # Multi-candidate generation: generate additional candidates and
                # pick the highest-scoring valid one
                if candidate_count > 1 and ir is not None:
                    _stage("candidates", "running",
                           f"Generating {candidate_count - 1} additional candidate(s)...")

                    for c_idx in range(2, candidate_count + 1):
                        try:
                            c_raw = self._backend.generate(
                                system_prompt=self._system_prompt,
                                messages=messages,
                                model=model,
                                max_tokens=effective_max_tokens,
                            )
                            c_dsl = _ensure_trailing_newline(_extract_dsl(c_raw))
                            c_lint = lint_dsl(c_dsl)
                            c_dsl = c_lint.text

                            c_ir = parse_dsl(c_dsl)
                            c_val = validate_ir(c_ir, domain_profile=domain_profile.profile_id)
                            if c_val.errors:
                                result.candidates_generated += 1
                                continue  # Invalid candidate, skip

                            c_shortcut = compile_ir(c_ir)
                            result.candidates_generated += 1
                            result.candidates_valid += 1

                            # Score this candidate
                            try:
                                c_cs = scorer.score(c_ir, mode=creative_mode)
                                c_score = c_cs.total
                            except Exception:
                                c_score = 0.0

                            # Keep the highest-scoring candidate
                            if best_score is None or c_score > best_score:
                                best_ir = c_ir
                                best_shortcut = c_shortcut
                                best_dsl = c_dsl
                                best_score = c_score
                                result.creativity_score = c_score
                        except Exception:
                            result.candidates_generated += 1
                            continue  # Generation/parse/compile failure

                    _stage("candidates", "success",
                           f"{result.candidates_valid}/{result.candidates_generated} valid"
                           + (f", best={best_score:.2f}" if best_score is not None else ""))

                # Use the best candidate for delivery
                ir = best_ir
                shortcut = best_shortcut
                dsl_text = best_dsl
                result.dsl_text = dsl_text
                # Fall through to delivery
            elif ftype == FailureType.COMPILE:
                # No retry for compile errors — return immediately
                if not result.errors:
                    result.errors = [f"Compile error: {compile_error}"]
                return result
            elif attempt < max_retries:
                # Route retry strategy by failure type
                if ftype == FailureType.SYNTAX and isinstance(self._backend, LocalBackend):
                    # Syntax failures → grammar constraint can help
                    use_grammar_next = True
                    _stage("retrying", "running", f"Syntax failure → enabling grammar constraint")
                elif ftype == FailureType.UNKNOWN_ACTION:
                    # Unknown action → grammar won't help, use error context
                    _stage("retrying", "running", f"Unknown action → error context retry")
                elif ftype == FailureType.BAD_PARAMS:
                    _stage("retrying", "running", f"Bad params → error context retry")

                messages.append({"role": "assistant", "content": raw_text})
                messages.append({"role": "user", "content": build_retry_message(dsl_text, last_errors)})
                continue
            else:
                # Max retries exceeded
                if not result.errors:
                    result.errors = last_errors or [f"{ftype.name} failure after {max_retries} attempts"]
                return result

            # If we reach here, compilation succeeded — skip compile error check
            if compile_error:
                return result

            # ── Stage: Delivering ──
            _stage("delivering", "running", "Saving and signing...")
            try:
                os.makedirs(output_dir, exist_ok=True)
                delivery = shortcut.deliver(
                    output_dir=output_dir,
                    sign=sign,
                    auto_import=auto_import,
                )
                result.shortcut_path = delivery.get("unsigned")
                result.signed_path = delivery.get("signed")
                result.imported = delivery.get("imported", False)
                instructions = delivery.get("instructions", "")
                _stage("delivering", "success", instructions)
            except Exception as e:
                _stage("delivering", "failed", f"Delivery error: {e}")
                result.errors = [f"Delivery error: {e}"]
                return result

            # ── Done ──
            result.success = True
            return result

        # Should not reach here, but just in case
        result.errors = last_errors or ["Max retries exceeded"]
        return result

    def compile_dsl(
        self,
        dsl_text: str,
        *,
        output_dir: str = "./output",
        sign: bool = True,
        auto_import: bool = False,
        on_stage_update: Callable[[StageResult], None] | None = None,
    ) -> GenerationResult:
        """Compile DSL text directly (no LLM generation).

        Useful for debugging and iterating on DSL files.
        """
        result = GenerationResult()
        result.dsl_text = dsl_text
        result.attempts = 0  # No LLM call

        def _stage(stage: str, status: str, message: str = "", duration_ms: int | None = None):
            sr = StageResult(stage=stage, status=status, message=message, duration_ms=duration_ms)
            result.stages.append(sr)
            if on_stage_update:
                on_stage_update(sr)

        _stage("generating", "skipped", "Direct DSL input (no LLM)")

        # Parse
        _stage("parsing", "running", "Parsing DSL...")
        try:
            ir = parse_dsl(dsl_text)
            result.name = ir.name
            _stage("parsing", "success", f"Parsed: \"{ir.name}\" ({ir.action_count()} actions)")
        except Exception as e:
            _stage("parsing", "failed", f"Parse error: {e}")
            result.errors = [f"Parse error: {e}"]
            return result

        # Validate
        _stage("validating", "running", "Validating...")
        validation = validate_ir(ir)
        result.warnings = [str(w) for w in validation.warnings]
        if validation.errors:
            error_msgs = [f"Line {e.line_number}: [{e.category}] {e.message}" for e in validation.errors]
            _stage("validating", "failed", f"{len(validation.errors)} error(s)")
            result.errors = error_msgs
            return result
        _stage("validating", "success", "Valid")

        # Compile
        _stage("compiling", "running", "Compiling...")
        try:
            shortcut = compile_ir(ir)
            _stage("compiling", "success", f"Compiled ({len(shortcut.actions)} actions)")
        except Exception as e:
            _stage("compiling", "failed", f"Compile error: {e}")
            result.errors = [f"Compile error: {e}"]
            return result

        # Deliver
        _stage("delivering", "running", "Saving...")
        try:
            os.makedirs(output_dir, exist_ok=True)
            delivery = shortcut.deliver(
                output_dir=output_dir,
                sign=sign,
                auto_import=auto_import,
            )
            result.shortcut_path = delivery.get("unsigned")
            result.signed_path = delivery.get("signed")
            result.imported = delivery.get("imported", False)
            _stage("delivering", "success", delivery.get("instructions", ""))
        except Exception as e:
            _stage("delivering", "failed", f"Delivery error: {e}")
            result.errors = [f"Delivery error: {e}"]
            return result

        result.success = True
        return result
