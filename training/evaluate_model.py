#!/usr/bin/env python3
"""
Evaluate a fine-tuned model on the held-out eval set.

Generates DSL for each eval example and measures:
  - Parse pass rate (syntactically valid DSL)
  - Validate pass rate (known actions/params) — strict and permissive
  - Compile pass rate (end-to-end success) — strict and permissive
  - Runtime unverified compile rate (permissive-only compiles)

Features:
  - Dynamic token budgeting (simple/medium/complex/very_complex)
  - Wall-clock timeout with streaming generation
  - Overflow-triggered budget escalation (single retry per example)
  - ENDSHORTCUT early stopping

Usage:
    # Default (dynamic budget, 90s timeout):
    python scripts/evaluate_model.py --adapter-path models/baseline-v1-mlx -v

    # Fixed budget (legacy behavior):
    python scripts/evaluate_model.py --adapter-path models/baseline-v1-mlx --fixed-budget 4096

    # Custom timeout:
    python scripts/evaluate_model.py --adapter-path models/baseline-v1-mlx --timeout 120

    # Standard 8B model (recommended):
    python scripts/evaluate_model.py --model-path mlx-community/Meta-Llama-3.1-8B-Instruct --adapter-path models/baseline-v1-mlx -v

    # Capture baseline snapshot:
    python scripts/evaluate_model.py --adapter-path models/baseline-v1-mlx --snapshot -v
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from dsl_parser import parse_dsl
from dsl_linter import lint_dsl
from dsl_validator import validate_ir
from dsl_bridge import compile_ir
from token_budget import estimate_budget, detect_overflow, next_budget, TokenBudget
from inference import generate_with_timeout, GenerationMeta, DEFAULT_TIMEOUT_S
import dsl_linter as _linter_module


# ============================================================
# DSL extraction (mirrors orchestrator._extract_dsl)
# ============================================================

_FENCE_RE = re.compile(r"```\w*\n?(.*?)```", re.DOTALL)
_SHORTCUT_RE = re.compile(r"(SHORTCUT\s+\".*)", re.DOTALL)


def _extract_dsl(text: str) -> str:
    """Extract DSL text from LLM response."""
    text = text.strip()
    if text.startswith("SHORTCUT"):
        return text
    fences = _FENCE_RE.findall(text)
    for fenced in fences:
        fenced = fenced.strip()
        if fenced.startswith("SHORTCUT"):
            return fenced
    match = _SHORTCUT_RE.search(text)
    if match:
        return match.group(1).strip()
    return text


# ============================================================
# Failure category classifier
# ============================================================

def _classify_failure(result: dict) -> str | None:
    """Classify a failed result into a failure category.

    Categories:
      - parse_timeout: generation timed out (wall-clock exceeded)
      - parse_overflow: gen_time > 120s (token budget exhausted, no timeout)
      - parse_syntax: parse failure not from overflow/timeout
      - validate_unknown_action: unknown_action error
      - validate_other: other validation error
      - compile_error: passes validation but fails compilation

    Returns None if the example passed all stages.
    """
    if result.get("compiled", False):
        return None

    error = result.get("error", "") or ""

    if not result.get("parsed", False):
        # Parse failure — check timeout first, then overflow
        if result.get("timed_out", False):
            return "parse_timeout"
        gen_time = result.get("gen_time", 0)
        if gen_time > 120:
            return "parse_overflow"
        return "parse_syntax"

    if not result.get("validated", False):
        if "unknown_action" in error.lower() or "Unknown action" in error:
            return "validate_unknown_action"
        return "validate_other"

    if not result.get("compiled", False):
        return "compile_error"

    return None


def _format_eval_prompt(system_msg: str, description: str, chat_template: str = "llama3") -> str:
    """Format evaluation prompt using the specified chat template.

    Args:
        system_msg: System message content.
        description: User prompt / shortcut description.
        chat_template: Template format - "llama3" or "chatml".

    Returns:
        Formatted prompt string ready for tokenization.
    """
    if chat_template == "chatml":
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{description}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    # Default: Llama 3 format
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n{system_msg}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n{description}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def evaluate(
    model_path: str,
    adapter_path: str | None,
    eval_file: str,
    max_examples: int | None = None,
    skip_examples: int = 0,
    max_tokens: int | None = None,
    fixed_budget: int | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    budget_retry: bool = True,
    verbose: bool = False,
    log_distillation: bool = False,
    distillation_output: str | None = None,
    append_distillation: bool = False,
    chat_template: str = "llama3",
) -> dict:
    """Evaluate model on eval set.

    Args:
        model_path: Base model path or HuggingFace repo.
        adapter_path: LoRA adapter directory (optional).
        eval_file: Path to eval JSONL file.
        max_examples: Max examples to evaluate (default: all).
        skip_examples: Number of examples to skip from start (for batched processing).
        max_tokens: Legacy max_tokens override (deprecated, use fixed_budget).
        fixed_budget: Fixed token budget for all examples (overrides dynamic).
        timeout_s: Wall-clock timeout per generation in seconds.
        budget_retry: Whether to retry on overflow with escalated budget.
        verbose: Show per-example results.
        log_distillation: Write distillation log.
        distillation_output: Custom output path for distillation log.
        append_distillation: Append to existing distillation log instead of truncating.
        chat_template: Chat template format ("llama3" or "chatml").

    Returns:
        Stats dict with metrics, results, and failure categories.
    """
    import mlx_lm

    # Load model
    print(f"  Loading model: {model_path}", flush=True)
    if adapter_path:
        print(f"  Adapter: {adapter_path}", flush=True)
    load_kwargs = {}
    if adapter_path:
        load_kwargs["adapter_path"] = adapter_path
    model, tokenizer = mlx_lm.load(model_path, **load_kwargs)
    print(f"  Model loaded.", flush=True)

    # Load eval examples — stream-skip to avoid loading unneeded data
    eval_examples = []
    with open(eval_file) as f:
        for line_idx, line in enumerate(f):
            if line_idx < skip_examples:
                continue
            if max_examples and len(eval_examples) >= max_examples:
                break
            eval_examples.append(json.loads(line))

    budget_mode = "fixed" if (fixed_budget or max_tokens) else "dynamic"
    effective_fixed = fixed_budget or max_tokens or None
    print(f"  Evaluating {len(eval_examples)} examples (budget={budget_mode}"
          f"{f', fixed={effective_fixed}' if effective_fixed else ''}"
          f", timeout={timeout_s}s)", flush=True)
    print(flush=True)

    # Stats
    total = len(eval_examples)
    parse_pass = 0
    validate_strict_pass = 0
    validate_permissive_pass = 0
    compile_strict_pass = 0
    compile_permissive_pass = 0
    generation_times = []
    timeout_count = 0
    budget_retries = 0
    fallback_count = 0  # Reserved for profile-based fallback chains
    results = []

    for i, example in enumerate(eval_examples):
        description = example["messages"][1]["content"]
        system_msg = example["messages"][0]["content"]
        shortcut_id = example.get("shortcut_id", f"example_{i}")

        # Format as chat template
        prompt = _format_eval_prompt(system_msg, description, chat_template)

        # Dynamic budget estimation
        if effective_fixed:
            budget = TokenBudget(
                max_tokens=effective_fixed,
                complexity="fixed",
                word_count=len(description.split()),
                complex_signal_count=0,
                simple_signal_count=0,
                prompt_char_len=len(description),
            )
            effective_max_tokens = effective_fixed
        else:
            budget = estimate_budget(description)
            effective_max_tokens = budget.max_tokens

        # Generate with timeout
        raw, gen_meta = generate_with_timeout(
            model, tokenizer, prompt=prompt,
            max_tokens=effective_max_tokens,
            timeout_s=timeout_s,
        )
        gen_time = gen_meta.gen_time_s
        generation_times.append(gen_time)

        was_timed_out = gen_meta.timed_out
        if was_timed_out:
            timeout_count += 1

        # Overflow detection + budget retry
        overflow_retried = False
        if budget_retry and not was_timed_out:
            dsl_check = _extract_dsl(raw)
            has_end = "ENDSHORTCUT" in dsl_check
            if detect_overflow(dsl_check, budget, has_end):
                escalated = next_budget(effective_max_tokens)
                if escalated is not None:
                    old_budget = effective_max_tokens
                    effective_max_tokens = escalated
                    budget_retries += 1
                    overflow_retried = True

                    raw, gen_meta = generate_with_timeout(
                        model, tokenizer, prompt=prompt,
                        max_tokens=effective_max_tokens,
                        timeout_s=timeout_s,
                    )
                    gen_time += gen_meta.gen_time_s
                    was_timed_out = gen_meta.timed_out
                    if was_timed_out:
                        timeout_count += 1

                    if verbose:
                        print(f"    ↳ Budget escalation: {old_budget}→{effective_max_tokens}", flush=True)

        # Extract DSL
        dsl_text = _extract_dsl(raw)
        if not dsl_text.endswith("\n"):
            dsl_text += "\n"

        # Save raw (pre-lint) text for distillation logging
        raw_dsl_text = dsl_text

        # Lint — repair hallucinated actions/conditions before parsing
        lint_result = lint_dsl(dsl_text)
        dsl_text = lint_result.text
        lint_repairs = len(lint_result.changes)

        lint_changes_list = [
            {"kind": c.kind, "original": c.original, "replacement": c.replacement,
             "confidence": c.confidence, "reason": c.reason}
            for c in lint_result.changes
        ] if lint_repairs > 0 else []

        result = {
            "shortcut_id": shortcut_id,
            "description": description[:80],
            "gen_time": round(gen_time, 1),
            "parsed": False,
            "validated": False,  # backward compat: strict result
            "validated_strict": False,
            "validated_permissive": False,
            "compiled": False,   # backward compat: strict result
            "compiled_strict": False,
            "compiled_permissive": False,
            "runtime_unverified": False,
            "compiler_risk_warnings": [],
            "error": None,
            "failure_category": None,
            "lint_repairs": lint_repairs,
            "lint_changes": lint_changes_list,
            # New Phase 9/10 fields
            "timed_out": was_timed_out,
            "token_budget": effective_max_tokens,
            "budget_complexity": budget.complexity,
            "overflow_retried": overflow_retried,
            "early_stopped": gen_meta.early_stopped,
            "tokens_generated": gen_meta.tokens_generated,
        }

        # Distillation v2: classify scenario/domain/architecture
        try:
            from scenario_profiles import select_scenario
            from domain_profile import select_domain_profile
            from architecture_reasoner import analyze_architecture

            scenario = select_scenario(description)
            domain = select_domain_profile(description)
            arch = analyze_architecture(description)
            result["scenario_profile"] = scenario.scenario_id
            result["domain_profile"] = domain.profile_id
            result["architecture_decision"] = arch.strategy
        except Exception:
            result["scenario_profile"] = "default"
            result["domain_profile"] = "general"
            result["architecture_decision"] = "shortcut_only"

        # For distillation logging, stash raw pre-lint and post-lint text
        if log_distillation:
            result["_raw_dsl_text"] = raw_dsl_text
            result["_canonicalized_dsl_text"] = dsl_text

        # Parse
        try:
            ir = parse_dsl(dsl_text)
            result["parsed"] = True
            parse_pass += 1
        except Exception as e:
            result["error"] = f"parse: {str(e)[:100]}"
            result["failure_category"] = _classify_failure(result)
            results.append(result)
            if verbose:
                status = "PARSE_FAIL"
                lint_note = f" [lint:{lint_repairs}]" if lint_repairs > 0 else ""
                cat = f" [{result['failure_category']}]" if result["failure_category"] else ""
                budget_note = f" [budget:{budget.complexity}={effective_max_tokens}]"
                timeout_note = " [TIMEOUT]" if was_timed_out else ""
                print(f"  [{i+1}/{total}] {status}{cat}{lint_note}{budget_note}{timeout_note} ({gen_time:.1f}s) {description[:55]}", flush=True)
            continue

        # Validate — strict mode
        try:
            validation_strict = validate_ir(ir, strict=True)
            if not validation_strict.errors:
                result["validated_strict"] = True
                result["validated"] = True  # backward compat
                validate_strict_pass += 1
            else:
                result["error"] = f"validate: {validation_strict.errors[0]}"
        except Exception as e:
            result["error"] = f"validate: {str(e)[:100]}"

        # Validate — permissive mode
        try:
            validation_permissive = validate_ir(ir, strict=False)
            if not validation_permissive.errors:
                result["validated_permissive"] = True
                validate_permissive_pass += 1
                # Record compiler risk warnings
                risk_warns = [w for w in validation_permissive.warnings if w.category == "compiler_risk"]
                if risk_warns:
                    result["compiler_risk_warnings"] = [w.message for w in risk_warns]
            else:
                if not result["error"]:
                    result["error"] = f"validate: {validation_permissive.errors[0]}"
        except Exception as e:
            if not result["error"]:
                result["error"] = f"validate_permissive: {str(e)[:100]}"

        # Compile — strict path (only if strict validation passed)
        if result["validated_strict"]:
            try:
                compile_ir(ir)
                result["compiled_strict"] = True
                result["compiled"] = True  # backward compat
                compile_strict_pass += 1
            except Exception as e:
                result["error"] = f"compile: {str(e)[:100]}"

        # Compile — permissive path (only if permissive validation passed)
        if result["validated_permissive"]:
            if result["compiled_strict"]:
                # Already compiled successfully via strict path
                result["compiled_permissive"] = True
                compile_permissive_pass += 1
            else:
                try:
                    compile_ir(ir)
                    result["compiled_permissive"] = True
                    compile_permissive_pass += 1
                except Exception as e:
                    if not result["error"]:
                        result["error"] = f"compile_permissive: {str(e)[:100]}"

        # Runtime unverified: compiles in permissive but NOT strict
        if result["compiled_permissive"] and not result["compiled_strict"]:
            result["runtime_unverified"] = True

        result["failure_category"] = _classify_failure(result)
        results.append(result)
        if verbose:
            if result["compiled_strict"]:
                status = "OK"
            elif result["compiled_permissive"]:
                status = "OK_PERM"
            elif result["validated_strict"]:
                status = "COMP_FAIL"
            elif result["validated_permissive"]:
                status = "VALID_STRICT_FAIL"
            else:
                status = "VALID_FAIL"
            lint_note = f" [lint:{lint_repairs}]" if lint_repairs > 0 else ""
            cat = f" [{result['failure_category']}]" if result["failure_category"] else ""
            budget_note = f" [{budget.complexity}={effective_max_tokens}]"
            timeout_note = " [TIMEOUT]" if was_timed_out else ""
            early_note = " [EARLY]" if gen_meta.early_stopped else ""
            print(f"  [{i+1}/{total}] {status}{cat}{lint_note}{budget_note}{timeout_note}{early_note} ({gen_time:.1f}s) {description[:50]}", flush=True)

    # Summary
    avg_time = sum(generation_times) / len(generation_times) if generation_times else 0
    p50 = sorted(generation_times)[len(generation_times)//2] if generation_times else 0
    max_time = max(generation_times) if generation_times else 0

    # Count lint stats
    total_lint_repairs = sum(r.get("lint_repairs", 0) for r in results)
    examples_linted = sum(1 for r in results if r.get("lint_repairs", 0) > 0)

    # Count failure categories
    failure_categories: dict[str, int] = {}
    for r in results:
        cat = r.get("failure_category")
        if cat:
            failure_categories[cat] = failure_categories.get(cat, 0) + 1

    # Runtime unverified count
    runtime_unverified = sum(1 for r in results if r.get("runtime_unverified", False))

    # Budget complexity distribution
    budget_dist: dict[str, int] = {}
    for r in results:
        bc = r.get("budget_complexity", "unknown")
        budget_dist[bc] = budget_dist.get(bc, 0) + 1

    stats = {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "eval_file": eval_file,
        "total": total,
        "parse_pass": parse_pass,
        "parse_rate": round(parse_pass / total * 100, 1) if total else 0,
        # Backward compat: "validate_pass" = strict
        "validate_pass": validate_strict_pass,
        "validate_rate": round(validate_strict_pass / total * 100, 1) if total else 0,
        # Dual metrics
        "validate_strict_pass": validate_strict_pass,
        "validate_strict_rate": round(validate_strict_pass / total * 100, 1) if total else 0,
        "validate_permissive_pass": validate_permissive_pass,
        "validate_permissive_rate": round(validate_permissive_pass / total * 100, 1) if total else 0,
        # Backward compat: "compile_pass" = strict
        "compile_pass": compile_strict_pass,
        "compile_rate": round(compile_strict_pass / total * 100, 1) if total else 0,
        # Dual compile metrics
        "compile_strict_pass": compile_strict_pass,
        "compile_strict_rate": round(compile_strict_pass / total * 100, 1) if total else 0,
        "compile_permissive_pass": compile_permissive_pass,
        "compile_permissive_rate": round(compile_permissive_pass / total * 100, 1) if total else 0,
        # Runtime unverified KPI
        "runtime_unverified_compile_pass": runtime_unverified,
        "runtime_unverified_compile_rate": round(runtime_unverified / total * 100, 1) if total else 0,
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / total * 100, 1) if total else 0,
        "failure_categories": failure_categories,
        "lint_repairs_total": total_lint_repairs,
        "examples_linted": examples_linted,
        "avg_time_s": round(avg_time, 1),
        "p50_time_s": round(p50, 1),
        "max_time_s": round(max_time, 1),
        # New Phase 10 fields
        "timeout_count": timeout_count,
        "budget_retries": budget_retries,
        "budget_mode": budget_mode,
        "timeout_s": timeout_s,
        "budget_distribution": budget_dist,
        "results": results,
    }

    # Write distillation log if requested
    if log_distillation:
        _write_distillation_log(
            results=results,
            model_path=model_path,
            adapter_path=adapter_path,
            eval_file=eval_file,
            output_path=distillation_output,
            append=append_distillation,
        )

    return stats


# ============================================================
# Distillation data logging
# ============================================================

def _get_git_hash() -> str:
    """Best-effort git rev-parse HEAD."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_SCRIPT_DIR),
        )
        return result.stdout.strip()[:12] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_catalog_version() -> str:
    """Read action catalog version from _meta.version."""
    catalog_path = _SCRIPT_DIR.parent / "references" / "action_catalog.json"
    try:
        with open(catalog_path) as f:
            catalog = json.load(f)
        return catalog.get("_meta", {}).get("version", "unknown")
    except Exception:
        return "unknown"


def _write_distillation_log(
    results: list[dict],
    model_path: str,
    adapter_path: str | None,
    eval_file: str,
    output_path: str | None = None,
    append: bool = False,
) -> None:
    """Write provenance-enriched distillation log to JSONL.

    Args:
        output_path: Custom output path. Defaults to training_data/distillation_log.jsonl.
        append: If True, append to existing file instead of truncating.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    git_hash = _get_git_hash()
    linter_version = getattr(_linter_module, "__version__", "unknown")
    catalog_version = _get_catalog_version()

    provenance = {
        "model_id": model_path,
        "adapter_id": adapter_path or "none",
        "linter_version": linter_version,
        "validator_mode": "strict+permissive",
        "grammar_used": False,
        "schema_version": f"action_catalog_v{catalog_version}",
        "eval_timestamp": timestamp,
        "eval_script_git_hash": git_hash,
    }

    if output_path is None:
        output_path = os.path.join(os.path.dirname(eval_file), "distillation_log.jsonl")
    written = 0
    mode = "a" if append else "w"

    with open(output_path, mode) as f:
        for r in results:
            entry = {
                "provenance": provenance,
                "shortcut_id": r.get("shortcut_id", ""),
                "prompt": r.get("description", ""),
                "raw_output": r.get("_raw_dsl_text", ""),
                "canonicalized_output": r.get("_canonicalized_dsl_text", ""),
                "lint_changes": r.get("lint_changes", []),
                "parsed": r.get("parsed", False),
                "validated_strict": r.get("validated_strict", False),
                "validated_permissive": r.get("validated_permissive", False),
                "compiled_strict": r.get("compiled_strict", False),
                "compiled_permissive": r.get("compiled_permissive", False),
                "runtime_unverified": r.get("runtime_unverified", False),
                "compiler_risk_warnings": r.get("compiler_risk_warnings", []),
                "failure_category": r.get("failure_category"),
                "gen_time_s": r.get("gen_time", 0),
                "timed_out": r.get("timed_out", False),
                "token_budget": r.get("token_budget"),
                "budget_complexity": r.get("budget_complexity"),
                # Distillation v2 metadata
                "scenario_profile": r.get("scenario_profile", "default"),
                "domain_profile": r.get("domain_profile", "general"),
                "architecture_decision": r.get("architecture_decision", "shortcut_only"),
                "creativity_score": r.get("creativity_score"),
            }
            f.write(json.dumps(entry) + "\n")
            written += 1

    print(f"\n  Distillation log: {written} entries written to {output_path}", flush=True)

    # Count hard-negative training pairs
    hard_negatives = sum(
        1 for r in results
        if r.get("parsed", False) and r.get("lint_repairs", 0) > 0
    )
    if hard_negatives:
        print(f"  Hard-negative pairs (linter-repaired + parsed): {hard_negatives}", flush=True)


def _print_stratified(results: list[dict], group_key: str, label: str) -> None:
    """Print stratified pass rates grouped by a result field."""
    from collections import defaultdict

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        group = r.get(group_key, "unknown")
        groups[group].append(r)

    print(f"\n  {'='*50}", flush=True)
    print(f"  STRATIFIED BY {label.upper()}", flush=True)
    print(f"  {'='*50}", flush=True)
    print(f"  {'Group':20s} {'N':>5s} {'Parse':>7s} {'Val-S':>7s} {'Val-P':>7s} {'Comp-S':>7s} {'Comp-P':>7s}", flush=True)
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}", flush=True)

    for group_name in sorted(groups.keys()):
        items = groups[group_name]
        n = len(items)
        if n == 0:
            continue
        parse = sum(1 for r in items if r.get("parsed", False))
        val_s = sum(1 for r in items if r.get("validated_strict", False))
        val_p = sum(1 for r in items if r.get("validated_permissive", False))
        comp_s = sum(1 for r in items if r.get("compiled_strict", False))
        comp_p = sum(1 for r in items if r.get("compiled_permissive", False))

        def pct(x: int) -> str:
            return f"{x/n*100:5.1f}%"

        print(f"  {group_name:20s} {n:5d} {pct(parse):>7s} {pct(val_s):>7s} {pct(val_p):>7s} {pct(comp_s):>7s} {pct(comp_p):>7s}", flush=True)

    # Totals
    n_total = len(results)
    print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}", flush=True)
    t_parse = sum(1 for r in results if r.get("parsed", False))
    t_val_s = sum(1 for r in results if r.get("validated_strict", False))
    t_val_p = sum(1 for r in results if r.get("validated_permissive", False))
    t_comp_s = sum(1 for r in results if r.get("compiled_strict", False))
    t_comp_p = sum(1 for r in results if r.get("compiled_permissive", False))

    def pct_t(x: int) -> str:
        return f"{x/n_total*100:5.1f}%"

    print(f"  {'TOTAL':20s} {n_total:5d} {pct_t(t_parse):>7s} {pct_t(t_val_s):>7s} {pct_t(t_val_p):>7s} {pct_t(t_comp_s):>7s} {pct_t(t_comp_p):>7s}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog="evaluate_model",
        description="Evaluate fine-tuned model on held-out eval set",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx-community/Meta-Llama-3.1-8B-Instruct",
        help="Base model path (default: standard 8B)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="MLX adapter directory",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="training_data/shortcutdsl_eval.jsonl",
        help="Eval JSONL file",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--skip-examples",
        type=int,
        default=0,
        help="Number of examples to skip from the start (for batched processing)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Legacy: fixed max tokens per generation (deprecated, use --fixed-budget)",
    )
    parser.add_argument(
        "--fixed-budget",
        type=int,
        default=None,
        help="Fixed token budget for all examples (overrides dynamic budgeting)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Wall-clock timeout per generation in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parser.add_argument(
        "--no-budget-retry",
        action="store_true",
        help="Disable overflow-triggered budget retry",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show per-example results",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Save baseline snapshot to training_data/baseline_snapshot.json for regression gating",
    )
    parser.add_argument(
        "--log-distillation",
        action="store_true",
        help="Write provenance-enriched distillation log to training_data/distillation_log.jsonl",
    )
    parser.add_argument(
        "--distillation-output",
        default=None,
        help="Custom output path for distillation log (default: training_data/distillation_log.jsonl)",
    )
    parser.add_argument(
        "--append-distillation",
        action="store_true",
        help="Append to existing distillation log instead of truncating",
    )
    parser.add_argument(
        "--chat-template",
        default="llama3",
        choices=["llama3", "chatml"],
        help="Chat template format for prompt construction (default: llama3)",
    )
    parser.add_argument(
        "--by-domain",
        action="store_true",
        help="Print stratified results by domain profile",
    )
    parser.add_argument(
        "--by-complexity",
        action="store_true",
        help="Print stratified results by complexity tier",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    eval_file = args.eval_file if os.path.isabs(args.eval_file) else str(project_root / args.eval_file)

    print(f"\nShortcutForge: Evaluating model\n", flush=True)

    stats = evaluate(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        eval_file=eval_file,
        max_examples=args.max_examples,
        skip_examples=args.skip_examples,
        max_tokens=args.max_tokens,
        fixed_budget=args.fixed_budget,
        timeout_s=args.timeout,
        budget_retry=not args.no_budget_retry,
        verbose=args.verbose,
        log_distillation=args.log_distillation,
        distillation_output=args.distillation_output,
        append_distillation=args.append_distillation,
        chat_template=args.chat_template,
    )

    print(f"\n  {'='*50}", flush=True)
    print(f"  EVALUATION RESULTS (with linter)", flush=True)
    print(f"  {'='*50}", flush=True)
    print(f"  Total examples:    {stats['total']}", flush=True)
    print(f"  Budget mode:       {stats['budget_mode']} (timeout={stats['timeout_s']}s)", flush=True)
    print(f"  Lint repairs:      {stats['lint_repairs_total']} repairs across {stats['examples_linted']} examples", flush=True)
    print(f"  Parse pass:        {stats['parse_pass']}/{stats['total']} ({stats['parse_rate']}%)", flush=True)
    print(f"  Validate (strict): {stats['validate_strict_pass']}/{stats['total']} ({stats['validate_strict_rate']}%)", flush=True)
    print(f"  Validate (perm):   {stats['validate_permissive_pass']}/{stats['total']} ({stats['validate_permissive_rate']}%)", flush=True)
    print(f"  Compile (strict):  {stats['compile_strict_pass']}/{stats['total']} ({stats['compile_strict_rate']}%)", flush=True)
    print(f"  Compile (perm):    {stats['compile_permissive_pass']}/{stats['total']} ({stats['compile_permissive_rate']}%)", flush=True)
    print(f"    runtime_unverif: {stats['runtime_unverified_compile_pass']}/{stats['total']} ({stats['runtime_unverified_compile_rate']}%)", flush=True)
    print(f"    fallback_rate:   {stats['fallback_count']}/{stats['total']} ({stats['fallback_rate']}%)", flush=True)
    print(f"  Avg gen time:      {stats['avg_time_s']}s", flush=True)
    print(f"  P50 gen time:      {stats['p50_time_s']}s", flush=True)
    print(f"  Max gen time:      {stats['max_time_s']}s", flush=True)

    # Timeout/budget stats
    if stats.get("timeout_count", 0) > 0:
        print(f"  Timeouts:          {stats['timeout_count']}", flush=True)
    if stats.get("budget_retries", 0) > 0:
        print(f"  Budget retries:    {stats['budget_retries']}", flush=True)

    # Budget distribution
    bd = stats.get("budget_distribution", {})
    if bd and stats["budget_mode"] == "dynamic":
        print(f"\n  Budget distribution:", flush=True)
        for tier, count in sorted(bd.items()):
            print(f"    {tier}: {count}", flush=True)

    # Failure category breakdown
    fc = stats.get("failure_categories", {})
    if fc:
        print(f"\n  Failure categories:", flush=True)
        for cat, count in sorted(fc.items()):
            print(f"    {cat}: {count}", flush=True)

    print(f"  {'='*50}", flush=True)

    # Save results
    output_path = os.path.join(os.path.dirname(eval_file), "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Results saved to: {output_path}", flush=True)

    # Stratified results by domain
    if args.by_domain:
        _print_stratified(stats["results"], "domain_profile", "Domain")

    # Stratified results by complexity
    if args.by_complexity:
        _print_stratified(stats["results"], "budget_complexity", "Complexity")

    # Save baseline snapshot if requested
    if args.snapshot:
        snapshot = {
            "snapshot_version": 3,
            "date": datetime.now(timezone.utc).isoformat(),
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "eval_file": eval_file,
            "n_examples": stats["total"],
            "budget_mode": stats["budget_mode"],
            "timeout_s": stats["timeout_s"],
            "metrics": {
                "parse_pass": stats["parse_pass"],
                "parse_rate": stats["parse_rate"],
                "validate_strict_pass": stats["validate_strict_pass"],
                "validate_strict_rate": stats["validate_strict_rate"],
                "validate_permissive_pass": stats["validate_permissive_pass"],
                "validate_permissive_rate": stats["validate_permissive_rate"],
                "compile_strict_pass": stats["compile_strict_pass"],
                "compile_strict_rate": stats["compile_strict_rate"],
                "compile_permissive_pass": stats["compile_permissive_pass"],
                "compile_permissive_rate": stats["compile_permissive_rate"],
                "runtime_unverified_compile_pass": stats["runtime_unverified_compile_pass"],
                "runtime_unverified_compile_rate": stats["runtime_unverified_compile_rate"],
                "fallback_count": stats["fallback_count"],
                "fallback_rate": stats["fallback_rate"],
            },
            "failure_categories": fc,
            "budget_distribution": bd,
        }
        snapshot_path = os.path.join(os.path.dirname(eval_file), "baseline_snapshot.json")
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"  Baseline snapshot saved to: {snapshot_path}", flush=True)


if __name__ == "__main__":
    main()
