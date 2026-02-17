#!/usr/bin/env python3
"""
ShortcutForge MCP Server — structured tool access for Claude Code.

Wraps the ShortcutForge pipeline as MCP tools so Claude Code can orchestrate
generation, evaluation, distillation, training, and promotion-gating through
structured calls instead of assembling bash commands.

Install:
    pip install mcp

Run standalone (for testing):
    python scripts/mcp_server.py

Configure in Claude Code (~/.claude/settings.json or project .mcp.json):
    {
      "mcpServers": {
        "shortcutforge": {
          "command": "python",
          "args": ["scripts/mcp_server.py"],
          "env": {}
        }
      }
    }
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = PROJECT_ROOT / "training"
TESTS_DIR = PROJECT_ROOT / "tests"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
MODELS_DIR = PROJECT_ROOT / "models"
REFERENCES_DIR = PROJECT_ROOT / "references"

# Legacy alias for backward compatibility
SCRIPTS_DIR = SRC_DIR

# Ensure src/ is importable
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("Error: mcp package not installed. Run: pip install mcp", file=sys.stderr)
    sys.exit(1)

_MCP_INSTRUCTIONS = (
    "ShortcutForge compiler pipeline for Apple Shortcuts DSL. "
    "Use tools to generate, lint, validate, analyze, compile, evaluate, distill, "
    "and run promotion gates. Prefer strict metrics for model quality and "
    "permissive metrics for best-effort utility."
)


def _build_mcp_server() -> FastMCP:
    """Instantiate FastMCP across mcp package versions."""
    try:
        sig = inspect.signature(FastMCP.__init__)
        if "instructions" in sig.parameters:
            return FastMCP("ShortcutForge", instructions=_MCP_INSTRUCTIONS)
        if "description" in sig.parameters:
            return FastMCP("ShortcutForge", description=_MCP_INSTRUCTIONS)
    except Exception:
        pass

    # Conservative fallback for older/newer API drift
    try:
        return FastMCP("ShortcutForge", instructions=_MCP_INSTRUCTIONS)
    except TypeError:
        return FastMCP("ShortcutForge")


mcp = _build_mcp_server()


# ── Helpers ──────────────────────────────────────────────────────────

def _run_script(script_name: str, args: list[str], timeout: int = 600) -> dict:
    """Run a script in the scripts/ directory and capture output."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout,
            "stderr": result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr,
            "truncated": len(result.stdout) > 8000,
        }
    except subprocess.TimeoutExpired:
        return {"returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s", "truncated": False}
    except Exception as e:
        return {"returncode": -1, "stdout": "", "stderr": str(e), "truncated": False}


def _read_json(path: Path) -> dict | list | None:
    """Read a JSON file, return None on failure."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _run_health_logger_scenario(
    model_path: str,
    adapter_path: str | None,
    engine: str = "local",
    timeout: int = 900,
) -> tuple[dict, float | None]:
    """Run health_logger scenario and return (raw result, average score)."""
    os.makedirs(PROJECT_ROOT / "output", exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        dir=str(PROJECT_ROOT / "output"),
    ) as f:
        output_path = f.name

    args = [
        "--scenario", str(REFERENCES_DIR / "scenario_packs" / "health_logger"),
        "--output", output_path,
        "--engine", engine,
        "-v",
    ]
    if engine == "local":
        args += ["--model-path", model_path]
        if adapter_path:
            args += ["--adapter-path", adapter_path]
    else:
        args += ["--model", model_path]

    try:
        result = _run_script("evaluate_scenario.py", args, timeout=timeout)
        exported = _read_json(Path(output_path))
        avg_score = None
        if isinstance(exported, dict):
            score = exported.get("average_score")
            if isinstance(score, (int, float)):
                avg_score = float(score)
            result["scenario_export"] = {
                "average_score": avg_score,
                "scenario_id": exported.get("scenario_id"),
                "scenario_name": exported.get("scenario_name"),
            }
        return result, avg_score
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


def _normalize_eval_for_gating(eval_results: dict) -> dict:
    """Normalize eval results so gate checks are backward-compatible."""
    normalized = dict(eval_results)
    if not isinstance(normalized.get("fallback_rate"), (int, float)):
        normalized["fallback_rate"] = 0.0
    return normalized


# ── Generation Tools ─────────────────────────────────────────────────

@mcp.tool()
def forge_generate(
    prompt: str,
    engine: str = "claude",
    model_path: str | None = None,
    adapter_path: str | None = None,
    creative_mode: str = "pragmatic",
    chat_template: str = "llama3",
    no_grammar: bool = False,
    verbose: bool = True,
) -> str:
    """Generate an Apple Shortcut from natural language.

    Uses the full ShortcutForge pipeline: generation → lint → parse → validate → simulate → compile → sign.

    Args:
        prompt: Natural language description (e.g. "Set a 5-minute timer for tea")
        engine: "claude" (API) or "local" (MLX model)
        model_path: Local model path (required when engine=local)
        adapter_path: LoRA adapter path (for engine=local)
        creative_mode: pragmatic|expressive|playful|automation_dense|power_user
        chat_template: Chat template format - "llama3" or "chatml" (for Qwen models)
        no_grammar: Disable grammar fallback retries for local generation
        verbose: Show detailed pipeline output
    """
    args = [prompt, "--engine", engine, "--creative-mode", creative_mode]
    if model_path:
        args += ["--model-path", model_path]
    if adapter_path:
        args += ["--adapter-path", adapter_path]
    if engine == "local":
        args += ["--chat-template", chat_template]
        if no_grammar:
            args.append("--no-grammar")
    if verbose:
        args.append("-v")
    result = _run_script("shortcutforge.py", args, timeout=120)
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_compile_dsl(dsl_text: str) -> str:
    """Compile raw DSL text through the pipeline (no LLM generation).

    Useful for testing hand-written or modified DSL.

    Args:
        dsl_text: ShortcutDSL text (must start with SHORTCUT and end with ENDSHORTCUT)
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dsl", delete=False, dir=str(PROJECT_ROOT / "output")) as f:
        f.write(dsl_text)
        dsl_path = f.name
    try:
        result = _run_script("shortcutforge.py", ["--dsl-file", dsl_path, "-v"])
        return json.dumps(result, indent=2)
    finally:
        os.unlink(dsl_path)


@mcp.tool()
def forge_decompile(shortcut_path: str, validate: bool = True) -> str:
    """Decompile a .shortcut file back to DSL (reverse compiler).

    Args:
        shortcut_path: Path to .shortcut file
        validate: Run validation on the decompiled DSL
    """
    args = [shortcut_path]
    if validate:
        args.append("--validate")
    result = _run_script("plist_to_dsl.py", args)
    return json.dumps(result, indent=2)


# ── Evaluation Tools ─────────────────────────────────────────────────

@mcp.tool()
def forge_eval(
    model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    adapter_path: str | None = None,
    max_examples: int | None = None,
    log_distillation: bool = False,
    snapshot: bool = False,
    by_domain: bool = False,
    by_complexity: bool = False,
    timeout_per_example: float = 90,
    chat_template: str = "llama3",
) -> str:
    """Run model evaluation on the frozen eval set (100 examples).

    Measures parse/validate/compile rates in strict and permissive modes.
    This is the primary metric for model quality.

    Args:
        model_path: Base model path (HuggingFace repo or local)
        adapter_path: LoRA adapter directory (e.g. "models/baseline-v1-mlx")
        max_examples: Limit evaluation to N examples (for quick checks)
        log_distillation: Write distillation log with raw→canonicalized pairs
        snapshot: Save results as the new regression baseline
        by_domain: Show stratified results by domain profile
        by_complexity: Show stratified results by complexity tier
        timeout_per_example: Wall-clock timeout per generation (seconds)
        chat_template: Chat template format - "llama3" or "chatml" (for Qwen models)
    """
    args = ["--model-path", model_path, "-v", "--timeout", str(timeout_per_example),
            "--chat-template", chat_template]
    if adapter_path:
        args += ["--adapter-path", adapter_path]
    if max_examples:
        args += ["--max-examples", str(max_examples)]
    if log_distillation:
        args.append("--log-distillation")
    if snapshot:
        args.append("--snapshot")
    if by_domain:
        args.append("--by-domain")
    if by_complexity:
        args.append("--by-complexity")

    # Long timeout: 100 examples * 90s each = 2.5 hours max
    total_timeout = int((max_examples or 100) * timeout_per_example * 1.2)
    result = _run_script("evaluate_model.py", args, timeout=total_timeout)

    # Also read the eval results JSON if it exists
    eval_results = _read_json(TRAINING_DATA_DIR / "eval_results.json")
    if eval_results:
        # Extract just the summary metrics, not per-example results
        summary = {k: v for k, v in eval_results.items() if k != "results"}
        result["eval_metrics"] = summary

    return json.dumps(result, indent=2)


@mcp.tool()
def forge_eval_scenario(
    scenario: str | None = None,
    all_scenarios: bool = False,
    score_reference: bool = False,
    model_path: str | None = None,
    adapter_path: str | None = None,
    engine: str = "local",
) -> str:
    """Run scenario-parity benchmark with rubric-based scoring.

    Available scenarios: api_pagination_fetcher, calendar_triage, clipboard_utility,
    file_router, health_logger, media_metadata_pipeline, morning_routine,
    share_sheet_text_cleaner

    Args:
        scenario: Scenario pack name (e.g. "health_logger") or full path
        all_scenarios: Evaluate all 8 scenario packs
        score_reference: Only score the reference DSL (no generation needed)
        model_path: Base model path (for generation)
        adapter_path: LoRA adapter path (for generation)
        engine: "local" or "claude"
    """
    args = []
    if all_scenarios:
        args.append("--all-scenarios")
    elif scenario:
        # Resolve scenario name to path if not already a path
        if not os.path.sep in scenario:
            scenario = str(REFERENCES_DIR / "scenario_packs" / scenario)
        args += ["--scenario", scenario]
    if score_reference:
        args.append("--score-reference")
    if model_path:
        args += ["--model-path", model_path]
    if adapter_path:
        args += ["--adapter-path", adapter_path]
    args += ["--engine", engine, "-v"]

    result = _run_script("evaluate_scenario.py", args, timeout=300)
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_regression_check() -> str:
    """Run the no-regression gate against the frozen baseline.

    Compares training_data/eval_results.json against training_data/baseline_snapshot.json.
    Returns PASS (exit 0) or FAIL (exit 1) with details on what regressed.
    """
    result = _run_script("check_regression.py", ["-v"])

    # Include the baseline for context
    baseline = _read_json(TRAINING_DATA_DIR / "baseline_snapshot.json")
    if baseline:
        result["baseline_metrics"] = baseline.get("metrics", {})

    return json.dumps(result, indent=2)


@mcp.tool()
def forge_promotion_check(
    model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    adapter_path: str | None = None,
    run_scenario: bool = False,
) -> str:
    """Check if a model meets all promotion gates from model_profiles.yaml.

    Gates (from PLAN.md Phase 4):
      - compile_strict_rate >= 95.0
      - compile_permissive_rate >= 97.0
      - runtime_unverified_compile_rate <= 2.0
      - fallback_rate <= 5.0
      - health_logger_scenario_score >= 0.85

    Reads the latest eval_results.json (run forge_eval first).

    Args:
        model_path: Model that was evaluated
        adapter_path: Adapter that was evaluated
        run_scenario: Also run health_logger scenario benchmark
    """
    eval_results = _read_json(TRAINING_DATA_DIR / "eval_results.json")
    if not eval_results:
        return json.dumps({"error": "No eval_results.json found. Run forge_eval first."})
    eval_results = _normalize_eval_for_gating(eval_results)

    scenario_result = None
    scenario_score = None
    if run_scenario:
        engine = "claude" if model_path.startswith("claude-") else "local"
        scenario_result, scenario_score = _run_health_logger_scenario(
            model_path=model_path,
            adapter_path=adapter_path,
            engine=engine,
        )
        if scenario_score is not None:
            eval_results = dict(eval_results)
            eval_results["health_logger_scenario_score"] = round(scenario_score, 3)

    # Try to load gates from model_profiles.yaml
    try:
        from model_profiles import check_promotion as _check_gates
        metrics = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
        gate_result = _check_gates(metrics)
    except Exception:
        # Fallback to hardcoded gates
        gate_result = _hardcoded_promotion_check(eval_results)

    baseline = _read_json(TRAINING_DATA_DIR / "baseline_snapshot.json")

    result = {
        "promoted": gate_result["passed"],
        "model_path": model_path,
        "adapter_path": adapter_path,
        "gates_passed": gate_result["gates_passed"],
        "total_gates": gate_result["total_gates"],
        "gate_details": gate_result["gates"],
        "current_baseline": baseline.get("metrics", {}) if baseline else None,
        "recommendation": "Model meets all promotion gates!" if gate_result["passed"]
            else "Model does NOT meet promotion gates. See failing metrics above.",
    }
    if eval_results.get("health_logger_scenario_score") is None and not run_scenario:
        result["note"] = (
            "health_logger_scenario_score is missing. "
            "Re-run with run_scenario=true for a complete gate check."
        )

    # Include scenario benchmark result when requested
    if run_scenario:
        result["health_logger_scenario"] = scenario_result
        if scenario_score is not None:
            result["health_logger_scenario_score"] = round(scenario_score, 3)

    return json.dumps(result, indent=2)


def _hardcoded_promotion_check(eval_results: dict) -> dict:
    """Fallback promotion check when model_profiles.yaml isn't available."""
    gates = {
        "compile_strict_rate": {"threshold": 95.0, "direction": "min"},
        "compile_permissive_rate": {"threshold": 97.0, "direction": "min"},
        "runtime_unverified_compile_rate": {"threshold": 2.0, "direction": "max"},
        "fallback_rate": {"threshold": 5.0, "direction": "max"},
        "health_logger_scenario_score": {"threshold": 0.85, "direction": "min"},
    }

    gate_details = []
    all_pass = True
    for metric, gate in gates.items():
        actual = eval_results.get(metric, 0)
        if gate["direction"] == "min":
            passed = actual >= gate["threshold"]
        else:
            passed = actual <= gate["threshold"]
        op = ">=" if gate["direction"] == "min" else "<="
        gate_details.append({
            "gate": f"{metric} {op} {gate['threshold']}",
            "metric": metric,
            "value": actual,
            "threshold": gate["threshold"],
            "passed": passed,
        })
        if not passed:
            all_pass = False

    return {
        "passed": all_pass,
        "gates": gate_details,
        "total_gates": len(gates),
        "gates_passed": sum(1 for g in gate_details if g["passed"]),
    }


# ── Distillation & Training Tools ────────────────────────────────────

@mcp.tool()
def forge_distill_generate(
    model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    adapter_path: str = "models/baseline-v1-mlx",
    input_file: str | None = None,
    max_examples: int | None = None,
    chat_template: str = "llama3",
) -> str:
    """Generate distillation data from the teacher model.

    Runs evaluate_model.py with --log-distillation to produce raw→canonicalized
    pairs suitable for training a student model.

    Args:
        model_path: Teacher model path
        adapter_path: Teacher adapter path
        input_file: Training JSONL to generate against (default: expanded training set)
        max_examples: Limit to N examples (for testing)
        chat_template: Chat template format - "llama3" or "chatml" (for Qwen models)
    """
    eval_file = input_file or str(TRAINING_DATA_DIR / "shortcutdsl_train_expanded.jsonl")
    args = [
        "--model-path", model_path,
        "--adapter-path", adapter_path,
        "--eval-file", eval_file,
        "--log-distillation",
        "--chat-template", chat_template,
        "-v",
    ]
    if max_examples:
        args += ["--max-examples", str(max_examples)]

    total_timeout = int((max_examples or 6679) * 95 * 1.2)
    result = _run_script("evaluate_model.py", args, timeout=min(total_timeout, 36000))
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_distill_curate(
    input_file: str | None = None,
    output_file: str | None = None,
) -> str:
    """Curate distillation data: quality-filter, dedup, scenario-balance.

    Input: distillation log JSONL (from forge_distill_generate)
    Output: curated training JSONL ready for fine-tuning

    Args:
        input_file: Path to distillation log (default: training_data/distillation_log.jsonl)
        output_file: Output path (default: training_data/distilled_curated.jsonl)
    """
    inp = input_file or str(TRAINING_DATA_DIR / "distillation_log.jsonl")
    out = output_file or str(TRAINING_DATA_DIR / "distilled_curated.jsonl")
    args = [inp, "--output", out]
    result = _run_script("distillation_curator.py", args)
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_train(
    model: str,
    data_dir: str | None = None,
    adapter_path: str | None = None,
    batch_size: int = 4,
    lora_layers: int = 16,
    lora_rank: int = 16,
    learning_rate: float = 1e-4,
    iters: int = 1000,
    steps_per_eval: int = 100,
    method: str = "lora",
) -> str:
    """Fine-tune a model with MLX using LoRA/DoRA/full training.

    This is the core training command. Uses mlx_lm.lora under the hood.
    The data directory must contain train.jsonl and valid.jsonl files.

    Args:
        model: Model path (e.g. "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
               "Qwen/Qwen2.5-0.5B-Instruct", "meta-llama/Llama-3.2-1B-Instruct")
        data_dir: Directory with train.jsonl and valid.jsonl (default: training_data/)
        adapter_path: Output adapter path (default: auto-generated in models/)
        batch_size: Training batch size
        lora_layers: Number of LoRA layers
        lora_rank: LoRA rank
        learning_rate: Learning rate
        iters: Number of training iterations
        steps_per_eval: Run validation every N steps
        method: "lora", "dora", or "full"
    """
    data = data_dir or str(TRAINING_DATA_DIR)

    if not adapter_path:
        # Auto-generate run name
        model_short = model.split("/")[-1].lower().replace("-", "_")[:20]
        run_id = f"{model_short}_{method}_r{lora_rank}_i{iters}"
        adapter_path = str(MODELS_DIR / "local-runs" / run_id)

    # Check that train.jsonl exists
    train_file = Path(data) / "train.jsonl"
    if not train_file.exists():
        # Try to help by checking for the expanded file
        expanded = Path(data) / "shortcutdsl_train_expanded.jsonl"
        eval_file = Path(data) / "shortcutdsl_eval.jsonl"
        hint = ""
        if expanded.exists():
            hint = (f"\n\nHint: Found {expanded.name}. Create symlinks:\n"
                    f"  ln -s {expanded.name} {data}/train.jsonl\n"
                    f"  ln -s {eval_file.name} {data}/valid.jsonl")
        return json.dumps({
            "error": f"train.jsonl not found in {data}. mlx_lm.lora expects train.jsonl and valid.jsonl.{hint}"
        })

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", model,
        "--data", data,
        "--train",
        "--batch-size", str(batch_size),
        "--num-layers", str(lora_layers),
        "--learning-rate", str(learning_rate),
        "--iters", str(iters),
        "--steps-per-eval", str(steps_per_eval),
        "--adapter-path", adapter_path,
        "--fine-tune-type", method,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=36000,  # 10 hours max
            cwd=str(PROJECT_ROOT),
        )
        return json.dumps({
            "returncode": result.returncode,
            "stdout": result.stdout[-8000:] if len(result.stdout) > 8000 else result.stdout,
            "stderr": result.stderr[-4000:] if len(result.stderr) > 4000 else result.stderr,
            "adapter_path": adapter_path,
            "model": model,
            "method": method,
            "next_step": f"Run: forge_eval(model_path='{model}', adapter_path='{adapter_path}')",
        }, indent=2)
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Training timed out after 10 hours", "adapter_path": adapter_path})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def forge_fuse_adapter(
    model: str,
    adapter_path: str,
    save_path: str | None = None,
) -> str:
    """Fuse a LoRA adapter into the base model for faster inference.

    After fusing, the model can be used without specifying an adapter path.

    Args:
        model: Base model path
        adapter_path: LoRA adapter directory to fuse
        save_path: Output path for fused model (default: adapter_path + "-fused")
    """
    if not save_path:
        save_path = adapter_path.rstrip("/") + "-fused"

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", model,
        "--adapter-path", adapter_path,
        "--save-path", save_path,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT),
        )
        return json.dumps({
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "fused_model_path": save_path,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def forge_quantize(
    model: str,
    bits: int = 4,
    save_path: str | None = None,
) -> str:
    """Quantize a model for smaller size and faster inference.

    Useful for the feral compression lane (PLAN Phase 7).

    Args:
        model: Model path to quantize
        bits: Quantization bits (2, 4, or 8)
        save_path: Output path (default: model + f"-{bits}bit")
    """
    if not save_path:
        save_path = str(MODELS_DIR / "quantized" / f"{Path(model).name}-{bits}bit")

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", model,
        "--mlx-path", save_path,
        "-q",
        "--q-bits", str(bits),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(PROJECT_ROOT),
        )
        return json.dumps({
            "returncode": result.returncode,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-2000:],
            "quantized_model_path": save_path,
            "bits": bits,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Pipeline Inspection Tools ────────────────────────────────────────

@mcp.tool()
def forge_lint(dsl_text: str) -> str:
    """Run just the DSL linter on raw text and return repairs.

    Useful for debugging what the linter changes in LLM output.

    Args:
        dsl_text: Raw DSL text to lint
    """
    from dsl_linter import lint_dsl
    result = lint_dsl(dsl_text)
    return json.dumps({
        "canonicalized_text": result.text,
        "num_changes": len(result.changes),
        "changes": [
            {"kind": c.kind, "original": c.original, "replacement": c.replacement,
             "confidence": c.confidence, "reason": c.reason}
            for c in result.changes
        ],
    }, indent=2)


@mcp.tool()
def forge_validate(dsl_text: str, strict: bool = True) -> str:
    """Parse and validate DSL text, returning detailed results.

    Args:
        dsl_text: DSL text (should be linted first)
        strict: Use strict validation mode
    """
    from dsl_parser import parse_dsl
    from dsl_validator import validate_ir

    # Lint first
    from dsl_linter import lint_dsl
    lint_result = lint_dsl(dsl_text)
    linted = lint_result.text

    try:
        ir = parse_dsl(linted)
    except Exception as e:
        return json.dumps({"parsed": False, "error": str(e), "lint_changes": len(lint_result.changes)})

    try:
        validation = validate_ir(ir, strict=strict)
        return json.dumps({
            "parsed": True,
            "validated": len(validation.errors) == 0,
            "errors": [str(e) for e in validation.errors],
            "warnings": [str(w) for w in validation.warnings],
            "lint_changes": len(lint_result.changes),
            "action_count": len([s for s in ir.statements if hasattr(s, "action_name")]),
        }, indent=2)
    except Exception as e:
        return json.dumps({"parsed": True, "validated": False, "error": str(e)})


@mcp.tool()
def forge_analyze(dsl_text: str) -> str:
    """Run static analysis (simulation harness) on DSL text.

    Performs 7 analyses: variable flow, loop bounds, menu completeness,
    dead code, API validation, type flow, contract validation.

    Args:
        dsl_text: DSL text to analyze
    """
    from dsl_parser import parse_dsl
    from dsl_linter import lint_dsl
    from simulation_harness import SimulationHarness

    lint_result = lint_dsl(dsl_text)
    try:
        ir = parse_dsl(lint_result.text)
    except Exception as e:
        return json.dumps({"error": f"Parse failed: {e}"})

    harness = SimulationHarness()
    report = harness.analyze(ir)

    return json.dumps({
        "findings_count": len(report.findings),
        "findings": [
            {"category": f.category.value, "severity": f.severity.value,
             "message": f.message, "location": f.location}
            for f in report.findings
        ],
        "summary": report.summary(),
    }, indent=2)


# ── Status & Inventory Tools ────────────────────────────────────────

@mcp.tool()
def forge_status() -> str:
    """Get current project status: models, data, latest metrics, baseline.

    Returns an overview of available adapters, training data sizes,
    eval results, and baseline metrics.
    """
    # Available adapters
    adapters = []
    if MODELS_DIR.exists():
        for d in sorted(MODELS_DIR.iterdir()):
            if d.is_dir() and (d / "adapter_config.json").exists():
                size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
                adapters.append({"name": d.name, "size_mb": round(size_mb, 1)})
        # Check local-runs subdirectory
        local_runs = MODELS_DIR / "local-runs"
        if local_runs.exists():
            for d in sorted(local_runs.iterdir()):
                if d.is_dir() and (d / "adapter_config.json").exists():
                    size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
                    adapters.append({"name": f"local-runs/{d.name}", "size_mb": round(size_mb, 1)})

    # Training data stats
    data_files = {}
    for f in TRAINING_DATA_DIR.glob("*.jsonl"):
        count = sum(1 for _ in open(f))
        data_files[f.name] = {"lines": count, "size_mb": round(f.stat().st_size / 1e6, 1)}

    # Latest eval results
    eval_results = _read_json(TRAINING_DATA_DIR / "eval_results.json")
    eval_summary = None
    if eval_results:
        eval_summary = {k: v for k, v in eval_results.items() if k != "results"}

    # Baseline
    baseline = _read_json(TRAINING_DATA_DIR / "baseline_snapshot.json")

    # Scenario packs
    packs = []
    packs_dir = REFERENCES_DIR / "scenario_packs"
    if packs_dir.exists():
        packs = [d.name for d in sorted(packs_dir.iterdir()) if d.is_dir()]

    return json.dumps({
        "project_root": str(PROJECT_ROOT),
        "adapters": adapters,
        "training_data": data_files,
        "latest_eval": eval_summary,
        "baseline": baseline.get("metrics", {}) if baseline else None,
        "scenario_packs": packs,
        "action_catalog_size": _count_actions(),
    }, indent=2)


def _count_actions() -> int:
    """Count actions in the catalog."""
    catalog = _read_json(REFERENCES_DIR / "action_catalog.json")
    if catalog and isinstance(catalog, dict):
        actions = catalog.get("actions")
        if isinstance(actions, dict):
            return len(actions)
        # Backward compatibility with legacy flat catalogs
        return len([k for k in catalog if not k.startswith("_")])
    return 0


@mcp.tool()
def forge_test(suite: str | None = None) -> str:
    """Run test suites.

    Args:
        suite: Specific test file name (e.g. "test_dsl_linter") or None for all tests.
               Available: test_dsl_linter, test_orchestrator, test_dsl_validator,
               test_compiler_unit, test_macro_expander, test_simulation_harness,
               test_snippet_extractor, test_contract_validator, test_execution_planner,
               test_domain_profile, test_architecture_reasoner, test_creative_scoring,
               test_scenario_profiles, test_xml_validator, test_scenario_packs,
               test_label_training_data
    """
    if suite:
        if not suite.startswith("test_"):
            suite = f"test_{suite}"
        if not suite.endswith(".py"):
            suite = f"{suite}.py"
        result = _run_script(suite, [], timeout=120)
        return json.dumps(result, indent=2)
    else:
        # Run all test suites
        test_files = sorted(TESTS_DIR.glob("test_*.py"))
        results = {}
        total_pass = 0
        total_fail = 0
        for tf in test_files:
            r = _run_script(tf.name, [], timeout=60)
            passed = r["returncode"] == 0
            results[tf.stem] = {"passed": passed, "returncode": r["returncode"]}
            if passed:
                total_pass += 1
            else:
                total_fail += 1
                results[tf.stem]["stderr_tail"] = r["stderr"][-500:] if r["stderr"] else ""

        return json.dumps({
            "total_suites": len(test_files),
            "passed": total_pass,
            "failed": total_fail,
            "suites": results,
        }, indent=2)


# ── Training Data Tools ──────────────────────────────────────────────

@mcp.tool()
def forge_build_training_data(verbose: bool = True) -> str:
    """Rebuild training data from the shortcut corpus.

    Decompiles downloaded .shortcut files → validates → produces JSONL.
    Input: downloaded/ directory (1,772 shortcuts)
    Output: training_data/shortcutdsl_train.jsonl + shortcutdsl_eval.jsonl

    Args:
        verbose: Show per-file details
    """
    args = []
    if verbose:
        args.append("-v")
    result = _run_script("build_training_data.py", args, timeout=600)
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_label_data() -> str:
    """Label training data with domain/complexity tags.

    Enriches training examples with domain profile and complexity tier labels
    for stratified evaluation.
    """
    result = _run_script("label_training_data.py", [])
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_extract_snippets(
    top_k: int = 200,
) -> str:
    """Extract micro-pattern snippets from training data for retrieval-augmented generation.

    Args:
        top_k: Number of top patterns to keep
    """
    args = [
        "--input", str(TRAINING_DATA_DIR / "shortcutdsl_train.jsonl"),
        "--output", str(REFERENCES_DIR / "snippet_registry.json"),
        "--top-k", str(top_k),
    ]
    result = _run_script("snippet_extractor.py", args, timeout=120)
    return json.dumps(result, indent=2)


# ── Sashimi Mode: Distillation & Training Pipeline ─────────────────

@mcp.tool()
def forge_distill_build(
    model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    adapter_path: str = "models/baseline-v1-mlx",
    batch_size: int = 500,
    chat_template: str = "llama3",
    curate_only: bool = False,
    merge_only: bool = False,
    no_shaping: bool = False,
) -> str:
    """Orchestrate the full Sashimi Mode distillation pipeline.

    Stages: generate distillation data → curate → convert to chat → merge gold + distilled.
    Applies lint-guided data shaping (weighting by repair profile).

    Args:
        model_path: Teacher model path
        adapter_path: Teacher adapter path
        batch_size: Examples per batch during generation
        chat_template: Chat template format - "llama3" or "chatml"
        curate_only: Skip generation, only curate existing log
        merge_only: Skip generation and curation, only merge
        no_shaping: Skip lint-based weighting in merge
    """
    args = [
        "--model-path", model_path,
        "--adapter-path", adapter_path,
        "--batch-size", str(batch_size),
        "--chat-template", chat_template,
        "-v",
    ]
    if curate_only:
        args.append("--curate-only")
    if merge_only:
        args.append("--merge-only")
    if no_shaping:
        args.append("--no-shaping")

    total_timeout = 36000 if not (curate_only or merge_only) else 600
    result = _run_script("build_distill_data.py", args, timeout=total_timeout)
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_train_preset(
    preset: str,
    run_id: str,
    method: str = "lora",
    merged: bool = False,
    iters: int | None = None,
    eval_after: bool = True,
) -> str:
    """Train a model using named ShortcutForge presets.

    Available presets: tiny_qwen (Qwen 0.5B), tiny_llama (Llama 1B), local_8b (Llama 8B)

    Args:
        preset: Preset name - "tiny_qwen", "tiny_llama", or "local_8b"
        run_id: Unique identifier for this training run (e.g. "qwen-gold-v1")
        method: Fine-tuning method - "lora", "dora", or "full"
        merged: Use merged (gold + distilled) training data instead of gold-only
        iters: Override default iteration count
        eval_after: Run evaluation on frozen eval set after training
    """
    args = ["--preset", preset, "--run-id", run_id, "--method", method, "-v"]
    if merged:
        args.append("--merged")
    if iters:
        args += ["--iters", str(iters)]
    if eval_after:
        args.append("--eval-after")

    result = _run_script("train_local_mlx.py", args, timeout=36000)
    return json.dumps(result, indent=2)


@mcp.tool()
def forge_profile_list() -> str:
    """List all available model profiles from configs/model_profiles.yaml.

    Shows each profile's model path, chat template, fallback order,
    and promotion gates.
    """
    try:
        from model_profiles import list_profiles, get_promotion_gates
        profiles = list_profiles()
        gates = get_promotion_gates()
        return json.dumps({
            "profiles": profiles,
            "promotion_gates": [str(g) for g in gates],
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def forge_prepare_training_data() -> str:
    """Create symlinks and validate training data format for MLX training.

    Ensures train.jsonl and valid.jsonl exist in training_data/ directory,
    pointing to the correct source files.
    """
    results = {}

    # Check and create symlinks
    train_link = TRAINING_DATA_DIR / "train.jsonl"
    valid_link = TRAINING_DATA_DIR / "valid.jsonl"
    gold_source = TRAINING_DATA_DIR / "shortcutdsl_train_expanded.jsonl"
    eval_source = TRAINING_DATA_DIR / "shortcutdsl_eval.jsonl"

    if not gold_source.exists():
        results["error"] = f"Gold training data not found: {gold_source}"
        return json.dumps(results, indent=2)

    if not eval_source.exists():
        results["error"] = f"Eval data not found: {eval_source}"
        return json.dumps(results, indent=2)

    # Create/verify symlinks
    for link, source, name in [
        (train_link, gold_source, "train.jsonl"),
        (valid_link, eval_source, "valid.jsonl"),
    ]:
        if link.exists() or link.is_symlink():
            target = os.readlink(link) if link.is_symlink() else "regular file"
            results[name] = f"exists (→ {target})"
        else:
            os.symlink(source.name, str(link))
            results[name] = f"created (→ {source.name})"

    # Count examples
    for f in [gold_source, eval_source]:
        count = sum(1 for _ in open(f))
        results[f.name] = f"{count} examples"

    # Check for merged data
    merged = TRAINING_DATA_DIR / "merged_train.jsonl"
    if merged.exists():
        count = sum(1 for _ in open(merged))
        results["merged_train.jsonl"] = f"{count} examples (available for --merged training)"

    return json.dumps(results, indent=2)


@mcp.tool()
def forge_compare_runs() -> str:
    """Compare eval metrics across all training runs in models/local-runs/.

    Returns runs sorted by compile_strict_rate descending.
    """
    try:
        from train_local_mlx import compare_runs
        runs = compare_runs()
        return json.dumps({
            "total_runs": len(runs),
            "runs": runs,
            "best_run": runs[0] if runs else None,
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def forge_promotion_report(
    model_path: str = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    adapter_path: str | None = None,
    chat_template: str = "llama3",
    run_eval: bool = True,
    run_scenario: bool = False,
) -> str:
    """One-call comprehensive promotion assessment: eval + scenario + gates.

    Runs evaluation if needed, checks all promotion gates, and optionally
    runs the health_logger scenario benchmark.

    Args:
        model_path: Model to evaluate
        adapter_path: LoRA adapter path
        chat_template: Chat template format - "llama3" or "chatml"
        run_eval: Run evaluation first (set False if eval_results.json is fresh)
        run_scenario: Also run health_logger scenario benchmark
    """
    report = {}

    # Step 1: Run eval if requested
    if run_eval:
        eval_args = [
            "--model-path", model_path, "-v",
            "--chat-template", chat_template,
            "--log-distillation",
        ]
        if adapter_path:
            eval_args += ["--adapter-path", adapter_path]
        eval_result = _run_script("evaluate_model.py", eval_args, timeout=12000)
        report["eval_output"] = eval_result["stdout"][-2000:]

    # Step 2: Optional scenario benchmark (model-generated, not reference-only)
    scenario_score = None
    if run_scenario:
        engine = "claude" if model_path.startswith("claude-") else "local"
        scenario_result, scenario_score = _run_health_logger_scenario(
            model_path=model_path,
            adapter_path=adapter_path,
            engine=engine,
        )
        report["health_logger_scenario"] = scenario_result
        if scenario_score is not None:
            report["health_logger_scenario_score"] = round(scenario_score, 3)

    # Step 3: Check promotion gates
    eval_results = _read_json(TRAINING_DATA_DIR / "eval_results.json")
    if eval_results:
        eval_results = _normalize_eval_for_gating(eval_results)
        if scenario_score is not None:
            eval_results = dict(eval_results)
            eval_results["health_logger_scenario_score"] = round(scenario_score, 3)
        try:
            from model_profiles import check_promotion as _check_gates
            metrics = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
            gate_result = _check_gates(metrics)
        except Exception:
            gate_result = _hardcoded_promotion_check(eval_results)

        report["promotion_gates"] = gate_result
        report["key_metrics"] = {
            "parse_rate": eval_results.get("parse_rate"),
            "validate_strict_rate": eval_results.get("validate_strict_rate"),
            "compile_strict_rate": eval_results.get("compile_strict_rate"),
            "compile_permissive_rate": eval_results.get("compile_permissive_rate"),
            "runtime_unverified_compile_rate": eval_results.get("runtime_unverified_compile_rate"),
            "fallback_rate": eval_results.get("fallback_rate"),
            "health_logger_scenario_score": eval_results.get("health_logger_scenario_score"),
        }
        if eval_results.get("health_logger_scenario_score") is None:
            report["note"] = (
                "health_logger_scenario_score is missing from eval_results. "
                "Run with run_scenario=true to evaluate all promotion gates."
            )

    # Step 4: Compare with baseline
    baseline = _read_json(TRAINING_DATA_DIR / "baseline_snapshot.json")
    if baseline:
        report["baseline_comparison"] = baseline.get("metrics", {})

    report["model_path"] = model_path
    report["adapter_path"] = adapter_path

    return json.dumps(report, indent=2)


@mcp.tool()
def forge_mine_errors(
    eval_results_path: str | None = None,
    distillation_log_path: str | None = None,
    min_frequency: int = 3,
) -> str:
    """Mine eval/distillation lint errors into linter improvement proposals.

    Closes the Sashimi feedback loop: analyzes lint repair profiles to discover
    new hallucination aliases, structural patterns, condition gaps, and handle
    anti-patterns that should be added to dsl_linter.py.

    Args:
        eval_results_path: Path to eval results JSON (default: training_data/eval_results.json)
        distillation_log_path: Path to distillation log (default: training_data/distillation_log.jsonl)
        min_frequency: Minimum frequency to report a discovery
    """
    args = ["--min-frequency", str(min_frequency), "-v"]
    if eval_results_path:
        args += ["--eval-results", eval_results_path]
    if distillation_log_path:
        args += ["--distillation-log", distillation_log_path]

    result = _run_script("mine_lint_errors.py", args, timeout=120)

    # Also read the mining report if it exists
    report = _read_json(TRAINING_DATA_DIR / "lint_mining_report.json")
    if report:
        result["mining_report"] = report

    return json.dumps(result, indent=2)


# ── MCP Resources & Prompts ──────────────────────────────────────────

@mcp.resource(
    "shortcutforge://status",
    name="ShortcutForge Status",
    description="Current model/data/eval inventory snapshot",
    mime_type="application/json",
)
def resource_status() -> str:
    """Resource: current project status snapshot."""
    return forge_status()


@mcp.resource(
    "shortcutforge://promotion-gates",
    name="Promotion Gates",
    description="Active promotion thresholds from configs/model_profiles.yaml",
    mime_type="application/json",
)
def resource_promotion_gates() -> str:
    """Resource: promotion gate policy."""
    try:
        from model_profiles import get_promotion_gates
        gates = [str(g) for g in get_promotion_gates()]
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)
    return json.dumps({"promotion_gates": gates}, indent=2)


@mcp.prompt(
    name="shortcutforge_generate_and_debug",
    description="Generate a shortcut, then lint/validate/analyze if it fails.",
)
def prompt_generate_and_debug(user_intent: str) -> str:
    return (
        f'Generate a shortcut for: "{user_intent}". '
        "Use forge_generate first. If it fails, run forge_lint, forge_validate "
        "(strict=true), and forge_analyze on the DSL, then propose a corrected DSL "
        "and confirm with forge_compile_dsl."
    )


@mcp.prompt(
    name="shortcutforge_promotion_audit",
    description="Run full promotion audit with scenario gate and report failures.",
)
def prompt_promotion_audit(model_path: str, adapter_path: str = "") -> str:
    adapter_note = f' adapter_path="{adapter_path}"' if adapter_path else ""
    return (
        f'Run forge_promotion_report(model_path="{model_path}"{adapter_note}, '
        "run_eval=true, run_scenario=true). Then summarize failing gates and "
        "the top two remediation actions."
    )


# ── Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
