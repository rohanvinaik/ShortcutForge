#!/usr/bin/env python3
"""Convenience wrapper around mlx_lm.lora with ShortcutForge presets.

Provides named presets for tiny/local/8B model training with
ShortcutForge-specific defaults: LoRA rank, batch size, iterations,
chat template, and data source selection.

Usage:
    # Train with a preset
    python train_local_mlx.py --preset tiny_qwen --run-id qwen-gold-v1

    # Train and auto-evaluate after
    python train_local_mlx.py --preset tiny_llama --run-id llama-gold-v1 --eval-after

    # DoRA variant
    python train_local_mlx.py --preset tiny_qwen --run-id qwen-dora-v1 --method dora

    # Use merged (gold + distilled) training data
    python train_local_mlx.py --preset tiny_llama --run-id llama-merged-v1 --merged
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
_PROJECT_ROOT = _SCRIPT_DIR.parent
_TRAINING_DIR = _PROJECT_ROOT / "training_data"
_MODELS_DIR = _PROJECT_ROOT / "models" / "local-runs"

# Training presets
PRESETS = {
    "tiny_qwen": {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "batch_size": 8,
        "lora_rank": 32,
        "lora_layers": 16,
        "iters": 1000,
        "learning_rate": 1e-4,
        "seq_length": 3072,
        "chat_template": "chatml",
    },
    "tiny_llama": {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "batch_size": 4,
        "lora_rank": 32,
        "lora_layers": 16,
        "iters": 1500,
        "learning_rate": 1e-4,
        "seq_length": 3072,
        "chat_template": "llama3",
    },
    "local_8b": {
        "model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "batch_size": 4,
        "lora_rank": 32,
        "lora_layers": 16,
        "iters": 1000,
        "learning_rate": 1e-4,
        "seq_length": 3072,
        "chat_template": "llama3",
    },
}


def resolve_data_dir(gold_only: bool = True) -> str:
    """Resolve training data directory.

    For gold-only mode, uses symlinks in training_data/ (train.jsonl → shortcutdsl_train_expanded.jsonl).
    For merged mode, expects merged_train.jsonl to be symlinked as train.jsonl.

    Returns:
        Path to the training data directory.
    """
    data_dir = str(_TRAINING_DIR)

    if not gold_only:
        merged_path = _TRAINING_DIR / "merged_train.jsonl"
        if not merged_path.exists():
            print(
                "  WARNING: merged_train.jsonl not found. Run build_distill_data.py first.",
                flush=True,
            )
            print("  Falling back to gold-only data.", flush=True)
            return data_dir

        # For merged mode, create a temp symlink pointing train.jsonl → merged_train.jsonl
        train_link = _TRAINING_DIR / "train.jsonl"
        if train_link.exists() or train_link.is_symlink():
            os.remove(train_link)
        os.symlink("merged_train.jsonl", str(train_link))
        print(
            "  Linked train.jsonl → merged_train.jsonl for merged training", flush=True
        )

    return data_dir


def restore_gold_symlink():
    """Restore train.jsonl symlink to point back to gold data."""
    train_link = _TRAINING_DIR / "train.jsonl"
    if train_link.exists() or train_link.is_symlink():
        os.remove(train_link)
    os.symlink("shortcutdsl_train_expanded.jsonl", str(train_link))


def train(
    preset_name: str,
    run_id: str,
    method: str = "lora",
    gold_only: bool = True,
    iters_override: int | None = None,
    max_examples: int | None = None,
    verbose: bool = False,
) -> dict:
    """Run LoRA/DoRA training with a preset.

    Args:
        preset_name: Preset name from PRESETS.
        run_id: Unique identifier for this training run.
        method: Fine-tuning method - "lora", "dora", or "full".
        gold_only: If True, use gold data only. If False, use merged data.
        iters_override: Override default iteration count.
        max_examples: Limit training examples (for smoke testing).
        verbose: Verbose training output.

    Returns:
        Training result dict with config, output path, timing.
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}"
        )

    preset = PRESETS[preset_name]
    output_dir = _MODELS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    iters = iters_override or preset["iters"]
    data_dir = resolve_data_dir(gold_only=gold_only)

    # Build mlx_lm.lora command
    # Note: mlx_lm >= 0.30 uses --num-layers (not --lora-layers)
    # and --fine-tune-type lora|dora|full (not --use-dora)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        preset["model"],
        "--data",
        data_dir,
        "--train",
        "--batch-size",
        str(preset["batch_size"]),
        "--num-layers",
        str(preset["lora_layers"]),
        "--iters",
        str(iters),
        "--learning-rate",
        str(preset["learning_rate"]),
        "--adapter-path",
        str(output_dir),
        "--fine-tune-type",
        method,
    ]

    # Log training config
    config = {
        "preset": preset_name,
        "run_id": run_id,
        "method": method,
        "gold_only": gold_only,
        "model": preset["model"],
        "chat_template": preset["chat_template"],
        "batch_size": preset["batch_size"],
        "lora_rank": preset["lora_rank"],
        "lora_layers": preset["lora_layers"],
        "iters": iters,
        "learning_rate": preset["learning_rate"],
        "seq_length": preset["seq_length"],
        "data_dir": data_dir,
        "output_dir": str(output_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    config_path = output_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n  ShortcutForge: Training with preset '{preset_name}'", flush=True)
    print(f"  Model:     {preset['model']}", flush=True)
    print(f"  Method:    {method}", flush=True)
    print(f"  Data:      {'gold-only' if gold_only else 'merged'}", flush=True)
    print(f"  Template:  {preset['chat_template']}", flush=True)
    print(f"  Iters:     {iters}", flush=True)
    print(f"  Output:    {output_dir}", flush=True)
    print(flush=True)

    t0 = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        timeout=3600 * 4,  # 4 hour timeout
    )
    elapsed = time.monotonic() - t0

    training_result = {
        "config": config,
        "returncode": result.returncode,
        "elapsed_s": round(elapsed, 1),
        "adapter_path": str(output_dir),
    }

    # Save training result
    result_path = output_dir / "training_result.json"
    with open(result_path, "w") as f:
        json.dump(training_result, f, indent=2)

    if result.returncode == 0:
        print(f"\n  Training complete in {elapsed:.0f}s", flush=True)
        print(f"  Adapter saved to: {output_dir}", flush=True)
    else:
        print(
            f"\n  Training FAILED (exit {result.returncode}) after {elapsed:.0f}s",
            flush=True,
        )

    # Restore gold symlink if we changed it
    if not gold_only:
        restore_gold_symlink()

    return training_result


def run_eval_after(
    preset_name: str,
    adapter_path: str,
    verbose: bool = False,
) -> dict | None:
    """Run evaluation on the frozen eval set after training.

    Uses the correct chat template for the preset.

    Returns:
        Eval stats dict, or None on failure.
    """
    preset = PRESETS[preset_name]
    chat_template = preset["chat_template"]
    eval_file = str(_TRAINING_DIR / "shortcutdsl_eval.jsonl")

    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "evaluate_model.py"),
        "--model-path",
        preset["model"],
        "--adapter-path",
        adapter_path,
        "--eval-file",
        eval_file,
        "--chat-template",
        chat_template,
        "--log-distillation",
    ]
    if verbose:
        cmd.append("--verbose")

    print("\n  Running evaluation on frozen eval set...", flush=True)
    print(f"  Template: {chat_template}", flush=True)

    result = subprocess.run(
        cmd,
        cwd=str(_PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=3600,
    )

    if result.returncode == 0:
        print(result.stdout, flush=True)
        # Try to load eval results
        eval_results_path = _TRAINING_DIR / "eval_results.json"
        if eval_results_path.exists():
            with open(eval_results_path) as f:
                return json.load(f)
    else:
        print(f"  Eval FAILED (exit {result.returncode})", flush=True)
        if verbose and result.stderr:
            print(f"  stderr: {result.stderr[:500]}", flush=True)

    return None


def compare_runs() -> list[dict]:
    """Compare eval metrics across all runs in models/local-runs/.

    Returns:
        List of run summary dicts sorted by compile_strict_rate descending.
    """
    runs = []

    if not _MODELS_DIR.exists():
        return runs

    for run_dir in sorted(_MODELS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "training_config.json"
        result_path = run_dir / "training_result.json"

        if not config_path.exists():
            continue

        with open(config_path) as f:
            config = json.load(f)

        summary = {
            "run_id": config.get("run_id", run_dir.name),
            "preset": config.get("preset", "unknown"),
            "method": config.get("method", "unknown"),
            "gold_only": config.get("gold_only", True),
            "model": config.get("model", "unknown"),
            "chat_template": config.get("chat_template", "unknown"),
            "timestamp": config.get("timestamp", ""),
        }

        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            summary["elapsed_s"] = result.get("elapsed_s", 0)
            summary["returncode"] = result.get("returncode", -1)

        # Check for eval results
        eval_path = run_dir / "eval_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                eval_data = json.load(f)
            summary["parse_rate"] = eval_data.get("parse_rate", 0)
            summary["validate_strict_rate"] = eval_data.get("validate_strict_rate", 0)
            summary["compile_strict_rate"] = eval_data.get("compile_strict_rate", 0)
            summary["compile_permissive_rate"] = eval_data.get(
                "compile_permissive_rate", 0
            )

        runs.append(summary)

    # Sort by compile_strict_rate descending
    runs.sort(key=lambda r: r.get("compile_strict_rate", 0), reverse=True)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="ShortcutForge training with MLX presets",
    )
    parser.add_argument(
        "--preset",
        required=False,
        choices=list(PRESETS.keys()),
        help="Training preset name",
    )
    parser.add_argument(
        "--run-id",
        required=False,
        help="Unique identifier for this training run",
    )
    parser.add_argument(
        "--method",
        default="lora",
        choices=["lora", "dora", "full"],
        help="Fine-tuning method (default: lora)",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use merged (gold + distilled) training data instead of gold-only",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override default iteration count",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit training examples (smoke testing)",
    )
    parser.add_argument(
        "--eval-after",
        action="store_true",
        help="Run evaluation on frozen eval set after training",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare metrics across all training runs and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.compare:
        runs = compare_runs()
        if not runs:
            print("  No training runs found.", flush=True)
            return

        print(f"\n  {'=' * 70}", flush=True)
        print(f"  TRAINING RUN COMPARISON ({len(runs)} runs)", flush=True)
        print(f"  {'=' * 70}", flush=True)

        for r in runs:
            compile_str = f"{r.get('compile_strict_rate', '?')}%"
            parse_str = f"{r.get('parse_rate', '?')}%"
            print(
                f"  {r['run_id']:<25} "
                f"preset={r['preset']:<12} "
                f"method={r['method']:<5} "
                f"compile_strict={compile_str:<6} "
                f"parse={parse_str}",
                flush=True,
            )

        print(flush=True)
        return

    if not args.preset or not args.run_id:
        parser.error("--preset and --run-id are required unless --compare is set")

    # Train
    result = train(
        preset_name=args.preset,
        run_id=args.run_id,
        method=args.method,
        gold_only=not args.merged,
        iters_override=args.iters,
        max_examples=args.max_examples,
        verbose=args.verbose,
    )

    if result["returncode"] != 0:
        sys.exit(1)

    # Eval after training
    if args.eval_after:
        eval_stats = run_eval_after(
            preset_name=args.preset,
            adapter_path=result["adapter_path"],
            verbose=args.verbose,
        )
        if eval_stats:
            # Save eval results to run directory
            eval_out = Path(result["adapter_path"]) / "eval_results.json"
            with open(eval_out, "w") as f:
                json.dump(eval_stats, f, indent=2)
            print(f"  Eval results saved to: {eval_out}", flush=True)


if __name__ == "__main__":
    main()
