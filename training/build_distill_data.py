#!/usr/bin/env python3
"""Linter-guided distillation data flywheel for ShortcutForge.

Orchestrates the Sashimi Mode distillation pipeline:
  1. Generate distillation data from teacher model (batched)
  2. Curate via DistillationCurator
  3. Convert to chat format
  4. Merge gold + distilled data with lint-guided shaping

The linter's repair records (LintChange) are first-class inputs:
each distillation example carries a lint profile that shapes training
data selection and weighting.

Usage:
    # Full pipeline
    python build_distill_data.py --batch-size 500

    # Curate-only (from existing distillation log)
    python build_distill_data.py --curate-only

    # Merge-only (gold + distilled)
    python build_distill_data.py --merge-only

    # Skip lint-based weighting
    python build_distill_data.py --no-shaping
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
_PROJECT_ROOT = _SCRIPT_DIR.parent
_TRAINING_DIR = _PROJECT_ROOT / "training_data"

# Ensure scripts/ is importable
sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Lint Profile (Sashimi layer)
# ---------------------------------------------------------------------------


@dataclass
class LintProfile:
    """Repair fingerprint for a single distillation example.

    Computed from the lint_changes list attached to each example.
    Used to shape training data: weight/oversample based on repair patterns.
    """

    total_repairs: int
    repairs_by_kind: dict[str, int]
    avg_confidence: float
    dominant_kind: str | None
    is_clean: bool  # True if total_repairs == 0
    is_hard_negative: bool  # True if parsed=True AND total_repairs > 0


def compute_lint_profile(lint_changes: list[dict], parsed: bool = True) -> LintProfile:
    """Compute lint profile from a list of lint change dicts.

    Args:
        lint_changes: List of lint change dicts with 'kind', 'confidence' keys.
        parsed: Whether the example parsed successfully after linting.

    Returns:
        LintProfile with repair fingerprint.
    """
    total = len(lint_changes)

    if total == 0:
        return LintProfile(
            total_repairs=0,
            repairs_by_kind={},
            avg_confidence=1.0,
            dominant_kind=None,
            is_clean=True,
            is_hard_negative=False,
        )

    kind_counts: Counter[str] = Counter()
    confidences: list[float] = []

    for change in lint_changes:
        kind = change.get("kind", "unknown")
        kind_counts[kind] += 1
        confidences.append(change.get("confidence", 1.0))

    dominant = kind_counts.most_common(1)[0][0] if kind_counts else None
    avg_conf = sum(confidences) / len(confidences) if confidences else 1.0

    return LintProfile(
        total_repairs=total,
        repairs_by_kind=dict(kind_counts),
        avg_confidence=round(avg_conf, 3),
        dominant_kind=dominant,
        is_clean=False,
        is_hard_negative=parsed and total > 0,
    )


def compute_shaping_weight(profile: LintProfile) -> float:
    """Compute training weight from lint profile.

    Shaping rules:
      - Clean examples (0 repairs): 1.0x (positive anchors)
      - Hard negatives (parsed + repairs): 2.0x
      - Action-dominated: 1.5x
      - Structure-dominated: 1.5x
      - Low-confidence repairs (avg < 0.75): 0.5x
    """
    if profile.is_clean:
        return 1.0

    if profile.avg_confidence < 0.75:
        return 0.5

    if profile.is_hard_negative:
        weight = 2.0
    else:
        weight = 1.0

    if profile.dominant_kind == "action":
        weight = max(weight, 1.5)
    elif profile.dominant_kind == "structure":
        weight = max(weight, 1.5)

    return weight


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def generate_distillation_batched(
    model_path: str,
    adapter_path: str | None,
    eval_file: str,
    output_path: str,
    batch_size: int = 500,
    chat_template: str = "llama3",
    timeout_s: float = 60,
    verbose: bool = False,
) -> int:
    """Generate distillation data by running evaluate_model in batched chunks.

    Appends to output_path so batches accumulate.

    Args:
        model_path: Base model path.
        adapter_path: LoRA adapter directory.
        eval_file: Training JSONL to distill from.
        output_path: Where to write the distillation log.
        batch_size: Examples per batch.
        chat_template: Chat template format.
        timeout_s: Timeout per generation.
        verbose: Print per-batch progress.

    Returns:
        Total examples processed.
    """
    # Count total examples
    with open(eval_file) as f:
        total = sum(1 for _ in f)

    processed = 0
    batch_num = 0

    while processed < total:
        batch_num += 1
        remaining = total - processed
        current_batch = min(batch_size, remaining)

        if verbose:
            print(
                f"\n  Batch {batch_num}: examples {processed + 1}-{processed + current_batch} of {total}",
                flush=True,
            )

        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / "evaluate_model.py"),
            "--model-path",
            model_path,
            "--eval-file",
            eval_file,
            "--skip-examples",
            str(processed),
            "--max-examples",
            str(current_batch),
            "--log-distillation",
            "--distillation-output",
            output_path,
            "--append-distillation",
            "--chat-template",
            chat_template,
            "--timeout",
            str(timeout_s),
        ]
        if adapter_path:
            cmd.extend(["--adapter-path", adapter_path])

        t0 = time.monotonic()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=int(timeout_s * current_batch * 2),  # generous timeout
        )

        elapsed = time.monotonic() - t0
        if result.returncode != 0:
            print(
                f"  WARNING: Batch {batch_num} failed (exit {result.returncode})",
                flush=True,
            )
            if verbose:
                print(f"  stderr: {result.stderr[:500]}", flush=True)
        else:
            if verbose:
                print(f"  Batch {batch_num} complete in {elapsed:.1f}s", flush=True)

        processed += current_batch

    return processed


def curate_distillation(
    input_path: str,
    output_path: str,
    creativity_threshold: float = 0.3,
    similarity_threshold: float = 0.85,
    verbose: bool = False,
) -> dict:
    """Curate distillation data using DistillationCurator.

    Args:
        input_path: Path to raw distillation log JSONL.
        output_path: Where to write curated JSONL.
        creativity_threshold: Min creativity score.
        similarity_threshold: Dedup similarity threshold.
        verbose: Print curation stats.

    Returns:
        Curation statistics dict.
    """
    from distillation_curator import DistillationCurator

    curator = DistillationCurator(
        creativity_threshold=creativity_threshold,
        similarity_threshold=similarity_threshold,
    )

    stats = curator.curate_file(input_path, output_path)

    if verbose:
        print("\n  Curation stats:", flush=True)
        print(f"    Input entries:  {stats.input_count}", flush=True)
        print(f"    Passed gates:   {stats.quality_passed}", flush=True)
        after_dedup = stats.quality_passed - stats.dedup_removed
        print(f"    After dedup:    {after_dedup}", flush=True)
        print(f"    Final output:   {stats.output_count}", flush=True)

    return {
        "total_input": stats.input_count,
        "passed_gates": stats.quality_passed,
        "after_dedup": stats.quality_passed - stats.dedup_removed,
        "final_count": stats.output_count,
    }


def convert_to_chat_format(
    curated_path: str,
    output_path: str,
    chat_template: str = "llama3",
    verbose: bool = False,
) -> int:
    """Convert curated distillation entries to chat training format.

    Input: {prompt, completion, lint_changes, ...} per line
    Output: {messages: [{role: system, content: ...}, {role: user, ...}, {role: assistant, ...}]}

    Returns:
        Number of examples written.
    """
    from generate_prompt import build_system_prompt

    system_prompt = build_system_prompt()
    written = 0

    with open(curated_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            prompt = entry.get("prompt", "")
            # Use canonicalized output (linter-repaired) as the target
            completion = entry.get("canonicalized_output") or entry.get(
                "completion", ""
            )

            if not prompt or not completion:
                continue

            chat_entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            }

            # Preserve lint metadata for downstream analysis
            if "lint_changes" in entry:
                chat_entry["_lint_changes"] = entry["lint_changes"]
            if "shortcut_id" in entry:
                chat_entry["shortcut_id"] = entry["shortcut_id"]

            fout.write(json.dumps(chat_entry) + "\n")
            written += 1

    if verbose:
        print(
            f"  Converted {written} entries to chat format → {output_path}", flush=True
        )

    return written


def merge_gold_and_distilled(
    gold_path: str,
    distilled_path: str,
    output_path: str,
    max_per_shortcut: int = 2,
    apply_shaping: bool = True,
    verbose: bool = False,
) -> dict:
    """Merge gold training data with distilled data, applying lint-guided shaping.

    PLAN.md merge policy:
      - max 2 rows per shortcut (gold + 1 distilled)
      - keep hard-negatives where raw failed but canonicalized passed
      - cap near-duplicate prompts by similarity threshold

    Args:
        gold_path: Path to gold training JSONL.
        distilled_path: Path to distilled chat JSONL.
        output_path: Where to write merged JSONL.
        max_per_shortcut: Max examples per shortcut ID.
        apply_shaping: Whether to apply lint-based weighting.
        verbose: Print merge stats.

    Returns:
        Merge statistics dict.
    """
    # Load gold data
    gold_entries = []
    with open(gold_path) as f:
        for line in f:
            entry = json.loads(line)
            entry["_source"] = "gold"
            gold_entries.append(entry)

    # Load distilled data
    distilled_entries = []
    if os.path.exists(distilled_path):
        with open(distilled_path) as f:
            for line in f:
                entry = json.loads(line)
                entry["_source"] = "distilled"
                distilled_entries.append(entry)

    # Group by shortcut_id
    by_shortcut: dict[str, list[dict]] = {}
    for entry in gold_entries:
        sid = entry.get("shortcut_id", f"gold_{len(by_shortcut)}")
        by_shortcut.setdefault(sid, []).append(entry)

    for entry in distilled_entries:
        sid = entry.get("shortcut_id", f"distilled_{len(by_shortcut)}")
        by_shortcut.setdefault(sid, []).append(entry)

    # Apply merge policy
    merged = []
    lint_profiles = []
    skipped = 0

    for sid, entries in by_shortcut.items():
        # Gold entries first, then distilled
        gold = [e for e in entries if e.get("_source") == "gold"]
        distilled = [e for e in entries if e.get("_source") == "distilled"]

        # Keep all gold (up to max_per_shortcut)
        kept = gold[:max_per_shortcut]
        remaining_slots = max_per_shortcut - len(kept)

        # Fill remaining slots with distilled
        if remaining_slots > 0:
            kept.extend(distilled[:remaining_slots])

        skipped += (len(gold) + len(distilled)) - len(kept)

        for entry in kept:
            # Compute lint profile
            lint_changes = entry.get("_lint_changes", [])
            parsed = True  # Assume parsed if it made it to training data
            profile = compute_lint_profile(lint_changes, parsed)
            weight = compute_shaping_weight(profile) if apply_shaping else 1.0

            entry["_lint_profile"] = asdict(profile)
            entry["_shaping_weight"] = weight
            merged.append(entry)
            lint_profiles.append(profile)

    # Write output
    with open(output_path, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    # Compute aggregate lint analysis
    lint_analysis = _compute_lint_analysis(lint_profiles)

    stats = {
        "gold_count": len(gold_entries),
        "distilled_count": len(distilled_entries),
        "merged_count": len(merged),
        "skipped_by_policy": skipped,
        "unique_shortcuts": len(by_shortcut),
        "lint_analysis": lint_analysis,
    }

    if verbose:
        print("\n  Merge stats:", flush=True)
        print(f"    Gold entries:       {stats['gold_count']}", flush=True)
        print(f"    Distilled entries:  {stats['distilled_count']}", flush=True)
        print(f"    Merged output:      {stats['merged_count']}", flush=True)
        print(f"    Skipped (policy):   {stats['skipped_by_policy']}", flush=True)
        print(f"    Unique shortcuts:   {stats['unique_shortcuts']}", flush=True)
        print("\n  Lint profile analysis:", flush=True)
        print(f"    Clean examples:     {lint_analysis['clean_count']}", flush=True)
        print(
            f"    Hard negatives:     {lint_analysis['hard_negative_count']}",
            flush=True,
        )
        print(
            f"    By dominant kind:   {lint_analysis['dominant_kind_dist']}", flush=True
        )
        print(
            f"    Avg repairs/example: {lint_analysis['avg_repairs']:.2f}", flush=True
        )

    # Write lint analysis
    analysis_path = os.path.join(os.path.dirname(output_path), "lint_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(lint_analysis, f, indent=2)

    if verbose:
        print(f"  Lint analysis → {analysis_path}", flush=True)

    return stats


def _compute_lint_analysis(profiles: list[LintProfile]) -> dict:
    """Compute aggregate lint statistics from a list of profiles."""
    if not profiles:
        return {
            "total_examples": 0,
            "clean_count": 0,
            "hard_negative_count": 0,
            "avg_repairs": 0.0,
            "repairs_by_kind": {},
            "dominant_kind_dist": {},
            "avg_confidence": 0.0,
            "weight_distribution": {},
        }

    total = len(profiles)
    clean = sum(1 for p in profiles if p.is_clean)
    hard_neg = sum(1 for p in profiles if p.is_hard_negative)
    total_repairs = sum(p.total_repairs for p in profiles)

    # Aggregate repairs by kind
    kind_totals: Counter[str] = Counter()
    for p in profiles:
        for kind, count in p.repairs_by_kind.items():
            kind_totals[kind] += count

    # Dominant kind distribution
    dominant_dist: Counter[str] = Counter()
    for p in profiles:
        if p.dominant_kind:
            dominant_dist[p.dominant_kind] += 1

    # Average confidence (only for non-clean)
    non_clean = [p for p in profiles if not p.is_clean]
    avg_conf = (
        sum(p.avg_confidence for p in non_clean) / len(non_clean) if non_clean else 1.0
    )

    # Weight distribution
    weight_dist: Counter[str] = Counter()
    for p in profiles:
        w = compute_shaping_weight(p)
        bucket = f"{w:.1f}x"
        weight_dist[bucket] += 1

    return {
        "total_examples": total,
        "clean_count": clean,
        "hard_negative_count": hard_neg,
        "avg_repairs": round(total_repairs / total, 2) if total else 0.0,
        "repairs_by_kind": dict(kind_totals.most_common()),
        "dominant_kind_dist": dict(dominant_dist.most_common()),
        "avg_confidence": round(avg_conf, 3),
        "weight_distribution": dict(weight_dist),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Linter-guided distillation data flywheel (Sashimi Mode)",
    )
    parser.add_argument(
        "--model-path",
        default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        help="Base model path for distillation teacher",
    )
    parser.add_argument(
        "--adapter-path",
        default="models/baseline-v1-mlx",
        help="LoRA adapter for teacher model",
    )
    parser.add_argument(
        "--train-file",
        default=str(_TRAINING_DIR / "shortcutdsl_train_expanded.jsonl"),
        help="Training data to distill from",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Examples per batch during distillation generation",
    )
    parser.add_argument(
        "--chat-template",
        default="llama3",
        choices=["llama3", "chatml"],
        help="Chat template for teacher model",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60,
        help="Timeout per generation in seconds",
    )
    parser.add_argument(
        "--max-per-shortcut",
        type=int,
        default=2,
        help="Max examples per shortcut ID in merged output",
    )
    parser.add_argument(
        "--curate-only",
        action="store_true",
        help="Only run curation on existing distillation log",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only run merge of gold + distilled data",
    )
    parser.add_argument(
        "--no-shaping",
        action="store_true",
        help="Skip lint-based weighting in merge",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Resolve paths
    distill_log_full = _TRAINING_DIR / "distillation_log_full.jsonl"
    distill_log_default = _TRAINING_DIR / "distillation_log.jsonl"
    distill_log = str(distill_log_full)
    curated_path = str(_TRAINING_DIR / "distilled_curated.jsonl")
    chat_path = str(_TRAINING_DIR / "distilled_chat.jsonl")
    merged_path = str(_TRAINING_DIR / "merged_train.jsonl")
    gold_path = str(_TRAINING_DIR / "shortcutdsl_train_expanded.jsonl")

    print("\n  ShortcutForge: Sashimi Mode — Distillation Data Flywheel\n", flush=True)

    if args.merge_only:
        # Skip generation and curation, just merge
        print("  [Step 3] Merging gold + distilled data...", flush=True)
        merge_stats = merge_gold_and_distilled(
            gold_path=gold_path,
            distilled_path=chat_path,
            output_path=merged_path,
            max_per_shortcut=args.max_per_shortcut,
            apply_shaping=not args.no_shaping,
            verbose=args.verbose,
        )
        print(
            f"\n  Done. Merged {merge_stats['merged_count']} examples → {merged_path}",
            flush=True,
        )
        return

    if args.curate_only:
        # Skip generation, just curate + convert + merge
        if not distill_log_full.exists() and distill_log_default.exists():
            distill_log = str(distill_log_default)
        if not os.path.exists(distill_log):
            print(
                f"  ERROR: Distillation log not found. Tried: "
                f"{distill_log_full} and {distill_log_default}",
                flush=True,
            )
            sys.exit(1)

        print("  [Step 2] Curating distillation data...", flush=True)
        curate_stats = curate_distillation(
            input_path=distill_log,
            output_path=curated_path,
            verbose=args.verbose,
        )

        print("  [Step 2b] Converting to chat format...", flush=True)
        n_converted = convert_to_chat_format(
            curated_path=curated_path,
            output_path=chat_path,
            chat_template=args.chat_template,
            verbose=args.verbose,
        )

        print("  [Step 3] Merging gold + distilled data...", flush=True)
        merge_stats = merge_gold_and_distilled(
            gold_path=gold_path,
            distilled_path=chat_path,
            output_path=merged_path,
            max_per_shortcut=args.max_per_shortcut,
            apply_shaping=not args.no_shaping,
            verbose=args.verbose,
        )

        print(
            f"\n  Done. {curate_stats['final_count']} curated → {n_converted} chat → {merge_stats['merged_count']} merged",
            flush=True,
        )
        return

    # Full pipeline
    print(
        f"  [Step 1] Generating distillation data (batch_size={args.batch_size})...",
        flush=True,
    )
    total = generate_distillation_batched(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        eval_file=args.train_file,
        output_path=distill_log,
        batch_size=args.batch_size,
        chat_template=args.chat_template,
        timeout_s=args.timeout,
        verbose=args.verbose,
    )
    print(f"  Generated distillation log: {total} examples → {distill_log}", flush=True)

    print("\n  [Step 2] Curating distillation data...", flush=True)
    curate_stats = curate_distillation(
        input_path=distill_log,
        output_path=curated_path,
        verbose=args.verbose,
    )

    print("  [Step 2b] Converting to chat format...", flush=True)
    n_converted = convert_to_chat_format(
        curated_path=curated_path,
        output_path=chat_path,
        chat_template=args.chat_template,
        verbose=args.verbose,
    )

    print("\n  [Step 3] Merging gold + distilled data...", flush=True)
    merge_stats = merge_gold_and_distilled(
        gold_path=gold_path,
        distilled_path=chat_path,
        output_path=merged_path,
        max_per_shortcut=args.max_per_shortcut,
        apply_shaping=not args.no_shaping,
        verbose=args.verbose,
    )

    print(f"\n  ={'=' * 50}", flush=True)
    print("  SASHIMI MODE: Distillation Complete", flush=True)
    print(f"  ={'=' * 50}", flush=True)
    print(f"  Teacher:      {args.model_path}", flush=True)
    print(
        f"  Distilled:    {total} → {curate_stats['final_count']} curated → {n_converted} chat",
        flush=True,
    )
    print(
        f"  Merged:       {merge_stats['merged_count']} total ({merge_stats['gold_count']} gold + distilled)",
        flush=True,
    )
    print(f"  Output:       {merged_path}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
