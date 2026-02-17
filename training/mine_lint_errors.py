#!/usr/bin/env python3
"""Sashimi feedback loop: mine eval/distillation lint errors into linter improvements.

Analyzes lint_changes from evaluation and distillation runs to discover:
  1. New hallucination aliases (action names LLMs frequently generate wrong)
  2. Structural patterns (recurring block-level errors)
  3. Condition keyword gaps (invalid IF conditions not yet caught)
  4. Handle access anti-patterns (property access on handles)

Produces:
  - lint_mining_report.json: structured report of discoveries
  - linter_patch_proposal.json: machine-readable patch for dsl_linter.py

The patch proposals are reviewed by a human, not auto-applied.

Usage:
    # Mine from latest eval + distillation
    python mine_lint_errors.py

    # Custom sources
    python mine_lint_errors.py \
      --eval-results training_data/eval_results.json \
      --distillation-log training_data/distillation_log.jsonl \
      --min-frequency 3

    # Output to custom directory
    python mine_lint_errors.py --output-dir training_data/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
_PROJECT_ROOT = _SCRIPT_DIR.parent
_TRAINING_DIR = _PROJECT_ROOT / "training_data"

# Ensure scripts/ is importable
sys.path.insert(0, str(_SRC_DIR))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MiningDiscovery:
    """A single discovery from lint error mining."""
    kind: str                    # "action", "condition", "structure", "handle"
    original: str                # What the model generated
    replacement: str | None      # What the linter fixed it to (if applicable)
    frequency: int               # How many times this appeared
    avg_confidence: float        # Average linter confidence for this repair
    examples: list[str] = field(default_factory=list)  # Sample prompt contexts
    proposed_action: str = ""    # What to do: "add_alias", "add_pattern", etc.


@dataclass
class MiningReport:
    """Complete report from an error mining pass."""
    action_hallucinations: list[MiningDiscovery]
    structural_patterns: list[MiningDiscovery]
    condition_gaps: list[MiningDiscovery]
    handle_patterns: list[MiningDiscovery]
    total_lint_changes_analyzed: int
    total_unique_discoveries: int
    dominant_error_kind: str | None
    error_kind_distribution: dict[str, int]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lint_data(
    eval_results_path: str | None = None,
    distillation_log_path: str | None = None,
) -> list[dict]:
    """Load lint changes from eval results and/or distillation log.

    Each returned dict has: lint_changes, prompt, parsed, shortcut_id.

    Returns:
        List of example dicts with lint_changes.
    """
    examples = []

    # Load from eval results
    if eval_results_path and os.path.exists(eval_results_path):
        with open(eval_results_path) as f:
            data = json.load(f)

        results = data.get("results", [])
        for r in results:
            lint_changes = r.get("lint_changes", [])
            if lint_changes:
                examples.append({
                    "lint_changes": lint_changes,
                    "prompt": r.get("description", ""),
                    "parsed": r.get("parsed", False),
                    "shortcut_id": r.get("shortcut_id", ""),
                    "source": "eval",
                })

    # Load from distillation log
    if distillation_log_path and os.path.exists(distillation_log_path):
        with open(distillation_log_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                lint_changes = entry.get("lint_changes", [])
                if lint_changes:
                    examples.append({
                        "lint_changes": lint_changes,
                        "prompt": entry.get("prompt", ""),
                        "parsed": entry.get("parsed", False),
                        "shortcut_id": entry.get("shortcut_id", ""),
                        "source": "distillation",
                    })

    return examples


def _get_existing_aliases() -> set[str]:
    """Load existing HALLUCINATION_ALIASES keys from dsl_linter.py."""
    try:
        from dsl_linter import HALLUCINATION_ALIASES
        return set(HALLUCINATION_ALIASES.keys())
    except (ImportError, AttributeError):
        # Fallback: try to parse the file directly
        linter_path = _SCRIPT_DIR / "dsl_linter.py"
        if not linter_path.exists():
            return set()

        aliases = set()
        in_dict = False
        with open(linter_path) as f:
            for line in f:
                if "HALLUCINATION_ALIASES" in line and "=" in line:
                    in_dict = True
                    continue
                if in_dict:
                    if line.strip().startswith("}"):
                        break
                    # Extract key from "key": "value" pattern
                    stripped = line.strip().strip(",")
                    if stripped.startswith('"') or stripped.startswith("'"):
                        key = stripped.split(":")[0].strip().strip('"').strip("'")
                        aliases.add(key.lower())
        return aliases


# ---------------------------------------------------------------------------
# Mining functions
# ---------------------------------------------------------------------------

def mine_action_hallucinations(
    examples: list[dict],
    min_frequency: int = 3,
) -> list[MiningDiscovery]:
    """Find frequent action hallucinations not yet in HALLUCINATION_ALIASES.

    Looks for kind="action" lint changes where the original action name
    appears frequently across examples.
    """
    existing_aliases = _get_existing_aliases()

    # Count original → replacement pairs
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_confidences: dict[tuple[str, str], list[float]] = defaultdict(list)
    pair_prompts: dict[tuple[str, str], list[str]] = defaultdict(list)

    for example in examples:
        for change in example.get("lint_changes", []):
            if change.get("kind") != "action":
                continue

            original = change.get("original", "").lower().strip()
            replacement = change.get("replacement", "").lower().strip()

            if not original or not replacement:
                continue

            # Skip if already known
            if original in existing_aliases:
                continue

            pair = (original, replacement)
            pair_counts[pair] += 1
            pair_confidences[pair].append(change.get("confidence", 1.0))
            if example.get("prompt") and len(pair_prompts[pair]) < 3:
                pair_prompts[pair].append(example["prompt"][:100])

    discoveries = []
    for (original, replacement), count in pair_counts.most_common():
        if count < min_frequency:
            continue

        confs = pair_confidences[(original, replacement)]
        avg_conf = sum(confs) / len(confs) if confs else 1.0

        discoveries.append(MiningDiscovery(
            kind="action",
            original=original,
            replacement=replacement,
            frequency=count,
            avg_confidence=round(avg_conf, 3),
            examples=pair_prompts[(original, replacement)],
            proposed_action=f'add_alias: "{original}": "{replacement}"',
        ))

    return discoveries


def mine_structural_patterns(
    examples: list[dict],
    min_frequency: int = 2,
) -> list[MiningDiscovery]:
    """Cluster structural errors by pattern and report frequency."""
    pattern_counts: Counter[str] = Counter()
    pattern_prompts: dict[str, list[str]] = defaultdict(list)
    pattern_confs: dict[str, list[float]] = defaultdict(list)

    for example in examples:
        for change in example.get("lint_changes", []):
            if change.get("kind") != "structure":
                continue

            reason = change.get("reason", "unknown_structure_error")
            # Normalize reason to pattern key
            pattern_key = _normalize_structural_reason(reason)

            pattern_counts[pattern_key] += 1
            pattern_confs[pattern_key].append(change.get("confidence", 1.0))
            if example.get("prompt") and len(pattern_prompts[pattern_key]) < 3:
                pattern_prompts[pattern_key].append(example["prompt"][:100])

    discoveries = []
    for pattern, count in pattern_counts.most_common():
        if count < min_frequency:
            continue

        confs = pattern_confs[pattern]
        avg_conf = sum(confs) / len(confs) if confs else 1.0

        discoveries.append(MiningDiscovery(
            kind="structure",
            original=pattern,
            replacement=None,
            frequency=count,
            avg_confidence=round(avg_conf, 3),
            examples=pattern_prompts[pattern],
            proposed_action=f"investigate_pattern: {pattern}",
        ))

    return discoveries


def _normalize_structural_reason(reason: str) -> str:
    """Normalize a structural lint reason to a pattern key."""
    reason_lower = reason.lower()

    if "orphan" in reason_lower and "else" in reason_lower:
        return "orphan_else"
    if "unclosed" in reason_lower:
        if "if" in reason_lower:
            return "unclosed_if"
        if "repeat" in reason_lower:
            return "unclosed_repeat"
        if "menu" in reason_lower:
            return "unclosed_menu"
        return "unclosed_block"
    if "truncat" in reason_lower:
        return "truncated_output"
    if "endshortcut" in reason_lower:
        return "missing_endshortcut"
    if "action" in reason_lower and "keyword" in reason_lower:
        return "action_as_keyword"
    if "incomplete" in reason_lower:
        return "incomplete_action"

    # Keep as-is if no pattern matched
    return reason_lower[:50]


def mine_condition_gaps(
    examples: list[dict],
    min_frequency: int = 2,
) -> list[MiningDiscovery]:
    """Extract invalid IF condition keywords that appear frequently."""
    cond_counts: Counter[str] = Counter()
    cond_replacements: dict[str, Counter[str]] = defaultdict(Counter)
    cond_prompts: dict[str, list[str]] = defaultdict(list)

    for example in examples:
        for change in example.get("lint_changes", []):
            if change.get("kind") != "condition":
                continue

            original = change.get("original", "").strip()
            replacement = change.get("replacement", "").strip()

            if not original:
                continue

            cond_counts[original] += 1
            if replacement:
                cond_replacements[original][replacement] += 1
            if example.get("prompt") and len(cond_prompts[original]) < 3:
                cond_prompts[original].append(example["prompt"][:100])

    discoveries = []
    for original, count in cond_counts.most_common():
        if count < min_frequency:
            continue

        best_replacement = None
        if cond_replacements[original]:
            best_replacement = cond_replacements[original].most_common(1)[0][0]

        discoveries.append(MiningDiscovery(
            kind="condition",
            original=original,
            replacement=best_replacement,
            frequency=count,
            avg_confidence=0.9,
            examples=cond_prompts[original],
            proposed_action=f'add_condition: "{original}" → "{best_replacement}"',
        ))

    return discoveries


def mine_handle_patterns(
    examples: list[dict],
    min_frequency: int = 2,
) -> list[MiningDiscovery]:
    """Extract handle property access anti-patterns."""
    handle_counts: Counter[str] = Counter()
    handle_replacements: dict[str, str] = {}
    handle_prompts: dict[str, list[str]] = defaultdict(list)

    for example in examples:
        for change in example.get("lint_changes", []):
            if change.get("kind") != "handle":
                continue

            original = change.get("original", "").strip()
            replacement = change.get("replacement", "").strip()

            if not original:
                continue

            handle_counts[original] += 1
            if replacement:
                handle_replacements[original] = replacement
            if example.get("prompt") and len(handle_prompts[original]) < 3:
                handle_prompts[original].append(example["prompt"][:100])

    discoveries = []
    for original, count in handle_counts.most_common():
        if count < min_frequency:
            continue

        discoveries.append(MiningDiscovery(
            kind="handle",
            original=original,
            replacement=handle_replacements.get(original),
            frequency=count,
            avg_confidence=0.95,
            examples=handle_prompts[original],
            proposed_action=f'add_handle_rule: "{original}" → "{handle_replacements.get(original, "?")}"',
        ))

    return discoveries


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_mining_report(
    examples: list[dict],
    min_frequency: int = 3,
) -> MiningReport:
    """Run all mining passes and produce a structured report."""
    # Compute error kind distribution
    kind_dist: Counter[str] = Counter()
    total_changes = 0

    for example in examples:
        for change in example.get("lint_changes", []):
            kind = change.get("kind", "unknown")
            kind_dist[kind] += 1
            total_changes += 1

    dominant = kind_dist.most_common(1)[0][0] if kind_dist else None

    # Run mining passes
    action_discoveries = mine_action_hallucinations(examples, min_frequency)
    structural_discoveries = mine_structural_patterns(examples, min_frequency)
    condition_discoveries = mine_condition_gaps(examples, min_frequency)
    handle_discoveries = mine_handle_patterns(examples, min_frequency)

    total_unique = (
        len(action_discoveries) +
        len(structural_discoveries) +
        len(condition_discoveries) +
        len(handle_discoveries)
    )

    return MiningReport(
        action_hallucinations=action_discoveries,
        structural_patterns=structural_discoveries,
        condition_gaps=condition_discoveries,
        handle_patterns=handle_discoveries,
        total_lint_changes_analyzed=total_changes,
        total_unique_discoveries=total_unique,
        dominant_error_kind=dominant,
        error_kind_distribution=dict(kind_dist.most_common()),
    )


def generate_linter_patch(report: MiningReport) -> dict:
    """Produce a machine-readable patch file for dsl_linter.py.

    Returns a dict with proposed additions grouped by category.
    """
    patch: dict[str, Any] = {
        "metadata": {
            "total_lint_changes_analyzed": report.total_lint_changes_analyzed,
            "total_discoveries": report.total_unique_discoveries,
            "dominant_error_kind": report.dominant_error_kind,
        },
        "hallucination_aliases": {},
        "structural_patterns": [],
        "condition_keywords": {},
        "handle_rules": {},
    }

    for d in report.action_hallucinations:
        if d.replacement:
            patch["hallucination_aliases"][d.original] = {
                "replacement": d.replacement,
                "frequency": d.frequency,
                "avg_confidence": d.avg_confidence,
                "suggested_confidence": min(0.95, d.avg_confidence),
            }

    for d in report.structural_patterns:
        patch["structural_patterns"].append({
            "pattern": d.original,
            "frequency": d.frequency,
            "avg_confidence": d.avg_confidence,
            "action": d.proposed_action,
        })

    for d in report.condition_gaps:
        if d.replacement:
            patch["condition_keywords"][d.original] = {
                "replacement": d.replacement,
                "frequency": d.frequency,
            }

    for d in report.handle_patterns:
        if d.replacement:
            patch["handle_rules"][d.original] = {
                "replacement": d.replacement,
                "frequency": d.frequency,
            }

    return patch


def print_report(report: MiningReport) -> None:
    """Print human-readable summary."""
    print(f"\n  {'='*50}", flush=True)
    print(f"  SASHIMI MODE: Lint Error Mining Report", flush=True)
    print(f"  {'='*50}", flush=True)
    print(f"  Total lint changes analyzed: {report.total_lint_changes_analyzed}", flush=True)
    print(f"  Dominant error kind: {report.dominant_error_kind}", flush=True)
    print(f"  Error distribution: {report.error_kind_distribution}", flush=True)
    print(f"  Unique discoveries: {report.total_unique_discoveries}", flush=True)

    if report.action_hallucinations:
        print(f"\n  NEW HALLUCINATION ALIASES ({len(report.action_hallucinations)}):", flush=True)
        for d in report.action_hallucinations[:10]:
            print(f"    \"{d.original}\" → \"{d.replacement}\" (×{d.frequency}, conf={d.avg_confidence})", flush=True)

    if report.structural_patterns:
        print(f"\n  STRUCTURAL PATTERNS ({len(report.structural_patterns)}):", flush=True)
        for d in report.structural_patterns[:10]:
            print(f"    {d.original} (×{d.frequency})", flush=True)

    if report.condition_gaps:
        print(f"\n  CONDITION GAPS ({len(report.condition_gaps)}):", flush=True)
        for d in report.condition_gaps[:10]:
            print(f"    \"{d.original}\" → \"{d.replacement}\" (×{d.frequency})", flush=True)

    if report.handle_patterns:
        print(f"\n  HANDLE PATTERNS ({len(report.handle_patterns)}):", flush=True)
        for d in report.handle_patterns[:10]:
            print(f"    \"{d.original}\" → \"{d.replacement}\" (×{d.frequency})", flush=True)

    if report.total_unique_discoveries == 0:
        print(f"\n  No new discoveries (linter already covers observed errors).", flush=True)

    print(flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sashimi feedback: mine lint errors into linter improvements",
    )
    parser.add_argument(
        "--eval-results",
        default=str(_TRAINING_DIR / "eval_results.json"),
        help="Path to eval results JSON",
    )
    parser.add_argument(
        "--distillation-log",
        default=str(_TRAINING_DIR / "distillation_log.jsonl"),
        help="Path to distillation log JSONL",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=3,
        help="Minimum frequency to report a discovery (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_TRAINING_DIR),
        help="Directory for output files (default: training_data/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print(f"\n  ShortcutForge: Sashimi Mode — Lint Error Mining\n", flush=True)

    # Load data
    examples = load_lint_data(
        eval_results_path=args.eval_results,
        distillation_log_path=args.distillation_log,
    )

    if not examples:
        print("  No lint data found. Run eval with --log-distillation first.", flush=True)
        sys.exit(0)

    print(f"  Loaded {len(examples)} examples with lint changes", flush=True)

    # Generate report
    report = generate_mining_report(examples, min_frequency=args.min_frequency)

    # Print summary
    print_report(report)

    # Write structured report
    report_path = os.path.join(args.output_dir, "lint_mining_report.json")
    report_dict = {
        "action_hallucinations": [asdict(d) for d in report.action_hallucinations],
        "structural_patterns": [asdict(d) for d in report.structural_patterns],
        "condition_gaps": [asdict(d) for d in report.condition_gaps],
        "handle_patterns": [asdict(d) for d in report.handle_patterns],
        "total_lint_changes_analyzed": report.total_lint_changes_analyzed,
        "total_unique_discoveries": report.total_unique_discoveries,
        "dominant_error_kind": report.dominant_error_kind,
        "error_kind_distribution": report.error_kind_distribution,
    }
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"  Report → {report_path}", flush=True)

    # Write linter patch proposal
    patch = generate_linter_patch(report)
    patch_path = os.path.join(args.output_dir, "linter_patch_proposal.json")
    with open(patch_path, "w") as f:
        json.dump(patch, f, indent=2)
    print(f"  Patch  → {patch_path}", flush=True)

    print(flush=True)


if __name__ == "__main__":
    main()
