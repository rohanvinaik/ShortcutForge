"""
Training Data Labeler for ShortcutForge.

Labels each training example with:
  - domain: best-matching domain profile (via DomainProfileManager)
  - complexity: token budget complexity tier (via estimate_budget)
  - action_count: number of ACTION statements in the DSL
  - construct_types: set of structural constructs used (IF, MENU, REPEAT, FOREACH)
  - word_count: prompt word count
  - dsl_char_len: character length of DSL output

Produces labeled_train.jsonl with original data + labels for stratified evaluation.

Usage:
    python scripts/label_training_data.py
    python scripts/label_training_data.py --input training_data/shortcutdsl_train.jsonl --output training_data/labeled_train.jsonl
    python scripts/label_training_data.py --stats  # Print label distribution only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from domain_profile import DomainProfileManager
from token_budget import estimate_budget

# ── Defaults ──────────────────────────────────────────────────────────

_PROJECT_ROOT = _SCRIPT_DIR.parent
_DEFAULT_INPUT = _PROJECT_ROOT / "training_data" / "shortcutdsl_train.jsonl"
_DEFAULT_OUTPUT = _PROJECT_ROOT / "training_data" / "labeled_train.jsonl"


# ── Construct Detection ──────────────────────────────────────────────

_CONSTRUCT_PATTERNS = {
    "IF":      re.compile(r"^IF\b", re.MULTILINE),
    "MENU":    re.compile(r"^MENU\b", re.MULTILINE),
    "REPEAT":  re.compile(r"^REPEAT\b", re.MULTILINE),
    "FOREACH": re.compile(r"^FOREACH\b", re.MULTILINE),
}

_ACTION_PATTERN = re.compile(r"^ACTION\b", re.MULTILINE)


def _detect_constructs(dsl: str) -> list[str]:
    """Detect which structural constructs are used in a DSL string."""
    return [name for name, pat in _CONSTRUCT_PATTERNS.items() if pat.search(dsl)]


def _count_actions(dsl: str) -> int:
    """Count the number of ACTION statements in a DSL string."""
    return len(_ACTION_PATTERN.findall(dsl))


# ── Labeler ──────────────────────────────────────────────────────────

class TrainingDataLabeler:
    """Labels training examples with domain, complexity, and structural metadata."""

    def __init__(self, profiles_dir: Path | None = None):
        self._profile_mgr = DomainProfileManager(profiles_dir)

    def label_example(self, example: dict) -> dict:
        """Label a single training example.

        Args:
            example: A training example dict with 'messages' key

        Returns:
            A new dict with original data plus 'labels' key
        """
        messages = example.get("messages", [])
        if len(messages) < 3:
            return {**example, "labels": self._empty_labels()}

        prompt = messages[1].get("content", "")
        dsl = messages[2].get("content", "")

        # Domain profile selection
        profile = self._profile_mgr.select_profile(prompt)

        # Token budget complexity estimation
        budget = estimate_budget(prompt)

        # Structural analysis of DSL
        action_count = _count_actions(dsl)
        construct_types = _detect_constructs(dsl)

        labels = {
            "domain": profile.profile_id,
            "complexity": budget.complexity,
            "action_count": action_count,
            "construct_types": construct_types,
            "word_count": budget.word_count,
            "dsl_char_len": len(dsl),
        }

        return {**example, "labels": labels}

    @staticmethod
    def _empty_labels() -> dict:
        """Return empty labels for malformed examples."""
        return {
            "domain": "unknown",
            "complexity": "unknown",
            "action_count": 0,
            "construct_types": [],
            "word_count": 0,
            "dsl_char_len": 0,
        }

    def label_file(
        self,
        input_path: Path,
        output_path: Path | None = None,
    ) -> list[dict]:
        """Label all examples in a JSONL file.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to write labeled JSONL (optional)

        Returns:
            List of labeled examples
        """
        labeled = []

        with open(input_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    example = json.loads(line)
                    labeled.append(self.label_example(example))
                except json.JSONDecodeError:
                    print(f"Warning: skipping malformed JSON on line {line_num}")
                    continue

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for item in labeled:
                    f.write(json.dumps(item) + "\n")

        return labeled


# ── Statistics ───────────────────────────────────────────────────────

def print_stats(labeled: list[dict]) -> None:
    """Print label distribution statistics."""
    total = len(labeled)
    print(f"\nTraining Data Labels ({total} examples)")
    print("=" * 60)

    # Domain distribution
    domain_counts = Counter(ex["labels"]["domain"] for ex in labeled)
    print(f"\nDomain Distribution:")
    for domain, count in domain_counts.most_common():
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {domain:20s} {count:5d} ({pct:5.1f}%) {bar}")

    # Complexity distribution
    complexity_counts = Counter(ex["labels"]["complexity"] for ex in labeled)
    print(f"\nComplexity Distribution:")
    for comp, count in sorted(complexity_counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {comp:20s} {count:5d} ({pct:5.1f}%) {bar}")

    # Construct usage
    construct_counts: Counter[str] = Counter()
    for ex in labeled:
        for ct in ex["labels"]["construct_types"]:
            construct_counts[ct] += 1
    print(f"\nConstruct Usage:")
    for construct, count in construct_counts.most_common():
        pct = count / total * 100
        print(f"  {construct:20s} {count:5d} ({pct:5.1f}%)")

    # Action count stats
    action_counts = sorted(ex["labels"]["action_count"] for ex in labeled)
    n = len(action_counts)
    print(f"\nAction Count Stats:")
    print(f"  Median: {action_counts[n // 2]}")
    print(f"  P90:    {action_counts[int(n * 0.9)]}")
    print(f"  P95:    {action_counts[int(n * 0.95)]}")
    print(f"  Max:    {action_counts[-1]}")

    # Domain x Complexity cross-tab
    print(f"\nDomain x Complexity:")
    print(f"  {'':20s} {'simple':>8s} {'medium':>8s} {'complex':>8s} {'v_complex':>8s}")
    for domain in domain_counts.most_common():
        d = domain[0]
        row = {c: 0 for c in ["simple", "medium", "complex", "very_complex"]}
        for ex in labeled:
            if ex["labels"]["domain"] == d:
                row[ex["labels"]["complexity"]] += 1
        print(f"  {d:20s} {row['simple']:>8d} {row['medium']:>8d} {row['complex']:>8d} {row['very_complex']:>8d}")


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Label ShortcutDSL training data")
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--stats", action="store_true", help="Print stats only (skip writing)")
    args = parser.parse_args()

    labeler = TrainingDataLabeler()

    output_path = None if args.stats else args.output
    labeled = labeler.label_file(args.input, output_path)

    print_stats(labeled)

    if output_path:
        print(f"\nLabeled data written to: {output_path}")
        print(f"Total examples labeled: {len(labeled)}")


if __name__ == "__main__":
    main()
