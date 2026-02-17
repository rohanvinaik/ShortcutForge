"""
ShortcutForge Distillation Curator — Training Data Quality Pipeline.

Processes distillation logs into curated training JSONL with:
  - Quality filters: must pass parse+validate+compile, no simulation errors
  - Creativity scoring: minimum threshold for inclusion
  - Diversity filters: dedup by prompt similarity, balance across scenario profiles
  - Metadata enrichment: scenario tags, domain profile, architecture decision

Input: distillation_log.jsonl (from evaluate_model.py --log-distillation)
Output: curated_training.jsonl (quality-filtered, deduplicated)

Run: python3 scripts/distillation_curator.py [input.jsonl] [--output out.jsonl]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__version__ = "1.0"


# ── Data Classes ─────────────────────────────────────────────────

@dataclass
class CurationStats:
    """Statistics from a curation run."""
    input_count: int = 0
    quality_passed: int = 0
    quality_failed: int = 0
    dedup_removed: int = 0
    output_count: int = 0
    # Breakdown
    failed_parse: int = 0
    failed_validate: int = 0
    failed_compile: int = 0
    failed_creativity: int = 0
    scenario_distribution: dict[str, int] = field(default_factory=dict)
    complexity_distribution: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Input: {self.input_count}",
            f"Quality gate: {self.quality_passed} passed, {self.quality_failed} failed",
        ]
        if self.quality_failed > 0:
            lines.append(f"  - Parse failures: {self.failed_parse}")
            lines.append(f"  - Validate failures: {self.failed_validate}")
            lines.append(f"  - Compile failures: {self.failed_compile}")
            lines.append(f"  - Below creativity threshold: {self.failed_creativity}")
        lines.append(f"Dedup removed: {self.dedup_removed}")
        lines.append(f"Output: {self.output_count}")
        if self.scenario_distribution:
            lines.append(f"Scenarios: {dict(self.scenario_distribution)}")
        if self.complexity_distribution:
            lines.append(f"Complexity: {dict(self.complexity_distribution)}")
        return "\n".join(lines)


@dataclass
class CuratedEntry:
    """A single curated training example."""
    prompt: str
    canonicalized_output: str
    # Quality signals
    parsed: bool = False
    validated: bool = False
    compiled: bool = False
    # Metadata
    scenario_profile: str = "default"
    domain_profile: str = "general"
    architecture_decision: str = "shortcut_only"
    budget_complexity: str = "medium"
    creativity_score: float | None = None
    # Provenance
    source_id: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_training_dict(self) -> dict[str, Any]:
        """Convert to training JSONL format."""
        return {
            "prompt": self.prompt,
            "completion": self.canonicalized_output,
            "metadata": {
                "scenario_profile": self.scenario_profile,
                "domain_profile": self.domain_profile,
                "architecture_decision": self.architecture_decision,
                "budget_complexity": self.budget_complexity,
                "creativity_score": self.creativity_score,
                "source_id": self.source_id,
            },
            "provenance": self.provenance,
        }


# ── Similarity ───────────────────────────────────────────────────

def _normalize_prompt(prompt: str) -> str:
    """Normalize a prompt for similarity comparison."""
    # Lowercase, remove extra whitespace, strip punctuation
    text = prompt.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _word_overlap_similarity(a: str, b: str) -> float:
    """Compute word overlap similarity between two prompts (Jaccard)."""
    words_a = set(_normalize_prompt(a).split())
    words_b = set(_normalize_prompt(b).split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union) if union else 0.0


# ── Curator ──────────────────────────────────────────────────────

class DistillationCurator:
    """Process distillation logs into curated training data.

    Usage:
        curator = DistillationCurator(
            creativity_threshold=0.3,
            similarity_threshold=0.85,
        )
        entries, stats = curator.curate(input_entries)
    """

    def __init__(
        self,
        creativity_threshold: float = 0.3,
        similarity_threshold: float = 0.85,
        max_per_scenario: int | None = None,
    ):
        self.creativity_threshold = creativity_threshold
        self.similarity_threshold = similarity_threshold
        self.max_per_scenario = max_per_scenario

    def curate(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[list[CuratedEntry], CurationStats]:
        """Curate a list of distillation log entries.

        Args:
            entries: List of dicts from distillation_log.jsonl

        Returns:
            (curated_entries, stats)
        """
        stats = CurationStats(input_count=len(entries))

        # Phase 1: Quality filter
        quality_passed: list[CuratedEntry] = []
        for entry in entries:
            curated = self._apply_quality_gate(entry, stats)
            if curated is not None:
                quality_passed.append(curated)

        stats.quality_passed = len(quality_passed)
        stats.quality_failed = stats.input_count - stats.quality_passed

        # Phase 2: Diversity dedup
        deduplicated = self._deduplicate(quality_passed, stats)

        # Phase 3: Scenario balancing (optional)
        if self.max_per_scenario is not None:
            deduplicated = self._balance_scenarios(deduplicated, stats)

        # Build output stats
        stats.output_count = len(deduplicated)
        for entry in deduplicated:
            scenario = entry.scenario_profile
            stats.scenario_distribution[scenario] = (
                stats.scenario_distribution.get(scenario, 0) + 1
            )
            complexity = entry.budget_complexity
            stats.complexity_distribution[complexity] = (
                stats.complexity_distribution.get(complexity, 0) + 1
            )

        return deduplicated, stats

    def _apply_quality_gate(
        self,
        entry: dict[str, Any],
        stats: CurationStats,
    ) -> CuratedEntry | None:
        """Apply quality filters to a single entry. Returns None if filtered out."""
        prompt = entry.get("prompt", "")
        output = entry.get("canonicalized_output", "")

        if not prompt or not output:
            stats.failed_parse += 1
            return None

        # Must have parsed
        parsed = entry.get("parsed", False)
        if not parsed:
            stats.failed_parse += 1
            return None

        # Must have validated (permissive OK)
        validated = entry.get("validated_permissive", False) or entry.get("validated_strict", False)
        if not validated:
            stats.failed_validate += 1
            return None

        # Must have compiled (permissive OK)
        compiled = entry.get("compiled_permissive", False) or entry.get("compiled_strict", False)
        if not compiled:
            stats.failed_compile += 1
            return None

        # Creativity threshold (if score available)
        creativity = entry.get("creativity_score")
        if creativity is not None and creativity < self.creativity_threshold:
            stats.failed_creativity += 1
            return None

        return CuratedEntry(
            prompt=prompt,
            canonicalized_output=output,
            parsed=parsed,
            validated=validated,
            compiled=compiled,
            scenario_profile=entry.get("scenario_profile", "default"),
            domain_profile=entry.get("domain_profile", "general"),
            architecture_decision=entry.get("architecture_decision", "shortcut_only"),
            budget_complexity=entry.get("budget_complexity", "medium"),
            creativity_score=creativity,
            source_id=entry.get("shortcut_id", ""),
            provenance=entry.get("provenance", {}),
        )

    def _deduplicate(
        self,
        entries: list[CuratedEntry],
        stats: CurationStats,
    ) -> list[CuratedEntry]:
        """Remove near-duplicate prompts using word overlap similarity."""
        if not entries:
            return entries

        kept: list[CuratedEntry] = []
        kept_prompts: list[str] = []

        for entry in entries:
            is_dup = False
            for existing_prompt in kept_prompts:
                sim = _word_overlap_similarity(entry.prompt, existing_prompt)
                if sim >= self.similarity_threshold:
                    is_dup = True
                    break

            if is_dup:
                stats.dedup_removed += 1
            else:
                kept.append(entry)
                kept_prompts.append(entry.prompt)

        return kept

    def _balance_scenarios(
        self,
        entries: list[CuratedEntry],
        stats: CurationStats,
    ) -> list[CuratedEntry]:
        """Cap entries per scenario profile."""
        if self.max_per_scenario is None:
            return entries

        scenario_counts: dict[str, int] = {}
        balanced: list[CuratedEntry] = []

        for entry in entries:
            scenario = entry.scenario_profile
            count = scenario_counts.get(scenario, 0)
            if count < self.max_per_scenario:
                balanced.append(entry)
                scenario_counts[scenario] = count + 1

        return balanced

    def curate_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> CurationStats:
        """Curate a distillation log JSONL file.

        Args:
            input_path: Path to distillation_log.jsonl
            output_path: Path for curated output (default: same dir, curated_training.jsonl)

        Returns:
            CurationStats
        """
        input_path = Path(input_path)
        if output_path is None:
            output_path = input_path.parent / "curated_training.jsonl"
        else:
            output_path = Path(output_path)

        # Read input
        entries: list[dict[str, Any]] = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Curate
        curated, stats = self.curate(entries)

        # Write output
        with open(output_path, "w") as f:
            for entry in curated:
                f.write(json.dumps(entry.to_training_dict()) + "\n")

        return stats


# ── Convenience ──────────────────────────────────────────────────

def curate_distillation_log(
    input_path: str | Path,
    output_path: str | Path | None = None,
    creativity_threshold: float = 0.3,
    similarity_threshold: float = 0.85,
) -> CurationStats:
    """Convenience function to curate a distillation log file."""
    curator = DistillationCurator(
        creativity_threshold=creativity_threshold,
        similarity_threshold=similarity_threshold,
    )
    return curator.curate_file(input_path, output_path)


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python distillation_curator.py <input.jsonl> [--output <out.jsonl>]")
        print("       [--creativity-threshold 0.3] [--similarity-threshold 0.85]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = None
    creativity = 0.3
    similarity = 0.85

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--creativity-threshold" and i + 1 < len(sys.argv):
            creativity = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--similarity-threshold" and i + 1 < len(sys.argv):
            similarity = float(sys.argv[i + 1])
            i += 2
        else:
            i += 1

    curator = DistillationCurator(
        creativity_threshold=creativity,
        similarity_threshold=similarity,
    )
    stats = curator.curate_file(input_file, output_file)

    print(f"\nDistillation Curation Report:")
    print(stats.summary())
