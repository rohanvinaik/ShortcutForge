"""Negative bank builder for contrastive margin loss training.

Generates hard negative examples from three sources:
1. Distillation error logs (raw vs. canonicalized pairs)
2. Synthetic mutations on typed IR examples (action swap, param drop, hallucinated name)
3. Linter alias -> canonical mappings

Mutation logic lives in src.mutations. MutationWeights can be derived from a
BehavioralFingerprint to bias mutation selection toward the model's observed
failure modes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import NamedTuple

# ── Path setup (must precede cross-project imports) ──────────────────
_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _RESEARCH_ROOT.parent
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))

from src.behavioral_fingerprint import BehavioralFingerprint  # noqa: E402
from src.contracts import (  # noqa: E402
    NegativeBankEntry,
    Tier2Block,
    TypedIRExample,
)
from src.data import load_typed_ir_jsonl  # noqa: E402
from src.mutations import apply_mutation, collect_action_metadata  # noqa: E402

# ---------------------------------------------------------------------------
# MutationWeights
# ---------------------------------------------------------------------------


class MutationWeights(NamedTuple):
    """Weights for biasing mutation type selection.

    Higher weight = more likely to pick that mutation type.
    Default (1.0, 1.0, 1.0) gives uniform selection.
    """

    action_swap: float = 1.0
    param_drop: float = 1.0
    hallucinated_name: float = 1.0


def weights_from_fingerprint(fp: BehavioralFingerprint) -> MutationWeights:
    """Derive mutation weights from a behavioral fingerprint.

    Heuristics:
        - High action_entropy -> higher hallucinated_name weight
          (model is confused about which actions to use)
        - Concentrated distribution (few actions dominate) -> higher action_swap weight
          (model over-selects certain actions, needs swap exposure)
        - Low discreteness_score -> higher param_drop weight
          (model is indecisive, needs param-level signal)
    """
    hallucinated_name = 1.0 + fp.action_entropy
    action_swap = 1.0 + max(0.0, 1.0 - len(fp.action_distribution) / 10.0)
    param_drop = 1.0 + max(0.0, 1.0 - fp.discreteness_score)
    return MutationWeights(
        action_swap=action_swap,
        param_drop=param_drop,
        hallucinated_name=hallucinated_name,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_distillation_record(record: dict, line_num: int) -> NegativeBankEntry | None:
    """Parse a single distillation log record into a NegativeBankEntry.

    Returns None if the record shows no actual correction.
    """
    shortcut_id = record.get("shortcut_id", f"distill_{line_num}")
    prompt = record.get("prompt", "")
    raw_output = record.get("raw_output", "")
    canonicalized_output = record.get("canonicalized_output", "")
    lint_changes = record.get("lint_changes", [])
    failure_category = record.get("failure_category")

    # Build error_tags from lint_changes and failure_category
    error_tags: list[str] = []
    if failure_category:
        error_tags.append(failure_category)
    for lc in lint_changes:
        kind = lc.get("kind", "unknown")
        if kind not in error_tags:
            error_tags.append(kind)

    # Only create entries where raw != canonicalized (actual correction)
    # or where there's a failure_category (the raw was wrong)
    if raw_output == canonicalized_output and not failure_category and not lint_changes:
        return None

    positive = TypedIRExample(
        shortcut_id=shortcut_id,
        system_prompt="",
        prompt=prompt,
        dsl=canonicalized_output,
        shortcut_name=shortcut_id,
        tier1_tokens=[],
        tier2_blocks=[],
        tier3_slots=[],
        metadata={"source": "distillation_positive"},
    )

    negative = TypedIRExample(
        shortcut_id=shortcut_id,
        system_prompt="",
        prompt=prompt,
        dsl=raw_output,
        shortcut_name=shortcut_id,
        tier1_tokens=[],
        tier2_blocks=[],
        tier3_slots=[],
        metadata={"source": "distillation_negative"},
    )

    lint_changes_dicts: list[dict[str, str | float]] = [
        {
            "kind": lc.get("kind", ""),
            "original": lc.get("original", ""),
            "replacement": lc.get("replacement", ""),
            "confidence": lc.get("confidence", 0.0),
        }
        for lc in lint_changes
    ]

    return NegativeBankEntry(
        prompt=prompt,
        shortcut_id=shortcut_id,
        positive=positive,
        negative=negative,
        error_tags=error_tags if error_tags else ["distillation_diff"],
        source="distillation",
        lint_changes=lint_changes_dicts,
    )


# ---------------------------------------------------------------------------
# Source 1: Distillation negatives
# ---------------------------------------------------------------------------


def generate_distillation_negatives(
    distillation_log: Path,
) -> list[NegativeBankEntry]:
    """Source 1: Build entries from distillation error logs."""
    print("=== Source 1: Distillation errors ===")
    if not distillation_log.exists():
        print(f"  WARNING: {distillation_log} not found, skipping")
        return []

    entries: list[NegativeBankEntry] = []
    with open(distillation_log) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            entry = _parse_distillation_record(record, line_num)
            if entry is not None:
                entries.append(entry)

    print(f"  Loaded {len(entries)} distillation error entries")
    return entries


# ---------------------------------------------------------------------------
# Source 2: Synthetic negatives
# ---------------------------------------------------------------------------


def generate_synthetic_negatives(
    typed_ir_train: Path,
    n_mutations: int,
    verbose: bool,
    weights: MutationWeights | None = None,
) -> list[NegativeBankEntry]:
    """Source 2: Build entries via synthetic mutations on typed IR examples.

    Args:
        typed_ir_train: Path to typed IR training JSONL.
        n_mutations: Number of mutations per example.
        verbose: Print progress.
        weights: Optional MutationWeights to bias mutation selection.
            Defaults to uniform (1.0, 1.0, 1.0).
    """
    print("\n=== Source 2: Synthetic mutations ===")
    if not typed_ir_train.exists():
        print(f"  ERROR: {typed_ir_train} not found.")
        return []

    typed_ir_examples = load_typed_ir_jsonl(typed_ir_train)
    print(f"  Loaded {len(typed_ir_examples)} typed IR examples")

    all_action_names, action_categories, hallucination_aliases = collect_action_metadata(
        typed_ir_examples
    )

    mutation_types = ["action_swap", "param_drop", "hallucinated_name"]
    w = weights or MutationWeights()
    weight_list = [w.action_swap, w.param_drop, w.hallucinated_name]
    entries: list[NegativeBankEntry] = []

    for ex_idx, ex in enumerate(typed_ir_examples):
        if verbose and ex_idx % 1000 == 0:
            print(f"  Processing example {ex_idx}/{len(typed_ir_examples)}...")
        if not ex.tier2_blocks:
            continue
        for _mut_i in range(n_mutations):
            mut_type = random.choices(mutation_types, weights=weight_list, k=1)[0]
            entry = apply_mutation(
                ex,
                mut_type,
                all_action_names,
                action_categories,
                hallucination_aliases,
            )
            if entry is not None:
                entries.append(entry)

    print(f"  Generated {len(entries)} synthetic mutation entries")
    return entries


# ---------------------------------------------------------------------------
# Source 3: Linter negatives
# ---------------------------------------------------------------------------


def _make_linter_example(action_name: str, label: str) -> TypedIRExample:
    """Build a minimal TypedIRExample for a linter taxonomy entry."""
    return TypedIRExample(
        shortcut_id=f"linter_{action_name}",
        system_prompt="",
        prompt=f"Action using {action_name}",
        dsl=f'SHORTCUT "Linter Example"\nACTION {action_name}\nENDSHORTCUT\n',
        shortcut_name="Linter Example",
        tier1_tokens=["SHORTCUT", "ACTION", "ENDSHORTCUT"],
        tier2_blocks=[Tier2Block(action_index=0, action_name=action_name, tokens=[])],
        tier3_slots=[],
        metadata={"source": label},
    )


def generate_linter_negatives(
    canonical_map: dict[str, str],
) -> list[NegativeBankEntry]:
    """Source 3: Build entries from linter alias -> canonical mappings."""
    print("\n=== Source 3: Linter taxonomy ===")
    entries: list[NegativeBankEntry] = []

    for alias, canonical in canonical_map.items():
        entries.append(
            NegativeBankEntry(
                prompt=f"Action: {alias} -> {canonical}",
                shortcut_id=f"linter_{alias}",
                positive=_make_linter_example(canonical, "linter_taxonomy"),
                negative=_make_linter_example(alias, "linter_taxonomy"),
                error_tags=["hallucination_alias"],
                source="linter_repair",
                lint_changes=[
                    {
                        "kind": "action",
                        "original": alias,
                        "replacement": canonical,
                        "confidence": 0.95,
                    }
                ],
            )
        )

    print(f"  Generated {len(entries)} linter taxonomy entries")
    return entries
