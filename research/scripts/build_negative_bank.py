#!/usr/bin/env python3
"""
Build hard negative bank from distillation logs and linter repair taxonomy.

Negative bank entries are contrastive pairs (positive, negative) with error
tags for margin loss training.

Usage:
    uv run python research/scripts/build_negative_bank.py -v
    uv run python research/scripts/build_negative_bank.py --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Imports from project ──────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dsl_linter import ActionResolver  # noqa: E402
from research.src.contracts import (  # noqa: E402
    NegativeBankEntry,
    Tier2Block,
    TypedIRExample,
)
from research.src.data import load_typed_ir_jsonl  # noqa: E402

# ---------------------------------------------------------------------------
# Sub-functions
# ---------------------------------------------------------------------------


def _generate_distillation_negatives(
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


def _parse_distillation_record(
    record: dict, line_num: int
) -> NegativeBankEntry | None:
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


def _action_category(name: str) -> str:
    """Classify an action name into a rough category by prefix."""
    if name.startswith("is.workflow.actions."):
        return "is.workflow.actions"
    if "." in name:
        return ".".join(name.split(".")[:2])
    return "short"


def _generate_synthetic_negatives(
    typed_ir_train: Path,
    n_mutations: int,
    verbose: bool,
) -> list[NegativeBankEntry]:
    """Source 2: Build entries via synthetic mutations on typed IR examples."""
    print("\n=== Source 2: Synthetic mutations ===")
    if not typed_ir_train.exists():
        print(f"  ERROR: {typed_ir_train} not found.")
        print("  Run build_typed_ir_data.py first to generate typed IR training data.")
        return []

    typed_ir_examples = load_typed_ir_jsonl(typed_ir_train)
    print(f"  Loaded {len(typed_ir_examples)} typed IR examples")

    # Collect all action names for action swap candidates
    all_action_names: list[str] = []
    for ex in typed_ir_examples:
        for block in ex.tier2_blocks:
            if block.action_name not in all_action_names:
                all_action_names.append(block.action_name)

    hallucination_aliases = list(ActionResolver.HALLUCINATION_ALIASES.keys())

    # Group action names by rough category for action swap
    action_categories: dict[str, list[str]] = {}
    for name in all_action_names:
        action_categories.setdefault(_action_category(name), []).append(name)

    mutation_types = ["action_swap", "param_drop", "hallucinated_name"]
    entries: list[NegativeBankEntry] = []

    for ex_idx, ex in enumerate(typed_ir_examples):
        if verbose and ex_idx % 1000 == 0:
            print(f"  Processing example {ex_idx}/{len(typed_ir_examples)}...")
        if not ex.tier2_blocks:
            continue
        for mut_i in range(n_mutations):
            mut_type = mutation_types[mut_i % len(mutation_types)]
            entry = _apply_mutation(
                ex, mut_type, all_action_names, action_categories,
                hallucination_aliases,
            )
            if entry is not None:
                entries.append(entry)

    print(f"  Generated {len(entries)} synthetic mutation entries")
    return entries


def _apply_mutation(
    ex: TypedIRExample,
    mut_type: str,
    all_action_names: list[str],
    action_categories: dict[str, list[str]],
    hallucination_aliases: list[str],
) -> NegativeBankEntry | None:
    """Apply a single mutation to produce one NegativeBankEntry, or None on skip."""
    block_idx = random.randint(0, len(ex.tier2_blocks) - 1)
    original_block = ex.tier2_blocks[block_idx]
    neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]

    if mut_type == "action_swap":
        result = _mutate_action_swap(
            original_block, neg_blocks, block_idx,
            all_action_names, action_categories,
        )
    elif mut_type == "param_drop":
        result = _mutate_param_drop(original_block, neg_blocks, block_idx)
    elif mut_type == "hallucinated_name":
        result = _mutate_hallucinated_name(
            original_block, neg_blocks, block_idx, hallucination_aliases,
        )
    else:
        return None

    if result is None:
        return None

    neg_blocks, lint_change_desc = result

    negative = TypedIRExample(
        shortcut_id=ex.shortcut_id,
        system_prompt=ex.system_prompt,
        prompt=ex.prompt,
        dsl=ex.dsl,
        shortcut_name=ex.shortcut_name,
        tier1_tokens=list(ex.tier1_tokens),
        tier2_blocks=neg_blocks,
        tier3_slots=[copy.deepcopy(s) for s in ex.tier3_slots],
        metadata={"source": "synthetic_negative", "mutation": mut_type},
    )

    return NegativeBankEntry(
        prompt=ex.prompt,
        shortcut_id=ex.shortcut_id,
        positive=ex,
        negative=negative,
        error_tags=[mut_type],
        source="synthetic_mutation",
        lint_changes=[lint_change_desc] if lint_change_desc else [],
    )


def _mutate_action_swap(
    original_block: Tier2Block,
    neg_blocks: list[Tier2Block],
    block_idx: int,
    all_action_names: list[str],
    action_categories: dict[str, list[str]],
) -> tuple[list[Tier2Block], dict[str, str | float]] | None:
    """Replace action_name with a different action from the same category."""
    orig_name = original_block.action_name
    cat = _action_category(orig_name)

    candidates = [
        n for n in action_categories.get(cat, all_action_names) if n != orig_name
    ]
    if not candidates:
        candidates = [n for n in all_action_names if n != orig_name]
    if not candidates:
        return None

    new_name = random.choice(candidates)
    neg_blocks[block_idx] = Tier2Block(
        action_index=original_block.action_index,
        action_name=new_name,
        tokens=list(original_block.tokens),
    )
    lint_change_desc = {
        "kind": "action_swap",
        "original": orig_name,
        "replacement": new_name,
        "confidence": 1.0,
    }
    return neg_blocks, lint_change_desc


def _mutate_param_drop(
    original_block: Tier2Block,
    neg_blocks: list[Tier2Block],
    block_idx: int,
) -> tuple[list[Tier2Block], dict[str, str | float]] | None:
    """Remove one parameter token from the block's tokens."""
    if len(original_block.tokens) < 2:
        return None

    param_positions = [
        ti for ti, tok in enumerate(original_block.tokens) if tok == "PARAM"
    ]

    if param_positions:
        drop_pos = random.choice(param_positions)
        new_tokens = list(original_block.tokens)
        end_pos = min(drop_pos + 2, len(new_tokens))
        dropped = new_tokens[drop_pos:end_pos]
        new_tokens = new_tokens[:drop_pos] + new_tokens[end_pos:]
        lint_change_desc: dict[str, str | float] = {
            "kind": "param_drop",
            "original": " ".join(dropped),
            "replacement": "<removed>",
            "confidence": 1.0,
        }
    else:
        drop_idx = random.randint(0, len(original_block.tokens) - 1)
        new_tokens = list(original_block.tokens)
        dropped_tok = new_tokens.pop(drop_idx)
        lint_change_desc = {
            "kind": "param_drop",
            "original": dropped_tok,
            "replacement": "<removed>",
            "confidence": 1.0,
        }

    neg_blocks[block_idx] = Tier2Block(
        action_index=original_block.action_index,
        action_name=original_block.action_name,
        tokens=new_tokens,
    )
    return neg_blocks, lint_change_desc


def _mutate_hallucinated_name(
    original_block: Tier2Block,
    neg_blocks: list[Tier2Block],
    block_idx: int,
    hallucination_aliases: list[str],
) -> tuple[list[Tier2Block], dict[str, str | float]]:
    """Replace action name with a hallucination alias."""
    orig_name = original_block.action_name
    matching_aliases = [
        alias
        for alias, target in ActionResolver.HALLUCINATION_ALIASES.items()
        if target == orig_name
        or (
            orig_name.startswith("is.workflow.actions.")
            and target == orig_name[len("is.workflow.actions."):]
        )
        or ("is.workflow.actions." + target) == orig_name
    ]
    if matching_aliases:
        hallucinated = random.choice(matching_aliases)
    else:
        hallucinated = random.choice(hallucination_aliases)

    neg_blocks[block_idx] = Tier2Block(
        action_index=original_block.action_index,
        action_name=hallucinated,
        tokens=list(original_block.tokens),
    )
    lint_change_desc: dict[str, str | float] = {
        "kind": "hallucinated_name",
        "original": orig_name,
        "replacement": hallucinated,
        "confidence": 1.0,
    }
    return neg_blocks, lint_change_desc


def _generate_linter_taxonomy_negatives(
    canonical_map: dict[str, str],
) -> list[NegativeBankEntry]:
    """Source 3: Build entries from linter alias -> canonical mappings."""
    print("\n=== Source 3: Linter taxonomy ===")
    entries: list[NegativeBankEntry] = []

    for alias, canonical in canonical_map.items():
        positive = TypedIRExample(
            shortcut_id=f"linter_{canonical}",
            system_prompt="",
            prompt=f"Action using {canonical}",
            dsl=f'SHORTCUT "Linter Example"\nACTION {canonical}\nENDSHORTCUT\n',
            shortcut_name="Linter Example",
            tier1_tokens=["SHORTCUT", "ACTION", "ENDSHORTCUT"],
            tier2_blocks=[Tier2Block(action_index=0, action_name=canonical, tokens=[])],
            tier3_slots=[],
            metadata={"source": "linter_taxonomy"},
        )

        negative = TypedIRExample(
            shortcut_id=f"linter_{alias}",
            system_prompt="",
            prompt=f"Action using {alias}",
            dsl=f'SHORTCUT "Linter Example"\nACTION {alias}\nENDSHORTCUT\n',
            shortcut_name="Linter Example",
            tier1_tokens=["SHORTCUT", "ACTION", "ENDSHORTCUT"],
            tier2_blocks=[Tier2Block(action_index=0, action_name=alias, tokens=[])],
            tier3_slots=[],
            metadata={"source": "linter_taxonomy"},
        )

        entry = NegativeBankEntry(
            prompt=f"Action: {alias} -> {canonical}",
            shortcut_id=f"linter_{alias}",
            positive=positive,
            negative=negative,
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
        entries.append(entry)

    print(f"  Generated {len(entries)} linter taxonomy entries")
    return entries


def _summarize_and_save(
    entries: list[NegativeBankEntry],
    source_counts: dict[str, int],
    out: Path,
    min_triples: int,
    dry_run: bool,
) -> None:
    """Print summary, check gate, and optionally write output."""
    total = len(entries)
    print("\n=== Summary ===")
    print(f"  Distillation errors:  {source_counts['distillation']:>6}")
    print(f"  Synthetic mutations:  {source_counts['synthetic_mutation']:>6}")
    print(f"  Linter taxonomy:      {source_counts['linter_repair']:>6}")
    print(f"  Total:                {total:>6}")

    if total < min_triples:
        print(
            f"\n  WARNING: {total} entries < {min_triples} minimum. Gate NOT met."
        )
        sys.exit(1)
    else:
        print(f"\n  Gate: {total} >= {min_triples} minimum. PASSED.")

    if dry_run:
        print("\n  --dry-run: skipping write")
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        print(f"\n  Wrote {total} entries to {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build hard negative bank for margin loss training",
    )
    parser.add_argument(
        "--distillation-log",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "distillation_log.jsonl",
        help="Distillation log JSONL (raw->canonicalized pairs)",
    )
    parser.add_argument(
        "--typed-ir-train",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_train.jsonl",
        help="Typed IR training JSONL (for positive examples)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "hard_negative_bank.jsonl",
        help="Output negative bank JSONL",
    )
    parser.add_argument(
        "--synthetic-mutations-per-example",
        type=int,
        default=3,
        help="Number of synthetic negative mutations per example",
    )
    parser.add_argument(
        "--min-triples",
        type=int,
        default=3000,
        help="Minimum entries for Phase 0 acceptance",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run", action="store_true", help="Count without writing output"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    random.seed(args.seed)

    resolver = ActionResolver()
    canonical_map: dict[str, str] = dict(resolver._canonical_map)

    # Generate entries from all three sources
    distillation_entries = _generate_distillation_negatives(args.distillation_log)
    synthetic_entries = _generate_synthetic_negatives(
        args.typed_ir_train, args.synthetic_mutations_per_example, args.verbose,
    )
    linter_entries = _generate_linter_taxonomy_negatives(canonical_map)

    entries = distillation_entries + synthetic_entries + linter_entries
    source_counts = {
        "distillation": len(distillation_entries),
        "synthetic_mutation": len(synthetic_entries),
        "linter_repair": len(linter_entries),
    }

    _summarize_and_save(entries, source_counts, args.out, args.min_triples, args.dry_run)


if __name__ == "__main__":
    main()
