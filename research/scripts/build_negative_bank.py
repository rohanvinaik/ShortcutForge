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
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description="Build hard negative bank for margin loss training",
    )
    parser.add_argument("--distillation-log", type=Path,
                       default=PROJECT_ROOT / "training_data" / "distillation_log.jsonl",
                       help="Distillation log JSONL (raw->canonicalized pairs)")
    parser.add_argument("--typed-ir-train", type=Path,
                       default=PROJECT_ROOT / "training_data" / "typed_ir_train.jsonl",
                       help="Typed IR training JSONL (for positive examples)")
    parser.add_argument("--out", type=Path,
                       default=PROJECT_ROOT / "training_data" / "hard_negative_bank.jsonl",
                       help="Output negative bank JSONL")
    parser.add_argument("--synthetic-mutations-per-example", type=int, default=3,
                       help="Number of synthetic negative mutations per example")
    parser.add_argument("--min-triples", type=int, default=3000,
                       help="Minimum entries for Phase 0 acceptance")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--dry-run", action="store_true",
                       help="Count without writing output")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose output")
    args = parser.parse_args()

    import copy
    import json
    import random

    random.seed(args.seed)

    # ── Imports from project ──────────────────────────────────────────
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from research.src.contracts import NegativeBankEntry, TypedIRExample, Tier2Block, Tier3Slot
    from research.src.data import load_typed_ir_jsonl
    from dsl_linter import ActionResolver

    # ── Build canonical_map from ActionResolver ───────────────────────
    resolver = ActionResolver()
    canonical_map: dict[str, str] = dict(resolver._canonical_map)

    entries: list[NegativeBankEntry] = []
    source_counts: dict[str, int] = {"distillation": 0, "synthetic_mutation": 0, "linter_repair": 0}

    # ══════════════════════════════════════════════════════════════════
    # Source 1: Distillation errors
    # ══════════════════════════════════════════════════════════════════
    print("=== Source 1: Distillation errors ===")
    if not args.distillation_log.exists():
        print(f"  WARNING: {args.distillation_log} not found, skipping")
    else:
        with open(args.distillation_log) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
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
                    continue

                # Build a minimal positive TypedIRExample from canonicalized output
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

                # Build negative from raw (errored) output
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

                # Convert lint_changes to the expected format
                lint_changes_dicts: list[dict[str, str | float]] = []
                for lc in lint_changes:
                    lint_changes_dicts.append({
                        "kind": lc.get("kind", ""),
                        "original": lc.get("original", ""),
                        "replacement": lc.get("replacement", ""),
                        "confidence": lc.get("confidence", 0.0),
                    })

                entry = NegativeBankEntry(
                    prompt=prompt,
                    shortcut_id=shortcut_id,
                    positive=positive,
                    negative=negative,
                    error_tags=error_tags if error_tags else ["distillation_diff"],
                    source="distillation",
                    lint_changes=lint_changes_dicts,
                )
                entries.append(entry)
                source_counts["distillation"] += 1

        print(f"  Loaded {source_counts['distillation']} distillation error entries")

    # ══════════════════════════════════════════════════════════════════
    # Source 2: Synthetic mutations from typed_ir_train
    # ══════════════════════════════════════════════════════════════════
    print("\n=== Source 2: Synthetic mutations ===")
    if not args.typed_ir_train.exists():
        print(f"  ERROR: {args.typed_ir_train} not found.")
        print("  Run build_typed_ir_data.py first to generate typed IR training data.")
    else:
        typed_ir_examples = load_typed_ir_jsonl(args.typed_ir_train)
        print(f"  Loaded {len(typed_ir_examples)} typed IR examples")

        # Collect all action names that appear in training data for action swap
        all_action_names: list[str] = []
        for ex in typed_ir_examples:
            for block in ex.tier2_blocks:
                if block.action_name not in all_action_names:
                    all_action_names.append(block.action_name)

        # Collect hallucination alias keys for hallucinated-name mutations
        hallucination_aliases = list(ActionResolver.HALLUCINATION_ALIASES.keys())

        # Group action names by rough category (prefix) for action swap
        action_categories: dict[str, list[str]] = {}
        for name in all_action_names:
            if name.startswith("is.workflow.actions."):
                cat = "is.workflow.actions"
            elif "." in name:
                cat = ".".join(name.split(".")[:2])
            else:
                cat = "short"
            action_categories.setdefault(cat, []).append(name)

        n_mutations = args.synthetic_mutations_per_example
        mutation_types = ["action_swap", "param_drop", "hallucinated_name"]

        for ex_idx, ex in enumerate(typed_ir_examples):
            if args.verbose and ex_idx % 1000 == 0:
                print(f"  Processing example {ex_idx}/{len(typed_ir_examples)}...")

            if not ex.tier2_blocks:
                continue

            # Generate N mutations per example
            for mut_i in range(n_mutations):
                mut_type = mutation_types[mut_i % len(mutation_types)]

                # Pick a random block to mutate
                block_idx = random.randint(0, len(ex.tier2_blocks) - 1)
                original_block = ex.tier2_blocks[block_idx]

                neg_blocks = [copy.deepcopy(b) for b in ex.tier2_blocks]
                error_tag = mut_type
                lint_change_desc: dict[str, str | float] = {}

                if mut_type == "action_swap":
                    # Replace action_name with a different action from the same category
                    orig_name = original_block.action_name
                    if orig_name.startswith("is.workflow.actions."):
                        cat = "is.workflow.actions"
                    elif "." in orig_name:
                        cat = ".".join(orig_name.split(".")[:2])
                    else:
                        cat = "short"

                    candidates = [n for n in action_categories.get(cat, all_action_names) if n != orig_name]
                    if not candidates:
                        candidates = [n for n in all_action_names if n != orig_name]

                    if candidates:
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
                    else:
                        continue

                elif mut_type == "param_drop":
                    # Remove one parameter token from the block's tokens
                    if len(original_block.tokens) >= 2:
                        # Find PARAM positions (tokens that start with "PARAM")
                        param_positions: list[int] = []
                        for ti, tok in enumerate(original_block.tokens):
                            if tok == "PARAM":
                                param_positions.append(ti)

                        if param_positions:
                            drop_pos = random.choice(param_positions)
                            # Drop the PARAM token and the following token (param name)
                            new_tokens = list(original_block.tokens)
                            end_pos = min(drop_pos + 2, len(new_tokens))
                            dropped = new_tokens[drop_pos:end_pos]
                            new_tokens = new_tokens[:drop_pos] + new_tokens[end_pos:]
                            neg_blocks[block_idx] = Tier2Block(
                                action_index=original_block.action_index,
                                action_name=original_block.action_name,
                                tokens=new_tokens,
                            )
                            lint_change_desc = {
                                "kind": "param_drop",
                                "original": " ".join(dropped),
                                "replacement": "<removed>",
                                "confidence": 1.0,
                            }
                        else:
                            # No PARAM tokens, just drop a random token
                            drop_idx = random.randint(0, len(original_block.tokens) - 1)
                            new_tokens = list(original_block.tokens)
                            dropped_tok = new_tokens.pop(drop_idx)
                            neg_blocks[block_idx] = Tier2Block(
                                action_index=original_block.action_index,
                                action_name=original_block.action_name,
                                tokens=new_tokens,
                            )
                            lint_change_desc = {
                                "kind": "param_drop",
                                "original": dropped_tok,
                                "replacement": "<removed>",
                                "confidence": 1.0,
                            }
                    else:
                        continue

                elif mut_type == "hallucinated_name":
                    # Replace action name with a hallucination alias
                    orig_name = original_block.action_name
                    # Find aliases that map to this action or a related one
                    matching_aliases = [
                        alias for alias, target in ActionResolver.HALLUCINATION_ALIASES.items()
                        if target == orig_name
                        or (orig_name.startswith("is.workflow.actions.")
                            and target == orig_name[len("is.workflow.actions."):])
                        or ("is.workflow.actions." + target) == orig_name
                    ]
                    if matching_aliases:
                        hallucinated = random.choice(matching_aliases)
                    else:
                        # Pick a random hallucination alias
                        hallucinated = random.choice(hallucination_aliases)

                    neg_blocks[block_idx] = Tier2Block(
                        action_index=original_block.action_index,
                        action_name=hallucinated,
                        tokens=list(original_block.tokens),
                    )
                    lint_change_desc = {
                        "kind": "hallucinated_name",
                        "original": orig_name,
                        "replacement": hallucinated,
                        "confidence": 1.0,
                    }

                # Build negative TypedIRExample
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

                entry = NegativeBankEntry(
                    prompt=ex.prompt,
                    shortcut_id=ex.shortcut_id,
                    positive=ex,
                    negative=negative,
                    error_tags=[error_tag],
                    source="synthetic_mutation",
                    lint_changes=[lint_change_desc] if lint_change_desc else [],
                )
                entries.append(entry)
                source_counts["synthetic_mutation"] += 1

        print(f"  Generated {source_counts['synthetic_mutation']} synthetic mutation entries")

    # ══════════════════════════════════════════════════════════════════
    # Source 3: Linter taxonomy (hallucination alias -> canonical pairs)
    # ══════════════════════════════════════════════════════════════════
    print("\n=== Source 3: Linter taxonomy ===")
    for alias, canonical in canonical_map.items():
        # Create a minimal entry for each alias -> canonical mapping
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
            lint_changes=[{
                "kind": "action",
                "original": alias,
                "replacement": canonical,
                "confidence": 0.95,
            }],
        )
        entries.append(entry)
        source_counts["linter_repair"] += 1

    print(f"  Generated {source_counts['linter_repair']} linter taxonomy entries")

    # ══════════════════════════════════════════════════════════════════
    # Summary and output
    # ══════════════════════════════════════════════════════════════════
    total = len(entries)
    print(f"\n=== Summary ===")
    print(f"  Distillation errors:  {source_counts['distillation']:>6}")
    print(f"  Synthetic mutations:  {source_counts['synthetic_mutation']:>6}")
    print(f"  Linter taxonomy:      {source_counts['linter_repair']:>6}")
    print(f"  Total:                {total:>6}")

    if total < args.min_triples:
        print(f"\n  WARNING: {total} entries < {args.min_triples} minimum. Gate NOT met.")
        sys.exit(1)
    else:
        print(f"\n  Gate: {total} >= {args.min_triples} minimum. PASSED.")

    if args.dry_run:
        print("\n  --dry-run: skipping write")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        print(f"\n  Wrote {total} entries to {args.out}")


if __name__ == "__main__":
    main()
