"""Mutation strategies for synthetic negative generation.

Implements action swap, parameter drop, and hallucinated name mutations
on typed IR examples. Used by negative_bank_builder to create contrastive
training pairs for margin loss.
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

_RESEARCH_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _RESEARCH_ROOT.parent
_COMPILER_SRC = _PROJECT_ROOT / "src"
if str(_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RESEARCH_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.append(str(_PROJECT_ROOT))
if str(_COMPILER_SRC) not in sys.path:
    sys.path.insert(0, str(_COMPILER_SRC))

from dsl_linter import ActionResolver  # noqa: E402

from src.contracts import (  # noqa: E402
    NegativeBankEntry,
    Tier2Block,
    TypedIRExample,
)


def action_category(name: str) -> str:
    """Classify an action name into a rough category by prefix."""
    if name.startswith("is.workflow.actions."):
        return "is.workflow.actions"
    if "." in name:
        return ".".join(name.split(".")[:2])
    return "short"


def collect_action_metadata(
    examples: list[TypedIRExample],
) -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Collect action names, categories, and hallucination aliases from examples.

    Returns (all_action_names, action_categories, hallucination_aliases).
    """
    all_action_names: list[str] = []
    for ex in examples:
        for block in ex.tier2_blocks:
            if block.action_name not in all_action_names:
                all_action_names.append(block.action_name)

    action_categories: dict[str, list[str]] = {}
    for name in all_action_names:
        action_categories.setdefault(action_category(name), []).append(name)

    hallucination_aliases = list(ActionResolver.HALLUCINATION_ALIASES.keys())
    return all_action_names, action_categories, hallucination_aliases


def mutate_action_swap(
    original_block: Tier2Block,
    neg_blocks: list[Tier2Block],
    block_idx: int,
    all_action_names: list[str],
    action_categories: dict[str, list[str]],
) -> tuple[list[Tier2Block], dict[str, str | float]] | None:
    """Replace action_name with a different action from the same category."""
    orig_name = original_block.action_name
    cat = action_category(orig_name)

    candidates = [n for n in action_categories.get(cat, all_action_names) if n != orig_name]
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
    return neg_blocks, {
        "kind": "action_swap",
        "original": orig_name,
        "replacement": new_name,
        "confidence": 1.0,
    }


def mutate_param_drop(
    original_block: Tier2Block,
    neg_blocks: list[Tier2Block],
    block_idx: int,
) -> tuple[list[Tier2Block], dict[str, str | float]] | None:
    """Remove one parameter token from the block's tokens."""
    if len(original_block.tokens) < 2:
        return None

    param_positions = [ti for ti, tok in enumerate(original_block.tokens) if tok == "PARAM"]

    if param_positions:
        drop_pos = random.choice(param_positions)
        new_tokens = list(original_block.tokens)
        end_pos = min(drop_pos + 2, len(new_tokens))
        dropped = new_tokens[drop_pos:end_pos]
        new_tokens = new_tokens[:drop_pos] + new_tokens[end_pos:]
        desc: dict[str, str | float] = {
            "kind": "param_drop",
            "original": " ".join(dropped),
            "replacement": "<removed>",
            "confidence": 1.0,
        }
    else:
        drop_idx = random.randint(0, len(original_block.tokens) - 1)
        new_tokens = list(original_block.tokens)
        dropped_tok = new_tokens.pop(drop_idx)
        desc = {
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
    return neg_blocks, desc


def mutate_hallucinated_name(
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
            and target == orig_name[len("is.workflow.actions.") :]
        )
        or ("is.workflow.actions." + target) == orig_name
    ]
    hallucinated = (
        random.choice(matching_aliases)
        if matching_aliases
        else random.choice(hallucination_aliases)
    )

    neg_blocks[block_idx] = Tier2Block(
        action_index=original_block.action_index,
        action_name=hallucinated,
        tokens=list(original_block.tokens),
    )
    return neg_blocks, {
        "kind": "hallucinated_name",
        "original": orig_name,
        "replacement": hallucinated,
        "confidence": 1.0,
    }


def apply_mutation(
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

    _DISPATCH = {
        "action_swap": lambda: mutate_action_swap(
            original_block, neg_blocks, block_idx, all_action_names, action_categories
        ),
        "param_drop": lambda: mutate_param_drop(original_block, neg_blocks, block_idx),
        "hallucinated_name": lambda: mutate_hallucinated_name(
            original_block, neg_blocks, block_idx, hallucination_aliases
        ),
    }
    handler = _DISPATCH.get(mut_type)
    if handler is None:
        return None
    result = handler()
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
