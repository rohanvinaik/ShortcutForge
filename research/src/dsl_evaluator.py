"""Lightweight DSL quality checker for cross-model evaluation.

Evaluates generated DSL text quality without depending on the full
Lark parser or compiler pipeline. Used by Phase B LoRA training to
compute DSL metrics at each evaluation checkpoint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ACTION_RE = re.compile(r"^\s*ACTION\s+(\S+)", re.MULTILINE)


@dataclass
class DSLMetrics:
    """Aggregated DSL quality metrics across a batch of outputs."""

    endshortcut_rate: float = 0.0
    valid_action_rate: float = 0.0
    avg_action_count: float = 0.0
    parse_rate: float = 0.0
    first_action_accuracy: float = 0.0


def evaluate_outputs(
    outputs: list[str],
    references: list[dict] | None = None,
) -> DSLMetrics:
    """Evaluate a batch of generated DSL outputs.

    Args:
        outputs: Generated DSL text strings.
        references: Optional list of reference dicts. Each may contain a
            ``messages`` key with chat-format messages where the assistant
            message holds the reference DSL, or a ``reference_dsl`` key.

    Returns:
        Aggregated DSLMetrics over the batch.
    """
    if not outputs:
        return DSLMetrics()

    n = len(outputs)
    has_refs = references is not None and len(references) >= n
    ref_first_actions = _build_ref_first_actions(references, n) if has_refs else []

    counts = _score_outputs(outputs, ref_first_actions)

    return DSLMetrics(
        endshortcut_rate=counts["endshortcut"] / n,
        valid_action_rate=counts["action_lines"] / n,
        avg_action_count=counts["total_actions"] / n,
        parse_rate=counts["parse"] / n,
        first_action_accuracy=_first_action_accuracy(
            ref_first_actions, counts["first_action_matches"]
        ),
    )


def _build_ref_first_actions(references: list[dict] | None, n: int) -> list[str | None]:
    """Extract the first action from each reference up to n."""
    if references is None:
        return []
    return [_extract_ref_first_action(ref) for ref in references[:n]]


def _score_outputs(outputs: list[str], ref_first_actions: list[str | None]) -> dict[str, int]:
    """Score each output and return aggregate counts."""
    counts = {
        "endshortcut": 0,
        "total_actions": 0,
        "action_lines": 0,
        "parse": 0,
        "first_action_matches": 0,
    }
    has_refs = bool(ref_first_actions)

    for i, output in enumerate(outputs):
        if "ENDSHORTCUT" in output:
            counts["endshortcut"] += 1

        actions = _extract_actions(output)
        counts["total_actions"] += len(actions)
        if actions:
            counts["action_lines"] += 1

        if _structural_check(output):
            counts["parse"] += 1

        if has_refs and actions and ref_first_actions[i] is not None:
            if actions[0] == ref_first_actions[i]:
                counts["first_action_matches"] += 1

    return counts


def _first_action_accuracy(ref_first_actions: list[str | None], matches: int) -> float:
    """Compute first-action accuracy from ref actions and match count."""
    if not ref_first_actions:
        return 0.0
    ref_count = sum(1 for a in ref_first_actions if a is not None)
    return matches / ref_count if ref_count > 0 else 0.0


def compute_tier_accuracies(
    outputs: list[str],
    references: list[dict] | None,
    action_catalog: set[str] | None = None,
) -> dict[str, float]:
    """Compute tier-level accuracies for generated outputs.

    Args:
        outputs: Generated DSL text strings.
        references: Reference dicts (same format as evaluate_outputs).
        action_catalog: Optional set of valid action names (unused for now,
            reserved for tier3 validation).

    Returns:
        Dict with ``tier1`` (first action match rate) and ``tier2``
        (parameter structure match rate).
    """
    if not outputs or references is None or len(references) < len(outputs):
        return {"tier1": 0.0, "tier2": 0.0}

    n = len(outputs)
    tier1_matches = 0
    tier2_matches = 0
    ref_count = 0

    for i in range(n):
        ref_dsl = _extract_ref_dsl(references[i])
        if ref_dsl is None:
            continue
        ref_count += 1

        out_actions = _extract_actions(outputs[i])
        ref_actions = _extract_actions(ref_dsl)

        if not ref_actions:
            continue

        # Tier 1: first action matches
        if out_actions and out_actions[0] == ref_actions[0]:
            tier1_matches += 1

            # Tier 2: parameter structure matches for first action block
            out_params = _extract_params_for_action(outputs[i], 0)
            ref_params = _extract_params_for_action(ref_dsl, 0)
            if out_params == ref_params:
                tier2_matches += 1

    if ref_count == 0:
        return {"tier1": 0.0, "tier2": 0.0}

    return {
        "tier1": tier1_matches / ref_count,
        "tier2": tier2_matches / ref_count,
    }


def _extract_actions(text: str) -> list[str]:
    """Extract all action identifiers from ACTION lines."""
    return _ACTION_RE.findall(text)


def _structural_check(text: str) -> bool:
    """Return True if text has basic DSL structure (SHORTCUT + ACTION + ENDSHORTCUT)."""
    has_shortcut = "SHORTCUT" in text and "ENDSHORTCUT" in text
    has_action = bool(_ACTION_RE.search(text))
    return has_shortcut and has_action


def _extract_ref_first_action(ref: dict) -> str | None:
    """Extract the first action from a reference dict."""
    dsl = _extract_ref_dsl(ref)
    if dsl is None:
        return None
    actions = _extract_actions(dsl)
    return actions[0] if actions else None


def _extract_ref_dsl(ref: dict) -> str | None:
    """Extract DSL text from a reference dict."""
    if "reference_dsl" in ref:
        return ref["reference_dsl"]
    messages = ref.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


_PARAM_KEY_RE = re.compile(r"^\s+(\w+):", re.MULTILINE)


def _extract_params_for_action(text: str, action_index: int) -> set[str]:
    """Extract parameter keys for the Nth ACTION block in DSL text."""
    block_lines = _get_action_block_lines(text, action_index)
    params: set[str] = set()
    for line in block_lines:
        m = _PARAM_KEY_RE.match(line)
        if m:
            params.add(m.group(1))
    return params


def _get_action_block_lines(text: str, action_index: int) -> list[str]:
    """Return the indented parameter lines for the Nth ACTION block."""
    lines = text.split("\n")
    action_count = 0
    result: list[str] = []
    in_target = False

    for line in lines:
        if _ACTION_RE.match(line):
            if in_target:
                break
            if action_count == action_index:
                in_target = True
            action_count += 1
            continue

        if not in_target:
            continue
        if line.strip() and not line.startswith(" "):
            break
        result.append(line)

    return result
