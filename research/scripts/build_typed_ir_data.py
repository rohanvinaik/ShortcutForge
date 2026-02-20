#!/usr/bin/env python3
"""
Build typed IR training/eval data from existing ShortcutDSL JSONL.

Decomposes each DSL example into three-tier representation (TypedIRExample):
Tier 1 structural tokens, Tier 2 parameter blocks, Tier 3 value slots.

Usage:
    uv run python research/scripts/build_typed_ir_data.py -v
    uv run python research/scripts/build_typed_ir_data.py --dry-run --max-examples 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Add src dirs to path for imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "research" / "src"))

from contracts import Tier2Block, Tier3Slot, TypedIRExample  # noqa: E402

from dsl_ir import (  # noqa: E402
    ActionStatement,
    BoolValue,
    Comment,
    DictLiteral,
    ForeachBlock,
    HandleRef,
    HeadersLiteral,
    IfBlock,
    InterpolatedString,
    IRValue,
    ListLiteral,
    MenuBlock,
    NumberValue,
    QuantityLiteral,
    RepeatBlock,
    SetVariable,
    ShortcutIR,
    Statement,
    StringValue,
    VarRef,
    iter_child_blocks,
)
from dsl_linter import lint_dsl  # noqa: E402
from dsl_parser import parse_dsl  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Value serialization
# ---------------------------------------------------------------------------


def _serialize_string(value: StringValue) -> tuple[str, str]:
    return value.value, "string"


def _serialize_number(value: NumberValue) -> tuple[str, str]:
    return str(value.value), "number"


def _serialize_bool(value: BoolValue) -> tuple[str, str]:
    return "true" if value.value else "false", "boolean"


def _serialize_var_ref(value: VarRef) -> tuple[str, str]:
    return f"${value.name}", "var_ref"


def _serialize_handle_ref(value: HandleRef) -> tuple[str, str]:
    return f"@{value.kind}", "handle_ref"


def _serialize_interpolated(value: InterpolatedString) -> tuple[str, str]:
    """Reconstruct backtick string from parts."""
    parts_str = []
    for part in value.parts:
        if isinstance(part, StringValue):
            parts_str.append(part.value)
        elif isinstance(part, VarRef):
            parts_str.append(f"{{{part.name}}}")
        elif isinstance(part, HandleRef):
            parts_str.append(f"{{{part.kind}}}")
        else:
            parts_str.append(str(part))
    return "`" + "".join(parts_str) + "`", "interpolated"


def _serialize_dict(value: DictLiteral) -> tuple[str, str]:
    entries = []
    for key, val in value.entries:
        val_str, _ = _serialize_ir_value(val)
        entries.append(f'"{key}": {val_str}')
    return "{" + ", ".join(entries) + "}", "dict"


def _serialize_list(value: ListLiteral) -> tuple[str, str]:
    items = []
    for item in value.items:
        item_str, _ = _serialize_ir_value(item)
        items.append(item_str)
    return "[" + ", ".join(items) + "]", "list"


def _serialize_quantity(value: QuantityLiteral) -> tuple[str, str]:
    if isinstance(value.magnitude, (VarRef, HandleRef)):
        mag_str, _ = _serialize_ir_value(value.magnitude)
    else:
        mag_str = str(value.magnitude)
    return f'QTY({mag_str}, "{value.unit}")', "quantity"


def _serialize_headers(value: HeadersLiteral) -> tuple[str, str]:
    entries = []
    for key, val in value.entries:
        val_str, _ = _serialize_ir_value(val)
        entries.append(f'"{key}": {val_str}')
    return "HEADERS {" + ", ".join(entries) + "}", "headers"


_SERIALIZE_DISPATCH: dict[type, Any] = {
    StringValue: _serialize_string,
    NumberValue: _serialize_number,
    BoolValue: _serialize_bool,
    VarRef: _serialize_var_ref,
    HandleRef: _serialize_handle_ref,
    InterpolatedString: _serialize_interpolated,
    DictLiteral: _serialize_dict,
    ListLiteral: _serialize_list,
    QuantityLiteral: _serialize_quantity,
    HeadersLiteral: _serialize_headers,
}


def _serialize_ir_value(value: IRValue) -> tuple[str, str]:
    """Convert an IR value to (string_repr, value_kind).

    Returns a human-readable string representation and a kind tag
    used by Tier3Slot.
    """
    handler = _SERIALIZE_DISPATCH.get(type(value))
    if handler is not None:
        return handler(value)
    return str(value), "unknown"


# ---------------------------------------------------------------------------
# IR walking — three-tier decomposition
# ---------------------------------------------------------------------------


# Mapping from context_label (from iter_child_blocks) to (open_token, close_token).
# IfBlock emits open/close at the statement level since it has ELSE handling;
# the child bodies produced by iter_child_blocks use "if_then" and "if_else".
_BLOCK_OPEN_CLOSE: dict[type, tuple[str, str]] = {
    RepeatBlock: ("REPEAT", "ENDREPEAT"),
    ForeachBlock: ("FOREACH", "ENDFOREACH"),
    MenuBlock: ("MENU", "ENDMENU"),
}


def _walk_action(
    stmt: ActionStatement,
    tier1: list[str],
    tier2_blocks: list[Tier2Block],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
    action_idx: int,
) -> tuple[int, int]:
    """Emit tier1/tier2/tier3 data for an ActionStatement."""
    tier1.append("ACTION")
    param_tokens: list[str] = []
    for param_name in stmt.params:
        tier1.append("PARAM")
        tier1.append(param_name)
        param_tokens.append("PARAM")
        param_tokens.append(param_name)

    tier2_blocks.append(
        Tier2Block(
            action_index=action_idx,
            action_name=stmt.action_name,
            tokens=param_tokens,
        )
    )

    for param_name, param_value in stmt.params.items():
        value_str, value_kind = _serialize_ir_value(param_value)
        tier3_slots.append(
            Tier3Slot(
                slot_id=f"s{slot_counter}",
                value_kind=value_kind,
                value=value_str,
                source_param=param_name,
            )
        )
        slot_counter += 1

    action_idx += 1
    return slot_counter, action_idx


def _walk_set_variable(
    stmt: SetVariable,
    tier1: list[str],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
) -> int:
    """Emit tier1/tier3 data for a SetVariable statement."""
    tier1.append("SET")
    value_str, value_kind = _serialize_ir_value(stmt.value)
    tier3_slots.append(
        Tier3Slot(
            slot_id=f"s{slot_counter}",
            value_kind=value_kind,
            value=value_str,
            source_param=stmt.var_name,
        )
    )
    return slot_counter + 1


def _walk_control_block(
    stmt: Statement,
    tier1: list[str],
    tier2_blocks: list[Tier2Block],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
    action_idx: int,
) -> tuple[int, int]:
    """Emit tier1 tokens and recurse into child blocks for control-flow statements.

    Handles IfBlock (with ELSE), RepeatBlock, ForeachBlock, and MenuBlock
    by using iter_child_blocks for recursion and emitting block-specific tokens.
    """
    if isinstance(stmt, IfBlock):
        tier1.append("IF")
        for body, ctx, _is_loop in iter_child_blocks(stmt):
            if ctx == "if_else":
                tier1.append("ELSE")
            slot_counter, action_idx = _walk_statements(
                body, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )
        tier1.append("ENDIF")
    else:
        open_tok, close_tok = _BLOCK_OPEN_CLOSE[type(stmt)]
        tier1.append(open_tok)
        for body, _ctx, _is_loop in iter_child_blocks(stmt):
            slot_counter, action_idx = _walk_statements(
                body, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )
        tier1.append(close_tok)

    return slot_counter, action_idx


def _walk_statements(
    stmts: list[Statement],
    tier1: list[str],
    tier2_blocks: list[Tier2Block],
    tier3_slots: list[Tier3Slot],
    slot_counter: int,
    action_idx: int,
) -> tuple[int, int]:
    """Recursive walk over Statement list, populating all three tiers.

    Returns updated (slot_counter, action_idx).
    """
    for stmt in stmts:
        if isinstance(stmt, Comment):
            continue
        elif isinstance(stmt, ActionStatement):
            slot_counter, action_idx = _walk_action(
                stmt, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )
        elif isinstance(stmt, SetVariable):
            slot_counter = _walk_set_variable(
                stmt, tier1, tier3_slots, slot_counter
            )
        elif isinstance(stmt, (IfBlock, RepeatBlock, ForeachBlock, MenuBlock)):
            slot_counter, action_idx = _walk_control_block(
                stmt, tier1, tier2_blocks, tier3_slots, slot_counter, action_idx
            )

    return slot_counter, action_idx


# ---------------------------------------------------------------------------
# Per-record conversion
# ---------------------------------------------------------------------------


def decompose_dsl_to_typed_ir(
    raw_record: dict[str, Any],
    verbose: bool = False,
) -> TypedIRExample | None:
    """Convert a single raw training record to TypedIRExample.

    Returns None on failure (with logging if verbose).
    """
    shortcut_id = raw_record.get("shortcut_id", "unknown")
    messages = raw_record.get("messages", [])

    if len(messages) < 3:
        if verbose:
            logger.warning("[%s] Fewer than 3 messages, skipping", shortcut_id)
        return None

    system_prompt = messages[0].get("content", "")
    user_prompt = messages[1].get("content", "")
    dsl_text = messages[2].get("content", "")

    if not dsl_text.strip():
        if verbose:
            logger.warning("[%s] Empty DSL text, skipping", shortcut_id)
        return None

    # Step 1: Try linting to canonicalize
    canonicalized = dsl_text
    lint_applied = False
    try:
        lint_result = lint_dsl(dsl_text)
        canonicalized = lint_result.text
        lint_applied = lint_result.was_modified
    except Exception as e:
        if verbose:
            logger.warning("[%s] Lint failed (%s), using raw DSL", shortcut_id, e)

    # Step 2: Parse to IR
    ir: ShortcutIR | None = None
    parse_error: str | None = None

    try:
        ir = parse_dsl(canonicalized)
    except Exception as e:
        parse_error = str(e)
        # Fallback: try parsing raw DSL if linting changed it
        if lint_applied:
            try:
                ir = parse_dsl(dsl_text)
                parse_error = None
            except Exception:
                pass

    if ir is None:
        if verbose:
            logger.warning("[%s] Parse failed: %s", shortcut_id, parse_error)
        return None

    # Step 3: Walk IR to produce three tiers
    tier1_tokens: list[str] = ["SHORTCUT"]
    tier2_blocks: list[Tier2Block] = []
    tier3_slots: list[Tier3Slot] = []

    slot_counter, action_idx = _walk_statements(
        ir.statements,
        tier1_tokens,
        tier2_blocks,
        tier3_slots,
        slot_counter=0,
        action_idx=0,
    )

    tier1_tokens.append("ENDSHORTCUT")

    # Build metadata
    metadata: dict[str, str | int | float | bool] = {
        "action_count": ir.action_count(),
        "tier1_len": len(tier1_tokens),
        "tier2_count": len(tier2_blocks),
        "tier3_count": len(tier3_slots),
        "lint_applied": lint_applied,
        "statement_count": len(ir.statements),
    }

    return TypedIRExample(
        shortcut_id=shortcut_id,
        system_prompt=system_prompt,
        prompt=user_prompt,
        dsl=dsl_text,
        shortcut_name=ir.name,
        tier1_tokens=tier1_tokens,
        tier2_blocks=tier2_blocks,
        tier3_slots=tier3_slots,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _process_file(
    input_path: Path,
    output_path: Path | None,
    label: str,
    max_examples: int | None,
    verbose: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Process a single JSONL file: load, convert, write.

    Returns stats dict for the conversion report.
    """
    if not input_path.exists():
        logger.warning("%s file not found: %s", label, input_path)
        return {
            "file": str(input_path),
            "label": label,
            "total": 0,
            "success": 0,
            "failed": 0,
            "failure_rate_pct": 0.0,
            "skipped_not_found": True,
        }

    print(f"\n{'=' * 60}")
    print(f"Processing {label}: {input_path}")
    print(f"{'=' * 60}")

    records = _load_jsonl(input_path, max_examples)
    total = len(records)
    print(f"  Loaded {total} records")

    results: list[TypedIRExample] = []
    failed_ids: list[str] = []

    t0 = time.time()
    for i, record in enumerate(records):
        result = decompose_dsl_to_typed_ir(record, verbose=verbose)
        if result is not None:
            results.append(result)
        else:
            failed_ids.append(record.get("shortcut_id", f"record_{i}"))

        # Progress reporting every 500 records
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  ... {i + 1}/{total} processed ({rate:.0f} rec/s)")

    elapsed = time.time() - t0
    success = len(results)
    failed = len(failed_ids)
    failure_rate = (failed / total * 100) if total > 0 else 0.0

    print(
        f"\n  Results: {success}/{total} succeeded, {failed} failed "
        f"({failure_rate:.1f}% failure rate)"
    )
    print(f"  Time: {elapsed:.1f}s")

    if failed_ids and verbose:
        print(f"  Failed IDs (first 20): {failed_ids[:20]}")

    # Write output
    if output_path and not dry_run and results:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for ex in results:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"  Written: {output_path} ({success} records)")
    elif dry_run:
        print(f"  [DRY RUN] Would write {success} records to {output_path}")

    # Tier distribution summary
    if results:
        t1_lens = [len(r.tier1_tokens) for r in results]
        t2_counts = [len(r.tier2_blocks) for r in results]
        t3_counts = [len(r.tier3_slots) for r in results]
        print(
            f"\n  Tier1 token counts: min={min(t1_lens)}, "
            f"max={max(t1_lens)}, mean={sum(t1_lens) / len(t1_lens):.1f}"
        )
        print(
            f"  Tier2 block counts: min={min(t2_counts)}, "
            f"max={max(t2_counts)}, mean={sum(t2_counts) / len(t2_counts):.1f}"
        )
        print(
            f"  Tier3 slot counts:  min={min(t3_counts)}, "
            f"max={max(t3_counts)}, mean={sum(t3_counts) / len(t3_counts):.1f}"
        )

    return {
        "file": str(input_path),
        "label": label,
        "total": total,
        "success": success,
        "failed": failed,
        "failure_rate_pct": round(failure_rate, 2),
        "failed_ids": failed_ids[:50],  # Cap for report size
        "elapsed_seconds": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build typed IR data from ShortcutDSL JSONL",
    )
    parser.add_argument(
        "--train-in",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "shortcutdsl_train_expanded.jsonl",
        help="Input training JSONL",
    )
    parser.add_argument(
        "--eval-in",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "shortcutdsl_eval.jsonl",
        help="Input eval JSONL (frozen, 100 examples)",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_train.jsonl",
        help="Output typed IR training JSONL",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_eval.jsonl",
        help="Output typed IR eval JSONL",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=PROJECT_ROOT / "training_data" / "typed_ir_conversion_report.json",
        help="Output conversion report JSON",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=5.0,
        help="Max acceptable failure rate percent (default: 5.0)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit to N examples (for testing)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate without writing output",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print("build_typed_ir_data.py")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  dry_run: {args.dry_run}")
    print(f"  max_examples: {args.max_examples}")
    print(f"  fail_threshold: {args.fail_threshold}%")

    report: dict[str, Any] = {"splits": []}
    overall_total = 0
    overall_failed = 0

    # Process training data
    train_stats = _process_file(
        input_path=args.train_in,
        output_path=args.train_out,
        label="train",
        max_examples=args.max_examples,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    report["splits"].append(train_stats)
    overall_total += train_stats["total"]
    overall_failed += train_stats["failed"]

    # Process eval data
    eval_stats = _process_file(
        input_path=args.eval_in,
        output_path=args.eval_out,
        label="eval",
        max_examples=args.max_examples,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )
    report["splits"].append(eval_stats)
    overall_total += eval_stats["total"]
    overall_failed += eval_stats["failed"]

    # Overall stats
    overall_failure_rate = (
        (overall_failed / overall_total * 100) if overall_total > 0 else 0.0
    )
    report["overall"] = {
        "total": overall_total,
        "success": overall_total - overall_failed,
        "failed": overall_failed,
        "failure_rate_pct": round(overall_failure_rate, 2),
    }

    # Write conversion report
    if not args.dry_run:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nConversion report: {args.report_out}")
    else:
        print(f"\n[DRY RUN] Would write report to {args.report_out}")

    # Final summary
    print(f"\n{'=' * 60}")
    print(
        f"OVERALL: {overall_total - overall_failed}/{overall_total} succeeded "
        f"({overall_failure_rate:.1f}% failure rate)"
    )
    print(f"{'=' * 60}")

    # Gate check
    if overall_failure_rate > args.fail_threshold:
        print(
            f"\nFAIL: Failure rate {overall_failure_rate:.1f}% exceeds "
            f"threshold {args.fail_threshold}%"
        )
        sys.exit(1)
    else:
        print(
            f"\nPASS: Failure rate {overall_failure_rate:.1f}% is within "
            f"threshold {args.fail_threshold}%"
        )


if __name__ == "__main__":
    main()
