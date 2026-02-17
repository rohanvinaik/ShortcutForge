#!/usr/bin/env python3
"""
ShortcutForge CLI: Natural language → Apple Shortcut.

Usage:
    python shortcutforge.py "Set a 5-minute timer and notify me"
    python shortcutforge.py --dsl-file my_shortcut.dsl
    echo "Copy clipboard to a note" | python shortcutforge.py -

Edit/modify existing shortcuts:
    python shortcutforge.py --edit my_shortcut.shortcut --show-dsl
    python shortcutforge.py --edit my_shortcut.shortcut --modify "Add a notification at the end"
    python shortcutforge.py --edit my_shortcut.shortcut --dsl-file patched.dsl

Options:
    --model MODEL        Anthropic model (default: claude-sonnet-4-20250514)
    --max-retries N      Max retry attempts (default: 3)
    --output-dir DIR     Output directory (default: ./output)
    --no-sign            Skip signing
    --auto-import        Import into Shortcuts.app after building
    --dsl-only           Print generated DSL only, don't compile
    --verbose / -v       Show DSL text, warnings, and timing
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure scripts/ is importable
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _print_stage(stage_result):
    """Print a stage update to the terminal."""
    from orchestrator import StageResult

    STAGE_ORDER = {
        "generating": 1,
        "parsing": 2,
        "validating": 3,
        "compiling": 4,
        "delivering": 5,
    }
    idx = STAGE_ORDER.get(stage_result.stage, 0)

    if stage_result.status == "skipped":
        print(f"  [{idx}/5] {stage_result.stage.capitalize()}... skipped", flush=True)
    elif stage_result.status == "running":
        print(f"  [{idx}/5] {stage_result.stage.capitalize()}...", end=" ", flush=True)
    elif stage_result.status == "success":
        duration = ""
        if stage_result.duration_ms:
            duration = f" ({stage_result.duration_ms}ms)"
        print(f"done{duration}", flush=True)
    elif stage_result.status == "failed":
        print(f"FAILED", flush=True)


def main():
    parser = argparse.ArgumentParser(
        prog="shortcutforge",
        description="ShortcutForge: Natural language → Apple Shortcut",
        epilog="Set ANTHROPIC_API_KEY env var before using LLM generation.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help='Natural language description, or "-" to read from stdin',
    )
    parser.add_argument(
        "--dsl-file",
        type=str,
        default=None,
        help="Skip LLM, compile this DSL file directly",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts on errors (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help="Skip signing",
    )
    parser.add_argument(
        "--auto-import",
        action="store_true",
        help="Import into Shortcuts.app after building",
    )
    parser.add_argument(
        "--dsl-only",
        action="store_true",
        help="Print generated DSL only, don't compile",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["claude", "local"],
        default="claude",
        help="Generation engine (default: claude)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model (required when --engine=local)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter directory (for --engine=local)",
    )
    parser.add_argument(
        "--no-grammar",
        action="store_true",
        help="Never use grammar-constrained retries (for --engine=local)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        choices=["llama3", "chatml"],
        default="llama3",
        help="Chat template for local engine prompt formatting (default: llama3)",
    )
    parser.add_argument(
        "--edit",
        type=str,
        default=None,
        help="Load an existing .shortcut file for editing/modification",
    )
    parser.add_argument(
        "--show-dsl",
        action="store_true",
        help="With --edit: show the decompiled DSL and exit",
    )
    parser.add_argument(
        "--modify",
        type=str,
        default=None,
        help="With --edit: modification instruction for LLM to apply to the loaded shortcut",
    )
    parser.add_argument(
        "--creative-mode",
        type=str,
        choices=["pragmatic", "expressive", "playful", "automation_dense", "power_user"],
        default="pragmatic",
        help="Creative scoring mode (default: pragmatic)",
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=1,
        help="Number of candidates to generate and score (default: 1)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["auto", "shortcut_only", "shortcut_plus_blueprint"],
        default="auto",
        help="Implementation strategy (default: auto)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DSL text, validation warnings, and timing",
    )

    args = parser.parse_args()

    # ── Resolve input ──

    # Edit mode: load existing .shortcut file
    if args.edit:
        _handle_edit(args)
        return

    if args.dsl_file:
        # Direct DSL compilation mode
        dsl_path = Path(args.dsl_file)
        if not dsl_path.exists():
            print(f"Error: DSL file not found: {args.dsl_file}", file=sys.stderr)
            sys.exit(1)
        dsl_text = dsl_path.read_text()
        _compile_dsl(dsl_text, args)
        return

    if args.prompt is None:
        parser.print_help()
        sys.exit(1)

    if args.prompt == "-":
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("Error: no input on stdin", file=sys.stderr)
            sys.exit(1)
    else:
        prompt = args.prompt

    # ── LLM generation mode ──

    from orchestrator import Orchestrator, ClaudeBackend, LocalBackend

    # Build backend based on engine choice
    backend = None
    if args.engine == "local":
        if not args.model_path:
            print("Error: --model-path is required when --engine=local", file=sys.stderr)
            sys.exit(1)
        backend = LocalBackend(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            use_grammar=False,
            never_grammar=args.no_grammar,
            chat_template=args.chat_template,
        )

    try:
        orch = Orchestrator(backend=backend)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    engine_label = args.engine.capitalize()
    print(f"\nShortcutForge: Generating shortcut ({engine_label})...\n", flush=True)
    t0 = time.monotonic()

    if args.dsl_only:
        # Generate DSL only, don't compile
        result = orch.generate(
            prompt,
            model=args.model,
            max_retries=args.max_retries,
            output_dir=args.output_dir,
            sign=not args.no_sign,
            auto_import=args.auto_import,
            candidate_count=args.candidate_count,
            creative_mode=args.creative_mode,
            implementation_strategy=args.strategy,
        )
        if result.dsl_text:
            print(result.dsl_text)
        if result.errors:
            print(f"\nErrors:", file=sys.stderr)
            for e in result.errors:
                print(f"  - {e}", file=sys.stderr)
        return

    result = orch.generate(
        prompt,
        model=args.model,
        max_retries=args.max_retries,
        output_dir=args.output_dir,
        sign=not args.no_sign,
        auto_import=args.auto_import,
        on_stage_update=_print_stage,
        candidate_count=args.candidate_count,
        creative_mode=args.creative_mode,
        implementation_strategy=args.strategy,
    )

    elapsed = time.monotonic() - t0
    print()

    if result.success:
        if result.signed_path:
            print(f"  \u2713 Signed:  {result.signed_path}")
        elif result.shortcut_path:
            print(f"  \u2713 Saved:   {result.shortcut_path}")
        if result.imported:
            print(f"  \u2713 Imported into Shortcuts.app")
        print(f"  \u2713 Total time: {elapsed:.1f}s ({result.attempts} attempt(s))")
    else:
        print(f"  \u2717 Generation failed after {result.attempts} attempt(s)", file=sys.stderr)
        for e in result.errors:
            print(f"    - {e}", file=sys.stderr)
        sys.exit(1)

    # Verbose output
    if args.verbose:
        print()
        if result.dsl_text:
            print("  --- Generated DSL ---")
            for line in result.dsl_text.split("\n"):
                print(f"  {line}")
            print("  --- End DSL ---")
        if result.warnings:
            print(f"\n  Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"    - {w}")
        # Phase 3 metadata
        if result.scenario_profile != "default":
            print(f"\n  Scenario: {result.scenario_profile}")
        if result.architecture_decision != "shortcut_only":
            print(f"  Architecture: {result.architecture_decision}")
        if result.creativity_score is not None:
            print(f"  Creativity: {result.creativity_score:.2f}")
        if result.candidates_generated > 1:
            print(f"  Candidates: {result.candidates_valid}/{result.candidates_generated} valid")

        print(f"\n  Stages:")
        for s in result.stages:
            duration = f" ({s.duration_ms}ms)" if s.duration_ms else ""
            print(f"    {s.stage:12s} {s.status:8s}{duration}  {s.message}")


def _handle_edit(args):
    """Handle --edit mode: load, view, modify, or recompile a .shortcut file."""
    from plist_to_dsl import shortcut_file_to_dsl_safe

    edit_path = Path(args.edit)
    if not edit_path.exists():
        print(f"Error: .shortcut file not found: {args.edit}", file=sys.stderr)
        sys.exit(1)

    if not edit_path.suffix == ".shortcut":
        print(f"Error: Expected .shortcut file, got: {edit_path.suffix}", file=sys.stderr)
        sys.exit(1)

    # Decompile .shortcut → DSL
    print(f"\nShortcutForge: Loading {edit_path.name}...\n", flush=True)
    dsl_text, error = shortcut_file_to_dsl_safe(str(edit_path))
    if error:
        print(f"Error decompiling: {error}", file=sys.stderr)
        sys.exit(1)

    assert dsl_text is not None

    print(f"  Decompiled: {len(dsl_text.splitlines())} lines of DSL", flush=True)

    # --show-dsl: print and exit
    if args.show_dsl:
        print(f"\n--- DSL from {edit_path.name} ---")
        print(dsl_text)
        print(f"--- End DSL ---")
        return

    # --dsl-file: replace DSL with provided file, then compile
    if args.dsl_file:
        dsl_path = Path(args.dsl_file)
        if not dsl_path.exists():
            print(f"Error: DSL file not found: {args.dsl_file}", file=sys.stderr)
            sys.exit(1)
        dsl_text = dsl_path.read_text()
        print(f"  Replaced DSL with: {args.dsl_file}", flush=True)
        _compile_dsl(dsl_text, args)
        return

    # --modify: send original DSL + instruction to LLM for modification
    if args.modify:
        from orchestrator import Orchestrator, LocalBackend
        from dsl_linter import lint_dsl

        print(f"  Modification: {args.modify}", flush=True)

        # Build backend
        backend = None
        if args.engine == "local":
            if not args.model_path:
                print("Error: --model-path is required when --engine=local", file=sys.stderr)
                sys.exit(1)
            backend = LocalBackend(
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                use_grammar=False,
                never_grammar=args.no_grammar,
                chat_template=args.chat_template,
            )

        try:
            orch = Orchestrator(backend=backend)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Build modification prompt that includes the original DSL
        modify_prompt = (
            f"Here is an existing Apple Shortcut in DSL format:\n\n"
            f"```\n{dsl_text}\n```\n\n"
            f"Please modify this shortcut according to the following instruction:\n"
            f"{args.modify}\n\n"
            f"Output the complete modified shortcut in ShortcutDSL format."
        )

        engine_label = args.engine.capitalize()
        print(f"\n  Modifying with {engine_label}...\n", flush=True)
        t0 = time.monotonic()

        result = orch.generate(
            modify_prompt,
            model=args.model,
            max_retries=args.max_retries,
            output_dir=args.output_dir,
            sign=not args.no_sign,
            auto_import=args.auto_import,
            on_stage_update=_print_stage,
        )

        elapsed = time.monotonic() - t0
        print()

        if result.success:
            if result.signed_path:
                print(f"  ✓ Modified & signed:  {result.signed_path}")
            elif result.shortcut_path:
                print(f"  ✓ Modified & saved:   {result.shortcut_path}")
            if result.imported:
                print(f"  ✓ Imported into Shortcuts.app")
            print(f"  ✓ Total time: {elapsed:.1f}s ({result.attempts} attempt(s))")
        else:
            print(f"  ✗ Modification failed after {result.attempts} attempt(s)", file=sys.stderr)
            for e in result.errors:
                print(f"    - {e}", file=sys.stderr)
            sys.exit(1)

        if args.verbose and result.dsl_text:
            print()
            print("  --- Modified DSL ---")
            for line in result.dsl_text.split("\n"):
                print(f"  {line}")
            print("  --- End DSL ---")
        return

    # No action specified for edit mode
    print(f"\nError: --edit requires --show-dsl, --modify, or --dsl-file", file=sys.stderr)
    sys.exit(1)


def _compile_dsl(dsl_text: str, args):
    """Compile DSL text directly (no LLM)."""
    from orchestrator import Orchestrator

    # For compile_dsl mode, we don't need an API key
    # Create a minimal orchestrator with a dummy key check bypass
    try:
        orch = Orchestrator()
    except ValueError:
        # No API key — that's OK for direct compilation
        # We'll call the pipeline directly
        pass

    # Direct pipeline: parse → validate → compile → deliver
    from dsl_parser import parse_dsl
    from dsl_validator import validate_ir
    from dsl_bridge import compile_ir

    print(f"\nShortcutForge: Compiling DSL...\n", flush=True)

    # Parse
    print(f"  [1/4] Parsing...", end=" ", flush=True)
    try:
        ir = parse_dsl(dsl_text)
        print(f"done (\"{ir.name}\", {ir.action_count()} actions)")
    except Exception as e:
        print(f"FAILED")
        print(f"\n  Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate
    print(f"  [2/4] Validating...", end=" ", flush=True)
    validation = validate_ir(ir)
    if validation.errors:
        print(f"FAILED ({len(validation.errors)} error(s))")
        for e in validation.errors:
            print(f"    - Line {e.line_number}: [{e.category}] {e.message}", file=sys.stderr)
        sys.exit(1)
    warn_note = f" ({len(validation.warnings)} warning(s))" if validation.warnings else ""
    print(f"done{warn_note}")

    # Compile
    print(f"  [3/4] Compiling...", end=" ", flush=True)
    try:
        shortcut = compile_ir(ir)
        print(f"done ({len(shortcut.actions)} actions)")
    except Exception as e:
        print(f"FAILED")
        print(f"\n  Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Deliver
    print(f"  [4/4] Delivering...", end=" ", flush=True)
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        delivery = shortcut.deliver(
            output_dir=args.output_dir,
            sign=not args.no_sign,
            auto_import=args.auto_import,
        )
        print(f"done")
    except Exception as e:
        print(f"FAILED")
        print(f"\n  Error: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    if delivery.get("signed"):
        print(f"  \u2713 Signed:  {delivery['signed']}")
    elif delivery.get("unsigned"):
        print(f"  \u2713 Saved:   {delivery['unsigned']}")
    if delivery.get("imported"):
        print(f"  \u2713 Imported into Shortcuts.app")

    if args.verbose:
        if validation.warnings:
            print(f"\n  Warnings ({len(validation.warnings)}):")
            for w in validation.warnings:
                print(f"    - {w}")


if __name__ == "__main__":
    main()
