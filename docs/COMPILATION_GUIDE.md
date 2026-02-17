# ShortcutForge Compilation Guide

Quick reference for compiling DSL files through the ShortcutForge pipeline.

---

## Small Files (<40KB): Use MCP Tools

```
forge_lint(dsl_text)           → Canonicalize action names, fix structure
forge_validate(dsl_text)       → Parse + validate against 615-action catalog
forge_analyze(dsl_text)        → 7-pass static analysis (variable flow, dead code, etc.)
forge_compile_dsl(dsl_text)    → Full pipeline → signed .shortcut binary
```

Typical workflow:
1. Write DSL incrementally (50-100 lines at a time)
2. `forge_validate` after each chunk to catch errors early
3. `forge_compile_dsl` when complete

## Large Files (>40KB): Direct Python

MCP tool parameters truncate around ~55KB. For large shortcuts, compile via Python:

```python
import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from dsl_linter import lint_dsl
from dsl_parser import parse_dsl
from dsl_validator import validate_ir
from dsl_bridge import compile_ir

with open('path/to/shortcut.dsl') as f:
    dsl = f.read()

# Stage 1: Lint
lint_result = lint_dsl(dsl)
print(f"Lint: {len(lint_result.changes)} changes")

# Stage 2: Parse
ir = parse_dsl(lint_result.text)
print(f"Parsed: {ir.action_count()} actions")

# Stage 3: Validate
result = validate_ir(ir, strict=True)
if not result.is_valid:
    for err in result.errors:
        print(f"  ERROR: {err}")
    raise SystemExit("Validation failed")
for warn in result.warnings:
    print(f"  WARN: {warn}")

# Stage 4: Compile + Sign
sc = compile_ir(ir)
output_path = 'output/My_Shortcut_signed.shortcut'
sc.save_and_sign(output_path)
print(f"Compiled: {len(sc.actions)} actions, saved to {output_path}")
```

Run from project root:
```bash
cd /Users/rohanvinaik/apple-shortcuts
python3 -c "$(cat above_script.py)"
```

## API Reference (Key Classes)

| Module | Function/Class | Returns | Notes |
|--------|---------------|---------|-------|
| `dsl_linter` | `lint_dsl(text)` | `LintResult` | `.text`, `.changes`, `.was_modified` |
| `dsl_parser` | `parse_dsl(text)` | `ShortcutIR` | `.action_count()`, `.statements` |
| `dsl_validator` | `validate_ir(ir, strict=)` | `ValidationResult` | `.is_valid`, `.errors`, `.warnings` |
| `dsl_bridge` | `compile_ir(ir)` | `Shortcut` | `.actions`, `.save_and_sign(path)` |

## Common Gotchas

1. **`lint_dsl()` returns an object, not a tuple.** Use `lint_result.text`.
2. **`compile_ir()` is in `dsl_bridge.py`**, not `shortcuts_compiler.py`.
3. **`Shortcut.save_and_sign(path)`** is the one-call method. Don't use `.sign()` directly.
4. **Module imports need `sys.path`** pointed at the `scripts/` directory.
5. **Always run from project root** so relative paths to `references/` resolve correctly.

## Metrics to Expect

| DSL Lines | Parsed Actions | Compiled Actions | Binary Size |
|-----------|---------------|-----------------|-------------|
| ~50 | ~30 | ~50 | ~22KB |
| ~200 | ~120 | ~200 | ~40KB |
| ~500 | ~300 | ~500 | ~65KB |
| ~1,000 | ~600 | ~900 | ~100KB |
| ~1,288 | ~661 | ~1,085 | ~119KB |

Compiled action count is typically 1.4-1.7x parsed action count due to control flow overhead (IF/ENDIF, MENU/CASE/ENDMENU each add mode markers).
