# ShortcutForge — Claude Code Guide

## What This Is

ShortcutForge compiles natural language into signed, installable Apple Shortcuts (`.shortcut` files).
LLM output is treated as intermediate code and run through a real compiler: linting → parsing → validation → static analysis → plist compilation → signing.

The current focus is **Balanced Sashimi**: a hybrid continuous-ternary architecture for domain-constrained program synthesis. See `research/docs/BALANCED_SASHIMI_RESEARCH.md` for theory and `research/docs/BALANCED_SASHIMI_PLAN.md` for the operational plan.

## Project Layout (Post-Reorg)

```
apple-shortcuts/
├── src/           # Core compiler pipeline (21 modules)
├── cli/           # Entry points (shortcutforge.py, server.py, mcp_server.py)
├── tests/         # All test suites (22 files)
├── training/      # ML training, eval, distillation (12 files)
├── tools/         # One-off utilities and scrapers (12 files)
├── research/      # Balanced Sashimi research project (new)
├── references/    # Grammars, catalogs, scenario packs
├── training_data/ # JSONL datasets, baselines
├── configs/       # Model profiles
├── models/        # LoRA adapters
├── downloaded/    # Shortcut corpus (1,772 files)
├── output/        # Generated .shortcut files
└── docs/          # Documentation
```

## Project Root

All paths below are relative to `/Users/rohanvinaik/apple-shortcuts/` unless absolute.

## Architecture At a Glance

```
Prompt → [Execution Planner] → [Architecture Reasoner] → [Scenario Profile]
       → LLM Generation (Claude API or local MLX)
       → DSL Linter (6-phase canonicalization)
       → Lark LALR(1) Parser → ShortcutIR
       → Semantic Validator (strict + permissive, 615-action catalog)
       → Simulation Harness (7 static analyses)
       → Creative Scoring
       → Compiler Bridge → Plist Compiler → Signing → .shortcut
```

## Key Commands

### Generate a shortcut
```bash
# Via Claude API
python cli/shortcutforge.py "Set a 5-minute timer" -v

# Via local MLX model
python cli/shortcutforge.py "Toggle DND" \
  --engine local \
  --model-path mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path models/baseline-v1-mlx -v
```

### Evaluate a model (frozen eval set, 100 examples)
```bash
python training/evaluate_model.py \
  --model-path mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path models/baseline-v1-mlx \
  -v --by-domain --by-complexity
```
Output: prints per-example results + summary metrics. Writes `training_data/eval_results.json`.

Add `--log-distillation` to also write `training_data/distillation_log.jsonl` with raw→canonicalized pairs.

Add `--snapshot` to freeze current metrics as the regression baseline.

### Scenario benchmark (rubric-scored, 8 packs)
```bash
# Single scenario
python training/evaluate_scenario.py \
  --scenario references/scenario_packs/health_logger/ \
  --model-path mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path models/baseline-v1-mlx

# All 8 scenario packs (reference DSL scoring, no generation)
python training/evaluate_scenario.py --all-scenarios --score-reference
```

### Regression gate
```bash
python training/check_regression.py -v
# exit 0 = PASS, exit 1 = FAIL
# compares training_data/eval_results.json against training_data/baseline_snapshot.json
```

### Run all tests (376 tests, 16 suites)
```bash
cd /Users/rohanvinaik/apple-shortcuts
python -m pytest tests/test_*.py -v 2>/dev/null || \
  for f in tests/test_*.py; do echo "=== $f ==="; python "$f"; done
```
Tests are standalone scripts (unittest), not pytest-based. Run them individually:
```bash
python tests/test_dsl_linter.py        # 34 tests
python tests/test_orchestrator.py      # 25 tests
python tests/test_dsl_validator.py     # 9 tests
python tests/test_compiler_unit.py     # 24 tests
python tests/test_macro_expander.py    # 58 tests
python tests/test_simulation_harness.py # 25 tests
python tests/test_snippet_extractor.py # 48 tests
python tests/test_contract_validator.py # 20 tests
python tests/test_execution_planner.py # 15 tests
# ... and more (see tests/test_*.py)
```

### Build training data from shortcut corpus
```bash
python training/build_training_data.py -v
# Input: downloaded/ (1,772 .shortcut files) + references/
# Output: training_data/shortcutdsl_train.jsonl, shortcutdsl_eval.jsonl
```

### Curate distillation data
```bash
python training/distillation_curator.py training_data/distillation_log.jsonl \
  --output training_data/distilled_curated.jsonl
```

### Label training data (domain/complexity tags)
```bash
python training/label_training_data.py
```

### Static analysis on a DSL file
```bash
python src/simulation_harness.py references/scenario_packs/health_logger/reference.dsl
```

### Decompile a .shortcut → DSL (reverse compiler)
```bash
python src/plist_to_dsl.py path/to/shortcut.shortcut --validate
```

## Distillation & Training Workflow

This is the core loop. Claude Code should be able to drive this end-to-end.

### Step 1: Generate distillation data from teacher model
Run eval with `--log-distillation` to produce raw→canonicalized pairs:
```bash
python training/evaluate_model.py \
  --model-path mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path models/baseline-v1-mlx \
  --eval-file training_data/shortcutdsl_train_expanded.jsonl \
  --log-distillation -v
```
This writes `training_data/distillation_log.jsonl`.

### Step 2: Curate the distillation data
```bash
python training/distillation_curator.py training_data/distillation_log.jsonl \
  --output training_data/distilled_curated.jsonl
```
Quality gates: must parse + validate + compile. Deduplication by prompt similarity. Scenario balancing.

### Step 3: Prepare training JSONL
Training data format is chat-style JSONL with `messages` array:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "prompt"}, {"role": "assistant", "content": "DSL output"}]}
```
The existing `training_data/shortcutdsl_train_expanded.jsonl` (6,679 examples) is in this format.
Curated distillation data needs conversion to this format before training.

### Step 4: Fine-tune with MLX
Use `mlx_lm.lora` directly:
```bash
# LoRA fine-tune (primary method)
python -m mlx_lm.lora \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --data training_data/ \
  --train \
  --batch-size 4 \
  --lora-layers 16 \
  --iters 1000 \
  --learning-rate 1e-4 \
  --adapter-path models/new-run-v1

# For tiny models (PLAN Phase 3 targets):
# Qwen 0.5B: --model Qwen/Qwen2.5-0.5B-Instruct --batch-size 8 --lora-rank 32
# Llama 1B:  --model meta-llama/Llama-3.2-1B-Instruct --batch-size 4 --lora-rank 32

# Fuse adapter into base model for faster inference
python -m mlx_lm.fuse \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit \
  --adapter-path models/new-run-v1 \
  --save-path models/new-run-v1-fused
```

The `mlx_lm.lora` command expects `train.jsonl` and `valid.jsonl` in the `--data` directory.
You may need to symlink or copy:
```bash
ln -s shortcutdsl_train_expanded.jsonl training_data/train.jsonl
ln -s shortcutdsl_eval.jsonl training_data/valid.jsonl
```

### Step 5: Evaluate the new model
```bash
python training/evaluate_model.py \
  --model-path <base-model> \
  --adapter-path models/new-run-v1 \
  -v --by-domain --by-complexity
```

### Step 6: Check promotion gates (from PLAN.md Phase 4)
The model is promotable if:
- `compile_strict_rate >= 95.0`
- `compile_permissive_rate >= 97.0`
- `runtime_unverified_compile_rate <= 2.0`
- `fallback_rate <= 5.0`
- health logger scenario score `>= 0.85`

Compare against current baseline in `training_data/baseline_snapshot.json`:
- Parse: 93%, Validate strict: 85%, Compile strict: 85%
- Validate permissive: 89%, Compile permissive: 89%

### Step 7: If gates pass, snapshot as new baseline
```bash
python training/evaluate_model.py \
  --model-path <base-model> \
  --adapter-path models/new-run-v1 \
  --snapshot -v
```

## Available Model Adapters

| Adapter | Path | Description |
|---|---|---|
| baseline-v1 | `models/baseline-v1-mlx/` | Current baseline (336MB, LoRA on 8B) |
| mini-v1 | `models/mini-v1-mlx/` | Smaller variant |
| expanded-v1 | `models/expanded-v1-mlx/` | Trained on expanded dataset |
| expanded-v1-step1000 | `models/expanded-v1-step1000-mlx/` | Expanded, 1000 steps |

Default base model: `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`

## Key Data Files

| File | Purpose |
|---|---|
| `training_data/shortcutdsl_train_expanded.jsonl` | 6,679 training examples |
| `training_data/shortcutdsl_eval.jsonl` | 100 frozen eval examples (DO NOT MODIFY) |
| `training_data/baseline_snapshot.json` | Frozen baseline metrics for regression gate |
| `training_data/eval_results.json` | Latest eval run results |
| `training_data/distillation_log.jsonl` | Raw distillation log from eval |
| `references/action_catalog.json` | 615 actions + 659 aliases |
| `references/shortcutdsl.lark` | DSL grammar (Lark LALR) |
| `references/snippet_registry.json` | ~200 micro-patterns for RAG |
| `references/macro_patterns.json` | 31 macro templates |
| `references/scenario_packs/` | 8 benchmark scenarios with rubrics |

## Scenario Packs (8 benchmarks)

`api_pagination_fetcher`, `calendar_triage`, `clipboard_utility`, `file_router`,
`health_logger`, `media_metadata_pipeline`, `morning_routine`, `share_sheet_text_cleaner`

Each pack has: `rubric.json`, `reference.dsl`, optional domain-specific data.

## Module Map (what to read when)

| Task | Start here |
|---|---|
| Understand generation flow | `src/orchestrator.py` (Orchestrator.generate) |
| Debug linter issues | `src/dsl_linter.py` (lint_dsl, 6 phases) |
| Debug parse failures | `src/dsl_parser.py` + `references/shortcutdsl.lark` |
| Debug validation failures | `src/dsl_validator.py` (validate_ir, action catalog) |
| Understand eval metrics | `training/evaluate_model.py` (evaluate function, line ~124) |
| Modify scenario scoring | `training/evaluate_scenario.py` (criterion registry) |
| Understand training data format | `training/build_training_data.py` |
| Curate distillation data | `training/distillation_curator.py` (DistillationCurator class) |
| Local MLX inference | `src/inference.py` (generate_with_timeout) |
| System/user prompts | `src/generate_prompt.py` |
| Static analysis | `src/simulation_harness.py` (7 analyses) |

## MCP Server (if installed)

If the ShortcutForge MCP server is running (`cli/mcp_server.py`), Claude Code has
structured tool access to the pipeline. See the server for available tools.

## Important Conventions

1. **Never modify `training_data/shortcutdsl_eval.jsonl`** — it's the frozen eval set.
2. **Always run regression gate after changes** — `python training/check_regression.py -v`
3. **Scripts expect to be run from the project root** or they resolve paths via `Path(__file__).resolve().parent.parent`.
4. **Tests are standalone unittest scripts**, not pytest. Run individually with `python tests/test_*.py`.
5. **DSL must end with `ENDSHORTCUT`** — the linter adds it if missing, but generation should include it.
6. **Linter version is in `src/dsl_linter.py.__version__`** — currently v2.4.
7. **Permissive mode exists for utility** — strict is the promotion metric, permissive tracks user-visible value.
