# Balanced Sashimi Environment Setup

## Prerequisites
- macOS with Apple Silicon (M1 Max tested)
- `uv` installed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Canonical Bootstrap (One-Time)

```bash
cd /Users/rohanvinaik/apple-shortcuts

# Create project-local venv
uv venv .venv --python 3.11.8

# Install project with research dependencies
uv pip install -e ".[research]"

# Generate lockfile
uv lock
```

## Canonical Sync (After Pull / Dependency Change)

```bash
cd /Users/rohanvinaik/apple-shortcuts

# Sync from lockfile
uv lock --check  # verify lock is fresh
uv pip install -e ".[research]"
```

## Canonical Run Commands

All research commands go through `uv run`:

```bash
# Environment preflight
uv run python research/scripts/env_doctor.py --strict

# Phase 0 data builds
uv run python research/scripts/build_typed_ir_data.py -v
uv run python research/scripts/build_decoder_vocab.py -v
uv run python research/scripts/build_negative_bank.py -v
uv run python research/scripts/build_ood_prompts.py -v

# Training
uv run python research/src/trainer.py --config research/configs/base.yaml --run-id test-v1

# Evaluation
uv run python research/src/evaluate.py --config research/configs/base.yaml --checkpoint path/to/ckpt --eval-file training_data/typed_ir_eval.jsonl

# Tests
uv run python research/tests/test_contracts.py
uv run python research/tests/test_data_io.py
```

## Expected env_doctor Output (Healthy)

```
============================================================
 Balanced Sashimi Environment Doctor
 Health: HEALTHY
 (N ok, 0 warnings, 0 errors)
============================================================
```

## Dependency Model

- **`pyproject.toml`**: Intent/ranges — defines what the project needs.
- **`uv.lock`**: Execution lock — exact versions, committed to git.
- **`research/requirements.txt`**: Generated reference (read-only).
- **`.python-version`**: Pins Python 3.11.8.

Never use global `python` for research work. Always `uv run`.
