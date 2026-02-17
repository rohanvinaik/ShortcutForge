# Balanced Sashimi — Research Module

Hybrid continuous-ternary architecture for domain-constrained program synthesis.

**Theory & Architecture:** `docs/BALANCED_SASHIMI_RESEARCH.md`
**Operational Plan:** `docs/BALANCED_SASHIMI_PLAN.md`
**Experiment Results:** `docs/EXPERIMENT_RESULTS.md`
**Environment Setup:** `ENV_SETUP.md`

## Structure

```
research/
├── configs/
│   └── base.yaml            # Base experiment config (others derived per-experiment)
├── scripts/
│   ├── env_doctor.py         # Environment preflight checker
│   ├── build_typed_ir_data.py   # ✅ DSL → three-tier decomposition (100% rate)
│   ├── build_decoder_vocab.py   # ✅ Tier1/Tier2 vocab extraction (99%/97% coverage)
│   ├── build_negative_bank.py   # ✅ Hard negative bank (19,840 entries)
│   ├── build_ood_prompts.py     # ✅ Balanced OOD prompt set (500+500)
│   └── run_ablation_matrix.py   # ⏳ Phase 4 (not yet implemented)
├── src/
│   ├── contracts.py          # ✅ Typed data contracts (single source of truth)
│   ├── encoder.py            # ✅ Frozen sentence-transformer wrapper (MiniLM-L6-v2)
│   ├── domain_gate.py        # ✅ Binary OOD classifier MLP
│   ├── intent_extractor.py   # ✅ Semantic frame projection
│   ├── bridge.py             # ✅ Information bottleneck (Linear + LayerNorm)
│   ├── ternary_decoder.py    # ✅ STE ternary structural decoder
│   ├── value_filler.py       # ⏳ Phase 3 (not yet implemented)
│   ├── lowering.py           # ✅ Deterministic IR → DSL conversion
│   ├── losses.py             # ✅ Composite loss + UW-SO adaptive weighting
│   ├── data.py               # ✅ Datasets for typed IR + negatives
│   ├── trainer.py            # ✅ Training loop with gradient safety
│   └── evaluate.py           # ✅ Evaluation harness
├── tests/
│   ├── fixtures/
│   │   └── tiny_train.jsonl  # 3 hand-crafted examples for smoke testing
│   ├── test_contracts.py     # Serialization roundtrips
│   ├── test_env_doctor.py    # Preflight check tests
│   ├── test_data_io.py       # JSONL loading + malformed row handling
│   ├── test_ternary.py       # Quantized values {-1,0,+1} + STE finite
│   ├── test_losses.py        # Loss components non-negative and finite
│   ├── test_lowering.py      # Lower-then-parse roundtrip
│   └── test_smoke_e2e.py     # End-to-end: fixture → vocab → model → forward → eval
├── models/                   # Checkpoints (gitignored)
├── requirements.txt          # Pinned dependency versions (generated)
└── ENV_SETUP.md              # Canonical environment setup instructions
```

## Quick Start

```bash
# 1. Environment
uv run python research/scripts/env_doctor.py --strict

# 2. Phase 0: Data prep (run in order)
uv run python research/scripts/build_typed_ir_data.py -v
uv run python research/scripts/build_decoder_vocab.py -v
uv run python research/scripts/build_negative_bank.py -v
uv run python research/scripts/build_ood_prompts.py -v

# 3. Phase 2: Training (dry-run to verify)
uv run python research/src/trainer.py --config research/configs/base.yaml --run-id smoke --dry-run --device cpu

# 4. Phase 2: Full training
uv run python research/src/trainer.py --config research/configs/base.yaml --run-id v1

# 5. Tests
uv run pytest research/tests/ -v
```

## Gates

| Gate | Criteria | Status |
|------|----------|--------|
| S1: Scaffold import | All `research.src.*` modules import cleanly | ✅ |
| S2: CLI | Every `research/scripts/*.py --help` exits 0 | ✅ |
| S3: Tests | All tests pass (0 skipped) | ✅ |
| S4: Env | `env_doctor.py --strict` passes | ✅ |
| P0: Data conversion | ≥95% conversion rate | ✅ 100% (6,779/6,779) |
| P0: Tier1 coverage | ≥98% on eval set | ✅ 99.01% |
| P0: Tier2 coverage | ≥95% on eval set | ✅ 97.53% |
| P0: Negative bank | ≥3,000 entries | ✅ 19,840 |
| P0: OOD prompts | 500 in-domain + 500 OOD | ✅ 1,000 |
| P2: Trainer dry-run | 1 step completes without error | ✅ |

## Not Yet Implemented

- **Phase 3: ValueFiller** (`src/value_filler.py`) — Tier 3 text generation for free-form parameter values. Currently stubbed.
- **Phase 4: Ablation Matrix** (`scripts/run_ablation_matrix.py`) — Systematic ablation experiments. Requires working trainer and evaluation loop.
