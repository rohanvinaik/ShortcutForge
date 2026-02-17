# Balanced Sashimi: Operational Research Plan

**Version:** 1.0
**Date:** February 16, 2026
**Corresponding theory document:** `BALANCED_SASHIMI_RESEARCH.md`
**Results ledger:** `EXPERIMENT_RESULTS.md`

---

## Overview

This document specifies the exact sequence of work, tooling, compute decisions, and stop/go criteria for implementing and evaluating the Balanced Sashimi architecture. Each phase is self-contained and produces a working, evaluable artifact.

**Time horizon:** 4–8 weeks for Phase 1–4. Phase 5+ is contingent on results.

---

## Phase 0: Data Preparation and Tooling (Week 1)

*Goal: Build the data artifacts and infrastructure needed before any model training.*

### 0.1 Construct Typed IR Training Data

**What**: Convert existing 6,679 training examples from DSL text targets to three-tier structural representations.

**How**:
```python
# research/scripts/build_typed_ir_data.py
# For each training example:
#   1. Parse DSL target with dsl_parser.py → ShortcutIR
#   2. Walk IR tree, emit Tier 1 tokens (actions, control flow, vars)
#   3. For each action, emit Tier 2 tokens (typed param keys + enum values)
#   4. Extract Tier 3 values (free strings, numbers)
#   5. Write structured JSONL
```

**Output**: `training_data/typed_ir_train.jsonl`, `training_data/typed_ir_eval.jsonl`

**Validation**: Round-trip test — lower typed IR back to DSL, re-parse, verify identical ShortcutIR.

**Compute**: Local, CPU-only, <1 hour.

**Stop/go**: If >5% of training examples fail to convert (due to parsing edge cases), fix the conversion before proceeding. The typed IR must be a lossless representation of the existing DSL.

### 0.2 Build Tier 1 + Tier 2 Vocabularies

**What**: Enumerate the concrete token vocabularies for the structural decoder.

**How**:
```python
# research/scripts/build_decoder_vocab.py
# Tier 1: Scan all parsed training data for:
#   - Structural keywords (SHORTCUT, IF, ACTION, SET, etc.)
#   - Canonical action short names (deduplicated from catalog)
#   - Variable patterns ($VAR_1...$VAR_N, abstract)
#   - Handle references (@prev, @input, @item, @index, @date)
#   - Special tokens (BOS, EOS, PAD, SEP_TIER)
#
# Tier 2: For each action identity in Tier 1:
#   - Extract known parameter keys from param_schemas.json
#   - Extract known enum values from catalog + training data
#   - Build per-action conditional vocabulary
```

**Output**: `references/tier1_vocab.json`, `references/tier2_vocab/` (one file per action)

**Validation**: Coverage check — what % of eval set actions and parameters are covered?

**Compute**: Local, CPU-only, <30 minutes.

### 0.3 Build Hard Negative Bank (Seed Version)

**What**: Construct initial hard negative triples from existing distillation logs.

**How**:
```python
# research/scripts/build_negative_bank.py
# Source 1: Existing distillation_log.jsonl
#   - For each entry where raw ≠ canonicalized:
#     - positive = canonicalized (post-linter) → convert to typed IR
#     - negative = raw (pre-linter) → convert to typed IR (may partially fail)
#     - error_tags = linter repair kinds applied
#
# Source 2: Synthetic perturbations of training data
#   - Action substitution (swap action for similar-name incorrect action)
#   - Structural corruption (remove ENDIF, duplicate CASE)
#   - Parameter mutation (change enum value to incorrect one)
#
# Output: training_data/hard_negative_bank.jsonl
# Schema: {"prompt": ..., "positive_ir": [...], "negative_ir": [...],
#           "error_tags": [...], "source": "linter|synthetic|adversarial"}
```

**Target**: ≥3,000 triples for initial training.

**Compute**: Local, CPU-only, <2 hours.

### 0.4 Build OOD Prompt Set

**What**: Balanced in-domain / out-of-domain prompt dataset for domain gate training.

**How**:
- **In-domain** (500+ examples): Sample prompts from training data.
- **Out-of-domain** (500+ examples): Construct manually + generate via Claude API:
  - General knowledge questions ("What is the capital of France?")
  - Math/science requests ("Solve this integral")
  - Non-Shortcuts coding ("Write a Python sorting algorithm")
  - Philosophical/open-ended ("What is consciousness?")
  - Adversarial near-domain ("Build an Android widget for...", "Write a SwiftUI view...")

**Output**: `references/ood_prompt_set.jsonl`
**Schema**: `{"prompt": "...", "label": "in_domain" | "ood", "category": "..."}`

**Compute**: Local + Claude API calls for OOD generation (~$2–5).

### 0.5 Set Up PyTorch Training Infrastructure

**What**: Create the project scaffolding for custom PyTorch training.

**Directory structure** (under `research/`, per post-reorg layout):
```
research/
├── models/           # Saved model checkpoints
├── configs/          # YAML experiment configs
├── src/
│   ├── encoder.py        # Sentence transformer wrapper
│   ├── domain_gate.py    # Binary classification head
│   ├── intent_extractor.py  # Semantic frame extraction
│   ├── bridge.py         # Information bottleneck
│   ├── ternary_decoder.py   # Ternary structural decoder + STE
│   ├── value_filler.py   # Tier 3 text generation
│   ├── lowering.py       # Deterministic IR → DSL conversion
│   ├── losses.py         # Composite loss with adaptive weighting
│   ├── data.py           # DataLoader for typed IR + negatives
│   ├── trainer.py        # Custom training loop
│   └── evaluate.py       # Evaluation bridge to ShortcutForge pipeline
├── scripts/
│   ├── build_typed_ir_data.py
│   ├── build_decoder_vocab.py
│   ├── build_negative_bank.py
│   └── build_ood_prompts.py
├── tests/
│   ├── test_ternary.py   # Unit tests for STE quantization
│   ├── test_losses.py    # Unit tests for each loss component
│   ├── test_lowering.py  # Round-trip lowering tests
│   └── test_data.py      # Data loading tests
└── requirements.txt
```

**Dependencies**:
```
torch>=2.0
sentence-transformers
pyyaml
```

**Compute**: Local, no GPU needed for setup.

**Stop/go for Phase 0**: All data artifacts pass validation. Round-trip tests pass. PyTorch scaffolding has passing unit tests for data loading and ternary quantization. Estimated: **3–5 days of work.**

---

## Phase 1: Component-Level Validation (Week 2)

*Goal: Validate each architectural component independently before integration.*

### 1.1 Encoder Evaluation (EXP-1.1)

**What**: Measure whether all-MiniLM-L6-v2 produces good semantic representations for our prompt distribution.

**Protocol**:
1. Encode all 6,679 training prompts.
2. For prompts with identical DSL targets, compute intra-class cosine similarity.
3. For prompts with different targets, compute inter-class similarity.
4. Compute silhouette scores per domain cluster (health, api, file, etc.).
5. Visualize with UMAP, colored by domain.

**Decision criteria**:
- Separation ratio (intra/inter) > 1.5: encoder sufficient as-is.
- Separation ratio 1.0–1.5: needs domain fine-tuning (contrastive, ~1 hour local training).
- Separation ratio < 1.0: consider different encoder.

**Compute**: Local CPU, <30 minutes.

### 1.2 Domain Gate Validation (EXP-1.2)

**What**: Train and evaluate three gate architectures.

**Protocol**:
1. Split OOD prompt set: 80% train, 20% test.
2. Train: (a) linear probe, (b) 2-layer MLP, (c) SVM over 384-d embeddings.
3. Evaluate precision, recall, F1 on held-out test set.

**Decision criteria**:
- Adopt simplest architecture with ≥99% precision and ≥95% recall.
- If linear probe suffices, use it (most interpretable).

**Compute**: Local CPU, <10 minutes per variant.

### 1.3 Vocabulary Coverage Analysis (EXP-1.3, EXP-1.4)

**What**: Verify that Tier 1 + Tier 2 vocabularies cover the frozen eval set.

**Protocol**:
1. Parse all 100 eval examples.
2. For each, extract Tier 1 and Tier 2 tokens.
3. Report % of tokens present in constructed vocabularies.
4. Identify any uncovered tokens (these need vocabulary additions or fall to Tier 3).

**Target**: ≥99% coverage for Tier 1, ≥95% for Tier 2 enum values.

**Compute**: Local CPU, <5 minutes.

**Stop/go for Phase 1**: Encoder separation ratio ≥1.5 (or fine-tuned to ≥1.5). Domain gate ≥99% precision. Vocabulary coverage ≥99%/95%. Estimated: **2–3 days.**

---

## Phase 2: Ternary Decoder Prototype (Weeks 2–3)

*Goal: Build and validate the core ternary structural decoder on a simplified task.*

### 2.1 Toy Problem Validation

**What**: Before training on full data, validate the ternary decoder + STE training on a toy structured prediction task.

**Protocol**:
1. Create a toy dataset: 500 prompts → 5-token structural sequences (from a vocabulary of 50 tokens).
2. Train a small ternary decoder (1 layer, 64-dim) with STE.
3. Verify it achieves >95% accuracy on the toy task.
4. Compare: continuous weights vs. ternary weights on the same task.

**Purpose**: Catches fundamental bugs in STE implementation, loss computation, and decoding before scaling up.

**Compute**: Local, <30 minutes.

### 2.2 Tier 1 Decoder (Structural Skeleton Only)

**What**: Train the ternary decoder to emit Tier 1 structural tokens only (action sequence + control flow), given oracle encoder representations.

**Configuration**:
- Input: Gold-standard encoder output (computed from correct prompts, not learned).
- Output: Tier 1 token sequences.
- Weight regime: Full ternary.
- Loss: L_ce only (no negatives yet — establish the positive baseline first).
- Bottleneck: 128-dim (mid-range, will sweep later).

**Training**:
- Data: 6,679 typed IR training examples (Tier 1 only).
- Iterations: 1000.
- Batch size: 16.
- Learning rate: 1e-4 with cosine decay.
- Hardware: Local M1 Max.
- Estimated time: 2–4 hours.

**Evaluation**:
- Tier 1 token accuracy (exact match per position).
- Tier 1 sequence accuracy (exact match full sequence).
- After lowering + Tier 2/3 fill (using oracle values): compile rate on eval set.

**Decision criteria**:
- Tier 1 sequence accuracy ≥ 80%: Proceed with full architecture.
- 60–80%: Investigate — is the bottleneck too narrow? Architecture too small?
- <60%: Ternary decoder may be fundamentally insufficient. Consider partial ternary or more decoder capacity.

**Compute**: Local, 2–4 hours.

### 2.3 Tier 1+2 Decoder (With Conditional Parameters)

**What**: Extend the decoder to emit Tier 2 tokens conditioned on Tier 1 choices.

**Additional complexity**: The Tier 2 vocabulary is dynamic — it depends on which action was selected in Tier 1. This requires either:
- (a) A large unified vocabulary with masking (mask invalid Tier 2 tokens based on Tier 1 context).
- (b) Per-action Tier 2 heads (separate small output layers per action).
- (c) Pointer mechanism (point into the action-specific vocab based on Tier 1 selection).

**Decision point**: Which approach to implement. Start with (a) — simplest implementation — and switch to (b) or (c) if accuracy is insufficient.

**Compute**: Local, 3–6 hours.

**Stop/go for Phase 2**: Tier 1+2 decoder achieves ≥70% sequence accuracy with ternary weights. Compile rate (with oracle Tier 3 values) ≥80%. Estimated: **5–7 days.**

---

## Phase 3: Full Pipeline Integration (Week 3–4)

*Goal: Connect all components into an end-to-end trainable system.*

### 3.1 End-to-End Wiring

**What**: Connect encoder → domain gate → intent extractor → bridge → decoder → value filler → lowering → existing pipeline.

**Key integration points**:
1. Encoder output feeds both domain gate and intent extractor (shared representation).
2. Bridge receives intent extractor output, emits bottleneck embedding.
3. Decoder receives bottleneck embedding, emits Tier 1+2 tokens.
4. Value filler receives Tier 1+2 skeleton + encoder representation, fills Tier 3.
5. Lowering converts structured output to DSL text.
6. DSL text enters existing pipeline (linter → parser → validator → compiler).

**Validation**: End-to-end forward pass on a single training example. Verify gradient flow from composite loss back through all components (gradient health check: no NaN, no vanishing, no explosion).

### 3.2 Full Composite Loss Training

**Configuration** (first full run):
- All components connected.
- Composite loss: L_ce + L_margin + L_repair with adaptive weighting.
- L_ood: Separate head, separate optimizer.
- Decoder: Full ternary.
- Bottleneck: 128-dim.
- Negative learning: On (hard negatives from bank).
- Training data: Full typed IR train set + negative bank.
- Iterations: 1000.
- Hardware: Local M1 Max.
- Estimated time: 4–8 hours.

**Evaluation**: Full ShortcutForge eval pipeline on frozen 100 examples.

### 3.3 First A/B Comparison

**Compare against baseline (BL-1)**:
- Compile strict rate: target ≥ baseline (85%) even on first full run.
- If significantly below baseline: diagnose which component is failing (encoder understanding? decoder accuracy? lowering bugs?).

**Stop/go for Phase 3**: End-to-end system runs without crashes. First compile rate ≥ 70% (allowing for initial training suboptimality). If < 50%, there's a fundamental architecture bug. Estimated: **5–7 days.**

---

## Phase 4: Ablation Matrix (Weeks 4–6)

*Goal: Systematic evaluation of architectural choices.*

### 4.1 Priority Ablations (Must Run)

These 8 runs answer the core research questions:

| Run | Weight Regime | Negative Learning | Output Target | Purpose |
|---|---|---|---|---|
| 4.1.1 | Continuous | Off | Typed IR | Baseline: architecture w/o ternary or negatives |
| 4.1.2 | Continuous | Full | Typed IR | Isolate negative learning contribution |
| 4.1.3 | Full ternary | Off | Typed IR | Isolate ternary contribution |
| 4.1.4 | Full ternary | Full | Typed IR | Full Balanced Sashimi |
| 4.1.5 | Full ternary | Full | DSL text | Typed IR vs DSL output comparison |
| 4.1.6 | Partial ternary (decoder only) | Full | Typed IR | Full vs partial ternary |
| 4.1.7 | Full ternary | Full | Typed IR (bottleneck=64) | Bottleneck sensitivity |
| 4.1.8 | Full ternary | Full | Typed IR (bottleneck=256) | Bottleneck sensitivity |

**Compute estimate**: 8 runs × 4–8 hours = 32–64 compute hours. **Run 2–3 concurrent on M1 Max** (6–10GB each, one GPU device shared): **12–24 hours wall-clock, $0.** Early-exit runs that fail fast (<70% Tier 1 accuracy by step 200) to free slots.

### 4.2 Secondary Ablations (If Phase 4.1 Shows Promise)

- Full bottleneck sweep: {32, 64, 128, 256, 384}
- Encoder fine-tuning: frozen vs. domain contrastive
- Loss weighting: adaptive vs. fixed {1, 0.1, 0.1} vs. fixed {1, 0.5, 0.5}
- Negative source: linter-only vs. linter+synthetic vs. linter+synthetic+adversarial

### 4.3 Track Comparison (If Architecture Validated)

- **Track A**: Best configuration from 4.1, trained on teacher-distilled data.
- **Track B**: Best configuration from 4.1, trained on gold training data directly.
- **Track C**: Best configuration from 4.1, trained from scratch (no pretrained encoder).

**Stop/go for Phase 4**: At least one configuration meets or exceeds baseline compile strict rate (85%). If no configuration reaches 75%, revisit fundamental architecture assumptions. Estimated: **10–14 days.**

---

## Phase 5: Refinement and Interpretability (Weeks 6–8)

*Contingent on Phase 4 showing at least one promising configuration.*

### 5.1 Hyperparameter Optimization

Fine-tune the best configuration from Phase 4:
- Learning rate sweep
- Training iterations (early stopping analysis)
- Batch size effects
- Ternarization schedule tuning

### 5.2 Interpretability Analysis

For the best ternary configuration:
1. Extract ternary weight patterns for top-20 actions.
2. Identify which input features (from encoder) each action attends to.
3. Visualize bottleneck representations (UMAP by domain, by compile outcome).
4. Analyze failure modes: localize failures to specific modules.

### 5.3 Promotion Assessment

Does any configuration meet the promotion gates?

```yaml
compile_strict_rate: ≥95.0%
compile_permissive_rate: ≥97.0%
runtime_unverified_compile_rate: ≤2.0%
ood_false_accept_rate: ≤1.0%
hard_negative_separability: ≥2.0 nats
inference_latency_p95_ms: ≤2000ms
model_total_params_M: ≤500M
```

If yes: snapshot as new baseline, integrate into orchestrator as production option.
If no: document findings, identify specific bottlenecks, decide whether to iterate or declare research conclusions.

---

## Compute Budget Summary

**Design constraint: $0 local-first.** The M1 Max (64GB unified memory, 32 GPU cores) is the primary and ideally only platform. Cloud is a last resort, CPU-only if used.

### Why $0 is realistic

$0 is plausible because each individual run fits comfortably in local memory and can be staged/pruned — not because of massive parallelism.

**Honest memory math for a 200M-param decoder with Adam (fp32 throughout STE training):**
- Weights (fp32 shadow for STE): ~0.8 GB
- Gradients: ~0.8 GB
- Adam optimizer states (m, v): ~1.6 GB
- Subtotal before activations: ~3.2 GB
- Activations, buffers, non-decoder modules (encoder, gate, bridge): ~3–7 GB depending on batch size
- **Realistic per-run total: 6–10 GB**

Note: ternary quantization helps *inference* memory (weights pack to ~50MB at 2 bits), but during STE training the backward pass operates on full fp32 shadow weights. Don't confuse inference footprint with training footprint.

**Parallelism is limited.** The M1 Max has one GPU device; multiple MPS processes contend for the same compute. Memory may fit 4+ runs but throughput does not scale linearly. Practical sweet spot is **2–3 concurrent runs**, sometimes 4 if models are small and underutilizing GPU. For Phase 4's 8 priority runs, expect 2–3 serial batches of 2–3 concurrent runs.

| Phase | Platform | Compute Hours | Wall-Clock (est.) | Cost |
|---|---|---|---|---|
| Phase 0: Data prep | Local CPU | 8–12h | 8–12h | $0 |
| Phase 1: Component validation | Local CPU | 2–4h | 2–4h | $0 |
| Phase 2: Decoder prototype | Local MPS | 8–16h | 8–16h | $0 |
| Phase 3: Full integration | Local MPS | 8–16h | 8–16h | $0 |
| Phase 4: Ablation matrix (8 runs) | Local MPS, 2–3 concurrent | 32–64h | **12–24h** | $0 |
| Phase 5: Refinement | Local MPS | 20–40h | 20–40h | $0 |
| **Total** | | **78–152h compute** | **58–112h wall-clock** | **$0** |

The real savings come from **early pruning**: if Phase 2 stop/go kills a configuration direction, we skip entire ablation branches. And runs that fail fast (< 70% Tier 1 accuracy by step 200) can be terminated early, freeing slots for the next experiment.

### Performance engineering (baked in from Phase 0)

1. **MPS (Metal) first.** All standard PyTorch ops (linear, embedding, softmax, STE quantization = `round(clip(W/γ, -1, 1))`) are MPS-compatible. Train on `device="mps"` by default.
2. **Gradient health checks every 100 steps.** MPS has a known "silent NaN" issue — less strict about float exceptions than CUDA. We add explicit NaN/inf checks on loss and gradients to catch this early.
3. **`torch.compile` where it helps.** MPS compile support has improved for simple op patterns (no exotic attention). Benchmark per-model — gains are workload-dependent. Profile first, compile the actual bottleneck.
4. **Pack ternary weights for inference only.** At inference time, ternary weights ({-1,0,+1}) can be packed 2 bits per weight. 200M params → ~50MB packed. But note: STE training still uses fp32 float ops with quantized values, not packed integer kernels. True integer-MAC ternary requires custom kernels/runtime (out of scope unless inference latency becomes a blocker).
5. **Profile before optimizing.** Data prep (Phase 0) is I/O-bound, not compute-bound — Numba/JIT won't help there. Optimization effort goes where profiling says it should, not pre-emptively.

### Cloud fallback (CPU-only, if needed)

If any single ablation run exceeds 12 hours locally, or if the ablation phase would exceed 2 weeks wall-clock:
- **CPU-only cloud instances** (Hetzner dedicated, ~€0.05/core-hour; or AWS c7a spot instances).
- Note: "ternary matmuls are integer ops" is true at inference with custom kernels, but during STE training the compute is standard fp32 — so cloud CPU is not inherently cheaper per-FLOP than GPU. The cost advantage is simply that CPU spot instances are dirt cheap compared to GPU instances.
- Estimated cloud budget if needed: **$10–30** for a burst of parallelism.
- No GPU instances. No A100s. The models are small enough that the per-run cost is pennies on cloud CPU.

---

## Risk Checkpoints

| After Phase | Risk Check | Action if Failed |
|---|---|---|
| 0 | Typed IR conversion fails for >5% of data | Fix conversion, investigate edge cases |
| 1 | Encoder separation ratio < 1.0 | Try domain fine-tuning or alternate encoder |
| 2 | Ternary decoder < 60% Tier 1 accuracy | Increase decoder capacity or use partial ternary |
| 3 | End-to-end compile rate < 50% | Debug component-level. Check lowering, check data pipeline |
| 4 | No configuration reaches 75% compile strict | Fundamental architecture concern. Write up findings, compare with standard distillation |
| 5 | Best config stalls below 90% compile strict | Document diminishing returns. The architecture may have a ceiling for this task |

---

## File Map (New Files This Project Creates)

### Data Preparation Scripts (in `research/scripts/`)
```
research/scripts/build_typed_ir_data.py     # Phase 0.1: DSL → typed IR conversion
research/scripts/build_decoder_vocab.py     # Phase 0.2: Tier 1+2 vocabulary construction
research/scripts/build_negative_bank.py     # Phase 0.3: Hard negative triple generation
research/scripts/build_ood_prompts.py       # Phase 0.4: OOD prompt set generation
```

### Model Code (in `research/src/`)
```
research/src/encoder.py
research/src/domain_gate.py
research/src/intent_extractor.py
research/src/bridge.py
research/src/ternary_decoder.py
research/src/value_filler.py
research/src/lowering.py
research/src/losses.py
research/src/data.py
research/src/trainer.py
research/src/evaluate.py
```

### Data Artifacts
```
training_data/typed_ir_train.jsonl
training_data/typed_ir_eval.jsonl
training_data/hard_negative_bank.jsonl
references/ood_prompt_set.jsonl
references/tier1_vocab.json
references/tier2_vocab/*.json
```

### Documentation
```
research/docs/BALANCED_SASHIMI_RESEARCH.md  # Theory and architecture (this pair)
research/docs/BALANCED_SASHIMI_PLAN.md      # Operations (this file)
research/docs/EXPERIMENT_RESULTS.md          # Data ledger
```

---

## Getting Started: First Session Checklist

When we're ready to begin implementation:

- [ ] Verify `research/` directory structure (src/, scripts/, tests/, configs/, models/)
- [ ] Install PyTorch + sentence-transformers
- [ ] Run `research/scripts/build_typed_ir_data.py` — convert training data
- [ ] Run `research/scripts/build_decoder_vocab.py` — build vocabularies
- [ ] Run round-trip validation on typed IR data
- [ ] Run `research/scripts/build_negative_bank.py` — seed negative bank
- [ ] Run `research/scripts/build_ood_prompts.py` — build OOD set
- [ ] Implement and test `research/src/ternary_decoder.py` STE quantization
- [ ] Implement and test `research/src/losses.py` composite loss
- [ ] Run toy problem validation (EXP-2.1)
- [ ] **First real training run**

---

*Plan version: 1.0. Created 2026-02-16. Subject to revision based on Phase 0 findings.*
