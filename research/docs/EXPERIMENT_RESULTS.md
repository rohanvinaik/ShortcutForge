# Balanced Sashimi: Experiment Results Ledger

**Project:** ShortcutForge — Hybrid Continuous-Ternary Architecture
**Started:** February 2026
**Hardware:** Apple M1 Max 64GB (local); cloud bursts as noted
**Eval set:** 100 frozen examples (`training_data/shortcutdsl_eval.jsonl`)

---

## Baselines (Established)

### BL-1: Existing 8B Monolithic (Llama 3.1 8B + LoRA)

| Metric | Value | Date | Notes |
|---|---|---|---|
| Parse rate | 93.0% | 2026-02-15 | Linter v2.4, max_tokens=4096 |
| Validate strict | 85.0% | 2026-02-15 | |
| Compile strict | 85.0% | 2026-02-15 | |
| Validate permissive | 89.0% | 2026-02-15 | |
| Compile permissive | 89.0% | 2026-02-15 | |
| Runtime unverified | 4.0% | 2026-02-15 | |
| Model params | ~8B (4-bit) | — | mlx-community/Meta-Llama-3.1-8B-Instruct-4bit |
| Adapter params | ~336MB | — | models/baseline-v1-mlx |
| Inference latency (p50) | TBD | — | To be measured |
| Inference latency (p95) | TBD | — | To be measured |

**Failure breakdown:** parse_overflow=2, validate_unknown_action=8, parse_syntax=5

---

## Experiment Cycle 1: Component Validation

*Purpose: Validate each architectural component independently before integration.*

### EXP-1.1: Input Encoder Evaluation

**Date:** ____
**Configuration:**
- Encoder: all-MiniLM-L6-v2 (22M params, 384-dim)
- Task: Prompt similarity ranking (do in-domain synonymous prompts cluster?)
- Data: Training prompts with manual similarity labels (subset, n=___)

**Protocol:**
1. Encode all training prompts.
2. Compute pairwise cosine similarity.
3. Evaluate: do prompts with identical DSL targets cluster together?
4. Measure intra-cluster / inter-cluster similarity ratio.

| Metric | Value | Notes |
|---|---|---|
| Mean intra-class cosine similarity | | Prompts with same DSL target |
| Mean inter-class cosine similarity | | Prompts with different DSL targets |
| Separation ratio (intra/inter) | | Higher = better |
| Silhouette score (domain clusters) | | Cluster quality by domain |

**Observations:**

**Decision:** [ ] Encoder sufficient as-is / [ ] Needs domain fine-tuning / [ ] Replace encoder

---

### EXP-1.2: Domain Gate Validation

**Date:** ____
**Configuration:**
- Gate architecture: ______ (linear probe / MLP / SVM)
- Training data: in-domain=___ examples, OOD=___ examples
- Eval split: ___/___

| Metric | Value | Notes |
|---|---|---|
| Precision (OOD detection) | | Target: ≥99% |
| Recall (OOD detection) | | Target: ≥95% |
| F1 | | |
| False accept rate | | In-domain classified as OOD |
| False reject rate | | OOD classified as in-domain |
| Inference time (ms) | | |

**Confusion matrix:**

|  | Predicted In-Domain | Predicted OOD |
|---|---|---|
| Actual In-Domain | | |
| Actual OOD | | |

**Observations:**

**Decision:** [ ] Gate architecture selected: ____________

---

### EXP-1.3: Tier 1 Vocabulary Construction

**Date:** ____
**Source:** Parsed from 6,679 training examples via dsl_parser.py

| Statistic | Value | Notes |
|---|---|---|
| Total Tier 1 token types | | Structural + action identifiers |
| Action identifier tokens | | Canonical short names |
| Structural keyword tokens | | SHORTCUT, IF, ACTION, etc. |
| Variable pattern tokens | | $VAR_1...$VAR_N |
| Handle tokens | | @prev, @input, @item, @index, @date |
| Mean Tier 1 sequence length | | Per training example |
| Max Tier 1 sequence length | | |
| Tier 1 vocabulary coverage | | % of eval set actions covered |

**Observations:**

---

### EXP-1.4: Tier 2 Vocabulary Construction

**Date:** ____
**Source:** Extracted from action catalog + param schemas

| Statistic | Value | Notes |
|---|---|---|
| Actions with enum parameters | | Out of 615 |
| Total Tier 2 token types | | Across all action-specific vocabs |
| Mean Tier 2 tokens per action | | |
| Max Tier 2 tokens (single action) | | Which action? |
| Coverage of eval set params | | % of eval param values in Tier 2 vocabs |

**Observations:**

---

## Experiment Cycle 2: Decoder Architecture Ablation

*Purpose: Determine whether ternary weights help or hurt for structural decoding.*

### Template for Decoder Runs

Copy this block for each configuration tested.

#### EXP-2.X: [Configuration Name]

**Date:** ____
**Configuration:**
- Weight regime: ______ (continuous / partial ternary / full ternary)
- Output target: ______ (DSL text / typed IR)
- Negative learning: ______ (off / margin only / margin + repair)
- Bottleneck dim: ______
- Encoder: ______ (frozen / fine-tuned)
- OOD gate: ______ (off / on)
- Training data: ______
- Training iterations: ______
- Batch size: ______
- Learning rate: ______
- Hardware: ______ (local M1 Max / cloud ______)
- Training time: ______

**Compilation Metrics (frozen eval set, n=100):**

| Metric | Value | vs BL-1 | Notes |
|---|---|---|---|
| Parse rate | | Δ= | |
| Validate strict | | Δ= | |
| Compile strict | | Δ= | |
| Validate permissive | | Δ= | |
| Compile permissive | | Δ= | |
| Runtime unverified | | Δ= | |

**Architecture-Specific Metrics:**

| Metric | Value | Notes |
|---|---|---|
| Bottleneck effective rank | | SVD of bridge output |
| Ternary weight % = -1 | | Per decoder layer avg |
| Ternary weight % = 0 | | Per decoder layer avg |
| Ternary weight % = +1 | | Per decoder layer avg |
| Decision sharpness (mean entropy) | | Lower = sharper |
| Hard-negative separability (nats) | | Log-likelihood gap |
| Failure category entropy | | Shannon entropy over failure types |

**Adaptive Loss Weights (final values):**

| Component | σ² (uncertainty) | Effective weight | Notes |
|---|---|---|---|
| L_ce | | | |
| L_margin | | | |
| L_repair | | | |

**Failure Analysis:**

| Failure Type | Count | % | Change from BL-1 |
|---|---|---|---|
| Parse overflow | | | |
| Parse syntax | | | |
| Unknown action | | | |
| Bad parameters | | | |
| Structural error | | | |
| Other | | | |

**Training Dynamics:**

| Checkpoint | L_total | L_ce | L_margin | L_repair | Val compile % |
|---|---|---|---|---|---|
| Iter 100 | | | | | |
| Iter 250 | | | | | |
| Iter 500 | | | | | |
| Iter 750 | | | | | |
| Iter 1000 | | | | | |

**Observations:**

**Decision:** [ ] Promising / [ ] Neutral / [ ] Abandon variant

---

## Experiment Cycle 3: Information Bottleneck Sweep

*Purpose: Determine intrinsic dimensionality of the intent → structure mapping.*

### Bottleneck Dimensionality Sweep

**Configuration held constant:** [best configuration from Cycle 2]

| Bottleneck dim (d) | Compile strict % | Compile permissive % | Effective rank | Separability | Notes |
|---|---|---|---|---|---|
| 32 | | | | | |
| 64 | | | | | |
| 128 | | | | | |
| 256 | | | | | |
| 384 (full) | | | | | |

**Critical dimension** (smallest d where compile strict ≥ 90%): ______

**Observations:**

**Interpretation (information-theoretic):**

---

## Experiment Cycle 4: Negative Learning Ablation

*Purpose: Isolate the contribution of each negative learning component.*

| Configuration | Compile strict % | Separability | Failure entropy | Notes |
|---|---|---|---|---|
| CE only (no negatives) | | | | |
| CE + L_margin | | | | |
| CE + L_repair | | | | |
| CE + L_margin + L_repair (full) | | | | |
| CE + L_margin + L_repair + adaptive λ | | | | |

**Key finding:** Which negative learning component contributes most?

**Observations:**

---

## Experiment Cycle 5: Track Comparison (A vs B vs C)

*Purpose: Compare distillation path vs. direct training vs. from-scratch.*

| Track | Description | Compile strict % | Compile permissive % | Separability | OOD F1 | Params (M) | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|
| A | Teacher-distilled | | | | | | |
| B | Direct tiny-specialist | | | | | | |
| C | From-scratch ablation | | | | | | |
| BL-1 | Existing 8B baseline | 85.0% | 89.0% | N/A | N/A | ~8000 | TBD |

**Observations:**

**Key finding:**

---

## Interpretability Analysis

### Weight Pattern Analysis (Best Ternary Configuration)

**Date:** ____

For the top-5 most common actions in the eval set, document the ternary weight patterns:

#### Action: ________________ (e.g., `downloadurl`)

| Feature dimension | Weight (-1/0/+1) | Interpretation |
|---|---|---|
| [Most positive-weighted input feature] | +1 | |
| [Second most positive] | +1 | |
| [Most negative-weighted input feature] | -1 | |
| [Second most negative] | -1 | |
| Total non-zero weights | ___/384 | Sparsity = ___% |

**Is the pattern interpretable?** [ ] Yes — clearly related to intent / [ ] Partially / [ ] No — opaque

### Bottleneck Representation Visualization

**Method:** UMAP or t-SNE of bottleneck representations, colored by:
1. Domain profile
2. Compilation outcome (success/failure)
3. Action type

**Observations:**

---

## Cumulative Findings

### Hypothesis Status Tracker

| # | Hypothesis | Status | Evidence | Cycle |
|---|---|---|---|---|
| H1 | Ternary weights improve structural decoding over continuous | ☐ Supported / ☐ Refuted / ☐ Inconclusive | | |
| H2 | Negative learning improves hard-negative separability | ☐ Supported / ☐ Refuted / ☐ Inconclusive | | |
| H3 | Intent → structure mapping has low intrinsic dimensionality (<128) | ☐ Supported / ☐ Refuted / ☐ Inconclusive | | |
| H4 | Split architecture (continuous encoder + ternary decoder) outperforms uniform precision | ☐ Supported / ☐ Refuted / ☐ Inconclusive | | |
| H5 | Typed IR output improves compile rate over raw DSL text | ☐ Supported / ☐ Refuted / ☐ Inconclusive | | |
| H6 | Adaptive loss weighting outperforms fixed coefficients | ☐ Supported / ☐ Refuted / ☐ Inconclusive | | |
| H7 | Teacher distillation (Track A) outperforms direct training (Track B) | ☐ Supported / ☐ Refuted / ☐ Inconclusive | | |

### Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| 2026-02-16 | Architecture: split continuous/ternary | Theoretical alignment with task asymmetry | Uniform ternary, uniform continuous |
| 2026-02-16 | Encoder: all-MiniLM-L6-v2 | Encoder-only, optimized for semantic similarity | Qwen 0.5B encoder layers, BERT-base |
| 2026-02-16 | Adaptive loss: UW-SO framework | Handles different convergence rates | Fixed λ, manual curriculum |
| | | | |

---

## Appendix A: Raw Eval Outputs

*For each experiment run, store the full eval output in:*
`training_data/balanced_sashimi/EXP-{cycle}.{number}_eval_results.json`

*Format: identical to existing `eval_results.json` schema for compatibility with `check_regression.py`.*

## Appendix B: Training Curves

*Store training loss curves as JSON arrays in:*
`training_data/balanced_sashimi/EXP-{cycle}.{number}_training_log.jsonl`

*Schema per line:*
```json
{
  "iter": 100,
  "L_total": 2.31,
  "L_ce": 1.85,
  "L_margin": 0.34,
  "L_repair": 0.12,
  "L_ood": 0.05,
  "sigma_ce": 0.8,
  "sigma_margin": 1.2,
  "sigma_repair": 1.5,
  "val_compile_strict": 0.72,
  "val_compile_permissive": 0.78,
  "ternary_sparsity": 0.45,
  "wall_time_s": 3600
}
```

## Appendix C: Compute Log

| Date | Experiment | Platform | GPU | Duration | Cost | Notes |
|---|---|---|---|---|---|---|
| | | Local M1 Max | — | | $0 | |
| | | Cloud | | | | |

**Running total cloud spend:** $____

---

*Ledger version: 1.0. Template created 2026-02-16. Skeleton awaiting experimental data.*
