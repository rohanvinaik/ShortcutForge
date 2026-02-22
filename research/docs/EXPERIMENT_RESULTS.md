# Balanced Sashimi: Experiment Results Ledger

**Project:** ShortcutForge — Hybrid Continuous-Ternary Architecture
**Version:** 2.0
**Started:** February 2026
**Hardware:** Apple M1 Max 64GB (local); cloud bursts as noted
**Eval set:** 100 frozen examples (`training_data/shortcutdsl_eval.jsonl`)

Dual-stream results ledger: architecture validation + process evaluation validation.

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
| Precision (OOD detection) | | Target: >=99% |
| Recall (OOD detection) | | Target: >=95% |
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

**PAB Trajectory Profile:**

*Full profile saved at: `research/models/EXP-2.X/pab_profile.json`*

| PAB Metric | Value | Assessment |
|---|---|---|
| Stability mean (S̄) | | ≤0.15 = stable, 0.15–0.30 = moderate, >0.30 = chaotic |
| Stability std | | Low = consistent dynamics |
| Predictability (final) | | ≤0.05 = structured training |
| Stability regime | | stable / chaotic / phase_transition |
| Tier 1 convergence step | | Step where tier1_acc > 0.80 |
| Tier 2 convergence step | | Step where tier2_acc > 0.70 |
| Ternary crystallization (final) | | % of weights settled |
| Crystallization rate | | Slope of crystallization curve |
| Representation evolution (final) | | Low = bottleneck stabilized |
| Convergence epoch | | Step where S < 0.10 for 5 consecutive checks |
| Early-exit triggered? | | [ ] No / [ ] Yes at step ___ |

**Domain Progression (PAB):**

| Domain | Classification | Convergence Step | Final Accuracy | Notes |
|---|---|---|---|---|
| health | | | | early / late / unstable |
| api | | | | |
| file | | | | |
| media | | | | |
| clipboard | | | | |
| calendar | | | | |
| share_sheet | | | | |
| morning_routine | | | | |

**Behavioral Fingerprint:**

| Metric | Value | Notes |
|---|---|---|
| Action selection entropy (mean) | | Lower = more discrete |
| Output variance eigenvalue ratio | | Higher = clearer restriction sites |
| Cross-seed fingerprint correlation | | Higher = more stable behavior |
| Fingerprint checkpoint: step 200 | | Captured [ ] Yes / [ ] No |
| Fingerprint checkpoint: step 500 | | Captured [ ] Yes / [ ] No |
| Fingerprint checkpoint: step 1000 | | Captured [ ] Yes / [ ] No |

**Observations:**

**Decision:** [ ] Promising / [ ] Neutral / [ ] Abandon variant

---

## Experiment Cycle 3: Information Bottleneck Sweep

*Purpose: Determine intrinsic dimensionality of the intent → structure mapping.*

### Bottleneck Dimensionality Sweep

**Configuration held constant:** [best configuration from Cycle 2]

| Bottleneck dim (d) | Compile strict % | Compile permissive % | Effective rank | Separability | PAB Stability (S̄) | PAB Tier1 Conv. Step | PAB Crystal. Rate | Fingerprint Stability | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 32 | | | | | | | | | |
| 64 | | | | | | | | | |
| 128 | | | | | | | | | |
| 256 | | | | | | | | | |
| 384 (full) | | | | | | | | | |

**Critical dimension** (smallest d where compile strict ≥ 90%): ______
**Fastest crystallization** (smallest d with crystallization_rate > 0.002): ______
**Most stable training** (d with lowest PAB stability_mean): ______
**Most stable fingerprint** (d with highest cross-seed fingerprint correlation): ______

**Observations:**

**Interpretation (information-theoretic):**

---

## Experiment Cycle 4: Negative Learning Ablation

*Purpose: Isolate the contribution of each negative learning component.*

| Configuration | Compile strict % | Separability | Failure entropy | PAB Stability (S̄) | PAB Predictability | Unstable Domains | Fingerprint Stability | Notes |
|---|---|---|---|---|---|---|---|---|
| CE only (no negatives) | | | | | | | | |
| CE + L_margin | | | | | | | | |
| CE + L_repair | | | | | | | | |
| CE + L_margin + L_repair (full) | | | | | | | | |
| CE + L_margin + L_repair + adaptive λ | | | | | | | | |

**Key finding (endpoint):** Which negative learning component contributes most to compile rate?

**Key finding (trajectory):** Does negative learning change the *stability* of training, even when endpoint metrics are similar? Do runs with negatives have fewer "unstable" domains?

**Key finding (behavioral):** Does negative learning produce more discrete/stable behavioral fingerprints?

**Observations:**

---

## Experiment Cycle 5: Track Comparison (A vs B vs C)

*Purpose: Compare distillation path vs. direct training vs. from-scratch.*

| Track | Description | Compile strict % | Compile permissive % | Separability | OOD F1 | Params (M) | Latency p95 (ms) | PAB Stability (S̄) | PAB Regime | Fingerprint Stability | Fingerprint Discreteness |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | Teacher-distilled | | | | | | | | | | |
| B | Direct tiny-specialist | | | | | | | | | | |
| C | From-scratch ablation | | | | | | | | | | |
| BL-1 | Existing 8B baseline | 85.0% | 89.0% | N/A | N/A | ~8000 | TBD | N/A | N/A | N/A | N/A |

**Trajectory comparison**: Do the three tracks have qualitatively different learning trajectories? Does Track A (distilled) learn more smoothly than Track B (direct)? Does Track C (from-scratch) show a longer "chaotic" phase before stabilization?

**Observations:**

**Key finding:**

---

## Experiment Cycle 6: PAB Framework Empirical Validation

*Purpose: Empirically test PAB's theoretical claims against traditional benchmark evaluation. Each experiment tests a specific claim by comparing PAB predictions against traditional evaluation on stress tests.*

### EXP-6.1: Similar Endpoints, Different Trajectories (PAB Claim 1)

**Date:** ____
**Configuration pair:**
- Config A: ______ (compile strict: ___%)
- Config B: ______ (compile strict: ___%)
- Endpoint difference: ≤2%

**PAB trajectory comparison:**

| Metric | Config A | Config B |
|---|---|---|
| PAB stability_mean | | |
| PAB predictability | | |
| PAB regime | | |

**Stress test results:**

| Stress Test | Config A | Config B | Better performer |
|---|---|---|---|
| Distribution shift (unseen domains) | | | |
| Adversarial near-miss prompts | | | |
| Data corruption (noisy final 10%) | | | |

**PAB prediction (better trajectory → more robust):** [ ] Confirmed / [ ] Refuted / [ ] Inconclusive
**Traditional benchmark prediction:** [ ] Both equivalent (correct) / [ ] Differentiated (wrong — shouldn't have)

**Observations:**

---

### EXP-6.2: Trajectory Stability vs Generalization (PAB Claim 2)

**Date:** ____

**Ranking comparison:**

| Configuration | PAB Stability Rank | Generalization Rank | Delta |
|---|---|---|---|
| | | | |

**Correlation:** Spearman rho = ______ (target: > 0.7)
**p-value:** ______

**Early stopping comparison:**

| Method | Mean held-out performance | Overfitting detected at step | Notes |
|---|---|---|---|
| Traditional (patience-based) | | | |
| PAB convergence detection | | | |

**PAB prediction (stability → generalization):** [ ] Confirmed / [ ] Refuted / [ ] Inconclusive

**Observations:**

---

### EXP-6.3: Class-wise Progression vs Domain Augmentation (PAB Claim 3)

**Date:** ____

**PAB-identified fragile domains:** ______

**Augmentation comparison:**

| Method | Data added | Previously-unstable domains stabilized | Compile rate change | Notes |
|---|---|---|---|---|
| PAB-targeted augmentation | | | | |
| Random augmentation (same size) | | | | |

**PAB prediction (targeted > random):** [ ] Confirmed / [ ] Refuted / [ ] Inconclusive

**Observations:**

---

### EXP-6.4: Feature Importance Consistency (PAB Claim 4)

**Date:** ____

**Feature importance consistency (L) by configuration type:**

| Config Type | L value | Adversarial vulnerability (% fooled) | Representation drift (200 more steps) | Notes |
|---|---|---|---|---|
| Ternary | | | | |
| Continuous | | | | |
| Partial ternary | | | | |

**Correlation (L vs adversarial vulnerability):** Spearman rho = ______ (prediction: negative, < -0.5)
**Ternary L > Continuous L:** [ ] Yes / [ ] No (prediction: Yes)

**Observations:**

---

### EXP-6.5: Predictability vs Reproducibility (PAB Claim 5)

**Date:** ____

| Configuration | PAB Predictability Var(ΔL) | Cross-seed variance (3 seeds) | Extrapolation error (step 500→1000) | Notes |
|---|---|---|---|---|
| | | | | |

**Correlation (predictability vs seed variance):** Spearman rho = ______ (prediction: positive, > 0.7)
**Correlation (predictability vs extrapolation error):** Spearman rho = ______ (prediction: positive, > 0.5)

**PAB prediction (low predictability → reproducible):** [ ] Confirmed / [ ] Refuted / [ ] Inconclusive

**Observations:**

---

### EXP-6.6: Approximate vs Exact Correctness (PAB Claim 6)

**Date:** ____

| Configuration | PAB Stability | Strict compile rate | Permissive compile rate | Strict variance (3 seeds) | Permissive variance (3 seeds) | Notes |
|---|---|---|---|---|---|---|
| PAB-stable configs (mean) | | | | | | |
| PAB-unstable configs (mean) | | | | | | |

**PAB prediction (stable → lower permissive variance):** [ ] Confirmed / [ ] Refuted / [ ] Inconclusive

**Observations:**

---

### PAB Framework Validation Summary

| Claim # | Claim | Verdict | Key Evidence | Statistical Significance |
|---|---|---|---|---|
| 1 | Similar endpoints, different trajectories → different robustness | | | |
| 2 | Trajectory stability predicts generalization | | | |
| 3 | Class-wise progression enables targeted augmentation | | | |
| 4 | Feature importance consistency detects fragile representations | | | |
| 5 | Predictability measures training structure | | | |
| 6 | Endpoint evaluation conflates approximate/exact correctness | | | |

**Overall PAB framework assessment:** ______ of 6 claims supported. Framework [ ] validated / [ ] partially validated / [ ] not validated for constrained program synthesis.

---

## Experiment Cycle 7: Behavioral Verification

*Purpose: Test whether training quality (PAB) predicts deployment reliability (behavioral fingerprinting).*

### EXP-7.1: Fingerprint Stability Across Training Checkpoints

**Date:** ____
**Model:** Best configuration from Phase 4

**Fingerprint evolution:**

| Checkpoint step | Action entropy | Variance eigenvalue ratio | Δ from previous | Notes |
|---|---|---|---|---|
| 200 | | | — | |
| 400 | | | | |
| 600 | | | | |
| 800 | | | | |
| 1000 | | | | |

**PAB-behavioral correlation:**

| PAB Metric | Behavioral Metric | Spearman rho | Notes |
|---|---|---|---|
| Stability S(t) | Fingerprint Δ | | Prediction: negative (stable training → stable fingerprint) |
| Crystallization rate | Variance ratio | | Prediction: positive (crystallized → discrete) |
| Tier1 accuracy | Action entropy | | Prediction: negative (accurate → focused) |

### EXP-7.2: Ternary vs Continuous Fingerprint Comparison

**Date:** ____

| Metric | Ternary model | Continuous model | Partial ternary model |
|---|---|---|---|
| Action selection entropy | | | |
| Variance eigenvalue ratio | | | |
| Cross-seed correlation | | | |
| Restriction site count | | | |

**Prediction (ternary → more discrete):** [ ] Confirmed / [ ] Refuted / [ ] Inconclusive

**Observations:**

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
| H8 | Ternary weights learn Tier 1 (structure) earlier/more stably than Tier 2 (parameters) | ☐ Supported / ☐ Refuted / ☐ Inconclusive | PAB tier-wise progression | |
| H9 | Negative learning reduces domain-level instability (fewer "unstable" domains) | ☐ Supported / ☐ Refuted / ☐ Inconclusive | PAB domain progression comparison | |
| H10 | PAB-informed distillation curation improves training stability and/or endpoint metrics | ☐ Supported / ☐ Refuted / ☐ Inconclusive | PAB profile comparison pre/post curation | |
| H11 | STE training exhibits characteristic phase transitions visible in PAB stability curves | ☐ Supported / ☐ Refuted / ☐ Inconclusive | PAB stability time series | |
| H12 | Ternary quantization bounds PAB feature importance variance (ternary forces commitment → low L) | ☐ Supported / ☐ Refuted / ☐ Inconclusive | PAB L metric comparison: ternary vs continuous | |
| H13 | PAB trajectory stability predicts behavioral fingerprint stability across training checkpoints | ☐ Supported / ☐ Refuted / ☐ Inconclusive | PAB-behavioral Spearman correlation | |
| H14 | Architectural decomposition produces more discrete behavioral signatures than monolithic architectures | ☐ Supported / ☐ Refuted / ☐ Inconclusive | Fingerprint comparison: decomposed vs monolithic | |
| H15 | PAB stability ranking correlates with generalization ranking (rho > 0.7) | ☐ Supported / ☐ Refuted / ☐ Inconclusive | EXP-6.2 ranking correlation | |
| H16 | PAB-targeted domain augmentation outperforms random augmentation for stabilizing weak domains | ☐ Supported / ☐ Refuted / ☐ Inconclusive | EXP-6.3 augmentation comparison | |
| H17 | PAB predictability predicts cross-seed reproducibility (low Var(ΔL) → low variance across seeds) | ☐ Supported / ☐ Refuted / ☐ Inconclusive | EXP-6.5 predictability-reproducibility correlation | |
| H18 | Configurations passing both endpoint AND trajectory gates are more robust to distribution shift than endpoint-only | ☐ Supported / ☐ Refuted / ☐ Inconclusive | EXP-6.1 stress test comparison | |

### Decision Log

| Date | Decision | Rationale | Alternatives Considered |
|---|---|---|---|
| 2026-02-16 | Architecture: split continuous/ternary | Theoretical alignment with task asymmetry | Uniform ternary, uniform continuous |
| 2026-02-16 | Encoder: all-MiniLM-L6-v2 | Encoder-only, optimized for semantic similarity | Qwen 0.5B encoder layers, BERT-base |
| 2026-02-16 | Adaptive loss: UW-SO framework | Handles different convergence rates | Fixed λ, manual curriculum |
| 2026-02-20 | PAB integration: adapt PABKit metrics for trajectory evaluation | STE training dynamics need monitoring beyond NaN checks; trajectory comparison enriches ablation analysis; distillation quality benefits from per-example difficulty profiling | Full PABKit dependency (rejected: too early-stage, vision-focused); no trajectory tracking (rejected: lose valuable training dynamics data) |
| 2026-02-20 | Trajectory promotion gates alongside endpoint gates | Prevents promoting chaotic/fragile models that happen to hit endpoint targets | Endpoint-only gates (rejected: misses training reliability signal) |
| 2026-02-20 | Reframed as dual-validation project (v2.0) | PAB/PAC theory provides foundational motivation, not add-on monitoring; behavioral verification connects training to deployment | PAB as monitoring-only (rejected: undervalues theoretical contribution); architecture-only evaluation (rejected: misses process quality signal) |
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
  "wall_time_s": 3600,
  "pab_stability": 0.12,
  "pab_predictability": 0.03,
  "pab_generalization_gap": 0.45,
  "pab_repr_evolution": 0.08,
  "pab_tier1_accuracy": 0.74,
  "pab_tier2_accuracy": 0.61,
  "pab_tier3_accuracy": 0.55,
  "pab_crystallization": 0.68
}
```

## Appendix C: Compute Log

| Date | Experiment | Platform | GPU | Duration | Cost | Notes |
|---|---|---|---|---|---|---|
| | | Local M1 Max | — | | $0 | |
| | | Cloud | | | | |

**Running total cloud spend:** $____

## Appendix D: PAB Profile Schema

*Full PAB profiles are saved per-run at: `research/models/<run_name>/pab_profile.json`*

*Schema:*
```json
{
  "experiment_id": "EXP-2.3",
  "config_hash": "abc123...",
  "checkpoints": [50, 100, 150, "..."],
  "stability": [0.35, 0.22, 0.18, "..."],
  "predictability": [0.12, 0.08, 0.04, "..."],
  "generalization_gap": [1.2, 0.8, 0.5, "..."],
  "representation_evolution": [0.45, 0.22, 0.11, "..."],
  "tier1_accuracy": [0.32, 0.55, 0.71, "..."],
  "tier2_accuracy": [0.18, 0.35, 0.52, "..."],
  "tier3_accuracy": [0.15, 0.28, 0.41, "..."],
  "ternary_crystallization": [0.33, 0.45, 0.62, "..."],
  "domain_progression": {
    "health": [0.40, 0.65, 0.78, "..."],
    "api": [0.25, 0.42, 0.58, "..."]
  },
  "domain_classification": {
    "health": "early",
    "api": "late",
    "clipboard": "unstable"
  },
  "loss_ce": [2.1, 1.5, 1.1, "..."],
  "loss_margin": [0.8, 0.5, 0.3, "..."],
  "loss_repair": [0.3, 0.2, 0.1, "..."],
  "loss_adaptive_weights": [
    {"ce": 0.5, "margin": 0.3, "repair": 0.2}
  ],
  "summary": {
    "stability_mean": 0.14,
    "stability_std": 0.08,
    "predictability_final": 0.03,
    "early_stop_epoch": null,
    "convergence_epoch": 450,
    "stability_regime": "stable",
    "tier1_convergence_step": 350,
    "tier2_convergence_step": 600,
    "crystallization_rate": 0.0015
  }
}
```

## Appendix E: PAB Validation Protocol Results

*Detailed statistical tables for PAB framework empirical validation (Experiment Cycle 6).*

### E.1 Configuration Pair Selection for Claim 1

| Pair | Config A | Config B | Compile strict A | Compile strict B | Δ | PAB stability A | PAB stability B |
|---|---|---|---|---|---|---|---|
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |

### E.2 Stability-Generalization Correlation Data

| Configuration | PAB S̄ | PAB Predictability | Held-out gap | Cross-domain transfer | Curriculum shift recovery |
|---|---|---|---|---|---|
| | | | | | |

**Bootstrap 95% CI for Spearman rho (S̄ vs held-out gap):** [_____, _____]
**p-value:** ______

### E.3 Domain Augmentation Results

| Domain | Pre-augmentation PAB class | Post-PAB-targeted class | Post-random class | PAB-targeted examples added | Random examples added |
|---|---|---|---|---|---|
| | | | | | |

### E.4 Feature Importance Consistency Detail

| Configuration | Weight type | L metric | Adversarial % fooled | Repr. drift (cosine) | Notes |
|---|---|---|---|---|---|
| | | | | | |

**Bootstrap 95% CI for correlation (L vs adversarial %):** [_____, _____]

### E.5 Predictability-Reproducibility Detail

| Configuration | Var(ΔL) | Seed 1 compile % | Seed 2 compile % | Seed 3 compile % | σ across seeds | Predicted step-1000 from step-500 | Actual step-1000 | Error |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

---

*Ledger version: 2.0. Template created 2026-02-16. Updated 2026-02-20: Fundamental reframing with dual-stream validation, PAB empirical validation experiments (Cycle 6), behavioral verification experiments (Cycle 7), hypotheses H12–H18, PAB Validation Protocol appendix. Skeleton awaiting experimental data.*
