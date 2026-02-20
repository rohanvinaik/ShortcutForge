# Balanced Sashimi: A Hybrid Continuous-Ternary Architecture for Domain-Constrained Program Synthesis

**Principal Investigator:** Rohan Vinaik
**Research Partner:** Claude (Opus 4.6), Anthropic
**Date:** February 16, 2026
**Status:** Pre-experimental design specification (v1.1 — PAB integration)
**Project:** ShortcutForge — Natural Language → Apple Shortcuts Compiler

---

## Abstract

We propose a novel neural architecture for domain-constrained program synthesis that exploits the asymmetry between a semi-open natural-language input space and a formally specified output space (ShortcutDSL, a typed intermediate representation for Apple Shortcuts). The architecture decomposes the generation task into functionally specialized stages — continuous input encoding, a learned information bottleneck, and a balanced ternary ({-1, 0, +1}) structural decoder — reflecting the hypothesis that *understanding intent* and *emitting correct programs* are fundamentally different computational problems requiring different computational primitives.

The system incorporates a multi-objective training regime featuring hard-negative contrastive learning derived from a deterministic linter's categorized repair taxonomy, adaptive loss weighting via learned homoscedastic uncertainty, and an explicit out-of-domain rejection gate. The existing ShortcutForge compiler stack (615-action semantic validator, 7-pass static analyzer, plist compiler with signing) serves as the deterministic verification layer, ensuring that neural output quality is measured by objective compilation metrics rather than proxy losses.

This work is positioned as fundamental research into how neural networks can display reliable intelligent behavior in formally constrained domains. ShortcutForge provides a uniquely complete experimental apparatus: a verified compiler stack that delivers unambiguous success/failure signals, a categorized error taxonomy from 215 hallucination aliases and 8 repair categories, and a frozen evaluation harness of 100 examples with established baselines. The broader research program connects to questions about minimal cognitive architectures, interpretable program synthesis, and the information-theoretic limits of constrained-domain intelligence — questions relevant to challenges such as ARC-AGI.

---

## 1. Introduction

### 1.1 Motivation

Large language models used for code generation typically operate as monolithic sequence-to-sequence transducers: natural language in, program text out. This approach inherits the LLM's general capabilities (broad language understanding, world knowledge, reasoning) but also its failure modes (hallucination, syntactic imprecision, inconsistent adherence to formal constraints). For *general* programming tasks, this tradeoff is often acceptable — the output space is vast and the constraints are loose.

ShortcutForge presents a different regime entirely. The output space is:

- **Formally specified**: ShortcutDSL has an LALR(1) grammar (`references/shortcutdsl.lark`) with unambiguous parse semantics.
- **Finitely vocabularied**: 615 valid actions, each with known parameter schemas and type constraints.
- **Deterministically verifiable**: A complete compiler stack (linter → parser → semantic validator → static analyzer → plist compiler → signer) provides binary success/failure at each stage.
- **Structurally regular**: Most valid shortcuts are 6–30 lines with 3–5 distinct structural patterns (linear sequence, conditional branch, loop, menu, API call chain).

The input space, conversely, is:

- **Natural language**: Arbitrary phrasing, synonymy, ambiguity, varying specificity.
- **Domain-constrained**: Restricted to "things Apple Shortcuts can do" — a large but bounded semantic domain.
- **Intent-dense**: A single sentence typically encodes 1–3 functional requirements.

This asymmetry — semi-open input, tightly constrained output — suggests that a monolithic transformer is architecturally mismatched to the task. It spends most of its capacity on capabilities (open-ended text generation, world knowledge) that are irrelevant or actively harmful (hallucination of plausible-but-invalid action names).

### 1.2 The ShortcutForge Baseline

The current ShortcutForge system uses Llama 3.1 8B (4-bit quantized) with LoRA fine-tuning, generating ShortcutDSL text that passes through a 6-phase deterministic linter before parsing and compilation. On a frozen evaluation set of 100 examples:

| Metric | Value |
|---|---|
| Parse rate | 93% |
| Validate (strict) | 85% |
| Compile (strict) | 85% |
| Validate (permissive) | 89% |
| Compile (permissive) | 89% |
| Runtime-unverified compile | 4% |

Failure analysis reveals that the 15% strict-compile failure rate decomposes into:
- Parse/syntax failures: 5% (structural errors the linter couldn't repair)
- Unknown action failures: 8% (hallucinated action names that fuzzy matching couldn't resolve)
- Parse overflow: 2% (generation exceeded token budget mid-structure)

The linter currently repairs an additional ~20% of outputs that would have failed without canonicalization, across 8 repair categories: `action` (name resolution), `alias_warning`, `condition` (keyword normalization), `handle` (property stripping), `interpolation` (string repair), `macro_expansion`, `structure` (block closure), and `trailing_newline`. The 215 curated hallucination aliases map known LLM-generated action names to their canonical forms.

### 1.3 Research Questions

This work addresses three interconnected questions:

1. **Architectural**: Can a functionally decomposed, hybrid continuous-ternary architecture outperform a monolithic transformer on constrained program synthesis, at equal or smaller parameter count?

2. **Representational**: What is the minimal sufficient representation of user intent for this task? How narrow can the information bottleneck be while maintaining compilation reliability?

3. **Training-theoretic**: Does hard-negative contrastive learning, using the linter's categorized repair taxonomy as structured error signal, produce measurably sharper decision boundaries than standard cross-entropy training?

### 1.4 Scope and Stance

This is a research project, not a product engineering effort. The existing ShortcutForge pipeline remains the production system. This work explores whether a fundamentally different architecture can yield insights about domain-constrained intelligence — insights applicable to problems beyond shortcut generation (including ARC-AGI-class reasoning in constrained domains).

Concretely:
- **Primary objective**: Deep understanding of architectural choices for constrained-domain neural program synthesis.
- **Secondary objective**: A working system that compiles at ≥95% strict on the frozen eval set with interpretable internal representations.
- **Non-goal**: Replacing the production system unless this architecture demonstrably surpasses it.

---

## 2. Theoretical Framework

### 2.1 The Task as Semantic Parsing

Mapping natural language to ShortcutDSL is an instance of **semantic parsing** — the NLP subfield concerned with mapping utterances to formal meaning representations. Unlike open-ended code generation (where the output is natural-language-like program text), semantic parsing targets a *logical form* with known structure.

Key parallels:
- **Natural language → SQL** (Spider, WikiSQL benchmarks): Maps questions to database queries with known schemas.
- **Natural language → lambda calculus** (GeoQuery, ATIS): Maps utterances to executable logical forms.
- **Natural language → ShortcutDSL** (this work): Maps intent descriptions to typed IR with known action vocabulary.

The semantic parsing literature establishes that output-structure-aware architectures consistently outperform unconstrained sequence-to-sequence models on these tasks. Grammar-constrained decoding, schema-aware attention, and typed output representations are standard tools. Our contribution extends this line of work by introducing *ternary-weight structural decoding* and *deterministic-linter-derived contrastive training* — techniques enabled by the unique properties of the ShortcutForge verification stack.

### 2.2 Information-Theoretic Framing

We hypothesize that the **mutual information** between user intent and correct ShortcutDSL output is dramatically lower than what a general-purpose 8B-parameter language model represents. Informally: the "bandwidth" of the channel from "what the user wants" to "what the shortcut should do" is narrow, even though both endpoints are individually complex.

This hypothesis is testable via the **information bottleneck** (Tishby et al., 2000): the learned bottleneck between encoder and decoder compresses input representations to retain only information relevant to output generation. The dimensionality and effective rank of the bottleneck representation directly measure how much the neural component needs to "know" to do its job.

If the bottleneck can be compressed aggressively (e.g., to 64–128 dimensions) while maintaining compile rates, that is evidence that:
1. The input-to-output mapping has low intrinsic dimensionality.
2. General-purpose language models are dramatically overparameterized for this task.
3. Purpose-built architectures can exploit this low dimensionality for efficiency.

### 2.3 Ternary Weights as Relevance Logic

Standard neural network weights are continuous real numbers, encoding graded feature importance. Balanced ternary weights ({-1, 0, +1}) enforce a qualitatively different computational regime:

- **+1**: "This feature is positively relevant to this decision."
- **-1**: "This feature is negatively relevant (evidence against this decision)."
- **0**: "This feature is irrelevant to this decision."

This is structurally isomorphic to a three-valued **relevance logic**, where propositions can be supporting, contradicting, or irrelevant to a conclusion. For a decoder operating over a finite action vocabulary (615 actions), each output neuron's weight pattern encodes: "which input features support this action, which contradict it, and which don't matter."

The key theoretical property: **interpretability by construction.** A ternary weight matrix can be read as a sparse, signed relevance map. Non-zero entries are meaningful; zero entries are provably irrelevant. This is not achievable with continuous weights, where "small but nonzero" is ambiguous between "slightly relevant" and "artifact of training dynamics."

Recent work on ternary neural networks, particularly Microsoft's BitNet b1.58 (Ma et al., 2024), demonstrates that ternary-weight transformers can match full-precision models at equivalent scale while providing substantial efficiency gains. BitNet b1.58 uses an "absmean" quantization function during training, with weights scaled and rounded to {-1, 0, +1}. Our work extends this paradigm from general language modeling to domain-specific structured prediction, where we hypothesize ternary weights are not merely *sufficient* but *architecturally preferred* due to the categorical nature of the output decisions.

### 2.4 The Argument for Architectural Decomposition

Monolithic transformer architectures learn to blur the boundaries between understanding, planning, and generation. This is powerful for general tasks but counterproductive when:

1. **The output space is formally structured**: The model must discover structure that we already know.
2. **Verification is deterministic**: Wrong outputs are unambiguously wrong — no need for the model to hedge.
3. **Failure modes are categorized**: We know *how* the model fails and can train against specific failure types.

We propose decomposing the generation task into modules with distinct computational characters:

| Module | Computational Character | Weights | Training Signal |
|---|---|---|---|
| Input Encoder | Continuous similarity over semantic space | Float16/32 | Contrastive (SimCSE-style) |
| Domain Gate | Binary classification | Float16 or classical ML | Cross-entropy (OOD) |
| Information Bridge | Learned compression / discretization | Float16 | Bottleneck pressure |
| Structural Decoder | Sharp categorical selection over finite vocabulary | Ternary {-1,0,+1} | CE + margin + repair-weighted |
| Value Filler | Open-ended text generation (short strings) | Float16 or low-bit quantized | Standard CE |

This decomposition is a testable architectural hypothesis, not merely an engineering convenience.

---

## 3. Architecture Specification

### 3.1 System Overview

```
┌──────────────────────────────────────────────────────────────┐
│                    BALANCED SASHIMI v1                        │
│                                                              │
│  [Natural Language Prompt]                                   │
│          │                                                   │
│          ▼                                                   │
│  ┌─────────────────────────────────────────┐                │
│  │  INPUT ENCODER (continuous, pretrained)  │                │
│  │  Model: all-MiniLM-L6-v2 (22M params)   │                │
│  │  Output: h_enc ∈ ℝ^384                  │                │
│  └─────────────┬───────────────────────────┘                │
│                │                                             │
│       ┌────────┴────────┐                                   │
│       │                 │                                    │
│  ┌────▼─────┐   ┌──────▼──────────────────────┐            │
│  │ DOMAIN   │   │  INTENT / SLOT EXTRACTOR    │            │
│  │ GATE     │   │  (continuous, task-specific)  │            │
│  │          │   │  Output: semantic frame f     │            │
│  │  Binary: │   │  = {domain, entities,         │            │
│  │  in/out  │   │    constraints, actions}      │            │
│  └──────────┘   └──────────┬──────────────────┘            │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────┐     │
│  │  INFORMATION BRIDGE                                 │     │
│  │  Continuous → discrete transition                   │     │
│  │  + Symbolic augmentation (snippet/macro retrieval)  │     │
│  │  Output: plan embedding z ∈ ℝ^d (d = TBD)         │     │
│  └─────────────────────────┬──────────────────────────┘     │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────┐     │
│  │  STRUCTURAL DECODER (TERNARY, Tiers 1+2)           │     │
│  │  Weights ∈ {-1, 0, +1}, trained via STE            │     │
│  │                                                     │     │
│  │  Tier 1 — Topology:                                 │     │
│  │    Action identifiers, control flow blocks,         │     │
│  │    variable binding patterns                        │     │
│  │    (finite vocab, ~700 structural tokens)           │     │
│  │                                                     │     │
│  │  Tier 2 — Typed Parameters:                         │     │
│  │    Enum-constrained values conditioned on Tier 1    │     │
│  │    (HealthKit types, HTTP methods, specifiers)      │     │
│  │    (action-dependent finite sets)                   │     │
│  │                                                     │     │
│  │  Loss: L_ce + λ₁·L_margin + λ₂·L_repair           │     │
│  │  (adaptive λ via uncertainty weighting)             │     │
│  └─────────────────────────┬──────────────────────────┘     │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────┐     │
│  │  VALUE FILLER (Tier 3, continuous / light quant)    │     │
│  │  Fills free-text parameter values:                  │     │
│  │    user strings, URLs, numbers, var references      │     │
│  │  Small autoregressive model or template fill        │     │
│  └─────────────────────────┬──────────────────────────┘     │
│                            │                                 │
│  ┌─────────────────────────▼──────────────────────────┐     │
│  │  DETERMINISTIC LOWERING                             │     │
│  │  Structured output → canonical ShortcutDSL text     │     │
│  └─────────────────────────┬──────────────────────────┘     │
│                            │                                 │
│                            ▼                                 │
│  [Existing Pipeline: Parser → Validator → Simulator →        │
│   Compiler → Signer → .shortcut binary]                     │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Input Encoder

**Model**: `all-MiniLM-L6-v2` (sentence-transformers)
**Parameters**: ~22M
**Output**: 384-dimensional dense embedding
**Precision**: Float16 (continuous)

**Rationale**: The input encoder's task is to map diverse natural language phrasings to a shared semantic space. "Log my coffee," "track my caffeine intake," and "record that I had an espresso" must produce similar representations. This is a *recognition* problem requiring continuous, graded similarity — exactly what sentence transformers are trained for.

We choose an encoder-only model rather than extracting encoder layers from a generative model (e.g., Qwen 0.5B) because:
1. Encoder-only models are explicitly optimized for bidirectional semantic understanding.
2. Generative model representations are optimized for next-token prediction, which encodes sequential dependencies rather than utterance-level meaning.
3. Architectural purity: separate computational problems should use separate computational primitives.

The encoder may be further fine-tuned via domain-specific contrastive learning (SimCSE-style) using in-domain prompt pairs.

### 3.3 Domain Gate

**Architecture**: Initially a separate classification head on the encoder output. May be implemented as:
- A learned linear probe on h_enc (simplest)
- A lightweight MLP (2 layers, ReLU, binary output)
- A classical SVM over the 384-d embedding (most interpretable baseline)

All three will be evaluated; the simplest approach that achieves ≥99% precision on OOD rejection will be adopted.

**Function**: Given h_enc, output `{in_domain: bool, confidence: float}`. If out-of-domain, the system refuses generation (returns clarification response). This is a safety gate, not a generation component.

**Training**: Binary cross-entropy (L_ood), trained on a dedicated in-domain vs. out-of-domain prompt set (`references/ood_prompt_set.jsonl`, to be constructed).

**Training-objective separation**: L_ood is trained as a separate head with its own optimization schedule. It does not compete for gradient bandwidth with the generation objectives. This reflects the architectural principle that *judgment about whether to act* is a different computation than *deciding how to act*.

### 3.4 Intent / Slot Extractor

**Architecture**: 1–2 transformer layers or MLP operating on h_enc.
**Output**: A typed semantic frame:

```python
@dataclass
class SemanticFrame:
    domain: str                    # e.g., "health", "api", "file", "media"
    primary_intent: str            # e.g., "log_measurement", "fetch_data"
    entities: list[Entity]         # extracted entities with types
    constraints: list[Constraint]  # e.g., "must include error handling"
    estimated_complexity: str      # "simple" | "medium" | "complex"
```

**Rationale**: This is essentially named entity recognition + intent classification — a solved problem at small scale. The semantic frame provides structured input to the planner, constraining the search space of valid output structures. By making intent extraction explicit, we gain:
1. Interpretable intermediate representations (we can inspect what the model "understood").
2. Error localization: if the shortcut is wrong, we can determine whether the *understanding* or the *generation* was at fault.
3. Natural integration with the existing domain profile system (the `domain` field maps directly to ShortcutForge's 8 domain profiles).

### 3.5 Information Bridge

**Architecture**: A learned bottleneck layer that transforms the continuous semantic frame into a form consumable by the ternary decoder. Options to explore:

1. **Linear projection** (simplest): Dense layer mapping semantic frame → d-dimensional vector, followed by layer normalization.
2. **VQ-like discretization** (most aligned with ternary philosophy): Vector quantization that snaps the continuous representation to the nearest entry in a learned codebook. Each codebook entry represents a distinct "intent configuration."
3. **Cross-attention with symbolic retrieval**: The semantic frame cross-attends to entries from the snippet registry (`references/snippet_registry.json`, ~200 micro-patterns) and macro registry (`references/macro_patterns.json`, 31 templates), allowing symbolic knowledge to augment the neural representation.

The bridge is the **critical measurement point** for the information-theoretic hypothesis. By varying the bottleneck dimensionality d and measuring compile rate degradation, we directly estimate the intrinsic dimensionality of the intent → structure mapping.

### 3.6 Structural Decoder (Ternary)

**Weight regime**: Balanced ternary {-1, 0, +1} for all affine (linear) layers. Non-affine parameters (layer norms, biases, selected embedding scales) remain in higher precision.

**Training method**: Straight-Through Estimator (STE) with absmean scaling (following BitNet b1.58). During forward pass, weights are quantized to ternary. During backward pass, gradients flow through the quantization function via STE.

**Output vocabulary**: The decoder emits tokens from a custom structural vocabulary, organized in two tiers:

**Tier 1 — Topology tokens** (~700 tokens):
- Structural keywords: `SHORTCUT`, `ENDSHORTCUT`, `ACTION`, `SET`, `IF`, `ELSE`, `ENDIF`, `MENU`, `CASE`, `ENDMENU`, `REPEAT`, `ENDREPEAT`, `FOREACH`, `ENDFOREACH`, `COMMENT`
- Action identifiers: 615 canonical action names (short forms, e.g., `downloadurl`, `ask`, `log.health`, `showresult`)
- Variable patterns: `$VAR_1`, `$VAR_2`, ... (abstract variable slots, bound during value fill)
- Handle references: `@prev`, `@input`, `@item`, `@index`, `@date`
- Assignment operator: `=`

**Tier 2 — Typed parameter tokens** (action-conditioned):
- For each action, a known set of parameter keys and enum values
- Example: Given `ACTION log.health`, valid Tier 2 tokens include `WFQuantitySampleType=`, `"Caffeine"`, `"Dietary Calcium"`, `"Steps"`, etc.
- The valid Tier 2 token set is *dynamically determined* by the Tier 1 action identity

**Tier 3 — Free values** (continuous text, separate model):
- User-specified strings, arbitrary URLs, numeric values
- Generated by a small continuous autoregressive model or filled via template
- These tokens rarely cause compilation failures; they are the "safe" part of generation

**Decoding strategy**: Autoregressive within each tier. Tier 1 tokens are emitted first (the structural skeleton), then Tier 2 tokens are emitted conditioned on the skeleton, then Tier 3 values are filled in. This mirrors compiler construction: first the AST structure, then the typed annotations, then the literal values.

### 3.7 Deterministic Lowering

A purely symbolic (non-neural, non-learned) module that converts the structured output of the decoder into canonical ShortcutDSL text. This is a template-based expansion:

```
Tier 1: [ACTION, log.health, SET, $VAR_1, =, @prev]
Tier 2: [WFQuantitySampleType="Caffeine", WFQuantityValue=...]
Tier 3: [VALUE_1 = user_string]

→ ACTION log.health WFQuantitySampleType="Caffeine" WFQuantityValue=...
  SET $VAR_1 = @prev
```

The lowered DSL text feeds directly into the existing ShortcutForge pipeline (`dsl_parser.py` → `dsl_validator.py` → `simulation_harness.py` → `shortcuts_compiler.py`).

### 3.8 Existing Pipeline Integration

The Balanced Sashimi model operates as a **drop-in replacement** for the LLM generation step in the existing orchestrator. The downstream pipeline is unchanged:

```python
# In orchestrator.py, the current flow:
#   raw_dsl = backend.generate(prompt)
#   linted = lint_dsl(raw_dsl)
#   ir = parse_dsl(linted.text)
#   validation = validate_ir(ir, strict=True)
#   ...

# With Balanced Sashimi:
#   structured_output = sashimi_model.generate(prompt)
#   raw_dsl = deterministic_lower(structured_output)
#   # Linter still runs but should have minimal work
#   linted = lint_dsl(raw_dsl)
#   ir = parse_dsl(linted.text)
#   # ... rest unchanged
```

The existing fallback path (current 8B model → Claude rescue) remains active for A/B comparison and safety.

---

## 4. Training Regime

### 4.1 Composite Loss Function

The model is trained with a multi-objective loss:

```
L = (1/2σ₁²)·L_ce + (1/2σ₂²)·L_margin + (1/2σ₃²)·L_repair + log(σ₁) + log(σ₂) + log(σ₃)
```

Where σ₁, σ₂, σ₃ are learned homoscedastic uncertainty parameters (Kendall et al., 2018), and:

**L_ce (Cross-Entropy)**: Standard next-token prediction loss on canonical typed IR targets. This is the primary learning signal: "generate the correct structural output."

**L_margin (Margin Ranking)**: For each training example (prompt, positive_ir, negative_ir):

```
L_margin = max(0, margin - log P(positive_ir | prompt) + log P(negative_ir | prompt))
```

This penalizes the model for assigning high likelihood to near-miss outputs. The hard negatives come from the linter's repair log: pre-canonicalization outputs that were *plausible but wrong*.

**L_repair (Repair-Weighted CE)**: Cross-entropy loss on the *same* canonical targets, but with per-example weights derived from the linter repair profile:

```
w_i = 1 + α · severity(repair_type_i) + β · count(repairs_i)
```

Examples that required more severe or more numerous linter repairs are upweighted, focusing the model's learning on the specific error types it's most prone to.

**Separation of L_ood**: The domain gate's out-of-domain loss is trained as a **separate head** with its own Adam optimizer and learning rate schedule. It does not participate in the adaptive uncertainty weighting because OOD detection is a fundamentally different task (classification vs. generation) that would distort the adaptive balance.

### 4.2 Hard Negative Construction

Hard negatives are sourced from three channels, rotated to prevent overfitting to known failure styles:

1. **Linter repair pairs** (primary source): The raw LLM output before canonicalization, paired with the canonical output after linter repair. These are categorized by repair type (action hallucination, structural error, condition error, etc.) with known taxonomic labels from the 8 repair categories.

2. **Synthetic perturbations**: Programmatically generated near-misses:
   - Action substitution: Replace a correct action with a semantically similar but incorrect one (e.g., `log.health` → `log.workout`)
   - Structural corruption: Remove an `ENDIF`, swap `ELSE` and `ENDIF`, duplicate a `CASE`
   - Parameter mutation: Change an enum value to a different valid-looking but incorrect one

3. **Adversarial generation**: Use the model's own beam search to find high-likelihood outputs that fail validation. These are the model's "best guesses" that are wrong — the most informative negatives.

### 4.3 Adaptive Loss Weighting

We adopt the **Soft Optimal Uncertainty Weighting (UW-SO)** framework, which extends Kendall et al.'s homoscedastic uncertainty weighting with analytically optimal weights and softmax normalization with an adaptable temperature parameter. This addresses known limitations of standard uncertainty weighting:

- **Bad initialization sensitivity**: UW-SO's softmax normalization prevents early-training instability.
- **Overfitting to easy tasks**: Temperature scheduling ensures harder tasks (L_margin, L_repair) retain adequate gradient signal as training progresses.
- **Implicit curriculum**: The adaptive weights naturally discover that L_ce converges first (learn what's right), L_margin converges next (sharpen boundaries), and L_repair converges last (handle edge cases) — a data-driven curriculum that emerges from training dynamics rather than being hand-specified.

### 4.4 Process-Aware Training: Trajectory Analysis for Distillation Quality

Traditional model evaluation judges training runs by final metrics alone (compile rate, accuracy). We integrate **Process-Aware Benchmarking (PAB)** — a framework for analyzing *how* models learn, not just *what* they learn — into both the training loop and the distillation pipeline. PAB was designed by Parama Pal (Pal, 2025) and formalizes learning trajectory analysis within a PAC learning theory extension.

#### 4.4.1 Trajectory Metrics During Training

Every training run records a **PAB profile**: a time series of trajectory-specific metrics computed at regular checkpoints (every 50 iterations). These metrics operate on values the training loop already computes (loss, accuracy, representations) — the overhead is negligible.

**Learning stability** quantifies smoothness of training dynamics:
```
S(t) = |L(t-1) - L(t)| / (L(t-1) + ε)
```
Low stability scores indicate structured, convergent learning; high scores indicate chaotic dynamics. This is especially critical for STE training, where the gradient mismatch between the quantized forward pass and continuous backward pass can introduce training instability that NaN checks alone won't detect.

**Learning curve predictability** measures how structured the loss trajectory is:
```
P = Var(ΔL_t) where ΔL_t = L(t) - L(t-1)
```
Low predictability scores indicate the loss decreases in a regular pattern; high scores indicate erratic jumps. A training run with high accuracy but high predictability variance may be unreliable — it reached the target by luck rather than structured learning.

**Representation evolution** tracks how the information bottleneck embedding changes over training:
```
R(t) = 1 - cos_sim(normalize(z̄(t-1)), normalize(z̄(t)))
```
where z̄(t) is the mean bottleneck embedding at checkpoint t. Declining R(t) indicates the bottleneck representation is stabilizing; persistent high R(t) suggests the model hasn't found a stable internal representation.

**Tier-wise progression** — analogous to PABKit's class-wise progression — tracks when each output tier reaches proficiency:
- *Early-learning tiers*: Accuracy exceeds threshold (e.g., 80%) in the first third of training.
- *Late-learning tiers*: Accuracy exceeds threshold only in the final third.
- *Unstable tiers*: Accuracy oscillates rather than converging monotonically.

If Tier 1 (structural tokens) learns early and stably while Tier 3 (free values) is late/unstable, that's direct evidence for the thesis that ternary weights are architecturally suited to structural decisions.

**Ternary crystallization** tracks the percentage of decoder weights that have settled into their final ternary value ({-1, 0, +1}) and stopped changing across checkpoints. Rising crystallization indicates the weight space is converging; stalled crystallization suggests the model is stuck between discrete states.

#### 4.4.2 PAB-Informed Distillation

The distillation pipeline (Section 5) currently uses binary quality gates: a distillation example either compiles or it doesn't. PAB trajectory analysis enables a richer, second-pass quality filter:

1. **Probe training pass**: Run a short training pass (200–300 iterations) on candidate distillation data, collecting per-example loss trajectories.

2. **Example difficulty profiling**: Classify each training example by its loss trajectory:
   - *Easy* (loss drops fast, stays low): Model already handles these. Over-representing them wastes training budget.
   - *Hard-but-learnable* (loss drops slowly, steadily): High-value examples that push the model forward.
   - *Unlearnable* (loss stays high or oscillates): Likely noisy or conflicting with other examples, even if they individually pass compile gates.
   - *Destabilizing* (loss spikes when this example appears in a batch): Actively harmful; may conflict with other training signal.

3. **Trajectory-informed curation**: Re-balance the distillation dataset based on difficulty profiles. Down-weight easy examples, prioritize hard-but-learnable ones, flag unlearnable examples for manual review, remove destabilizing examples.

4. **Iterative refinement**: After training on curated data, compare the PAB trajectory of the new run against the probe run. If specific domains or action types remain unstable, generate targeted distillation data for those failure modes and repeat.

This creates a closed feedback loop: training trajectory → data curation → training trajectory → data curation, converging on a distillation dataset that produces maximally stable, structured learning.

### 4.5 Ternary Training Protocol

Following BitNet b1.58's established protocol:

1. **Weight initialization**: Initialize ternary layers from a pretrained continuous model (if available from Track A distillation) or from Kaiming initialization (Track C from-scratch).
2. **Absmean quantization**: During forward pass, compute scale factor γ = mean(|W|), then quantize: W_ternary = round(clip(W/γ, -1, 1)).
3. **STE gradient**: During backward pass, gradients flow through the quantization function unchanged (straight-through).
4. **Staged ternarization schedule**: Begin with continuous weights for the first N warmup iterations, then progressively ternarize layers from output (decoder) toward input (bridge). This allows early training to establish good continuous representations before imposing ternary constraints.
5. **Activation monitoring**: Track weight distribution statistics (fraction of {-1, 0, +1}) per layer to detect collapse (all weights going to 0 or all to ±1).

---

## 5. Data Pipeline

### 5.1 Existing Data Assets

| Asset | Count | Format | Source |
|---|---|---|---|
| Training examples | 6,679 | Chat JSONL | Decompiled shortcuts + synthetic |
| Frozen eval set | 100 | Chat JSONL | Curated, DO NOT MODIFY |
| Action catalog | 615 actions | JSON | Apple Shortcuts reverse engineering |
| Hallucination aliases | 215 | Python dict | Curated from eval failures |
| Macro templates | 31 | JSON | 8 categories |
| Snippet registry | ~200 | JSON | Mined from training data |
| Scenario packs | 8 | DSL + rubric | Hand-crafted benchmarks |
| Baseline snapshot | 1 | JSON | Frozen metrics for regression gate |

### 5.2 New Data Artifacts Required

1. **Typed IR training set** (`training_data/typed_ir_train.jsonl`): Convert existing training examples from raw DSL targets to three-tier structural representations. This is a deterministic conversion (parse existing DSL → extract structure → emit tier tokens).

2. **Hard negative bank** (`training_data/hard_negative_bank.jsonl`): Triplets of (prompt, positive_ir, negative_ir, error_tags). Seeded from existing distillation logs, augmented with synthetic perturbations.

3. **OOD prompt set** (`references/ood_prompt_set.jsonl`): Balanced set of in-domain prompts (from training data) and out-of-domain prompts (general questions, math problems, philosophical queries, coding requests for non-Shortcuts platforms). Target: 500+ examples per class.

4. **Tier 2 parameter vocabularies** (`references/tier2_vocab/`): Per-action enumeration of valid typed parameter values, extracted from the action catalog and param schemas. These define the conditional output vocabulary for the ternary decoder's Tier 2.

### 5.3 Data Conversion Pipeline

```
Existing DSL training examples
    │
    ▼
Parse with dsl_parser.py → ShortcutIR
    │
    ▼
Extract Tier 1 tokens (actions, control flow, variables)
    │
    ▼
Extract Tier 2 tokens (typed params, conditioned on actions)
    │
    ▼
Extract Tier 3 values (free strings, numbers)
    │
    ▼
Emit typed_ir_train.jsonl with all three tiers separated
```

This conversion is deterministic and lossless — the existing validated DSL contains all the information needed.

---

## 6. Evaluation Framework

### 6.1 Primary Metrics (Per-Run)

| Metric | Description | Baseline | Target |
|---|---|---|---|
| Parse rate | % outputs that parse after lowering | 93% | ≥98% |
| Validate strict | % outputs passing strict semantic validation | 85% | ≥95% |
| Compile strict | % outputs that compile to signed .shortcut | 85% | ≥95% |
| Validate permissive | % outputs passing permissive validation | 89% | ≥97% |
| Compile permissive | % outputs compiling in permissive mode | 89% | ≥97% |
| Runtime unverified | % compiling but with unresolvable semantic risk | 4% | ≤2% |

### 6.2 Architecture-Specific Metrics

| Metric | Description | Purpose |
|---|---|---|
| Bottleneck dimensionality | Effective rank of information bridge output | Measures intrinsic task dimensionality |
| Ternary weight distribution | Fraction of {-1, 0, +1} per layer | Detects weight collapse |
| Ternary sparsity | Fraction of weights = 0 per layer | Measures structural sparsity |
| Decision sharpness | Entropy of softmax distributions at decode time | Lower = sharper categorical decisions |
| Hard-negative separability | Log-likelihood gap between positive and nearest negative | Measures boundary quality |
| Failure category entropy | Shannon entropy over failure type distribution | Higher = diverse failures (good); lower = concentrated (mode collapse risk) |
| OOD precision / recall | Domain gate performance on OOD test set | ≥99% precision, ≥95% recall |
| Inference latency (p50/p95) | End-to-end generation time on M1 Max | Target: <2s p95 |
| Model size | Total parameters, memory footprint | Target: <500M total |

### 6.3 Ablation Matrix

Each experiment cycle evaluates variants along these dimensions:

| Dimension | Levels | Purpose |
|---|---|---|
| Decoder weight regime | Continuous / Partial ternary / Full ternary | Isolate ternary contribution |
| Negative learning | Off / On (L_margin only) / On (L_margin + L_repair) | Isolate negative learning contribution |
| Output target | Raw DSL text / Typed IR (three-tier) | Isolate structural vocabulary contribution |
| Bottleneck size | 32 / 64 / 128 / 256 / 384 | Map intrinsic dimensionality |
| Encoder | Frozen / Fine-tuned (domain contrastive) | Isolate encoder adaptation contribution |
| OOD gate | Off / On | Measure gate impact on in-domain accuracy |

Full factorial is 3 × 3 × 2 × 5 × 2 × 2 = 360 combinations. We use a **fractional factorial design** (resolution IV) to cover the design space in ~40–60 runs, with full factorial sweeps only for the dimensions showing strongest interactions.

### 6.4 Process-Aware Benchmarking (PAB) Metrics

In addition to endpoint metrics, every training run produces a **PAB profile** — a structured time series capturing the learning trajectory. These metrics are not merely diagnostic; they are formal evaluation criteria that complement endpoint metrics by measuring *how* a model reached its final performance.

#### PAB Profile Schema

Each profile is saved as `research/models/<run_name>/pab_profile.json` alongside the model checkpoint:

```python
@dataclass
class PABProfile:
    """Process-Aware Benchmark profile for a training run."""
    # Experiment identification
    experiment_id: str              # e.g., "EXP-2.3"
    config_hash: str                # SHA of training config for reproducibility

    # Core PAB metrics (time series, one value per checkpoint)
    stability: list[float]          # S(t) = |ΔL| / (L_prev + ε)
    predictability: list[float]     # Var(ΔL) over recent window
    generalization_gap: list[float] # val_loss - train_loss
    representation_evolution: list[float]  # 1 - cos_sim of bottleneck means

    # Balanced Sashimi-specific extensions
    tier1_accuracy: list[float]     # Structural token accuracy over time
    tier2_accuracy: list[float]     # Parameter token accuracy over time
    tier3_accuracy: list[float]     # Value fill accuracy over time
    ternary_crystallization: list[float]  # % of weights settled at {-1,0,+1}

    # Per-domain progression (analogous to PABKit's class-wise progression)
    domain_progression: dict[str, list[float]]  # {domain: [accuracy_per_checkpoint]}
    domain_classification: dict[str, str]  # {domain: "early"|"late"|"unstable"}

    # Per-action difficulty (top-N tracked actions)
    action_progression: dict[str, list[float]]  # {action_name: [accuracy_per_checkpoint]}

    # Loss component trajectories
    loss_ce: list[float]
    loss_margin: list[float]
    loss_repair: list[float]
    loss_adaptive_weights: list[dict[str, float]]  # [{ce: w, margin: w, repair: w}]

    # Derived summary statistics
    stability_mean: float
    stability_std: float
    predictability_final: float
    early_stop_epoch: int | None    # Epoch where val loss first increases
    convergence_epoch: int | None   # Epoch where stability < threshold for 5 consecutive checks
    stability_regime: str           # "stable" | "chaotic" | "phase_transition"
    tier1_convergence_step: int | None  # Step where tier1_accuracy > 0.8
    tier2_convergence_step: int | None
    crystallization_rate: float     # Slope of ternary_crystallization curve
```

#### PAB Comparison Protocol

When comparing configurations (ablation matrix, track comparison), PAB profiles are overlaid to answer:

1. **Which configuration learns fastest?** Compare tier1_convergence_step and tier2_convergence_step.
2. **Which configuration learns most stably?** Compare stability_mean and predictability_final.
3. **Which configuration produces the most structured representations?** Compare representation_evolution trajectories — faster stabilization indicates more structured internal representations.
4. **Do ternary weights crystallize differently under different training regimes?** Compare ternary_crystallization curves across weight regime ablations.
5. **Which domains are fragile?** Domains classified as "unstable" across multiple configurations indicate inherent difficulty, not configuration-specific issues.

### 6.5 Regression and Promotion Gates

Extended from the existing ShortcutForge regression gate (`training/check_regression.py`):

```yaml
balanced_sashimi_gates:
  # Endpoint gates (existing)
  compile_strict_rate: {min: 95.0}
  compile_permissive_rate: {min: 97.0}
  runtime_unverified_compile_rate: {max: 2.0}
  ood_false_accept_rate: {max: 1.0}
  hard_negative_separability: {min: 2.0}  # log-likelihood gap, nats
  inference_latency_p95_ms: {max: 2000}
  model_total_params_M: {max: 500}

  # Trajectory gates (PAB — new)
  pab_stability_mean: {max: 0.15}         # Training was stable, not chaotic
  pab_predictability_final: {max: 0.05}   # Loss trajectory was structured
  pab_tier1_converged_by: {max: 500}      # Structure learned in first half of training
  pab_no_domain_regression: {value: true} # No domain accuracy regressed in final 20%
  pab_crystallization_rate: {min: 0.001}  # Ternary weights are converging, not stuck
```

The trajectory gates prevent promoting a model that hits endpoint targets through chaotic, unreliable training. A model that achieves 95% compile rate via a stable, predictable trajectory is more trustworthy (and more likely to generalize) than one that oscillated wildly and happened to end at the same number.

### 6.6 Mandatory Test Scenarios

1. **Short prompts** (2–8 words): "Toggle DND," "Set a timer for 5 minutes," "Log 200mg caffeine."
2. **Long/ambiguous prompts**: "Every morning check the weather and if it's raining send me a notification, otherwise open my running playlist."
3. **Multi-domain workflows**: Health + API + notification in one shortcut.
4. **Known failure classes**: Prompts drawn from the distillation log where the baseline model failed.
5. **Adversarial near-miss prompts**: "Use the getContentOfURL action to download..." (tests whether the model uses the hallucinated name or the canonical one).
6. **OOD prompts**: "What is the capital of France?" "Write a Python function to sort a list." "Explain photosynthesis."

---

## 7. Experimental Tracks

### 7.1 Track A: Teacher-Distilled Specialist

**Method**: Use the existing 8B baseline as teacher. Generate clean positive/negative pairs via the distillation pipeline. Train the Balanced Sashimi architecture on the distilled data.

**Purpose**: Establishes whether the architecture can learn effectively from high-quality training data, independent of training-from-scratch challenges.

**Expected outcome**: Strong initial compile rates due to clean training signal. The primary question is whether ternary weights preserve or degrade quality relative to continuous weights on the same architecture.

### 7.2 Track B: Direct Tiny-Specialist Training

**Method**: Train the Balanced Sashimi architecture directly on the existing 6,679-example training set (converted to typed IR format), without teacher distillation.

**Purpose**: Tests whether the architecture's structural inductive biases compensate for the lack of teacher-distilled training signal.

**Expected outcome**: Lower initial compile rates than Track A, but potentially better final performance if the negative learning and structural constraints provide stronger inductive bias than the teacher's implicit knowledge.

### 7.3 Track C: From-Scratch Ablation

**Method**: Train a minimal Balanced Sashimi model with random initialization on both encoder and decoder. No pretrained encoder, no teacher distillation.

**Purpose**: Ablation to isolate the contribution of pretrained representations. If Track C approaches Track A performance, the pretrained encoder is unnecessary for this task — evidence that the task's intrinsic complexity is very low.

**Expected outcome**: Significantly worse than Tracks A/B, serving as a lower bound. However, if OOD detection and hard-negative separability metrics are competitive, that suggests the *structural* contributions (ternary, negative learning, typed IR) are carrying most of the weight.

---

## 8. Compute Requirements and Infrastructure

### 8.1 Hardware

**Primary (and ideally only) platform**: Apple M1 Max, 64GB unified memory.

**Design constraint: $0 local-first.** Each individual run fits comfortably in local memory and can be staged/pruned. Cloud is unnecessary for this architecture scale.

**Memory budget per training run** (honest math for STE + Adam in fp32):
- Decoder weights (fp32 shadow for STE): ~0.8 GB
- Decoder gradients: ~0.8 GB
- Adam optimizer states (m, v): ~1.6 GB
- Decoder subtotal before activations: ~3.2 GB
- Activations, buffers, encoder (22M params), gate, bridge: ~3–7 GB depending on batch size
- **Realistic per-run total: 6–10 GB**

Note: ternary quantization helps *inference* memory (~50MB packed at 2 bits), but STE training operates on full fp32 shadow weights. Don't confuse inference footprint with training footprint.

**Parallelism is limited.** One GPU device; multiple MPS processes contend for the same compute. Memory may fit 4+ runs but throughput does not scale linearly. Practical sweet spot: **2–3 concurrent runs**.

**Training feasibility estimate**:
- Input encoder (22M params, frozen or light fine-tuning): Trivial, minutes per epoch.
- Domain gate (linear probe or small MLP): Trivial.
- Structural decoder (ternary, estimated 50–200M params): Estimated 2–8 hours per training run at 1000 iterations on MPS.
- Full ablation matrix (8 priority runs): 12–24 hours wall-clock (2–3 concurrent batches), reduced by early-exit of failing configurations.

**Cloud fallback (CPU-only, if needed)**:
- Only if a single run exceeds 12h locally, or ablation phase exceeds 2 weeks wall-clock.
- CPU-only instances (Hetzner dedicated, AWS c7a spot). Note: STE training is standard fp32 ops (not packed integer kernels), so cloud CPU is not inherently cheaper per-FLOP — the advantage is simply that CPU spot instances are dirt cheap.
- Estimated budget if used: **$10–30**. No GPU instances.

### 8.2 Software Stack

| Component | Tool | Purpose |
|---|---|---|
| Research training | PyTorch 2.x + MPS backend (custom training loop) | STE, composite loss, ternary quantization on Apple GPU |
| Encoder | sentence-transformers (HuggingFace) | Pretrained input embeddings |
| Inference deployment | MLX (Apple Silicon optimized) | On-device inference after training |
| Ternary runtime | Packed 2-bit weights (inference only; STE training uses fp32) | Efficient ternary inference — custom kernels only if latency is a blocker |
| Performance profiling | `torch.profiler` + `torch.compile` (MPS) where beneficial | Benchmark per-model — MPS gains are workload-dependent |
| Gradient safety | Explicit NaN/inf checks every 100 steps | MPS silent-NaN mitigation |
| Experiment tracking | Local JSON + structured markdown | Results recording |
| Evaluation | Existing ShortcutForge eval harness | Compilation metrics |
| Version control | Git, with experiment branches | Reproducibility |

### 8.3 PyTorch Training Architecture

The training loop is fully custom. Key components:

```python
# Pseudocode for core training loop
class BalancedSashimiTrainer:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.domain_gate = DomainGateHead(384, 1)
        self.intent_extractor = IntentSlotExtractor(384, frame_dim)
        self.bridge = InformationBridge(frame_dim, bottleneck_dim)
        self.structural_decoder = TernaryDecoder(bottleneck_dim, tier1_vocab, tier2_vocabs)
        self.value_filler = ValueFiller(...)

        # Adaptive loss weights (learned)
        self.log_sigma_ce = nn.Parameter(torch.zeros(1))
        self.log_sigma_margin = nn.Parameter(torch.zeros(1))
        self.log_sigma_repair = nn.Parameter(torch.zeros(1))

    def forward(self, prompt, positive_ir, negative_ir, repair_weight, ood_label):
        h_enc = self.encoder.encode(prompt)

        # Domain gate (separate loss path)
        ood_pred = self.domain_gate(h_enc)
        L_ood = F.binary_cross_entropy_with_logits(ood_pred, ood_label)

        # Main generation path
        frame = self.intent_extractor(h_enc)
        z = self.bridge(frame)
        logits_pos = self.structural_decoder(z, positive_ir)  # teacher forcing
        logits_neg = self.structural_decoder(z, negative_ir)  # negative scoring

        L_ce = cross_entropy(logits_pos, positive_ir)
        L_margin = margin_ranking_loss(logits_pos, logits_neg, margin=2.0)
        L_repair = repair_weight * cross_entropy(logits_pos, positive_ir)

        # Adaptive weighting
        L_main = (0.5 / self.log_sigma_ce.exp()**2) * L_ce \
               + (0.5 / self.log_sigma_margin.exp()**2) * L_margin \
               + (0.5 / self.log_sigma_repair.exp()**2) * L_repair \
               + self.log_sigma_ce + self.log_sigma_margin + self.log_sigma_repair

        return L_main, L_ood  # optimized by separate optimizers
```

---

## 9. Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Full-network ternary degrades language understanding | High | Ternary only in decoder; encoder stays continuous. Staged ternarization with collapse monitoring. |
| Typed IR reduces expressive flexibility | Medium | Keep DSL-text baseline path for A/B comparison. Tier 3 (free values) remains unconstrained. |
| Hard-negative mining overfits to known failure styles | Medium | Rotate negative sources (linter pairs, synthetic perturbations, adversarial beam search). Monitor failure category entropy. |
| STE gradient noise destabilizes small models | High | Warmup with continuous weights before ternarization. Learning rate warmup + cosine decay. Gradient clipping. |
| Custom training loop bugs mask real performance | Medium | Unit test every loss component. Verify training on toy problems before full data. Sanity check: continuous-weight ablation must match expected baselines. |
| Three-tier decoding introduces integration complexity | Medium | Build and validate one tier at a time. Each tier is a standalone model that can be evaluated independently. |
| Scope creep | High | Fixed ablation matrix. Cycle-level stop/go decisions. No ad hoc experiments outside the matrix. |
| Information bridge bottleneck too narrow | Medium | Sweep dimensionality {32, 64, 128, 256, 384}. If no setting works, remove bottleneck constraint (bridge becomes full-rank). |

---

## 10. Relation to Broader Research Program

### 10.1 ARC-AGI Connections

The Abstraction and Reasoning Corpus (ARC-AGI) challenges require identifying transformation rules from input-output grid pairs and applying them to novel inputs. The core difficulty is *abstraction*: extracting the right features from examples and ignoring irrelevant details.

Balanced Sashimi tests a specific hypothesis about how to engineer this capability:

1. **Explicit relevance encoding** (ternary weights): The network must commit to which input features are relevant (+1), irrelevant (0), or contradictory (-1) for each output decision. This is the same kind of relevance judgment ARC-AGI requires.

2. **Constrained output spaces**: ARC-AGI outputs are grids with known dimensions and finite color sets — formally constrained, like ShortcutDSL. Architectures that exploit output constraints here should transfer conceptually to ARC-AGI.

3. **Negative learning from near-misses**: ARC-AGI solutions fail on *plausible but wrong* transformations. Training against near-misses — as we do with linter repair pairs — should improve discrimination between correct and almost-correct solutions.

4. **Minimal representations**: If the information bottleneck experiment shows that domain-constrained program synthesis requires surprisingly low-dimensional intent representations, that has implications for how we think about the representational demands of ARC-AGI.

### 10.2 Interpretability

A successful Balanced Sashimi system would be *structurally interpretable* in ways that monolithic transformers are not:

- **Encoder representations**: Visualizable as semantic similarity maps. "What prompts does the model consider similar?"
- **Bottleneck representations**: Directly measurable dimensionality. "How much does the model need to know?"
- **Ternary decoder weights**: Readable as signed relevance maps. "Why did the model choose this action?"
- **Failure modes**: Localizable to specific modules. "The encoder understood correctly, but the decoder selected the wrong action."

This interpretability is not an afterthought — it's a design goal that flows directly from the architectural decomposition.

---

## 11. References

### Foundational
- Tishby, N., Pereira, F., & Bialek, W. (2000). The Information Bottleneck Method.
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. CVPR 2018.

### Ternary Networks
- Ma, S., Wang, H., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. *BitNet b1.58*.
- Li, F., Zhang, B., & Liu, B. (2016). Ternary Neural Networks for Resource-Efficient AI Applications. arXiv:1609.00222.
- Communications Physics (2025). Annealing-inspired training of an optical neural network with ternary weights.

### Semantic Parsing
- Yin, P. & Neubig, G. (2017). A Syntactic Neural Model for General-Purpose Code Generation. ACL 2017.
- Wang, B., Shin, R., et al. (2021). Grounded Adaptation for Zero-shot Executable Semantic Parsing. EMNLP 2021.

### Contrastive Learning
- ACM TOSEM (2024). Effective Hard Negative Mining for Contrastive Learning-Based Code Search.
- Robinson, J., et al. (2021). Contrastive Learning with Hard Negative Samples. ICLR 2021.

### Adaptive Loss Weighting
- Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning. arXiv:2408.07985 (2024).
- Investigating Uncertainty Weighting for Multi-Task Learning. IJCV 2025.

### Process-Aware Benchmarking
- Pal, P. (2025). PABKit: Process-Aware Benchmarking Toolkit. GitHub: parama/pabkit. MIT License.
  *Introduces trajectory-aware evaluation metrics (learning stability, generalization efficiency, rule evolution, class-wise progression) grounded in PAC learning theory extensions. We adapt PABKit's class-wise progression to tier-wise and domain-wise progression, and extend the stability metrics to address STE-specific training dynamics.*

### Sentence Encoders
- Wang, W., et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.

---

*Document version: 1.1. Updated 2026-02-20: Added Process-Aware Benchmarking (PAB) integration (Sections 4.4, 6.4–6.5). Pre-experimental specification. To be updated as experimental results accumulate.*
