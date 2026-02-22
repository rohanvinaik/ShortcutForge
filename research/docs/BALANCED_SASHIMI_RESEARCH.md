# Balanced Sashimi: A Hybrid Continuous-Ternary Architecture for Domain-Constrained Program Synthesis

**Principal Investigator:** Rohan Vinaik
**Research Partner:** Claude (Opus 4.6), Anthropic
**Date:** February 20, 2026
**Status:** Pre-experimental design specification (v2.0 — PAC/PAB-grounded dual-validation framework)
**Project:** ShortcutForge — Natural Language → Apple Shortcuts Compiler

---

## Abstract

We propose a dual-validation research project centered on a novel neural architecture for domain-constrained program synthesis. The project advances two co-equal claims:

**Architectural claim.** A hybrid continuous-ternary architecture — with continuous encoding, a learned information bottleneck, and a balanced ternary ({-1, 0, +1}) structural decoder — outperforms monolithic transformers on constrained program synthesis at equal or smaller parameter count, by exploiting the asymmetry between a semi-open natural-language input space and a formally specified output space (ShortcutDSL, a typed intermediate representation for Apple Shortcuts).

**Epistemological claim.** The architecture is designed to produce measurable learning trajectories, enabling Process-Aware Benchmarking (PAB) to evaluate *how* it learns, not just *what* it achieves. Trajectory evaluation reveals reliability properties — stability, structured progression, predictable convergence — invisible to endpoint metrics. Crucially, the architecture's transparency validates the evaluation framework itself: if PAB's trajectory metrics are informative anywhere, they should be maximally informative for an architecture that physically exposes its learning process through discrete weight crystallization, tier-wise progression, and modular decomposition.

The system incorporates a multi-objective training regime featuring hard-negative contrastive learning derived from a deterministic linter's categorized repair taxonomy, adaptive loss weighting via learned homoscedastic uncertainty, and an explicit out-of-domain rejection gate. The existing ShortcutForge compiler stack (615-action semantic validator, 7-pass static analyzer, plist compiler with signing) serves as the deterministic verification layer, providing unambiguous compilation signals that anchor both the endpoint evaluation stream and the trajectory evaluation stream.

The experimental pipeline is designed so that a single set of experiments simultaneously validates the architecture (does it work?) and the evaluation framework (does PAB predict what endpoint metrics cannot?). This dual-validation structure means negative results on either axis are informative: an architecture that trains stably but compiles poorly reveals PAB's limitations; an architecture that compiles well through chaotic training reveals endpoint evaluation's blind spots.

---

## 1. Introduction

### 1.1 The Process Problem in Machine Learning

PAC (Probably Approximately Correct) learning theory (Valiant, 1984) establishes the foundational guarantees for machine learning: given sufficient samples from a distribution, a learner can achieve ε-approximate, δ-probable correctness on unseen examples. Traditional ML benchmarks operationalize PAC by testing the endpoint — given a trained model, does it achieve target accuracy on a held-out set? This implicitly assumes an **Exact Learning** paradigm: the held-out set is IID with respect to training, and performance on it is a binary pass/fail signal.

This assumption is adequate when training is stable and the deployment distribution matches the test distribution. It breaks down when we need to know *how reliable* a model is — not just its accuracy today, but its expected behavior under distribution shift, adversarial perturbation, or continued training. Two models with identical held-out accuracy can have vastly different reliability profiles depending on how they reached that accuracy.

Process-Aware Benchmarking (PAB), introduced by Pal (2025) and formalized in PABKit, provides the correction. PAB's core insight is that **trajectory evaluation** — evaluating *how* a model reaches its endpoint — reveals reliability properties invisible to endpoint metrics. A model that converges smoothly and predictably, learning structure before details, domain by domain in a stable progression, is fundamentally more trustworthy than one that oscillated wildly and happened to land at the same final number.

But PAB faces an architectural challenge: most neural architectures are opaque. PAB can measure their loss curves, but cannot decompose learning into interpretable sub-processes. The loss trajectory of a monolithic transformer conflates representation learning, structural understanding, and parametric memorization into a single scalar signal. PAB can detect instability, but cannot localize it.

This motivates an architecture designed for **evaluability** — one whose internal structure exposes the learning trajectory in decomposed, interpretable terms. If PAB's theoretical claims are correct (that trajectory evaluation predicts reliability), then an architecture designed to be maximally transparent to trajectory analysis should produce maximally informative PAB metrics. If even this architecture fails to produce informative trajectories, that constrains PAB's applicability claims.

The Balanced Sashimi architecture is designed with this dual purpose: to solve the program synthesis task *and* to be the ideal test case for process-aware evaluation.

### 1.2 The ShortcutForge Testbed

ShortcutForge compiles natural language into signed, installable Apple Shortcuts. The output space is:

- **Formally specified**: ShortcutDSL has an LALR(1) grammar (`references/shortcutdsl.lark`) with unambiguous parse semantics.
- **Finitely vocabularied**: 615 valid actions, each with known parameter schemas and type constraints.
- **Deterministically verifiable**: A complete compiler stack (linter → parser → semantic validator → static analyzer → plist compiler → signer) provides binary success/failure at each stage.
- **Structurally regular**: Most valid shortcuts are 6–30 lines with 3–5 distinct structural patterns (linear sequence, conditional branch, loop, menu, API call chain).

The input space, conversely, is:

- **Natural language**: Arbitrary phrasing, synonymy, ambiguity, varying specificity.
- **Domain-constrained**: Restricted to "things Apple Shortcuts can do" — a large but bounded semantic domain.
- **Intent-dense**: A single sentence typically encodes 1–3 functional requirements.

This asymmetry — semi-open input, tightly constrained output — makes ShortcutForge uniquely suited for joint architecture-evaluation validation:

1. **Deterministic verification provides unambiguous trajectory signals.** Compilation is binary — there are no proxy metrics, no subjective scores. When PAB tracks compile rate over training, the signal is exact.
2. **Categorized errors enable PAB's class-wise analysis.** The linter's 8 repair categories (`action`, `alias_warning`, `condition`, `handle`, `interpolation`, `macro_expansion`, `structure`, `trailing_newline`) and 215 hallucination aliases map to learnable failure classes. PAB can track when each failure class is learned (resolved) during training.
3. **The constrained output space makes behavioral fingerprinting tractable.** With 615 actions and finite structural patterns, the space of possible model behaviors is bounded — behavioral signatures are discrete and measurable.
4. **Eight semantic domains provide natural curriculum structure.** The domain profiles (`api_pagination_fetcher`, `calendar_triage`, `clipboard_utility`, `file_router`, `health_logger`, `media_metadata_pipeline`, `morning_routine`, `share_sheet_text_cleaner`) provide natural categories for PAB's class-wise progression analysis.

### 1.3 The Baseline

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

These metrics are an endpoint snapshot. The 85% compile rate tells us the model succeeds on 85 of 100 test cases — but it cannot tell us *how* the model learned to compile those 85, or *why* it fails on the other 15. It does not reveal whether the model learned structure first and details second (a healthy curriculum suggesting robust generalization) or memorized examples in arbitrary order (fragile, likely to degrade under distribution shift). It does not tell us which of the 8 domains are reliably learned versus sporadically correct — a domain that happens to compile 100% today may be the first to fail when prompts shift slightly.

PAB trajectory analysis addresses exactly these gaps. By tracking the learning trajectory across training — when each domain stabilizes, when structural tokens are mastered relative to parameters, whether convergence is smooth or chaotic — we can distinguish robust learning from lucky endpoints.

### 1.4 Research Questions

This work addresses five interconnected questions:

1. **Architectural (Q1)**: Can a functionally decomposed, hybrid continuous-ternary architecture outperform a monolithic transformer on constrained program synthesis, at equal or smaller parameter count?

2. **Representational (Q2)**: What is the minimal sufficient representation of user intent for this task? How narrow can the information bottleneck be while maintaining compilation reliability?

3. **Training-theoretic (Q3)**: Does hard-negative contrastive learning, using the linter's categorized repair taxonomy as structured error signal, produce measurably sharper decision boundaries than standard cross-entropy training?

4. **Process (Q4)**: Does the architecture's learning trajectory satisfy PAB's quality criteria (stability, predictability, structured progression)? Can PAB trajectory metrics predict final performance before training completes?

5. **Behavioral (Q5)**: Does training quality (as measured by PAB) predict deployment reliability (as measured by behavioral fingerprinting)?

### 1.5 Scope and Stance

This is a **dual-validation** research project:

- **Primary objective**: Deep understanding of architectural choices for constrained-domain neural program synthesis — a working system that compiles at ≥95% strict on the frozen eval set with interpretable internal representations.
- **Co-equal objective**: Empirical validation of PAB's theoretical claims — does trajectory evaluation actually predict things endpoint evaluation cannot? The architecture is designed to be maximally transparent to PAB, making this the strongest possible test of PAB's framework.
- **Tertiary objective**: Empirical validation of PAB's theoretical claims in the specific sense — if trajectory evaluation is informative for *this* architecture (designed to expose trajectories), how informative is it for opaque architectures (by comparison)?
- **Non-goal**: Replacing the production system unless this architecture demonstrably surpasses it.

The existing ShortcutForge pipeline remains the production system. This work explores whether a fundamentally different architecture can yield insights about domain-constrained intelligence — insights applicable to problems beyond shortcut generation (including ARC-AGI-class reasoning in constrained domains).

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

#### 2.3.1 Ternary-PAB Synergy

Ternary weights force discrete commitment: each weight must be +1, 0, or -1. This bounds PAB's feature importance variance metric L. In continuous networks, a weight at 0.001 is ambiguously "slightly relevant" or "artifact of training dynamics" — its importance can drift during fine-tuning without the network "noticing." In ternary networks, the weight is exactly 0 (provably irrelevant) or exactly ±1 (committed). This means:

- **Feature importance is inherently stable across checkpoints once crystallized.** A ternary weight that has settled to +1 at checkpoint t and remains +1 at checkpoint t+k has *provably* stable feature importance. There is no continuous drift within the committed state.
- **PAB's consistency metric L has a natural lower bound determined by crystallization rate.** As more weights crystallize (commit to their final ternary value), the fraction of stable features increases monotonically. PAB can track crystallization directly as a process metric.
- **The architecture physically instantiates the distinction PAB measures.** PAB distinguishes stable features (consistent importance across training) from unstable features (drifting importance). Ternary weights make this distinction literal: a crystallized weight is stable by definition; an uncrystallized weight is unstable by definition.

This is not a coincidence — ternary weights and process-aware evaluation are naturally synergistic because both privilege discrete commitment over continuous hedging. The architecture makes PAB's abstract metrics concrete, and PAB provides the evaluation framework that makes the architecture's learning process legible.

### 2.4 PAC Learning and Process-Aware Evaluation

PAC (Probably Approximately Correct) learning theory (Valiant, 1984) establishes sample complexity bounds for achieving ε-approximate, δ-probable correctness. Traditional ML benchmarks operationalize PAC by testing: given enough training data, does the model achieve target accuracy on a held-out set? This treats learning as a black box — only the endpoint matters.

Process-Aware Benchmarking (PAB), introduced by Pal (2025) and formalized in PABKit, extends this framework by defining a quality function over learning *trajectories*, not just endpoints. The key insight: two models with identical final accuracy can have vastly different trajectory properties (stability, predictability, domain progression order), and these properties predict generalization reliability and deployment robustness in ways endpoint metrics cannot.

PAB defines several core trajectory metrics:

- **Learning stability S(T)**: Smoothness of loss trajectory. Quantifies whether training converges in a structured or chaotic manner. Computed as the mean of per-step stability scores S(t) = |L(t-1) - L(t)| / (L(t-1) + ε).
- **Generalization efficiency G(t)**: Rate at which validation performance approaches training performance. Measures sample efficiency — how quickly the model generalizes from training signal to held-out data.
- **Rule evolution diversity R_div**: How many distinct learning "phases" the model traverses. Low diversity suggests memorization (monotonic loss decrease); high diversity suggests structured exploration (distinct phases for different competencies).
- **Learning predictability P_learn**: Variance of loss deltas. Low variance indicates regular, extrapolable learning (the next step's loss is predictable from the trajectory so far); high variance indicates erratic, unreliable learning.
- **Class-wise progression**: Which output classes are learned early, late, or unstably. Reveals curriculum structure — the order in which the model acquires competencies.

In our context, PAB is not merely monitoring — it is *specification*. We define what good learning looks like for this architecture as a set of testable predictions:

1. **Stability should decrease monotonically as ternary weights crystallize.** As more weights commit to their final values, the loss surface becomes smoother. If stability increases during crystallization, the STE gradient mismatch is causing pathological dynamics.
2. **Tier 1 (structural tokens) should learn before Tier 2 (parameters), which should learn before Tier 3 (values).** This is the expected curriculum for a structured decoder: learn the skeleton first, then the types, then the specifics. Violation of this order suggests the architecture is not exploiting its structural decomposition.
3. **Domain progression should show curriculum structure.** Simple domains (e.g., `clipboard_utility`) should stabilize early; complex domains (e.g., `api_pagination_fetcher`) should stabilize late. No domain should be unstable at convergence — instability at convergence indicates the training data is contradictory or the architecture lacks capacity for that domain.
4. **Feature importance (proxied by ternary weight sign stability) should increase monotonically after warmup.** Once the continuous warmup phase ends and ternarization begins, the fraction of crystallized weights should increase steadily. Plateaus indicate the model is stuck between discrete states.

These are testable predictions that connect architectural design to PAB's evaluation framework. If the predictions hold, the architecture validates PAB (the theory correctly predicts observable trajectory properties). If PAB detects trajectory pathologies the predictions don't anticipate, PAB validates itself (it reveals information beyond what the architectural theory predicts).

### 2.5 Behavioral Verification

The Variance-Mediated approach from behavioral verification research (PoT/REV/HBT) provides a framework for connecting training quality to deployment reliability. The core mechanism: a model's behavioral fingerprint — the pattern of activations and outputs it produces on diagnostic probes — should be stable if the underlying representations are stable.

The prediction connecting PAB to behavioral verification:

- **PAB trajectory stability → behavioral fingerprint stability.** A model that trains stably (low S̄, low P) should produce consistent behavioral signatures across runs with different seeds, because stable training implies the model converges to similar representations regardless of initialization.
- **Ternary weights produce inherently discrete behavioral signatures.** Each weight is exactly -1, 0, or +1, so output distributions are step functions over feature relevance rather than smooth curves. Discrete signatures have lower intrinsic noise.
- **Discrete signatures → cleaner restriction sites → more reliable fingerprinting.** When behavioral fingerprinting decomposes model behavior into dimensions, discrete weights produce activation patterns that cluster tightly around a small number of distinct states. This makes restriction sites (dimensions that distinguish model behaviors) more pronounced and easier to identify.

This connects training-time evaluation (PAB) to deployment-time evaluation (behavioral fingerprinting): a model that trains stably should behave predictably. The ternary architecture provides a strong test case for this connection because its discrete weight space should amplify the relationship between training stability and behavioral predictability.

### 2.6 The Argument for Architectural Decomposition

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

This decomposition is a testable architectural hypothesis, not merely an engineering convenience. Critically, the decomposition is motivated not just by engineering (each module performs different computation) but by *evaluability*:

- **Each module produces a distinct trajectory measurable by PAB.** The encoder's representation evolution, the bridge's information compression dynamics, and the decoder's tier-wise progression and crystallization rate are separate, interpretable time series. PAB can analyze each independently and in combination.
- **Each module produces a distinct behavioral signature measurable by fingerprinting.** The encoder's semantic space, the bridge's bottleneck activations, and the decoder's ternary output patterns provide three independent fingerprinting dimensions.
- **Failures are localizable.** If PAB shows the decoder trajectory is stable but the bridge representation is oscillating, the bridge is the bottleneck. If the encoder representations stabilize early but tier-1 accuracy plateaus, the decoder is undertrained. This localization is impossible in a monolithic architecture where all components share a single loss trajectory.
- **This is the architectural instantiation of PAB's core claim**: process evaluation reveals information that endpoint evaluation cannot, *when the architecture exposes its process*. A monolithic transformer with 85% compile rate gives PAB one scalar trajectory. This architecture gives PAB five decomposed trajectories, tier-wise progression curves, crystallization dynamics, and domain-wise accuracy series. If PAB cannot extract useful information from *this* architecture, its claims are weak. If it can, the architecture demonstrates what evaluable design looks like.

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

**PAB affordance**: Encoder representations enable tracking of representation evolution R(t) and domain clustering quality over training. As the encoder adapts to in-domain prompts, PAB can measure whether semantically similar prompts cluster more tightly and whether domain boundaries in the embedding space become sharper.

### 3.3 Domain Gate

**Architecture**: Initially a separate classification head on the encoder output. May be implemented as:
- A learned linear probe on h_enc (simplest)
- A lightweight MLP (2 layers, ReLU, binary output)
- A classical SVM over the 384-d embedding (most interpretable baseline)

All three will be evaluated; the simplest approach that achieves ≥99% precision on OOD rejection will be adopted.

**Function**: Given h_enc, output `{in_domain: bool, confidence: float}`. If out-of-domain, the system refuses generation (returns clarification response). This is a safety gate, not a generation component.

**Training**: Binary cross-entropy (L_ood), trained on a dedicated in-domain vs. out-of-domain prompt set (`references/ood_prompt_set.jsonl`, to be constructed).

**Training-objective separation**: L_ood is trained as a separate head with its own optimization schedule. It does not compete for gradient bandwidth with the generation objectives. This reflects the architectural principle that *judgment about whether to act* is a different computation than *deciding how to act*.

**PAB affordance**: Domain gate confidence distribution provides a binary signal for PAB's class-wise progression on in-domain vs. OOD classification. PAB can track when the gate achieves reliable separation and whether that separation is stable or oscillates during training.

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

**PAB affordance**: Semantic frame extraction quality enables PAB to decompose learning into intent recognition vs. structural generation — localizing failures to understanding or production. If the extractor learns correct domains early but the decoder still fails, PAB can attribute failures to the generation stage rather than the comprehension stage.

### 3.5 Information Bridge

**Architecture**: A learned bottleneck layer that transforms the continuous semantic frame into a form consumable by the ternary decoder. Options to explore:

1. **Linear projection** (simplest): Dense layer mapping semantic frame → d-dimensional vector, followed by layer normalization.
2. **VQ-like discretization** (most aligned with ternary philosophy): Vector quantization that snaps the continuous representation to the nearest entry in a learned codebook. Each codebook entry represents a distinct "intent configuration."
3. **Cross-attention with symbolic retrieval**: The semantic frame cross-attends to entries from the snippet registry (`references/snippet_registry.json`, ~200 micro-patterns) and macro registry (`references/macro_patterns.json`, 31 templates), allowing symbolic knowledge to augment the neural representation.

The bridge is the **critical measurement point** for the information-theoretic hypothesis. By varying the bottleneck dimensionality d and measuring compile rate degradation, we directly estimate the intrinsic dimensionality of the intent → structure mapping.

**PAB affordance**: The bridge is the critical measurement point for both information-theoretic analysis (bottleneck dimensionality) and process evaluation. Representation evolution R(t) tracks how the compressed representation stabilizes over training — fast stabilization with maintained compile rate indicates the bottleneck has found a sufficient compression; oscillation indicates the bottleneck dimensionality may be too narrow for the task.

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

**PAB affordance**: Tier-wise accuracy trajectories enable PAB's class-wise progression analysis — tracking when Tier 1 (structure), Tier 2 (parameters), and Tier 3 (values) each reach proficiency provides a decomposed learning curve unavailable in monolithic decoders. Ternary crystallization provides a unique process metric unavailable in continuous architectures: the fraction of weights committed to their final ternary value is a direct, unambiguous measure of learning progress. **Behavioral verification affordance**: Ternary weights produce discrete behavioral signatures suitable for fingerprinting — each weight is exactly -1, 0, or +1, producing step-function activation patterns that cluster into identifiable behavioral states.

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

**PAB affordance**: Deterministic lowering means compilation success/failure is a pure function of the neural output — no stochastic post-processing obscures the trajectory signal. When PAB tracks compile rate over training, every change in that metric is attributable to a change in the neural model's behavior, not to randomness in the lowering step.

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

### 4.4 Process-Aware Training

PAB is not an external monitoring tool applied after training — it is a training-loop citizen whose metrics serve as formal quality criteria for the training run. This section describes how PAB integrates with the training loop at three levels: real-time trajectory assessment, early exit decisions, and distillation data curation.

#### 4.4.1 PAB as Training-Loop Citizen

Every training run records a **PAB profile**: a time series of trajectory-specific metrics computed at every checkpoint (every 50 iterations). These metrics operate on values the training loop already computes (loss, accuracy, representations) — the overhead is negligible.

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

These metrics are not diagnostics — they are *formal quality criteria*. A training run that achieves target compile rate but shows chaotic stability, erratic predictability, or stalled crystallization is flagged as unreliable regardless of its endpoint metrics. This is PAB's central claim operationalized: process quality is a first-class evaluation dimension, not a secondary concern.

#### 4.4.2 PAB-Informed Early Exit

The early exit protocol implements PAB's core claim — that process quality predicts outcome quality — as a concrete training decision. When PAB metrics indicate the training trajectory has become pathological, exit early rather than wasting compute on a doomed run.

Early exit triggers:
- **Stability explosion**: If S̄(t) over a 100-step window exceeds 0.5, the training dynamics have become chaotic. The STE gradient mismatch is likely causing oscillation between ternary states.
- **Predictability collapse**: If Var(ΔL_t) over a 100-step window exceeds 3× its value at the end of warmup, the training has become erratic.
- **Crystallization stall**: If the crystallization rate drops below 0.0001 per step for 200 consecutive steps, the ternary weights are stuck between states — further training is unlikely to resolve this without intervention (e.g., learning rate adjustment or re-initialization).
- **Domain regression**: If any domain's accuracy drops by more than 10% from its peak for more than 100 consecutive steps, the model is catastrophically forgetting that domain.

This is not an operational hack — it is PAB's claim operationalized. If process quality predicts outcome quality, then bad process (chaotic dynamics, stalled crystallization) means the outcome is unreliable regardless of whether it accidentally hits a target number. Early exit conserves compute for configurations with healthy trajectories.

#### 4.4.3 PAB-Informed Distillation Curation

The distillation pipeline (Section 5) currently uses binary quality gates: a distillation example either compiles or it doesn't. PAB trajectory analysis enables a richer, second-pass quality filter:

1. **Probe training pass**: Run a short training pass (200–300 iterations) on candidate distillation data, collecting per-example loss trajectories.

2. **Example difficulty profiling**: Classify each training example by its loss trajectory:
   - *Easy* (loss drops fast, stays low): Model already handles these. Over-representing them wastes training budget.
   - *Hard-but-learnable* (loss drops slowly, steadily): High-value examples that push the model forward.
   - *Unlearnable* (loss stays high or oscillates): Likely noisy or conflicting with other examples, even if they individually pass compile gates.
   - *Destabilizing* (loss spikes when this example appears in a batch): Actively harmful; may conflict with other training signal.

3. **Trajectory-informed curation**: Re-balance the distillation dataset based on difficulty profiles. Down-weight easy examples, prioritize hard-but-learnable ones, flag unlearnable examples for manual review, remove destabilizing examples.

4. **Iterative refinement**: After training on curated data, compare the PAB trajectory of the new run against the probe run. If specific domains or action types remain unstable, generate targeted distillation data for those failure modes and repeat.

#### 4.4.4 The Feedback Loop

The feedback loop is the point where PAB's theoretical framework becomes operationally concrete. The loop is: training trajectory → data curation → training trajectory → data curation. PAB trajectory metrics from a training run inform which examples to prioritize, which to discard, and where to generate new targeted data. The next training run on curated data produces a new trajectory, which informs the next round of curation.

This creates a testable prediction: **trajectory-curated data should produce better models than randomly-curated data of the same size.** If PAB's claim that process quality predicts outcome quality is correct, then data selected to optimize process quality (stable trajectories, structured progression) should produce models with better endpoint metrics than data selected by endpoint criteria alone (compile rate of individual examples).

Concretely, the test is:
- **Control**: Select N distillation examples by binary quality gate (compiles/doesn't compile), uniformly at random from passing examples.
- **Treatment**: Select N distillation examples by PAB-informed curation (difficulty profiling, domain balancing, destabilizer removal).
- **Measure**: Train identical architectures on both datasets. Compare endpoint metrics (compile rate) AND trajectory metrics (stability, predictability, domain progression).
- **Prediction**: Treatment produces higher compile rate AND better trajectory. If treatment produces better trajectory but equal compile rate, PAB is measuring real properties but they don't matter for this task at this scale. If treatment produces higher compile rate with worse trajectory, PAB's curation heuristics need revision.

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

### 6.1 Two-Stream Evaluation

The evaluation framework operates two streams that are individually necessary and jointly sufficient.

**Stream 1 (Endpoint):** Does the architecture produce better programs? Measured by compilation metrics — parse rate, validation rate, compile rate, latency, model size. These are the metrics that matter for deployment: a model that compiles at 95% is better than one that compiles at 85%, regardless of how it learned to do so.

**Stream 2 (Process):** Does the architecture learn the right way? Measured by PAB trajectory metrics — stability, predictability, tier-wise progression, crystallization rate, domain-wise accuracy evolution. These are the metrics that predict reliability: a model that trains stably and progressively is more likely to generalize, less likely to fail under distribution shift, and more amenable to continued improvement.

Neither stream alone is sufficient:
- A model that achieves high compile rates through chaotic training is fragile. It passes Stream 1 but fails Stream 2. Its 95% compile rate may drop to 80% under distribution shift because its internal representations are unstable.
- A model that trains beautifully but compiles poorly is elegant but useless. It passes Stream 2 but fails Stream 1. Stable convergence to a bad solution is still a bad solution.

The evaluation framework measures both streams and their interaction. The key question is whether Stream 2 metrics *predict* Stream 1 outcomes — does training stability predict compile rate? Does structured tier-wise progression predict generalization? If the streams correlate, PAB adds predictive power. If they don't, endpoint metrics remain the only relevant evaluation.

### 6.2 Endpoint Metrics

| Metric | Description | Baseline | Target |
|---|---|---|---|
| Parse rate | % outputs that parse after lowering | 93% | ≥98% |
| Validate strict | % outputs passing strict semantic validation | 85% | ≥95% |
| Compile strict | % outputs that compile to signed .shortcut | 85% | ≥95% |
| Validate permissive | % outputs passing permissive validation | 89% | ≥97% |
| Compile permissive | % outputs compiling in permissive mode | 89% | ≥97% |
| Runtime unverified | % compiling but with unresolvable semantic risk | 4% | ≤2% |

### 6.3 Architecture-Specific Metrics

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

### 6.4 Process Metrics (PAB)

Every training run produces a **PAB profile** — a structured time series capturing the learning trajectory. These metrics are formal evaluation criteria that complement endpoint metrics by measuring *how* a model reached its final performance.

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

#### PAB Validation Protocol

Beyond comparing configurations, we use the PAB metrics to test whether PAB's framework is itself predictive. Four specific validation experiments:

1. **Do stable trajectories produce models that generalize better to unseen prompts?** Take all configurations that achieve ≥90% compile rate on the frozen eval set. Rank by PAB stability_mean. Evaluate on a *new* held-out prompt set (never seen during training or tuning). Prediction: lower stability_mean → higher held-out compile rate.

2. **Do trajectory metrics correlate with behavioral fingerprint stability?** For each configuration, compute behavioral fingerprint (Section 6.5) across 3 seeds. Compute fingerprint variance across seeds. Rank configurations by PAB stability_mean and by fingerprint variance. Prediction: Spearman rho > 0.5 between the rankings.

3. **Can PAB convergence_epoch predict final performance within 10% at the halfway point?** At step 500 (of 1000), compute PAB's convergence prediction based on trajectory extrapolation. Compare to actual step-1000 performance. Prediction: prediction error < 10% for configurations with low predictability variance.

4. **Does PAB-informed data curation improve outcomes vs. random curation of the same size?** Run the feedback loop experiment from Section 4.4.4. Prediction: PAB-curated data produces higher compile rate.

### 6.5 Behavioral Metrics

Behavioral metrics connect training-time evaluation (PAB) to deployment-time evaluation (behavioral fingerprinting). They test the prediction that training quality predicts deployment reliability.

**Behavioral fingerprint discreteness.** For each decoder output, compute the distribution over the action vocabulary (615 actions). In ternary models, these distributions should be more peaked (lower entropy) than in continuous models because ternary weights enforce discrete relevance judgments. Measure: mean entropy of action selection distributions across the eval set. Lower entropy = sharper, more decisive action selection.

**Restriction site clarity.** Compute the variance tensor on decoder outputs across a set of diagnostic probes (the 100 frozen eval prompts plus 50 adversarial near-miss prompts). Clear restriction sites (low variance in some dimensions, high in others) indicate the model has learned stable feature-action mappings — some features are consistently used for specific decisions, while others vary appropriately with input. Measure: ratio of max to min eigenvalues of the output variance matrix. Higher ratio = clearer restriction sites = more structured behavioral signature.

**PAB-behavioral correlation.** The key prediction connecting the two evaluation streams. Rank all configurations by PAB stability_mean (training-time process quality). Independently rank by behavioral fingerprint stability (cross-seed fingerprint correlation — compute fingerprint for 3 seeds, measure pairwise correlation, average). Compute Spearman rank correlation between the two rankings. Prediction: rho > 0.7. If confirmed, training stability predicts deployment predictability. If rho < 0.3, the connection between training process and deployment behavior is weaker than PAB's framework suggests.

### 6.6 PAB Framework Empirical Validation

This section specifies six experiments that test PAB's theoretical claims against traditional benchmarks. Each experiment is designed so that the result is informative regardless of direction: confirmation validates PAB; refutation constrains PAB's applicability. Together, they constitute a systematic empirical assessment of process-aware evaluation.

**PAB Claim 1: "Models with similar endpoint metrics can have vastly different learning trajectories."**

- **Setup**: From the Phase 4 ablation matrix (Section 6.7), identify pairs of configurations that achieve similar compile rates (within 2% on the frozen eval set) but differ in PAB trajectory properties (stability_mean differs by >2×, or predictability_final differs by >2×).
- **Traditional evaluation**: Both configurations look equally good — compile rates are indistinguishable.
- **PAB evaluation**: The configurations have measurably different trajectories. PAB predicts the more stable configuration is more reliable.
- **Stress test**: Evaluate both configurations under three perturbation conditions:
  - *Distribution shift*: Unseen domains and phrasings (prompts not in training distribution).
  - *Adversarial near-miss prompts*: Prompts that use hallucinated action names or structurally ambiguous phrasing.
  - *Data corruption*: Retrain with noisy training data in the final 10% of iterations (label noise, action substitution).
- **Comparison**: Measure degradation (compile rate drop) under each perturbation. Prediction: the PAB-stable configuration degrades less.
- **Statistical significance**: Bootstrap 95% CIs on degradation difference. Report p-values from paired permutation tests.

**PAB Claim 2: "Trajectory stability predicts generalization reliability."**

- **Setup**: Rank all configurations from the ablation matrix by PAB stability_mean (ascending — lower is more stable).
- **Traditional evaluation**: Rank by held-out compile rate on frozen eval set.
- **PAB evaluation**: Rank by stability_mean. Prediction: these rankings correlate.
- **Stress test**: Construct three independent generalization tests:
  - *Held-out generalization gap*: Compile rate on a NEW held-out set (separate from frozen eval), minus compile rate on training examples.
  - *Cross-domain transfer*: Train on 6 domains, evaluate on 2 held-out domains.
  - *Performance under curriculum shift*: Change the training data ordering mid-training and measure recovery.
- **Comparison**: Compute Spearman rho between PAB stability ranking and each generalization ranking. Prediction: rho > 0.7 for at least two of three generalization tests.
- **Additional comparison**: Traditional early stopping (patience-based, stop when val loss increases for K consecutive checks) vs. PAB convergence detection (stop when stability and predictability meet threshold criteria). Prediction: PAB convergence detection produces models that generalize at least as well, with fewer wasted training steps.

**PAB Claim 3: "Class-wise progression reveals curriculum structure that endpoint metrics miss."**

- **Setup**: Use PAB domain progression to identify fragile domains — domains classified as "unstable" or "late" across multiple configurations.
- **Traditional evaluation**: Per-domain compile rates on frozen eval set. Identifies weak domains but not *why* they're weak.
- **PAB evaluation**: Domain progression trajectories reveal whether weak domains are consistently late-learned (hard but learnable), oscillating (training signal is contradictory), or regressing (catastrophic forgetting).
- **Stress test**: For each fragile domain identified by PAB, inject 50 targeted augmentation examples. Separately, inject 50 random augmentation examples (from any domain). Retrain and compare.
- **Comparison**: PAB-targeted augmentation (injecting examples for PAB-identified fragile domains) vs. random augmentation (same number of examples, randomly selected). Prediction: PAB-targeted augmentation stabilizes weak domains more efficiently — higher compile rate improvement per augmentation example.
- **Specifics**: The 8 ShortcutForge domains provide the class structure. PAB's domain_classification field provides the fragility signal.

**PAB Claim 4: "Feature importance consistency detects fragile representations."**

- **Setup**: Compute PAB feature importance consistency L — operationalized as ternary weight sign stability across checkpoints. For each weight, track whether its ternary value (+1, 0, -1) changes between consecutive checkpoints. L = fraction of weights that don't change, averaged over the last 20% of training.
- **Traditional evaluation**: No direct analog — traditional evaluation doesn't measure weight stability.
- **PAB evaluation**: High L (>0.9) predicts robust representations. Low L (<0.7) predicts fragile representations.
- **Stress test**: Compare high-L configurations against low-L configurations on:
  - *Adversarial vulnerability*: Fraction of eval examples that flip from compile-success to compile-failure when one word in the prompt is substituted with a synonym.
  - *Representation drift under continued training*: Train for 200 additional steps on the same data. Measure how much the bottleneck representation z̄ changes (cosine distance).
- **Comparison**: Prediction: high L → low adversarial vulnerability AND low representation drift. Additionally, ternary models should have higher L than continuous models (Section 2.3.1 predicts this — ternary weights enforce discrete commitment).
- **Statistical significance**: Report L values for all configurations, correlation with vulnerability and drift, and bootstrap CIs.

**PAB Claim 5: "Learning curve predictability measures how 'structured' training is."**

- **Setup**: Compute PAB predictability P = Var(ΔL_t) for all configurations.
- **Traditional evaluation**: Loss curve visualization (qualitative) or final loss value (uninformative about trajectory structure).
- **PAB evaluation**: Low P indicates structured, regular training. High P indicates erratic training.
- **Stress test**:
  - *Reproducibility*: Train each configuration with 3 different random seeds. Measure variance of final compile rate across seeds.
  - *Extrapolation accuracy*: At step 500 (halfway), use the trajectory so far to predict step-1000 compile rate via linear extrapolation of the compile rate trajectory. Measure prediction error.
- **Comparison**: Correlate P with (a) seed variance and (b) extrapolation error. Prediction: low P → low seed variance AND low extrapolation error. A configuration with structured, predictable training should be reproducible (same result regardless of seed) and forecastable (trajectory shape at step 500 predicts step 1000).
- **Statistical significance**: Spearman rho for P vs. seed variance and P vs. extrapolation error. Bootstrap CIs.

**PAB Claim 6: "Endpoint evaluation conflates approximate with exact correctness."**

- **Setup**: Separate configurations into PAB-stable (stability_mean < median) and PAB-unstable (stability_mean > median) groups.
- **Traditional evaluation**: Compare compile_strict rates between groups. Both groups may have similar mean compile rates.
- **PAB evaluation**: PAB predicts that stable configurations have more *consistent* compile rates — lower variance across seeds, across domains, and across eval subsets.
- **Stress test**: For each configuration, compute:
  - *Cross-seed compile rate variance*: Variance of compile_strict across 3 seeds.
  - *Cross-domain compile rate variance*: Variance of per-domain compile rates.
  - *Strict-permissive gap*: compile_permissive - compile_strict (large gap = many "almost right" outputs).
- **Comparison**: Prediction: PAB-stable configurations have lower cross-seed variance, lower cross-domain variance, and smaller strict-permissive gap. The strict-permissive gap is especially informative: a large gap means many outputs are "approximately correct" (pass permissive but fail strict), which endpoint evaluation treats as partial success. PAB's process metrics should predict which configurations produce genuinely correct outputs vs. outputs that happen to pass permissive thresholds.

**Head-to-Head Protocol**: For each of the six claims, the experimental writeup follows a standard structure:
1. **Setup**: Configuration selection, metric computation.
2. **Traditional evaluation**: What standard benchmarks show.
3. **PAB evaluation**: What PAB trajectory metrics show.
4. **Stress test**: Perturbation or generalization challenge.
5. **Comparison**: Side-by-side results with quantitative measures.
6. **Statistical significance**: Bootstrap 95% confidence intervals and p-values from appropriate nonparametric tests (permutation tests for paired comparisons, Spearman for rank correlations).

### 6.7 Ablation Matrix

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

### 6.8 Regression and Promotion Gates

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

  # Trajectory gates (PAB)
  pab_stability_mean: {max: 0.15}         # Training was stable, not chaotic
  pab_predictability_final: {max: 0.05}   # Loss trajectory was structured
  pab_tier1_converged_by: {max: 500}      # Structure learned in first half of training
  pab_no_domain_regression: {value: true} # No domain accuracy regressed in final 20%
  pab_crystallization_rate: {min: 0.001}  # Ternary weights are converging, not stuck

  # Behavioral gates
  behavioral_fingerprint_stability: {min: 0.85}  # Cross-seed fingerprint correlation
  restriction_site_clarity: {min: 3.0}  # Eigenvalue ratio of output variance
  pab_behavioral_correlation: {min: 0.5}  # PAB-behavioral Spearman rho
```

The trajectory gates prevent promoting a model that hits endpoint targets through chaotic, unreliable training. A model that achieves 95% compile rate via a stable, predictable trajectory is more trustworthy (and more likely to generalize) than one that oscillated wildly and happened to end at the same number.

The behavioral gates ensure that training quality translates to deployment predictability. A model whose behavioral fingerprint is unstable across seeds (below 0.85 correlation) may produce inconsistent results in deployment regardless of its endpoint metrics.

### 6.9 Mandatory Test Scenarios

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

**PAB trajectory comparison**: Track A should produce the smoothest, most stable PAB trajectories of all three tracks. The teacher provides clean, consistent training signal — if PAB shows chaotic dynamics even with clean data, the STE training protocol needs adjustment. Expect: low stability_mean, early tier-1 convergence, fast crystallization. The teacher's implicit curriculum (the order in which examples appear) should transfer to the student's trajectory — PAB domain progression should mirror the teacher's domain competency profile.

### 7.2 Track B: Direct Tiny-Specialist Training

**Method**: Train the Balanced Sashimi architecture directly on the existing 6,679-example training set (converted to typed IR format), without teacher distillation.

**Purpose**: Tests whether the architecture's structural inductive biases compensate for the lack of teacher-distilled training signal.

**Expected outcome**: Lower initial compile rates than Track A, but potentially better final performance if the negative learning and structural constraints provide stronger inductive bias than the teacher's implicit knowledge.

**PAB trajectory comparison**: Track B should show slower convergence but potentially more structured domain progression than Track A. Without teacher guidance, the architecture must discover its own curriculum — PAB's domain progression will reveal whether the structural decoder's inductive bias (ternary weights, tier decomposition) naturally discovers a curriculum (simple domains first, complex domains later) or learns all domains concurrently. Expect: higher stability_mean than Track A (noisier training signal), but potentially more structured tier-wise progression (Tier 1 before Tier 2 before Tier 3, since the architecture explicitly separates them).

### 7.3 Track C: From-Scratch Ablation

**Method**: Train a minimal Balanced Sashimi model with random initialization on both encoder and decoder. No pretrained encoder, no teacher distillation.

**Purpose**: Ablation to isolate the contribution of pretrained representations. If Track C approaches Track A performance, the pretrained encoder is unnecessary for this task — evidence that the task's intrinsic complexity is very low.

**Expected outcome**: Significantly worse than Tracks A/B, serving as a lower bound. However, if OOD detection and hard-negative separability metrics are competitive, that suggests the *structural* contributions (ternary, negative learning, typed IR) are carrying most of the weight.

**PAB trajectory comparison**: Track C should show the longest chaotic phase before stabilization, serving as the PAB baseline for "unguided learning." Without pretrained representations, the encoder must learn semantic similarity from scratch — PAB's representation evolution R(t) will show prolonged high values before stabilizing. Expect: high stability_mean, late tier-1 convergence, slow crystallization, and potentially chaotic domain progression (no natural curriculum emerges without pretrained features). Track C's PAB profile provides the reference point for "what learning looks like without architectural advantages" — improvements in Tracks A/B relative to Track C are attributable to the pretrained encoder and/or teacher distillation.

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
| PAB metrics don't correlate with deployment quality | Medium | This is itself a research finding worth documenting — it constrains PAB's applicability claims. If PAB-behavioral correlation (Section 6.5) is below threshold (rho < 0.3), proceed with endpoint-only evaluation and report the negative result. |
| Behavioral fingerprinting is too noisy for ternary models | Low | Ternary discreteness should help, not hurt (Section 2.5). If noisy despite discrete weights, the variance-mediated approach may need adaptation for step-function activation patterns. Report noise levels as empirical data for the behavioral verification research program. |
| Dual-validation scope creep | Medium | PAB validation is co-equal to architecture validation but must not dominate compute. If PAB experiments (Section 6.6) consume >30% of total compute budget, deprioritize Claims 4–6 and focus on Claims 1–3. The architectural result always takes priority over the evaluation-framework result. |

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

### 10.3 Relation to Process-Aware Evaluation Research

This project serves as a test case for PAB's theoretical framework. The Balanced Sashimi architecture is designed to be maximally transparent to trajectory analysis — its modular decomposition, tier-wise output structure, and ternary weight crystallization provide PAB with richer, more interpretable trajectory data than any monolithic architecture could offer.

If PAB's trajectory metrics are informative for this architecture — if they predict generalization, correlate with behavioral stability, and enable more efficient data curation — that validates PAB's core claim that process evaluation adds value beyond endpoint evaluation. The strength of the validation is proportional to the architecture's transparency: because the architecture *exposes* its learning process, PAB has the best possible data to work with. Success here establishes an upper bound on PAB's utility.

Conversely, if PAB's metrics are uninformative even for this architecture — if trajectory stability doesn't predict generalization, if crystallization rate doesn't correlate with reliability, if PAB-curated data doesn't improve outcomes — that constrains PAB's applicability claims. If the framework fails on its best test case, its claims for opaque architectures are weaker.

Either way, the result is valuable for the process-aware evaluation research program. Positive results motivate extending PAB to less transparent architectures (with appropriate adaptations for reduced trajectory visibility). Negative results motivate revising PAB's metric definitions or reconsidering its theoretical foundations. The dual-validation structure ensures that the PAB research program gains knowledge regardless of the architectural outcome.

### 10.4 Relation to Behavioral Verification Research

The ternary decoder's discrete weight space provides a natural test case for behavioral fingerprinting approaches from the Variance-Mediated research program (PoT/REV/HBT). The core question: does representational discreteness aid verification?

In continuous-weight architectures, behavioral fingerprints are derived from smooth activation distributions — fingerprinting requires statistical analysis to distinguish meaningful behavioral patterns from continuous noise. In ternary architectures, the weights are exactly {-1, 0, +1}, producing activation patterns that are step functions rather than smooth curves. This should produce "crisper" behavioral signatures — fingerprints with higher signal-to-noise ratio and more clearly defined restriction sites.

If the experiments confirm this (restriction site clarity is higher for ternary configurations than continuous configurations in the ablation), it supports the hypothesis that representational discreteness aids verification. This connects to broader questions about model governance: if discrete-weight architectures are inherently easier to fingerprint and verify, that's a reason to prefer them in safety-critical deployments, independent of performance considerations.

The connection between PAB and behavioral verification is the thread running through the project: training quality (PAB) → representation quality (interpretability) → deployment reliability (behavioral fingerprinting). Each link in this chain is testable within the experimental design.

---

## 11. References

### PAC Learning
- Valiant, L. (1984). A Theory of the Learnable. *Communications of the ACM*, 27(11), 1134–1142.

### Process-Aware Benchmarking
- Pal, P. (2025). PABKit: Process-Aware Benchmarking Toolkit. GitHub: parama/pabkit. MIT License.
  *Introduces trajectory-aware evaluation metrics (learning stability, generalization efficiency, rule evolution, class-wise progression) grounded in PAC learning theory extensions. We adapt PABKit's class-wise progression to tier-wise and domain-wise progression, and extend the stability metrics to address STE-specific training dynamics.*

### Behavioral Verification
- Variance-Mediated Behavioral Verification (PoT/REV/HBT research program). *Connects training-time process evaluation to deployment-time behavioral fingerprinting via representational stability analysis.*

### Information Theory
- Tishby, N., Pereira, F., & Bialek, W. (2000). The Information Bottleneck Method.

### Multi-Task Learning
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

### Sentence Encoders
- Wang, W., et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.

---

*Document version: 2.0. Updated 2026-02-20: Fundamental reframing with PAC/PAB as foundational motivation, behavioral verification integration, PAB empirical validation experiments, dual-stream evaluation framework.*
