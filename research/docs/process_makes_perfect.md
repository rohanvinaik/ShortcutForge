# Position: Process Makes Perfect — Transforming How We Benchmark Machine Learning Models

**Anonymous Authors**

*Preliminary work. Under review by the International Conference on Machine Learning (ICML). Do not distribute.*

---

## Abstract

Machine learning benchmarks have long measured success through the proxy of correctness, assessing whether models produce strictly defined answers on test sets. This approach works for tasks like classification where the answers aren't often complex, but is inadequate for higher-order tasks such as reasoning, decision-making, and structured generalization. Current benchmarks do not distinguish between models that learn meaningful abstractions and those that memorize, nor do they track how learning evolves over time.

This position paper argues that benchmarking must evolve in a way consistent with the field's foundations and shift to track learning, rather than data points we hope are the product of learning. We introduce **Process-Aware Benchmarking (PAB)**, a framework that evaluates models based on *learning trajectories* rather than static correctness. PAB aligns evaluation with formal computational learning theory, particularly PAC learning, and offers practical recommendations for integrating process-aware evaluation into widely used benchmarks. We also address concerns about computational feasibility and argue that AI benchmarking must evolve alongside AI itself. The future of evaluation should measure not only what a model gets right, but, in addition, *how it learns to get there.*

---

## 1. Introduction: Benchmarking Must Evolve with AI Progress

### 1.1. A Brief History of Benchmarking

Benchmarks have shaped how we evaluate machine learning models. Datasets like ImageNet (Deng et al., 2009) and GLUE (Wang et al., 2018) have standardized evaluation and provided clear performance targets, allowing researchers to compare models and track progress. These benchmarks have driven advances in image classification, language processing, and speech recognition.

Benchmarks do not just measure success—they define it. When correctness is the primary metric, models are optimized for correctness. This approach worked for first-order tasks, such as classification and sorting, where correctness is well-defined. However, as AI tackles higher-order problems, correctness alone is no longer enough. A model can generate the right answer while failing to generalize. Reinforcement learning agents can maximize reward while learning brittle policies (Gulcehre et al., 2020). Large language models can produce fluent text yet hallucinate facts (McIntosh et al., 2024). Correctness-based benchmarks provide no way to differentiate between true learning and heuristic-driven pattern matching.

### 1.2. Grounding Benchmarking in Learning Theory

If we are to evaluate intelligence correctly, we must align benchmarking with formal learning theory. **Benchmarks should reflect learning itself.** Machine learning belongs to a hierarchy of learning paradigms:

- **Computational Learning Theory (CoLT):** The broad framework for analyzing learning systems.
- **Exact Learning:** Models must perfectly reconstruct a function.
- **PAC Learning:** Models infer rules that generalize with high probability (Valiant, 1984).
- **VC Learning:** A subset of PAC learning that measures hypothesis complexity.
- **Statistical Learning Theory:** Provides probabilistic generalization guarantees.

Correctness-based benchmarks implicitly assume **Exact Learning**, where learning is defined as finding the single correct function. However, most real-world AI problems align better with **PAC Learning**, where generalization efficiency matters more than correctness alone. If PAC Learning is the right framework for AI, our benchmarks must reflect that.

### 1.3. Position: Benchmarking Must Measure Learning, Not Just Correctness

**This position paper argues that machine learning benchmarking must evolve beyond correctness-based evaluation and shift toward measuring the learning process itself.** Existing benchmarks, designed for first-order AI tasks, assess models based solely on static correctness. However, as AI systems tackle higher-order problems, this rigid evaluation paradigm no longer suffices.

We introduce **Process-Aware Benchmarking (PAB)**, a framework that evaluates models based on *learning trajectories* rather than final correctness. PAB tracks how models refine decision boundaries, develop structured representations, and improve generalization efficiency over time. Unlike previous benchmarks, which assume that learning is reducible to correctness, PAB aligns evaluation with formal computational learning theory.

In the sections that follow, we examine the limitations of correctness-based evaluation (Section 2), formalize the learning trajectory-based framework of PAB (Section 3), compare PAB to traditional benchmarks (Section 4), and explore its computational feasibility (Section 5). Finally, we discuss alternative views (Section 6) and conclude with recommendations for the future of AI benchmarking.

---

## 2. The Limitations of Correctness-Based Benchmarks

### 2.1. Why Correctness Was a Good Start, But Is No Longer Enough

Correctness-based benchmarks have driven the progress of machine learning for decades. They provided a **clear, structured way to measure progress**, allowing researchers to compare models, optimize architectures, and set performance milestones. Without them, we would not have seen the rapid advancements that followed datasets like ImageNet (Deng et al., 2009) or GLUE (Wang et al., 2018).

But benchmarks do not just measure success—they define it. When correctness is the primary measure of performance, models optimize for correctness. That works well for **first-order tasks** like classification and sorting, where correctness is well-defined. But as AI expands to **higher-order tasks**, correctness is no longer enough. A large language model can generate fluent text while hallucinating facts (McIntosh et al., 2024). A reinforcement learning agent can maximize reward while failing to generalize (Gulcehre et al., 2020). Correctness-based benchmarks provide no way to distinguish between models that learn structured generalization and those that exploit dataset-specific heuristics.

### 2.2. Benchmarks Cannot Distinguish Memorization from Learning

One fundamental issue with correctness-based evaluation is that it does not differentiate between a model that has truly learned generalizable patterns and one that has simply memorized its training data. Zhang et al. (Zhang et al., 2017) demonstrated that deep networks can achieve high accuracy even when trained on datasets where labels are randomly assigned, proving that correctness alone is not evidence of meaningful learning.

Further research reinforces this concern. McIntosh et al. (McIntosh et al., 2024) highlight how large language model (LLM) benchmarks fail to test for true reasoning, allowing models to appear more capable than they actually are. Similarly, Geirhos et al. (Geirhos et al., 2020) showed that image classification models often **rely on background cues rather than object features**, and McCoy et al. (McCoy et al., 2019) found that NLP models trained on large corpora frequently **exploit statistical patterns rather than learning syntactic or semantic structure**. These issues result in models that perform well under benchmark conditions but fail in real-world applications where shortcuts do not generalize.

### 2.3. Benchmarks Are Misaligned with Learning Trajectories

Correctness-based benchmarks evaluate models at a single endpoint, but real-world intelligence is a process. Unlike humans, who value their understanding over time, deep learning models are rarely assessed based on **how their representations evolve during training**. The assumption that final test accuracy is the best indicator of success ignores the fact that models with similar accuracy scores can have dramatically different learning trajectories.

Research has shown that models follow distinct **learning phases** during training, exhibiting predictable generalization behaviors (Schindler et al., 2023). However, existing benchmarks do not track these learning phases, providing no insight into the progression of training and eliminating the need for process-aware evaluation.

### 2.4. Robustness Failures and Distribution Shifts

Correctness-based benchmarks also fail to capture how a model will perform under distribution shift. Research (Recht et al., 2019) replicated ImageNet's dataset collection process years after its original release and found that model accuracy dropped significantly on newly collected data, despite the test set being drawn from the same distribution. Similarly, robustness studies in reinforcement learning reveal that policies trained for high performance can fail catastrophically when exposed to slightly modified conditions (Gleave et al., 2020).

Benchmarks also struggle with **adversarial robustness** (Goodfellow et al., 2015; Madry et al., 2018), which cause sharp drops in accuracy without corresponding changes in model confidence or representation structure (Szegedy et al., 2014). If evaluation is based only on correctness, benchmarks provide **false confidence in a model's ability to generalize safely.**

### 2.5. The Need for Process-Aware Benchmarking

Rather than evaluating models only on final outputs, a paradigm shift is needed. Benchmarks should track learning, ensuring that performance reflects structural generalization rather than dataset-specific heuristics. This shift aligns with computational learning theory, particularly PAC learning, which evaluates learning in terms of generalization probability rather than deterministic correctness.

**Process-Aware Benchmarking (PAB)** provides a principled way to assess models as they learn, requiring that a model learns structured rules rather than exploiting dataset artifacts. We now formalize PAB into a model evaluation framework built on trajectory-based evaluation (Raghu et al., 2017).

### 3. Process-Aware Benchmarking (PAB)

#### 3.1. Why We Need Process-Aware Benchmarking

Traditional benchmarks evaluate models at a single endpoint, ignoring the developmental process of learning. In contrast, PAB shifts evaluation toward continuous measurement of learning trajectories. Correctness tells us whether a model got the right answer, but it does not tell us how the model learned to do so.

If intelligence is a process, then evaluating AI requires tracking that process. Human learners do not master skills at a single endpoint—they develop understanding through experience, making gradual adjustments until behavior becomes truly generalizable—until it encounters an unfamiliar scenario. **Process-Aware Benchmarking (PAB)** addresses this by providing a systematic way to evaluate not just what models produce but also how they improve over time.

#### 3.2. PAB as a Theoretically Grounded Framework

To evaluate learning effectively, we must ground benchmarking in **computational learning theory**. Correctness-based benchmarks assume that learning is a process of finding a single correct function, akin to **Exact Learning**, where a model is expected to perfectly reconstruct a function. However, real-world AI aligns more closely with **PAC Learning**, which evaluates whether models infer rules that generalize with high probability (Valiant, 1984). This requires:

- **Exact Learning:** Models must perfectly reconstruct a function.
- **PAC Learning:** Models infer rules that generalize with high probability.
- **VC Learning:** A subset of PAC learning that measures hypothesis complexity.
- **Statistical Learning Theory:** Provides probabilistic generalization guarantees.

Correctness-based benchmarks evaluate models as if they must learn exact functions, but **most real-world AI tasks align better with PAC Learning**, where the goal is to generalize efficiently rather than to achieve perfect correctness. PAB shifts evaluation toward this paradigm, ensuring models are assessed based on how they develop learning trajectories over time.

#### 3.3. Mathematical Formulation of PAB

Process-Aware Benchmarking evaluates models not just on whether they are correct but on how they learn. We define a **learning trajectory** as:

$$\mathcal{T} = \{h_{t_0}, h_{t_1}, ..., h_{t_n}\}$$

where *h* represents the hypothesis a model forms at step *t* during training. A well-structured learning trajectory should exhibit the following properties:

1. **Learning Trajectory Stability:** A robust model should demonstrate a gradual and structured learning process. We define **stability** as the smoothness of the loss trajectory:

$$S(\mathcal{T}) = \frac{1}{n} \sum_{t=1}^{n} |R(h_{t+1}) - R(h_t)|$$  (2)

where smaller *S(T)* values indicate a stable learning trajectory.

2. **Generalization Efficiency:** Instead of evaluating generalization only at convergence, we define **instantaneous generalization efficiency** *G(t)* as:

$$G(t) = P_{train}(h_t) - P_{test}(h_t)$$  (3)

A well-trained model should maintain low *G(t)* throughout training, indicating consistent benchmarks on unseen data.

3. **Rule Evolution:** We measure the structured abstraction of features rather than abrupt shifts in representation. We measure **rule formation divergence** as:

$$R_{div} = \frac{1}{n} \sum_{t=1}^{n} ||h_t - h_{t-1}||$$  (4)

where *d_t* is the distance between model representations at time *t*. A model that learns structured rules rather than heuristics will show lower, more interpretable rule evaluation (Raghu et al., 2017).

#### 3.4. Connecting Process-Aware Learning to Human Learning

In human learning, expertise develops through structured exposure and refinement. Children learning language, for example, do not memorize sentences—they gradually refine grammatical rules (Lust et al., 2017). A process-aware benchmark should track whether models follow a similar structured trajectory. We define **learning curve predictability** as:

$$P_{learn} = \mathbb{E}\left[\frac{1}{n}\sum_{t=1}^{n}|R(h_{t+1}) - R(h_t)|\right]$$  (5)

where lower values indicate more predictable and stable learning, akin to structured human learning.

#### 3.5. Implementing Process-Aware Benchmarks

To make PAB into an ME evaluation, we propose the following experimental protocols:

1. **Checkpoint-Based Evaluation:** Store model snapshots throughout training to track representation evolution.
2. **Dynamic Curriculum Testing:** Increase task difficulty dynamically based on model evaluation performance.
3. **Adaptive Benchmarking Pipelines:** Modify test sets dynamically based on model learning progress.

**Algorithm 1** Process-Aware Benchmarking Evaluation:
```
Input: Training dataset D, Model M
for t = 1 to T do
    Train M on D_t
    Compute generalization G(h_t)
    Compute stability S(h_t)
    Log learning curve measurements {G_t, S_t, P_t}
end for
Return process-aware matrices S(T), G(T), P_t, and R_div
```

#### 3.6. Conclusion: Moving Towards a Learning-Centric Evaluation Paradigm

Correctness-based benchmarks fail to capture how models learn. PAB provides a structured, theoretically grounded approach to benchmarking that evaluates learning trajectories rather than static correctness. This shift ensures that AI evaluation aligns with **how models refine their understanding over time** rather than simply maximizing static metrics.

In the next section, we compare PAB to traditional benchmarks, illustrating where correctness-based evaluation falls short and how PAB provides deeper insights into model performance over time.

---

## 4. Comparing Process-Aware Benchmarking to Traditional Benchmarks

### 4.1. The Role and Limits of Correctness-Based Benchmarks

Correctness-based benchmarks have played an essential role in machine learning. They have standardized evaluation, driven competition, and enabled direct model comparisons. Datasets like ImageNet (Deng et al., 2009) and GLUE (Wang et al., 2018) have been instrumental in AI progress, setting clear performance targets that helped advance deep learning.

However, benchmarks do not just measure success—they follow it. Correctness-based evaluation confirms that models can perform specific tasks, but it does not assess whether models generalize well. To address this, PAB introduces a new evaluation approach that tracks learning trajectories rather than just correctness.

### 4.2. Core Differences Between Correctness-Based Benchmarks and PAB

Correctness-based benchmarks assess models at a single endpoint. They do not measure how a model refines its representations, adapts to new data, or stabilizes its decision boundaries over time. Process-Aware Benchmarking (PAB) moves beyond static correctness, tracking how a model's decision-making structure evolves in an interpretable way.

Process-aware benchmarking enables a more principled approach to interpretability by tracking feature importance changes throughout training.

### 4.3. Empirical Evidence Supporting Process-Aware Benchmarking

Several studies provide empirical evidence that learning trajectories contain valuable insights beyond what final correctness can measure. Research on internal model representations has shown (Raghu et al., 2017) that model representations evolve throughout training, and tracking these changes reveals whether models develop structured generalization or exploit spurious correlations.

Area learning research also supports this position. Finn et al. (Finn et al., 2017) demonstrated that models optimized for rapid adaptation exhibit structured changes in learning behavior that cannot be captured by state evaluation. Similarly, curriculum learning (Bengio et al., 2009) has shown that models trained on progressively harder tasks generalize better, suggesting that measuring *how* a model learns is as important as measuring final performance.

Correctness-based benchmarks can also be misleading due to variance, driven competition, and enabled direct model comparisons. Datasets like ImageNet (Deng et al., 2009) and GLUE (Wang et al., 2018) have been important performance targets that helped advance deep learning workflows.

---

## 5. Computational Feasibility and Implementation Strategies

A common argument against process-aware benchmarking is that it introduces additional computational overhead. Correctness-based benchmarks evaluate models at a single point in time, whereas PAB requires monitoring and storing model progress throughout training, which results in efficient checkpointing and sparse trajectory sampling being required.

Sparse checkpointing techniques, such as those proposed by Liu et al. (Liu et al., 2020), allow models to be evaluated at select intervals rather than continuously tracking each training step. This reduces storage and computational costs, while still capturing meaningful trends in learning trajectories. Additionally, neural scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022) suggest that learning trajectories follow predictable scaling properties, reducing unnecessary evaluations for models that fail to generalize.

### 5.1. Efficient Integration with Existing Training Pipelines

To minimize disruption to existing workflows, PAB can be implemented within standard deep learning training frameworks. We propose the following computational strategies:

- **Adaptive Checkpointing:** Instead of saving full model states at every epoch, training pipelines can implement dynamic checkpointing based on representation stability metrics such as SVCA (Raghu et al., 2017).
- **Dynamic Test Set Adaptation:** Benchmarks can progressively increase task difficulty dynamically, increasing complexity as the model progresses (Bengio et al., 2009).
- **Efficient Learning Curve Analysis:** Rather than evaluating models on a full test set at every checkpoint, benchmarks can use adaptive trajectory sampling techniques based on entropy-based sampling techniques (Gleave et al., 2020).

These techniques allow for efficient tracking of learning progress without incurring excessive computational costs.

### 5.2. Practical Trade-Offs and Solutions

Despite its benefits, implementing PAB introduces certain challenges:

- **Storage Requirements:** Tracking intermediate training states requires additional storage. However, solutions such as weight sharing (Graft & Tai, 2017) can significantly reduce storage requirements by compressing checkpoint representations.
- **Computational Complexity:** Monitoring representation shifts during training may increase training time. However, hardware-aware optimizations (Hoffmann et al., 2022) and hardware-aware optimization can mitigate these challenges.
- **Scalability for Large Models:** For large-scale foundation models, logging full training trajectories is computationally prohibitive. Instead, sparse trajectory evaluation, evaluating only critical phases of model progression, can minimize unnecessary computations.

The feasibility of PAB depends on advancing advances in adaptive evaluation, dynamic evaluation, and representation analysis. The next section explores how process-aware benchmarking can improve AI safety, robustness, and interpretability.

---

## 6. AI Safety, Robustness, and Interpretability

Correctness-based benchmarking has long been criticized for its inability to capture model **fragility** and model **interpretability** in real-world settings. Models that score highly on static benchmarks frequently fail under adversarial conditions, domain shifts, or real-world deployment scenarios. Process-Aware Benchmarking (PAB) offers a potential solution by providing deeper insight into how models develop over time, incorporating stable learning structures rather than brittle heuristics.

### 6.1. Detecting Model Fragility Through Learning Trajectories

Robustness is a key concern in AI real-world applications, and PAB provides a way to measure whether models develop stable feature representations that resist adversarial attacks and distribution shifts (Szegedy et al., 2014; Recht et al., 2019). Models that score highly on static benchmarks frequently fail under adversarial conditions, domain shifts, or real-world deployment scenarios. Process-Aware Benchmarking (PAB) offers a potential solution by providing deeper insight into how models develop over time, incorporating stable learning structures rather than brittle heuristics.

PAB provides a mechanism for detecting model fragility by analyzing model stability during training. Instead of only measuring final performance, we track *large fluctuations in Δh*, where:

$$\Delta h_t = ||h_t - h_{t-1}||$$  (6)

where *h_t* represents a model's latent feature representations at step *t*. Large fluctuations in *Δh_t* at late *t* indicate susceptibility to adversarial perturbations, raising questions about their reliability in safety-critical AI applications.

### 6.2. Generalization Under Distribution Shifts

Correctness-based benchmarks provide no measure of how models respond to controlled distribution shifts throughout training. Recht et al. (Recht et al., 2019) demonstrated a significant performance degradation when evaluated on a new but related test set, and stability in reinforcement learning policies trained via self-play (OpenAI Five, et al., 2020), allows benchmarks to track generalization patterns across diverse environments, as PAB metrics predict performance drops more accurately than test accuracy alone.

### 6.3. Improving Model Interpretability

Interpretability remains one of the most critical challenges in deep learning (Rudin et al., 2019; Ribeiro et al., 2016). Current explainability techniques such as LIME and SHAP evaluate a model's decision-making structure at a single moment in time. However, **process-aware benchmarking enables a more principled approach** to interpretability by tracking feature importance changes throughout training.

$$L = \frac{1}{n} \sum_{t=1}^{n} (w_t - \bar{w})^2$$  (7)

where *L* measures the consistency of important features throughout training. A model with low *L* maintains consistent feature importance across all training time is more interpretable than one that abruptly shifts its decision rules at late training stages.

### 6.4. AI Safety Implications of Process-Aware Benchmarking

By evaluating learning trajectories rather than endpoint accuracy alone, PAB improves AI safety in two key ways:

- **Early Failure Detection:** Unstable learning trajectories indicate when a model is failing to generalize properly, allowing for early intervention.
- **Improved Trustworthiness:** Models that refine their decision-making in a structured way are more likely to safely rely on brittle heuristics, making them more predictable and reliable.

---

## 7. Addressing Alternative Views

While Process-Aware Benchmarking (PAB) offers a promising alternative to correctness-based evaluation, several potential objections arise regarding its fairness, objectivity, and practicality. In this section, we address these concerns and provide counterarguments.

### 7.1. Objection 1: Correctness-Based Benchmarks Are Not Effective

A common argument in favor of correctness-based benchmarking is that it has driven major advances in machine learning. Benchmarks like ImageNet (Deng et al., 2009), GLUE (Wang et al., 2018), SuperGLUE (Wang et al., 2019) and others (Hendrycks et al., 2021) have led to major advances in vision, language, and reasoning tasks. Some argue that adding complexity to benchmarking may not provide additional benefits.

**Response:** While these benchmarks have been effective for tracking progress, they are fundamentally correctness-based evaluation and true correctness orientation (Zhang et al., 2017). A model that achieves high test accuracy may do so by exploiting spurious dataset-related correlations rather than developing structured representations. PAB complements correctness-based evaluation by providing insights into *how* models develop generalization capacity over time, rather than simply measuring whether they succeed on static test sets.

### 7.2. Objective 2: Process-Aware Benchmarking Introduces Subjective Evaluation Standards

A key criticism of PAB is that evaluating learning trajectories rather than final correctness introduces **subjectivity** into model assessment. Unlike accuracy-based benchmarks, which are based on simple measurement, trajectory-based evaluation may be viewed as inconsistent.

**Response:** While it is true that process-aware evaluation is more complex, it is not necessarily subjective. Trajectory-based metrics—such as representation stability, rule formation divergence, and generalization efficiency—can be computed objectively from model checkpoints. These metrics are grounded in formal learning theory and provide more information about model behavior than a single accuracy score.

---

## 8. Conclusion: Redefining Benchmarking for the Future of AI

Machine learning benchmarking has long been dominated by correctness-based evaluation. Models are assessed solely on whether they produce the right answer at a fixed test set. While this approach has tracked AI progress, it has also led to systematic failures in distinguishing memorization from true generalization, exposing spurious correlations, and measuring model robustness in real-world tasks.

The future of AI evaluation lies in evaluating learning trajectories rather than endpoint accuracy. By building process-based evaluation into ML benchmarking, we can develop AI systems that are not just more accurate but **interpretable, reliable, and aligned with human reasoning workflows**.

### Key Takeaways

This paper has argued for a fundamental shift in benchmarking: shifting evaluation metrics toward **learning trajectories** rather than final static performance metrics. Our key contributions include:

- Identifying the limitations of correctness-based evaluation and its failure to capture meaningful structural generalization.
- Defining learning trajectories and proposing new evaluation metrics, including model evaluation stability.
- Comparing PAB with traditional benchmarks, demonstrating its effectiveness in assessing role formation efficiency and learning efficiency.
- Discussing practical implementation strategies and computational feasibility methods for PAB, showing that trajectory-based evaluation can be integrated into existing ML pipelines.
- Proposing a roadmap for process-aware evaluation for AI safety, interpretability, and robustness.
- Providing empirical insights, outlining future empirical work that can establish PAB's effectiveness.

### 8.2. The Broader Impact of Process-Aware Benchmarking

As AI systems continue to scale, their evaluation must grow alongside them. The next frontier of machine learning requires not merely improving static AI performance metrics but understanding how models learn. Process-aware benchmarking represents a necessary step toward achieving this goal.

---

## 5. Computational Feasibility and Implementation Strategies

*(Appendix sections)*

### A. Benchmark Summary and Evaluation Metrics

This appendix provides additional context on the dominance of correctness-based evaluation in modern machine learning. Table 2 categorizes major benchmarks by task type, while Table 3 describes the most commonly used evaluation standards.

Table 2 highlights the diversity of machine learning benchmarks but also illustrates the widespread reliance on correctness-based evaluation. While some benchmarks (e.g., HELM) incorporate additional evaluation dimensions such as fairness and robustness, most remain centered on single-point correctness assessments.

Table 3 describes common evaluation standards and their use cases. Most of these metrics measure correctness-based performance, providing little insight into how models develop structural generalization or refine their internal representations over time. This underscores the need for **Process-Aware Benchmarking (PAB)**, which explicitly evaluates learning trajectories rather than just final performance.

---

**Table 1: Comparison of Process-Aware Benchmarking (PAB) and Traditional Correctness-Based Benchmarks**

| Evaluation Aspect | Correctness-Based Benchmarks | Process-Aware Benchmarking (PAB) |
|---|---|---|
| Evaluation Focus | Static correctness on test set | Learning trajectory over training |
| Generalization Assessment | Single-point evaluation | Longitudinal tracking of role evolution |
| Robustness Testing | Often neglected | Adaptive testing during learning |
| Interpretability | Feature attributions | Tracking representation shifts over time |
| Computational Cost | Single evaluation at convergence | Trajectory tracking throughout training |
| Optimization Goal | Maximizes overfitting to test set | Rewards structured generalization |

---

**Table 2: Major Machine Learning Benchmarks by Task Type**

| Benchmark | Task Type | Primary Evaluation Metric |
|---|---|---|
| ImageNet (Deng et al., 2009) | Image classification | Accuracy |
| COCO (Lin et al., 2014) | Object detection | Precision, recall, mAP |
| GLUE (Wang et al., 2018) | NLP benchmark suite | Accuracy, F1-score |
| SuperGLUE (Wang et al., 2019) | NLP understanding | Accuracy, F1-score |
| SQuAD (Rajpurkar et al., 2016; 2018) | Question answering | Exact match, F1-score |
| MS MARCO (Nguyen et al., 2016) | Information retrieval | MRR, NDCG |
| KITTI (Geiger et al., 2012) | Autonomous driving | Precision, recall, mAP |
| LibriSpeech (Panayotov et al., 2015) | Speech recognition | Word error rate (WER) |
| MLPerf (Mattson et al., 2020) | Hardware benchmark | Latency, throughput |
| HELM (Liang et al., 2022) | Language model eval | Robustness, fairness, calibration |

---

**Table 3: Explanation of Common Machine Learning Evaluation Metrics**

| Metric | Description |
|---|---|
| Accuracy | Ratio of correctly predicted instances to total instances. Common in classification tasks. |
| Precision | Ratio of correctly predicted positives to total predicted positives. Important in information retrieval. |
| Recall (Sensitivity) | Ratio of correctly predicted positives to all actual positives. Used when false negatives are costly. |
| F1-Score | Harmonic mean of precision and recall. Used when classification has class imbalances. |
| Mean Absolute Error (MAE) | Average of absolute differences between predicted and actual values. Used in regression. |
| Mean Squared Error (MSE) | Average of squared differences between predicted and actual values. Penalizes larger errors more than MAE. |
| Mean Reciprocal Rank (MRR) | Average of reciprocal ranks of results for queries. Applied in information retrieval. |
| Normalized Discounted Cumulative Gain (NDCG) | Measure of ranking quality that accounts for position of correct results. Used in search ranking. |
| Word Error Rate (WER) | Ratio of incorrect words (substitutions, deletions, insertions) to total words. Used in speech recognition. |
| Perplexity | Measure of how confidently a model predicts text per unit of time. Used in language benchmarking. |
| Calibration Error | Measure of how well model confidence scores align with actual probabilities. Particularly useful for uncertainty/worthiness assessment. |
| Fairness Metrics | Metrics designed to evaluate bias and disparate impact in ML models, particularly in decision-making tasks. |

---

### B. Suggestions for Integration with Existing ML Frameworks

To ensure the practical adoption of Process-Aware Benchmarking (PAB), it is essential to integrate it into widely used machine learning frameworks and evaluation tools. In this section, we outline strategies for incorporating PAB into existing training workflows.

#### B.1. Leveraging Training Frameworks for Process Tracking

Most modern ML frameworks already support intermediate model evaluation. PAB can be integrated into existing training pipelines using checkpointing and evaluation tools. In particular:

- **PyTorch:** Using PyTorch's built-in `forward_hooks` and `backward_hooks`, we can log intermediate activations and monitor representational shifts during training (Paszke et al., 2019).
- **TensorFlow:** TensorFlow's `fit.summary` and `checkpointing` allow for lightweight monitoring of training progress, enabling efficient implementation of trajectory-based evaluation.
- **JAX:** JAX's functionally differentiable programming style supports checkpointing and tracking changes in model parameters over time.

These tools allow us to integrate process-aware evaluation without modifying the core architecture of existing ML training pipelines.

#### B.2. Adapting PAB for Benchmarking Suites

To ensure widespread adoption, PAB should be integrated into established benchmarking ecosystems. Below, we describe how PAB can be incorporated for major benchmarking frameworks:

- **MLPerf:** MLPerf currently evaluates models based on static inference latency and throughput. PAB can be incorporated by incorporating trajectory logging to track representational maturity over time.
- **Dynabench:** Dynabench enables dynamic benchmarking by incorporating human-in-the-loop evaluation. PAB can enhance Dynabench by tracking whether models improve rule-based generalization across diverse iterative updates.
- **HELM (Holistic Evaluation of Language Models):** HELM evaluates large language models across multiple dimensions, including fairness and bias. PAB can add an additional dimension that tracks learning efficiency throughout fine-tuning.

By integrating learning trajectory analysis into these frameworks, we can ensure that process-aware evaluation is scalable and accessible to researchers and practitioners.

#### B.3. Designing Efficient PAB Infrastructure

While trajectory-based benchmarking introduces additional computational requirements, recent advances in model introspection and efficient storage techniques help mitigate these concerns:

- **Sparse Checkpointing:** Instead of saving all model states, benchmarks can log snapshots at key learning milestones, reducing storage requirements while preserving trajectory insights (Liu et al., 2020).
- **Lightweight Feature Probing:** Rather than fully re-evaluating models at every checkpoint, feature importance tracking via SVCA (Raghu et al., 2017) can efficiently monitor representational shifts.
- **Dynamic Benchmark Adaptation:** Adaptive difficulty scaling can be integrated into benchmarks, ensuring that models are evaluated on progressively harder tasks (Bengio et al., 2009).

These optimizations allow process-aware benchmarking to be implemented without significantly increasing computational costs.

#### B.4. Implementation Roadmap

To facilitate adoption, we propose the following roadmap for integrating PAB into existing ML frameworks:

1. **Short-Term:** Implement lightweight trajectory tracking tools (e.g., PyTorch Hooks) to monitor representation evolution in deep networks.
2. **Mid-Term:** Integrate process-aware evaluation modules into existing benchmarks like MLPerf and HELM, ensuring compatibility with industry standards.
3. **Long-Term:** Establish new benchmarks that are fully designed for process-aware evaluation, incorporating dynamic test sets and adaptive learning curve assessments.

By following this roadmap, we can transition machine learning evaluation from a static correctness-based paradigm to a more dynamic, learning-centric approach.

#### B.5. Conclusion: The Path Forward for Machine Learning Evaluation

The successful integration of PAB into existing frameworks will require collaboration between researchers, industry practitioners, and the benchmarking organizations. As ML systems become increasingly complex, the need for process-aware evaluation will grow. By embedding trajectory-based assessment into ML workflows, we can ensure that future benchmarks provide more reliable insights into model learning dynamics, robustness, and interpretability.

---

### C. Roadmap for Future Experimental Validation

While the theoretical and practical arguments for Process-Aware Benchmarking (PAB) are compelling, empirical validation is necessary to establish its effectiveness in real-world settings. This section outlines a roadmap for empirical validation of PAB, focusing on its ability to measure structured learning, distinguish between memorization and generalization, and improve benchmarking efficiency.

#### C.1. Establishing the Empirical Case for PAB

A strong theoretical foundation is not enough. For Process-Aware Benchmarking (PAB) to gain adoption, it must demonstrate clear empirical benefits. This requires running controlled experiments to show that PAB identifies limitations that traditional benchmarks miss.

Three key areas must be explored:

- **Do learning trajectories reveal insights beyond correctness?** If two models achieve the same accuracy but one shows a more structured learning trajectory, does that model generalize better?
- **Does PAB improve robustness evaluation?** If models trained with PAB-based feedback perform better under distribution shifts, that is strong evidence in favor of structured learning.
- **How does PAB interact with different model architectures?** Some architectures may naturally exhibit more structured learning and others may need explicit mechanisms.

#### C.2. Evaluating Learning Trajectories in Vision and NLP

One of the primary advantages of PAB is its ability to track the evolution of model representations over time. A key experiment to validate this claim involves comparing standard benchmarking results with trajectory-based evaluation on established datasets.

- **Vision Tasks:** Train image classification models on ImageNet (Deng et al., 2009) while physically measuring representational shifts over time, using metrics such as centered kernel alignment (CKA).
- **Language Tasks:** Train transformer-based NLP models on SuperGLUE (Wang et al., 2019) and track how their syntactic and semantic representations evolve.

By comparing learning trajectory with trajectory-based metrics such as representation stability and generalization efficiency (Raghu et al., 2020; Hoffmann et al., 2022), we can determine whether PAB provides a more informative measure of learning than simply tracking accuracy.

#### C.3. Benchmark and Rule Formation

To test whether PAB successfully differentiates between models that generalize and those that memorize, we propose the following protocol:

1. Train deep learning models on datasets with controlled spurious correlations (e.g., Waterbirds dataset for bias evaluation (Sagawa et al., 2019)).
2. Evaluate models using traditional correctness-based benchmarks and compare them to PAB metrics such as rule formation divergence:

$$R_{div} = \frac{1}{n}\sum_{t=1}^{n}||h_t - h_{t-1}||_2$$  (8)

3. Analyze whether models that exhibit stable, structured learning trajectories perform better on out-of-distribution generalization tasks.

If models ranked highly by PAB outperform models ranked by correctness-based benchmarks on unseen test sets, it would provide strong empirical validation for process-aware evaluation.

#### C.4. Assessing Robustness Through Adversarial and Distribution Shift Testing

Robustness is a key concern in AI real-world applications, and PAB provides a benchmark to measure whether models develop stable feature representations that resist adversarial attacks and distribution shifts (Szegedy et al., 2014; Recht et al., 2019).

- **Adversarial Testing:** Evaluate model robustness by tracking stability of decision boundaries throughout training. Stable representations should correlate with adversarial vulnerability.
- **Domain Adaptation:** Test models on datasets with domain shifts, such as WILDS (Koh et al., 2021), and compare whether PAB metrics predict performance drops more accurately than test accuracy alone.

By designing benchmarks that track generalization patterns across diverse environments, we can ensure that PAB improves real-world AI reliability.

#### D. Roadmap for Future Experimental Validation

Adopting PAB in practice requires a structured rollout:

- **Phase 1 (Short-Term):** Implement PAB tracking in small-scale benchmarks and synthetic datasets to validate trajectory-based evaluation of robustness and generalization.
- **Phase 2 (Mid-Term):** Conduct large-scale experiments on diverse benchmarks to measure PAB's impact on robustness and generalization.
- **Phase 3 (Long-Term):** Integrate PAB into industry-standard benchmarks, making it a recognized evaluation metric.

#### D.1. Conclusion

Process-aware benchmarking must be tested before it can be widely adopted. By running controlled experiments in vision, NLP, and reinforcement learning, we can provide concrete evidence that tracking learning trajectories improves AI evaluation. The final section discusses the broader impact of this shift and the future of AI benchmarking.

---

## References

- Bengio, Y., Louradour, J., Collobert, R., and Weston, J. Curriculum learning. *International Conference on Machine Learning (ICML)*, 2009.
- Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2009.
- Finn, C., Abbeel, P., and Levine, S. Model-agnostic meta-learning for fast adaptation of deep networks. *International Conference on Machine Learning (ICML)*, 2017.
- Geirhos, R., Lena, P., and Ulman, R. Are we ready for autonomous driving? The KITTI vision benchmark suite. *2012 IEEE Conference on Computer Vision and Pattern Recognition*. Precision: 40, June 16-21, 2012. doi: 10.1109/CVPR.2012.6248074.
- Geirhos, R., Jacobsen, J.H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., and Wichmann, F.A. Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11):665–673, 2020.
- Gleave, A., Dennis, M., Wild, C., Kant, N., Levine, S., Russell, S., and Wechel, W. Adversarial policies: Attacking deep reinforcement learning. In *Proceedings of the 8th Annual International Conference on Learning Representations*, Addis Ababa, Ethiopia, April 26-30, 2020.
- Gulcehre, C., Wang, Z., Novikov, A., Paine, T.L., Gomes, G., Zolna, K., Agarwal, R., Merel, J.S., Ergul, D., Bohm, A., Deng, L., Paduraru, C., Dulac-Arnold, G., Li, J., Norouzi, M., Hoffman, J., Heess, N., and de Freitas, N. RL unplugged: A suite of benchmarks for offline reinforcement learning. *Advances in Neural Information Processing Systems*, volume 33, pp. 7248–7259, 2020.
- Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring mathematical problem solving using the math dataset. *arXiv preprint arXiv:2103.03874*, 2021.
- Kaplan, J., McCandlish, S., Henighan, T., Brown, T.B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models. *arXiv preprint arXiv:2001.08361*, 2020.
- Koh, P.W., Sagawa, S., Marklund, H., Xie, S.M., Zhang, M., Balsubramani, A., Hu, W., Yasunaga, M., Phillips, R.L., Gao, I., Lee, T., David, E., Stavness, I., Guo, W., Earnshaw, B., Haque, I., Beery, S., Lees, J., Fang, A., Karthikeyan, K., Anand, T., Goel, K., Sagawa, S., Raghunathan, A., Koh, P.W., Liang, P., and Levine, S. WILDS: A benchmark of in-the-wild distribution shifts. In *Proceedings of the 38th International Conference on Machine Learning*, 2021.
- Langley, P. Crafting papers on machine learning. In Langley, P. (Ed.), *Proceedings of the 17th International Conference on Machine Learning (ICML 2000)*, pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.
- Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., Narayanan, D., Wu, Y., Kumar, A., Newman, B., Yuan, B., Yan, B., Zhang, C., Cosgrove, C., Manning, C.D., Re, C., Achaiam, D., Han, E., Goel, E., Doshi-Velez, F., Ren, H., Leskovec, J., Yang, J., Ching, J., Choi, K., Srinivasan, K., Suresh, L., Xing, E., Zhiyuan, L., Roberts, M., Hashimoto, T., Zha, H., and Bommasani, R. Holistic evaluation of language models. *Transactions on Machine Learning Research (TMLR)*, 2022.
- Lin, T., Maire, M., Belongie, S.J., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C.L. Microsoft COCO: Common objects in context. In *Computer Vision — ECCV 2014*, Zurich, Switzerland, September 6-12, 2014. *Proceedings, Part V*, pp. 740–755, 2014. doi: 10.1007/978-3-319-10602-1.
- Liu, L., Jiang, H., He, P., Chen, W., Liu, X., Gao, J., and Han, J. On the variance of the adaptive learning rate and beyond. In *Proceedings of the 8th International Conference on Learning Representations (ICLR)*, 2020.
- Lust, B., Flynn, V., Foley, C., and Gupta, R. Language and mind. In Lust, B. (Ed.), *Child Language: Acquisition and Growth*, pp. 1–75. Cambridge University Press, 2017.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. Towards deep learning models resistant to adversarial attacks. In *Proceedings of the 6th International Conference on Learning Representations*, 2018.
- McCoy, T., Pavlick, E., and Linzen, T. Right for the wrong reasons: Diagnosing syntactic heuristics in natural language inference. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, Florence, Italy, July 28, 2019. doi: 10.18653/v1/P19-1334.
- McIntosh, T.R., Ruder, S., Raffel, C., Liu, Y., Xin, L., Fang, L., and Ye, J. Inadequacies of large language model benchmarks in the era of generative artificial intelligence. *arXiv preprint arXiv:2402.09748*, 2024.
- Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S., Majumder, R., and Deng, L. MS MARCO: A human generated machine reading comprehension dataset. *arXiv preprint arXiv:1611.09268*, 2016.
- Panayotov, V., Chen, G., Povey, D., and Khudanpur, S. LibriSpeech: An ASR corpus based on public domain audio books. In *2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2015.
- Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., and Chintala, S. PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32, 2019.
- Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J. SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In *Proceedings of the 31st International Conference on Neural Information Processing Systems (NeurIPS 2017)*, pp. 6078–6087, 2017.
- Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing*, pp. 2383–2392, 2016.
- Recht, B., Roelofs, R., Schmidt, L., and Shankar, V. Do imagenet classifiers generalize to imagenet? In *Proceedings of the 36th International Conference on Machine Learning*, 2019.
- Ribeiro, M.T., Singh, S., and Guestrin, C. "Why should I trust you?": Explaining the predictions of any classifier. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016.
- Rudin, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5):206–215, 2019.
- Sagawa, S., Koh, P.W., Hashimoto, T.B., and Liang, P. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2020.
- Schindler, A., Lidy, T., and Rauber, A. Investigating learning trajectories in deep learning models. *arXiv preprint arXiv:2308.12255*, 2023.
- Szegedy, C., Zaremba, W., Sutskever, I., Bruna, J., Erhan, D., Goodfellow, I., and Fergus, R. Intriguing properties of neural networks. In *Proceedings of the International Conference on Learning Representations (ICLR)*, 2014.
- Valiant, L.G. A theory of the learnable. *Communications of the ACM*, 27(11):1134–1142, 1984.
- Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In *Proceedings of the Workshop: Analyzing and Interpreting Neural Networks for NLP, at EMNLP 2018*, 2018.
- Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
- Zhang, C., Bengio, S., Hardt, M., Recht, B., and Vinyals, O. Understanding deep learning requires rethinking generalization. In *Proceedings of the 5th International Conference on Learning Representations*, Toulon, France, May 22-24, 2017.
