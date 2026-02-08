---
layout: blog_post
title: "RL Excursions During Pre-Training: How Early Is Too Early for On-Policy Learning?"
date: 2026-02-08
published: false
excerpt: "We study what happens if we introduce on-policy reinforcement learning (RL) much earlier than the standard LLM pipeline. In a controlled math setting, RL can improve reasoning surprisingly early, sometimes matches the SFT→RL pipeline, and can either sharpen or expand the model’s output distribution depending on the training setup."
---

*This post is a blog-format companion to our paper **“RL Excursions During Pre-Training: How Early Is Too Early for On-Policy Learning?”**.*

<!-- Optional: add links once public -->
<!-- Paper PDF: /assets/papers/rl-excursions-pretraining.pdf -->
<!-- Code: https://... -->

## Table of contents

- [Motivation](#motivation)
- [What we study](#what-we-study)
- [Experimental setup](#experimental-setup)
  - [Pretraining checkpoints and base model](#pretraining-checkpoints-and-base-model)
  - [Post-training data and evaluation](#post-training-data-and-evaluation)
  - [Three training pipelines](#three-training-pipelines)
- [Main results](#main-results)
  - [RL is effective early in pretraining](#rl-is-effective-early-in-pretraining)
  - [Where early RL struggles: limitations on MATH](#where-early-rl-struggles-limitations-on-math)
- [Sharpening vs. expansion: two opposing RL behaviors](#sharpening-vs-expansion-two-opposing-rl-behaviors)
  - [Standard pipeline: sharpening](#standard-pipeline-sharpening)
  - [Direct RL: expansion](#direct-rl-expansion)
  - [Brittleness on very early checkpoints](#brittleness-on-very-early-checkpoints)
- [Rollout budgets for early RL](#rollout-budgets-for-early-rl)
  - [Why rollouts matter early](#why-rollouts-matter-early)
  - [Experimental design for rollout scaling](#experimental-design-for-rollout-scaling)
  - [Results: sample vs compute efficiency trade-off](#results-sample-vs-compute-efficiency-trade-off)
- [Prior work](#prior-work)
- [Discussion and future directions](#discussion-and-future-directions)
- [Appendix: additional training details and ablations](#appendix-additional-training-details-and-ablations)
  - [A.1 RL training convergence](#a1-rl-training-convergence)
  - [A.2 Seed dependence](#a2-seed-dependence)
  - [A.3 SFT convergence](#a3-sft-convergence)
  - [A.4 Evaluating base checkpoints with n-shot prompting](#a4-evaluating-base-checkpoints-with-n-shot-prompting)
- [References](#references)

---

## Motivation

The standard training pipeline for Large Language Models (LLMs) proceeds sequentially through **pretraining**, **supervised fine-tuning (SFT)**, and **reinforcement learning (RL)**.

Pretraining and SFT typically use a **Next-Token Prediction (NTP)** objective via cross-entropy loss. This corresponds to approximating a static, external dataset—effectively an **off-policy** regime.

RL, in contrast, uses a **policy optimization** objective and learns from the model’s **own** generations—an **on-policy** regime.

The use of these two distinct objectives raises a basic but underexplored question:

> At what point during training does an LLM become capable of learning from its own generations?

Motivated by growing interest in RL-style pretraining, we investigate the transition between off-policy and on-policy objectives. Our guiding question is:

> **How and when should an RL objective be used in LLM training?**

We focus on **math reasoning**, which provides a controlled setting with unambiguous, verifiable rewards.

---

## What we study

We ask what happens if we introduce RL **earlier** than the conventional pipeline.

We:
- pretrain a **1B parameter** model from scratch,
- save intermediate checkpoints $M_t$ throughout pretraining,
- and at each checkpoint $M_t$, compare three “post-training” pipelines.

We focus on **Reinforcement Learning from Verifiable Rewards (RLVR)** using **Group Relative Policy Optimization (GRPO)** and the **VeRL** codebase.

---

## Experimental setup

### Pretraining checkpoints and base model

#### Why we pretrain our own model

Open-source suites with checkpoints exist, but they often use internet-scale corpora on the order of trillions of tokens, making it difficult to precisely measure how much math and reasoning content is in the pretraining mix. Instead, we pretrain our own model to maintain control over data exposure.

#### Data: DOLMino (high-quality portion of OLMo2 pretraining mix)

We use the **high-quality portion** of the OLMo2 pretraining mix, **DOLMino**, which contains **50B tokens** and includes:
- Wikipedia (**7%**)
- high-quality web data (**60%**, from DCLM and FLAN)
- high-quality math data (**20%**)
- other reasoning/code data such as StackExchange (**2%**) and STEM papers (**5%**)

(Dataset identifier: `allenai/dolmino-mix-1124`.)

#### Model and training

We use the **OLMo2** architecture and training infrastructure, and pretrain:
- a **1B** parameter decoder-only model
- on **50B tokens** (\(\sim 2.5\times\) Chinchilla-optimal)
- optimizer: **AdamW**
- LR schedule: **cosine decay**
- peak LR: \(4 \times 10^{-4}\)
- sequence length: **4096**
- batch size: **512**

We save intermediate checkpoints $M_t$ at different training stages (measured in pretraining tokens).

---

### Post-training data and evaluation

#### Training data: OpenMathInstruct

For both SFT and RL we use **OpenMathInstruct**, which consists of:
- math questions,
- multiple ground-truth solutions per question.

SFT trains on the dataset’s **ground-truth solutions**.

RL trains on **model-generated solutions**, scored using a verifiable reward based on the **final answer**.

OpenMathInstruct contains two main categories:
- questions inspired by **MATH** (competition-level; harder; majority)
- questions inspired by **GSM8K** (grade-school; easier; minority)

#### Two experimental settings

We run two settings to probe different aspects of early RL:

1. **Training on the GSM8K-inspired subset** of OpenMathInstruct (base checkpoints have non-trivial performance).
2. **Training on the full OpenMathInstruct** (MATH-heavy; remains challenging even for later checkpoints).

#### Evaluation benchmarks and metric

- For GSM8K-subset training, we evaluate on **GSM8K**.
- For full OpenMathInstruct training, we evaluate on **MATH**.

We report **pass@k** (for \(k \in \{1, 8, 32\}\)) at temperature \(T = 0.6\).
pass@k estimates the probability of obtaining at least one correct response when sampling \(k\) responses.

#### Prompting detail for base checkpoints

Pretraining checkpoints $M_t$ have only seen pretraining corpora and cannot consistently follow QA instructions. Therefore:
- we evaluate $M_t$ using **8-shot** prompts (in-context examples).

For post-trained models:
- we apply a formatting reward during RL and format OpenMathInstruct into QA format for SFT,
- so these models can follow the format without in-context examples,
- and we default to **0-shot** evaluation.

(We also include an appendix section below on the n-shot prompting ablation.)

---

### Three training pipelines

Let:
- $M_t$ denote the pretraining checkpoint at step \(t\),
- $M_t$ denote the final fully-pretrained model.

At each checkpoint $M_t$, we train and compare:

1. **RL only** ($M^{RL}_t$)  
   Start from $M_t$ and train with the RL objective until convergence.

2. **SFT only** ($M^{SFT}_t$)  
   Start from $M_t$ and perform SFT on ground-truth solutions until convergence.

3. **Gold standard pipeline** ($M^{SFT\rightarrow RL}_t$)  
   First train SFT to obtain $M^{SFT}_t$, then run GRPO on the same set of questions.

Why these baselines matter:
- Comparing $M^{RL}_t$ vs $M^{SFT}_t$ isolates the effect of the **training objective** (RL vs SFT).
- Comparing $M^{RL}_t$ vs $M^{SFT\rightarrow RL}_t$ evaluates whether RL alone can compete with the **current gold standard**.

We train all RL and SFT runs **until convergence**, with convergence checks included in the appendix.

---

## Main results

In the paper, Figures 2 and 3 report performance on GSM8K and MATH across pretraining checkpoints and training pipelines. Figure 4 analyzes training dynamics (sharpening vs expansion). Figure 5 studies rollout budgets.

> **Image note:** If you want the original figures in this post, export the paper’s figures as images and include them via:
>
> `![Figure X: caption](/assets/img/rl-pretraining/figureX.png)`

---

### RL is effective early in pretraining

Our first key finding is that **RL is effective surprisingly early**.

#### GSM8K: large gains from RL as early as 4B tokens

On GSM8K (training on the GSM8K-like subset of OpenMathInstruct), we find that:

- as early as **\(t = 4B\)** pretraining tokens,
- RL significantly improves GSM8K performance.

For example:
- pass@1 increases from **~2%** (base checkpoint) to **~18%** (after RL).

This improvement occurs **prior to reaching** the Chinchilla-optimal number of tokens.

#### RLVR competes with the gold standard pipeline on GSM8K

After around **\(t = 10B\)** tokens:
- $M^{RL}_t$ **outperforms** $M^{SFT}_t$ on pass@1,
- and performs **on par** with $M^{SFT\rightarrow RL}_t$ on pass@1.

For pass@8 and pass@32:
- $M^{RL}_t$ performs on par with both $M^{SFT}_t$ and $M^{SFT\rightarrow RL}_t$.

A notable point is that $M^{RL}_t$ never observes ground-truth reasoning traces; it develops reasoning capabilities entirely from self-generated on-policy traces and verifiable feedback—yet can match supervised learning on this benchmark.

#### Figure 2 (paper): GSM8K performance

**Caption (paper):** We report accuracy on pass@k for $M_t$, $M^{SFT}_t$, $M^{SFT\rightarrow RL}_t$, and $M^{RL}_t$ evaluated on GSM8K, after training on the GSM8K-like subset of OpenMathInstruct. The base model is evaluated using 8-shot prompts, while all other models are evaluated with 0-shot prompts. For each method on top of $M_t$, we report accuracy of the final converged checkpoint.

**Include in blog (optional):**
```md
![Figure 2: GSM8K performance across pretraining checkpoints and training pipelines.](/assets/img/rl-pretraining/figure2.png)
```

---

### Where early RL struggles: limitations on MATH

On the harder MATH benchmark (training on full OpenMathInstruct), the picture is different.

#### MATH: RL improves but does not catch up to SFT or SFT→RL

On MATH:
- $M^{RL}_t$ consistently improves pass@1, pass@8, and pass@32 by **5% to 10%** over the base checkpoint $M_t$,
- but it never catches up to $M^{SFT}_t$ and $M^{SFT\rightarrow RL}_t$.

This suggests a limitation of relying on the RLVR objective (i.e., on-policy data) from pretraining checkpoints, potentially related to the difficulty of the task.

#### Figure 3 (paper): MATH performance

**Caption (paper):** We report pass@k accuracy for $M_t$, $M^{SFT}_t$, $M^{SFT\rightarrow RL}_t$, and $M^{RL}_t$ evaluated on MATH after training on full OpenMathInstruct. Unlike GSM8K, on this harder benchmark, $M^{RL}_t$ brings substantial gains over $M_t$ but fails to fully match $M^{SFT}_t$ and $M^{SFT\rightarrow RL}_t$, indicating a limitation of using RL early in pretraining.

**Include in blog (optional):**
```md
![Figure 3: MATH performance across pretraining checkpoints and training pipelines.](/assets/img/rl-pretraining/figure3.png)
```

---

## Sharpening vs. expansion: two opposing RL behaviors

We refer to:

- **Sharpening** as the phenomenon where training improves pass@1 but has little to no effect on pass@k for larger \(k\).
- **Expansion** as the phenomenon where RL increases pass@k across many values of \(k\).

Recent work has argued that RLVR often sharpens the output distribution without bringing “new” reasoning capabilities. In our experiments, we see **two opposing effects depending on the training pipeline**.

---

### Standard pipeline: sharpening

When we apply the standard pipeline:
\[
M_t \rightarrow M^{SFT}_t \rightarrow M^{SFT\rightarrow RL}_t,
\]
we reproduce sharpening behavior.

In the paper’s example training dynamics:
- pass@1 continues to improve from $M_t$ to $M^{SFT}_t$ and then during RL,
- but pass@32 gains during SFT are followed by a **slight decrease** during RL.

We hypothesize that sharpening occurs because during SFT the model has already seen ground-truth solutions on the same set of questions, so RL primarily refines existing capabilities rather than discovering new reasoning paths.

---

### Direct RL: expansion

In contrast, when training RL directly from the same pretraining checkpoint:
\[
M_t \rightarrow M^{RL}_t,
\]
we observe expansion behavior.

In the paper’s direct RL training dynamics example:
- both pass@1 and pass@32 improve during RL.

Without prior exposure to ground-truth solutions, the model explores and discovers new reasoning paths through on-policy learning.

#### Figure 4: RL training dynamics

**Caption:** GSM8K accuracy (pass@1 and pass@32) across training stages. Left: Standard SFT→RL pipeline shows sharpening (pass@1 improves, pass@32 decreases during RL). Right: Direct RL training expands distribution (both metrics improve).

<!-- **Include in blog (optional):** -->
```md
![Figure 4: RL training dynamics with and without SFT (sharpening vs expansion).](/assets/img/rl-pretraining/figure4.png)
```

---

### Brittleness on very early checkpoints

Despite the promising results, direct RL exhibits instability on early checkpoints.

Between **\(t = 4B\)** and **\(t = 10B\)** pretraining tokens:
- $M^{RL}_t$ can be highly non-deterministic across training seeds.
- Some seeds yield significant improvements on GSM8K.
- Others fail to improve over $M_t$ at all.

For earlier checkpoints, we therefore ran RL across **4 seeds** and reported the best-performing seed in the GSM8K results.

In the appendix seed analysis:
- training reward curves can look identical across seeds,
- yet validation and test performance diverge substantially,
suggesting that early on-policy RL can sometimes learn superficial patterns rather than robust reasoning.

This brittleness resolves after around **\(t = 10B\)** tokens, when the model has developed enough reasoning foundation to generate higher-quality on-policy data.

---

## Rollout budgets for early RL

When applying RL to early pretraining checkpoints, pass@k on training questions can be low, exacerbating reward sparsity.

A natural strategy to increase the learning signal is to sample more rollouts per prompt. We study the effect of rollout count \(n\) in GRPO.

---

### Why rollouts matter early

Early checkpoints may generate few correct rollouts. Without enough positive samples, the learning signal can be sparse or noisy and training may struggle.

Increasing the number of rollouts per prompt is one knob to counter this.

---

### Experimental design for rollout scaling

We partition the training set into two subsets to simulate different stages of pretraining:

1. Focus on the GSM8K-like subset of OpenMathInstruct (80K examples).
2. For each question:
   - generate 64 responses from the base model at temperature 1,
   - count the number of correct solutions.
3. Define:
   - **GSM8K-Easy**: 16 to 64 correct responses
   - **GSM8K-Hard**: at most 8 correct responses
4. Randomly sample 10K questions from each split.
5. Train GRPO with:
   - **few rollouts**: \(n = 5\)
   - **many rollouts**: \(n = 64\)
6. Evaluate on GSM8K test using pass@k for \(k \in \{1, 8\}\).

Because \(n = 64\) consumes significantly more FLOPs per RL step, we analyze accuracy as a function of:
- total FLOPs, and
- samples seen (examples processed).

We train until validation pass@1 converges.

---

### Results: sample vs compute efficiency trade-off

We observe a distinct trade-off:

- As a function of **samples seen**, \(n = 64\) improves convergence speed compared to \(n = 5\).
- As a function of **FLOPs**, \(n = 5\) is more compute-efficient early in training.

As training progresses toward \(10^6\) FLOPs, the efficiency gap narrows, with \(n = 64\) eventually matching or surpassing \(n = 5\) in some regimes.

Across difficulty splits and metrics, we find three main takeaways:

1. **Asymptotic performance is largely independent of rollout count:**  
   \(n = 5\) and \(n = 64\) typically converge to similar pass@k peaks.

2. **There is a clear sample vs compute efficiency trade-off:**  
   more rollouts extract more signal per example but cost more compute; fewer rollouts can be much more FLOP-efficient.

3. **The compute advantage of fewer rollouts can be pronounced under sparse rewards:**  
   especially on the GSM8K-Hard split for pass@8, suggesting large rollout scaling may yield diminishing returns per FLOP compared to processing more batches with fewer rollouts.

#### Figure 5 (paper): Effect of number of rollouts

**Caption (paper):** We report pass@k on GSM8K test. We sub-sample GSM8K-Easy and GSM8K-Hard training sets based on the proportion of positive rollouts from the base model on each example. GSM8K-Hard simulates an early pretraining setting where the likelihood of sampling a positive rollout is very low. Both \(n = 5\) and \(n = 64\) typically asymptote to similar performance across training sets and pass@k metrics.

**Include in blog (optional):**
```md
![Figure 5: Effect of rollout count (n) on RL performance, by difficulty split.](/assets/img/rl-pretraining/figure5.png)
```

---

## Prior work

### Reinforcement learning for reasoning

Early work on RLHF demonstrated benefits of continuing to train with an RL objective after NTP pretraining. More recent methods such as DPO, PPO, and GRPO have proven effective in improving downstream performance. RL on verifiable reward tasks (math/coding) is common.

A prominent view is that RL tends to sharpen the output distribution.

### Prerequisites for post-training (“readiness”)

Several studies investigate whether models require a sufficient level of capability before they benefit from RL, and examine how the difficulty of RL data affects outcomes. Our results challenge a rigid notion of readiness by showing that RL can be effective earlier than previously thought in a controlled math setting.

### Integrating RL into pretraining

There is increasing interest in using pretraining data (more abundant and general) for RL-style training. Various approaches apply RL objectives on pretraining data in different ways, often relying on a pretrained model initially trained with next-token prediction. Our work focuses on the more fundamental question of **when** and **how** an RL objective can be effective relative to training stage.

---

## Discussion and future directions

This work is a proof-of-concept study of early-stage RL in a controlled environment (math).

We find:

- RL is effective **starting very early** during standard pretraining, improving reasoning metrics well before the end of pretraining.
- On GSM8K, RL-only training can approach (and sometimes match) the standard SFT→RL pipeline.
- RL can either **sharpen** or **expand** the model distribution depending on training pipeline:
  - SFT→RL can sharpen (pass@1 up, pass@k down for larger \(k\)),
  - direct RL can expand (pass@1 and pass@k both improve).
- Early RL can be brittle and seed-dependent, but this instability resolves after the model has developed a stronger reasoning foundation.

We see several future directions:

### Incorporating RL into pretraining

If RL can work early, can we incorporate RL training objectives during pretraining itself? There remain open questions about:
- what the best RL-flavored pretraining objective should be,
- how to interleave NTP and RL objectives,
- and how scaling behavior changes.

### Pretraining data mixtures

Our work suggests that if pretraining includes substantial math content, RL on math becomes effective early. This raises questions about whether the optimal pretraining mixture changes when the training objective changes.

### Rollouts and curriculum

Increasing rollouts can help in sparse-reward regimes, but our experiments show diminishing returns per FLOP in some settings. This suggests exploring:
- adaptive rollout budgets,
- curricula over problem difficulty,
- and improved reward design (while keeping rewards verifiable).

---

## Appendix: additional training details and ablations

### A.1 RL training convergence

For all $M^{RL}_t$ runs across checkpoints, we track:
- RL training reward,
- validation reward (on a manually split subset of OpenMathInstruct),
- and GSM8K reward / pass@1.

These curves converge by the end of training. For earlier checkpoints that exhibit seed brittleness, we report a favorable seed in the convergence plots.

**Include in blog (optional):**
```md
![Figure 6: RL training convergence across pretraining checkpoints.](/assets/img/rl-pretraining/figure6.png)
```

---

### A.2 Seed dependence

We compare a favorable seed and an unfavorable seed for $M^{RL}_t$ at \(t = 4B\) tokens:

- Training reward curves are nearly identical across seeds.
- Validation reward diverges.
- GSM8K evaluation diverges:
  - favorable seed improves pass@1 and pass@32,
  - unfavorable seed shows minimal improvement in pass@1 and decreased pass@32.

This reveals a disconnect between training reward and actual reasoning capability early in training.

**Include in blog (optional):**
```md
![Figure 7: RL seed brittleness on early pretraining checkpoints.](/assets/img/rl-pretraining/figure7.png)
```

---

### A.3 SFT convergence

We compare different numbers of SFT epochs to train $M^{SFT}_t$. Performance converges after **5 epochs**, which we use as the standard protocol.

**Include in blog (optional):**
```md
![Figure 8: SFT training convergence (5 epochs vs 10 epochs).](/assets/img/rl-pretraining/figure8.png)
```

---

### A.4 Evaluating base checkpoints with n-shot prompting

We evaluate base checkpoints $M_t$ with different numbers of in-context examples (n-shot).

On GSM8K:
- 0-shot, 1-shot, and 8-shot prompting are compared.

On MATH:
- 1-shot and 8-shot prompting are compared.

Performance peaks at **8-shot prompting** for both benchmarks.

**Include in blog (optional):**
```md
![Figure 9: Impact of n-shot prompting on pretraining checkpoint evaluation.](/assets/img/rl-pretraining/figure9.png)
```

---

## References

- Biderman, S., et al. **Pythia: A suite for analyzing large language models across training and scaling.** ICML, 2023.
- Chen, F., et al. **The coverage principle: How pre-training enables post-training.** arXiv:2510.15020, 2025.
- Chen, M., et al. **Evaluating large language models trained on code.** arXiv:2107.03374, 2021.
- Cheng, Z., et al. **Isocompute playbook: Optimally scaling sampling compute for RL training of LLMs.** 2026.
- Cobbe, K., et al. **Training verifiers to solve math word problems.** arXiv:2110.14168, 2021.
- Dai, J., et al. **Safe RLHF: Safe reinforcement learning from human feedback.** arXiv:2310.12773, 2023.
- Dong, Q., et al. **Reinforcement pre-training.** arXiv:2506.08007, 2025.
- Foster, D. J., et al. **Is a good foundation necessary for efficient reinforcement learning?** arXiv:2503.07453, 2025.
- Guo, D., et al. **DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.** arXiv:2501.12948, 2025.
- Hatamizadeh, A., et al. **RLP: Reinforcement as a pretraining objective.** arXiv:2510.01265, 2025.
- Hendrycks, D., et al. **Measuring mathematical problem solving with the MATH dataset.** NeurIPS, 2021.
- Hoffmann, J., et al. **Training compute-optimal large language models.** arXiv:2203.15556, 2022.
- Li, J., et al. **DataComp-LM: In search of the next generation of training sets for language models.** NeurIPS 2024.
- Li, S., et al. **Reinforcement learning on pre-training data.** arXiv:2509.19249, 2025.
- Loshchilov, I., & Hutter, F. **Decoupled weight decay regularization.** 2019.
- Team OLMo, et al. **2 OLMo 2 Furious.** arXiv:2501.00656, 2024.
- Ouyang, L., et al. **Training language models to follow instructions with human feedback.** NeurIPS 2022.
- Rafailov, R., et al. **Direct preference optimization.** NeurIPS 2023.
- Shao, Z., et al. **DeepSeekMath: Pushing the limits of mathematical reasoning in open language models.** arXiv:2402.03300, 2024.
- Sheng, G., et al. **HybridFlow: A flexible and efficient RLHF framework.** arXiv:2409.19256, 2024.
- Toshniwal, S., et al. **OpenMathInstruct-1: A 1.8 million math instruction tuning dataset.** NeurIPS 2024.
- Wei, J., et al. **Finetuned language models are zero-shot learners.** ICLR.
- Wu, F., et al. **The invisible leash: Why RLVR may or may not escape its origin.** arXiv:2507.14843, 2025.
- Xing, X., et al. **PretrainZero: Reinforcement active pretraining.** arXiv:2512.03442, 2025.
- Yue, Y., et al. **Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model?** arXiv:2504.13837, 2025.
- Zhang, C., et al. **On the interplay of pre-training, mid-training, and RL on reasoning language models.** arXiv:2512.07783, 2025.
- Zheng, R., et al. **Secrets of RLHF in large language models part I: PPO.** arXiv:2307.04964, 2023.
- Zhou, C., et al. **LIMA: Less is more for alignment.** NeurIPS 2023.
