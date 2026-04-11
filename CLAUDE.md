# CLAUDE.md

Project context for Claude Code working in this repo.

## Project

**Goal:** Develop a **label-free MLLM framework for hateful video detection**.

The north star is a method that (a) does not rely on human-annotated hate labels for training/adaptation, and (b) leverages multimodal LLMs in a way that is *principled*, not bolted together. Any proposed technique must earn its place by telling a coherent story about *why* it fits the hateful-video problem — not just by nudging a number on a benchmark.

---

## Execution constraints

- **All code must be submitted via Slurm.** Never run Python scripts directly on the login node. Use `sbatch --gres=gpu:1 --wrap "python ..."` for GPU jobs and `sbatch --wrap "python ..."` for CPU-only jobs.
- **No job chaining.** Submit one job at a time. Wait for it to complete before submitting the next. Do not use Slurm dependency chains (`--dependency`), background submission with `&`, or any form of parallel job submission.
- **1 GPU max at any given time.** This is not per-job — it means the total number of GPUs occupied across all running jobs must never exceed 1.
- **Conda env:** `SafetyContradiction` (vllm 0.11.0). Always activate before running: `conda activate SafetyContradiction && python ...`
- **Model:** Qwen3-VL-8B-Instruct (bf16, no quantization).
- **Label mapping:** For MHClip_EN and MHClip_ZH, `Offensive` maps to `1` (hateful), not `0`. The 3-class annotation collapses to binary as `Hateful+Offensive` vs `Normal`.

---

## Anti-patterns (do NOT do these)

These are the failure modes that have repeatedly derailed this line of work. Treat them as hard constraints, not soft preferences. If you find yourself reaching for one of these, stop and reconsider the design.

### Anti-pattern 1 — Ensembling or repeatedly querying the MLLM

**Do not** propose methods whose gain comes from:

- Querying the same MLLM multiple times and aggregating (majority vote, self-consistency, temperature sampling, N-best reranking).
- Ensembling multiple MLLMs (e.g., Qwen-VL + LLaVA + InternVL) and fusing their outputs.
- Multi-prompt ensembling where the "method" is really "try K prompts and pool".
- Chain-of-thought sampling that generates many rationales and picks/votes among them.
- Any "query more → score more" pattern dressed up as a new component.

**Why this is forbidden:**
1. **Not a contribution.** Paying more inference compute for a better number is not a scientific claim; it is a budget statement. Reviewers (correctly) read ensembles as *"the authors had no idea, so they averaged"*.
2. **Confounds the story.** If Method-X only works at K=8 samples but not K=1, the load-bearing component is the sampling, not Method-X. The paper then has nothing to say about hateful video specifically.
3. **Deployment-hostile.** Hate detection is a moderation setting. An approach that needs 8× MLLM calls per video will not be used. A label-free framework must be *single-pass* or have a clearly justified, bounded number of calls with a structural reason (e.g., one call per modality, not one call per sample).
4. **Hides the real win.** Ensembling tends to mask whether the underlying signal is actually the hateful-content cue we care about, or just variance reduction on an unrelated axis.

**What to do instead:** Single forward pass by default. If you truly need more than one call, each call must correspond to a *distinct, named role* in the method (e.g., "observe" vs "judge") — not K i.i.d. samples of the same role. Every extra call must be defended in the ablation: remove it, and the story must still make sense even if the number drops.

### Anti-pattern 2 — Engineering tricks without a scientific story

**Do not** adopt a technique just because it worked in a different paper or because it bumped the metric in a pilot. Every design choice must answer:

> **"What is the story, specific to hateful video detection, that makes this technique the right tool?"**

Red flags that indicate you are in engineering-trick territory:

- "We added [component] and F1 went up 1.2" — but you cannot say *why* hateful video, in particular, benefits from it.
- Borrowed tricks from unrelated domains (temperature schedules, LoRA ranks, prompt templates from code LLMs, etc.) with no argument for why they transfer to moderation of multimodal hate.
- Hyperparameter-sensitive gains (works at lr=3e-5 but not 1e-5) dressed up as methodological contributions.
- Post-hoc rationalization: choosing the technique first, then writing the motivation paragraph backwards from the result.
- "Label-free" framing bolted on top of a method that, in practice, secretly depends on a labeled dev set for tuning.
- Pipelines whose blocks are individually defensible but whose *composition* has no narrative — a grab-bag of good ideas rather than a method.

**The bar for every technique in this project:**

1. **Phenomenon.** Name a concrete property of hateful video (e.g., "hate is often carried by the *juxtaposition* of benign visuals and targeted audio", "irony inverts surface sentiment", "targets are identified by group cues the MLLM already knows from pretraining", "the offending signal is sparse and localized in time") that motivates the technique.
2. **Mechanism.** Explain *mechanistically* how the proposed technique exploits that property. A mechanism is falsifiable: "if property P did not hold, technique T would not help".
3. **Prediction.** Before running the experiment, state what you expect to see and — crucially — what result would **disconfirm** the story. If every possible result is consistent with the story, the story is unfalsifiable and therefore not a contribution.
4. **Counterfactual ablation.** The ablation must target the *story*, not just the component. Removing the technique should break the specific hateful-video phenomenon it claims to address, not just lower a generic metric.

If a technique passes 1–4, it has a story. If it only has "it works on the val set", it is an engineering trick and does not belong in this project.

### Anti-pattern 3 — Using external datasets to acquire extra information

**Do not** propose methods whose gain comes from:

- Pulling in additional labeled or unlabeled hate/meme/toxicity datasets beyond the target benchmark's own training split (e.g., training on FHM + HarMeme then evaluating on HateMM).
- Pre-training or fine-tuning on auxiliary datasets that supply domain knowledge unavailable in the target data (e.g., using Hateful Memes as a warm-up for video hate detection).
- Retrieval-augmented approaches that search an external hate-example database at inference time.
- Cross-dataset pooling, domain adaptation bridges, or transfer sets that smuggle in out-of-distribution supervision.
- Using external knowledge bases, ontologies, or hate-term lexicons as supplementary signal (e.g., Hatebase, GDELT).

**Why this is forbidden:**
1. **Confounds the label-free claim.** If the method needs HarMeme labels to work on HateMM, it is not label-free — it is *differently-labeled*. The contribution becomes "we found a good transfer set," not a method.
2. **Not reproducible or fair.** Every extra dataset is a degree of freedom. Reviewers will ask "what if you picked a different auxiliary set?" and there is no principled answer.
3. **Masks the method's real capability.** If the gain vanishes when the auxiliary data is removed, the load-bearing component is the data, not the framework. A method that only works with dataset X is a method about dataset X, not about hateful video in general.
4. **Already tried and failed.** Cross-dataset pooling (Iteration 4 in IDEA_REPORT) showed domain gap and label distribution mismatch. This is not a theoretical concern — it is an empirical dead end in this project.

**What is allowed:** Using the target benchmark's own *unlabeled* training split is fine — that is the point of label-free. Using general-purpose pre-trained models (CLIP, Qwen-VL, etc.) is fine — they carry general world knowledge, not task-specific hate supervision. The line is: **no additional data source that is chosen because it is relevant to hate detection.**

---

## How to use this document

- Before writing code for a new component, write the 4-point story (phenomenon → mechanism → prediction → counterfactual) in the PR description or experiment plan. If you cannot write it, do not implement the component yet.
- Before running an experiment, check: does success require K>1 MLLM calls per sample? If yes, justify the *role* of each call, not the *count*.
- When reviewing results, ask: "If I deleted this component, which part of the hateful-video story collapses?" If the answer is "none, just the number", cut the component.
- When tempted by a trick from another paper, ask: "What is the hateful-video-specific reason this transfers?" No reason → no trick.

The guiding question for every decision in this repo is:

> **Is there a story to be told, in the hateful-video application context, behind the technique we just adopted?**

If the honest answer is no, change the technique.
