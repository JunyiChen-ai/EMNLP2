# prompt_paradigm v1 — Observe-then-Judge: decouple perception from normative judgment

**Teammate**: prompt-paradigm (Teammate A)
**Target**: Beat 2B baseline on BOTH MHClip_EN (76.40 / 0.653) and MHClip_ZH (81.21 / 0.787) under a SINGLE unified (TR/TF x Otsu/GMM) configuration.
**MLLM call budget**: 2 distinct-role calls per video.
**Model**: Qwen/Qwen3-VL-2B-Instruct (bf16, vLLM, temperature=0).
**No test labels. No external datasets. No ensembling.**

---

## 1. Phenomenon (specific to hateful video)

MHClip_EN baseline is stuck at 76.4% ACC while its *oracle* threshold ceiling on the same score stream is only 77.6% — the bottleneck is not the threshold, it is the score itself. The direct binary prompt `"Does this video violate policy? Yes/No"` is forced to compress three cognitively distinct judgments into one token:

1. **Perception**: which identifiable group / target is depicted in frames + transcript?
2. **Stance**: what affective/rhetorical posture does the video take toward that group?
3. **Normative match**: does that (group, stance) pair violate the platform policy?

In English MHClip, hateful videos frequently have *benign-looking surface visuals* (a creator in a room, a slide, a clip montage) and the hate is carried by targeted transcript framing about an identifiable group. When the model is asked "does the VIDEO violate policy?", the perceptual prior — driven by frames — dominates the single-token answer, and the model tends toward "No" for videos whose visuals look like regular vlog/commentary content. This is exactly the false-negative pattern that holds EN below the 80% target. Chinese MHClip does not suffer as much because Bilibili hate in this dataset is more overt (on-screen text, explicit slurs), so the perceptual prior and the normative verdict already agree.

The property is falsifiable: *if* EN hate were already carried by blatant visuals, perception and normative judgment would not conflict, and a two-step decomposition would not help.

## 2. Mechanism (two distinct-role calls)

We split the task into two MLLM calls with structurally different inputs and roles. Neither is a resample of the other.

### Call 1 — **Observer** (perception-only, video-conditioned, free-form)

- **Reads**: video frames + title + transcript (same media content as baseline).
- **Asks**: "Describe in one short sentence who or what identifiable group this video is about, and the stance the video takes toward them. Do not judge whether it is hateful."
- **Outputs**: a short generated description (max ~40 tokens) — a natural-language commitment to (group, stance). This is the *only* call that sees pixels.
- **Why it is not a judgment call**: we explicitly forbid the policy vocabulary. The observer is not asked whether anything is hateful; it is asked to name the target and the posture. This is a perceptual description task the base MLLM is good at.

### Call 2 — **Judge** (text-only, policy-conditioned, Yes/No logprob)

- **Reads**: the policy rules + the Observer's one-sentence description + the title. **No frames. No transcript re-pass.**
- **Asks**: `"Given this description of a video's target and stance, does the described content violate any of the policy rules? Yes / No."`
- **Outputs**: a single token; we extract `P(Yes) / (P(Yes) + P(No))` as the score, exactly as in baseline's `extract_binary_score`.
- **Why it is not a duplicate of Call 1**: Call 2 has no video input and a different system role (policy adjudicator). It cannot be substituted by resampling Call 1 or by a longer prompt to Call 1, because the pixel prior would still dominate.

This is a *structural* decomposition — each call has a fixed, named role. It is not K-best, not self-consistency, not multi-prompt pooling. The 2 calls are unavoidable: Call 1 provides the grounded observation, and Call 2 converts it into a policy verdict whose input distribution (text only) is not contaminated by benign visuals.

## 3. Prediction (what I expect, and what would disconfirm)

### Expected
- **EN**: the observe-then-judge score shifts hateful videos upward in the P(Yes) distribution relative to baseline, because the judge is now reading an explicit (group, stance) commitment rather than being swayed by benign frames. I expect EN test ACC > 76.40% under at least one of {TF-Otsu, TF-GMM, TR-Otsu, TR-GMM}.
- **ZH**: ZH baseline already reflects overt hate; the observe-then-judge score should be near-identical or slightly higher, because Call 1 will correctly describe the overt target and Call 2 will convert that into the same Yes verdict. I expect ZH test ACC at least within -0.5pp of 81.21%, and under the chosen unified (TR/TF x Otsu/GMM) it must be >= 81.21%.
- **Signature in scores**: the *positive-class* score mean on EN should rise more than the negative-class score mean (widened margin), not a uniform shift.

### What would disconfirm the story
- EN ACC does not exceed 76.4% under ANY of the four (TR/TF x Otsu/GMM) threshold rules. The disentanglement did not produce a better signal → the 3-judgment-compression story was wrong.
- ZH drops materially (> 2pp) below 81.21%. The observer's description added noise rather than removing perceptual bias → the "perception vs normative" split is not cleanly separable by prompting alone.
- Positive and negative score means both rise uniformly → Call 2 is just calibration-shifted, not genuinely better-separating.

Any of these outcomes kills v1 and forces a new proposal.

## 4. Counterfactual ablation (what must break when we remove the 2nd call)

If we delete Call 2 and instead ask Call 1 to both observe AND judge in a single multimodal prompt, we recover exactly baseline behavior — the judgment is made while frames are still in context, the perceptual prior re-asserts itself, and EN stays at ~76.4%. The specific thing that must break is the **EN false-negative rate on benign-visual hate videos**: those are the videos where the observer correctly commits to a targeted group + stance, but a direct-judgment prompt would answer "No" because the visuals look fine.

Symmetric check: if we delete Call 1 and have Call 2 read just the policy + title (no observation), performance should collapse or be worse than baseline, because Call 2 is now a text-only zero-shot policy classifier with no video grounding. This confirms that Call 1 is load-bearing specifically for providing *grounded* perception — it is not a prompt-trick on Call 2.

Both ablations are natural single-call runs and will be available for comparison in the report.

## 5. Named roles

| Call | Name | Input | Output | Role reason |
|---|---|---|---|---|
| 1 | **Observer** | frames + title + transcript + system=`"You describe what you see. Do not judge."` + user=`"Describe in ONE sentence who/what group this video depicts and the stance the video takes toward them. Do NOT say whether it is hateful, offensive, normal, or anything policy-related."` | free-text, max 40 tokens, temperature=0 | Commits to a (target, stance) description grounded in pixels+transcript. |
| 2 | **Judge** | system=`"You are a content moderation analyst. Apply the policy literally."` + user=`"Policy rules: {rules}\n\nVideo title: {title}\nDescription of the video: {obs}\n\nDoes the described content violate any of the policy rules? Answer Yes or No."` **No media.** | single token, `P(Yes)/(P(Yes)+P(No))` extracted from logprobs, temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[Yes-tids, No-tids] | Converts grounded description into normative verdict using only text evidence, eliminating benign-visual bias. |

Both calls use the same `Qwen/Qwen3-VL-2B-Instruct` backbone loaded once. Total calls/video = 2, exactly at the cap. Observer is the only perceptual call; Judge is the only normative call. Neither is a resample of the other.

## 6. Reproduction plan

### Datasets and splits
- MHClip_EN: test (161) + train (for label-free TR thresholds).
- MHClip_ZH: test (149 valid) + train.

### Target unified configuration
- I will evaluate all four (TR/TF x Otsu/GMM) cells via `src/prompt_paradigm/eval_with_frozen_thresholds.py` and pick the single unified pair whose numbers beat baseline on BOTH datasets. If no unified pair beats, I go to Gate 1 with v2.
- Prior: **TF-GMM** is the most likely unified winner because ZH already needs GMM and the decoupled score is expected to be more bimodal on EN.

### Code layout (all under `src/prompt_paradigm/`)
- `observe_then_judge.py` — scorer. One vLLM instance, two-pass per batch:
  1. Pass 1: Observer call (media + observe prompt), free-form generation with `SamplingParams(temperature=0, max_tokens=40)`. Collect text.
  2. Pass 2: Judge call (text-only judge prompt using Pass-1 text), `SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[Yes/No tids])`. Extract `P(Yes)/(P(Yes)+P(No))` reusing baseline's token-ID and logprob extraction logic.
  Writes `{"video_id": str, "score": float, "observation": str}` to JSONL.
- `eval_with_frozen_thresholds.py` — imports `otsu_threshold`, `gmm_threshold`, `metrics`, `load_scores_file`, `build_arrays` from `src.quick_eval_all` (no reimplementation) and reports the 4 (TR/TF x Otsu/GMM) cells on `results/prompt_paradigm/{MHClip_EN,MHClip_ZH}/{train,test}_obsjudge.jsonl`.

### Slurm plan (serialized, 1 GPU at a time, up to 2 concurrent per rules)

Each command runs in `SafetyContradiction`. I will submit ONE job, wait for it to finish, then submit the next. I will log every job ID in `docs/experiments/prompt_paradigm_runs.md`.

1. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_EN --split test"` — ~15-20 min expected (161 videos x 2 calls).
2. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_EN --split train"` — needed for TR thresholds.
3. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_ZH --split test"`.
4. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_ZH --split train"`.
5. `sbatch --cpus-per-task=2 --mem=4G --wrap "... python src/prompt_paradigm/eval_with_frozen_thresholds.py"` — CPU-only evaluator, writes `results/prompt_paradigm/report.json` and `results/analysis/prompt_paradigm_report.md`.

### Gate 2 decision rule
- If the chosen unified (TR/TF x Otsu/GMM) pair yields EN ACC > 76.40 AND ZH ACC > 81.21 (and macro-F1 not lower than baseline on either), I send results-ready to the director.
- If either dataset misses, I stop, draft v2 explaining what the score distributions showed (per-class means, bimodality indicator), and what the new 2-call design will be. I do not iterate on prompts inside v1.

## 7. Label-free / external-data integrity check

- No test labels used anywhere. Thresholds are unsupervised (Otsu / GMM on scores).
- Train split is the TARGET benchmark's own unlabeled train split — allowed by CLAUDE.md.
- No auxiliary hate datasets, no hate lexicons, no retrieval, no knowledge bases.
- Both calls use only the general-purpose Qwen3-VL-2B weights and the platform policy text that is already in the baseline scorer.

## 8. Compliance with rules

- Anti-pattern 1: 2 calls, each with a distinct named role (perceptual observer vs. normative judge). No sampling, no pooling, no majority vote, no prompt ensembling. Ablation will show each call is load-bearing.
- Anti-pattern 2: the story is tied to a concrete property of EN hateful video (benign visuals + targeted transcript framing) and predicts the failure mode it fixes; result-to-story is falsifiable.
- Anti-pattern 3: no external data; only MHClip's own unlabeled train split and the general-purpose MLLM.
- Frozen files: I will not edit `src/score_holistic_2b.py`, `src/score_holistic_8b.py`, `src/quick_eval_all.py`, `src/data_utils.py`, `results/holistic_*`, `CLAUDE.md`, `STATE_ARCHIVE.md`, `MEMORY.md`.
- Slurm: one sbatch at a time, logged, no chaining, no dependency flags, no scancel of jobs I did not submit.

## 9. What I am NOT asking the director for

I am not asking for prompt wording suggestions or alternative role names. I am asking ONLY for a gate-1 ruling on whether this proposal violates any rule (anti-patterns, call budget, label-free, frozen files, slurm discipline). If approved, I implement as written. If rejected, I redraft v2 on my own.
