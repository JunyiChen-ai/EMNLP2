# prompt_paradigm v2 — Factored Verdict: Target-Presence × Stance-Valence

**Teammate**: prompt-paradigm (Teammate A)
**Target**: strict-beat 2B baseline on BOTH MHClip_EN (ACC 0.7640 / mF1 0.653) and MHClip_ZH (ACC 0.8121 / mF1 0.787) under ONE unified (TR/TF × Otsu/GMM) configuration, mF1 non-regression on each.
**MLLM call budget**: 2 distinct-role calls per video, both reading RAW media (no text compression between calls).
**Model**: Qwen/Qwen3-VL-2B-Instruct, bf16, vLLM, temperature=0.
**Label-free, no external datasets, no ensembling.**

---

## 0. What v1 actually showed (and why v2 is not a reshuffle of v1)

v1 failed Gate 2. The director's verification and my own re-analysis of the v1 score stream produced the following *empirical* findings (pulled from `results/prompt_paradigm/report.json` plus a per-video join against baseline `results/holistic_2b/*/test_binary.jsonl`):

| Signal | Baseline (2B binary_nodef) | v1 Observe-then-Judge | Δ |
|---|---|---|---|
| EN oracle ACC | 0.7764 | 0.7391 | −0.037 |
| ZH oracle ACC | 0.8121 | 0.7785 | −0.034 |
| EN pairwise AUC-like (P(s_pos > s_neg)) | 0.7250 | 0.7170 | −0.008 |
| ZH pairwise AUC-like | 0.8479 | 0.8340 | −0.014 |
| EN pos_mean / neg_mean | 0.222 / 0.075 | 0.549 / 0.238 | margin 0.146 → 0.311 |
| ZH pos_mean / neg_mean | 0.191 / 0.030 | 0.892 / 0.429 | margin 0.161 → 0.463 |

The **rank-ordering is essentially preserved** (AUC-like barely moved on either dataset), yet the oracle ACC dropped by ~3-4 pp on both. This is a striking pattern: v1 did not destroy the discriminative signal, it destroyed the score distribution's *calibration*, pushing BOTH the positive and negative masses upward into a narrow high-probability region where Otsu/GMM pick a worse cut than on the baseline distribution.

**Root cause of v1's failure** (my reading of the failed mechanism, required by the director's v2 constraint):

v1's Observer-Judge decomposition forced a *lossy text compression* step: the Observer produced a free-form 1-sentence description, and the Judge then adjudicated policy from that description *without ever seeing pixels or transcript again*. Two things went wrong:

1. **Compression loss of distinguishing detail.** Observer descriptions repeatedly used abstract stance labels like "mocking", "critical", "satirical", "humorous" which occur in *both* benign satire AND identity-targeted derogation. The Judge then cannot tell these apart from text alone. On benign satire the Judge returns ~0.3 P(Yes) (uncertain under policy); on target-derogation it returns ~0.6. The correct answers are 0.02 and 0.9. Margin collapses, distributions overlap.
2. **Calibration drift on the text-only Judge.** A text-only policy-reader over a natural-language description of *any* video has a non-trivial base-rate of "Yes" (~0.24 on EN, ~0.43 on ZH) because the Judge cannot ground its "no" on visual benignity — so it hedges toward "maybe". This is unrelated to the perception question and is the direct cause of the negative-class mass drift.

**Load-bearing lesson for v2**: any decomposition whose second call operates on a *natural-language summary* of the first call's output inherits both failure modes — lossy compression and calibration drift from lack of grounding. The second call MUST read the raw media, not a summary.

v2 abandons the "process → summarize → judge from summary" structure entirely. v2 factors the *question*, not the *input*.

---

## 1. Phenomenon (specific to hateful video)

A hateful video is the intersection of two orthogonal properties:

- **(T) Target**: the video identifies (visually or verbally) a protected-identity group — race, religion, ethnicity, gender, sexual orientation, disability, nationality, or a group defined by any of these.
- **(S) Negative stance**: the affective/rhetorical stance the video takes toward that identified group is derogatory, mocking, dehumanizing, disparaging, hostile, or inciting — as opposed to neutral, descriptive, celebratory, or self-referential.

Concretely, hateful := T ∧ S. Benign content violates AT LEAST ONE of these: either no protected-identity target is involved (e.g., a rooster in a farmyard; a product review), or a target is named but the stance is non-derogatory (e.g., a documentary about a minority, a congratulatory video, a factual news report).

This is not a philosophical decomposition — it is observable in the MHClip data. Re-scanning the v1 Observer outputs plus the baseline label distribution:

- Many EN false-negatives of baseline are videos where (T) is clearly satisfied (transcript names a group) but the visual frames look benign, and the direct single-call prompt under-weights the transcript-only target signal.
- Many EN false-positives of baseline are videos where (T) is NOT satisfied (no protected-identity target at all — just generic meanness, political commentary about institutions, interpersonal conflict) but the visual/textual intensity triggers the single-call "is this hateful" prompt.
- The symmetric failure mode also appears: videos where (T) is satisfied with neutral (S) — factual discussions of a minority group — can trip a direct prompt because the presence of the target biases the model's "is hateful" token.

**Falsifiability of the phenomenon**: if T and S were *not* separable (i.e., if the model's P(T) and P(S) collapse into the same judgment and correlate perfectly with the direct P(Yes|hateful) call), then factoring T × S would not help. A test for this is: if for baseline-correct videos both sub-calls agree with the baseline, and for baseline-wrong videos at least one sub-call disagrees with the baseline and points to the correct answer, then T and S carry independent information.

## 2. Mechanism (two distinct-role calls, both video-grounded)

Two MLLM calls per video. Both calls receive the **same raw media** (frames + title + transcript). They differ in the question asked and the role played.

### Call 1 — **Target-Detector** (identifies protected-group presence in the video)

- **Role**: perception + grounding. "Is there a protected-identity group that this video is about?"
- **Inputs**: video frames + title + transcript (same as baseline).
- **System prompt**: `"You analyze video content. You identify whether a video concerns a protected-identity group."`
- **User prompt**: `"Video title: {title}\nTranscript: {transcript}\n\nDoes this video identify, depict, mention, or focus on a protected-identity group? Protected-identity groups include: race, ethnicity, nationality, religion, gender, sexual orientation, gender identity, disability, or any group defined by these attributes.\n\nIf yes, the video is about (or targets, mentions, or depicts) such a group. If no, the video is about something else (e.g., products, pets, entertainment without identity targeting, generic opinions unrelated to protected groups).\n\nAnswer \"Yes\" or \"No\"."`
- **Output**: single token. Score `P_T = P(Yes)/(P(Yes)+P(No))` extracted from position-0 logprobs with `allowed_token_ids=[Yes, No]`, `max_tokens=1`, `logprobs=20`. This is the **target-presence probability**.
- **Why this is a distinct role**: Target-Detector is a *perception / existence* question. It does not invoke policy rules. It does not ask "is this hateful". Its training signal in pretraining is zero-shot VQA on "does this image/video depict/mention X", which Qwen-VL is strong at.

### Call 2 — **Stance-Judge** (measures derogatory valence given the video)

- **Role**: affective valence toward whoever/whatever the video is about. "Is the stance the video takes toward its subject derogatory, mocking, dehumanizing, or hostile?"
- **Inputs**: video frames + title + transcript (same as baseline). **The raw media again, NOT the output of Call 1.**
- **System prompt**: `"You analyze video content. You identify the affective stance a video takes toward its subject, on a derogatory / non-derogatory axis."`
- **User prompt**: `"Video title: {title}\nTranscript: {transcript}\n\nDoes this video take a derogatory, mocking, dehumanizing, disparaging, or hostile stance toward any person or group it depicts or discusses?\n\nConsider only the stance/attitude of the video, not whether its subject is a protected group. A video about a celebrity spoken of admiringly is non-derogatory; a video about an animal shown with affection is non-derogatory; a video using slurs, dehumanization, hostile humor, or contempt toward ANY person or group is derogatory.\n\nAnswer \"Yes\" or \"No\"."`
- **Output**: single token. Score `P_S = P(Yes)/(P(Yes)+P(No))`.
- **Why this is a distinct role**: Stance-Judge is an *affective/sentiment* question. It explicitly abstracts away from "is the target a protected group", so Call 2's answer does not need to know which group (Call 1 handles that orthogonally). Its training signal is zero-shot sentiment/valence reading, which the model is also strong at.

### Final score

```
score(v) = P_T(v) × P_S(v)
```

This is the joint probability under an independence factorization: `P(hateful) ≈ P(has protected-target) × P(derogatory-stance toward it)`. Per-video, it is a deterministic function of two single-token logprob extractions. No sampling, no pooling, no resampling, no ensemble. If either factor is near 0 the score is near 0; only when BOTH are high does the score approach 1. Otsu/GMM on this score stream will see a distribution pushed toward the {0,1} corners because multiplying two sub-probabilities amplifies the margin.

### Why this is not a duplicate of Call 1 or a reword of baseline

- **Not baseline**: baseline's single call asks "does this violate policy rules" with a 9-item rule list in-context. The model must simultaneously weigh (identity present?), (stance negative?), (does that combination match rule k?). In v2 the rules are *dropped* from the user calls entirely — the two factors are general-purpose perception and sentiment, and the policy specification (hateful = T ∧ S) is encoded in the *score formula*, not the prompt. This is structurally different: the prompt asks atomic questions, the score encodes the compositional rule.
- **Not Call 1 twice**: Target-Detector ignores stance, Stance-Judge ignores whether the target is protected. They commit to orthogonal axes and are each solvable with a different pretrained skill (entity detection vs. sentiment valence). Swapping the prompts would not change the score.
- **Not v1**: both calls read the raw video. No natural-language summary is passed between them. The v1 failure mode (text-only adjudication with calibration drift) cannot occur because Stance-Judge is video-grounded, and Target-Detector does not need adjudication at all.

## 3. Prediction (what I expect and what would disconfirm)

### Pre-pilot prediction (required by director)

- **EN**: the Factored Verdict score on EN test beats baseline's ACC 0.7640 with mF1 ≥ 0.653 under at least one of the four (TR/TF × Otsu/GMM) unified cells. Specifically, I expect EN ACC ∈ [0.78, 0.82] and mF1 ∈ [0.70, 0.76] under TF-Otsu or TF-GMM.
- **ZH**: Factored Verdict beats baseline's ACC 0.8121 with mF1 ≥ 0.787. Specifically, I expect ZH ACC ∈ [0.82, 0.87] and mF1 ∈ [0.79, 0.84] under the same unified cell.
- **Oracle ceiling**: the oracle ceiling of the product score must EXCEED baseline oracle on both datasets (EN > 0.7764, ZH > 0.8121). If the oracle ceiling is below baseline, v2 is dead before any threshold rule is applied — I will stop and go to v3 without claiming a Gate 2 pass.
- **Distribution signature**: the product-score histogram on each dataset is more bimodal than baseline — specifically, the Bhattacharyya overlap between pos and neg class score distributions is strictly lower than baseline's, and the pos_mean/neg_mean margin (normalized by pooled std) is strictly larger.

### What would disconfirm the v2 mechanism (and trigger v3)

Any of the following kills v2 and forces a new proposal:

1. **Oracle drop**: EN or ZH oracle ACC under the product score is below baseline oracle on the same dataset. The rank information was not preserved by the factorization → the independence assumption failed and the sub-calls were not really orthogonal.
2. **Strict-miss on either dataset**: after evaluating all four (TR/TF × Otsu/GMM) cells, no single unified cell beats baseline on BOTH datasets with mF1 non-regression.
3. **Ablation inconsistency**: single-factor scores (P_T alone, P_S alone) each beat baseline. This would mean the factorization is not load-bearing — one of the factors is already a better single-call score and the other is noise. v2's claim is that BOTH are needed.

## 4. Counterfactual ablation (the story-level ablations, not tuning knobs)

**Ablation A — "P_T alone"**: use only Target-Detector, no Stance-Judge. Prediction: this will flag all videos about protected-identity groups as hateful, producing many false positives on neutral/factual/positive videos that discuss minority groups. The ZH dataset has multiple such neutral-discussion false positives; P_T alone should under-perform the product on ZH.

**Ablation B — "P_S alone"**: use only Stance-Judge, no Target-Detector. Prediction: this will flag all videos with hostile/mocking tone as hateful, producing many false positives on interpersonal conflict, political institutional critique, ordinary meanness, and adversarial humor. The EN dataset contains many such non-protected-target hostile videos; P_S alone should under-perform the product on EN.

**Ablation C — "P_T + P_S (sum)" vs "P_T × P_S (product)"**: if sum beats product, the mechanism is just averaging two signals (suspicious — looks like a prompt ensemble rather than a factorization). Product is the prediction-claim because it encodes AND semantics: a video must be BOTH about a protected group AND carry a derogatory stance to score high. If product does not strictly dominate sum or max on at least one dataset, the AND interpretation is wrong.

**Ablation D — "both calls with video vs. Call 2 with only text (v1-style)"**: deleting the video from Call 2 should degrade to v1 behavior (oracle drop, distribution compression). This ablation is already done — v1 is the data point. It confirms that video grounding on Call 2 is load-bearing.

Ablations A, B, C will be run using the same v2 scoring files via alternate aggregation formulas at eval time (no extra MLLM calls). Ablation D is already established by v1.

## 5. Named roles (2-call cap compliance)

| Call | Role name | Input | Output | Distinct-role justification |
|---|---|---|---|---|
| 1 | **Target-Detector** | frames + title + transcript, system="you identify whether a video concerns a protected-identity group" | single token P(Yes)/(P(Yes)+P(No)), allowed_token_ids=[Yes,No], max_tokens=1, logprobs=20 | Perception/existence query about protected-identity target. Ignores stance. |
| 2 | **Stance-Judge** | frames + title + transcript, system="you identify the derogatory stance of a video toward its subject" | single token P(Yes)/(P(Yes)+P(No)) | Sentiment/valence query about derogatory stance. Ignores whether target is protected. |

Score = P_T × P_S. Both calls use the same vLLM instance. Total calls/video = 2, at the cap.

Neither call is a resample, self-consistency pass, or multi-prompt pool of the other. Neither is a reword of baseline. Baseline's single call is *not* equivalent to either sub-call: baseline asks "does this violate policy" with the full 9-rule list; Call 1 asks "is there a protected-group target" with no rules; Call 2 asks "is the stance derogatory" with no rules and no identity specification.

## 6. Reproduction plan

### Code (under `src/prompt_paradigm/`, all NEW)

- `factored_verdict.py` — scorer. One vLLM instance, two-pass per batch:
  - Pass 1 (Target-Detector): build same media_content as baseline (frames or mp4), prompt with Call 1 system+user, `SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[Yes/No tids])`. Extract P_T via baseline's `extract_binary_score`-equivalent helper.
  - Pass 2 (Stance-Judge): same media_content, prompt with Call 2 system+user, same SamplingParams. Extract P_S.
  - Output: `{"video_id": str, "p_target": float, "p_stance": float, "score": p_target*p_stance}` per line to `results/prompt_paradigm/{MHClip_EN,MHClip_ZH}/{train,test}_factored.jsonl`. `score` field satisfies the `load_scores_file` contract in `src/quick_eval_all.py:92`.
- `eval_factored.py` — evaluator. Imports `otsu_threshold`, `gmm_threshold`, `metrics`, `load_scores_file`, `build_arrays` from `src.quick_eval_all` unchanged. Reports 4 (TR/TF × Otsu/GMM) cells. Also computes ablations A (P_T alone), B (P_S alone), C (sum) for diagnostic reporting — these are NOT candidate Gate 2 submissions, only confirmations of the load-bearing factorization. Writes `results/prompt_paradigm/report_v2.json` and `results/analysis/prompt_paradigm_report.md` (overwriting v1 report).

### Slurm plan (strict one-at-a-time-or-two-concurrent, per teammate brief §7)

All jobs logged in `docs/experiments/prompt_paradigm_runs.md` under a new "v2 — Factored Verdict" section. Each row: job ID, command, expected runtime, status, observed runtime, output path.

1. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/factored_verdict.py --dataset MHClip_EN --split test"` — ~20 min expected (161 videos × 2 calls).
2. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/factored_verdict.py --dataset MHClip_EN --split train"` — ~1 h expected.
3. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/factored_verdict.py --dataset MHClip_ZH --split test"` — ~20 min expected.
4. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/factored_verdict.py --dataset MHClip_ZH --split train"` — ~1 h expected.
5. `sbatch --cpus-per-task=2 --mem=4G --wrap "... python src/prompt_paradigm/eval_factored.py"` — ~1 min CPU.

Concurrent pairing: jobs 1+2 or 2+3 may run simultaneously (max 2 GPU jobs at once per teammate brief §7). No `--dependency`, no `&`, no chained submission.

### Decision rule at Gate 2

I will report the 4 (TR/TF × Otsu/GMM) cells on both datasets plus ablations A, B, C. The Gate 2 submission is the SINGLE unified cell (TR/TF × Otsu/GMM pair) — if no such cell exists, I report the miss and go to v3. I will NOT cherry-pick different cells for different datasets.

## 7. Label-free / external-data / frozen-file integrity

- No test labels touched in scoring, threshold fitting, or decision rule.
- Unlabeled train splits used only for TR-Otsu/TR-GMM threshold fitting, same as baseline.
- No auxiliary hate datasets, lexicons, retrieval corpora, or knowledge bases. Target/stance vocabulary is the model's pretrained general knowledge.
- No edits to `src/score_holistic_2b.py`, `src/score_holistic_8b.py`, `src/quick_eval_all.py`, `src/data_utils.py`, `results/holistic_*`, `CLAUDE.md`, `STATE_ARCHIVE.md`, `MEMORY.md`. Evaluator imports from `src.quick_eval_all` without modification.

## 8. Compliance self-audit (director's review categories)

- **Anti-pattern 1 (ensembling / repeated query)**: 2 calls, each with a distinct structural role (T-detection vs S-valence). Factorization via product is NOT sampling/voting/pooling — it is a compositional score formula where each factor targets a different failure mode of the single-call baseline. Distinct roles are justified mechanistically (perception vs. sentiment) and ablation-separable (ablation A, B, C). Within the 2-call hard cap.
- **Anti-pattern 2 (engineering trick without story)**: 4-point story present. Phenomenon: hateful = T ∧ S, falsifiable. Mechanism: factorize into perception and valence, both video-grounded. Prediction: pre-registered strict improvement on both EN and ZH + distribution signature. Counterfactual: ablations A/B/C predict specific directional failures. Removing EITHER call must break a NAMED failure mode (A: neutral discussions of minorities; B: non-target hostility).
- **Anti-pattern 3 (external datasets)**: none. Only MHClip's own unlabeled train splits and the pretrained Qwen3-VL-2B.
- **v1 failure continuity**: v2's root-cause diagnosis of v1 (text-only Judge calibration drift + description compression loss) is directly addressed by v2's design: both calls read raw video, no text summary flows between them. The v1 Observer-Judge sequential structure is abandoned; v2 is a parallel factorization of the question, not a sequential decomposition of the inference path.
- **Frozen files**: unchanged. Evaluator imports.
- **Slurm discipline**: one-at-a-time or 2-concurrent, no dependency chains, no scancel of foreign jobs, full run-log.
- **Model fix**: Qwen/Qwen3-VL-2B-Instruct only, bf16, vLLM, temperature=0.

## 9. What I am asking the director for

A Gate-1 rules-compliance ruling on whether this proposal violates the anti-patterns, the 2-call cap, the label-free integrity, the frozen-file rule, or the slurm discipline. If approved, I implement as written without further director input. If rejected, I redesign v3 on my own — I am not asking the director to suggest alternative factorizations or prompt wordings.
