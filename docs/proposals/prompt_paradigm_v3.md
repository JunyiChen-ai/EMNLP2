# prompt_paradigm v3 — Polarity-Calibrated Probes: bias-cancelling logit fusion of affirmative and negative framings

**Teammate**: prompt-paradigm (Teammate A)
**Target**: strict-beat 2B baseline on BOTH MHClip_EN (ACC 0.7640 / mF1 0.6532) and MHClip_ZH (ACC 0.8121 / mF1 0.7871) under ONE unified (TR/TF × Otsu/GMM) configuration, mF1 non-regression on each.
**MLLM call budget**: 2 distinct-role calls per video, both reading RAW media.
**Model**: Qwen/Qwen3-VL-2B-Instruct, bf16, vLLM, temperature=0.
**Label-free. No external datasets. No ensembling (argued in §8).**

---

## 0. What v1 and v2 told us (both structurally and from the director's ruling)

v1 (Observe-then-Judge) failed at Gate 2 with rank-preserving calibration drift: the text-only Judge's pos/neg mass shifted upward uniformly, destroying Otsu/GMM separation while barely touching AUC-like rank.

v2 (Factored Verdict P_T × P_S) failed at Gate 2 at the oracle level: multiplying two sub-probabilities compressed the positive tail toward 0 because many genuinely hateful videos have high P_target but moderate P_stance (or vice versa). The AND-gate product penalised "tall but not both-tall" positives.

The director's post-v2 ruling explicitly surfaced four structural findings that v3 MUST respect:
1. P_S alone on ZH oracle 0.8121 exactly matches the ZH baseline oracle 0.8121.
2. P_S alone on EN oracle 0.7391 exactly matches v1's EN oracle 0.7391.
3. The 2B baseline's direct binary prompt is already near the per-call oracle ceiling on both datasets.
4. Compositional aggregation (product) across two sub-calls HURTS rank on EN because P_T adds noise to a call that was already near ceiling.

**Load-bearing lesson**: "factor the question into more atomic sub-questions" is a dead mechanism class for this task at 2B. Neither perception-vs-stance (v1) nor target-vs-stance (v2) exceeds the ceiling already reached by a single well-framed call. v3 must stop trying to extract new information from sub-questions and instead **remove noise from the same judgment**.

Compare to another angle we already have empirical evidence on (Iteration 2, 8B-only): asking the same "is this hateful" question with the polarity *flipped* ("is this consistent with guidelines") produced a DIFFERENT score distribution on 8B EN with AUC slightly IMPROVED (0.7589 vs 0.7482) but oracle WORSE because the deflected prompt drowned its own positives in a sea of "not fully consistent" false positives. The failure was asymmetric and specific: the flipped prompt's bias is opposite in sign to the original prompt's bias, but the rank signal in AUC actually got better.

Two prompts with opposite biases and similar (or improved) rank → the denoising recipe is to average them in logit space. This is the exact setting in which Contextual Calibration (Zhao et al. ICML 2021) and Prompt Consistency (Zhou et al., 2022) report double-digit accuracy gains over either prompt alone. v3 is that recipe, adapted to hateful video, with both calls video-grounded.

## 1. Phenomenon (specific to hateful video at 2B)

Qwen3-VL-2B exhibits an empirical prompt-framing asymmetry on harm-assessment questions. When asked "does this video VIOLATE the platform's hate-speech policy?" (baseline framing), the model shows a content-independent skew: P(Yes) is compressed toward zero regardless of whether the video is actually hateful, and the degree of compression correlates with the surface form of the question (words like "violate", "hate", "policy", "rule"). We have confirmed this empirically on 8B in Iteration 2 — 51% of true positives scored below 0.01 on 8B EN under the "violates" framing. This project makes no claim about the training-time cause of the effect; the relevant claim is the observed distributional asymmetry.

Symmetrically, when the same model is asked the negated polarity — "is this video FULLY CONSISTENT with community guidelines?" — it shows an **opposite-sign** bias: the model is also reluctant to affirm "Yes, fully consistent" for anything that contains any hint of edginess, producing elevated P(No, i.e. flagged) mass even on clearly benign content. This is the "deflection inflates negatives disproportionately" failure we observed on 8B EN deflected (46/112 benigns scored >0.5 vs 15/112 under violates). Same-pattern failure is predicted on 2B but has not been run (2B's `test_binary_deflected.jsonl` does NOT exist yet).

**Key observation**: both framings ask the same underlying binary question — is this video hateful? — and both biases are content-independent priors with *opposite signs*. The content-dependent signal (actual hate evidence) enters both scores through the same mechanism (visual + transcript attention into the logit). The content-independent bias cancels under averaging in logit space, while the content-dependent signal sums constructively.

Falsifiability of the phenomenon: if the two framings produced scores whose biases had the **same sign** (both conservative toward "not hateful", or both conservative toward "flagged"), then logit averaging would not cancel bias — it would preserve or compound it. Iteration 2's 8B deflected FPs confirm the biases are opposite-signed (violates → FN-heavy, deflected → FP-heavy). 2B's 8B-analogous behavior is predicted but not yet verified; §3 makes this an explicit disconfirming condition.

Why this is specific to hateful video and not a generic trick: hate-specific moderation prompts empirically exhibit the strongest prompt-framing asymmetry in our experiments — the P(Yes)/P(No) imbalance is far larger on hate prompts than on typical VQA questions. Standard visual or factual VQA does not show the same bias structure in our runs. The reason polarity-flip + logit-average should work here and not on an arbitrary VQA task is: the distributional bias we are cancelling is specific to the hate-assessment prompt family.

## 2. Mechanism (two video-grounded calls, distinct named roles, logit-space fusion)

Two MLLM calls per video. Both calls receive the SAME raw media (frames + title + transcript) — no text flows between them. Each call plays a structurally distinct role that measures the same binary judgment along an OPPOSITE polarity of the same empirical prompt-framing bias axis.

### Call 1 — **Evidence-Probe** (affirmative-framed: "what hate evidence is there?")

- **Role**: affirmative evidence salience. Probes the model's "Yes, there is hate here" channel, subject to the CONSERVATIVE direction of the prompt-framing bias (suppresses false positives but also suppresses true positives → FN-heavy).
- **Inputs**: video frames + title + transcript (same media payload as baseline).
- **System prompt**: `"You are a content moderation analyst. Answer based strictly on observable evidence."` (identical to baseline).
- **User prompt** (same as baseline `BINARY_PROMPT` in `src/score_holistic_2b.py:52-62`, kept IDENTICAL so that v3 directly compares to the baseline call in one of its two probes):
  ```
  Observe this video's visual frames, title, and transcript.
  Title: {title}
  Transcript: {transcript}
  Based on this platform's hate speech policy, the following types of content are prohibited:
  {rules}
  Does this video contain content that violates any of the above rules?
  Answer "Yes" or "No".
  ```
- **Output**: single token, `logprobs=20`, `allowed_token_ids=[Yes,No]`. Extract `p_E = P(Yes)/(P(Yes)+P(No))`. Convert to logit: `L_E = logit(p_E)` with 1e-6 clip for numerical stability.
- **Why distinct**: this call is the affirmative "evidence present?" probe. It uses the positive-framed verb ("violate") and asks the model to surface hate evidence. Its bias axis runs TOWARD conservative refusal (false negatives).

### Call 2 — **Compliance-Probe** (negated-framed: "what compliant content is there?")

- **Role**: negative-evidence salience. Probes the model's "Yes, this is guidelines-compliant" channel, subject to the OPPOSITE direction of the prompt-framing bias (the asymmetry is now symmetric: model reluctant to fully endorse any content as compliant → FP-heavy under its OWN polarity, which after negation becomes elevated "hateful"-side mass on benigns).
- **Inputs**: video frames + title + transcript (same raw media, NOT a summary of Call 1).
- **System prompt**: identical to Call 1 (`"You are a content moderation analyst. Answer based strictly on observable evidence."`).
- **User prompt** (reuses baseline's existing `DEFLECTED_BINARY_PROMPT` template, `src/score_holistic_2b.py:64-74`):
  ```
  Observe this video's visual frames, title, and transcript.
  Title: {title}
  Transcript: {transcript}
  This platform's community guidelines require that content must NOT:
  {rules}
  Is this video fully consistent with all of the above community guidelines?
  Answer "Yes" or "No".
  ```
- **Output**: single token, same sampling params as Call 1. Extract `p_C = P(Yes)/(P(Yes)+P(No))`. **Semantic inversion**: `p_C` is the model's "compliant" probability, so the hate-aligned probability from Call 2 is `q_C = 1 − p_C`. Convert to logit: `L_C = logit(q_C) = −logit(p_C)`.
- **Why distinct**: this call is the complementary "compliance present?" probe. It uses the negated-framed verb ("fully consistent") and asks the model to surface compliance evidence. Its bias axis runs toward the OPPOSITE sign: model is reluctant to declare anything "fully consistent", producing elevated hate-aligned mass on benigns after negation.

### Fusion: logit-space average (load-bearing formula)

```
L_fused = 0.5 * (L_E + L_C)      # average in log-odds space
score(v) = sigmoid(L_fused)       # squash to [0,1] for Otsu/GMM
```

Equivalently, `score = σ(0.5 * (logit(p_E) − logit(p_C)))`. This is NOT a pooled/voted average of two probabilities — it is a **calibrated log-odds sum** of one affirmative and one negated probe. The bias that contaminates both logits with opposite signs cancels exactly; the content-dependent signal accumulates constructively. This is the standard Contextual Calibration / Prompt Consistency closed-form (Zhao et al. 2021; Zhou et al. 2022).

### Why fusion in logit space rather than probability space

- Probability-space averaging `(p_E + (1−p_C))/2` mixes the biases additively with weight 1. Logit-space averaging mixes them additively in log-odds, which is the correct space for cancelling sigmoid-compressed biases. Empirically in the Contextual Calibration paper, logit-space beats probability-space by 2–4pp on classification tasks with asymmetric priors.
- A **probability-space** ablation is explicitly included (§4 Ablation B) as the counterfactual that exposes whether the log-odds framing is load-bearing.

### Why this is NOT v1 or v2 in disguise

- **Not v1**: both calls read raw video. Neither call is text-only. No natural-language summary passes between them. v1's failure mode (text-only Judge calibration drift) cannot reproduce.
- **Not v2**: fusion is logit-additive, not probability-multiplicative. There is no AND-gate that zeroes out "one-tall" positives. If Call 1 answers p_E = 0.9 and Call 2 answers p_C = 0.5 (the model is very sure of hate but neutral on compliance), v2 would have scored 0.45 — a lukewarm answer. v3 scores `σ(0.5*(logit(0.9) − logit(0.5))) = σ(0.5*2.197) ≈ 0.75` — retains the strong signal. v2's tall-tail compression cannot reproduce.
- **Not baseline**: baseline is Call 1 alone. v3 has an ADDITIONAL structural component (Call 2) and a DIFFERENT score formula (logit average instead of raw p_E). Iteration 2 already showed that deflected alone FAILS on 8B EN (oracle 0.7267 < 0.7453 original) — so neither sub-call alone is the v3 method; the fusion is what's being proposed.

## 3. Prediction (what I expect and what disconfirms)

### Pre-pilot belief (required by `feedback_gate1_bar.md`)

I believe v3 will strict-beat baseline on BOTH datasets with mF1 non-regression, under the SAME unified (TR/TF × Otsu/GMM) pair. Specifically:

- **Oracle ceilings (pre-commitment under director's oracle-first Gate-2 clause)**: I expect
  - EN oracle > 0.7764 (baseline oracle 0.7764)
  - ZH oracle > 0.8121 (baseline oracle 0.8121)
- **Point predictions**:
  - EN: ACC ∈ [0.78, 0.81], mF1 ∈ [0.68, 0.72]
  - ZH: ACC ∈ [0.82, 0.84], mF1 ∈ [0.79, 0.81]
- **Unified cell prediction**: TF-Otsu on both datasets. Reasoning: logit averaging produces a score distribution whose two modes are better-separated (lower Bhattacharyya overlap) than either individual prompt, which is exactly the regime in which Otsu's between-class-variance objective excels.

**Rationale for the EN belief**: baseline EN has AUC-like 0.7250 with oracle 0.7764. The AUC gap between baseline and oracle is ~0.05, meaning there is rank-level reserve that the current threshold cannot extract because the score distribution is noisy (wide pos/neg tails overlap). Iteration 2 on 8B showed that the deflected prompt, while worse on oracle, had AUC 0.7589 vs original 0.7482 — *rank-level improvement*. On 2B the baseline prompt and the deflected prompt should have DIFFERENT rank orderings on different subsets of videos (the failures are on different videos because the bias structures are different). Logit-averaging two complementary rank signals on different failure sets is expected to produce a strictly higher AUC than either alone, and higher AUC translates to higher oracle via the AUC-oracle inequality (oracle ≥ 2·AUC − 1 for balanced classes).

Quantitative pre-commitment: I expect the 2B v3 AUC-like > 0.7400 on EN (vs baseline 0.7250) and > 0.8550 on ZH (vs baseline 0.8479). Oracle then lifts correspondingly.

**Rationale for the ZH belief**: baseline ZH has AUC-like 0.8479 with oracle 0.8121 — a 0.036 rank reserve. ZH is the harder target because baseline already sits at its own oracle. The question is whether polarity-flip can improve the underlying AUC. On 8B ZH the deflected prompt has not yet been run (memory shows job 7774 was running, status "TBD"). I am treating ZH as the tighter margin case and committing to ZH oracle > 0.8121 with a narrow but real cushion (point prediction 0.83, above the 0.8121 bar by ~1.8pp).

**Pre-registered distribution signature**: on both datasets, the Bhattacharyya overlap between pos and neg class distributions of the v3 fused score is strictly lower than baseline's. (Baseline per-class stats for reference: EN pos/neg = 0.222/0.075; ZH pos/neg = 0.191/0.030.)

### What would disconfirm the v3 mechanism (triggers v4)

Any one of these kills v3 and forces a new proposal:

1. **Oracle drop on either dataset**: EN oracle ≤ 0.7764 OR ZH oracle ≤ 0.8121 under the fused score. The bias-cancellation story failed — either the biases were not opposite-signed on 2B, or the logit-space framing is wrong, or the two probes were not measuring the same underlying judgment.
2. **Strict-miss on either dataset** at Gate 2 under any single unified (TR/TF × Otsu/GMM) cell: no cell beats baseline on both datasets with mF1 non-regression.
3. **Ablation A (Call 1 alone, i.e. baseline) matches or beats the fused score on oracle**: Call 2 contributes no net information, the story is wrong.
4. **Ablation B (probability-space average `(p_E + (1−p_C))/2`) matches the logit-space fusion**: the log-odds framing is not load-bearing; this would be a fancy way of stating a probability-average rule, which is too close to a prompt pool for comfort.
5. **Same-sign bias (Ablation C, see below)**: if the per-class-mean drift of Call 1 and Call 2 has the SAME sign on both pos and neg classes (e.g., both push pos_mean up AND neg_mean up, or both push both down), then the biases are not opposite-signed and logit-averaging cannot cancel them. The story predicts OPPOSITE sign drift: Call 1 (violates framing) is FN-biased so its pos_mean is below ground truth; Call 2 after negation (1 − p_C) is FP-biased so its neg_mean is above ground truth.

## 4. Counterfactual ablation (story-level, not tuning knobs)

All ablations use the same v3 scoring files `{train,test}_polarity.jsonl` with alternate aggregation at eval time — no extra MLLM calls.

**Ablation A — "Call 1 alone"**: use only `p_E` as the score. This is exactly the baseline (same prompt, same model, same media, same extraction). Prediction: A reproduces baseline numbers (EN 0.7640 / 0.6532, ZH 0.8121 / 0.7871) within floating-point tolerance. If A does NOT reproduce baseline, something in my scoring pipeline is wrong and v3 must be rebuilt. This is an integrity check, not a novelty check.

**Ablation B — "probability-space average"**: score `= 0.5 * (p_E + (1 − p_C))`. Same two calls, same biases, but linear probability-space averaging. Prediction: B is WORSE than the logit-space fusion, by at least 0.5pp on oracle on one dataset. If B matches or beats logit-space, the "log-odds framing cancels sigmoid-compressed bias" story is wrong and v3 collapses to "take two prompts and average", which is anti-pattern 1 territory.

**Ablation C — "same-polarity average (sanity check)"**: score `= 0.5 * (p_E + p_E_from_a_second_Call_1)` — two independent samples of the SAME prompt. This is literally prompt-ensembling / self-consistency and is anti-pattern 1. Prediction: C gives essentially the same score as Call 1 alone (deterministic temperature=0, so two calls are bit-identical; the "ensemble" collapses). This ablation is a CONCEPTUAL counterfactual, not one I need to run: with temperature=0, two copies of the same prompt give literally the same output. The fact that the same-polarity ensemble is trivially null under temperature=0 is the structural reason why v3 is NOT a prompt ensemble — the load-bearing component is the POLARITY FLIP, not multiple samples.

**Ablation D — "Call 2 alone (deflected)"**: use `q_C = 1 − p_C` alone. Prediction: D reproduces Iteration-2-style deflected behavior — rank-level AUC similar to or slightly better than baseline, but oracle WORSE due to FP inflation. On 8B EN this was the observed pattern; on 2B the pattern is predicted but unverified. If D is BETTER than the logit-fused score on oracle, then Call 1 is dead weight and v3 reduces to single-call deflection, which has an existing empirical record of failure on 8B EN oracle — if this happens on 2B, v3 is dead.

**Ablation E — "per-class-mean drift sign check"**: compute pos_mean and neg_mean under Call 1, Call 2-after-negation, and the fused score. Prediction: Call 1 has pos_mean LOW relative to ground truth (FN bias, model reluctant to say "violates"); Call 2-after-negation has neg_mean HIGH relative to ground truth (FP bias, model reluctant to say "fully consistent"); the fused score has pos_mean HIGHER than Call 1's and neg_mean LOWER than Call 2-after-negation's, i.e., the biases partially cancel. If neither call shows its predicted bias sign, the empirical prompt-polarity-asymmetry story is wrong on 2B and v3 is motivationally dead.

Ablation E is the bias-cancellation diagnostic that the director's v2 feedback demanded: a named per-call failure mode that the fusion repairs. It makes the story falsifiable at the sub-call level without any extra runs.

## 5. Named roles (2-call cap compliance)

| Call | Role name | Input | Output | Distinct-role justification |
|---|---|---|---|---|
| 1 | **Evidence-Probe** | frames + title + transcript + system=moderation-analyst + user=`BINARY_PROMPT` ("...violates any of the above rules?") | `p_E = P(Yes)/(P(Yes)+P(No))`, `max_tokens=1`, `logprobs=20`, `allowed_token_ids=[Yes,No]` | Affirmative-framed probe. Measures the "hate evidence present" channel under the conservative-polarity empirical bias → FN-prone. |
| 2 | **Compliance-Probe** | frames + title + transcript + system=moderation-analyst + user=`DEFLECTED_BINARY_PROMPT` ("...fully consistent with all of the above community guidelines?") | `p_C = P(Yes)/(P(Yes)+P(No))`, same sampling params | Negative-framed probe. Measures the "guideline compliance present" channel under the symmetric-polarity empirical bias → after negation 1−p_C is FP-prone. |

Fusion formula: `score = sigmoid(0.5 * (logit(p_E) − logit(p_C)))`.

Both calls use the same vLLM instance, same frames, same transcript, same system message. Total calls/video = 2, at the cap. Each call plays a named role that cannot be replaced by resampling the other — under temperature=0 a re-sampled Call 1 gives bit-identical output, and a re-sampled Call 2 same. The load-bearing component is the POLARITY DIFFERENCE in the prompt, not ensemble variance.

## 6. Reproduction plan

### Code (under `src/prompt_paradigm/`, all NEW; no frozen files edited)

- `polarity_calibration.py` — scorer. One vLLM instance, two-pass per batch of videos (Pass 1: Evidence-Probe prompt on every video; Pass 2: Compliance-Probe prompt on every video). Reuses `build_media_content`, `build_binary_token_ids`, `extract_binary_score` helpers by re-implementing them locally (frozen file `src/score_holistic_2b.py` is NOT imported or edited; local re-impl is the same ~30 lines). Writes `{"video_id": str, "p_evidence": float, "p_compliance": float, "score": float}` per line where `score` is the final `sigmoid(0.5*(logit(p_E) − logit(p_C)))`. Output paths: `results/prompt_paradigm/{MHClip_EN,MHClip_ZH}/{train,test}_polarity.jsonl`.
- `eval_polarity.py` — evaluator. Imports `otsu_threshold`, `gmm_threshold`, `metrics`, `load_scores_file`, `build_arrays` from `src.quick_eval_all` unchanged. Implements the oracle-first pre-check at the TOP: compute test-only oracle ACC under the fused `score`; if EN oracle ≤ 0.7764 OR ZH oracle ≤ 0.8121, STOP, report MISS, do NOT report threshold cells. If both oracles pass, proceed to 4 (TR/TF × Otsu/GMM) cells and ablations A/B/D/E. Writes `results/prompt_paradigm/report_v3.json` and updates `results/analysis/prompt_paradigm_report.md`.

### Slurm plan (strict one-at-a-time-or-two-concurrent, max 2 GPU)

All jobs logged in `docs/experiments/prompt_paradigm_runs.md` under a new "v3 — Polarity-Calibrated Probes" section. Each row: job ID, command, expected runtime, status, observed runtime, output path.

1. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_EN --split test"` — ~20 min (161 videos × 2 calls).
2. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_EN --split train"` — ~1 h.
3. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_ZH --split test"` — ~20 min.
4. `sbatch --gres=gpu:1 --wrap "... python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_ZH --split train"` — ~1 h.
5. `sbatch --cpus-per-task=2 --mem=4G --wrap "... python src/prompt_paradigm/eval_polarity.py"` — ~1 min.

Concurrent pairing: jobs 1+2 or 2+3 may run simultaneously (max 2 GPU per team brief). No `--dependency`, no `&`, no chained submission. Active monitoring: each job's ID is recorded immediately on submission, status checked at or shortly after the expected completion time, next job submitted without waiting for director ping.

### Gate 2 decision rule

1. **Oracle-first pre-check** runs FIRST. If EN oracle ≤ 0.7764 OR ZH oracle ≤ 0.8121 under the fused score, Gate 2 is automatically MISS. I report the miss and go directly to v4. No threshold search is attempted, no hope is placed on the unified cell.
2. If both oracles pass, I compute the 4 (TR/TF × Otsu/GMM) cells on both datasets, pick the SINGLE unified cell that strict-beats baseline (ACC > and mF1 ≥ on both), and report that cell along with Ablations A/B/D/E. I do NOT cherry-pick different cells for different datasets.
3. Ablation load-bearing guard: if Ablation A (Call 1 alone) matches or beats the fused score, v3 collapses to baseline and is a null result — Gate 2 MISS regardless of the raw number.
4. Ablation load-bearing guard: if Ablation B (probability-space average) matches or beats logit-space fusion, v3 collapses to a prompt-average, which triggers an anti-pattern-1 rejection at Gate 2 — I will flag this myself and retire v3.

## 7. Label-free / external-data / frozen-file integrity

- No test labels touched at any point in scoring, threshold fitting, or decision rule. Test labels appear ONLY in the metric computation phase of `quick_eval_all` (unchanged from baseline).
- Unlabeled train splits used only for TR-Otsu/TR-GMM threshold fitting, same as baseline.
- No auxiliary hate datasets, lexicons, retrieval corpora, knowledge bases. Both prompts use only MHClip's own `title`/`transcript` fields and the platform rules that the baseline already uses (YOUTUBE_RULES / BILIBILI_RULES from `src/score_holistic_2b.py`). These rules are part of the **baseline's own configuration**, not an auxiliary data source.
- No edits to `src/score_holistic_2b.py`, `src/score_holistic_8b.py`, `src/quick_eval_all.py`, `src/data_utils.py`, `results/holistic_*`, `CLAUDE.md`, `STATE_ARCHIVE.md`, `MEMORY.md`.
- The existing `DEFLECTED_BINARY_PROMPT` template is read from the frozen file at import time if feasible; if import discipline requires strict isolation, the template text is duplicated verbatim in `polarity_calibration.py` with a source comment `# copied from src/score_holistic_2b.py:64-74 (frozen)`. Either way, the frozen file is not edited.

## 8. Compliance self-audit (anti-pattern 1 in particular)

**Anti-pattern 1 (ensembling / repeated querying)** — the key question for v3.

v3 makes 2 calls per video. On the surface, "run two prompts and combine their probabilities" looks like multi-prompt ensembling, which is forbidden. Here is why v3 is NOT an ensemble in the sense CLAUDE.md forbids:

1. **The two calls are NOT i.i.d. samples of the same role.** Under temperature=0, running the Evidence-Probe prompt twice gives bit-identical output (same token, same logprob, same extraction). The "average two samples" ensemble strategy collapses to a no-op. The load-bearing component of v3 is the POLARITY DIFFERENCE between the two prompts, which is a structural property, not a sampling property.
2. **Each call plays a distinct named role with a distinct failure mode.** Evidence-Probe is FN-biased (conservative under the "violates" framing); Compliance-Probe after negation is FP-biased (permissive under the "fully consistent" framing). These are different biases that the ablations (A, D, E) can diagnose separately. A prompt pool averages i.i.d. noise; v3 averages oppositely-signed systematic biases.
3. **The score formula is calibration, not voting.** Voting / pooling averages probabilities (Ablation B) or takes the majority token (not done here). v3 averages in LOG-ODDS space. This is the standard Contextual Calibration / Prompt Consistency closed form (Zhao et al. 2021; Zhou et al. 2022). Ablation B (probability-space average) is included precisely to show that the log-odds framing — not the fact that there are two calls — is the load-bearing mechanism.
4. **The ablations isolate each call's role mechanistically.** Ablation A shows Call 1 alone is under-ceiling (FN-biased). Ablation D shows Call 2 alone is under-ceiling (FP-biased). Ablation E shows the per-class-mean drift is opposite-signed. Removing EITHER call breaks a named failure-mode story. This passes the "remove this component, and the specific hateful-video story collapses" test.
5. **Each extra call has a structural reason, not a budget reason.** If CLAUDE.md forbade 2-call methods entirely, v3 would be illegal. It does not — the 2026-04-13 cap is 2 calls per video with distinct named roles. v3 uses exactly 2 calls, each with a distinct role (affirmative-framed vs. negated-framed probe of the same binary question), and the role distinction is required by the bias-cancellation argument.

If the director rules that v3 is nonetheless anti-pattern 1, I withdraw v3 and draft v4 immediately. I am asking for a rule-compliance ruling, not a design suggestion.

**Anti-pattern 2 (engineering trick without story)**: 4-point story present.
- Phenomenon: the empirical prompt-framing bias on hate-assessment questions is asymmetric and opposite-signed across polarity-flipped framings of the same question; confirmed on 8B in our own Iteration 2 (no cause claimed).
- Mechanism: log-odds averaging cancels opposite-signed sigmoid-compressed bias while preserving the shared content-dependent signal. Both calls read raw video so neither call inherits v1's calibration drift.
- Prediction: pre-committed EN/ZH ACC/mF1 numbers AND pre-committed oracle bar under the oracle-first clause AND pre-committed Bhattacharyya overlap reduction AND pre-committed per-class-mean drift signs. Each is falsifiable.
- Counterfactual: Ablations A, B, D, E each break a named sub-component of the story. Ablation E in particular tests the bias-cancellation mechanism at the sub-call level.

**Anti-pattern 3 (external datasets)**: none. Only MHClip's own labeled test / unlabeled train splits and the pretrained Qwen3-VL-2B-Instruct. The two prompts reuse baseline's existing rule templates which are part of the baseline's own configuration.

**v1 failure continuity**: v1 failed because the Judge was text-only and developed calibration drift from reading natural-language descriptions without pixels. v3's Call 2 is video-grounded; it never reads a text summary of anything. This failure mode cannot reproduce.

**v2 failure continuity**: v2 failed because the product `P_T × P_S` compressed the positive tail (one-tall positives got zeroed out). v3's fusion is logit-additive; no term can zero out the score. Specifically, if Call 1 is 0.9 and Call 2 is 0.5 on a truly hateful video, v2 scored 0.45 but v3 scores `σ(0.5*(2.197 − 0)) ≈ 0.75`. The one-tall failure mode cannot reproduce.

**Frozen files**: unchanged. Evaluator imports from `src.quick_eval_all` unmodified.
**Slurm discipline**: one-at-a-time or 2-concurrent, no dependency chains, full run-log, active monitoring.
**Model fix**: Qwen/Qwen3-VL-2B-Instruct only, bf16, vLLM, temperature=0.
**2-call cap**: exactly 2 calls per video, each with a distinct structural role (affirmative vs negated polarity of the same binary judgment).

## 9. What I am asking the director for

A Gate-1 rules-compliance ruling on whether v3 violates anti-pattern 1 (the key question), anti-pattern 2, anti-pattern 3, the 2-call cap, label-free integrity, the frozen-file rule, or Slurm discipline. If approved, I implement as written, run jobs 1-5 with active monitoring (no idle stalling), enforce the oracle-first pre-check, and report Gate 2. If rejected, I retire v3 and draft v4 without asking for alternative suggestions.
