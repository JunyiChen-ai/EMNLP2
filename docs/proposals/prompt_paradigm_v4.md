# prompt_paradigm v4 — Modality-Split Evidence Probes (Gate 1 proposal)

Author: prompt-paradigm (Teammate A)
Date: 2026-04-13
Status: Gate 1 — awaiting team-lead approval

---

## TL;DR

Two MLLM calls per video, each the SAME "violates rules?" evidence probe, but
operating on **disjoint input subsets**: Call 1 sees ONLY video frames
(no title, no transcript); Call 2 sees ONLY title + transcript (no frames).
Fusion is rank-space noisy-OR against a train-split reference distribution:

    score = 1 - (1 - rank_train(p_vis)) * (1 - rank_train(p_txt))

The two calls are not polarity-flipped, not sequential, not multiplicative in
probability space, and not using a shared input. They are two narrow
specialists running the same probe over structurally disjoint evidence
sources. The fusion is union-semantics (either specialist can fire), so
coverage is additive over the support of what a hateful video can look like.

---

## 1. Phenomenon (specific to hateful video)

**Hateful video is not modality-uniform.** In this project's benchmarks
(MHClip_EN, MHClip_ZH) and in the literature, the carrier of the hate signal
is split across videos:

- Some videos carry hate primarily in the **visual channel**: Nazi imagery,
  blackface, costumed mockery of a group's sacred symbol, caricatured
  depictions, dehumanizing animation. The title and transcript are benign or
  absent.
- Some videos carry hate primarily in the **text channel**: slurs, identity
  attacks, dog-whistle phrases, incitement calls. The visual content is
  generic or decorative — talking-head, lyric video, stock footage.
- A minority carry hate in BOTH channels simultaneously (overt, easy cases).

A MLLM fed all modalities in one prompt will, for compute and attention
reasons, implicitly average the two channels when issuing a single Yes/No
token. On a "half visual / half text" video, the visual evidence is diluted
by the neutral transcript, and vice versa. This is not a calibration error;
it is a **mixing error**: the joint prompt blurs a signal that is locally
strong on one modality but weak on the other.

This is a hateful-video-specific phenomenon. It does not arise in
image-only hate meme detection (one modality), and it does not arise in
text-only toxicity (one modality). It is a property of the multimodal,
variable-carrier nature of video hate content.

---

## 2. Mechanism

Run the same "violates rules?" probe TWICE with the same prompt template
and the same rule list, but with structurally disjoint evidence:

- **Call 1 — Visual-Evidence-Probe.** Media = video/frames. Text = rules
  + an empty title placeholder + an empty transcript placeholder. The
  MLLM's Yes-probability is a function of the visual channel alone.
- **Call 2 — Text-Evidence-Probe.** Media = none (no video, no frames).
  Text = rules + the actual title + the actual transcript. The MLLM's
  Yes-probability is a function of the text channel alone.

Both are still "evidence probes" (affirmative rule-violation framing) —
the polarity flip of v3 is not reused. Neither is a calibration step or a
bias-cancellation step. Both are perceptual queries over disjoint supports.

**Fusion.** Two probes with unknown and almost-certainly-different scale
factors cannot be linearly averaged safely (v3's empirical lesson). Rank
the raw p_vis and p_txt against the same probes' **train-split** score
distribution (label-free; train is just unlabeled population reference),
producing r_vis and r_txt in [0,1]. Combine with noisy-OR:

    score = 1 - (1 - r_vis) * (1 - r_txt)

Noisy-OR is the correct aggregation when (a) both specialists are
"positive detectors" (a high score means "I see evidence of hate") and
(b) their failures are decorrelated across the input support. Rank-space
makes the combination scale-free across the two heterogeneous calls.

---

## 3. Prediction and pre-commit

Directional prediction (no labels needed for the prediction — only for the
eventual check):

**H1.** The rank-AUC of visual-only probe ∪ text-only probe will exceed
the rank-AUC of the holistic probe (v3's p_evidence, which used both
modalities simultaneously) on at least one of EN / ZH. Specifically:

    AUC(noisy_OR_rank(p_vis, p_txt)) > max(AUC(p_vis), AUC(p_txt))
    AUC(noisy_OR_rank(p_vis, p_txt)) > AUC(v3 evidence_only) on at least one dataset

**H2.** At LEAST ONE of {tf_otsu, tf_gmm, tr_otsu, tr_gmm} on the fused
rank-space score strict-beats the current baseline under the same label-free
threshold family:

    EN: ACC > 0.7640 AND mF1 >= 0.6532
    ZH: ACC > 0.8121 AND mF1 >= 0.7871

**H3 (oracle pre-commit, binding).**

    EN oracle > 0.7764 (strict)
    ZH oracle > 0.8121 (strict)

If the oracle check fails on EITHER dataset, v4 Gate 2 is an automatic
MISS. v4 is retired and I will propose v5.

**What would falsify the story?** If rank-AUC of the fused score is NOT
greater than max(visual-only, text-only) — i.e., one modality is always
dominant and the other contributes no information — the
variable-carrier phenomenon does not hold in this data, and the mechanism
fails the load-bearing test.

---

## 4. Counterfactual ablations

Ablations reported at Gate 2 *even if the main result passes*. Each targets
the story, not just the component.

- **A. Visual-only specialist.** Score = p_vis alone. Test: if its oracle
  is already ≥ fused oracle on either dataset, the text channel contributes
  nothing and the story collapses (call-2-is-noise, v3 Ablation A analogue).
- **B. Text-only specialist.** Score = p_txt alone. Symmetric test.
- **C. Holistic specialist replica.** Use v3's existing p_evidence
  (same prompt, both modalities at once). Test: if v3 p_evidence oracle ≥
  v4 fused oracle on either dataset, the modality split yielded no
  signal beyond what the joint prompt already captured. This is the
  AP2-style "are we just re-using prior work?" self-check — the new
  contribution must come from the split, not from any prompt drift.
- **D. Aggregator replacement.** Replace rank-space noisy-OR with (i)
  prob-space average, (ii) rank-space average, (iii) rank-space max. Report
  all four. If any of them matches or beats noisy-OR, the
  "union-semantics" story is not load-bearing and v4 retires.
- **E. Rank decorrelation.** Report correlation(r_vis, r_txt) and the
  per-class cross-rescue rate:
  - Fraction of positives with low r_vis that have high r_txt (text
    rescues visual)
  - Fraction of positives with low r_txt that have high r_vis (visual
    rescues text)
  The story predicts both rates are non-trivial (> ~30%). If correlation
  is > 0.8 or if rescue rates are < 10%, the two specialists are
  redundant, not complementary, and the mechanism fails.

---

## 5. Binding Gate 2 clauses (carry forward)

v4's Gate 2 report must include, as binding pre-commit:

1. **Oracle-first (H3).** Strict beat on BOTH datasets, else MISS.
2. **Macro-F1 non-regression.** Unified cell's mF1 must be ≥ baseline
   mF1 on both datasets.
3. **Ablation A/B load-bearing (AP1 analogue).** If max(visual-only,
   text-only) oracle ≥ fused oracle on EITHER dataset, v4 retires —
   the fusion added nothing.
4. **Ablation C prior-art self-check.** If v3 p_evidence oracle ≥ v4
   fused oracle on EITHER dataset, v4 retires — the split added nothing
   over the holistic prompt.
5. **Ablation D aggregator robustness.** If any of {prob-avg, rank-avg,
   rank-max} matches or beats rank-noisy-OR oracle on both datasets,
   noisy-OR is not the key and v4 retires (same
   "not-the-specific-technique" rule as v3 AP1).
6. **Ablation E rescue-rate.** Cross-rescue rate ≥ 30% on both
   datasets; rank correlation < 0.7. If not met, the
   variable-carrier phenomenon is not empirically present.
7. **N_test reconciliation.** EN N=161, ZH N=149 (same as v2/v3).

---

## 6. How v4 addresses each prior failure mode

- **v1 — text-only Judge calibration drift.** v4 does NOT have text
  flow between calls. Call 2 is not a Judge of Call 1's output; it's a
  separate perceptual probe over disjoint inputs. No conditioning chain
  between the calls, no LLM-paraphrase drift.
- **v2 — multiplicative AND-gate compression.** v4's fusion is union
  (noisy-OR), not intersection (AND). A positive video only needs ONE
  specialist to fire. The AND-gate pathology cannot arise.
- **v3 — equal-weight logit fusion over polarity-flipped probes.** v4
  does NOT flip polarity. Both probes use the same "violates rules?"
  framing (not "consistent with rules?"). The failure mode we observed in
  v3 (Call 2 compliance probe has small, same-direction content signal on
  2B, so logit averaging adds noise) cannot arise because v4's Call 2 is
  the SAME framing as Call 1 and its content signal is (predicted to be)
  class-separating in the SAME direction as Call 1. The correlation
  between the two calls is what we're exploiting as a split — but
  crucially, on disjoint input supports, not on the same support with
  different wording.

All three prior mechanisms (text-cascade, AND-gate, polarity-flip logit
fusion) are structurally excluded from v4 by design.

---

## 7. Anti-pattern self-audit

- **AP1 (no ensembling).** v4 makes exactly 2 MLLM calls per video, each
  with a distinct, named role (visual specialist, text specialist). These
  are NOT K i.i.d. samples of the same prompt. They differ structurally
  in their input support, not in their random seed or decoding params.
  Ablation D tests whether the specific aggregator (rank-noisy-OR) is
  load-bearing; if a simple average matches it, v4 auto-retires. This
  binds v4 against the "we averaged things" failure mode.

- **AP2 (engineering trick).** The 4-point story above is the entire
  justification. The mechanism is NOT borrowed from another paper — it
  is motivated by a specific property of hateful video (variable
  modality carriage) that I can state and defend. Ablation C tests
  against v3's p_evidence (the joint-prompt replica); if the split adds
  nothing, the method is pure prompt engineering and retires. Ablation E
  tests whether the variable-carriage phenomenon is empirically present;
  if it is not, the story is unfalsifiable and retires.

- **AP3 (external datasets).** v4 uses only MHClip_EN and MHClip_ZH
  train-split as the rank reference (label-free — train labels are never
  read). No auxiliary hate dataset, no retrieval, no external lexicon.

---

## 8. Why I cannot give a numeric pre-pilot

I ran a pre-pilot on v3's existing scoring artefacts before drafting v4.
The result: under every threshold family (tf_otsu, tf_gmm, tr_otsu,
tr_gmm), ANY aggregator of v3's two score columns (p_evidence,
p_compliance) fails to strict-beat the unified cell on both datasets.
The best oracle achievable by any rank-fusion of v3's two probes is:

- noisy_OR_rank: EN 0.7888 ✓, ZH 0.8121 = (tie, not strict)
- avg_rank:      EN 0.7578 ✗, ZH 0.8322 ✓
- max_rank:      EN 0.7764 = (tie), ZH 0.8054 ✗

This tells me v3's two scoring columns DO NOT contain enough
separable information to clear the bar on both datasets under any
reweighting. v4 must change the *content* of the calls, not just the
fusion.

v4 requires new scoring runs because both Call 1 (frames-only, no
title/transcript) and Call 2 (title/transcript-only, no frames) are
genuinely new input configurations. I cannot pre-pilot them from v3
data. The 4-point story is my pre-experiment belief; H1/H2/H3 are my
falsifiable predictions. If H3 fails, v4 retires at Gate 2 — I will not
rescue it post-hoc.

---

## 9. Resources

- 2 new MLLM calls per video × 4 splits (EN/ZH × train/test) × ~700
  videos. ~2× the GPU hours of v3 = roughly 2.5 hours under the 2-GPU
  pairing strategy.
- New files:
  - `src/prompt_paradigm/modality_split.py` — scorer, outputs
    `{video_id, p_visual, p_text, score}` per split.
  - `src/prompt_paradigm/eval_modality.py` — evaluator with the 7
    binding clauses above. Writes
    `results/prompt_paradigm/report_v4.json` and updates
    `results/analysis/prompt_paradigm_report.md`.
- No edits to any frozen file. Evidence-probe prompt text copied verbatim
  from `src/score_holistic_2b.py:52-62`.
- Standing rules honored: Slurm discipline, 2-GPU cap, active monitoring.

---

## 10. Open question for team-lead

Is there a structurally different v4 you want me to consider instead?
I chose modality-split because:
(a) it is the *only* remaining 2-call structure that is not v1
(text-cascade), v2 (AND-gate), or v3 (polarity-flip) — the design space
under the 2-call cap is getting narrow;
(b) it has a hateful-video-specific story I can defend;
(c) the failure mode of every prior attempt has been "second call
doesn't carry orthogonal information" — disjoint inputs are the most
direct way to enforce orthogonality.

If you'd like me to reconsider — e.g., toward a 1-call v4 that
re-examines the existing evidence-only signal in a new light — I will
draft an alternative before submitting Slurm jobs. Awaiting Gate 1
approval.
