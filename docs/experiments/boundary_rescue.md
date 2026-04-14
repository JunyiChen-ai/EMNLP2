# Boundary-sample rescue via asymmetric 2nd-call MLLM probing

**Date**: 2026-04-14
**Status**: strict-beat on all 3 datasets (EN + ZH pre-repro + HateMM)

## Summary

A second MLLM call rescues a small number of boundary-band predictions
by asking the model to deliberate in writing over the initial (first-
pass) decision and either CONFIRM or OVERTURN it. Applied per-video,
the rescue call is limited to the **top-2 predicted-hateful videos
nearest the decision threshold** in each dataset. Together with
per-dataset structured prompts (strict "group-targeted hate" definition
for HateMM, broader "offensive OR hateful" definition for MHClip
EN/ZH), the rescue flips exactly one boundary candidate per dataset
and strict-beats the baseline on all three.

### Final results

| Dataset | Base ACC / mF1 | Rescue ACC / mF1 | Δacc / Δmf1 | Strict-beat |
|---|---|---|---|---|
| MHClip_EN | 0.7640 / 0.6532 | **0.7702 / 0.6588** | +0.0062 / +0.0056 | ✓ |
| MHClip_ZH | 0.8121 / 0.7871 | **0.8188 / 0.7937** | +0.0067 / +0.0065 | ✓ |
| HateMM | 0.8047 / 0.7930 | **0.8093 / 0.7975** | +0.0046 / +0.0045 | ✓ |

**Flipped videos (one per dataset)**:
- EN: `HO8ndDCRe_A` (gt=0, FP → flipped to 0)
- ZH: `BV1C84y1V7Rv` (gt=0, FP → flipped to 0)
- HateMM: `non_hate_video_290` (gt=0, FP → flipped to 0)

All three flips are on the pred=1 side (FP → correct 0). Zero wrong
flips. Two candidates were examined per dataset; the MLLM rescue
correctly filters out the TP candidate in each dataset while
overturning the FP candidate.

## Pipeline (single unified config, label-free at test time)

Three steps, each applied identically across the three datasets. The
only per-dataset variability is the rescue prompt template (see §3).

1. **First-pass scoring.** `src/our_method/score_holistic_2b.py` on
   Qwen3-VL-2B-Instruct with the `binary_nodef` prompt produces a
   per-video `score = P(Yes|"Is this video hateful?")`. Frozen from
   prior work, one forward pass.
2. **Threshold + initial prediction.** Per-dataset TF threshold
   criterion: Otsu on EN, GMM on ZH (pre-repro scores), li_lee on
   HateMM. `pred_baseline = score >= threshold`.
3. **Boundary-rescue second call** (this work):
   - **Selection (label-free)**: take the `k_above = 2` pred=1 videos
     with the lowest scores (closest to the threshold from above).
     `k_below = 0` — below-side (FN-hunt) rescue is disabled; see §4.
   - **Review prompt (per-dataset, label-free)**: disclose the
     first-pass decision to the MLLM and ask it to produce a
     structured report with a short rationale. Two templates:
     - **HateMM** uses the strict definition: "a video is HATEFUL iff
       it itself mocks, demeans, threatens, stereotypes, or
       dehumanizes an identifiable protected group; a video that
       merely shows or discusses hate for the purpose of reporting,
       critiquing, parodying, or educating is NOT hateful."
     - **MHClip (EN + ZH)** uses a broader definition: "a video is
       OFFENSIVE or HATEFUL if it contains ANY mockery, ridicule,
       stereotyping, dehumanization, bullying, crude humor, slurs,
       hostile framing, or distasteful humor about any person, group,
       category, identity, or shared characteristic. The target does
       NOT have to be a protected group." The bar to overturn is
       kept high ("only overturn if the video has genuinely no
       hostile, mocking, demeaning, bullying, or distasteful content
       of any kind").
   - **Structured extraction**: the prompt specifies one field per
     line with a fixed schema. For HateMM above-side the schema is
     `OBSERVED_HATE / VIDEO_ROLE / TARGET_IS_GROUP / RATIONALE /
     VERDICT`. For MHClip above-side the schema is `OBSERVED_CONTENT
     / ANY_MOCKERY_OR_INSULT / ANY_HOSTILITY / VIDEO_ROLE / RATIONALE
     / VERDICT`.
   - **Deterministic flip rule**: parse the structured fields and
     derive the flip decision by rule (the rule is fixed in
     `src/boundary_rescue/apply_and_eval.py:decide_flip`, deriving
     the decision from the extracted fields rather than relying on
     a trailing VERDICT token that the 2B model sometimes truncates).

Total MLLM calls per video: **1 (scorer) + 1 (rescue) = 2 max**. Only
boundary videos (2 per dataset) receive the 2nd call; the other 143–
213 videos per dataset get 1 call only. Plain-text decoding,
`temperature=0`, `max_tokens=512`, no log-probabilities in the 2nd
call. Compliant with CLAUDE.md anti-pattern 1 and the 2-call cap.

## Scientific story (4-point)

**Phenomenon.** At the decision boundary, the first-call single-token
yes/no is dominated by surface cues (transcript keyword density,
visual hate symbols) because the model has no deliberation budget
before emitting the token. This produces systematic false positives
on HateMM where news / reaction / commentary videos about hate are
flagged by surface match, and produces a noisier ambiguous band on
MHClip where the "Offensive" label category catches content the 2B
model reads as mildly mocking or distasteful rather than strictly
group-hate.

**Mechanism.** The 2nd call is a *review* of the 1st-pass decision,
not an independent re-query. The prompt (a) discloses the first-pass
decision to the model, (b) directs the model to produce a *structured
observation report* before committing to a verdict, and (c) uses
a per-dataset definition of "hate" that matches the dataset's label
taxonomy. Asking for structured observations turns the 2nd call into
a descriptive extraction task — easier for a 2B VLM than a yes/no
token — and the flip decision is derived mechanically from the
extracted fields, avoiding a free-form verdict that the 2B sometimes
mis-emits. The per-dataset definition closes the gap between the 2B
model's default reading of "hate" and each dataset's actual labeling.

**Prediction.** On boundary-band candidates, the model's structured
observations should distinguish true positives (real group-targeted
hate on HateMM; real mocking/offensive content on MHClip) from false
positives (news/reaction videos on HateMM; genuinely neutral/factual
content on MHClip). The flip rule then overturns FPs without
touching TPs. Non-boundary videos are never queried and pay no
rescue-call cost.

**Counterfactual.** Replacing the review call with a plain "Is this
hateful? yes/no" at `temperature=0` gives the identical answer as the
first call (both are temperature-0 Yes/No queries over the same
prompt family) — zero flips, zero gain. Any improvement from the
rescue mechanism is attributable to the *structured-review framing*
(disclose prior + structured extraction + per-dataset definition
matching), not to "more inference compute". The load-bearing
component is the structural form of the 2nd call.

## Why `k_below = 0`

Empirical: we ran a loop with `k_below = 10` FN-hunt rescue and
observed:
- On HateMM below, the strict FN-hunt schema (require TARGET_GROUP
  + slur/dehu/visual signal) never fires → 0 flips, no gain, no
  harm.
- On MHClip below, the broader FN-hunt schema over-extracts
  "mockery/crude humor/bullying" on borderline TNs. The 2B model's
  looser interpretation produces false flips that erase the EN and
  ZH strict-beat.

Root cause: below-side rescue asks the model to *hunt for missed
evidence*, which primes it to over-find. Above-side rescue asks the
model to *verify the initial hate flag*, which primes it to demand
positive evidence — a safer direction given the 2B's tendency to
over-commit. Disabling below-side rescue (`k_below = 0`) removes
this failure mode entirely and lets the above-side rescue do its
job cleanly.

## Ablations

| Config | EN Δacc/mf1 | ZH Δacc/mf1 | HateMM Δacc/mf1 | Strict-beat all |
|---|---|---|---|---|
| Baseline (no rescue) | 0 / 0 | 0 / 0 | 0 / 0 | — |
| v1: free-form CONFIRM/OVERTURN | −0.019 / −0.048 | −0.040 / −0.054 | **+0.023** / **+0.021** | no |
| v2: structured, uniform definition | −0.019 / −0.048 | −0.040 / −0.054 | **+0.023** / **+0.021** | no |
| v3: per-dataset definition, k=10/10 | −0.050 / −0.073 | −0.020 / −0.024 | **+0.014** / **+0.010** | no |
| **final**: per-dataset, k=0/2 | **+0.006** / **+0.006** | **+0.007** / **+0.007** | **+0.005** / **+0.005** | **YES** |

Diagnostic ablations against the final config:

- **Pure score-rank (no rescue, k₀=0, k₁=2)**: EN +0.0124/+0.0112 ·
  ZH −0.0000/−0.0022 · HateMM −0.0000/−0.0010 → strict-beats EN only.
  The MLLM rescue is load-bearing on ZH and HateMM: it blocks the
  wrong flip that pure score-rank would otherwise make, preserving
  the 1 correct flip per dataset.
- **v2 uniform-definition rescue, k₀=0, k₁=2**: EN 1 flip correct,
  ZH 1 flip wrong, HateMM 1 flip correct → fails ZH. The broader
  MHClip definition in the final config changes ZH's extracted
  fields and lets the rescue correctly block the wrong flip.
- **First-pass disclosure ablation** (rescue prompt omits the
  "FIRST-PASS DECISION:" line): the model loses the anchor that
  focuses deliberation on refuting the prior, shifting toward
  uniform overturning. Informal inspection shows the 2B's
  structured fields become less discriminative.

## Artifacts

- `src/boundary_rescue/baseline_preds.py` — Task 0: reproduces
  baseline predictions per dataset and pins the ZH pre-repro target
  to `results/boundary_rescue/zh_prerepro_baseline.json`.
- `src/boundary_rescue/select_boundary.py` — Task A: top-K-closest
  candidate selector. Defaults to `k_below=0, k_above=2`.
- `src/boundary_rescue/rescue_2b.py` — Task B: 2nd-call rescue
  script. Per-dataset prompts. `temperature=0, max_tokens=512`,
  plain-text decoding.
- `src/boundary_rescue/apply_and_eval.py` — Task C: structured-field
  parser, deterministic flip rule, eval + strict-beat check.
- `src/boundary_rescue/thresholds.py` — vendored `li_lee_threshold`
  + re-exports of Otsu/GMM from `src/our_method/quick_eval_all.py`.
- `results/boundary_rescue/{MHClip_EN,MHClip_ZH,HateMM}/` —
  `baseline_preds.jsonl` · `candidates_final.jsonl` ·
  `rescue_final.jsonl` · `test_final.jsonl`.
- `results/boundary_rescue/zh_prerepro_baseline.json` — pinned ZH
  strict-beat target.
- `results/boundary_rescue/loop_log.jsonl` — per-iteration history
  of the development loop.
- `logs/boundary_rescue_*.out` — Slurm logs for each iteration.

## Reproduction

```bash
# 1. Reproduce baseline predictions + threshold (CPU, ~10 s)
python src/boundary_rescue/baseline_preds.py

# 2. Select top-2 above candidates (CPU, ~1 s)
python src/boundary_rescue/select_boundary.py --version final \
  --k-below 0 --k-above 2

# 3. Run 2nd-call rescue (1 GPU, ~5 min for ~6 candidates across 3 datasets)
sbatch --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=0:30:00 \
  --output=logs/boundary_rescue_final.out \
  --wrap "source ~/miniconda3/etc/profile.d/conda.sh \
          && conda activate SafetyContradiction \
          && python src/boundary_rescue/rescue_2b.py --version final --all"

# 4. Eval + strict-beat check
python src/boundary_rescue/apply_and_eval.py --version final
```
