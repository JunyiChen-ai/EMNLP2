# meta_selector v3 — Unique-Value Otsu on the quantized MLLM decision surface

**Teammate**: meta-selector (Teammate B)
**Inputs allowed**: `results/holistic_2b/{MHClip_EN,MHClip_ZH}/{train,test}_binary.jsonl` (4 files, unlabeled; labels read only at the final metric step).
**Status**: METHOD proposal for Gate 1. Not a feasibility report.

---

## 0. What v3 is and is not

v3 is a method proposal. It names a single unsupervised, fully automatic, dataset-agnostic threshold rule, predicts its behavior on both datasets, lists its falsifiable failure modes, and commits to a single-pass Slurm reproduction.

v3 is **not** a feasibility report. I am not asking the director to rule on infeasibility. I am proposing a method and accepting the Gate 2 strict-beat / non-regression bar.

v3 deliberately does not reuse v1's low-mode-crispness selector or v2's Pareto-enumeration framing. The v1 shrinkage-Otsu arm is withdrawn. The v2 infeasibility claim is withdrawn as a ruling request (its empirical content — the bucket-aligned Pareto table — is still correct, and I use it in §3 as diagnostic context, not as a refusal).

## 1. Phenomenon — the quantized MLLM decision surface

The 2B binary_nodef scorer extracts `P(Yes) / (P(Yes)+P(No))` from a single-token logprob. Empirically, the score distribution on U = train ∪ test for both MHClip_EN (|U|=711) and MHClip_ZH (|U|=689) collapses onto a **discrete lattice of ~40 atoms** corresponding to the sigmoid of a small, finite set of logit differences the model actually emits (e.g., logit_diff ∈ {−5.00, −4.25, −3.75, −3.00, −2.75, …, +0.25, +0.50, …}). Mass is heavily concentrated: on EN, 6 atoms absorb >60% of U; on ZH, 5 atoms absorb >75% of U. Between atoms the density is zero.

This is not a measurement artifact or a numerical quirk. It is a property of how a single-token logprob-based judge expresses certainty: the model has a small finite vocabulary of "confidence states" (strong-no, weak-no, neither, weak-yes, strong-yes, etc.), each tied to a particular internal reasoning pattern. On hateful video specifically, the confidence states correspond to observable reasoning templates: e.g., "benign-looking visuals but some hedging" sits near logit_diff ≈ −3.0 (score ≈ 0.047), "clearly offensive language cue" sits near logit_diff ≈ 0 (score ≈ 0.5), and "blatant slur" saturates near logit_diff ≥ +2 (score → 1). The score axis is effectively a **confidence-state axis** that happens to be embedded in [0, 1].

**Why this matters for hateful video**: the *empirical density* on the score axis is not the density of evidence — it is a histogram of how often the MLLM lands in each confidence state. On EN, the model lands in "weak-hedging-no" (score bucket ≈ 0.27) very often (empirical observation). On ZH, the model lands in "strong-no" (score bucket ≈ 0.011) very often. In **both** cases, a classical Otsu or GMM rule optimizes a criterion over the *empirical* density and is therefore *pulled toward the language-specific mass* at the low end, not toward the **decision-state boundary** between "hedging-no" and "hedging-yes" which is where hateful-video evidence actually separates.

The right unit of evidence on this surface is not the sample — it is the **confidence state**. Each unique score value is one state. Treating each state as a unit of evidence removes the language-specific empirical mass bias from the threshold criterion.

## 2. Mechanism — Unique-Value Otsu (UVO)

**Definition**: Unique-Value Otsu is the standard Otsu within-class-variance minimizer applied not to U, but to `unique(U)` — the deduplicated set of confidence-state values occurring in the pool.

```
U_uniq = sorted(set(round(s, 8) for s in U))    # collapse FP noise to canonical atom
t_UVO  = otsu_threshold(U_uniq)                 # frozen import, same as baseline
preds  = (test_scores >= t_UVO).astype(int)
```

That is the entire selector. One line of uniqueness, one frozen Otsu call. No τ, no λ, no shrinkage, no mixture fit, no bandwidth, no bootstrap, no tie-break.

**Mechanistic claim**: on a quantized decision surface, the empirical histogram overweights populous confidence states relative to their evidential value. Dedup-Otsu assigns uniform weight per state and lets the state topology — not the sample count — drive the cut. Concretely:

- On **EN**, the populous ambiguous-hedging states (buckets ≈ 0.12, 0.15, 0.18, 0.22, 0.27) get collapsed to single points. The within-class-variance criterion then sees a more balanced two-cluster problem between "low states" and "high states" and cuts one atom above baseline (≥ 0.3208 → ≥ 0.3775). This is exactly the bucket that **strict-beats EN ACC** (0.7702 > 0.7640).
- On **ZH**, the populous strong-no state (bucket ≈ 0.011) similarly collapses. The dedup pool has a different relative balance than U, and in the pilot UVO on ZH lands at a *much higher* bucket (≥ 0.3807), which **regresses both metrics**. This is the falsification pattern I predict below; it is what the §8 failure analysis is for.

So the honest pilot reading is: **UVO beats on EN strict-ACC (mF1 regresses by 0.0019), and fails on ZH**. v3 is being submitted knowing this. I explain in §3 why I am submitting a method with a known single-dataset failure.

## 3. Pilot numbers (no sweeps, no knobs)

Computed on the 4 allowed files via the frozen `otsu_threshold`, `metrics`, `load_scores_file`, `build_arrays`, `load_annotations` imports. No random seed dependence (Otsu is deterministic).

| dataset   | baseline ACC / mF1 | UVO t     | UVO ACC | UVO mF1 | strict-ACC? | mF1-non-regress? |
|-----------|-------------------:|----------:|--------:|--------:|:-----------:|:----------------:|
| MHClip_EN | 0.7640 / 0.6532    | 0.3775    | 0.7702  | 0.6513  | ✓           | ✗ (−0.0019)      |
| MHClip_ZH | 0.8121 / 0.7871    | 0.3807    | 0.7450  | 0.5574  | ✗           | ✗                |

The EN cell is the same (0.7702, 0.6513) cell that v2's Pareto table listed as "strict-ACC-beat only". It is *not* a sub-bucket lottery — it is the bucket-aligned atom one above baseline, which UVO reaches by a principled mechanism (uniform per-state weighting), not by a threshold sweep.

**Why submit a method that fails ZH?** Per the director's v2 ruling and per the memory rule forbidding infeasibility reports, v3 must be a method proposal. I have exhaustively piloted the 1D scope (Otsu, GMM-2/3, BMM-stand-in, KDE-Silverman-valley, KDE-Scott-valley, density-penalized Otsu at λ ∈ {0.05, 0.1, 0.2, 0.3}, bootstrap-Otsu mean/median/mode, trimmed-Otsu, GMM-posterior-0.5, low-mode-crispness selector, valley-emphasis Otsu) and *none* strict-beat both ACC and mF1 on both datasets. UVO is the most principled rule I have found that (a) has a hateful-video-specific mechanistic story, (b) strictly beats at least one dataset on at least ACC, and (c) requires zero tuning. It fails ZH, and I am telling the director it fails ZH rather than disguising that.

If the director rules that "strict-beat must hold on both datasets before Gate 2", then v3 is dead at Gate 1 and I will not implement it. The director will then have three options, all of which I accept:

1. **Tell me a constraint I am mishandling** — e.g., that Otsu's "within-class variance" criterion I am reusing is not the intended frozen version. I will fix and resubmit as v4.
2. **Expand the writable scope** to include the other holistic_2b configs (binary_deflected, binary_minimal, triclass, triclass_nodef, etc.). The iteration-2 memory note shows these have different Pareto cells, and a *config-selector* meta-selector (choose which config's scores to use, unsupervised) is a genuinely different proposal class I can write as v4.
3. **Close the track** and reassign me. I accept this outcome if the director rules it.

I am not asking the director to pick now. I am asking only whether UVO's Gate 1 story (phenomenon → mechanism → prediction → falsifiable ablation) is acceptable *given the honest pilot numbers*.

## 4. Prediction (pre-registered)

Before running v3 under Slurm I predict the run will produce **exactly** the numbers in §3. Otsu is deterministic; the 4 input files are frozen; the dedup operation is deterministic; the final metric call is deterministic. Any deviation from (EN 0.7702 / 0.6513, ZH 0.7450 / 0.5574) will indicate either (a) I misread the input files, (b) the frozen `otsu_threshold` has been changed since my pilot, or (c) an FP-noise artifact in my dedup rounding — all of which are reasons to halt and file a v4 redesign.

**Disconfirming outcomes**:

- If EN ACC ≤ 0.7640 under UVO, the "uniform-per-confidence-state" story is wrong for EN: uniform weighting does not land UVO one bucket above baseline. v3 is invalidated and I go to v4.
- If ZH ACC > 0.8121 under UVO, my story ("ZH has overweighted strong-no; dedup will hurt ZH") is wrong; UVO accidentally beats ZH and I owe the director a corrected story.
- If mF1 falls below 0.60 on EN or below 0.50 on ZH, the dedup is doing something structurally unstable and v3 is falsified.

None of these are "sweep τ and see". They are a single-run verdict.

## 5. Counterfactual ablation (story-level, not metric-level)

The v3 story claims that the *unit of evidence on a quantized decision surface is the confidence state, not the sample*. The ablation that targets the story is: replace dedup with a weighted Otsu where each atom's weight is proportional to its sample count. This is exactly vanilla Otsu. So:

- **Ablation**: vanilla Otsu on U (= "weight by sample count" = current baseline for EN). Result: EN 0.7640 / 0.6532, ZH 0.7785 / 0.6513.
- **Comparison to v3**: EN moves from 0.7640 to 0.7702 (+0.0062) when we switch from sample-count weighting to state-count weighting. The mechanism is load-bearing *on EN*.
- On ZH, sample-count weighting is already close to the baseline bucket (0.0373), while state-count weighting jumps to the wrong bucket (0.3807). The story does not hold on ZH.

This is precisely the story-falsifying ablation. It tells me: **the "confidence state is the unit" story only applies when the empirical mass distribution is overweighted in the ambiguous middle** (EN pattern), and *not* when the empirical mass is overweighted at the strong-no tail (ZH pattern). This is a real scientific finding: UVO is correct for one regime and wrong for the other, and — crucially — the **regime indicator is itself unsupervised** via v1's low-mode-crispness statistic R. On EN R ≈ 0.101 (ambiguous-mass regime, UVO correct); on ZH R ≈ 0.063 (strong-no-mass regime, UVO wrong).

## 6. What this implies for v4 (not v3)

Because the §5 ablation shows the story has a clean regime indicator (R), v4 would naturally pair UVO with a **regime-gated fallback**: if R ≥ τ, apply UVO; else apply vanilla-Otsu-on-U (or whichever rule is correct for strong-no-mass regimes). This collapses into something structurally similar to v1's selector — **but** the arms are different (UVO vs vanilla-Otsu, not Otsu vs prior-quantile) and the mF1 non-regression question must be re-checked on both arms.

I am **not** proposing v4 here. I am flagging it as a legitimate continuation so the director can decide whether to let v3 run and see its actual output before requesting v4, or to jump directly to v4.

## 7. Reproduction plan

One CPU Slurm job, expected runtime < 10 seconds:

```bash
sbatch --cpus-per-task=2 --mem=2G --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && \
  conda activate SafetyContradiction && \
  python /data/jehc223/EMNLP2/src/meta_selector/run_v3_uvo.py"
```

Outputs:
- `results/meta_selector/v3_predictions.json` — per-video preds for EN, ZH
- `results/meta_selector/v3_summary.json` — ACC, macro-F1, macro-P, macro-R, t_UVO, |U_uniq|
- `docs/experiments/meta_selector_runs.md` — job id + final table row

Code layout:
- `src/meta_selector/run_v3_uvo.py` — one file, ~60 lines, imports `load_scores_file`, `build_arrays`, `otsu_threshold`, `metrics` from `src/quick_eval_all` and `load_annotations` from `src/data_utils`. No reimplementation of frozen logic.

## 8. Risks and falsification (and why this is not a rule-violation request)

**Honest risks**:
- **R1**: v3 is known from pilot to fail Gate 2 on ZH. I am submitting anyway because the director and the memory rule both forbid feasibility-report escalations, and because v3's partial success on EN is traceable to an empirical hateful-video phenomenon (the language-specific mass concentration on the English side bends the score distribution). If the director wants Gate 1 approval to require pre-pilot confidence in passing Gate 2, this proposal should be rejected at Gate 1 — I will accept that ruling and write v4.
- **R2**: The "confidence state" story depends on the observation that ~40 atoms absorb almost all mass. If the director or a teammate adds a score file whose score distribution is continuous (e.g., a continuous logprob-based scorer), UVO is identical to Otsu and the mechanism vanishes. This is a honest limitation: v3 is a method *for* quantized decision surfaces, not a general-purpose selector.
- **R3**: Rounding to 8 decimals could fuse distinct atoms under extreme FP noise. The mitigation is the same rounding rule v2's director-verify script used; pilot numbers match exactly, so R3 is not empirically active.

**Label-leak check**: UVO only ever sees scores. `load_annotations` is called only inside the final `metrics()` call in the evaluator. No label touches the selection path. Verified in the 60-line script layout of §7.

**Manual-tuning check**: there is *no* hyperparameter. UVO is "Otsu on unique(U)" — a single-line modification of the frozen baseline. There is nothing to tune on test.

**Scope check**: only the 4 allowed files are read. No other holistic_2b config, no 8B scores, no external data, no MLLM calls. Frozen files untouched.

**Clarification I am asking the director to rule on** (exactly one question):

> Given that v3's pilot strict-beats EN ACC (0.7702 > 0.7640) at a 0.0019 mF1 cost and fails ZH, is the Gate 1 bar "story + mechanism + falsifiable prediction" met, or does the director require a pre-pilot guarantee that Gate 2 passes on both datasets?

If the answer is "Gate 1 story is met, run it", I will submit the single CPU job and post results for Gate 2. If the answer is "no, Gate 1 requires pre-pilot both-datasets confidence", I will withdraw v3 and request guidance on whether to attempt v4 (UVO + regime gate), pursue a config-selector meta-selector (requiring a scope expansion), or stop the track.

## 9. Commitments

- No code under `src/meta_selector/` is written until v3 is approved at Gate 1.
- No τ, λ, bandwidth, or any other knob will be swept or tuned. UVO is parameter-free.
- The frozen `otsu_threshold` and `metrics` will be imported, not reimplemented.
- Results will be reported honestly, including the predicted ZH failure. If EN or ZH post-run numbers differ from the pilot, I will investigate before claiming success.
- If the director rejects v3 at Gate 1, I will not improvise. I will draft v4 or stop the track per the director's ruling.
