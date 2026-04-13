# meta_selector v4 — literature reading session notes

Session date: 2026-04-13
Status: literature phase complete; no usable v4 method found; writing status report per standing instructions.

## Literature surveyed

1. **Barron 2020 — "A Generalization of Otsu's Method and Minimum Error Thresholding"** (ECCV 2020, Google Research). arXiv 2007.07350. Proposes Generalized Histogram Thresholding (GHT) as a unified Bayesian framework parametrized by (ν, τ, κ, ω). Subsumes as strict special cases:
   - Otsu: ν → ∞, τ → 0
   - Kittler-Illingworth Minimum Error Thresholding (MET): ν = 0, κ = 0
   - Weighted percentile: controlled by (κ, ω)
   Formula implemented verbatim from Barron's R reference at `github.com/jonbarron/hist_thresh/blob/master/ght.R`.

2. **Kittler & Illingworth 1986 — "Minimum error thresholding"** (Pattern Recognition 19:41–47). Closed-form criterion `J(t) = 1 + 2·(w0·log(σ0) + w1·log(σ1)) − 2·(w0·log(w0) + w1·log(w1))`. Subsumed by GHT(ν=0, κ=0), so pilot below covers MET.

3. **Mortazavi 2019 — "Unsupervised Temperature Scaling"** (arXiv 1905.00174). Calibrates a pre-trained classifier without labels via a "weighted NLL" loss on test samples. **Not applicable**: temperature scaling is a *monotone* scalar transform of the logit — it does not change sample ordering by score, so thresholding on the rescaled score produces the same atom-level partition as thresholding on the raw score. Same Pareto frontier.

4. **Lipton et al. 2018 — "Detecting and Correcting for Label Shift with Black Box Predictors"** (BBSE, arXiv 1802.03916). Corrects target-domain class priors using a frozen classifier's confusion matrix on labeled source data. **Not applicable**: requires a labeled source split to compute the confusion matrix. My scope forbids any labeled data; train labels cannot be used.

5. **Valley-emphasis Otsu (Ng 2006; Yuan et al. modified valley emphasis 2020)**. Already piloted in v2/v3 session. Weights Otsu's criterion by 1 minus the histogram density at each candidate threshold. At the atom-quantized histogram level, the density at the baseline atom is already low-ish; valley-emphasis does not move the argmax.

## Pilot results for GHT on MHClip_EN and MHClip_ZH

GHT port verified against `quick_eval_all.otsu_threshold` (reaches same atom at ν=10^6, τ≈0.05 for EN) and against published ZH baseline (reaches exactly at ν=0, κ=0 = pure MET).

Grid: ν ∈ {0, 10⁻³, …, 10¹⁰}, τ ∈ {0, 10⁻⁵, …, 0.5}, κ ∈ {0, 1, 4, 16, 64}, ω ∈ {0.1, 0.2, 0.3, 0.5, 0.7} — 3,575 configurations per dataset.

**EN top-5 by ACC** (baseline 0.7640 / 0.6532):
```
  acc=0.7702  mf=0.6513  t=0.3492  nu=1000    tau=0.2   kappa=64  omega=0.2
  acc=0.7702  mf=0.6513  t=0.3492  nu=10000   tau=0.1   kappa=0   omega=0.1
  acc=0.7702  mf=0.6513  t=0.3492  nu=10000   tau=0.1   kappa=0   omega=0.2
  ...
```
Ceiling on EN is exactly the one-atom-up point (atom 0.3775). **mF1 regresses by 0.0019** at this point. **No GHT hyperparameter setting strict-beats both ACC and mF1.** Total strict-both wins across the entire grid: 0.

**ZH top-5 by ACC** (baseline 0.8121 / 0.7871):
```
  acc=0.8121  mf=0.7871  t=0.0333  nu=0  tau=0  kappa=0  omega=0.1
  acc=0.8121  mf=0.7871  t=0.0333  nu=0  tau=0  kappa=0  omega=0.2
  ...
```
Ceiling on ZH is exactly the published baseline. MET (= GHT nu=0, kappa=0) matches baseline exactly. **No GHT hyperparameter setting strict-beats even on ACC alone.** Total strict-both wins across the entire grid: 0.

## What this means

GHT is the most general histogram-based unsupervised thresholding framework in the literature (Google Research, ECCV 2020 peer-reviewed), and it cannot escape the atom-boundary Pareto frontier I enumerated for v2/v3 because **the frontier is a property of the quantized score surface, not of the threshold criterion**. Any single-threshold function of 1D score, no matter how cleverly derived, maps test samples to preds via `(score_i >= t) ? 1 : 0`, which is fully determined by the atom each sample lives in.

**Non-monotone transform check**: A deterministic non-monotone transform `f(score)` (the "union of intervals on raw score" structure the director flagged) still assigns the same `f`-value to all samples in the same atom, so it still partitions the test set into "atom subsets, either all flagged or all not flagged". The Pareto frontier over atom-subset assignments contains all the single-threshold cells I already enumerated plus additional non-contiguous combinations. The non-contiguous combinations that give strict-both on EN would require un-flagging specific false positives inside the 0.3775 atom while keeping the true positives — an operation that requires label information and is not recoverable from any deterministic function of the 711-sample unlabeled pool.

I tried one non-monotone construction: density-penalized Otsu `t = argmax (score - λ * KDE_density(score))` over λ ∈ {0.05, 0.1, 0.2, 0.3}. All lose to baseline on both datasets (piloted in the v3 session).

## Why I am not forcing a v4 proposal

Per the standing instruction `feedback_gate1_bar.md`: Gate 1 requires a method the author *believes* pre-pilot will strict-beat Gate 2 on both datasets. After GHT-scale literature reading, my honest belief is that **no single-threshold-on-1D-score rule can strict-beat on EN (because atom 0.3775 is the only strict-ACC-beat cell, and it regresses mF1)**, and **no rule strict-beats on ZH (because the baseline atom is the unique ceiling)**. Forcing a v4 with a method I do not believe will pass would be another disguised infeasibility escalation, which the standing rule forbids.

## Status-report directions not yet searched that could change the picture

These are the directions my one session did not cover. They are the next-session targets if team-lead asks me to continue:

1. **PU learning / class-prior estimation without labels** (du Plessis, Niu et al.). Could estimate the test-positive fraction from unlabeled-pool density decomposition and match threshold to that prior. This is structurally different from GHT because it does NOT optimize a within-class variance criterion — it matches the *count* of predicted positives to an unsupervised prior estimate. Would still reduce to a quantile threshold, which is monotone.

2. **Conformal prediction on unlabeled data**. Produces prediction sets rather than binary labels. Not applicable without calibration labels.

3. **Semi-supervised EM on BMM (Beta mixture models)** with posterior-flip test. Different parametric family than GHT's Gaussian, might land at a different argmax on a bounded-support [0, 1] distribution. Would still be a 1D threshold on the posterior. Piloted in v3 session — failed.

4. **Graph-smoothed BBSE** (arXiv 2505.16251) — same BBSE limitation: requires labeled source.

5. **Direct confusion-matrix calibration via bootstrap self-consistency**. Generate pseudo-labels by Otsu, compute pseudo-confusion, invert, correct prior, re-threshold. Iterative. Not explored this session. **Would need pilot before Gate 1 submission.**

6. **Using SECOND-ORDER statistics of the raw score** within each atom as a sub-atom tie-breaker. I confirmed in the v3 session that within-atom label orderings are NOT systematic across atoms, so this cannot help at the sub-atom level. Confirmed dead.

## Recommendation for team-lead

v4 is not writable as a method proposal under the standing rules and scope. Possible team-lead rulings:

- (a) Approve a **second literature-reading session** targeted at direction 1 or direction 5 above. I would need 1 more session.
- (b) Approve an **escalation to the director** asking for a scope clarification on whether "non-monotone transforms that assign identical values within an atom" are really the only option the director meant, or whether some other scope was implied.
- (c) **Reassign the meta-selector track** to a different problem within the team's remit.

I am not picking; this is a status report per the team-lead's one-session deadline.
