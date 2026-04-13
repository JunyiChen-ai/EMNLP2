# Label-free selector scanfold probe — 2026-04-13

## Question

Given a pool of K classical unsupervised thresholding methods, does there
exist a **pre-registered, non-self-referential, label-free criterion** C
such that `argmax_{method} C(method's threshold, pool)` equals the
**labeled-best** method on every tested dataset?

A positive result would give the current unified baseline `(EN→Otsu,
ZH→GMM)` a genuine label-free cover story instead of its current
"peeked at labels to pair methods" status. A negative result kills that
line of defense and becomes a publishable structural finding.

## Pre-registration (frozen BEFORE any results were computed)

### Method pool (K = 10)

`otsu`, `gmm` (K=2), `met` (Kittler-Illingworth), `triangle` (Zack), `kapur`
(max-entropy), `li_lee` (min cross-entropy), `yen` (max correlation),
`rosin` (unimodal), `renyi` (α=0.5), `median`.

### Criterion pool (6, non-self-referential)

1. `silhouette` — sklearn silhouette coefficient on 1D binary partition
2. `neg_davies_bouldin` — sklearn Davies-Bouldin, negated so higher=better
3. `dunn` — min inter-cluster distance / max intra-cluster diameter
4. `gap_statistic` — Tibshirani 2001 gap vs uniform-null
5. `kde_valley_depth` — `1 - KDE(t) / max(KDE)` where KDE is Gaussian pool
6. `balance_penalty` — `-|log(w1/w0)|`, discourages degenerate splits

### Excluded criteria (self-referential — would trivially pick their own method)

- Calinski-Harabasz (≡ Otsu's between/within variance ratio)
- 2-Gaussian BIC/log-likelihood (trivially picks GMM/MET)
- Otsu criterion value, MET J(t), Kapur/Li-Lee/Yen entropies

### Passing rule

A criterion **passes** iff its `argmax_{method}` equals the labeled-best
method on **every** dataset. No post-hoc criterion tweaks. No winner
cherry-picking.

### Datasets

- **MHClip_EN** — 2B binary_nodef test scores, n=161
- **MHClip_ZH** — 2B binary_nodef test scores, n=149
- **HateMM** — 2B binary_nodef test scores, n=215 (YouTube rule set, scored
  this session via job 8229 after the three additive edits to
  `src/score_holistic_2b.py` for HateMM support)

## Results (full 3-dataset run)

### Labeled-best method per dataset (diagnostic only — selectors do not see labels)

| Dataset | Labeled-best | ACC | mF1 | threshold |
|---|---|---|---|---|
| MHClip_EN | **otsu** | 0.7640 | 0.6532 | 0.2705 |
| MHClip_ZH | **gmm** | 0.8121 | 0.7871 | 0.0362 |
| HateMM | **li_lee** | 0.8047 | 0.7930 | 0.2410 |

**Key observation**: the three datasets have **three different** labeled-best
methods from the pool. This is the first structural finding of the probe —
the baseline's `(EN→Otsu, ZH→GMM)` pairing is not just a label-peek between
two options, it's a label-peek among ten.

### Criterion × dataset → argmax method

| criterion | EN pick | ZH pick | HateMM pick | EN ✓ | ZH ✓ | HateMM ✓ | PASS? |
|---|---|---|---|---|---|---|---|
| silhouette | otsu | otsu | otsu | ✓ | ✗ | ✗ | fail |
| neg_davies_bouldin | otsu | otsu | kapur | ✓ | ✗ | ✗ | fail |
| dunn | otsu | otsu | otsu | ✓ | ✗ | ✗ | fail |
| gap_statistic | otsu | otsu | otsu | ✓ | ✗ | ✗ | fail |
| kde_valley_depth | otsu | otsu | yen | ✓ | ✗ | ✗ | fail |
| balance_penalty | median | median | median | ✗ | ✗ | ✗ | fail |

**0 of 6 criteria pass.**

### Why the criteria fail — structural interpretation

The 5 non-degenerate geometric criteria (silhouette, DB, Dunn, gap, KDE-valley)
**unanimously prefer Otsu** on all three datasets with one near-exception
(DB picks Kapur on HateMM by a razor-thin margin; kde_valley_depth picks Yen on
HateMM). That is, **Otsu's threshold sits in the cleanest density valley**
on every dataset — it cuts where the data geometrically looks like it
should be cut.

But:

- On **MHClip_ZH**, the labeled winner is GMM at t=0.036, which sits almost
  directly on top of the negative mode. Every geometric metric ranks this
  partition as *terrible* (silhouette 0.55 vs Otsu's 0.84, KDE-valley 0.04
  vs Otsu's 0.93). Yet GMM **beats** Otsu by +5.4pp ACC. The reason: many
  ZH positives have scores just barely above zero. The 2B model is
  systematically under-confident on Chinese input; a low threshold captures
  those low-confidence positives, a geometrically-clean threshold throws
  them out.

- On **HateMM**, the labeled winner is Li-Lee at t=0.241, a mid-range
  cross-entropy threshold. Otsu sits higher at 0.380 and is slightly
  "too aggressive" (misses 2-3 borderline positives around 0.24-0.38).
  Li-Lee wins by a small +1.4pp ACC margin — essentially noise at n=215,
  but stable enough that it is the labeled argmax.

- On **MHClip_EN**, Otsu is both the geometrically-cleanest AND the
  labeled-best. The label and the geometry agree.

### The structural conclusion

**The labeled-best threshold method is an artifact of dataset-specific
MLLM calibration quirks**, not a property of the score distribution's
geometric structure. No cluster-quality-based criterion can recover it.

Concretely:
1. **EN** is the "well-behaved" case where geometry and labels agree.
2. **ZH** is a systematic model-underconfidence case: the correct threshold
   is *pathologically* close to the negative mode because the 2B model
   crushes Chinese hateful content's Yes-probabilities toward zero. This is
   a calibration issue, not a separation issue.
3. **HateMM** is a noise-dominated small-sample case where li_lee happens
   to land 2-3 videos better than otsu. Not replicable as a rule.

A label-free selector grounded in score geometry **must** pick Otsu on all
three (because that is what geometry says), and that commits it to a
regression of ~5.4pp on ZH relative to the current label-peeked baseline.

## Implications

### For the current baseline's label-free claim

**The current `(EN→Otsu, ZH→GMM)` baseline is genuinely label-selected.**
No criterion in the tested family recovers it; the reason is not "we tested
the wrong criteria" but "ZH's labeled-best isn't geometric". A fair
label-free baseline would have committed to a single method in advance —
silhouette-optimal (Otsu) is the natural commitment — and would have
numbers of:
- EN: 0.7640 / 0.6532 (same as current)
- ZH: 0.7584 / 0.6042 (−5.37pp ACC, −18.3pp mF1 vs current)
- HateMM: 0.7907 / 0.7674

This is the **honest label-free baseline**, and it is 5.4pp worse on ZH
than the currently reported baseline.

### For publishable framing

Two possible framings:

1. **"The label-free threshold problem is harder than it looks"** — a
   structural finding that no geometry-based selector can recover the
   oracle pairing on MHClip, and the reason is MLLM calibration
   asymmetry between languages, not threshold-method inadequacy. This is
   a short, self-contained negative result. Publishable as a workshop
   note or as a cautionary section in a larger paper.

2. **"The honest label-free baseline"** — redefine the project's baseline
   as silhouette-selected Otsu and report 0.7640 / 0.7584 / 0.7907. The
   "method-beats-baseline" bar drops on EN (unchanged), drops on ZH (to
   0.7584, which several prompt_paradigm ablation cells already beat),
   and becomes attainable on HateMM. This is a stronger paper direction
   but requires re-running the ablation matrix against the honest
   baseline.

### What is NOT killed by this result

- ~~Readout-perturbation exploration (v3 p_evidence had ZH +0.67pp over
  current baseline).~~ **CORRECTED twice (2026-04-13 later session)**:
  v3 p_evidence was not a readout perturbation at all. It was computed on a
  **shifted prompt** — `polarity_calibration.py:83` has the sentence
  `"You are a content moderation analyst."` removed from the user message
  (baseline keeps that sentence). 78% of EN and 73% of ZH videos differ
  from baseline (max |Δ|=0.2151). Under classical methods on the shifted
  prompt, all 4 TR/TF × Otsu/GMM cells are below baseline. No empirical
  basis for "readout perturbation softens ZH".
- Input-side prompt reformulations (v6 Coarse Axes, not fully verified).
- Iterative/self-training atom-level flag methods (nothing in this probe
  addresses atom-level non-monotone assignment).

What IS killed: the hope that a classical cluster-quality criterion can
back-justify the current unified baseline as label-free.

## Artifacts

- `src/probe_selector_scanfold.py` — the probe (self-contained, CPU, ~30s)
- `results/analysis/probe_selector_scanfold.json` — full result dict
- `logs/probe_scanfold_v2.out` — stdout from final 3-dataset run
- `results/holistic_2b/HateMM/test_binary.jsonl` — new HateMM baseline
  scores (job 8229, 2B binary_nodef, YouTube rules)
- `logs/score_hatemm_test.out` — HateMM scoring log

## Pre-registration audit

- Method pool: pre-registered at plan-write time, unchanged.
- Criterion pool: pre-registered at plan-write time, unchanged.
- Self-referential exclusion list: pre-registered, unchanged.
- Datasets: pre-registered as EN, ZH, HateMM (with HateMM scored this
  session via the minimal edits authorized in the plan).
- Passing rule: pre-registered (all-datasets-match), unchanged.
- No post-hoc criterion was added after results were seen. No criterion
  was removed. The probe ran twice (once at 20-video partial HateMM, once
  at full HateMM); both runs used the identical frozen criterion set.

Verdict was **negative on both runs**, consistent across data availability.
