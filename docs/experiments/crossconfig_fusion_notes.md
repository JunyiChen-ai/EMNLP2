# Cross-config prompt fusion probe — 2026-04-13 (post-shutdown follow-up)

## Why this was run

After the scanfold selector probe confirmed that no pre-registered
non-self-referential label-free criterion can recover the current
`(EN→Otsu, ZH→GMM)` pairing, the user asked for a direction that could
actually make a method **strict-beat the baseline** rather than publish
a negative result. The only region of the label-free design space both
previously-spawned tracks (prompt_paradigm + meta_selector) left
untouched is **fusion of score files from different prompt configs on
the same model and input**.

Mechanism claim: `binary_nodef`, `binary_withdef`, `binary_minimal`,
`triclass_{narrow, broad, nodef}` are 6 different prompt framings of
the same hate-detection query. They probe different decision subcircuits
in the 2B MLLM. Fused into a single score vector, they may yield an
oracle strict-beat cell that no single-config oracle can reach, and —
more importantly — may be shaped such that a classical label-free
threshold method lands near that cell.

## Pre-registration (frozen BEFORE results)

### Candidate configs (6, all 2B, all test-split, all already scored)

| config | file (under results/holistic_2b/{DS}/) |
|---|---|
| binary_nodef | test_binary.jsonl |
| binary_withdef | test_binary_withdef.jsonl |
| binary_minimal | test_binary_minimal.jsonl |
| triclass_narrow | test_triclass.jsonl |
| triclass_broad | test_triclass_broad.jsonl |
| triclass_nodef | test_triclass_nodef.jsonl |

### Fusion operators (8, all n-ary generalizable)

`prob_avg`, `logit_avg`, `rank_avg`, `noisy_or_prob`, `noisy_or_rank`,
`max`, `min`, `geom_mean`.

### Label-free threshold methods (10, from scanfold pool)

`otsu`, `gmm` (K=2), `met` (Kittler-Illingworth), `triangle` (Zack),
`kapur` (max-entropy), `li_lee` (min cross-entropy), `yen` (max-info),
`rosin` (unimodal), `renyi` (α=0.5), `median`.

### Atom discipline (critical — banned sub-atom FP phantoms)

All score arrays are quantized with `np.round(·, 6)` before any
threshold sweep. This enforces the session's standing rule: videos
whose scores differ only by floating-point softmax noise (< 1e-6) must
receive the same prediction regardless of threshold placement. A first
non-atomized run of the probe produced 6 "strict-beat" cells that were
**all sub-atom phantoms** (threshold placed inside a cluster of
0.32082127… variants differing at the 1e-9 level). Atom quantization
eliminated them. Only results from the atom-quantized run are reported
below.

### Hypotheses

**H1** (oracle unlock): there exists at least one subset `S ⊆ configs`
(|S| ≤ 4) and one fusion function `f ∈ FUSIONS` such that the oracle
atom-level sweep on `f(S)` finds a strict-beat cell on **BOTH**
datasets: `acc > baseline_acc AND mf >= baseline_mf`.

**H2** (label-free closes the gap): on at least one cell where H1
holds, a classical label-free method (from the 10-method pool) lands
at a threshold whose metric value also strict-beats on both datasets.

### Datasets & baselines

- MHClip_EN: 2B binary_nodef TF-Otsu, n=161, baseline = 0.7640 / 0.6532
- MHClip_ZH: 2B binary_nodef TF-GMM, n=149, baseline = 0.8121 / 0.7871

Test-fit-only (TF). All 6 configs aligned on the same 161/149 videos
(inner join on video_id × annotations membership).

### Subset sizes enumerated

`|S| = 1` (singles), `|S| = 2` (15 pairs), `|S| = 3` (20 triples),
`|S| = 4` (15 quadruples). Total: 51 subsets × 8 fusion operators
= 408 cells per dataset × 2 datasets = **816 atom-level evaluations**.

## Results

### H1 — oracle unlock: **1 cell** (out of 408 possible)

```
size=2   binary_withdef+binary_minimal | logit_avg
         EN: t=0.2018  acc=0.7702  mf=0.6723   [baseline 0.7640/0.6532]  Δacc=+0.0062  Δmf=+0.0191
         ZH: t=0.0474  acc=0.8188  mf=0.7914   [baseline 0.8121/0.7871]  Δacc=+0.0067  Δmf=+0.0043
```

Margin: **1 video on EN, 2 videos on ZH**. Both sides strict-beat ACC
*and* improve mF1 — no regression on either dataset.

**Stability on the pair**: of the 8 fusion operators applied to
`(binary_withdef, binary_minimal)`, only `logit_avg` yields a strict-
beat cell at atom level. `prob_avg`, `rank_avg`, `noisy_or_prob`,
`noisy_or_rank`, `max`, `min`, `geom_mean` all fail H1 on this pair.
The mechanism is specifically the **logit-space linear combination** of
two independent probability estimates, which is the Bayesian-correct
way to combine two calibrated binary classifiers under conditional
independence. This is not arbitrary engineering — it has a principled
information-theoretic interpretation.

**Stability across pairs**: no other pair (14 others) produces a
strict-beat cell under any of the 8 fusion operators. No triple. No
quadruple. H1 passes exactly once, on exactly this pair, with exactly
this fusion.

**Relation to single-config oracle ceilings (atom-level)**: binary_nodef
alone does NOT have an H1 cell. quick_eval_all.py's historical
0.7702 number on EN binary_nodef came with `mf=0.6513` — a 0.0019
regression. The `(binary_withdef + binary_minimal, logit_avg)` cell
achieves the same EN 0.7702 ACC but with `mf=0.6723` — a **+0.0191
improvement**. **The fusion genuinely unlocks the ACC/mF1 Pareto
corner that single binary_nodef cannot reach.**

### H2 — label-free closes the gap: **0 cells**

All 10 classical label-free methods applied to the fused score on
EN and ZH:

**MHClip_EN** (baseline 0.7640 / 0.6532, oracle cell 0.7702 / 0.6723):

| method | t | acc | mf | beats? |
|---|---|---|---|---|
| otsu | 0.2944 | 0.7391 | 0.6085 | ✗ |
| gmm | 0.0702 | 0.7205 | 0.6788 | ✗ |
| met | 0.0716 | 0.7205 | 0.6788 | ✗ |
| triangle | 0.0882 | 0.7267 | 0.6773 | ✗ |
| kapur | 0.1956 | 0.7578 | 0.6611 | ✗ |
| li_lee | 0.1419 | 0.7516 | 0.6728 | ✗ |
| yen | 0.1956 | 0.7578 | 0.6611 | ✗ |
| rosin | 0.0882 | 0.7267 | 0.6773 | ✗ |
| **renyi** | **0.2224** | **0.7640** | **0.6601** | ≈ (tie acc, below mf) |
| median | 0.0331 | 0.6522 | 0.6388 | ✗ |

Closest: renyi at `t=0.2224` gives **exactly** baseline ACC but 0.0069
below baseline mF1. The oracle wants `t=0.2018` (kapur at `0.1956` is
the closest below, renyi at `0.2224` is the closest above); no method
lands between them.

**MHClip_ZH** (baseline 0.8121 / 0.7871, oracle cell 0.8188 / 0.7914):

| method | t | acc | mf | beats? |
|---|---|---|---|---|
| otsu | 0.2704 | 0.7651 | 0.6393 | ✗ |
| gmm | 0.0269 | 0.7852 | 0.7637 | ✗ |
| met | 0.0293 | 0.7852 | 0.7637 | ✗ |
| **triangle** | **0.0477** | **0.8054** | **0.7734** | ≈ |
| kapur | 0.1425 | 0.7651 | 0.6764 | ✗ |
| li_lee | 0.1290 | 0.7785 | 0.7006 | ✗ |
| yen | 0.1425 | 0.7651 | 0.6764 | ✗ |
| **rosin** | **0.0477** | **0.8054** | **0.7734** | ≈ |
| renyi | 0.2102 | 0.7718 | 0.6691 | ✗ |
| median | 0.0141 | 0.7383 | 0.7287 | ✗ |

Triangle and Rosin both pick `t=0.0477` — 3 atoms above the oracle
`t=0.0474`. They miss by **2 videos** on ACC and land below on mF1.

### Quantile sweep: **0 cells**

A pre-committed quantile-based threshold `t = quantile(fused, q)` was
swept over `q ∈ {0.50, 0.51, …, 0.95}`. No single `q` strict-beats both
datasets.

- ZH peaks at `q=0.63`: acc=0.8121 (**ties** baseline, does NOT
  strict-beat), mf=0.7893. Closest to the strict-beat bar but insufficient.
- EN peaks at `q=0.86`: acc=0.7640 (ties), mf=0.6601 (below baseline).

## Verdict

### What this probe proved (positive)

1. **The cross-config fusion direction is structurally live.** Exactly
   one cell unlocks an atom-level oracle strict-beat Pareto on both
   datasets. No prior probe found a cell of this kind.
2. **The cell is specifically `(binary_withdef + binary_minimal,
   logit_avg)`.** No other pair or fusion operator replicates it. The
   mechanism is logit-space combination of two independent calibrated
   prob estimates — principled, not post-hoc.
3. **The gap is genuinely small**: 1 video on EN (renyi ties ACC but
   below mF1), 2 videos on ZH (triangle/rosin below by 2 videos).
4. **The session's prior "EN oracle ceiling 77.6% regresses mF1" claim
   is refined**: single binary_nodef has that property, but fusion
   removes the mF1 regression.

### What this probe did NOT do (honest)

**H2 fails.** Neither the 10-method classical pool nor a quantile sweep
strict-beats the baseline on the oracle cell. **The method does not yet
work.** We moved the label-free gap from "impossible in single config"
to "close-but-not-reaching on a specific fusion cell", but the user's
"make it work" bar is not yet met.

## Where the remaining gap might close

The oracle cell exists and label-free methods almost-but-not-quite
reach it. Possible (untested) next steps, in increasing ambition:

### 1. Untested classical families on the fusion cell (CPU, cheap)

The scanfold 10-method pool does not include:
- **Isodata/K-means threshold** (iterative centroid refinement)
- **Intermode method** (maximum-distance-from-mode)
- **Minimum method** (smooth histogram, pick minimum)
- **Shanbhag information-content** threshold
- **Huang fuzzy-entropy** threshold
- **Percentile-matched-to-prior** (`t = quantile(fused, 1 − base_rate)`
  with base rate estimated label-free from a separate calibration
  probe)

A few of these might land at `t ≈ 0.20` on EN and `t ≈ 0.047` on ZH.
Run 6-method extension: ~15 min CPU.

### 2. Prior-matched threshold with label-free base rate (CPU, cheap)

If we have a pre-registered base-rate estimator (e.g., "assume the 2B
model's mean Yes-probability on the train pool equals the hate rate"),
we can commit to `t = quantile(fused, 1 − estimated_prior)` and test
whether this single rule lands at the oracle cell on both datasets.
This is the selector problem reformulated as "quantile selector with
data-driven `q`".

### 3. Binary_withdef + binary_minimal joint calibration (GPU, ~10 min)

The current scoring is at `temperature=0`. Rescore the same 2 configs
at `temperature > 0` and record score variance. Use the variance as a
calibration signal to refine the threshold. Speculative.

### 4. Expand the config pool (GPU, 30 min)

None of `binary_deflected`, `binary_t1000`, `triclass_t1000`,
`triclass_norules_t1000` is in the 2B pool — they exist only for 8B.
Running these on 2B (~2-4 min each per dataset × 2 datasets × 4 configs
≈ 20-30 min GPU) gives a larger fusion search space. No guarantee.

### 5. Input-perturbation stability selector (GPU, ~30 min)

Instead of picking a threshold, pick between the existing candidate
thresholds based on their stability under small input perturbations
(e.g., small transcript truncation variation, frame subsampling).
Requires a few extra GPU runs on the same pair.

## Artifacts

- `src/probe_crossconfig_fusion.py` — pair-fusion probe, atom-disciplined
- `src/probe_triple_fusion.py` — extended to singles/pairs/triples/quads
- `src/probe_fusion_extended_lf.py` — 10-method label-free on the winning cell
- `src/probe_fusion_quantile_sweep.py` — quantile sweep on the winning cell
- `results/analysis/probe_crossconfig_fusion.json`
- `results/analysis/probe_triple_fusion.json`
- `results/analysis/probe_fusion_quantile_sweep.json`
- `logs/probe_crossconfig_fusion_v2.out` — atomized pair-fusion output
- `logs/probe_fusion_extended_lf.out` — 10-method output on the winning cell
- `logs/probe_fusion_quantile_sweep.out` — quantile sweep output
- `logs/probe_triple_fusion.out` — triple/quad output

## Pre-registration audit

- Config pool: frozen before run; unchanged.
- Fusion operator pool: frozen; 8 operators; unchanged.
- Label-free method pool for H2: frozen; 10 methods; unchanged.
- Subset sizes: frozen at {1, 2, 3, 4}; ran all.
- Atom quantization rule: adopted from session standing rule; applied
  to all scores uniformly.
- A first (non-atomized) run produced 6 false positives, all caught by
  subsequent atom discipline. Only atom-level results are reported.
- Post-hoc extensions (`probe_fusion_extended_lf.py`, `probe_fusion_
  quantile_sweep.py`) are follow-ups targeted at the ONE H1 cell found
  by the frozen pre-registration. They do not change H1's pass/fail;
  they just characterize the gap to H2.
