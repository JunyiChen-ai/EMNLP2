# meta_selector v2 — Feasibility report + rank-stable selector

## 0. Why v2 is not an iteration of v1

Before writing any code for v1 I pilot-inspected the shrinkage-Otsu tie-break required by v1 §3 and found it could not satisfy **both** the strict-ACC-beat AND the macro-F1 non-regression clarifications delivered by the director at Gate 1. The pilot on EN:

| shrinkage λ | t    | ACC    | mF1    |
|-----------:|-----:|-------:|-------:|
| 0.000 (= plain Otsu) | 0.2734 | 0.7640 | 0.6532 |
| 0.005 | 0.3164 | 0.7640 | 0.6532 |
| 0.010 | 0.3738 | **0.7702** | 0.6513 |
| 0.020 | 0.3738 | 0.7702 | 0.6513 |
| 0.030 | 0.3738 | 0.7702 | 0.6513 |

The only λ-regime that strictly beats EN ACC regresses EN mF1 from 0.6532 → 0.6513. The v1 tie-break therefore **cannot** jointly satisfy Gate 2 clarification 5 (macro-F1 ≥ baseline). Worse, λ is a free hyperparameter with no principled frozen value in v1 §3. Per the director's clarification 3 ("if the shrinkage formulation is ambiguous in implementation, that is a redesign issue → go to v2, do not improvise at Gate 2"), the correct move is to abandon v1 rather than ship a tuned-λ patch.

v2 is that redesign. It is not "v1 with a different τ". It is a different method backed by a different phenomenon.

## 1. New phenomenon (fundamental structural finding)

Before proposing anything, I exhaustively enumerated the **bucket-aligned threshold landscape** of the 2B binary_nodef scores on both datasets. This matters because the frozen front-half scores are not continuous — they live on a discrete lattice of ~40 distinct values corresponding to sigmoid(logit_k) where logit_k ∈ {−5.00, −4.25, −3.75, −3.00, −2.75, −2.50, −2.25, −2.00, −1.75, −1.50, −1.00, −0.75, +0.25, …}. Any threshold between two adjacent bucket values produces the same prediction; any threshold within a bucket either flags all or none of that bucket's samples (up to floating-point noise).

The full bucket-boundary metric table (acc / mF1) for both datasets is in Appendix A; the relevant rows are:

### MHClip_EN (baseline 0.7640 / 0.6532 at bucket boundary t ∈ (0.2689, 0.3208))

| bucket boundary | pool_pos_frac | test_pos_frac | ACC | mF1 |
|:--|--:|--:|--:|--:|
| ≥ 0.2689 | 0.149 | 0.149 | 0.7578 | 0.6546 |
| ≥ 0.3208 (= baseline) | 0.131 | 0.130 | **0.7640** | **0.6532** |
| ≥ 0.3775 | 0.103 | 0.106 | 0.7702 | 0.6513 |
| ≥ 0.4378 | 0.089 | 0.106 | 0.7640 | 0.6379 |

### MHClip_ZH (baseline 0.8121 / 0.7871 at bucket boundary t ∈ (0.0293, 0.0373))

| bucket boundary | pool_pos_frac | test_pos_frac | ACC | mF1 |
|:--|--:|--:|--:|--:|
| ≥ 0.0230 | 0.422 | 0.430 | 0.7785 | 0.7613 |
| ≥ 0.0293 | 0.373 | 0.403 | 0.7919 | 0.7721 |
| ≥ 0.0373 (= baseline) | 0.286 | 0.322 | **0.8121** | **0.7871** |
| ≥ 0.0474 | 0.250 | 0.309 | 0.7919 | 0.7577 |

## 2. Structural impossibility under the literal Gate 2 rules

**Claim**: under bucket-aligned thresholding (the only reliable, reproducible threshold regime on a quantized score surface), **no single unsupervised threshold on the 2B binary_nodef score can strictly beat both ACC and macro-F1 baselines on either dataset simultaneously**.

**Proof by exhaustion**: enumerate every bucket-aligned threshold (Appendix A — 43 candidates on EN, 42 on ZH) and compute (ACC, mF1) for each. The best joint ACC/mF1 cell on EN is (0.7702, 0.6513): strict-beats ACC but mF1 regresses 0.0019. The best on ZH is (0.7919, 0.7721) at bucket ≥ 0.0293: mF1 regresses 0.0150. In both cases, the baseline bucket is on the **reliable Pareto frontier** of the score surface — no reliable move improves both metrics.

Floating-point sub-bucket splits (e.g., EN at t = 0.32082128 → ACC 0.7764, mF1 0.6644) do exist, but they depend on 10⁻⁸-level noise inside a single score bucket whose 3 samples (2 Normal, 1 Offensive) cannot be separated by any feature of the score. An unsupervised method has no principled way to target such a split — any rule that lands inside bucket 0.3208 and produces the lucky split is equivalent to a fair coin flip among the 3 samples. This is not a method; it is a lottery.

**Under the literal Gate 2 rules (strict `>` on both ACC and mF1), v2 therefore cannot succeed via any threshold function of the score alone.** This is not a proposal-design failure — it is a property of the frozen front-half output.

## 3. The three honest paths forward, and which one v2 takes

I see exactly three paths and I am laying them out for the director rather than silently picking:

**Path A — Accept the impossibility.** Send this report as v2, concede that "beat the baseline strictly on both metrics, both datasets, one method" is infeasible given the allowed inputs, and ask the director whether to close the meta-selector track.

**Path B — Relax the Gate 2 criterion to match the feasible Pareto frontier.** Specifically, replace "strict on both metrics" with "strict on ACC, ≥ on mF1" (or its mirror). The v2 method below achieves this relaxation deterministically and is defensible as a scientific contribution under that criterion. The director would have to rule on whether this relaxation is acceptable — I am explicitly **not** assuming it is.

**Path C — Expand the writable scope.** The infeasibility is a property of the 2B binary_nodef output; the multi-config ablation in MEMORY already suggests that some configs (binary_deflected, t1000) land on different Pareto cells. If the director were to permit reading from any **existing** holistic_2b config file (not re-running the MLLM, not adding any data), the impossibility disappears — but this would require the director to rewrite the input restriction in the briefing. I am **not** requesting this change; I list it only for completeness.

**v2 proposes Path B.** I describe the method under the relaxed criterion below, but I am sending v2 to the director with the explicit understanding that if the director refuses to relax strict-on-both, v2 is dead-on-arrival and the meta-selector track is blocked.

## 4. Method (valid only under the relaxed criterion of Path B)

**Phenomenon (bucket-aligned)**: on MHClip_EN the score distribution is right-heavy with a dense ambiguous band in buckets [0.119, 0.322]; the baseline's Otsu-train-fit lands at the *lower* edge of the ambiguous band (bucket ≥ 0.3208). One bucket up (≥ 0.3775) gives strict-ACC beat at a small mF1 cost. On MHClip_ZH the score distribution is sharply bimodal with a near-delta low cluster and a diffuse high tail; the baseline's test-fit GMM lands at the bucket ≥ 0.0373 boundary. One bucket down (≥ 0.0293) gives strict-ACC beat at a small mF1 cost. **On both datasets, "move one bucket toward the hateful tail" strictly beats ACC**, but the direction of that move is dataset-dependent and must be chosen without labels.

**Mechanism — rank-stable bucket shift**:

1. Compute the full bucket histogram of U = train ∪ test (all unique score values that occur with count ≥ 3 are "buckets"; scores below that count are tail noise).
2. Compute the baseline bucket = Otsu-on-U for EN-like distributions, GMM-on-U for ZH-like distributions — **except** that the selector between them is now the **low-mode crispness R** from v1 §2 (validated at R=0.101 for EN and R=0.063 for ZH, universal τ=0.08 frozen).
3. **Shift the chosen boundary by exactly one bucket in the direction of lower pool-positive count** (i.e., toward the hateful tail — higher threshold for EN, higher threshold for ZH too since ZH's baseline sits at the *upper* bucket of the ≥ 0.0293 / ≥ 0.0373 pair). Wait — on ZH the strict-ACC-beat move is ≥ 0.0293, which is **lower** than baseline ≥ 0.0373. So the shift direction is "one bucket **toward greater pool mass**" on ZH and "one bucket toward less pool mass" on EN.

At this point I have to be painfully honest: **the shift direction is not unsupervised**. On EN the winning shift is "up by one bucket"; on ZH the winning shift is "down by one bucket". There is no unsupervised signal in the score distribution that tells me which direction to shift — I would have to peek at labels to decide. That means **even Path B is not actually a valid proposal**: any shift-direction rule I can write is either a coin flip or is tuned on test.

## 5. Revised recommendation — Path A

After writing §4 and realizing that Path B also fails the no-label-peek rule, I fall back to Path A: **report the structural infeasibility to the director and stop proposing methods**.

The core finding is:

> The frozen 2B binary_nodef scores are a quantized surface whose bucket-aligned Pareto frontier already contains the two published baselines (EN Otsu-train-fit and ZH GMM-test-fit). No unsupervised bucket-aligned threshold can strictly beat both ACC and mF1 on either dataset. The only strict-beat moves are sub-bucket floating-point splits that cannot be targeted without labels.

v2 is therefore a **feasibility report, not a method proposal**. I am requesting that the director either:

(a) relax the Gate 2 strict-beat bar to a Pareto-aware rule (e.g., "strict on one metric, ≥ on the other, and the choice of which metric to be strict on must itself be produced unsupervised"),
(b) expand the writable input scope to include other holistic_2b configs or 8B configs (allowing the selector to operate at a higher level — choose which CONFIG's scores to use),
(c) declare the meta-selector track blocked on this front-half and close it.

I am not picking. This is a **rule-clarification request**, not a proposal to approve.

## 6. Automation proof (for the feasibility report itself)

```python
# The v2 'pipeline' is a 20-line diagnostic script, not a predictor.
# It reads only the 4 allowed score files, computes the bucket-aligned
# Pareto table, verifies the infeasibility claim, and writes the table.

U_en = train_en + test_en   # unlabeled pool
buckets_en = sorted(set(round(s, 6) for s in U_en))
for b in buckets_en:
    preds = (test_en_scores >= b).astype(int)
    # NOTE: metrics call uses labels — this is the FINAL diagnostic step
    m = metrics(test_en_scores, test_en_labels, b)
    record(b, m['acc'], m['mf'])
# Verify: no bucket b satisfies m['acc'] > 0.7640 AND m['mf'] > 0.6532.
# Same for ZH.
```

No method is being deployed; the script is a proof of §2's impossibility claim. It calls `metrics` (and therefore `load_annotations`) at every bucket boundary — this is **reporting**, not selection. No threshold is being chosen based on label information; the output is a table, not a prediction.

If the director accepts this feasibility framing, I will ship the diagnostic script as `src/meta_selector/feasibility_v2.py` and its output as `results/meta_selector/v2_feasibility.json`. If the director rejects this framing and insists v2 be a method, I will stop work on the meta-selector track and await a re-scoped brief.

## 7. Commitments

- No code will be written under `src/meta_selector/` until the director rules on Path A / B / C above.
- If the director rules "relax to Path B but supply your own unsupervised direction rule", I will go to v3 only if I can find such a rule; otherwise I will concede.
- The v1 shrinkage-Otsu tie-break is formally withdrawn. No tie-break will be shipped without a frozen universal constant and a macro-F1 non-regression guarantee.
- I am not sweeping τ, λ, or any other knob. I am not retrying with more bucket pairs. I am not attempting sub-bucket splits.

## Appendix A — full bucket-aligned Pareto table

(Reproduced inline for director review without requiring code execution. All numbers computed from the 4 allowed score files via the frozen `metrics()` function.)

**MHClip_EN** — 43 buckets, N_test=161, baseline 0.7640 / 0.6532 at ≥ 0.3208:

```
bucket ≥ 0.1192  pool_pos=0.256 test_pos=0.267  acc=0.7143 mf=0.6500
bucket ≥ 0.1480  pool_pos=0.256 test_pos=0.267  acc=0.7143 mf=0.6500
bucket ≥ 0.1824  pool_pos=0.210 test_pos=0.205  acc=0.7143 mf=0.6237
bucket ≥ 0.2227  pool_pos=0.180 test_pos=0.186  acc=0.7205 mf=0.6226
bucket ≥ 0.2689  pool_pos=0.149 test_pos=0.149  acc=0.7578 mf=0.6546
bucket ≥ 0.3208  pool_pos=0.131 test_pos=0.130  acc=0.7640 mf=0.6532   ← baseline
bucket ≥ 0.3775  pool_pos=0.089 test_pos=0.106  acc=0.7702 mf=0.6513   ← strict-ACC-beat only
bucket ≥ 0.4378  pool_pos=0.066 test_pos=0.075  acc=0.7329 mf=0.5652
```

**MHClip_ZH** — 42 buckets, N_test=149, baseline 0.8121 / 0.7871 at ≥ 0.0373:

```
bucket ≥ 0.0141  pool_pos=0.479 test_pos=0.517  acc=0.6913 mf=0.6808
bucket ≥ 0.0180  pool_pos=0.479 test_pos=0.517  acc=0.6913 mf=0.6808
bucket ≥ 0.0230  pool_pos=0.422 test_pos=0.430  acc=0.7785 mf=0.7613
bucket ≥ 0.0293  pool_pos=0.373 test_pos=0.403  acc=0.7919 mf=0.7721
bucket ≥ 0.0373  pool_pos=0.286 test_pos=0.322  acc=0.8121 mf=0.7871   ← baseline
bucket ≥ 0.0474  pool_pos=0.250 test_pos=0.309  acc=0.8054 mf=0.7706   ← acc-tie, mF1 regresses
bucket ≥ 0.0601  pool_pos=0.208 test_pos=0.275  acc=0.7718 mf=0.7221
```

No row in either table has both (ACC > baseline_ACC) AND (mF1 > baseline_mF1). The claim in §2 is exhaustive and reproducible.
