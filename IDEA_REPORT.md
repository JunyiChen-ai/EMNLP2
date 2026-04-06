# Research Idea Report: Hateful Video Detection

**Direction**: Novel methods for hateful video detection, drawing from general ML advances NOT yet applied in hateful video/meme detection
**Generated**: 2026-04-06
**Updated**: 2026-04-07 (pilot results added)
**Ideas evaluated**: 12 generated → 12 reviewed → 5 piloted → 0 passed strict threshold
**Source literature**: 72 papers from LITERATURE_BY_CATEGORY.md + web search for 2025-2026 top venues

---

## Landscape Summary

The hateful video detection space (2024-2026) is dominated by three paradigms: (1) MLLM-as-text-generator → classifier pipelines (HVGuard, MoRE), (2) cross-modal attention fusion (MM-HSD, ImpliHateVid), and (3) adversarial/debate reasoning (RAMF, ARCADE). All share a common blind spot: they treat multimodal fusion as a symmetric problem where all modalities contribute roughly equally. Our empirical facts show otherwise—MLLM text is the dominant signal (+6-8pp), base modalities add only 0.7-3.2pp, and seed variance is 5-8pp (an elephant nobody addresses).

Meanwhile, the broader ML community has produced powerful tools that haven't entered this task:
- **Hallucination/grounding verification** (HalLoc, ContextualLens, MARINE) — detecting when MLLM text isn't supported by raw evidence
- **Modality imbalance theory** (representation collapse, boosting-based balancing, gradient modulation) — principled asymmetric fusion
- **Evidential deep learning** (Dempster-Shafer, belief entropy) — separating conflict from ignorance in multimodal decisions
- **Feature distribution alignment** (Proxy-FDA, StarFT) — stabilizing training across seeds
- **Calibrated decoding** (C-PMI, AVCD) — modality-aware output calibration

The opportunity is to use these general tools to address the specific pathologies of hateful video detection: text dominance, weak auxiliary modalities, high seed variance, and implicit/relational hate signals.

---

## Pilot Experiment Results (10 seeds, 50 epochs)

All 5 ideas were piloted on 3 datasets. Positive signal threshold: >1pp mean F1 gain OR meaningful stability win.

### HateMM (Binary, 1066 videos)

| Variant | Mean F1 | Std F1 | Worst F1 | Best F1 | Δ Mean F1 | Verdict |
|---------|---------|--------|----------|---------|-----------|---------|
| baseline (text MLP) | 86.89 | 1.39 | 84.39 | 88.50 | — | Baseline |
| GTT | 84.44 | 1.29 | 81.45 | 86.06 | -2.45 | No signal |
| VMBT | 86.95 | 1.71 | 83.53 | 88.83 | +0.05 | No signal |
| BORF | 86.81 | 1.23 | 84.35 | 88.00 | -0.09 | No signal |
| **RCD** | **87.25** | **1.14** | **85.25** | **88.50** | **+0.36** | **Borderline** |
| CMDE | 85.59 | 1.14 | 84.30 | 87.50 | -1.31 | No signal |

### MHClip-EN Binary (Offensive+Hateful→Hate, 891 videos)

| Variant | Mean F1 | Std F1 | Worst F1 | Best F1 | Δ Mean F1 | Verdict |
|---------|---------|--------|----------|---------|-----------|---------|
| baseline | 70.56 | 2.49 | 65.75 | 73.09 | — | Baseline |
| **VMBT** | **75.75** | 2.66 | 68.99 | 78.72 | **+5.19** | **Positive** |
| BORF | 69.85 | 3.40 | 62.53 | 72.85 | -0.71 | No signal |
| RCD | 70.17 | 2.98 | 65.44 | 74.00 | -0.39 | No signal |
| **CMDE** | **71.69** | 4.27 | 66.13 | 76.98 | **+1.14** | **Positive** |

### MHClip-ZH Binary (Offensive+Hateful→Hate, 897 videos)

| Variant | Mean F1 | Std F1 | Worst F1 | Best F1 | Δ Mean F1 | Verdict |
|---------|---------|--------|----------|---------|-----------|---------|
| baseline | 78.66 | 2.33 | 74.00 | 81.34 | — | Baseline |
| VMBT | 75.22 | 2.88 | 69.70 | 78.43 | -3.44 | No signal |
| BORF | 78.80 | 2.13 | 73.38 | 81.03 | +0.14 | Borderline |
| RCD | 78.61 | 2.59 | 73.16 | 81.12 | -0.05 | No signal |
| CMDE | 77.93 | 1.61 | 75.02 | 80.30 | -0.73 | No signal |

### Cross-Dataset Consistency

| Method | HateMM | MHClip-EN | MHClip-ZH | Pattern |
|--------|--------|-----------|-----------|---------|
| **GTT** | -2.45 | — | — | Failed: AV too coarse to validate fine-grained text |
| **VMBT** | +0.05 | **+5.19** | -3.44 | Inconsistent: huge gain on EN, hurt on ZH |
| **BORF** | -0.09 | -0.71 | +0.14 | Most stable: never hurts significantly, never helps much |
| **RCD** | +0.36 | -0.39 | -0.05 | Only HateMM signal, didn't transfer |
| **CMDE** | -1.31 | +1.14 | -0.73 | Inconsistent |

**No method consistently beats baseline across all 3 datasets.**

---

## Ideas (with Pilot Outcomes)

### Idea 1: Grounded Token Trust (GTT) — ❌ KILLED

- **Hypothesis**: Raw modalities help most by telling you which MLLM-generated text tokens are trustworthy.
- **Combines**: HalLoc + ContextualLens
- **Pilot result**: -2.45 F1 on HateMM. Worst performer.
- **Why it failed**: Global AV features [768] are too coarse to validate token-level text. Implicit hate cues are inferential/abstract — not verifiable from raw audio/frames. The trust head suppressed exactly the most discriminative tokens.
- **Lesson**: "AV as token-level validator" is the wrong abstraction for this task.

### Idea 2: Variance-Minimized Boundary Training (VMBT) — ⚠️ MIXED

- **Hypothesis**: Seed variance comes from unstable boundary pockets; EMA + distribution alignment fixes it.
- **Combines**: Proxy-FDA + boosting-based modality balancing
- **Pilot result**: +0.05 on HateMM (neutral), **+5.19 on MHClip-EN** (strong), -3.44 on MHClip-ZH (hurt).
- **Why mixed**: On MHClip-EN the text-only baseline is weak (70.56 F1), so the multimodal concat [text;audio;frame] backbone genuinely helps. On HateMM where text already dominates (86.89 F1), adding AV via concat adds noise. On MHClip-ZH the EMA regularization destabilized instead of helping.
- **Lesson**: VMBT's value comes from the multimodal backbone, not the stability mechanism. The EMA+FDA didn't deliver the intended variance reduction.

### Idea 3: Boundary-Only Residual Fusion (BORF) — ⚠️ NEUTRAL

- **Hypothesis**: Audio/frame should only correct the boundary where text is ambiguous.
- **Combines**: MTS Taylor series + boosting
- **Pilot result**: -0.09 / -0.71 / +0.14 across 3 datasets. Never hurts badly, never helps.
- **Why neutral**: The ambiguity gate works as intended (residuals activate only on uncertain samples), but the residual branches don't learn useful corrections. Audio/frame may genuinely lack complementary signal on these datasets.
- **Lesson**: Most stable method. The asymmetric design is sound, but there may not be enough correctable signal in audio/frame features.

### Idea 4: Residual Correlation Distillation (RCD) — ⚠️ BORDERLINE (HateMM only)

- **Hypothesis**: Audio/frame collapse toward text shortcuts; distilling residuals preserves complementarity.
- **Combines**: Modality collapse diagnosis + CMAD
- **Pilot result**: **+0.36 on HateMM** with lower std (1.14 vs 1.39) and better worst-seed (85.25 vs 84.39). But -0.39 on MHClip-EN and -0.05 on MHClip-ZH.
- **Why HateMM-only**: The residual target (text errors) is meaningful on HateMM where the text teacher is strong. On MHClip where text baseline is weaker, residuals are noisier.
- **Lesson**: Residual distillation shows the most coherent directional pattern on HateMM (all stability metrics improve simultaneously). Worth bounded follow-up on that dataset, but not a general solution.

### Idea 5: Cross-Modal Description Editor (CMDE) — ❌ MOSTLY KILLED

- **Hypothesis**: Best use of AV is to edit text representations before classification.
- **Combines**: C-PMI + AVCD
- **Pilot result**: -1.31 on HateMM, +1.14 on MHClip-EN, -0.73 on MHClip-ZH. Inconsistent.
- **Why it failed on HateMM**: Same core problem as GTT — AV-based gating removes useful implicit-hate text fields. On MHClip-EN where text is weaker, the editing occasionally helps by suppressing noisy fields.
- **Lesson**: "AV edits text" shares the same failure mode as "AV validates text" — coarse AV features can't reliably judge fine-grained text quality.

---

## Eliminated Ideas (pre-pilot)

| Idea | Reason eliminated |
|------|-------------------|
| Conflict Is Signal (evidential DS fusion) | Novelty too low — "evidential fusion on a new task"; likely a calibration story, not accuracy |
| Latent Questions, Not Prompts (VERA-style) | Borderline — "learned query vectors" ≈ rebranded cross-attention; hard to differentiate |
| Rescue-Then-Adapt (dynamic routing + TTA) | Two techniques stapled together; too complex for 2 months; routing errors cascade |
| Relational Anchors (CMAD + SAR) | Too close to killed directions (prototypes, concept bottleneck); easy reviewer reduction |
| Short Cue, Long Context (dual memory + TACA) | Datasets may not require dual-memory reasoning; inflated architecture for the signal |
| Ungrounded = Suspect (HalLoc + StarFT) | Too close to contrastive/spuriosity territory; may suppress useful implicit-hate tokens |
| Disentangled Support Maps (MASH-VLM + HalLoc) | Spatial/temporal split is artificial for social/contextual hate signals |

---

## Key Lessons from Pilot

1. **"AV as text validator/editor" is wrong for this task.** GTT and CMDE both failed because global audio/frame features cannot reliably judge whether MLLM-generated text tokens/fields are correct. Hate signals are often abstract, inferential, or culturally contextual — not directly verifiable from raw modalities.

2. **No method consistently beats text-only MLP across all datasets.** This is the hardest finding. It suggests that on these datasets, the text-only representation is already near the ceiling achievable with these audio/frame features.

3. **The multimodal gain depends on how strong text-only already is.** MHClip-EN (text baseline 70.56 F1) benefits most from multimodal methods. HateMM (text baseline 86.89 F1) barely benefits. This matches the intuition: when text is already near-sufficient, adding weak modalities mostly adds noise.

4. **RCD is the only method with coherent stability improvement on HateMM**: lower std, better worst-seed, slight mean gain. But this didn't transfer to other datasets.

5. **BORF is the most robust design**: it never significantly hurts any dataset. The asymmetric "residual only when uncertain" design prevents the damage that symmetric fusion causes. But it also doesn't help enough to be a contribution.

---

## Remaining Options

1. **Bounded RCD follow-up on HateMM** — 2-3 targeted changes (selective residual application, better alpha/beta calibration). Stop rule: if no clear win after that.
2. **Investigate why MHClip-EN benefits from multimodal but HateMM doesn't** — this could lead to a dataset-adaptive method.
3. **Pivot entirely** — the pilot evidence suggests that for MLLM-textualized hateful video detection, the multimodal fusion problem may be fundamentally limited by the quality of audio/frame features, not by the fusion method.
