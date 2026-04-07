# Research Idea Report

**Direction**: MLLM rationale → multimodal fusion pipeline for hateful video detection
**Generated**: 2026-04-07
**Ideas evaluated**: 10 generated → 6 survived filtering → 3 recommended
**Source**: GPT-5.4 xhigh brainstorm + landscape analysis + constraint filtering

---

## Landscape Summary

The proven pipeline — MLLM generates text rationale → encode → fuse with audio/frame → classify — is the dominant paradigm for video hate detection (HVGuard, RAMF, MARS, our baseline). The rationale provides +6-8pp over no rationale, but a simple MLP on rationale text alone reaches ~87 F1, making raw modalities nearly redundant (+0.7-3.2pp).

**The core paradox**: Raw modalities (frame, audio) ARE the ground truth about the video, yet they add almost nothing on top of the MLLM's text summary. This happens because: (1) the MLLM rationale already summarizes what the raw modalities show, making them redundant as parallel features; (2) current fusion treats all information sources as additive feature channels, rather than distinguishing their epistemic roles.

**The untapped opportunity**: Our diagnostic prompt ALREADY produces structured rationales with explicit [Grounded Observations] vs [Diagnostic Interpretation] separation. But the downstream pipeline ignores this structure entirely — it compresses everything into one [CLS] embedding. Nobody in this field (or in adjacent tasks like fake news, propaganda, sarcasm) has exploited observation/interpretation structure for modality-grounded classification.

**Philosophies from adjacent literature ready to adapt**:
- FaithScore (EMNLP 2024): decompose → verify atomic claims → score
- Representation Collapse (ICML 2025 Spotlight): modality collapse via shared neurons → orthogonal subspaces
- AVCD (NeurIPS 2025): contrastive decoding across modalities
- DSANet (AAAI 2026): normality prototypes + deviation scoring
- VERA (CVPR 2025): learnable guiding questions optimized via verbal feedback

---

## Recommended Ideas (ranked)

### Idea 1: Observation-Grounded Classification (OGC)

**Core Motivation**: MLLM rationales mix two epistemically different types of content: *grounded observations* ("the speaker uses a racial slur," "a burning cross is visible") and *inferential interpretations* ("the video expresses hate toward group X"). Current pipelines compress both into one embedding, so the classifier cannot distinguish reliable observation from potentially hallucinated interpretation. Raw modalities can verify observations but NOT interpretations — if we separate them, raw modalities finally have a principled role.

**Problem Solved**: (1) Hallucinated observations mislead the classifier — verification filters them out. (2) Raw modalities are underused (+0.7-3.2pp) because they're treated as parallel features instead of verification signals. (3) The motivation is clean and reviewer-friendly: "trust the interpretation only if the observations are grounded."

**Philosophy Borrowed**: FaithScore (EMNLP 2024) — decompose → verify → score. Adapted from caption faithfulness to rationale grounding in a classification pipeline.

**Method Sketch**:
1. Parse the diagnostic rationale into `observation_text` (Section 2: Grounded Observations) and `interpretation_text` (Sections 3-5)
2. Encode separately: `e_obs = Encoder(observation_text)`, `e_int = Encoder(interpretation_text)`
3. Compute grounding score: `g = σ(MLP([e_obs; e_frame; e_audio]))` — measures how well raw modalities corroborate the observations
4. Gated classification: `logits = Classifier(g · e_int + (1-g) · [e_frame; e_audio])` — high grounding → trust interpretation; low grounding → fall back to raw modalities
5. Loss: standard CE + optional grounding regularizer

**Why Novel**:
- No work in hate detection, fake news, propaganda, or sarcasm separates rationale observations from interpretations for modality verification
- FaithScore verifies caption faithfulness but doesn't use it for downstream classification gating
- Closest work (RAMF, MARS) generates multi-hypothesis rationales but never verifies them against raw modalities
- The contribution is a NEW PARADIGM: raw modalities as **evidence verifiers**, not feature sources

**Novelty**: 9/10 — no direct competitor in any neighboring task
**Feasibility**: HIGH — uses existing diagnostic rationales, just changes encoding/fusion
**Risk**: LOW-MEDIUM
**Main failure mode**: Grounding score is noisy because observation text and CLIP/Wav2Vec embeddings occupy different semantic spaces. Mitigation: use cross-modal alignment (e.g., CLIP text encoder for observations, CLIP visual encoder for frames).
**Estimated effort**: 2-3 weeks implementation, no additional MLLM calls needed

---

### Idea 2: Contradiction-Aware Evidence Fusion (CAEF)

**Core Motivation**: In hateful videos, the most diagnostic signal is often not what each modality says individually, but whether modalities AGREE or CONTRADICT. A calm, friendly tone (audio) paired with extremist text overlay (visual) is more suspicious than either signal alone. Current fusion adds modalities — it should detect their *consistency*.

**Problem Solved**: (1) Additive fusion cannot represent contradiction — [calm_audio + extremist_text] averages out to "neutral," losing the diagnostic signal. (2) MLLM rationales already note cross-modal contradictions (our diagnostic prompt has a "Cross-modal contradiction" field), but this is just text — not computed from actual modality features. (3) Implicit hate specifically works by creating contradiction between surface content and underlying meaning.

**Philosophy Borrowed**: AVCD (NeurIPS 2025) — contrastive decoding across modalities + entropy-based weighting. Adapted from generation-time decoding to post-hoc fusion feature construction.

**Method Sketch**:
1. For each modality pair (rationale-frame, rationale-audio, frame-audio), compute agreement/contradiction features: `c_ij = e_i ⊙ e_j` (element-wise product captures alignment), `d_ij = |e_i - e_j|` (absolute difference captures divergence)
2. Construct evidence tensor: `E = [e_rat; e_frame; e_audio; c_rf; c_ra; d_rf; d_ra]`
3. Lightweight attention over E → classification
4. Optional: compare computed contradiction with MLLM's stated contradiction → meta-faithfulness signal

**Why Novel**:
- Contradiction/inconsistency has been used in fake news detection (MDAM3, DEFAME) but NEVER in hate detection
- No hate detection paper treats cross-modal disagreement as a first-class feature
- The contribution: multimodal hate detection should model CONSISTENCY, not just CONTENT

**Novelty**: 8/10 — idea exists in fake news but not hate detection; adaptation to rationale-based pipeline is new
**Feasibility**: HIGH — simple feature engineering on existing embeddings
**Risk**: MEDIUM
**Main failure mode**: Embedding spaces are too different for meaningful agreement/contradiction computation. Mitigation: project into shared space via learned linear projection before computing interactions.
**Estimated effort**: 1-2 weeks

---

### Idea 3: Normality-Anchored Deviation Detection (NADD)

**Core Motivation**: Implicit hate is hard because the surface content looks normal — ordinary visuals, conversational tone, no explicit slurs. The hate signal is in DEVIATIONS from normal cross-modal patterns: when what's said doesn't quite match what's shown, or when the topic-tone combination is subtly unusual. Current classifiers look for explicit hate features; they should look for *abnormal normalcy*.

**Problem Solved**: (1) Implicit hate detection is the hardest subproblem (ImpliHateVid ACL 2025 introduced it as a benchmark). (2) Current methods trained on explicit hate overfit to surface features (slurs, symbols) and fail on implicit cases. (3) The motivation is compelling: "Implicit hate hides in plain sight — detect it by finding what's subtly wrong, not what's obviously hateful."

**Philosophy Borrowed**: DSANet (AAAI 2026) — normality prototypes + deviation scoring. Adapted from video anomaly detection to hateful video detection, where "normal" means benign cross-modal alignment patterns.

**Method Sketch**:
1. From training set benign videos, learn topic-conditioned normality prototypes in the joint (rationale, frame, audio) embedding space
2. For each test video, compute deviation score: `dev = d(x, nearest_prototype(x))` — distance from the nearest normality prototype
3. Classify from: `[e_rat; e_frame; e_audio; dev; prototype_residual]`
4. Key insight: explicit hate has HIGH deviation from normality; implicit hate has MODERATE deviation (looks almost normal but not quite); benign has LOW deviation

**Why Novel**:
- Anomaly detection paradigm has NEVER been applied to hate speech detection
- Closest: DSANet (video anomaly), VadCLIP (video anomaly) — but these detect visual anomalies, not semantic/pragmatic ones
- The contribution: reframing implicit hate detection as a deviation-from-normality problem

**Novelty**: 8/10 — paradigm transfer from anomaly detection to hate detection is novel
**Feasibility**: MEDIUM — requires careful prototype learning, may need larger datasets
**Risk**: HIGH
**Main failure mode**: Normality prototypes are dataset-specific and don't transfer. Hateful and benign videos may overlap too much in embedding space.
**Estimated effort**: 3-4 weeks

---

## Tier 2 Ideas (Promising but Higher Risk)

### Idea 4: Anti-Collapse Orthogonal Fusion

- **Hypothesis**: Rationale dominance causes representation collapse (shared classifier neurons absorb text signal, leaving no rank for AV)
- **Philosophy**: Representation Collapse (ICML 2025 Spotlight)
- **Method**: Orthogonal subspace constraint — each modality gets private + shared dimensions, cross-modal KD prevents collapse
- **Risk**: MEDIUM — gains may be marginal if the real ceiling is low (+0.7-3.2pp means AV truly has little signal)
- **Closest prior**: Representation Collapse paper itself, but applied to rationale-dominant pipeline is new

### Idea 5: Rationale-Conditioned Temporal Grounding

- **Hypothesis**: Pooling frame/audio over entire video washes out temporally sparse hate signals
- **Philosophy**: FaithScore claim decomposition adapted as temporal selector
- **Method**: Decompose rationale into atomic claims → predict which video segments support each claim → aggregate modality features over support regions only
- **Risk**: MEDIUM-HIGH — most hate videos are short (<60s), temporal selection may not help
- **Closest prior**: MultiHateLoc (WWW 2026) does temporal localization but differently (MIL objective)

### Idea 6: Learnable Evidence Schema (VERA-style)

- **Hypothesis**: The diagnostic prompt fields are hand-designed; learning optimal evidence fields could improve rationale quality
- **Philosophy**: VERA (CVPR 2025) + TextGrad (ICML 2025)
- **Method**: Start with diagnostic prompt → train classifier → use loss as signal → optimize prompt fields via verbal feedback → iterate on small dev set
- **Risk**: MEDIUM-HIGH — budget constraint ($20-100 for MLLM calls), may need 3-5 iterations
- **Closest prior**: VERA does this for anomaly detection; nobody for hate detection

---

## Eliminated Ideas (from GPT-5.4 brainstorm)

| Idea | Reason eliminated |
|------|-------------------|
| Grounded Information Bottleneck | Too close to killed direction #34 (LSRB = concept/information bottleneck) |
| Residual Multimodal Correction | Too close to killed directions #12 (reliability-aware fusion) and #13 (error-focused two-regime, too few hard samples) |
| Interpretable Evidence-Interaction MoE | HVGuard already uses MoE; differentiation too weak; close to killed #22 (weakness-aware routing) |
| Weak-Modality Boosting | Close to killed #12 and #22; ceiling limited by inherent AV weakness (+0.7-3.2pp) |
| Multi-prompt distillation variants | Violates constraint #6 (no cross-prompt ensembling) |

---

## Constraint Compliance Check

| Constraint | Idea 1 (OGC) | Idea 2 (CAEF) | Idea 3 (NADD) |
|---|---|---|---|
| No human annotation | ✅ Auto-parsed | ✅ Auto | ✅ Auto |
| No multi-prompt ensemble | ✅ One prompt | ✅ One prompt | ✅ One prompt |
| Genuinely multimodal | ✅ Modalities verify observations | ✅ Contradiction requires ≥2 modalities | ✅ Joint normality prototypes |
| One core principle | ✅ Observation grounding | ✅ Consistency modeling | ✅ Deviation from normality |
| No model upgrade | ✅ Uses existing MLLM outputs | ✅ Same | ✅ Same |
| Builds on proven pipeline | ✅ Extends encoding + fusion | ✅ Extends fusion features | ✅ Adds prototype layer |
| Not killed direction | ✅ | ✅ | ✅ |
| EMNLP novelty bar | ✅ New paradigm | ✅ New for hate detection | ✅ Paradigm transfer |

---

## Novelty Verification Results

| Idea | Hate Detection | Neighboring Tasks (fake news, sarcasm, sentiment) | Assessment |
|------|---------------|---------------------------------------------------|------------|
| OGC (Idea 1) | No competitor | FaithScore decomposes claims but doesn't gate classification; fact-checking verifies but not in a fusion pipeline | **HIGH** — the classification-gating mechanism is new even across adjacent fields |
| CAEF (Idea 2) | No competitor | Contradiction signal used in fake news (CAFE, BMR) and sarcasm (incongruity models) — these are neighboring tasks | **MEDIUM** — core mechanism exists in neighboring tasks; adaptation to rationale verification adds value but reviewers may question novelty |
| NADD (Idea 3) | No competitor | Anomaly detection paradigm (DSANet, VadCLIP) not applied to semantic/pragmatic hate detection | **HIGH** — genuine paradigm transfer, no precedent in any neighboring NLP task |
| Anti-Collapse (Idea 4) | No competitor | MISA, Self-MM do private/shared orthogonal subspace in sentiment analysis — directly neighboring | **LOW** — well-known technique in multimodal sentiment, too close to neighboring tasks |

**User's novelty standard**: "方法只要没被邻近/相似任务就叫做novel" — a method is novel if it hasn't been done in neighboring tasks. By this standard, **Idea 1 and Idea 3 are clearly novel; Idea 2 is borderline; Idea 4 is not novel enough**.

**Revised ranking**: Idea 1 (OGC) >> Idea 3 (NADD) > Idea 2 (CAEF, as extension of Idea 1) >> Idea 4 (demoted)

---

## Suggested Execution Order

1. **Start with Idea 1 (OGC)** — lowest risk, strongest motivation, no additional MLLM calls, clean paper story. If observation grounding works, it also validates the foundation for Idea 2.

2. **Combine with Idea 2 (CAEF)** as an ablation/extension — once you have separate observation/interpretation embeddings, computing contradiction features is trivial. This can be the "full model" while OGC alone is the ablation.

3. **Idea 3 (NADD) as backup** — if OGC + CAEF don't yield sufficient gains, NADD offers a completely different angle. Best tested on ImpliHateVid specifically.

## Recommended Paper Framing

**Title candidates**:
- "Observation-Grounded Multimodal Fusion for Hateful Video Detection"
- "Trust but Verify: Evidence-Grounded MLLM Rationales for Video Hate Detection"
- "From Rationale to Evidence: Grounding MLLM Judgments with Multimodal Verification"

**Story arc**:
1. MLLM rationales are strong but unaccountable — they can hallucinate, and the classifier can't tell
2. Raw modalities are ground truth but underused in current pipelines (+0.7-3.2pp)
3. We redefine the role of raw modalities: from redundant features to evidence verifiers
4. The key insight: separate observations (verifiable) from interpretations (must be grounded), use modalities to verify observations, gate interpretations by grounding confidence
5. Result: better accuracy, better stability, and a principled multimodal story

---

## Next Steps

- [ ] Implement Idea 1 (OGC): parse existing diagnostic rationales into observation/interpretation, encode separately, build gated fusion
- [ ] Run pilot on HateMM with 3 seeds — compare against rationale-only MLP baseline
- [ ] If positive signal, add Idea 2 (CAEF) contradiction features as extension
- [ ] If confirmed, invoke `/auto-review-loop` for full iteration
