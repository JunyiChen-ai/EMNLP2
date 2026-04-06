# EMNLP2 Brainstorm Log: What We Tried and Why It Failed

**Last updated**: 2026-04-06
**Status**: Stuck — no surviving direction yet

---

## The Problem We're Trying to Solve

Hateful video detection using frozen MLLM textualization pipeline:
```
frozen MLLM (gpt-5.4-nano) → text rationale → BERT encode + AV features → classifier
```

Baseline performance: 87.3% ACC (text-only MLP on generic rationale). Need a novel EMNLP paper.

---

## Key Empirical Facts

1. MLLM rationale as features + simple MLP = 87.3% ACC (strong baseline)
2. MLLM as direct classifier = 74.2% ACC (poor) — rationale is more useful as features than as judgment
3. Generic prompt ≈ theory-specific prompt — SCM/ITT/IET all fail to beat generic
4. AV adds only 0.7-3.2pp on top of text, but reduces seed variance (std 1.43→0.79)
5. Seed variance is huge: 5-8pp max-mean gap
6. MLLM makes 139 FP (13% of all data): describes content correctly as "song/recording" but still judges hateful
7. MLLM makes 38 FN (3.6%): completely misses implicit/subtle hate
8. 77.7% of FPs have performance/historical cues that MLLM itself describes but ignores in judgment
9. 8 samples always wrong across ALL models (text-only, text+AV, SCM-MoE)
10. kNN on MLLM text embeddings adds only ~2.4 F1

---

## User's Core Concerns

1. **"I know how to improve performance, but I cannot find a novel story"** — better prompting works but isn't a contribution
2. **Any hand-designed taxonomy (frame types, stance labels, theory fields) must have either strong theoretical grounding or be intuitively obvious** — user is strict about this, will not accept ad hoc categories
3. **Solution must co-design prompt + downstream model under ONE principle** — not two ideas stapled together
4. **AV must be genuinely necessary, not decorative**
5. **Use-mention / communicative framing already tried in previous project** — effect was marginal
6. **Just adding more MLLM outputs as features will improve performance** — this is known engineering, not a contribution
7. **SCM lesson**: imposing external theory onto MLLM doesn't work; decomposition must emerge from actual failure modes
8. **Application paper needs an "aha" moment**: either a clearly overlooked error pattern, or a new paradigm

---

## Killed Ideas (with reasons)

### Session 1: 100-Round Brainstorm (20 directions killed)

| # | Idea | Why Killed |
|---|---|---|
| 1 | Adversarial/dual-hypothesis | = RAMF (2025), already published |
| 2 | Hand-designed prompt fields (stance, endorsement, gap_type, target) | "dirty prompting", no principled basis |
| 3 | Temporal localization + multiple losses | Bag of tricks, not unified |
| 4 | ALARM minor adaptation | Incremental over KDD 2026 |
| 5 | Intra-video self-contrast | Not universal (short videos, uniform hate/benign) |
| 6 | Corpus-level schema/pattern mining | Lossy compression of already-strong embeddings |
| 7 | Theory-guided methods (SCM/ITT) | Don't consistently beat generic prompt (empirical fact) |
| 8 | Counterfactual generation | Unrealistic for video |
| 9 | Debate/courtroom agents | = ARCADE (2026), efficiency problems |
| 10 | Training-free MLLM classification | Ceiling too low (62-84 F1) |
| 11 | Multi-prompt distillation | Not novel enough |
| 12 | Reliability-aware fusion | **NOT KILLED** — multimodal fusion method, keep for consideration |
| 13 | Error-focused two-regime learning | Too few hard samples (~50-100) |
| 14 | Latent discourse structure | Lossy compression of 150-token text |
| 15 | Narrative state transition (multi-call) | 7-29 calls/video, too expensive |
| 16 | Cross-channel commitment | Too expensive, weak signal |
| 17 | Prefix-based ambiguity resolution | Too expensive, noisy on short videos |
| 18 | Entity graph / target grounding | Hand-designed schema |
| 19 | Factorized privileged distillation | Pragmatic residual concept not well-defined |
| 20 | NormGraph distillation | Too ontology-dependent for 2 months |

### Session 2: 20-Round Discussion (7 directions killed)

| # | Idea | Why Killed |
|---|---|---|
| 21 | Multimodal Polarity Residual Learning | **NOT KILLED** — multimodal fusion method, keep for consideration (note: overlaps with AUG NeurIPS 2025) |
| 22 | SIDM (Seed-Induced Decision Multiplicity) | Collapses into Rashomon set literature (NeurIPS 2022, FAccT 2025) |
| 23 | Rationale-Conditioned Modality Residualization | **NOT KILLED** — multimodal fusion method, keep for consideration (note: related to FactorCL NeurIPS 2023) |
| 24 | Synergy-Only Multimodal Learning | **NOT KILLED** — multimodal fusion method, keep for consideration (note: related to PID literature) |
| 25 | SAM/SWA/ensembling as core method | Generic optimization trick, not a contribution |
| 26 | Add-on to SCM-MoE | Too incremental, blurs problem formulation |
| 27 | Domain-general method framing | Overclaiming from one task |

### Session 2 Survivor: COBRA (4.5/10 confidence)

**COBRA**: AV neighborhoods identify text-space directions likely induced by stochastic textualization; Jacobian regularizer penalizes sensitivity along those directions.
**Status**: **NOT KILLED** — multimodal fusion method, keep for consideration (note: user felt it was trivial as standalone story, but the fusion mechanism itself is valid)

### Session 3: Application Paper Discussion (7 directions killed)

| # | Idea | Why Killed |
|---|---|---|
| 28 | Multi-rationale consistency training | Violates budget + no-ensemble constraint |
| 29 | Structured evidence decomposition | = killed structured prompting |
| 30 | Instance ambiguity-aware training | Needs multiple rationales or ensembles |
| 31 | Expert routing by failure mode | Needs oracle labels |
| 32 | Cross-dataset robustness alone | Too thin for paper |
| 33 | Unsigned evidence-unit + AV gating | **NOT KILLED** — multimodal fusion method, keep for consideration (note: R2 concern = HAN + gated fusion) |
| 34 | Merging evidence auditing + COBRA | Two principles stapled = bag of tricks |

### Session 3 Survivor: Signed Evidence Auditing (6.5/10 confidence)

**Signed Evidence Auditing**: Split rationale into sentence units, AV-conditioned signed accept/ignore/reject scores per unit, separate accept/reject aggregation.
**Kill test result (implemented and run)**:
- text_unit_attn (84.65) < whole_mlp (86.79) — unit decomposition HURTS
- signed_auditor (84.28) < baseline — worst of all 4 variants
- Seed 46: all-accept collapse (100% accept, 0% reject)
- Higher seed variance (2.34) than baseline (1.57)
**Conclusion**: Unit splitting loses holistic [CLS] representation. Post-hoc auditing of already-wrong rationale doesn't fix root cause.

### Session 4: Story Search (this session)

| # | Idea | Why Killed |
|---|---|---|
| 35 | Communicative Framing + Frame Bottleneck Fusion | Taxonomy is ad hoc (4 labels: ENDORSED/QUOTED_REPORTED/PERFORMED_FICTIONAL/IMPLICIT_AMBIGUOUS). No theoretical source for the enumeration. User challenged "have you enumerated all frames?" — couldn't defend. Also, Gligoric (NAACL 2024) already did use-vs-mention for text hate speech. |
| 36 | Use-vs-Mention (2 labels) | Too trivial. User already tried in previous project — effect marginal. |
| 37 | Stance detection | "stance detection" is a mature NLP task; would be perceived as "stance detection applied to hate video". CASE 2025 already has multimodal hate+stance shared task. |
| 38 | "Only use description, remove judgment" | Too obvious, not a contribution |
| 39 | AV-Conditioned Rationale Purification | Too close to killed Signed Evidence Auditing (cross-attention on rationale tokens) |
| 40 | Corruption-Aware Fusion | **NOT KILLED** — multimodal fusion method, keep for consideration |
| 41 | Evidential Conflict Fusion | **NOT KILLED** — multimodal fusion method, keep for consideration |
| 42 | Neighborhood Self-Distillation | **NOT KILLED** — multimodal fusion method, keep for consideration |
| 43 | Segment-Grounded Evidence Accumulation | = killed temporal localization (#3) |
| 44 | Joint-Manifold Label Diffusion | **NOT KILLED** — multimodal fusion method, keep for consideration |
| 45 | Prototype Reorientation at Test Time | **NOT KILLED** — multimodal fusion method, keep for consideration |
| 46 | Case-Based Multimodal Calibration | **NOT KILLED** — multimodal fusion method, keep for consideration |
| 47 | Temporal Regularity as Shortcut Breaker | Too narrow (only works on songs) |
| 48 | Norm-Disentangled Moderation | Sounds like more hand-designed prompt fields |
| 49 | Counterfactual Boundary Witnesses | = killed counterfactual generation (#8) |
| 50 | Moderation as Precedent Retrieval | kNN on MLLM embeddings only adds 2.4 F1 (empirical fact #10); close to killed retrieval (#26); reviewer says "this is kNN with better features" |
| 51 | Harm-Transaction Graphs | = hand-designed schema (#18), graph extraction is noisy |
| 52 | Context-Demand Signatures | Interesting conceptually but impractical to validate |

---

## Surviving Multimodal Fusion Methods (not killed)

These were previously killed for lack of novel story, but the fusion mechanisms themselves remain valid candidates if paired with a good story/paradigm.

| # | Method | Mechanism | Related Work |
|---|---|---|---|
| 12 | Reliability-aware fusion | Dynamic modality weighting by estimated reliability | General multimodal fusion literature |
| 21 | Multimodal Polarity Residual Learning | AV learns residual to flip/sharpen borderline decisions | AUG (NeurIPS 2025 Oral) |
| 23 | Rationale-Conditioned Modality Residualization | Predict AV from text, use residual (text-unpredictable AV) | FactorCL (NeurIPS 2023) |
| 24 | Synergy-Only Multimodal Learning | Only learn joint text+AV signal, suppress redundant | PID literature (NeurIPS 2023) |
| 33 | Evidence-unit + AV gating | Per-unit text + AV-conditioned gating | HAN + gated fusion family |
| 40 | Corruption-Aware Fusion | Text+AV diagnose when MLLM judgment is corrupted | Noisy label learning |
| 41 | Evidential Conflict Fusion | Text-AV disagreement as explicit predictive signal | Evidential deep learning |
| 42 | Neighborhood Self-Distillation | Label averaging over joint text+AV neighbors | Self-distillation literature |
| 44 | Joint-Manifold Label Diffusion | Graph label propagation on text+AV manifold | GLPN-LLM (ACL 2025) |
| 45 | Prototype Reorientation | Test-time prototype adaptation in joint space | Test-time adaptation |
| 46 | Case-Based Multimodal Calibration | Retrieve similar cases, use relation features for correction | Retrieval-augmented classification |
| COBRA | Cross-Modal Boundary Anchoring | AV-graph Jacobian regularization on text classifier | JacHess, attribution reg |

---

## What's Left

### Known working engineering (not novel):
- Better/longer MLLM prompts → more features → higher ACC
- Adding AV features → reduces FP and seed variance
- More MLLM outputs concatenated → marginal improvements

### Open questions:
- Is there a paradigm shift hiding in the "MLLM describes correctly but judges wrongly" finding?
- Can the finding itself (systematic judgment bias in frozen MLLMs) be the contribution?
- Is there something from a completely different field (not NLP/CV) that applies here?
- Should we reconsider the paper format (findings paper? analysis paper? benchmark paper?)

### User's state:
Frustrated. Has been brainstorming for 200+ rounds across multiple sessions. Every direction either overlaps with existing work, is too incremental, or requires hand-designed components that can't be defended. Needs fresh perspective, not more brainstorming in the same space.

---

## Hard Constraints (unchanged)

1. NO human annotation in method
2. NO multi-prompt ensemble
3. NO model upgrade
4. ONE unified framework, one core principle
5. Must be genuinely multimodal
6. Must be novel and elegant
7. CCF-A venues for references
8. ~$20-100 MLLM budget
9. ~2 months timeline
10. 4 datasets: HateMM, MultiHateClip-EN, MultiHateClip-ZH, ImpliHateVid
