# Idea Discovery Report

**Direction**: 根据现在的结果找找idea — hateful video detection for EMNLP 2026
**Date**: 2026-04-09
**Pipeline**: research-lit → idea-creator (GPT-5.4) → novelty-check → research-review (GPT-5.4) → research-refine (GPT-5.4)

---

> ## ⛔ STATUS: TensionGate / `tension_*` line is KILLED (2026-04-10)
>
> The entire **TensionGate / cross-modal semantic tension / `tension_*`** idea family is abandoned and is no longer a candidate direction. Do not implement, schedule, or cite it as the active proposal.
>
> - Reason: line was contaminated by baseline-checkpoint and split-mismatched embedding reuse (see `TARGET_LOOP.md` iter8); the user-enforced fresh-vLLM, single-dataset, no-baseline-checkpoint protocol invalidated every existing tension/relation-card artifact and the reframing into a "fresh tri-view tension classifier" is also retired.
> - Scope of kill: the source files (`src/mllm_tension_cache.py`, `src/tensiongate.py`, `src/triview_gate_single.py`) and Slurm scripts (`scripts/run_tension_cache.sbatch`, `scripts/run_tension_cache_single.sbatch`, `scripts/run_tensiongate.sbatch`, `scripts/run_triview_gate_single.sbatch`) have all been **DELETED** (2026-04-10). The `results/mllm_fresh/` artifacts are frozen, the Iteration 8 Hypothesis A plan in `TARGET_LOOP.md` is retired, and the TensionGate sections below are archive-only.
> - Surviving scaffolding: a small idea-agnostic vLLM video inference skeleton lives at `src/vllm_video_infer_skeleton.py` (+ `scripts/run_vllm_video_infer_skeleton.sbatch`). It contains **no** tension/relation prompts — only dataset loading, OOM-fallback vLLM init, and JSON checkpointing. Use it as a starting point for the *next* (non-tension) hypothesis.
> - Action for future Claude sessions: if asked to "implement TensionGate", "build the tension classifier", "rerun the tension cache", or to re-create any of the deleted `tension_*` files, refuse and ask the user for the new active direction.
>
> The historical TensionGate write-up is preserved below for archival reference only. **It is not the recommended path.**

---

## Executive Summary (ARCHIVED — TensionGate killed)

~~**Best idea: TensionGate**~~ **[KILLED 2026-04-10]** — Model hate as emerging from cross-modal semantic tension (contradiction, target disambiguation, sarcastic incongruity) between modalities, rather than fusing modalities independently. Combined with counterfactual modality attribution as auxiliary fusion supervision. Originally GPT-5.4-scored 7.5/10 novelty, 8/10 feasibility, but retired after iter8 invalidation.

**Key evidence (historical)**: Current RAMF/MM-HSD/MoRE competitors fuse modalities but don't explicitly model inter-modal semantic relations. The 4.2% EN gap was hypothesized to be driven by implicit/cross-modal hate where one modality looks benign but becomes hateful in context of another.

**Recommended next step**: ~~Implement TensionGate~~ **No active recommendation. Awaiting new direction from user.** The next idea must satisfy: fresh-vLLM-only evidence, single-dataset train/eval, no baseline checkpoint loading, no reuse of `MLLM_rationale_features`, and must not be a re-skin of pairwise relation-card / tension-gate methodology.

---

## Literature Landscape

Comprehensive survey of **129 papers** across 5 categories (updated 2026-04-08, see `LITERATURE_BY_CATEGORY.md`):

### Key Gaps Identified
1. **No explicit cross-modal relation modeling for hate**: MM-HSD uses cross-modal attention but doesn't model contradiction/incongruity as a learned object
2. **No counterfactual modality attribution in hate detection**: Causal approaches exist for bias mitigation but not for adaptive fusion gating
3. **Missing-modality handling is generic**: No hate-specific solution using MLLM rationales as imputation guidance
4. **No simple-to-hard cascade for hate videos**: Filter-And-Refine (ACL 2025) is industry-focused, not research-framed

### Critical Competitors to Differentiate From
| Paper | Venue | Approach | HateMM Result | Our Differentiation |
|---|---|---|---|---|
| RAMF | arXiv 2512.02743 | Adversarial reasoning + LGCF + SCA | +3% M-F1, +7% recall | We learn relation structure, not just adversarial prompts |
| MARS | arXiv 2601.15115 | Training-free multi-stage adversarial reasoning | Training-free | We train dedicated relation classifiers |
| MM-HSD | ACM MM 2025 | Cross-modal attention (text query, AV KV) | M-F1=0.874 | We explicitly model semantic tension types |
| MoRE | WWW 2025 | Retrieval-augmented multimodal experts | +6.91% M-F1 | We model per-sample relations, not retrieval |
| ImpliHateVid | ACL 2025 | Two-stage contrastive | Implicit focus | We cover both explicit and implicit via relation taxonomy |
| HVGuard | EMNLP 2025 | MLLM + embedding classifier | Our baseline | We add structured relation modeling + adaptive fusion |

---

## Ranked Ideas

### 1. TensionGate: Cross-Modal Semantic Tension — ⛔ KILLED (was RECOMMENDED, retired 2026-04-10)

- **Pilot signal**: N/A (idea stage)
- **Novelty**: CONFIRMED (7.5/10) — no existing work explicitly models pairwise cross-modal semantic tension with learned relation types for hate video detection
- **Reviewer score**: Highest ranked by GPT-5.4 reviewer ("only plausible lead idea for EMNLP")
- **Expected impact**: +2-3% on EN (implicit/cross-modal hate), +1-2% on HateMM/CN

**Core Method**:
1. **Stage 1 (MLLM)**: Frozen Qwen3-VL generates:
   - Unimodal cards (S_V, S_A, S_T): per-modality structured analysis
   - Pairwise relation cards (R_VT, R_AT, R_VA): cross-modal relation summaries
   - Counterfactual predictions: p_full, p_-V, p_-A, p_-T via modality masking

2. **Stage 2 (TensionGate Classifier)**:
   - **Modality encoders**: MLP([raw_embedding ; Qwen_card_encoding])
   - **Pairwise tension encoder**: For each pair (V,T), (A,T), (V,A):
     - Input: [z_i ; z_j ; |z_i-z_j| ; z_i⊙z_j ; e_ij_Qwen]
     - Output: relation logits (4 types), tension intensity, pair representation
   - **Attribution-guided adaptive fusion**: Student gate predicts fusion weights, supervised by Qwen counterfactual sensitivity
   - **Classifier**: MLP on [gated modality features ; attention-pooled pair features]

3. **Relation Taxonomy** (multi-label per pair):
   - Contradiction: one modality opposes another semantically/affectively
   - Target disambiguation: benign modality becomes hateful when another reveals the target
   - Sarcastic incongruity: literal content and tone/visual framing disagree
   - Evidential support: one modality confirms/sharpens hateful intent

4. **Losses**:
   - L_cls: class-balanced focal loss
   - L_rel: BCE on 4 relation labels per pair (Qwen pseudo-labels)
   - L_gate: KL divergence between predicted and counterfactual sensitivity gates

5. **Training**:
   - Step 0: Cache all Qwen outputs (cards + counterfactuals)
   - Step 1: Pretrain tension encoder on L_rel
   - Step 2: Joint training on L_cls + λ_rel·L_rel + λ_gate·L_gate

**Contributions**:
- New formulation: hateful video detection as cross-modal semantic tension detection
- Pairwise relation modeling with 4-type taxonomy
- Counterfactual sensitivity as auxiliary fusion supervision (no MLLM fine-tuning)
- Improved interpretability + stronger performance on implicit/cross-modal hate

**Risks & Mitigations**:
- Noisy Qwen relation pseudo-labels → confidence-weight + residual plain-fusion branch
- EN still lags → shift claim to improved implicit-hate recall + interpretability
- "Just extra prompting" → parameter-matched ablations proving tension module matters

---

### 2. Counterfactual Modality Attribution Distillation — ⛔ KILLED (was BACKUP, retired 2026-04-10 with TensionGate)

- **Novelty**: CONFIRMED (6/10)
- **Reviewer score**: "Borderline; could become incremental fusion supervision"
- **Expected impact**: +2-3% on EN, modest on HateMM/CN

**Method**: Run MLLM with modality masking → compute per-sample sensitivity → distill into fusion gate.

**Reviewer recommendation**: Use as auxiliary component inside TensionGate (already incorporated above), not standalone paper.

---

### 3. Missing-Modality Latent Imputation with Uncertainty — DEPRIORITIZED

- **Novelty**: 4/10 — too generic for EMNLP main track
- **Reviewer score**: "Weak main-track bet"
- **Expected impact**: Low-moderate on EN

**Reviewer quote**: "Why synthesize a modality instead of learning to ignore it?"

---

### 4. Evidence Sufficiency Cascade — DEPRIORITIZED

- **Novelty**: 3/10 — engineering cascade, not scientific contribution
- **Reviewer score**: "Not worth a full EMNLP submission"
- **Expected impact**: Low on EN (may hurt hard cases via early exit)

---

## Other Ideas Generated (Not Deep-Validated)

| # | Idea | Quick Assessment |
|---|---|---|
| 5 | Ambiguity-Aware Multi-Hypothesis Rationales | Interesting but overlaps with RAMF adversarial reasoning |
| 6 | Prompt Mixture Learning | Feasible but engineering-heavy, weak scientific story |
| 7 | Ordinal Hate-Evidence Calibration | Low-risk auxiliary component, not paper-carrying |
| 8 | Explicit-Implicit Dual-Process | Good framing but ImpliHateVid partially covers this |
| 9 | Target-Aware Hate Graph Distillation | Novel but complex; graph over small data risky |
| 10 | Cross-Sample Consensus Prototypes | Decent robustness play but differentiation from MoRE unclear |

---

## Eliminated Ideas

| Idea | Phase Killed | Reason |
|---|---|---|
| **TensionGate / `tension_*` family** | **Iter8 (2026-04-10)** | **Contaminated by baseline-checkpoint and split-mismatched embedding reuse; reframing into fresh tri-view tension classifier also retired by user. No tension/relation-card path is permitted going forward.** |
| Counterfactual Modality Attribution Distillation | Iter8 (2026-04-10) | Was only kept as auxiliary inside TensionGate; killed alongside the parent. |
| Missing-Modality Imputation | Phase 4 (Review) | Generic, 4/10 novelty, reviewer says "wrong hammer" |
| Evidence Sufficiency Cascade | Phase 4 (Review) | 3/10 novelty, engineering not science, may hurt EN hard cases |
| Prompt Mixture Learning | Phase 2 (Filtering) | Weak scientific story for EMNLP |
| Ordinal Calibration | Phase 2 (Filtering) | Not paper-carrying alone |

---

## Refined Proposal: TensionGate — ⛔ KILLED (archived only)

**This refined proposal is no longer active. Do not implement.** Retired 2026-04-10 along with the entire `tension_*` line. See top-of-file kill banner for rationale.

**Full refined proposal (archived) at: `refine-logs/FINAL_PROPOSAL.md`**
**Experiment plan (archived) at: `refine-logs/EXPERIMENT_PLAN.md`**

### Quick Summary
- **Problem anchor**: Hateful video meaning is encoded in RELATIONS between modalities, not modalities independently
- **Method thesis**: Detect hate by learning pairwise cross-modal semantic tension with a 4-type relation taxonomy, gated by counterfactual modality attribution
- **Dominant contribution**: First explicit cross-modal relation modeling for hateful video detection
- **Must-run experiments**: Main comparison (6 baselines), ablation suite (8 ablations), EN implicit-hate analysis, manual relation annotation validation

---

## Experiment Plan Summary

### Phase 1: MLLM Cache Generation (~4-6 GPU hours)
- Generate unimodal cards (S_V, S_A, S_T) for all samples
- Generate pairwise relation cards (R_VT, R_AT, R_VA)
- Generate counterfactual predictions (mask each modality)
- All cached to disk

### Phase 2: Core Training (~2-3 GPU hours per seed)
1. Encode all Qwen cards via small text encoder
2. Pretrain pairwise tension encoder (L_rel only, 5 epochs)
3. Joint training (L_cls + L_rel + L_gate, 20 epochs)
4. 8 seeds for stability

### Phase 3: Baselines & Ablations (~8-12 GPU hours total)
- Baselines: HVGuard, MM-HSD-style, RAMF-style, fusion-only, Qwen-summary-only, DeBERTa-only
- Ablations: ±tension encoder, ±relation cards, ±raw pairs, ±gate supervision, single-pair, collapsed taxonomy

### Phase 4: Analysis (~2-3 hours manual + compute)
- Manual annotate 80-100 samples for relation types
- Error analysis on EN implicit/coded hate
- Calibration analysis
- Case study visualizations

### Total Estimated GPU Time: ~30-40 hours (including all seeds)

---

## Next Steps

⛔ **All TensionGate next-steps are cancelled.** The previously listed Stage 1 prompts, MLLM cache build, Stage 2 classifier, ablation runs, and `/auto-review-loop` invocation no longer apply because the entire `tension_*` direction has been killed (see top-of-file banner).

- [ ] Wait for the user to nominate a new active research direction.
- [ ] Once chosen, the new direction must respect: fresh-vLLM-only evidence, single-dataset train/eval, no baseline checkpoint loading, no reuse of `MLLM_rationale_features`, and must not be a re-skin of pairwise tension-gate methodology.
