# Experiment Plan: Psychology-Theory-Guided Hateful Video Detection

**Date**: 2026-03-27
**Strategy**: Phased funnel — screen all 4 theories quickly, then implement full architectures for top performers.

---

## Overview

### The 4 Theories

| ID | Theory | Fields | Fusion Type | Middle Module |
|----|--------|:------:|-------------|---------------|
| ITT | Integrated Threat Theory | 6 | 4 parallel channels → accumulator | Threat Moderation Block |
| IET | Intergroup Emotion Theory | 5 | 3-stage sequential gating | Appraisal-to-Emotion Transport |
| ATT | Attribution Theory | 5 | Strict causal chain | Counterfactual Blame Filter |
| SCM | Stereotype Content Model | 5 | Dual-stream → Quadrant Composer | Quadrant Attractor |

### Datasets

| Dataset | Train | Dev | Test | Label |
|---------|:-----:|:---:|:----:|-------|
| HateMM | 744 | 107 | 215 | Non-Hate=0, Hate=1 |
| MultiHateClip-EN | 569 | 81 | 164 | Normal=0, Hateful=1 |
| MultiHateClip-ZH | 558 | 80 | 160 | Normal=0, Hateful=1 |

---

## Phase A: Quick Screening (Week 1-2)

### Goal
Identify which theory's structured rationale embeddings are most discriminative, using the **same baseline fusion** for all theories.

### A1. LLM Rationale Generation

**Model**: gpt-5.4-nano
**Settings**: temperature=0.0, top_p=1.0, max_tokens=350, structured JSON output
**Concurrency**: max_concurrent=10
**Retry**: up to 3 retries on malformed JSON, then 1 repair prompt, then empty string

**5 prompt variants to run per sample**:

#### Prompt 0: Generic Rationale (Control)
```
Analyze this video for potential hateful content:
- Title: {title}
- Transcript: {transcript}
- Visual: {frame_description}

Provide structured analysis:
Step 1 - Content Summary: What is this video about?
Step 2 - Target Analysis: Is any group targeted? Who?
Step 3 - Sentiment/Tone: What is the overall tone?
Step 4 - Harm Assessment: Is there potential harm? What kind?
Step 5 - Overall Judgment: Is this hateful? Why or why not?
```

#### Prompt 1: ITT (Integrated Threat Theory)
```
Analyze this video through the lens of intergroup threat perception:
- Title: {title}
- Transcript: {transcript}
- Visual: {frame_description}

Reason step by step:
Step 1 - Target Group: Which social group is being discussed or depicted? How central is this group to the video's message?
Step 2 - Realistic Threat: Does the video frame this group as a material threat — to jobs, safety, health, resources, or territory? What evidence?
Step 3 - Symbolic Threat: Does it frame the group as a threat to values, culture, religion, morals, or way of life? What evidence?
Step 4 - Anxiety/Discomfort: Does the video create feelings of unease, fear, or discomfort about this group? Through what means — tone, imagery, music, editing?
Step 5 - Stereotype Support: What negative stereotypes are invoked or reinforced? How are they presented as evidence?
Step 6 - Hostility Prescription: Does the accumulated threat perception lead to endorsing exclusion, hostility, or harm toward the group?
```
**Fields**: target_salience, realistic_threat, symbolic_threat, anxiety_discomfort, stereotype_support, hostility_prescription

#### Prompt 2: IET (Intergroup Emotion Theory)
```
Analyze this video through the lens of intergroup emotional dynamics:
- Title: {title}
- Transcript: {transcript}
- Visual: {frame_description}

Reason step by step:
Step 1 - Group Framing: Who is positioned as "us" (ingroup) and "them" (outgroup)? What markers define each group?
Step 2 - Appraisal Evidence: What is the outgroup portrayed as doing to the ingroup? What harm, threat, or violation is implied?
Step 3 - Emotion Inference: Given this appraisal, what group-based emotion is the viewer invited to feel? (anger / disgust / contempt / fear / none) Why?
Step 4 - Action Tendency: What collective response does the video endorse or imply? (exclude / shame / punish / attack / none)
Step 5 - Endorsement Stance: Does the video endorse or oppose this action tendency? (endorse / oppose / neutral) What signals this?
```
**Fields**: group_framing, appraisal_evidence, emotion_inference, action_tendency, endorsement_stance

#### Prompt 3: Attribution Theory
```
Analyze this video through the lens of blame and responsibility attribution:
- Title: {title}
- Transcript: {transcript}
- Visual: {frame_description}

Reason step by step:
Step 1 - Negative Outcome: What problem, harm, or negative situation is highlighted in this video? (crime, economic decline, moral decay, cultural loss, social disorder, etc.)
Step 2 - Causal Attribution: Who or what group is presented as the cause of this problem? How is causality established — direct claim, implication, juxtaposition, or montage?
Step 3 - Controllability: Is the cause portrayed as intentional and controllable by the blamed group? Or as accidental, structural, or beyond their control?
Step 4 - Responsibility & Blame: Given the attribution and controllability, how much responsibility and moral condemnation is assigned to this group?
Step 5 - Punitive Response: What punishment, exclusion, or hostile response is endorsed or implied as appropriate?
```
**Fields**: negative_outcome, causal_attribution, controllability, responsibility_blame, punitive_tendency

#### Prompt 4: SCM+BIAS (Stereotype Content Model)
```
Analyze this video through the lens of social group perception:
- Title: {title}
- Transcript: {transcript}
- Visual: {frame_description}

Reason step by step:
Step 1 - Target Group: Which social group is being discussed or depicted? What defines them?
Step 2 - Warmth Assessment: Is this group framed as warm (friendly, trustworthy, well-intentioned) or cold (hostile, untrustworthy, threatening)? What evidence from text, visuals, or tone?
Step 3 - Competence Assessment: Is this group framed as competent (capable, skilled, intelligent) or incompetent (foolish, backward, primitive)? What evidence?
Step 4 - Social Perception: Given the warmth and competence framing, what social perception does the video construct? (contempt / envy / pity / admiration / mixed)
Step 5 - Behavioral Implication: What treatment of this group does the video imply or endorse? (active harm / passive exclusion / patronizing / respect / none)
```
**Fields**: target_group, warmth_evidence, competence_evidence, social_perception, behavioral_tendency

### A2. Embedding Generation

- Encode each field independently with `bert-base-uncased` [CLS] → 768d
- Store as `{theory}_field_{name}_features.pth` per dataset
- Also compute pooled rationale embedding (mean of all fields) → 768d → `{theory}_rationale_features.pth`

### A3. Quick Screening Training

**Fusion**: Existing Multi-Head Gated Routing (4 heads, hidden=192, modality_dropout=0.15)
**Inputs**: text(768d) + audio(768d) + frame(768d) + rationale(768d)
**Training**: AdamW lr=2e-4, weighted CE 1:1.5, label_smoothing=0.03, EMA=0.999, 45 epochs
**Seeds**: 10 per theory per dataset → seeds={11, 17, 23, 31, 47, 59, 71, 89, 101, 131}

**Total Phase A runs**: 5 prompts × 3 datasets × 10 seeds = **150 runs**

### A4. Screening Metrics & Selection

**Metrics per theory per dataset** (10-seed statistics):
- Macro-F1 (mean, std, max)
- AUROC
- Accuracy

**Selection rule**:
1. Compute mean macro-F1 rank across 3 datasets
2. Must beat generic rationale (Prompt 0) on ≥2/3 datasets
3. Must have ≥+1.0 mean macro-F1 improvement over generic
4. **Phase B candidate** = best rank; **Phase C candidate** = second-best

**Also compute** (for diagnosis):
- Linear probe on rationale embedding only (logistic regression)
- Class centroid distance in rationale embedding space (Fisher ratio)

### A5. Estimated Cost

| Item | Count | Time |
|------|:-----:|:----:|
| LLM calls | ~14,700 (5 prompts × 2,678 samples + 10% retry) | ~2h |
| BERT encoding | ~64,000 field encodes | ~2h |
| Training runs | 150 | ~90 GPU-hours |

---

## Phase B: Full Implementation — Top Theory (Week 3-4)

### B1. Architecture Implementation

Implement theory-specific fusion + middle module for the winning theory.

#### If ITT wins:

**Fusion: 4 Parallel Threat Channels + Accumulator**
```
Each channel (realistic/symbolic/anxiety/hostility):
  Input: field_embedding + multimodal_context
  Architecture: FiLM-conditioned 2-head cross-attention
  Output: 192d

Threat Accumulator:
  Input: concat(4 channels) → 768d
  Gated aggregation → 384d
  Channel weights via softmax for interpretability
```

**Middle Module: Threat Moderation Block**
```
salience = MLP(target_salience_embedding) → scalar [0,1]
stereo_fit = cosine(stereotype_repr, target_repr) → scalar [0,1]
moderation = sigmoid(W_s·salience + W_t·stereo_fit + W_st·salience⊙stereo_fit)
output = threat_accumulator * (0.5 + 0.5 * moderation)
```

**Hyperparameter search**:
- channel_hidden: {128, 192, 256}
- moderation_gate: {scalar, vector-64, vector-384}
- moderation_dropout: {0.0, 0.1, 0.2}
- lr: {1e-4, 2e-4, 3e-4}
- dropout: {0.10, 0.15, 0.20}

#### If IET wins:

**Fusion: 3-Stage Sequential Gating**
```
Stage 1 - Appraisal:
  Input: appraisal_evidence + group_framing + multimodal_context
  Cross-attention (4 heads, hidden 384)
  Output: h_A

Stage 2 - Emotion:
  gate = sigmoid(W[h_A; emotion_inference; context])
  h_E = gate * f_emotion(inputs) + (1-gate) * h_A

Stage 3 - Action:
  gate = sigmoid(W[h_E; action_tendency; endorsement_stance; context])
  h_T = gate * f_action(inputs) + (1-gate) * h_E
```

**Middle Module: Appraisal-to-Emotion Transport**
```
6 emotion simplex vertices in 128d latent space
h_A → softmax projection → simplex weights
Reconstructed emotion = weighted sum of vertices
Coherence = cosine(reconstructed, actual emotion embedding)
Append coherence scalar + simplex weights to classifier input
```

**Hyperparameter search**:
- hidden: {256, 384}
- simplex_dim: {64, 128}
- transport_loss_weight: {0.05, 0.1, 0.2}
- gate_temperature: {0.7, 1.0, 1.3}

#### If Attribution wins:

**Fusion: Strict Causal Chain**
```
Stage 1: Outcome → Attribution
  Input: negative_outcome + multimodal_context
  Residual gated transition → attribution_state

Stage 2: Attribution → Blame
  Input: attribution_state + controllability + responsibility_blame
  Residual gated transition → blame_state

Stage 3: Blame → Punishment
  Input: blame_state + punitive_tendency
  Residual gated transition → punishment_state
```

**Middle Module: Counterfactual Blame Filter**
```
g_B = sigmoid(W_c·controllability + W_b·blame_state + b)
blame_filtered = g_B * blame_state + (1-g_B) * neutral_anchor
neutral_anchor: learned vector or running mean of non-hate examples
```

**Hyperparameter search**:
- chain_depth: {1, 2, 3}
- chain_hidden: {256, 384}
- gate_sharpness: {0.7, 1.0, 1.5}
- anchor_type: {learned, running_mean}

#### If SCM wins:

**Fusion: Dual-Stream + Quadrant Composer**
```
Warmth Stream:
  Input: warmth_evidence + multimodal_context + target_conditioning
  2-head cross-attention → 256d

Competence Stream:
  Same architecture → 256d

Quadrant Composer:
  concat(warmth, competence) → 4 quadrant logits → softmax
  quadrant_repr = weighted sum of 4 learned basis vectors (256d)
```

**Middle Module: Quadrant Attractor**
```
4 learned prototypes (contempt/envy/pity/admiration) in 256d
prototype_pull_loss for positive examples near hostile quadrants
harm_score = quadrant_dist · [1.0, 0.7, 0.3, 0.0]
```

**Hyperparameter search**:
- stream_hidden: {192, 256, 384}
- prototype_dim: {128, 256}
- prototype_loss_weight: {0.05, 0.1, 0.2}

### B2. Training Protocol

**3-stage pruning for seed search**:

1. **Coarse grid**: top ~24 hyperparameter configs × 5 seeds each = 120 runs
   - Select top 3 configs by dev macro-F1
2. **Medium search**: top 3 configs × 20 seeds = 60 runs
   - Select best config
3. **Full seed search**: best config × 600 seeds (200 seeds × 3 offsets) = 600 runs

**Training settings**: AdamW, weighted CE 1:1.5, label_smoothing=0.03, EMA=0.999, 60 epochs max, patience=10

### B3. Retrieval Augmentation

After seed search, apply same retrieval pipeline as AppraiseHate:
- Shrinkage-PCA whitening (Ledoit-Wolf, rank 32/48/64)
- CSLS kNN interpolation (k=10-40, temp=0.02-0.1, alpha=0.05-0.5)
- Threshold tuning on dev set

### B4. Estimated Cost

| Item | Runs | GPU-hours |
|------|:----:|:---------:|
| Coarse grid (3 datasets) | 360 | ~250 |
| Medium search | 180 | ~130 |
| Full seed search | 1,800 | ~1,100 |
| Retrieval sweep | ~50 configs × 3 datasets | ~20 |
| **Total Phase B** | | **~1,500** |

---

## Phase C: Second-Best Theory (Week 5-6)

Same as Phase B but for the second-ranked theory. Same 3-stage pruning.

**Estimated cost**: ~1,500 GPU-hours

---

## Phase D: Optional (Week 7 if time)

If time and compute allow, implement 3rd and 4th theories with abbreviated search (coarse grid + 200-seed search only).

---

## Ablation Matrix (Mandatory for Submission)

Run on top 2 full models, all 3 datasets, 5 seeds each.

### A. Rationale Source Ablations

| ID | Description |
|----|-------------|
| ABL-R0 | No rationale (raw modalities only) |
| ABL-R1 | Generic rationale (Prompt 0, same field count) |
| ABL-R2 | Shuffled field assignment (permute field names) |
| ABL-R3 | Random text control (length-matched unrelated text) |
| ABL-R4 | Different theory's fields (e.g., ITT model with IET rationale) |

### B. Architecture Ablations

| ID | Description |
|----|-------------|
| ABL-A0 | Mean-pooled fields (no theory-aware fusion) |
| ABL-A1 | Concat + MLP (no theory-aware fusion) |
| ABL-A2 | Theory-aware fusion WITHOUT middle module |
| ABL-A3 | Middle module WITHOUT theory-aware fusion |
| ABL-A4 | Parallel → sequential (or vice versa, depending on theory) |

### C. Theory-Specific Ablations

**ITT**:
| ID | Description |
|----|-------------|
| ABL-ITT1 | Remove realistic threat channel |
| ABL-ITT2 | Remove symbolic threat channel |
| ABL-ITT3 | Remove anxiety channel |
| ABL-ITT4 | Collapse 4 channels → 1 general_threat |
| ABL-ITT5 | Remove salience gating |
| ABL-ITT6 | Remove stereotype_fit gating |

**IET**:
| ID | Description |
|----|-------------|
| ABL-IET1 | Remove Appraisal→Emotion gate |
| ABL-IET2 | Remove Emotion→Action gate |
| ABL-IET3 | Remove coherence score |
| ABL-IET4 | Replace simplex transport with MLP |
| ABL-IET5 | Remove group_framing field |
| ABL-IET6 | Remove endorsement_stance field |

**Attribution**:
| ID | Description |
|----|-------------|
| ABL-ATT1 | Bypass controllability (no blame filter) |
| ABL-ATT2 | Bypass attribution→blame step |
| ABL-ATT3 | Replace chain with fully connected MLP |
| ABL-ATT4 | Learned counterfactual vs fixed neutral |

**SCM**:
| ID | Description |
|----|-------------|
| ABL-SCM1 | Warmth-only |
| ABL-SCM2 | Competence-only |
| ABL-SCM3 | Remove Quadrant Attractor |
| ABL-SCM4 | Hard quadrant assignment vs soft |

### D. Modality Ablations

| ID | Description |
|----|-------------|
| ABL-M0 | Text only |
| ABL-M1 | Text + rationale only |
| ABL-M2 | Text + audio + visual (no rationale) |
| ABL-M3 | Full multimodal + rationale (full model) |

### Ablation Cost Estimate
Top 2 theories × 3 datasets × ~12 ablations × 5 seeds = **360 runs** → ~250 GPU-hours

---

## Evaluation Protocol

### Primary Metrics
- **Macro-F1** (main metric for ranking)
- **AUROC**
- **AUPRC**

### Secondary Metrics
- Weighted-F1, Accuracy, Balanced Accuracy, MCC
- ECE (15 bins), Brier score

### Reporting Standard
For each dataset × method:
- 10-seed mean ± std (Phase A)
- Top-5 seed mean, all-seed mean, best seed (Phase B/C)
- This prevents "lucky seed" criticism

### Cross-Dataset Transfer
1. Train HateMM → test MHC-EN, MHC-ZH (zero-shot)
2. Train MHC-EN → test HateMM, MHC-ZH
3. Train MHC-ZH → test HateMM, MHC-EN
4. Train 2 datasets → test held-out third

**Transfer metrics**: macro-F1, AUROC, relative drop from in-domain

### Statistical Testing
- Paired bootstrap (10,000 resamples) for each comparison
- McNemar's test on prediction disagreement
- Approximate randomization test (10,000 permutations) for macro-F1
- Holm-Bonferroni correction for multiple comparisons
- Report: p-value, 95% CI, Cohen's d or Cliff's delta

---

## Risk Mitigation

### Risk 1: No theory beats generic rationale in Phase A
**Action**: Check if theory prompts produce low-variance outputs. Run rationale-only linear probes. If all theories equally weak → prompt quality bottleneck, not theory choice. Reposition as "structured rationale helps, theory-specific structure does not."

### Risk 2: Ceiling effect (94% baseline already high)
**Action**: Move to harder settings — cross-dataset transfer, low-resource (25%/50%/75% train), calibration metrics. Theory-aware models may matter more under distribution shift.

### Risk 3: Theory helps only one dataset
**Action**: Present theory-domain alignment analysis. E.g., ITT better for immigration/national identity content; Attribution better for blame-heavy content.

### Risk 4: LLM outputs unstable
**Action**: Run 3 generations per sample, aggregate by field-wise mean embedding or self-consistency selection.

### Risk 5: Full model overfits
**Action**: Increase dropout to 0.3, weight_decay to 0.05, freeze field projections, reduce hidden to 192.

---

## Success Criteria

### Minimum publishable (workshop / borderline main):
- ≥1 theory beats strongest baseline by +1.5 macro-F1 on 2/3 datasets
- Gains statistically significant (corrected p<0.05) on ≥2 datasets
- ≥4 ablations degrade in expected direction
- Cross-dataset transfer improves by ≥+2.0 macro-F1 average

### Strong main-track:
- Top theory beats generic rationale AND non-rationale baseline on all 3 datasets
- Average gain: +2.0 to +3.0 macro-F1
- Middle module contributes independently (removing it → ≥0.8 macro-F1 drop)
- Theory-aware fusion contributes independently
- Transfer also improves
- Qualitative analysis shows intermediate states align with theory

### Best-case claim:
"Psychology-theory-guided rationale structuring creates more discriminative multimodal representations than generic rationales, and theory-specific inductive bias improves both in-domain detection and cross-dataset robustness."

---

## Timeline

| Week | Task |
|------|------|
| 1 | Finalize prompt schemas, run all Phase A LLM generations, parse & validate |
| 2 | Phase A: encode embeddings, train 150 screening runs, select top 2 theories |
| 3 | Phase B: implement top theory full model, coarse hyperparameter search |
| 4 | Phase B: seed search + retrieval augmentation |
| 5 | Phase C: implement second theory, coarse search |
| 6 | Phase C: seed search + ablations for both theories |
| 7 | Cross-dataset transfer, statistical testing, tables, error analysis |

---

## Experiment Tracker

| Exp ID | Phase | Theory | Dataset | Status | Best Macro-F1 | Notes |
|--------|-------|--------|---------|--------|:-------------:|-------|
| A-ITT-HM | A | ITT | HateMM | TODO | — | |
| A-ITT-EN | A | ITT | MHC-EN | TODO | — | |
| A-ITT-ZH | A | ITT | MHC-ZH | TODO | — | |
| A-IET-HM | A | IET | HateMM | TODO | — | |
| A-IET-EN | A | IET | MHC-EN | TODO | — | |
| A-IET-ZH | A | IET | MHC-ZH | TODO | — | |
| A-ATT-HM | A | ATT | HateMM | TODO | — | |
| A-ATT-EN | A | ATT | MHC-EN | TODO | — | |
| A-ATT-ZH | A | ATT | MHC-ZH | TODO | — | |
| A-SCM-HM | A | SCM | HateMM | TODO | — | |
| A-SCM-EN | A | SCM | MHC-EN | TODO | — | |
| A-SCM-ZH | A | SCM | MHC-ZH | TODO | — | |
| A-GEN-HM | A | Generic | HateMM | TODO | — | Control |
| A-GEN-EN | A | Generic | MHC-EN | TODO | — | Control |
| A-GEN-ZH | A | Generic | MHC-ZH | TODO | — | Control |
