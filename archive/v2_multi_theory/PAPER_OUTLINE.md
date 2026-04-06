# Paper Outline: Theory-Guided Hateful Video Detection via Stereotype Content Model

---

## 1. Introduction

### Para 1: Problem & Importance
- Hateful video detection is critical for online safety
- Videos are inherently multimodal (visual + audio + text transcript)
- Hate expression is diverse: explicit slurs, implicit dehumanization, sarcasm, dogwhistles
- Existing methods rely on surface-level feature matching without understanding the social-psychological mechanism behind hate

### Para 2: Limitation of Existing Approaches
- **Feature-based methods**: Extract multimodal features (CLIP, BERT, audio) → classifier. No structured reasoning about WHY content is hateful.
- **LLM-based methods**: Prompt LLM to directly judge hateful/not. Black-box reasoning, no intermediate decomposition of hate mechanism. Also expensive and hard to control.
- Neither answers: "What social group is targeted? How are they framed? What behavioral harm does this imply?"

### Para 3: Our Insight — Theory-Guided Reasoning
- Social psychology provides structured frameworks for understanding prejudice and discrimination
- **Stereotype Content Model (SCM)** (Fiske et al., 2002): People perceive social groups along two fundamental dimensions:
  - **Warmth**: Is this group friendly, trustworthy, well-intentioned? Or cold, hostile, threatening?
  - **Competence**: Is this group capable, skilled, intelligent? Or incompetent, foolish, primitive?
- The 2×2 combination forms 4 quadrants, each predicting distinct behavioral tendencies:
  - Low warmth + Low competence → **Contempt** → Active harm
  - Low warmth + High competence → **Envy** → Passive exclusion
  - High warmth + Low competence → **Pity** → Patronizing
  - High warmth + High competence → **Admiration** → Respect
- This gives us a theory-driven decomposition: instead of "is this hateful?", ask "how is the target group framed on warmth and competence, and what behavior does this framing endorse?"

### Para 4: Challenges — From Theory to Detection System
- **C1: Theory-Guided Reasoning** — How to make a model reason about video content through the lens of SCM, producing structured intermediate analysis rather than a binary judgment?
- **C2: Theory-Aware Representation** — How to encode the theoretical dimensions (warmth, competence, quadrant) into a computational multimodal representation that preserves the theory's structure?
- **C3: Imbalanced Classification with Theoretical Structure** — Hateful content is the minority class. How to leverage the quadrant structure to improve classification under class imbalance?

### Para 5: Our Method & Contributions
- We propose **[Method Name]**, a theory-guided framework for hateful video detection grounded in SCM
- **Component 1**: Theory-Guided MLLM Reasoning — prompt a multimodal LLM to analyze videos through SCM, producing 5 structured fields (target_group, warmth_evidence, competence_evidence, social_perception, behavioral_tendency) (addresses C1)
- **Component 2**: SCM-Grounded Multimodal Fusion — dual warmth/competence streams → Quadrant Composer → Quadrant Mixture-of-Experts (Q-MoE), where different experts specialize in different quadrant-specific hate patterns (addresses C2)
- **Component 3**: Quadrant-Conditioned Imbalanced Optimization — Quadrant-Entropy Adaptive Label Smoothing (QELS) for uncertain samples + feature compactness regularization for tighter class geometry (addresses C3)
- SOTA on 4 benchmarks: HateMM, MHC-English, MHC-Chinese, ImpliHateVid
- Contributions:
  1. First work to systematically integrate SCM theory into hateful video detection
  2. Q-MoE: quadrant-conditioned expert routing grounded in SCM's behavioral prediction framework
  3. QELS + feature compactness: theory-aware imbalanced learning that uses quadrant uncertainty as a signal
  4. Comprehensive evaluation on 4 datasets across 2 languages

---

## 2. Related Work

### 2.1 Multimodal Hateful Content Detection
- Image-based: hateful meme detection (HatefulMemes, MOMENTA, etc.)
- Video-based: HateMM, MultiHateClip — multimodal fusion approaches
- Limitation: treat hate detection as pattern matching, no social-psychological grounding

### 2.2 LLMs for Content Moderation
- Prompting LLMs for hate speech classification
- Chain-of-thought reasoning for toxicity detection
- Limitation: unstructured reasoning, no theoretical framework guiding the decomposition

### 2.3 Social Psychology in Computational Methods
- SCM in NLP: limited prior work, mostly text-only sentiment/stereotype analysis
- Intergroup theories in hate speech: mostly as post-hoc annotation, not as model architecture design
- Our contribution: first to use SCM as the architectural backbone for a detection pipeline

---

## 3. Method

### 3.1 Overview
- Input: video (frames + audio + transcript)
- Pipeline: MLLM + SCM prompt → structured fields → BERT encoding → SCM-Grounded Fusion → Q-MoE → Binary classification
- Figure: architecture diagram

### 3.2 Theory-Guided MLLM Reasoning (C1)
- **Motivation**: Binary "hateful or not" prompting loses the reasoning process. SCM provides intermediate reasoning steps grounded in social psychology.
- **SCM Prompt Design**: Given video frames + transcript, the MLLM outputs:
  - `target_group`: Which social group is discussed/depicted?
  - `warmth_evidence`: Is the group framed as warm or cold? Evidence?
  - `competence_evidence`: Is the group framed as competent or incompetent? Evidence?
  - `social_perception`: What social perception does this construct? (contempt/envy/pity/admiration)
  - `behavioral_tendency`: What treatment does the video imply/endorse? (active harm / passive exclusion / patronizing / respect / none)
- **Theory Selection**: We screened 5 candidate theories (Generic, ITT, IET, ATT, SCM) across 3 datasets. SCM achieved the best overall performance, validating its fit for this task. (Details in §4.4)
- **Encoding**: Each SCM field text → BERT-base-uncased mean pool → 768d embedding

### 3.3 SCM-Grounded Multimodal Fusion (C2)
- **Motivation**: The 5 SCM fields are NOT independent features to concatenate. Warmth and competence are two orthogonal theoretical dimensions whose interaction determines the quadrant → behavioral tendency. The architecture should preserve this structure.
- **Dual-Stream Processing**:
  - Warmth stream: warmth_evidence + target_group + base_modality_context → warmth representation
  - Competence stream: competence_evidence + target_group + base_modality_context → competence representation
  - Base modalities (text transcript, audio, visual) provide context for interpreting SCM fields
- **Quadrant Composer**: Concatenated warmth + competence representation → 4-class softmax → quadrant probability distribution
  - Learnable quadrant prototypes capture the embedding of each quadrant
  - Harm score derived from quadrant distribution × theory-based harm weights [1.0, 0.7, 0.3, 0.0]
- **Quadrant Mixture-of-Experts (Q-MoE)**:
  - 4 lightweight expert classifiers, one per SCM quadrant
  - Final prediction = soft mixture weighted by quadrant distribution
  - **Motivation**: Different quadrants manifest hate differently — contempt uses direct slurs, envy uses systemic threat framing, pity uses infantilization. Specialized experts capture these distinct patterns.

### 3.4 Quadrant-Conditioned Imbalanced Optimization (C3)
- **Motivation**: Hateful video detection is inherently class-imbalanced (hateful = minority). Standard cross-entropy with fixed class weights is suboptimal because:
  - Not all samples are equally uncertain — quadrant entropy captures this
  - Minority class features are more dispersed in the learned space, causing decision boundary overlap
- **QELS (Quadrant-Entropy Adaptive Label Smoothing)**:
  - Per-sample smoothing: ε_i = ε_min + ε_λ × H(quadrant_dist_i) / log(4)
  - High quadrant entropy → sample is ambiguous on warmth/competence → softer supervision to avoid overfitting noise
- **Feature Compactness Regularization**:
  - Minimize within-class variance in the penultimate (shared) feature space
  - Reduces minority class feature spread → larger effective margin at decision boundary
  - Based on MR2 principle: margin should be proportional to class-wise feature variability (ICLR 2026)

---

## 4. Experiments

### 4.1 Experimental Setup
- **Datasets**: HateMM (binary, English), MultiHateClip-English (3-class→binary), MultiHateClip-Chinese (3-class→binary), ImpliHateVid (binary, English)
- **Metrics**: Accuracy, Macro-F1, Macro-Precision, Macro-Recall
- **MLLM**: gpt-5.4-nano (frozen, inference only)
- **Text Encoder**: BERT-base-uncased (frozen, mean pool)
- **Training**: AdamW, lr=2e-4, 45 epochs, cosine schedule, EMA (decay=0.999), class weight [1.0, 1.5]
- **Baselines**: [list of comparison methods]

### 4.2 Main Results
- **Claim: Our framework achieves competitive or SOTA performance on all 4 datasets**
- Main comparison table: our method vs baselines (report best of 20 random seeds, following prior work)
- Our results: HateMM=92.6, MHClip-Y=85.3, MHClip-B=89.8, ImpliHateVid=93.5
- Report ACC, M-F1, M-P, M-R
- Improvements are consistent across English and Chinese, explicit and implicit hate

### 4.3 Ablation Study
- **Claim: Each component and its theory-grounded design contributes positively**
- Key principle: not just "remove X" but also **"replace theory-guided X with non-theory X"** to isolate the contribution of SCM theory vs general capacity

**4.3.1 Component 1: Theory-Guided MLLM Reasoning**
| Variant | Description | Tests |
|---------|-------------|-------|
| w/o SCM branch | Base modalities only (text+audio+frame) | Is LLM reasoning needed at all? |
| Generic prompt | Same MLLM, same number of fields, but non-SCM structured rationale (e.g., "summarize content, identify targets, assess tone, evaluate harm, judge intent") | Is the gain from SCM theory or just extra structured text? |
| w/o individual fields | Remove one SCM field at a time (×5) | Which fields are essential? |

**4.3.2 Component 2: SCM-Grounded Fusion**
| Variant | Description | Tests |
|---------|-------------|-------|
| Flat concat | All 5 fields concatenated, no warmth/competence separation | Is the dual-stream structure needed? |
| w/o Quadrant Composer | No quadrant routing, direct concat → classifier | Is quadrant decomposition needed? |
| Unconstrained MoE | 4 experts with learned gating (not quadrant-conditioned), matched parameters | Is the gain from SCM-guided routing or just more experts? |
| Single expert (matched params) | One classifier with same total parameters as Q-MoE | Is expert specialization needed? |

**4.3.3 Component 3: Imbalanced Optimization**
| Variant | Description | Tests |
|---------|-------------|-------|
| w/o QELS | Fixed label smoothing (ε=0.1) | Does quadrant-conditioned smoothing help? |
| w/o MR2 | Standard weighted CE only | Does compactness regularization help? |
| w/o both | No QELS, no MR2 | Joint contribution |
| Focal loss | Replace QELS with focal (γ=2.0) | Is QELS better than generic hard-example mining? |

### 4.4 Theory Consistency Analysis
- **Claim: The learned model behaves consistently with SCM's theoretical predictions**

**(a) Quadrant Distribution by Label**
- Hateful vs non-hateful videos: do they cluster in different SCM quadrants?
- Expectation: hateful → low warmth quadrants (contempt, envy); non-hateful → high warmth (admiration, pity)

**(b) Expert Specialization**
- Per-expert utilization rate and accuracy
- Do experts learn distinct decision patterns per quadrant as SCM predicts?
- Visualize expert weight distribution for different hate types

**(c) Performance by Quadrant and Router Entropy**
- Accuracy breakdown by dominant quadrant assignment
- Does low router entropy (confident quadrant) correlate with higher accuracy?

**(d) Implicit vs Explicit Hate (if split available)**
- Does the SCM framework help more on implicit hate cases?
- Compare SCM-guided vs generic-prompt performance specifically on implicit subset

### 4.5 SCM Extraction Quality
- **Claim: The MLLM produces reliable SCM analysis**
- Human evaluation on a sampled subset (~100 videos): accuracy of target_group extraction, agreement on warmth/competence framing
- Or: automated consistency check — does the model's social_perception match the warmth×competence quadrant?
- Supports the validity of Component 1 as a trustworthy reasoning stage

### 4.6 Interpretability & Case Study
- **Claim: Theory-guided reasoning produces interpretable intermediate outputs**
- Side-by-side SCM field outputs for hateful vs non-hateful videos
- Show how warmth/competence framing captures implicit hate strategies (e.g., infantilization as high-warmth/low-competence = pity quadrant)
- Contrast with generic prompt outputs — SCM fields are more structured and actionable

### Appendix
- Error analysis: categorize failures by pipeline stage (LLM error, boundary ambiguity, annotation noise)
- Hyperparameter sensitivity: MR2 α/β, QELS ε_λ across datasets
- Full per-seed distribution plots

---

## 5. Conclusion
- First systematic integration of Stereotype Content Model into hateful video detection
- Theory-guided MLLM reasoning produces interpretable intermediate analysis
- SCM-grounded architecture (dual-stream + Q-MoE) preserves theoretical structure in the computational pipeline
- Quadrant-conditioned optimization (QELS + compactness) leverages theory for imbalanced learning
- SOTA on 4 benchmarks across 2 languages
- **Limitation**: Depends on MLLM reasoning quality; SCM is one of several possible social psychology theories
- **Future work**: End-to-end training, exploring other theories, extending to more languages
