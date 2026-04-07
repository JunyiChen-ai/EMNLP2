# EMNLP2 Project: Complete Constraints & Killed Directions

**Last updated**: 2026-04-05
**Source**: 100-round brainstorm + 20-round Session 2 + 30-round Session 3 + user feedback

---

## I. Hard Constraints (User-Imposed, Non-Negotiable)

### Method Constraints
1. **NO human annotation dependency** — all methods must be fully automatic, end-to-end
2. **NO "instruct LLM with different prompts to ensemble/aggregate decisions"** — explicitly forbidden
3. **NO "provide different augmented views from different prompts"** — explicitly forbidden
4. **NO resources/insights from `/home/junyi/EMNLP2/archive/`** — that folder is dead, do not reference
5. **NO upgrading LLM model** — user has budget constraints (feedback_no_model_change.md)
6. **NO cross-prompt ensembling** — methods must be in ONE unified framework, not combining outputs from different prompt versions (feedback_no_cross_prompt.md)
7. **Solution must be NOVEL and ELEGANT** — not a bag of tricks, one core principle
8. **Must be a genuine multimodal method** — cannot just operate on a single text embedding and ignore other modalities. "只在MLLM文字embedding上做减法" is NOT acceptable as a multimodal video paper

### Quality Constraints
9. **Only CCF-A venues for references** — use latest papers, concise/scientific prompts (feedback_research_quality.md)
10. **Search general methods, not task-specific** — apply general ML advances to our domain for novelty (feedback_search_strategy.md)
11. **High novelty standard** — user rejects incremental work, wants ideas that are clearly differentiated from existing literature (feedback_idea_quality.md)

### Budget & Infrastructure Constraints
12. **Conda env: HVGuard**, max_concurrent=10 (feedback_dev_prefs.md)
13. **~$20-100 budget** for MLLM API calls (batch API)
14. **~2 months timeline** to EMNLP 2026 deadline
15. **4 datasets available**: HateMM, MultiHateClip-EN, MultiHateClip-ZH, ImpliHateVid
16. **Existing assets**: MLLM pipeline, 5 prompt outputs, BERT embeddings, base modality features (audio/frame), classifiers (MLP, MoE, retrieval)

---

## II. Empirical Facts (Cannot Be Contradicted)

These are established experimental results that any proposed method must be consistent with:

1. **MLLM structured text is the dominant feature**: +6-8pp over no rationale
2. **Generic prompt ≈ theory-specific prompt**: SCM, ITT, IET, ATT add nothing
3. **Base modalities (audio/frame) add only 0.7-3.2pp** on top of MLLM text
4. **MLLM direct classification is poor**: 62-84 F1
5. **Simple MLP on MLLM text ≈ full complex model**: ~87-88 F1
6. **Seed variance is huge**: max-mean gap 5-8pp across random seeds
7. **Retrieval (kNN) on MLLM text embeddings adds only ~2.4 F1**
8. **Gap between training-free (~76 F1) and supervised (~87-97 F1)**: ~15-20pp room
9. **All 3 accuracy targets reached on 2026-03-17** via feature geometry optimization (project_targets_reached.md)

### Implication
Any new method that adds complexity but doesn't substantially outperform "simple MLP on generic MLLM text" has no contribution. The bar is high.

---

## III. Killed Directions (30+ Methods, Do NOT Revisit)

### From 100-Round Brainstorm
1. **Adversarial/dual-hypothesis** — = RAMF (2025), already published
2. **Hand-designed prompt fields** (stance, endorsement, gap_type, target, abuse_type, confidence) — "dirty prompting", no principled basis
3. **Temporal localization + multiple losses** — bag of tricks, not unified
4. **ALARM minor adaptation** (just changing similarity metric) — incremental over KDD 2026
5. **Intra-video self-contrast** — not universal (short videos, uniform hate/benign)
6. **Corpus-level schema/pattern mining** — lossy compression of already-strong embeddings
7. **Theory-guided methods** (SCM/ITT) — don't consistently beat generic prompt (empirical fact)
8. **Counterfactual generation** — unrealistic for video
9. **Debate/courtroom agents** — = ARCADE (2026), efficiency problems
10. **Training-free relying on MLLM classification** — ceiling too low (62-84 F1)
11. **Multi-prompt distillation alone** — not novel enough
12. **Reliability-aware fusion** — base modalities too weak as fallback (+0.7-3.2pp)
13. **Error-focused two-regime learning** — too few hard samples (~50-100) to train specialist
14. **Latent discourse structure over rationales** — lossy compression of 150-token text
15. **Any "structured rationale" with hand-designed fields** — still dirty
16. **Narrative state transition (multi-call)** — 7-29 calls per video, too expensive
17. **Cross-channel commitment** — too expensive, weak signal
18. **Prefix-based ambiguity resolution** — too expensive, noisy on short videos
19. **Entity graph / target grounding** — hand-designed schema
20. **Factorized privileged distillation** — pragmatic residual concept not well-defined
21. **NormGraph distillation** — too ontology-dependent for 2 months
22. **Weakness-aware teacher routing** — bag of tricks, not unified
23. **Debate-with-memory** — = ARCADE variant
24. **Modality reliability court** — circular (MLLM judges own reliability)
25. **Retrieval-by-implication** — too close to MoRE (WWW 2025)
26. **Self-contradiction probing** — = RAMF
27. **Prototype games** — reduces to contrastive learning
28. **Cross-lingual semantic anchoring** — dataset too small for ZH
29. **Temporal Consensus Distillation (TCD)** — incremental, 2-7x more expensive than MARS
30. **Pure analysis "MLLM text is all you need"** — not surprising enough in 2026 alone
31. **Evidence taxonomy requiring human annotation** — user explicitly rejected

### From 20-Round Session 2
32. **TVRM (Transfer-Validated Reference Memory)** — mock review 3/5, "just retrieval with extra steps", novelty fragile
33. **Cross-View Consistency as backup** — violates "no multi-prompt ensemble" constraint

### From 30-Round Session 3
34. **Latent Social Relation Bottleneck (LSRB)** — = concept bottleneck models (Koh et al. 2020) / information bottleneck / slot attention applied to text
35. **Comparative Social Calibration (CSC)** — = supervised contrastive learning (SupCon, Khosla et al. 2020) / metric learning
36. **Semantic Ambiguity Calibration (SAC)** — = neighborhood-based label noise correction / label smoothing, published
37. **Stability regularization via graph smoothing** — = graph Laplacian regularization (Zhu et al. 2003, Kipf & Welling 2017)
38. **Foil Subtraction / NOR on single-modality text embedding** — user killed: "latent embedding只用一个模块" is not acceptable for a multimodal video paper. Operating only on MLLM text embedding while ignoring other modalities is too reductive.

---

## IV. Published Competitive Landscape (Cannot Replicate These)

| Paper | Venue | Key Idea |
|-------|-------|----------|
| HVGuard | EMNLP 2025 | MLLM CoT + MoE for video |
| MoRE | WWW 2025 | Modality experts + retrieval |
| RAMF | 2025 | Adversarial multi-hypothesis |
| ImpliHateVid | ACL 2025 | Implicit hate video benchmark (97.58 acc HateMM) |
| ALARM | KDD 2026 | Self-improvement moderation (memes) |
| MARS | ICASSP 2026 | Training-free adversarial reasoning (4 calls, ~75.8 F1) |
| ARCADE | 2026 | Debate agents |
| TANDEM | 2026 | - |
| MIND | ACL 2025 | Zero-shot harmful meme via retrieval + debate |
| LELA | 2026 | Localization |
| WWW 2025 meme-to-video transfer | arXiv:2501.15438 | - |

---

## V. What a Valid New Idea Must Satisfy

A proposed method MUST pass ALL of these checks:

1. **Not reducible to any single published technique** — if a reviewer can say "this is just X applied to Y", it's dead
2. **Genuinely multimodal** — must involve multiple modalities in a meaningful way, not just text embeddings
3. **Naturally follows from empirical insights** — not bolted-on complexity
4. **One core principle** — elegant, not a bag of tricks
5. **No human annotation** — fully automatic
6. **No multi-prompt ensemble/aggregation** — unified framework
7. **Implementable in ~2 months** with existing infrastructure
8. **Addresses a real bottleneck** — either accuracy, stability, transfer, or interpretability
9. **Competitive** — must at minimum match "simple MLP on MLLM text" while adding a new dimension of value
10. **Novel enough for EMNLP main track** — not just a feature trick or preprocessing step

---

## VI. MLLM Prompt Landscape (Existing Methods — Do NOT Replicate)

### A. HVGuard (EMNLP 2025) — 3-Step CoT + MoE

**Full Architecture**:
1. **Multimodal Extraction**: Audio → FunASR → transcript + emotion cues; Video → 32 uniformly sampled frames; Text = title + transcript
2. **MLLM Reasoning Module** (GPT-4o as default MLLM, 3-step CoT):
   - **Step 1 — Adaptation Prompt (system)**: "This is a video that may contain harmful content, such as hate speech, explicit violence, discrimination, or other forms of harmful behavior. You are a content moderation specialist. Your task is to identify any instances of hate speech, violent imagery, discriminatory actions, or any other content that could be considered harmful, abusive, or offensive. Ensure the answer's accuracy while keeping it concise and avoiding over-explanation."
   - **Step 2 — Visual Meaning Understanding**: "Describe the video content based on {video frames}, ignoring subtitles in the frames. Pay attention to any special characters or scenes." → res1
   - **Step 3 — Textual Meaning Understanding**: "The video title is {video title}. The text in the video is {video transcript}. Please analyze the meaning of this text. Note that there may be homophonic memes and puns; distinguish and explain them." → res2
   - **Step 4 — Fusion Meaning Understanding**: "Please combine the {video title}, {video transcript}, {video frames}, {voice emotion}, {response1}, {response2} and analyze both the visual, textual and audio elements of the video to detect and flag any hateful content. No need to describe the content of the video, only answer implicit meanings and whether this video expresses hateful content further." → v_M (final rationale)
3. **Multimodal Fusion Module**:
   - Modality-specific encoders: XLM (text), Wav2Vec (audio), ViT (vision)
   - Rationale encoded by same text encoder: E_M = f_T(v_M)
   - Concat all: E_i = concat(E_T, E_A, E_F, E_M) — 4 modality embeddings
   - MoE: 8 identical expert networks + 1 gating network with dropout
   - Gating: w_k = Dropout(softmax(g_k(E_i; φ)))
   - Output: O_fusion = Σ w_k · O_k → cross-entropy loss
4. **Results**: HateMM binary ACC 85.63, M-F1 84.79; MHClip-EN binary ACC 85.39, M-F1 77.14
5. **Ablation**: w/o CoT → ACC 79.21, M-F1 55.12 (huge drop); MoE→MLP → M-F1 74.66; MoE→Cross-attn → M-F1 80.37

**Prompt pattern**: Describe visual → Describe textual → Fuse & judge. Sequential CoT, no adversarial reasoning, no explicit evidence decomposition.

### B. RAMF (arXiv 2512.02743, 2025) — 3-Stage Adversarial Reasoning + LGCF + SCA

**Full Architecture**:
1. **VLM Adversarial Reasoning** (Qwen2.5-VL-32B, 16 sampled frames, temp 0.7):
   - **Stage 1 — Objective Description (T_O)**: Factual visual/textual observations without interpretive judgment
   - **Stage 2 — Hate-Assumed Inference (T_H)**: Assume content contains hate; extract discriminatory expressions targeting specific groups with contextual evidence and visual grounding
   - **Stage 3 — Non-Hate-Assumed Inference (T_N)**: Assume content lacks hate; explore alternative interpretations (artistic expression, satire, personal conflict) with supporting evidence
2. **Two-Layer Fusion**:
   - **Layer 1**: LGCF processes {T, A, V, T_O} independently → SCA cross-attention fusion → Y_1
     - LGCF: Local (Conv1D kernel=3) + Global (AdaptiveAvgPool) with learned gating g = σ(W[local ⊕ global] + b)
     - SCA: Cross-head convolution (2D conv across attention heads) + Structural mixing convolution (odd-even head interleaving)
   - **Layer 2**: SCA fuses {Y_1, T_H, T_N} → Y_2 → AvgPool → MLP {128, 64, 2}
3. **Results**: HateMM M-F1 83.7, ACC 84.3; 3% M-F1 and 7% hate-recall improvement over prior SOTA

**Prompt pattern**: Objective → Pro-hate → Anti-hate. Adversarial multi-hypothesis, no diagnostic/evidence decomposition.

### C. MARS (arXiv 2601.15115, 2026) — 4-Stage Adversarial Reasoning (Training-Free)

**Full Architecture**:
1. **4-Stage MLLM Reasoning** (Qwen2.5-VL-32B / GPT5-mini / Gemini2.5-Flash):
   - **Stage 1 — Objective Representation (P^obj)**: Factual video description without interpretation
   - **Stage 2 — Hate Hypothesis (P^hate)**: Under assumption "content contains hate", extract hate-supporting evidence + confidence score
   - **Stage 3 — Non-Hate Hypothesis (P^non)**: Under assumption "content is non-hateful", collect counter-evidence + confidence
   - **Stage 4 — Meta-Synthesis (P^meta)**: Integrate competing hypotheses via structured meta-analysis, weigh evidence strength and contextual relevance
2. **Training-Free**: No downstream classifier. Final decision from synthesis function Ψ → (predicted label, confidence, key factors, rationale)
3. **Results**: HateMM ACC 75.8, M-F1 75.8 (much lower than supervised methods)

**Prompt pattern**: Objective → Pro-hate → Anti-hate → Meta-synthesis. Extended version of RAMF's adversarial pattern, no training.

### D. Our Current Pipeline (Baseline)

- **MLLM**: Qwen3-VL-32B-Instruct (DP=2)
- **Input**: Raw video (not sampled frames) + transcript + title
- **Current prompt**: Generic CoT (similar to HVGuard pattern but with raw video input)
- **Encoding**: BERT [CLS] on rationale → [768]; WavLM → audio [768]; ResNet/ViT → frame [768]
- **Fusion**: MLP or concat → MLP
- **Result**: ~87% ACC text-only MLP baseline

### Summary of Prompt Landscape Gaps

| Method | Prompt Strategy | Evidence Structure | Modality Grounding | Observation/Inference Separation |
|--------|----------------|-------------------|-------------------|--------------------------------|
| HVGuard | Sequential CoT (describe→analyze→fuse) | None (free-form) | Weak (ignores subtitles) | NO |
| RAMF | Adversarial (objective→pro→anti) | None (free-form per hypothesis) | Weak (16 frames) | NO |
| MARS | Adversarial + meta-synthesis | Confidence scores only | Weak (frames) | NO |
| **Gap** | **Diagnostic / evidence-decomposed** | **Structured evidence dimensions** | **Strong (raw video)** | **YES** |

**Key observation**: ALL existing methods produce free-form rationales without separating grounded observations from inferential judgments. None explicitly structure the MLLM's reasoning around evidence dimensions (temporal, cross-modal, implicit/explicit). This is the prompt-level gap we can exploit.

---

## VIII. Open Insights Worth Building On

These insights from 150+ rounds of discussion are still valid and usable:

1. **Stability is an underexplored evaluation axis** — seed variance 5-8pp is huge, nobody studies this for hate video
2. **The bottleneck is boundary fitting, not representation** — MLP ≈ complex model, generic ≈ theory prompt
3. **Hate is relational** — target-act-stance structure, contextual polarity flips (quotation, counterspeech, satire, reclaimed language)
4. **MLLM textualization as semantic interface** — video multimodal content becomes compressible into text, but something is lost in that compression
5. **Paradigm shifts 2025-2026**: static classifiers → adaptive systems, passive encoding → active evidence gathering
6. **Cross-dataset transfer** as high-value evaluation dimension
7. **Legitimate uncertainty vs arbitrary instability** — important nuance for moderation systems
