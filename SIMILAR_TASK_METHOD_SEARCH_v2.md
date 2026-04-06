# Similar-Task Method Search v2: Broad Categorized Inventory

**Date**: 2026-04-06 (updated)
**Target Task**: Hateful video detection via frozen MLLM textualization pipeline
**Paper Mode**: hybrid
**Thread ID**: 019d5da2-8dda-7f51-be79-a089d602bc60

---

## Search Setup
- **target task**: Hateful video detection. Pipeline: frozen MLLM (gpt-5.4-nano) → text rationale → BERT encode + AV features → classifier
- **observed bottleneck**: MLLM text dominates (+6-8pp) but is brittle (5-8pp seed variance), lossy (omits tone/symbols/context), AV adds only 0.7-3.2pp
- **similarity axes**:
  1. Pipeline structure (frozen LLM/VLM → text → classifier)
  2. Decision bottleneck (text-dominant but needs multimodal grounding)
  3. Supervision setting (small datasets, binary, noisy labels)
  4. Robustness (seed instability, cross-domain transfer)
  5. Input structure (video = visual + audio + text)
  6. Failure modes (sarcasm, implicit meaning, context-dependent polarity)
  7. **[NEW]** Prompt sensitivity (small prompt changes → large accuracy swings)

---

## FINAL CANDIDATE INVENTORY

### Tier 1: Strong Transfer (17 candidates)

---

**B1.** "FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models"
- Jing et al., **EMNLP 2024 Findings**
- paper: https://aclanthology.org/2024.findings-emnlp.290/
- category: Hallucination / Grounding Verification
- source task: faithfulness evaluation of VLM-generated text against visual evidence
- code: **yes** — https://github.com/bcdnlp/FAITHSCORE
- pipeline:
  1. VLM generates free-form caption/description of image
  2. LLM decomposes caption into atomic factual claims (sub-sentence level)
  3. For each claim, generate a verification question
  4. VQA model answers the question against the original image
  5. Compare VQA answer with claim → supported / not supported
  6. Aggregate per-claim scores → overall faithfulness score
- transfer value: claim decomposition + evidence verification directly targets rationale brittleness

---

**B3.** "Knowledge-Centric Hallucination Detection" (RefChecker)
- **EMNLP 2024**
- paper: https://aclanthology.org/2024.emnlp-main.395/
- category: Hallucination / Grounding Verification
- source task: claim-triplet decomposition for detecting hallucinations in LLM outputs
- code: unclear
- pipeline:
  1. LLM generates response text
  2. Extract claim-triplets (subject, predicate, object) from response
  3. For each triplet, retrieve reference knowledge (KB or context)
  4. Compare triplet against reference → entailed / contradicted / neutral
  5. Aggregate triplet-level verdicts → hallucination score
- transfer value: structured claim decomposition makes rationale checkable against multimodal evidence

---

**B7.** "HalLoc: Token-level Localization of Hallucinations for Vision Language Models" **[NEW]**
- Park et al., **CVPR 2025**
- paper: https://openaccess.thecvf.com/content/CVPR2025/html/Park_HalLoc_Token-level_Localization_of_Hallucinations_for_Vision_Language_Models_CVPR_2025_paper.html
- category: Hallucination / Grounding Verification
- source task: real-time token-level hallucination detection during VLM generation
- code: **yes** — https://github.com/dbsltm/cvpr25_halloc
- pipeline:
  1. VLM generates response (caption, VQA answer, instruction-following)
  2. Lightweight add-on model runs concurrently during generation
  3. For each generated token, predict hallucination probability
  4. Localize tokens with high hallucination probability in real time
  5. Flag unreliable content → user/system alert
- transfer value: token-level confidence layer over MLLM rationale; can flag unreliable claims before they reach classifier

---

**B8.** "MARINE: Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance" **[NEW]**
- Zhao et al., **ICML 2025 Spotlight**
- paper: https://arxiv.org/abs/2402.08680
- category: Hallucination / Grounding Verification
- source task: training-free hallucination mitigation for frozen LVLMs
- code: **yes** — https://github.com/Linxi-ZHAO/MARINE
- pipeline:
  1. DETR extracts object-level visual detections from image
  2. RAM++ generates image tags as additional visual evidence
  3. Combine detections + tags → object-level visual guidance
  4. During LVLM decoding, inject image-grounded guidance to steer generation
  5. Output: grounded descriptions with reduced hallucination (no model fine-tuning)
- transfer value: training-free grounding guidance for frozen MLLM; directly applicable to our frozen textualization pipeline to reduce omissions and hallucinations

---

**F1.** "MultiClimate: Multimodal Stance Detection on Climate Change Videos"
- Wang et al., **EMNLP 2024 Workshop (NLP4PI)**
- paper: https://aclanthology.org/2024.nlp4pi-1.27/
- category: Stance Detection
- source task: multimodal video stance detection (endorse / oppose / neutral)
- code: **yes** — https://github.com/werywjw/MultiClimate
- pipeline:
  1. Extract video frames at fixed intervals
  2. Extract transcript via ASR, align with frames → frame-transcript pairs
  3. Encode text with BERT, encode frames with ResNet50/ViT
  4. Fuse text + image features (concatenation / cross-attention)
  5. Classify stance: support / oppose / neutral toward climate claims
- transfer value: "reporting hate" vs "endorsing hate" IS a stance problem; video-native

---

**I3.** "MoReVQA: Exploring Modular Reasoning Models for Video Question Answering"
- Min et al., **CVPR 2024**
- paper: https://openaccess.thecvf.com/content/CVPR2024/html/Min_MoReVQA_Exploring_Modular_Reasoning_Models_for_Video_Question_Answering_CVPR_2024_paper.html
- category: Grounded Reasoning
- source task: modular video QA with plan-ground-reason pipeline and external memory
- code: unclear
- pipeline:
  1. Frozen VLM generates per-frame captions → store in external memory
  2. Frozen LLM decomposes question into sub-queries (planning)
  3. For each sub-query, retrieve relevant captions from memory (grounding)
  4. Frozen LLM reasons over retrieved evidence to answer sub-queries
  5. Frozen LLM synthesizes sub-answers into final answer (reasoning)
- transfer value: modular plan → ground → reason structure attacks lossy ungrounded textualization

---

**K1.** "Cracking the Code: Enhancing Implicit Hate Speech Detection Through Coding Classification"
- **2025**
- paper: https://aclanthology.org/2025.trustnlp-main.9/
- category: Implicit Hate / Coded Language
- source task: detecting coded hate via classification of coding strategies
- code: likely yes
- pipeline:
  1. Input text (tweet, post, transcript)
  2. Classify coding strategy: sarcasm / insinuation / metaphor / euphemism / dog whistle / direct
  3. Encode text with language model
  4. Concatenate text embedding + coding-strategy features
  5. Classify: hateful / not hateful
- transfer value: predict HOW hate is encoded as auxiliary features; targets implicit-hate failure mode

---

**K2.** "Specializing General-Purpose LLM Embeddings for Implicit Hate Speech Detection"
- Cheremetiev et al., **ACM DHOW Workshop 2025**
- paper: https://publications.idiap.ch/publications/show/5671
- category: Implicit Hate / Coded Language
- source task: adapting LLM embeddings specifically for implicit hate
- code: **yes** — https://github.com/idiap/implicit-hsd
- pipeline:
  1. Take general-purpose LLM (e.g., Llama, Mistral)
  2. Extract sentence embeddings from last hidden states
  3. Fine-tune embedding space via contrastive or classification objective on implicit-hate data
  4. Use specialized embeddings as features → linear probe classifier
- transfer value: specialized text embeddings for implicit hate; drop-in replacement or teacher for BERT

---

**K3.** "ImpliHateVid: A Benchmark Dataset and Two-stage Contrastive Learning Framework for Implicit Hate Speech Detection in Videos" **[NEW]**
- Rehman et al., **ACL 2025**
- paper: https://aclanthology.org/2025.acl-long.842/
- category: Implicit Hate / Video Hate Detection
- source task: implicit hate speech detection in videos (2009 videos: 509 implicit, 500 explicit, 1000 non-hate)
- code: **yes** — https://github.com/videohatespeech/Implicit_Video_Hate
- pipeline:
  1. Extract modality-specific features: audio (wav2vec), text (transcript), image (keyframes)
  2. Stage 1: Train modality-specific encoders using contrastive loss on concatenated features
  3. Stage 2: Train cross-encoders via contrastive learning to refine multimodal representations
  4. Incorporate auxiliary features: sentiment, emotion, caption-based features
  5. Classify: implicit hate / explicit hate / non-hate
- transfer value: directly addresses implicit hate in video; two-stage contrastive learning for multimodal fusion; auxiliary affect features complement our rationale-based pipeline

---

**K4.** "MM-HSD: Multi-Modal Hate Speech Detection in Videos" **[NEW]**
- Céspedes-Sarrias et al., **ACM Multimedia 2025**
- paper: https://dl.acm.org/doi/10.1145/3746027.3754558
- category: Implicit Hate / Video Hate Detection
- source task: multi-modal hate speech detection in videos using cross-modal attention
- code: **yes** — https://github.com/idiap/mm-hsd
- pipeline:
  1. Extract video frames, audio features, ASR transcript, and on-screen text (OCR)
  2. Encode each modality independently (ViT/ResNet for video, wav2vec for audio, BERT for text)
  3. Cross-Modal Attention (CMA): on-screen text as query, other modalities as key/value
  4. Concatenate transcript + audio + video + on-screen text + CMA features
  5. Classify: hateful / not hateful (SOTA on HateMM: M-F1 = 0.874)
- transfer value: SOTA competitor on our target dataset (HateMM); shows OCR + CMA matters; directly comparable baseline

---

**M1.** "Describe What You See with Multimodal Large Language Models to Enhance Video Recommendations"
- Spotify Research, **ICLR 2026**
- paper: https://openreview.net/pdf?id=MiV3WXDYJb
- category: MLLM Description as Features
- source task: video recommendation using frozen MLLM descriptions
- code: no
- pipeline:
  1. Sample frames from video
  2. Frozen MLLM (e.g., GPT-4V) generates text description of video content
  3. Text encoder (e.g., sentence-transformers) encodes description → text embedding
  4. Text embedding replaces or augments traditional video features
  5. Feed into standard recommender model (SASRec / two-tower retrieval)
- transfer value: closest positive precedent for our entire textualization recipe; +4-18% gains

---

**P1.** "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"
- BehnamGhader et al., **ICLR 2024**
- paper: https://arxiv.org/abs/2404.05961
- category: Text Encoding
- source task: converting frozen decoder-only LLM into bidirectional sentence encoder
- code: **yes** — https://github.com/McGill-NLP/llm2vec
- pipeline:
  1. Take frozen decoder-only LLM (Llama, Mistral, etc.)
  2. Enable bidirectional attention (remove causal mask)
  3. Masked next token prediction (lightweight adaptation)
  4. Unsupervised contrastive learning on text pairs
  5. Output: strong sentence embeddings for downstream classification via linear probe
- transfer value: replace BERT with stronger rationale encoder; may capture reasoning structure better

---

**M4.** "Harnessing Large Language Models for Training-free Video Anomaly Detection" (LAVAD)
- Zanella et al., **CVPR 2024**
- paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf
- category: MLLM Description as Features / Pipeline
- source task: training-free video anomaly detection
- code: **yes** — https://github.com/lucazanella/lavad
- pipeline:
  1. Sample keyframes from video at temporal intervals
  2. Frozen VLM generates text description per keyframe
  3. Clean descriptions via image-text similarity filtering (remove hallucinated content)
  4. CLIP text encoder encodes cleaned descriptions → temporal sequence of text embeddings
  5. Compute text-video alignment scores per temporal window
  6. Aggregate temporal scores → anomaly detection (no training)
- transfer value: closest structural pipeline match; description quality and scoring quality are separable

---

**NEW2.** "Does VLM Classification Benefit from LLM Description Semantics?" (DisCLIP)
- **AAAI 2025**
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/32638
- category: Methodology / Pipeline Analysis
- source task: testing whether LLM descriptions improve VLM zero-shot classification
- code: unclear
- pipeline:
  1. LLM generates multiple class-specific text descriptions
  2. Select discriminative descriptions (quality > quantity)
  3. CLIP text encoder embeds selected descriptions
  4. CLIP image encoder embeds input image
  5. Compute image-description similarity → zero-shot classification
- transfer value: description quality > quantity; supports our "generic ≈ structured prompt" finding

---

**APO1.** "TextGrad: Automatic 'Differentiation' via Text" **[NEW]**
- Yuksekgonul et al., **Nature / ICML 2025**
- paper: https://arxiv.org/abs/2406.07496
- category: Automatic Prompt Optimization
- source task: end-to-end optimization of AI systems via textual gradients
- code: **yes** — https://github.com/zou-group/textgrad
- pipeline:
  1. Define AI system as computation graph with text variables (prompts, outputs)
  2. Run forward pass: input → frozen LLM → output → evaluation
  3. Evaluator LLM generates textual feedback ("gradients") on output quality
  4. Backpropagate textual gradients through graph to update prompt variables
  5. Iterate until convergence on task metric
- transfer value: directly optimizes our MLLM prompt as a trainable variable against downstream F1 + stability; black-box friendly, no model access needed

---

**APO2.** "VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models" **[NEW — also in M5]**
- Ye et al., **CVPR 2025**
- paper: https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.html
- category: Automatic Prompt Optimization / MLLM Pipeline
- source task: optimizing VLM guiding questions for video anomaly detection
- code: **yes** — https://github.com/vera-framework/VERA
- pipeline:
  1. Frozen VLM processes video frames
  2. Define guiding questions as learnable parameters (e.g., "what is unusual?")
  3. Learner VLM answers questions → generates verbalized description
  4. Evaluate answers against coarsely labeled training data
  5. Optimizer VLM provides verbal feedback on question quality
  6. Update guiding questions based on verbal feedback → iterate
- transfer value: **closest domain match**: video + frozen VLM + prompt-as-learnable-parameter; directly applicable to optimizing our hate-detection guiding questions

---

**APO3.** "Trace is the Next AutoDiff: Generative Optimization with Rich Feedback, Execution Traces, and LLMs" **[NEW]**
- Cheng et al., **NeurIPS 2024**
- paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/83ba7056bce2c3c3c27e17397cf3e1f0-Paper-Conference.pdf
- category: Automatic Prompt Optimization
- source task: end-to-end optimization of arbitrary AI workflows
- code: **yes** — https://github.com/microsoft/Trace
- pipeline:
  1. Instrument AI workflow as traceable computation graph (PyTorch-like API)
  2. Mark prompts, schemas, hyperparameters as trainable variables
  3. Run forward pass → capture execution trace + rich feedback (numerical, textual, errors)
  4. OptoPrime optimizer (LLM-based) reads trace + feedback → proposes parameter updates
  5. Iterate until convergence
- transfer value: optimizes not just prompt but entire pipeline (prompt + output schema + rationale post-processing); captures execution failures as direct optimization signal

---

### Tier 2: Moderate Transfer (16 candidates)

---

**A2.** "Classifier-Guided Gradient Modulation for Enhanced Multimodal Learning"
- **NeurIPS 2024**
- paper: https://openreview.net/forum?id=oe5ZEqTOaz
- category: Modality Imbalance / Fusion
- source task: balancing multimodal learning under modality dominance
- code: unclear
- pipeline:
  1. Standard multimodal encoder (text + image/audio branches)
  2. During training, monitor per-modality gradient contributions via classifier signals
  3. Adaptively modulate gradient magnitude AND direction for each modality
  4. Weaker modalities receive amplified gradients; dominant modality dampened
  5. Standard multimodal classifier on fused features
- transfer value: anti-dominance balancing when text overwhelms AV

---

**A4.** "A Closer Look at Multimodal Representation Collapse" **[NEW]**
- Chaudhuri et al., **ICML 2025 Spotlight**
- paper: https://arxiv.org/abs/2505.22483
- project: https://abhrac.github.io/mmcollapse/
- category: Modality Imbalance / Fusion
- source task: understanding and preventing modality collapse in multimodal fusion
- code: **yes** (project page)
- pipeline:
  1. Standard multimodal encoder with shared fusion head
  2. Diagnose: noisy features from one modality entangle with predictive features from another via shared neurons
  3. Result: dominant modality suppresses weaker modality (collapse)
  4. Solution: cross-modal knowledge distillation → frees rank bottlenecks in student encoder
  5. Alternative: explicit basis reallocation algorithm to prevent collapse
- transfer value: directly explains why AV adds only 0.7-3.2pp despite being informative; cross-modal KD or basis reallocation could unlock suppressed AV signal

---

**D1.** "Improving Explainable Fact-Checking with Claim-Evidence Interaction" (CorXFact)
- **COLING 2025**
- paper: https://aclanthology.org/2025.coling-main.108/
- category: Evidence Quality / Claim Verification
- source task: explicit claim-evidence interaction modeling
- code: unclear
- pipeline:
  1. Decompose input claim into sub-claims
  2. Retrieve relevant evidence passages
  3. Model pairwise claim-evidence interactions (cross-attention / co-attention)
  4. Aggregate interaction features → per-claim support/refute scores
  5. Generate explainable verification verdict with evidence pointers
- transfer value: models evidence sufficiency between rationale claims and multimodal evidence

---

**D6.** "Multimodal Fact-Checking with Vision Language Models: A Probing Classifier based Solution"
- Cekinel et al., **COLING 2025**
- paper: https://aclanthology.org/2025.coling-main.310/
- category: Evidence Quality / Probing
- source task: probing frozen VLM embeddings for veracity classification
- code: **yes** — https://github.com/firatcekinel/Multimodal-Fact-Checking-with-Vision-Language-Models
- pipeline:
  1. Frozen VLM (CLIP, BLIP, etc.) extracts text embeddings + image embeddings
  2. Test fusion strategies: early fusion / late fusion / separate encoders
  3. Train lightweight neural probing classifier (small MLP) on frozen embeddings
  4. Classify veracity: true / false / partly true
- transfer value: strong non-generative comparator; tests if frozen embeddings suffice

---

**E1.** "MUStReason: A Benchmark for Diagnosing Pragmatic Reasoning in Video-LMs"
- Saha et al., **arXiv 2025**
- paper: https://arxiv.org/abs/2510.23727
- category: Multimodal Sarcasm / Non-Literal Meaning
- source task: pragmatic reasoning for sarcasm in video
- code: unclear
- pipeline:
  1. Input: video clip with speech, visual, and audio modalities
  2. Annotate modality-specific cues (tone, facial expression, context) and reasoning steps
  3. PragCoT prompting: steer VideoLM to reason about implied intent (not literal meaning)
  4. VideoLM generates pragmatic reasoning chain
  5. Classify sarcasm based on reasoning chain
- transfer value: pragmatic reasoning for non-literal meaning (ironic praise, quoting, polarity flips)

---

**G2.** "Improving Multimodal Hateful Meme Detection Exploiting LMM-Generated Knowledge"
- **CVPR 2025 Workshops**
- paper: https://arxiv.org/abs/2504.09914
- category: Hateful Memes
- source task: hateful meme detection using frozen LMM descriptions
- code: unclear
- pipeline:
  1. Frozen LMM generates semantic description of meme image
  2. Frozen LMM generates emotion/sentiment labels for the meme
  3. CLIP text encoder embeds description + emotion text
  4. CLIP image encoder embeds original meme image
  5. Fuse text embeddings + image embeddings → hateful meme classifier
- transfer value: same "generated text as features" pattern in safety domain

---

**G7.** "MemeCLIP: Leveraging CLIP Representations for Multimodal Meme Classification"
- Bikram et al., **EMNLP 2024**
- paper: https://aclanthology.org/2024.emnlp-main.959/
- category: Hateful Memes
- source task: meme classification with frozen CLIP + adapters
- code: **yes** — https://github.com/SiddhantBikram/MemeCLIP
- pipeline:
  1. Frozen CLIP text encoder processes meme text (OCR)
  2. Frozen CLIP image encoder processes meme image
  3. Lightweight Feature Adapters (small MLPs) transform each modality's embeddings
  4. Fuse adapted embeddings via concatenation or cross-modal attention
  5. Multi-task classification heads: hateful / target / stance
- transfer value: parameter-efficient adaptation of frozen encoders via tiny adapters

---

**L1.** "SemEval-2024 Task 4: Multilingual Detection of Persuasion Techniques in Memes"
- Dimitrov et al., **SemEval 2024**
- paper: https://aclanthology.org/2024.semeval-1.275/
- category: Propaganda / Persuasion Technique Detection
- source task: detecting 22 persuasion techniques in memes (153 teams)
- code: **yes** — https://github.com/Exploration-Lab/IITK-SemEval-2024-Task-4
- pipeline (typical winning system):
  1. OCR extracts text from meme image
  2. VLM or CLIP encodes image + text
  3. Multi-label classifier predicts presence of each of 22 techniques (name calling, loaded language, appeal to fear, etc.)
  4. Hierarchical technique taxonomy allows coarse-to-fine prediction
- transfer value: persuasion-technique vectors as auxiliary features for hate detection

---

**L2.** "PropXplain: Can LLMs Enable Explainable Propaganda Detection?" **[NEW]**
- Hasanain et al., **EMNLP 2025 Findings**
- paper: https://aclanthology.org/2025.findings-emnlp.1296/
- category: Propaganda / Persuasion Technique Detection
- source task: explainable propaganda detection with LLM-generated rationales
- code: unclear
- pipeline:
  1. Input text (news article, social media post)
  2. LLM generates explanation-enhanced labels: technique type + rationale
  3. Build multilingual explanation-enhanced dataset from LLM annotations
  4. Train smaller model to predict both labels and rationale-based explanations
  5. Classify persuasion techniques with explainable output
- transfer value: rationale generation + classification mirrors our textualization pipeline; explanation supervision transferable

---

**M5.** "VERA: Explainable Video Anomaly Detection via Verbalized Learning of Vision-Language Models"
- **CVPR 2025**
- paper: https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.html
- category: MLLM Description as Features
- source task: explainable video anomaly detection via verbalized descriptions
- code: **yes** — https://github.com/vera-framework/VERA
- pipeline:
  1. Frozen VLM processes video frames
  2. Design guiding questions to steer VLM verbalization (e.g., "what is unusual?")
  3. VLM generates verbalized descriptions focused on anomaly-relevant aspects
  4. Encode verbalized descriptions → text features
  5. Train anomaly classifier on verbalized features
- transfer value: "verbalized learning" = our textualization; verbalization quality directly affects performance

---

**M6.** "MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection" **[NEW]**
- **NeurIPS 2025**
- paper: https://openreview.net/forum?id=6Had86RHix
- category: MLLM Description as Features / Video Pipeline
- source task: training-free online video anomaly detection with LLM instruction
- code: **yes** — https://github.com/YsTvT/MoniTor
- pipeline:
  1. Frozen VLM generates descriptions of video segments
  2. Dynamic Memory Gating Module: long-term episodic memory + short-term working memory
  3. LLM instruction guides scoring based on temporal context
  4. LSTM-inspired prediction mechanism for temporal dependency modeling
  5. Memory-based online scoring queue → anomaly detection (no training)
- transfer value: temporal memory architecture for video understanding; LLM instruction without training matches our frozen-MLLM paradigm

---

**P2.** "FLAME: Frozen Large Language Models Enable Data-Efficient Language-Image Pre-training"
- Cao et al., **CVPR 2025**
- paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Cao_FLAME_Frozen_Large_Language_Models_Enable_Data-Efficient_Language_Image_Pre_training_CVPR_2025_paper.pdf
- category: Text Encoding
- source task: data-efficient vision-language alignment
- code: unclear
- pipeline:
  1. Frozen LLM generates rich semantic text from image captions
  2. Frozen LLM text encoder produces text embeddings
  3. Trainable image encoder produces image embeddings
  4. Contrastive alignment loss between text and image embeddings
  5. Aligned embeddings → downstream zero-shot or few-shot image classification
- transfer value: stronger frozen-LLM text representations as multimodal bridge

---

**H2.** "Dynamic Content Moderation in Livestreams: Combining Supervised Classification with MLLM-Boosted Similarity Matching"
- **arXiv 2025**
- paper: https://arxiv.org/abs/2405.15074
- category: Video Content Moderation
- source task: livestream content moderation
- code: unclear
- pipeline:
  1. Frozen MLLM extracts embeddings from video content
  2. Reference-based similarity matching: compare against known violation database
  3. Supervised classifier trained on frozen MLLM embeddings for novel violations
  4. Combine similarity score + classifier score → moderation decision
- transfer value: validates frozen-MLLM + supervised classifier as serious moderation baseline

---

**APO4.** "metaTextGrad: Automatically Optimizing Language Model Optimizers" **[NEW]**
- Xu et al., **NeurIPS 2025**
- paper: https://arxiv.org/abs/2505.18524
- category: Automatic Prompt Optimization
- source task: meta-optimization of LLM-based prompt optimizers
- code: **yes** — https://github.com/zou-group/metatextgrad
- pipeline:
  1. Start with a base prompt optimizer (e.g., TextGrad)
  2. Meta prompt optimizer: refine the optimizer's own prompts for task alignment
  3. Meta structure optimizer: determine optimal combination/sequence of optimization modules
  4. Task-specific alignment: both components adapt to characteristics of target task
  5. Output: task-specialized optimizer achieving up to 6% absolute improvement over baselines
- transfer value: our task has unusually high prompt sensitivity; meta-optimizing the optimizer itself could discover hate-specific optimization strategies

---

**APO5.** "DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines" **[NEW]**
- Khattab et al., **ICLR 2024**
- paper: https://openreview.net/forum?id=sY5N0zY5Od
- category: Automatic Prompt Optimization
- source task: automatic compilation and optimization of LLM pipelines
- code: **yes** — https://github.com/stanfordnlp/dspy
- pipeline:
  1. Define LM pipeline as declarative modules with typed signatures
  2. Each module is parameterized (prompt template, few-shot demos, etc.)
  3. Compiler collects demonstrations, optimizes prompts against task metric
  4. MIPRO optimizer: multi-stage instruction proposal + Bayesian optimization
  5. Output: optimized pipeline with tuned prompts + demos for target metric
- transfer value: mature framework for wrapping our textualization pipeline as DSPy modules and automatically optimizing prompt + schema against downstream F1

---

**APO6.** "GReaTer: Gradients over Reasoning Makes Smaller Language Models Strong Prompt Optimizers" **[NEW]**
- **ICLR 2025**
- paper: https://openreview.net/forum?id=fWRBheSJth
- category: Automatic Prompt Optimization
- source task: gradient-informed prompt optimization using small LLMs
- code: **yes** — https://github.com/psunlpgroup/GreaTer
- pipeline:
  1. Calculate forward token probabilities to generate probable token candidates
  2. LLM generates reasoning chain for problem solution
  3. Extract final answer logits → calculate loss
  4. Compute gradient for probable token candidates → select best tokens
  5. Update prompt with gradient-selected tokens → iterate
- transfer value: enables prompt optimization with smaller models (not dependent on GPT-4); useful if budget-constrained; requires surrogate model access

---

### Tier 3: Niche / Supplementary (11 candidates)

---

**J1.** "On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines"
- Mosbach et al., **ICLR 2021**
- paper: https://openreview.net/forum?id=nzpLWnVAyah
- code: **yes** — https://github.com/uds-lsv/bert-stable-fine-tuning
- pipeline: BERT fine-tuning with bias-corrected Adam, warmup, learning rate schedule → reduces seed variance
- transfer value: training protocol baseline for seed robustness

---

**B6.** "DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts"
- Braun et al., **ICML 2025**
- paper: https://arxiv.org/abs/2412.10510
- code: **yes** — https://github.com/multimodal-ai-lab/DEFAME
- pipeline:
  1. Claim extraction from input (frozen MLLM)
  2. Dynamic evidence search (web + image reverse search)
  3. Evidence summarization (frozen MLLM)
  4. Cross-modal consistency check (frozen MLLM)
  5. Per-sub-claim verdict
  6. Final aggregation → veracity label + explanation
- transfer value: design template for multi-stage modular rationale checking

---

**A3.** "Robust Multimodal Learning through Dynamic Modality Attention" (EMMA-Net)
- **2025 preprint**
- code: unclear
- pipeline: symmetric cross-attention across modalities → dynamic reliability-based attention weights → fusion
- transfer value: reliability-weighted dynamic attention for fusion when modality trust varies

---

**N3.** "VadCLIP: Adapting Vision-Language Models for Weakly Supervised Video Anomaly Detection"
- **arXiv 2024**
- paper: https://arxiv.org/abs/2308.11681
- code: **yes** — https://github.com/nwpu-zxr/VadCLIP
- pipeline:
  1. Frozen CLIP encodes video frames and text prompts
  2. Fine-grained temporal associations between text prototypes and video segments
  3. Temporal aggregation of segment-level matching scores
  4. Weakly supervised anomaly scoring (video-level labels only)
- transfer value: temporal fine-grained matching between text claims and video segments

---

**H1.** "Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation"
- **ACL 2025 Industry**
- paper: https://aclanthology.org/2025.acl-industry.62/
- code: unclear
- pipeline:
  1. Lightweight router extracts features from frozen MLLM embeddings
  2. Router classifies easy positive/negative cases → fast decision
  3. Hard cases escalated to full MLLM-based ranker
  4. Ranker produces detailed moderation verdict
- transfer value: cascade routing pattern from same deployment family

---

**G4.** "Explainable Detection of Propagandistic and Hateful Memes"
- **EMNLP 2025**
- paper: https://aclanthology.org/2025.emnlp-main.1539/
- code: unclear
- pipeline:
  1. Strong MLLM generates step-by-step reasoning about meme
  2. Weak supervision: use reasoning as pseudo-labels for smaller model
  3. Train smaller model with SFT + RL to output binary labels + fine-grained explanations
- transfer value: weak supervision from strong MLLMs for interpretable detection

---

**APO7.** "Large Language Models as Optimizers" (OPRO) **[NEW]**
- Yang et al., **ICLR 2024**
- paper: https://openreview.net/forum?id=Bb4VGOWELI
- category: Automatic Prompt Optimization
- source task: iterative prompt refinement via meta-prompting
- code: **yes** — https://github.com/google-deepmind/opro
- pipeline:
  1. Define optimization task in natural language (meta-prompt)
  2. Include previously evaluated prompts + their scores in meta-prompt
  3. LLM generates new candidate prompts
  4. Evaluate candidates on task metric
  5. Add best candidates to meta-prompt → iterate
- transfer value: simple baseline for iterative prompt search; easy to implement but sample-inefficient and sensitive to noisy evaluation (problematic with 5-8pp seed variance)

---

**APO8.** "PromptQuine: Evolving Prompts In-Context" **[NEW]**
- Wang et al., **ICML 2025**
- paper: https://openreview.net/forum?id=jXZR3XinPg
- category: Automatic Prompt Optimization
- source task: evolutionary search for prompt pruning strategies
- code: **yes** — https://github.com/jianyu-cs/PromptQuine
- pipeline:
  1. Start with standard prompt + in-context demonstrations
  2. Prune demonstrations into compressed token sequences
  3. Evolutionary search over pruning strategies (self-replicating)
  4. Evaluate pruned prompts on task metric
  5. Select best pruning strategy → output compressed but effective prompt
- transfer value: interesting for discovering whether nano model prefers compressed prompts; high risk of exploiting model-specific quirks; poor interpretability for moderation use case

---

**APO9.** "Rethinking Prompt Optimizers: From Prompt Merits to Optimization" (MePO) **[NEW]**
- Zhu et al., **EACL 2026**
- paper: https://aclanthology.org/2026.eacl-long.38.pdf
- category: Automatic Prompt Optimization
- source task: merit-guided lightweight prompt optimization
- code: unclear
- pipeline:
  1. Identify model-agnostic prompt quality merits (Clarity, Precision, Conciseness)
  2. Build preference dataset from merit-aligned prompts
  3. Train lightweight merit-guided prompt optimizer (MePO)
  4. Given task prompt, MePO rewrites it to maximize merit scores
  5. Output: optimized prompt that is clearer, more precise, more concise
- transfer value: low-cost prompt quality baseline; could extend merits with hate-specific coverage (symbols, targets, coding strategies); risk of over-conciseness causing omissions

---

**APO10.** "A Systematic Survey of Automatic Prompt Optimization Techniques" **[NEW — REFERENCE]**
- Ramnath et al., **EMNLP 2025**
- paper: https://aclanthology.org/2025.emnlp-main.1681/
- category: Automatic Prompt Optimization (Survey)
- summary: Comprehensive survey covering 100+ APO papers with 5-part unifying framework; categorizes methods by initialization, optimization strategy (gradient descent, evolutionary, RL), and evaluation
- transfer value: reference for selecting and comparing APO methods for our pipeline

---

**G5.** "Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection" (GLPN-LLM) **[NEW]**
- Hu et al., **ACL 2025**
- paper: https://aclanthology.org/2025.acl-long.72/
- category: Fake News / Content Moderation
- source task: multimodal fake news detection with LLM pseudo-labels + graph propagation
- code: unclear
- pipeline:
  1. LLM generates pseudo-labels for unlabeled multimodal samples
  2. Build sample similarity graph across all data
  3. Global label propagation: propagate LLM pseudo-labels through graph
  4. Mask-based mechanism prevents label leakage during training
  5. Train classifier on propagated labels → multimodal fake news detection
- transfer value: semi-supervised expansion pattern; useful if scaling to large unlabeled video sets; LLM pseudo-labels alone are weak but graph propagation strengthens them

---

## Literature Categories Summary

### Category: Hallucination / Grounding Verification
- B1 (FaithScore, EMNLP 2024), B3 (RefChecker, EMNLP 2024), **B7 (HalLoc, CVPR 2025) [NEW]**, **B8 (MARINE, ICML 2025) [NEW]**, B6 (DEFAME, ICML 2025)

### Category: Modality Imbalance / Fusion
- A2 (Classifier-Guided Gradient, NeurIPS 2024), **A4 (Representation Collapse, ICML 2025) [NEW]**, A3 (EMMA-Net, 2025 preprint)

### Category: Implicit Hate / Coded Language / Video Hate
- K1 (Coding Classification, 2025), K2 (LLM Embeddings for Implicit Hate, 2025), **K3 (ImpliHateVid, ACL 2025) [NEW]**, **K4 (MM-HSD, ACM MM 2025) [NEW]**

### Category: MLLM Description as Features / Pipeline
- M1 (Spotify MLLM Descriptions, ICLR 2026), M4 (LAVAD, CVPR 2024), M5 (VERA, CVPR 2025), **M6 (MoniTor, NeurIPS 2025) [NEW]**

### Category: Automatic Prompt Optimization **[NEW CATEGORY]**
- **APO1 (TextGrad, Nature/ICML 2025)**, **APO2 (VERA verbalized learning, CVPR 2025)**, **APO3 (Trace/OptoPrime, NeurIPS 2024)**, **APO4 (metaTextGrad, NeurIPS 2025)**, **APO5 (DSPy, ICLR 2024)**, **APO6 (GReaTer, ICLR 2025)**, **APO7 (OPRO, ICLR 2024)**, **APO8 (PromptQuine, ICML 2025)**, **APO9 (MePO, EACL 2026)**, **APO10 (Survey, EMNLP 2025)**

### Category: Stance Detection
- F1 (MultiClimate, EMNLP 2024 Workshop)

### Category: Grounded Reasoning
- I3 (MoReVQA, CVPR 2024)

### Category: Hateful Memes
- G2 (LMM-Generated Knowledge, CVPR 2025 Workshop), G7 (MemeCLIP, EMNLP 2024)

### Category: Propaganda / Persuasion
- L1 (SemEval Persuasion Techniques, 2024), **L2 (PropXplain, EMNLP 2025) [NEW]**

### Category: Text Encoding
- P1 (LLM2Vec, ICLR 2024), P2 (FLAME, CVPR 2025)

### Category: Evidence Quality / Claim Verification
- D1 (CorXFact, COLING 2025), D6 (Probing Classifier, COLING 2025)

### Category: Video Content Moderation
- H1 (Filter-And-Refine, ACL 2025 Industry), H2 (Livestream Moderation, arXiv 2025)

### Category: Video Anomaly Detection
- N3 (VadCLIP, arXiv 2024)

### Category: Methodology / Pipeline Analysis
- NEW2 (DisCLIP, AAAI 2025)

### Category: Multimodal Sarcasm / Non-Literal Meaning
- E1 (MUStReason, arXiv 2025)

### Category: Fake News / Content Moderation
- **G5 (GLPN-LLM, ACL 2025) [NEW]**

### Category: Training Robustness
- J1 (BERT Stability, ICLR 2021)

### Category: Explainable Detection
- G4 (Propagandistic/Hateful Memes, EMNLP 2025)

---

## GPT Discussion Summary (Thread: 019d5da2-8dda-7f51-be79-a089d602bc60)

### APO Category Ranking for Transfer Fit:
1. **High priority**: VERA (APO2), TextGrad (APO1), Trace/OptoPrime (APO3), DSPy (APO5), metaTextGrad (APO4)
2. **Secondary**: MePO (APO9), OPRO (APO7)
3. **Conditional/speculative**: GReaTer (APO6), PromptQuine (APO8)

### Key GPT Insights:
- **VERA** has the closest domain match: video + frozen VLM + prompt-as-learnable-parameter
- **TextGrad** is the most general-purpose; risk of optimizing wording without fixing coverage
- **Trace/OptoPrime** best if optimizing entire pipeline (prompt + schema + post-processing)
- **DSPy** strongest practical engineering path for repeated prompt search
- **metaTextGrad** justified by our task's unusually high prompt sensitivity
- **GReaTer** requires surrogate model (can't access frozen MLLM gradients directly)
- **PromptQuine** high risk of exploiting model quirks; poor interpretability for moderation
- **OPRO** good baseline but sample-inefficient given 5-8pp seed variance noise

### New Papers Assessment:
- **ImpliHateVid** (K3): Very strong — directly about implicit hate in video, ACL 2025
- **MM-HSD** (K4): Very strong — SOTA on HateMM (our dataset), ACM MM 2025
- **HalLoc** (B7): Strong — token-level reliability layer for generated rationales
- **MARINE** (B8): Strong — training-free grounding for frozen MLLM
- **Representation Collapse** (A4): Strong — explains our AV suppression mechanistically
- **MoniTor** (M6): Medium — temporal memory useful but anomaly semantics far from hate
- **PropXplain** (L2): Medium — explanation supervision transferable
- **GLPN-LLM** (G5): Medium-low — semi-supervised expansion for unlabeled data

---

**Total candidates**: 44 (17 Tier 1, 16 Tier 2, 11 Tier 3)
**New papers added**: 18 (including 10 APO papers as new category)
**Categories**: 16 (1 new: Automatic Prompt Optimization)
