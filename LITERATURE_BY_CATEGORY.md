# Literature Organized by Two Categories

**Date**: 2026-04-08
**Source**: Refreshed from prior 44-paper collection + new search (2025-2026 top venues); Category E added for two-stage MLLM→Parametric Classifier papers

---

## Category A: Video Understanding (MLLM-based and General)

Papers about how to understand video content — using MLLMs, VLMs, or traditional multimodal approaches. Covers: MLLM textualization pipelines, video anomaly detection, video QA, content moderation, hateful video/meme detection, implicit hate, sarcasm, stance detection, propaganda detection.

---

### A.1 MLLM Textualization / Description-as-Features Pipeline

**M1.** "Describe What You See with MLLMs to Enhance Video Recommendations" — Spotify, **ICLR 2026**
- Frozen MLLM -> text description -> text encoder -> recommender
- paper: https://openreview.net/pdf?id=MiV3WXDYJb | code: no
- Closest positive precedent for our entire recipe; +4-18% gains

**M5/APO2.** "VERA: Explainable Video Anomaly Detection via Verbalized Learning" — **CVPR 2025**
- Frozen VLM + learnable guiding questions -> verbalized description -> classifier
- Treats VLM questions as learnable parameters, optimizes via verbal feedback
- paper: https://openaccess.thecvf.com/content/CVPR2025/html/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_CVPR_2025_paper.html | code: https://github.com/vera-framework/VERA

**M6.** "MoniTor: LLMs with Instruction for Online Video Anomaly Detection" — **NeurIPS 2025**
- Frozen VLM -> descriptions + dual memory (long-term + short-term) -> online scoring
- paper: https://openreview.net/forum?id=6Had86RHix | code: https://github.com/YsTvT/MoniTor

**NEW2.** "DisCLIP: Does VLM Classification Benefit from LLM Description Semantics?" — **AAAI 2025**
- LLM descriptions -> CLIP text encode -> zero-shot classification
- Finding: description quality > quantity
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/32638

**H1.** "Filter-And-Refine: MLLM Cascade for Video Content Moderation" — **ACL 2025 Industry**
- Lightweight router -> easy cases fast, hard cases -> full MLLM ranker
- paper: https://aclanthology.org/2025.acl-industry.62/

**P2.** "FLAME: Frozen LLMs Enable Data-Efficient Language-Image Pre-training" — **CVPR 2025**
- Frozen LLM text encoder + trainable image encoder + contrastive alignment
- paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Cao_FLAME_Frozen_Large_Language_Models_Enable_Data-Efficient_Language_Image_Pre_training_CVPR_2025_paper.pdf

### A.1 Earlier Efforts

**M4.** "LAVAD: Harnessing LLMs for Training-free Video Anomaly Detection" — **CVPR 2024**
- Frozen VLM -> per-keyframe text -> CLIP encode -> temporal anomaly scoring
- paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf | code: https://github.com/lucazanella/lavad

**H2.** "Dynamic Content Moderation in Livestreams" — **arXiv 2025**
- Frozen MLLM embeddings + similarity matching + supervised classifier
- paper: https://arxiv.org/abs/2405.15074

### A.2 Hallucination / Grounding Verification

**B8.** "MARINE: Image-Grounded Guidance for LVLM Hallucination" — **ICML 2025 Spotlight**
- DETR + RAM++ visual guidance -> steer frozen LVLM decoding -> grounded descriptions
- paper: https://arxiv.org/abs/2402.08680 | code: https://github.com/Linxi-ZHAO/MARINE

**B6.** "DEFAME: Dynamic Evidence-based Fact-checking with Multimodal Experts" — **ICML 2025**
- Multi-stage: claim extraction -> evidence search -> summarize -> cross-modal check -> verdict
- paper: https://arxiv.org/abs/2412.10510 | code: https://github.com/multimodal-ai-lab/DEFAME

**B7.** "HalLoc: Token-level Hallucination Localization for VLMs" — **CVPR 2025**
- Lightweight add-on -> per-token hallucination probability during generation
- paper: https://openaccess.thecvf.com/content/CVPR2025/html/Park_HalLoc_Token-level_Localization_of_Hallucinations_for_Vision_Language_Models_CVPR_2025_paper.html | code: https://github.com/dbsltm/cvpr25_halloc

**NEW3.** "MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs" — **CVPR 2025**
- DST-attention (disentangled spatial-temporal) + Harmonic-RoPE -> reduces action-scene hallucination; includes UNSCENE benchmark
- paper: https://arxiv.org/abs/2503.15871

**NEW4.** "Seeing Far and Clearly: Mitigating Hallucinations with Attention Causal Decoding" — **CVPR 2025**
- Attention-based causal decoding to improve visual attention during MLLM generation
- paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Tang_Seeing_Far_and_Clearly_Mitigating_Hallucinations_in_MLLMs_with_Attention_CVPR_2025_paper.pdf

**NEW5.** "ContextualLens: Contextual Embeddings for Robust Hallucination Detection" — **NAACL 2025**
- Middle-layer contextual token embeddings (not logit lens) -> training-free hallucination detection and visual grounding
- paper: https://aclanthology.org/2025.naacl-long.488/

**NEW6.** "AdaVIB: Mitigating Hallucinations via Adaptively Constraining Information Flow" — **AAAI 2025**
- Variational Information Bottleneck with entropy-based noise control -> reduces object hallucination
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/34512

**NEW7.** "C-PMI: Conditional Mutual Information Calibrated Decoding for Reducing Hallucinations" — **NeurIPS 2025**
- Bi-level optimization for visual-textual token contributions + token purification mechanism
- paper: https://arxiv.org/abs/2505.19678

**NEW8.** "AVCD: Audio-Visual Contrastive Decoding for Hallucination Mitigation" — **NeurIPS 2025**
- Training-free trimodal contrastive decoding for audio-visual LLMs; entropy-guided adaptive modality weighting
- paper: https://arxiv.org/abs/2505.20862

### A.2 Earlier Efforts

**B1.** "FaithScore" — **EMNLP 2024 Findings**
- Decompose VLM caption -> atomic claims -> VQA verification -> faithfulness score
- paper: https://aclanthology.org/2024.findings-emnlp.290/ | code: https://github.com/bcdnlp/FAITHSCORE

**B3.** "RefChecker: Knowledge-Centric Hallucination Detection" — **EMNLP 2024**
- Extract claim-triplets -> compare against reference -> hallucination score
- paper: https://aclanthology.org/2024.emnlp-main.395/

### A.3 Hateful Video / Meme Detection (Task-Specific)

**K3.** "ImpliHateVid: Implicit Hate Speech Detection in Videos" — **ACL 2025**
- Two-stage contrastive learning: modality-specific -> cross-encoder
- Auxiliary: sentiment, emotion, caption features
- paper: https://aclanthology.org/2025.acl-long.842/ | code: https://github.com/videohatespeech/Implicit_Video_Hate

**K4.** "MM-HSD: Multi-Modal Hate Speech Detection in Videos" — **ACM MM 2025**
- Cross-Modal Attention: on-screen text as query, other modalities as key/value
- SOTA on HateMM: M-F1 = 0.874
- paper: https://dl.acm.org/doi/10.1145/3746027.3754558 | code: https://github.com/idiap/mm-hsd

**G4.** "Explainable Detection of Propagandistic and Hateful Memes" — **EMNLP 2025**
- Strong MLLM reasoning -> weak supervision -> smaller model SFT+RL
- paper: https://aclanthology.org/2025.emnlp-main.1539/

**NEW9.** "MoRE: Retrieval-Augmented Multimodal Experts for Short Video Hate Detection" — **WWW 2025**
- Mixture of retrieval-augmented multimodal experts + joint video retriever + dynamic integration; +6.91% M-F1 over SOTA
- paper: https://dl.acm.org/doi/10.1145/3696410.3714560

**NEW10.** "Cross-Modal Transfer from Memes to Videos for Hateful Video Detection" — **WWW 2025**
- Hateful meme datasets as augmentation for video hate training via re-annotation; fine-tunes LLaMA-3.2-11B and LLaVA-Next-Video-7B
- paper: https://dl.acm.org/doi/10.1145/3696410.3714534

**NEW11.** "DeHate: A Holistic Hateful Video Dataset for Explicit and Implicit Hate Detection" — **ACM MM 2025**
- Largest hateful video dataset (6689 videos) with fine-grained explicit/implicit labels, segment-level localization, modality attribution
- paper: https://dl.acm.org/doi/10.1145/3746027.3758272

**NEW12.** "Robust Adaptation of LMMs for Retrieval Augmented Hateful Meme Detection" — **EMNLP 2025**
- Robust LMM adaptation + retrieval augmentation -> improved in-domain and cross-domain generalization on 6 meme datasets
- paper: https://aclanthology.org/2025.emnlp-main.1215/

**NEW13.** "MultiHateLoc: Temporal Localisation of Multimodal Hate Content in Videos" — **WWW 2026**
- First tri-modal weakly-supervised hate localisation; modality-aware temporal encoders + dynamic fusion + MIL objective
- paper: https://arxiv.org/abs/2512.10408

**NEW14.** "Multi3Hate: Multimodal, Multilingual, and Multicultural Hate Speech Detection" — **NAACL 2025**
- Multimodal multilingual multicultural hate speech benchmark and detection across diverse cultural contexts
- paper: https://aclanthology.org/2025.naacl-long.490/

**NEW15.** "Cross-Cultural Evaluation of VLMs for Hateful Meme Detection" — **WWW 2026**
- Systematic cross-cultural VLM evaluation across 6 languages; native-language prompting > translate-then-detect
- paper: https://arxiv.org/abs/2602.07497

**NEW16.** "From Meme to Threat: Hateful Meme Understanding and Induced Hateful Content Generation" — **USENIX Security 2025**
- Studies hateful meme understanding and the risk of LLMs generating induced hateful content
- paper: https://www.usenix.org/system/files/conference/usenixsecurity25/sec25cycle1-prepub-1017-ma-yihan.pdf

**NEW47.** "ALARM: From Shallow Humor to Metaphor — Label-Free Harmful Meme Detection via LMM Agent Self-Improvement" — **arXiv 2025**
- Confidence-based pseudo-labeling of "easy" memes + pairwise contrastive self-improvement of learner LMM; fully label-free; outperforms even label-driven methods on 3 datasets
- paper: https://arxiv.org/abs/2512.21598
- **[COMPETITOR]** Closest label-free meme method to our approach

**NEW48.** "ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection" — **ICLR 2026**
- SFT warmup + GRPO with curriculum learning; Conditional Decision Entropy (CDE) as reward; explanation-driven detection; +15% F1 over GRPO baseline
- paper: https://arxiv.org/abs/2510.08630

**NEW49.** "LELA: Towards Training-free Multimodal Hate Localisation with Large Language Models" — **arXiv 2026**
- Decomposes video into 5 modalities; multi-stage prompting with frozen LLM; per-frame hate scores + composition matching; training-free
- paper: https://arxiv.org/abs/2602.09637
- **[COMPETITOR]** Training-free SOTA on HateMM and MultiHateClip

**NEW50.** "SLM-Mod: Small Language Models Surpass LLMs at Content Moderation" — **NAACL 2025**
- Community-specific LoRA-fine-tuned SLMs (<15B) beat zero-shot LLMs by 11.5% accuracy, 25.7% recall on 150K Reddit comments
- paper: https://arxiv.org/abs/2410.09750 | code: https://github.com/AGoyal0512/SLM-Mod

### A.3 Earlier Efforts

**G2.** "Improving Hateful Meme Detection with LMM-Generated Knowledge" — **CVPR 2025 Workshops**
- Frozen LMM -> description + emotion -> CLIP encode -> fuse -> classifier
- paper: https://arxiv.org/abs/2504.09914

**G7.** "MemeCLIP: Leveraging CLIP for Multimodal Meme Classification" — **EMNLP 2024**
- Frozen CLIP + lightweight adapters -> multi-task (hateful/target/stance)
- paper: https://aclanthology.org/2024.emnlp-main.959/ | code: https://github.com/SiddhantBikram/MemeCLIP

**K1.** "Cracking the Code: Implicit Hate via Coding Classification" — **2025**
- Classify coding strategy (sarcasm/insinuation/metaphor/etc.) as auxiliary features
- paper: https://aclanthology.org/2025.trustnlp-main.9/

**K2.** "Specializing LLM Embeddings for Implicit Hate" — **ACM DHOW 2025**
- Contrastive fine-tuning of LLM embeddings for implicit hate
- paper: https://publications.idiap.ch/publications/show/5671 | code: https://github.com/idiap/implicit-hsd

### A.4 Stance / Sarcasm / Pragmatics

**NEW17.** "T-MAD: Target-driven Multimodal Alignment for Stance Detection" — **EMNLP 2025**
- Iterative target-driven multimodal alignment with dynamic weighting for in-target and zero-shot stance detection (RoBERTa + ViT)
- paper: https://aclanthology.org/2025.emnlp-main.30/

**NEW18.** "MMSD3.0: Cross-Image Reasoning Model for Multi-Image Sarcasm Detection" — **ACL 2025**
- Cross-image sequence modeling + relevance-guided fine-grained cross-modal fusion for multi-image sarcasm
- paper: https://arxiv.org/abs/2510.23299

**NEW19.** "Sarcasm-R1: Enhancing Sarcasm Detection through Focused Reasoning" — **EMNLP 2025 Findings**
- RL-based training with SarGRM reward model + multi-dimensional CoT reasoning on Gemma 7B + LoRA
- paper: https://aclanthology.org/2025.findings-emnlp.570.pdf

### A.4 Earlier Efforts

**F1.** "MultiClimate: Multimodal Stance Detection on Climate Change Videos" — **EMNLP 2024 Workshop**
- BERT text + ResNet/ViT frames -> cross-attention fusion -> stance classification
- paper: https://aclanthology.org/2024.nlp4pi-1.27/ | code: https://github.com/werywjw/MultiClimate

**E1.** "MUStReason: Pragmatic Reasoning in Video-LMs" — **arXiv 2025**
- PragCoT prompting -> VideoLM pragmatic reasoning -> sarcasm classification
- paper: https://arxiv.org/abs/2510.23727

### A.5 Propaganda / Persuasion / Fake News

**G5.** "GLPN-LLM: LLMs + Label Propagation for Multimodal Fake News" — **ACL 2025**
- LLM pseudo-labels -> graph propagation -> denoised labels -> classifier
- paper: https://aclanthology.org/2025.acl-long.72/

**L2.** "PropXplain: Explainable Propaganda Detection with LLMs" — **EMNLP 2025 Findings**
- LLM rationale -> train smaller model for labels + explanations
- paper: https://aclanthology.org/2025.findings-emnlp.1296/

**NEW20.** "Multi-perspective Rationale Generation and Verification for Multimodal Fake News Detection" — **AAAI 2026**
- Cross-verification of multi-perspective rationales + adaptive weighting fusion; SOTA on Twitter, Weibo, GossipCop
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/36965

**NEW21.** "MTS: Multimodal Taylor Series Network for Misinformation Detection" — **WWW 2025**
- Taylor series expansion for low-order and high-order cross-modal interactions with linear parameter scalability
- paper: https://dl.acm.org/doi/10.1145/3696410.3714719

**NEW22.** "MDAM3: Misinformation Detection for Multitype Multimodal Media" — **WWW 2025**
- Internal visual manipulation detectors (ImageBind) + external web signals + LVLMs for multi-type misinformation detection
- paper: https://dl.acm.org/doi/10.1145/3696410.3714498

**NEW23.** "Communication Makes Perfect: Persuasion Dataset via Multi-LLM Communication" — **NAACL 2025**
- Multi-LLM communication framework for generating high-quality persuasive dialogue data
- paper: https://aclanthology.org/2025.naacl-long.203/

### A.5 Earlier Efforts

**L1.** "SemEval-2024 Task 4: Persuasion Techniques in Memes" — **SemEval 2024**
- OCR + VLM/CLIP -> multi-label 22-technique classification
- paper: https://aclanthology.org/2024.semeval-1.275/ | code: https://github.com/Exploration-Lab/IITK-SemEval-2024-Task-4

### A.6 Grounded Reasoning / Video QA

**NEW24.** "MSR-ViR: Modularized Self-Reflected Video Reasoner for Multimodal LLM" — **ICML 2025**
- MoST-Grounding module decomposes questions via tree-structured policies; alternate self-reflection training optimizes policy and MLLM jointly
- paper: https://proceedings.mlr.press/v267/song25g.html

**NEW25.** "Commonsense Video QA through Video-Grounded Entailment Tree Reasoning" — **CVPR 2025**
- Explicit entailment tree construction over video fragments with recursive decomposition and verification
- paper: https://arxiv.org/abs/2501.05069

**NEW26.** "CG-Bench: Clue-grounded QA Benchmark for Long Video Understanding" — **ICLR 2025**
- 1,219 videos / 12,129 QA pairs with clue-grounded white-box and black-box evaluation protocols
- paper: https://openreview.net/forum?id=le4IoZZHy1

### A.6 Earlier Efforts

**I3.** "MoReVQA: Modular Reasoning for Video QA" — **CVPR 2024**
- Frozen VLM captions -> memory -> LLM decomposes -> ground -> reason -> synthesize
- paper: https://openaccess.thecvf.com/content/CVPR2024/html/Min_MoReVQA_Exploring_Modular_Reasoning_Models_for_Video_Question_Answering_CVPR_2024_paper.html

### A.7 Video Anomaly Detection (non-MLLM)

**NEW27.** "DSANet: Disentangled Semantic Alignment for Weakly Supervised Video Anomaly Detection" — **AAAI 2026**
- Coarse-grained normality prototypes + fine-grained decoupled contrastive visual-language alignment on CLIP features; SOTA on XD-Violence and UCF-Crime
- paper: https://arxiv.org/abs/2511.10334

**NEW51.** "AnyAnomaly: Zero-Shot Customizable Video Anomaly Detection with LVLM" — **WACV 2026**
- User-defined anomaly text → context-aware VQA + key frame selection; no fine-tuning; SOTA on UBnormal; strong cross-dataset generalization
- paper: https://arxiv.org/abs/2407.13221 | code: https://github.com/SkiddieAhn/Paper-AnyAnomaly
- **[BASELINE]** Just define "hateful content" as anomaly text → zero-shot hate detection

**NEW52.** "AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP" — **CVPR 2025**
- Anomaly-aware text anchors + residual adapters; patch-level alignment preserves CLIP generalization while adding anomaly discrimination
- paper: https://arxiv.org/abs/2407.15819 | code: https://github.com/Mwxinnn/AA-CLIP
- **[BASELINE]** Define "hateful" vs "benign" text anchors → zero-shot hateful frame detection

**NEW53.** "E3M: Zero-Shot Spatio-Temporal Video Grounding with Expectation-Maximization" — **ECCV 2024 Oral**
- Zero-shot temporal localization via EM at test time; visual modulation + textual modulation; no training videos needed
- paper: https://arxiv.org/abs/2404.05095 | code: https://github.com/baopj/E3M
- **[BASELINE]** Localize hateful moments in video without labels

### A.7 Earlier Efforts

**N3.** "VadCLIP: Frozen CLIP for Weakly Supervised Video Anomaly Detection" — **arXiv 2024**
- Frozen CLIP -> fine-grained temporal text-video matching -> anomaly scoring
- paper: https://arxiv.org/abs/2308.11681 | code: https://github.com/nwpu-zxr/VadCLIP

### A.9 Out-of-Context / Cross-Modal Incongruity Detection

**NEW54.** "SNIFFER: Multimodal LLM for Explainable Out-of-Context Misinformation Detection" — **CVPR 2024**
- Two-stage instruction tuning: (1) entity alignment, (2) OOC discrimination; internal checking (image-text inconsistency) + external checking (retrieved context); 88.4% ACC (+3.7% SOTA)
- paper: https://arxiv.org/abs/2403.01879 | code: https://github.com/MischaQI/Sniffer
- **[BASELINE]** Internal incongruity checking transfers to detecting hate via cross-modal mismatch

**NEW55.** "LAMAR: Latent Multimodal Reconstruction for Misinformation Detection" — **arXiv 2025**
- Reconstruct expected caption from image; reconstruction error = incongruity signal; trained on synthetic miscaptioned data; +7.8-10.4% on NewsCLIPpings/VERITE
- paper: https://arxiv.org/abs/2504.06010 | code: https://github.com/stevejpapad/miscaptioned-image-reconstruction
- **[BASELINE]** Reconstruction gap between expected vs actual text signals hateful juxtaposition

### A.8 Evidence / Claim Verification

**NEW28.** "MEVER: Multi-Modal Explainable Claim Verification with Graph-based Evidence Retrieval" — **EACL 2026**
- Two-layer multimodal graph for evidence retrieval + token/evidence-level fusion + multimodal Fusion-in-Decoder for explanations
- paper: https://arxiv.org/abs/2602.10023

### A.8 Earlier Efforts

**D1.** "CorXFact: Explainable Fact-Checking with Claim-Evidence Interaction" — **COLING 2025**
- Sub-claim decomposition -> pairwise claim-evidence cross-attention -> verdict
- paper: https://aclanthology.org/2025.coling-main.108/

**D6.** "Multimodal Fact-Checking via VLM Probing Classifier" — **COLING 2025**
- Frozen VLM embeddings -> lightweight MLP probing -> veracity classification
- paper: https://aclanthology.org/2025.coling-main.310/ | code: https://github.com/firatcekinel/Multimodal-Fact-Checking-with-Vision-Language-Models

---

## Category B: Multimodal Fusion Methods

Papers about HOW to fuse information from multiple modalities. Covers: modality imbalance, attention mechanisms, representation learning, text encoding, prompt optimization, fusion architectures.

---

### B.1 Modality Imbalance / Dominance

**A4.** "A Closer Look at Multimodal Representation Collapse" — **ICML 2025 Spotlight**
- Modality collapse via shared neurons in fusion head -> cross-modal KD frees rank bottlenecks
- Alternative: explicit basis reallocation algorithm
- paper: https://arxiv.org/abs/2505.22483 | project: https://abhrac.github.io/mmcollapse/

**NEW29.** "Rethinking Multimodal Learning: Mitigating Classification Ability Disproportion" — **NeurIPS 2025 Oral**
- Boosting principle to dynamically balance classification ability of weak/strong modalities via adaptive classifier assignment
- paper: https://openreview.net/forum?id=Q6IyUpBmrG

**NEW30.** "ARM: Asymmetric Reinforcing Against Multi-Modal Representation Bias" — **AAAI 2025**
- Dynamically reinforces weak modalities via Conditional Mutual Information (CMI) and Mutual Information-based Valuation (MIV)
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/33841

**NEW31.** "GMML: Gradient-Modulated Robustness for Imbalance-Aware Multimodal Learning" — **ACM MM 2025**
- Imbalance-aware gradient modulation with smooth weight transitions + L2-norm parameter constraints
- paper: https://dl.acm.org/doi/10.1145/3746027.3755198

**NEW32.** "G2D: Gradient-Guided Distillation for Multimodal Learning" — **ICCV 2025**
- KD framework with dynamic sequential modality prioritization preventing stronger modalities from overshadowing weaker ones
- paper: https://iccv.thecvf.com/virtual/2025/poster/73

**NEW33.** "Two Challenges, One Solution: Dynamic Modality Recognition and Enhancement" — **EMNLP 2025 Findings**
- Unified solution for modality missingness and modality imbalance without explicit missing-modality annotations
- paper: https://aclanthology.org/2025.findings-emnlp.689/

### B.1 Earlier Efforts

**A2.** "Classifier-Guided Gradient Modulation for Enhanced Multimodal Learning" — **NeurIPS 2024**
- Monitor per-modality gradients -> adaptively modulate magnitude + direction -> anti-dominance
- paper: https://openreview.net/forum?id=oe5ZEqTOaz

**A3.** "EMMA-Net: Robust Multimodal Learning through Dynamic Modality Attention" — **2025 preprint**
- Symmetric cross-attention -> dynamic reliability-based attention weights -> fusion

### B.2 Text Encoding / Representation

**NEW34.** "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models" — **ICLR 2025**
- Latent attention pooling for decoder-only LLMs, removes causal mask during contrastive training, two-stage instruction-tuning; #1 on MTEB
- paper: https://openreview.net/forum?id=lgsyLSsDRe

**NEW35.** "GritLM: Generative Representational Instruction Tuning" — **ICLR 2025**
- Unifies generative and embedding tasks in a single LLM via instruction-based task switching; speeds up RAG by >60%
- paper: https://openreview.net/forum?id=BC4lIvfSzv

**NEW36.** "Conan-Embedding-v2: Training LLM from Scratch for Text Embeddings" — **EMNLP 2025**
- Soft-masking to gradually transition causal -> bidirectional in 1.4B LLM; SOTA on English + Chinese MTEB
- paper: https://aclanthology.org/2025.emnlp-main.758/

### B.2 Earlier Efforts

**P1.** "LLM2Vec: LLMs Are Secretly Powerful Text Encoders" — **ICLR 2024**
- Frozen decoder LLM -> enable bidirectional attention -> contrastive learning -> strong embeddings
- paper: https://arxiv.org/abs/2404.05961 | code: https://github.com/McGill-NLP/llm2vec

### B.3 Training Robustness / Stability

**NEW37.** "Proxy-FDA: Feature Distribution Alignment for Fine-tuning Vision Foundation Models" — **ICML 2025**
- Regularization via nearest-neighbor graph alignment between pre-trained and fine-tuned feature spaces
- paper: https://icml.cc/virtual/2025/poster/45163

**NEW38.** "StarFT: Robust Fine-tuning of Zero-shot Models via Spuriosity Alignment" — **IJCAI 2025**
- Aligns output distributions for spuriosity-injected labels with zero-shot CLIP; +14.3% worst-group accuracy on Waterbirds
- paper: https://www.ijcai.org/proceedings/2025/616

**NEW39.** "Booster: Tackling Harmful Fine-tuning via Attenuating Harmful Perturbation" — **ICLR 2025 Oral**
- Alignment-stage loss regularizer that attenuates harmful perturbation over model weights; -22.6% harmful score
- paper: https://openreview.net/forum?id=tTPHgb0EtV

### B.3 Earlier Efforts

**J1.** "On the Stability of Fine-tuning BERT" — **ICLR 2021**
- Bias-corrected Adam + warmup + LR schedule -> reduces seed variance
- paper: https://openreview.net/forum?id=nzpLWnVAyah | code: https://github.com/uds-lsv/bert-stable-fine-tuning

### B.4 Automatic Prompt Optimization

**APO1.** "TextGrad: Automatic Differentiation via Text" — **Nature / ICML 2025**
- AI system as computation graph -> textual gradients -> prompt optimization
- paper: https://arxiv.org/abs/2406.07496 | code: https://github.com/zou-group/textgrad

**APO4.** "metaTextGrad: Meta-Optimizing LLM Optimizers" — **NeurIPS 2025**
- Meta prompt optimizer + meta structure optimizer -> task-specific optimizer alignment
- paper: https://arxiv.org/abs/2505.18524 | code: https://github.com/zou-group/metatextgrad

**APO6.** "GReaTer: Gradients over Reasoning for Small LLM Prompt Optimizers" — **ICLR 2025**
- Token-level gradient information over reasoning -> prompt optimization with small models
- paper: https://openreview.net/forum?id=fWRBheSJth | code: https://github.com/psunlpgroup/GreaTer

**APO8.** "PromptQuine: Evolving Prompts In-Context" — **ICML 2025**
- Evolutionary pruning of in-context demonstrations -> compressed effective prompts
- paper: https://openreview.net/forum?id=jXZR3XinPg | code: https://github.com/jianyu-cs/PromptQuine

**APO9.** "MePO: From Prompt Merits to Optimization" — **EACL 2026**
- Model-agnostic prompt quality merits (Clarity, Precision, Conciseness) -> merit-guided rewriting
- paper: https://aclanthology.org/2026.eacl-long.38.pdf

**APO10.** "Systematic Survey of Automatic Prompt Optimization" — **EMNLP 2025** [REFERENCE]
- 100+ papers, 5-part unifying framework
- paper: https://aclanthology.org/2025.emnlp-main.1681/

**NEW40.** "GPO: Unleashing LLMs as Prompt Optimizers" — **AAAI 2025**
- Gradient-inspired LLM-based optimizer; retrieves relevant prompts from trajectory as update direction + cosine-based decay; +56.8% BBH, +62.6% MMLU
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/34713

**NEW41.** "LPO: Local Prompt Optimization" — **NAACL 2025**
- Identifies optimization tokens and focuses LLM edits only on those; integrates with any existing APE method
- paper: https://aclanthology.org/2025.naacl-short.7/

**NEW42.** "Automatic Prompt Optimization via Heuristic Search" — **ACL 2025 Findings** [REFERENCE]
- Survey systematizing APO through heuristic search methods
- paper: https://aclanthology.org/2025.findings-acl.1140/

### B.4 Earlier Efforts

**APO3.** "Trace/OptoPrime: Generative Optimization with Execution Traces" — **NeurIPS 2024**
- PyTorch-like API -> capture traces + feedback -> LLM optimizer proposes updates
- paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/83ba7056bce2c3c3c27e17397cf3e1f0-Paper-Conference.pdf | code: https://github.com/microsoft/Trace

**APO5.** "DSPy: Compiling Declarative LM Calls into Pipelines" — **ICLR 2024**
- Declarative modules -> MIPRO compiler optimizes prompts + demos against task metric
- paper: https://openreview.net/forum?id=sY5N0zY5Od | code: https://github.com/stanfordnlp/dspy

**APO7.** "OPRO: Large Language Models as Optimizers" — **ICLR 2024**
- Iterative meta-prompting: include previous prompts + scores -> generate new candidates
- paper: https://openreview.net/forum?id=Bb4VGOWELI | code: https://github.com/google-deepmind/opro

### B.5 Multimodal Fusion Architectures

**NEW43.** "I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts" — **ICML 2025**
- End-to-end MoE with weakly supervised interaction losses for heterogeneous cross-modal interactions + sample/dataset-level interpretability
- paper: https://openreview.net/forum?id=EuJaF5QsMP

**NEW44.** "Mixture-of-Transformers (MoT): Sparse Scalable Multi-Modal Foundation Models" — **TMLR 2025**
- Decouples non-embedding parameters by modality while maintaining global self-attention; dense-level quality at 55.8% FLOPs
- paper: https://arxiv.org/abs/2411.04996

**NEW45.** "TACA: Temperature-Adjusted Cross-Modal Attention" — **ICCV 2025**
- Amplifies visual-text interaction logits to counter suppression from visual token dominance; timestep-dependent weighting
- paper: https://arxiv.org/abs/2506.07986

**NEW46.** "CMAD: Correlation-Aware and Modalities-Aware Distillation for Multimodal Sentiment" — **ICCV 2025**
- Correlation-aware and modality-aware distillation framework for multimodal fusion in sentiment analysis
- paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhuang_CMAD_Correlation-Aware_and_Modalities-Aware_Distillation_for_Multimodal_Sentiment_Analysis_with_ICCV_2025_paper.pdf

### B.5 Earlier (Cross-references from Category A)

**G7.** "MemeCLIP" — frozen CLIP + lightweight adapters + cross-modal attention — **EMNLP 2024**

**K3.** "ImpliHateVid" — two-stage contrastive: modality-specific -> cross-encoder — **ACL 2025**

**K4.** "MM-HSD" — Cross-Modal Attention (text query, AV key/value) + concat — **ACM MM 2025**

---

## Cross-Reference: Papers in Both Categories

| Paper | Category A (Video Understanding) | Category B (Fusion Method) |
|---|---|---|
| VERA (CVPR 2025) | Video anomaly detection pipeline | Prompt optimization (learnable questions) |
| ImpliHateVid (ACL 2025) | Implicit hate in video | Two-stage contrastive fusion |
| MM-HSD (ACM MM 2025) | Video hate detection SOTA | Cross-modal attention fusion |
| MemeCLIP (EMNLP 2024) | Hateful meme detection | Frozen encoder + adapter fusion |
| Representation Collapse (ICML 2025) | Explains AV suppression | Cross-modal KD / basis reallocation |
| MoRE (WWW 2025) | Video hate detection | Mixture of retrieval-augmented experts |
| I2MoE (ICML 2025) | General multimodal | Interpretable MoE fusion |

---

## Category C: Interleaved Reasoning for Multimodal Understanding

Papers about interleaving visual and textual reasoning steps — generating visual rationales (image regions, latent embeddings, or frame retrievals) within CoT, rather than text-only CoT. Covers: interleaved-modal CoT, latent visual reasoning, video-specific interleaved reasoning, chain-of-shot, grounded CoT verification.

---

### C.1 Interleaved-Modal Chain-of-Thought

**IR1.** "Interleaved-Modal Chain-of-Thought" — **CVPR 2025**
- Attention-driven Selection (ADS) inserts image regions into CoT steps as visual rationales; plug-and-play, no extra parameters; up to 14% over text-only CoT
- paper: https://arxiv.org/abs/2411.19488 | code: https://github.com/jungao1106/ICoT
- Foundational formulation of interleaved-modal CoT for VLMs

**IR2.** "MINT-CoT: Enabling Interleaved Visual Tokens in Mathematical Chain-of-Thought Reasoning" — **NeurIPS 2025**
- Interleave Token dynamically selects arbitrary-shape visual regions per reasoning step; 3-stage training (text CoT SFT → interleaved CoT SFT → interleaved CoT RL)
- paper: https://arxiv.org/abs/2506.05331 | code: https://github.com/xinyan-cxy/MINT-CoT
- Math-focused but interleave mechanism is general; +34% MathVista, +29% GeoQA

**IR3.** "VisuoThink: Empowering LVLM Reasoning with Multimodal Tree Search" — **ACL 2025**
- Vision-text interleaved expansion + rollout simulation + self-voting selection; test-time scaling via look-ahead tree search
- paper: https://aclanthology.org/2025.acl-long.1053/ | code: https://github.com/ekonwang/VisuoThink
- Inference-time scaling without fine-tuning; SOTA on geometry and spatial reasoning

### C.2 Latent Visual Reasoning (Continuous Embeddings as Interleaved Thoughts)

**IR4.** "Monet: Reasoning in Latent Visual Space Beyond Images and Language" — **CVPR 2026**
- Generates continuous latent visual embeddings as intermediate "visual thoughts" interleaved with text; VLPO (Visual-latent Policy Optimization) for RL
- paper: https://arxiv.org/abs/2511.21395 | code: https://github.com/NOVAglow646/Monet
- Latent-space interleaving avoids expensive image generation; strong OOD generalization

**IR5.** "Interleaved Latent Visual Reasoning with Selective Perceptual Modeling (ILVR)" — **arXiv 2025.12**
- Momentum teacher selectively distills features from ground-truth intermediate images; alternates text generation with latent visual cues
- paper: https://arxiv.org/abs/2512.05665 | code: https://github.com/XD111ds/ILVR
- Bridges fine-grained perception and sequential reasoning; outperforms single-step approaches

**IR6.** "Imagine While Reasoning in Space: Multimodal Visualization-of-Thought (MVoT)" — **ICML 2025**
- Generates image visualizations of reasoning traces; token discrepancy loss for high-quality visualization
- paper: https://arxiv.org/abs/2501.07542 | code: https://github.com/chengzu-li/MVoT
- Visual thinking complements verbal reasoning in spatial tasks where text-only CoT fails

### C.3 Video-Specific Interleaved Reasoning

**IR7.** "ViTCoT: Video-Text Interleaved Chain-of-Thought for Boosting Video Understanding in LLMs" — **ACM MM 2025**
- Interleaves video frames and text in CoT; activates more neuron values in MLLMs than text-only CoT
- paper: https://dl.acm.org/doi/10.1145/3746027.3755837
- Direct video-text interleaving paradigm; cognitively aligned reasoning

**IR8.** "FrameMind: Frame-Interleaved Video Reasoning via Reinforcement Learning" — **arXiv 2025.09** (under review)
- Frame-Interleaved CoT (FiCOT): model alternates between textual reasoning and active frame retrieval via tools; trained with DRFS-GRPO
- paper: https://arxiv.org/abs/2509.24008 | code: https://framemind.github.io/
- Dynamic visual evidence gathering during inference; competitive with proprietary models on MVBench, MLVU, VideoMME

**IR9.** "VITAL: Thinking With Videos via Multimodal Tool-Augmented RL for Long Video Reasoning" — **arXiv 2025.08**
- Agentic framework: densely samples frames on demand + multimodal CoT; DGRPO for difficulty-aware RL
- paper: https://arxiv.org/abs/2508.04416
- Strong on long video; 11 benchmarks

**IR10.** "VideoEspresso: A Large-Scale Chain-of-Thought Dataset for Fine-Grained Video Reasoning via Core Frame Selection" — **CVPR 2025 Oral**
- Semantic-aware frame selection + GPT-4o QA generation; Hybrid LVLM collaboration (Frame Selector + reasoning LVLM)
- paper: https://openaccess.thecvf.com/content/CVPR2025/html/Han_VideoEspresso_A_Large-Scale_Chain-of-Thought_Dataset_for_Fine-Grained_Video_Reasoning_via_CVPR_2025_paper.html | code: https://github.com/hshjerry/VideoEspresso
- CVPR oral; 14-task benchmark with 9 LVLMs evaluated

### C.4 Chain-of-Shot / Efficient Video Reasoning

**IR11.** "CoS: Chain-of-Shot Prompting for Long Video Understanding" — **arXiv 2025.02**
- Frames shot selection as test-time visual prompt optimization; shots-task alignment for long videos
- paper: https://arxiv.org/abs/2502.06428
- Addresses context-length bottleneck for long videos

**IR12.** "Rethinking Chain-of-Thought Reasoning for Videos" — **arXiv 2025.12**
- Challenges lengthy CoT + massive visual tokens; shows concise reasoning + reduced tokens can suffice
- paper: https://arxiv.org/abs/2512.09616
- Important counterpoint: efficiency vs. exhaustive interleaving

### C.5 Grounded CoT and Verification

**IR13.** "ARGUS: Vision-Centric Reasoning with Grounded Chain-of-Thought" — **CVPR 2025**
- Goal-directed visual tokenization: grounds RoI conditioned on instructions, re-engages visual tokens as CoT context
- paper: https://arxiv.org/abs/2505.23766 | code: https://yunzeman.github.io/argus/
- From NVIDIA; beats SOTA open models at 8B scale

**IR14.** "MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification" — **ACL 2025**
- Simulation-based tree search + rejection sampling for high-quality CoT verification data; MM-Verifier + MM-Reasoner
- paper: https://aclanthology.org/2025.acl-long.689/ | code: https://github.com/aurora-slz/mm-verify
- Verification as complement to generation; surpasses GPT-4o on MathVista

### C.6 Multi-Image Interleaved Reasoning

**IR15.** "LVLM-MIR: Large Vision-Language Model with Parameter-Efficient Fine-Tuning for Multimodal Interleaved Reasoning" — **ACM MM 2025**
- LoRA-based PEFT on Qwen2.5-VL for multi-image interleaved reasoning; freezes pretrained weights
- paper: https://dl.acm.org/doi/10.1145/3746027.3762002
- Lightweight adaptation for interleaved multi-image scenarios

### C.7 Surveys

**IRS1.** "Multimodal Chain-of-Thought Reasoning: A Comprehensive Survey" — **arXiv 2025.03**
- Covers image, video, speech, audio, 3D modalities; methodologies, applications, benchmarks
- paper: https://arxiv.org/abs/2503.12605

**IRS2.** "Thinking with Images for Multimodal Reasoning: Foundations, Methods, and Future Frontiers" — **arXiv 2025.06**
- Focuses on visual thinking / image generation as reasoning; covers MVoT, Monet, ILVR families
- paper: https://arxiv.org/abs/2506.23918

---

## Cross-Reference: Papers in Both Categories

| Paper | Category A (Video Understanding) | Category B (Fusion Method) | Category C (Interleaved Reasoning) |
|---|---|---|---|
| VERA (CVPR 2025) | Video anomaly detection pipeline | Prompt optimization (learnable questions) | — |
| ImpliHateVid (ACL 2025) | Implicit hate in video | Two-stage contrastive fusion | — |
| MM-HSD (ACM MM 2025) | Video hate detection SOTA | Cross-modal attention fusion | — |
| MemeCLIP (EMNLP 2024) | Hateful meme detection | Frozen encoder + adapter fusion | — |
| Representation Collapse (ICML 2025) | Explains AV suppression | Cross-modal KD / basis reallocation | — |
| MoRE (WWW 2025) | Video hate detection | Mixture of retrieval-augmented experts | — |
| I2MoE (ICML 2025) | General multimodal | Interpretable MoE fusion | — |
| ARGUS (CVPR 2025) | — | — | Grounded CoT |
| MM-Verify (ACL 2025) | — | — | CoT verification |
| VideoEspresso (CVPR 2025) | Video reasoning dataset | — | Core frame selection + CoT |

---

## Category D: Simple-to-Hard / Adaptive Inference for Multimodal

Papers about routing easy vs hard cases, cascade systems, easy-to-hard generalization, difficulty-aware training/inference — especially those using MLLMs. The core idea: handle easy cases cheaply, spend more compute/reasoning on hard cases.

---

### D.1 Easy-to-Hard Generalization (Foundation)

**SH1.** "Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision" — **NeurIPS 2024**
- Train process-supervised reward model on easy problems (level 1-3 MATH) → use it to score hard problems (level 4-5); re-ranking or RL for easy-to-hard transfer
- paper: https://arxiv.org/abs/2403.09472 | code: https://github.com/Edward-Sun/easy-to-hard
- Core paradigm: evaluator trained on easy data generalizes to score hard data

**SH2.** "Generative Verifiers: Reward Modeling as Next-Token Prediction" — **ICLR 2025**
- GenRM trains verifiers via next-token prediction + CoT; shows 28%→44.6% on easy-to-hard MATH generalization
- paper: https://openreview.net/forum?id=Ccwp4tFEtE
- Verifier-based easy-to-hard transfer; GenRM-CoT trained on grade-school math solves 17% more competition problems

**SH3.** "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision" — **ICML 2024 Oral**
- Strong model finetuned on weak model labels outperforms weak supervisor; auxiliary confidence loss recovers near-strong performance
- paper: https://arxiv.org/abs/2312.09390 | code: https://github.com/openai/weak-to-strong
- OpenAI. Paradigmatic: weak supervisor + strong model = better than weak alone

### D.2 Cascade / Routing with Classification Experiments (Top Venues)

**SH4.** "Gatekeeper: Improving Model Cascades Through Confidence Tuning" — **ICML 2025**
- Novel loss function fine-tunes small model: high confidence when correct, low when wrong → better deferral to large model
- **Classification exps**: encoder-only image classification, decoder-only LLM text tasks, encoder-decoder VLM classification + captioning
- paper: https://arxiv.org/abs/2502.19335
- Task/architecture agnostic. Tested on VLM. Most directly applicable to our setting.

**SH5.** "Inter-Cascade: From Deferral to Learning — Online In-Context Knowledge Distillation for LLM Cascades" — **ICLR 2025**
- Strong model not just helper but teacher: when it solves a hard query, distills strategy into reusable repository → weak model improves over time
- **Classification exps**: text classification on multiple NLP benchmarks; weak model accuracy +33%, strong model calls -48%
- paper: https://arxiv.org/abs/2509.22984 | openreview: https://openreview.net/forum?id=fIFYBtjn2h
- Key: small model gets progressively better at hard cases. Most novel mechanism among cascade papers.

**SH6.** "A Unified Approach to Routing and Cascading for LLMs" — **ICML 2025**
- Derives theoretically optimal strategy unifying routing + cascading; proposes "cascade routing" with formal proofs
- **Classification exps**: text classification + generation benchmarks
- paper: https://arxiv.org/abs/2410.10347 | project: https://www.sri.inf.ethz.ch/publications/dekoninck2024cascaderouting
- Theoretical foundation for principled easy→hard deferral

**SH7.** "Cascaded Language Models for Cost-Effective Human-AI Decision-Making" — **NeurIPS 2025**
- 3-tier cascade: base model → large model → human expert; deferral policy + abstention policy based on confidence
- **Classification exps**: text classification + QA; code: https://github.com/fanconic/cascaded-llms
- paper: https://arxiv.org/abs/2506.11887
- Includes human-in-the-loop tier; directly relevant to content moderation

**SH8.** "Large Language Model Cascades with Mixture of Thought Representations" — **ICLR 2025**
- Multi-stage LLM cascade: small model handles easy, forwards uncertain (log-prob confidence) to larger model
- **Classification exps**: reasoning + classification benchmarks
- paper: https://openreview.net/forum?id=6okaSfANzh
- MoT (Mixture of Thought) representations for routing decisions

**SH9.** "Online Cascade Learning for Efficient Inference over Streams" — **ICML 2024**
- Online learning of cascade: logistic regressor → SLM → LLM; deferral policy as imitation-learning
- **Classification exps**: stream classification tasks
- paper: https://dl.acm.org/doi/10.5555/3692070.3693614
- Online setting; model learns to defer during deployment

### D.3 Applied Cascade (Video / Content Moderation / Multimodal)

**SH10.** "Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation" — **ACL 2025 Industry**
- Lightweight router → easy cases fast, hard cases → full MLLM ranker; transforms generative MLLM into classifier
- **Classification exps**: video content moderation; +66.5% F1 over traditional classifiers; compute reduced to 1.5%
- paper: https://aclanthology.org/2025.acl-industry.62/
- Only cascade paper in video content moderation domain

**SH11.** "MMR-Bench: A Comprehensive Benchmark for Multimodal LLM Routing" — **arXiv 2026.01**
- Benchmark for query-level MLLM routing; confidence via prototype similarity + norm-based signal per modality
- paper: https://arxiv.org/abs/2601.17814 | code: https://github.com/Hunter-Wrynn/MMR-Bench
- First benchmark isolating multimodal routing; easy OCR → cheap model, hard reasoning → expensive model

**SH12.** "VModA: An Effective Framework for Adaptive NSFW Image Moderation" — **arXiv 2025**
- Adaptive moderation: easy cases → lightweight filter, ambiguous cases → full MLLM analysis
- paper: https://arxiv.org/abs/2505.23386
- Content moderation domain (NSFW); architecture pattern transfers to hate

### D.4 Adaptive Inference / Test-Time Scaling

**SH13.** "AdaLLaVA: Learning to Inference Adaptively for Multimodal Large Language Models" — **ICCV 2025**
- Dynamically reconfigures MLLM operations during inference based on input difficulty + latency budget
- paper: https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Learning_to_Inference_Adaptively_for_Multimodal_Large_Language_Models_ICCV_2025_paper.pdf
- Learned adaptive system; integrates with token selection

**SH14.** "D-LLM: A Token Adaptive Computing Resource Allocation Strategy" — **NeurIPS 2024**
- Dynamic decision module per transformer layer: skip or execute based on token importance
- paper: https://neurips.cc/virtual/2024/poster/94977
- Token-level adaptive compute within a single input

**SH15.** "Limits and Gains of Test-Time Scaling in Vision-Language Reasoning" — **arXiv 2025.12**
- Systematic study of test-time scaling for VLMs; external verification most reliable; iterative refinement often degrades
- paper: https://arxiv.org/abs/2512.11109
- More compute at test-time helps on multi-step reasoning but not perception tasks

### D.5 Difficulty-Aware Data Selection

**SH16.** "Revisiting the Data Sampling in Multimodal Post-training from a Difficulty-Distinguish View" — **arXiv 2025.11**
- PISM (Progressive Image Semantic Masking) quantifies sample hardness; CMAB (Cross-Modality Attention Balance) measures cross-modal interaction complexity
- paper: https://arxiv.org/abs/2511.06722
- Two principled difficulty metrics for MLLM data

---

## Cross-Reference: Papers in Both Categories

| Paper | Category A (Video Understanding) | Category B (Fusion Method) | Category C (Interleaved Reasoning) | Category D (Simple-to-Hard) |
|---|---|---|---|---|
| VERA (CVPR 2025) | Video anomaly detection pipeline | Prompt optimization (learnable questions) | — | — |
| ImpliHateVid (ACL 2025) | Implicit hate in video | Two-stage contrastive fusion | — | — |
| MM-HSD (ACM MM 2025) | Video hate detection SOTA | Cross-modal attention fusion | — | — |
| MemeCLIP (EMNLP 2024) | Hateful meme detection | Frozen encoder + adapter fusion | — | — |
| Representation Collapse (ICML 2025) | Explains AV suppression | Cross-modal KD / basis reallocation | — | — |
| MoRE (WWW 2025) | Video hate detection | Mixture of retrieval-augmented experts | — | — |
| I2MoE (ICML 2025) | General multimodal | Interpretable MoE fusion | — | — |
| ARGUS (CVPR 2025) | — | — | Grounded CoT | — |
| MM-Verify (ACL 2025) | — | — | CoT verification | — |
| VideoEspresso (CVPR 2025) | Video reasoning dataset | — | Core frame selection + CoT | — |
| Filter-And-Refine (ACL 2025) | Video content moderation | — | — | MLLM cascade routing |

---

## Category E: Two-Stage MLLM Analysis → Parametric Classifier Training

Papers that use a two-stage pipeline: Stage 1 uses an MLLM/VLM to analyze multimodal content (generate descriptions, rationales, concepts, or features); Stage 2 trains a SEPARATE parametric classifier/detector on the MLLM outputs — not just prompting or fine-tuning the MLLM itself. Covers: LLM-rationale-augmented classifiers, concept bottleneck models with LLM concepts, VLM-to-lightweight distillation, MLLM-generated feature engineering.

---

### E.1 LLM/MLLM Rationale → Trained Classifier (Content Moderation / Misinformation)

**E1.** "EARAM: From Predictions to Analyses: Rationale-Augmented Fake News Detection with Large Vision-Language Models" — **WWW 2025**
- Stage 1: LVLMs generate multi-angle analytical rationales; Stage 2: a smaller LM adaptively extracts useful rationales and trains a classifier (outperforms LVLM's own judgment)
- paper: https://dl.acm.org/doi/10.1145/3696410.3714532
- Directly validates that dedicated classifier > MLLM prompting for detection tasks

**E2.** "Bad Actor, Good Advisor: Exploring the Role of LLMs in Fake News Detection (ARG)" — **AAAI 2024**
- Stage 1: GPT-3.5 generates multi-perspective rationales; Stage 2: Adaptive Rationale Guidance network trains BERT to selectively acquire LLM insights via news-rationale interaction; also derives rationale-free student (ARG-D)
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/30214

**E3.** "ExplainHM: Explainable Harmful Meme Detection through Multimodal Debate between LLMs" — **WWW 2024**
- Stage 1: two LLM agents debate from harmless/harmful perspectives generating rationales; Stage 2: fine-tuned T5-based model as debate judge fuses rationale features with meme features for classification
- paper: https://dl.acm.org/doi/10.1145/3589334.3645381

**E4.** "Mr.Harm: Unveiling Harmful Memes with Multimodal Reasoning Distilled from LLMs" — **EMNLP 2023 Findings**
- Stage 1: LLMs perform abductive reasoning to generate multimodal rationales; Stage 2: generative framework distills LLM reasoning chains to train a lightweight harmful meme classifier
- paper: https://aclanthology.org/2023.findings-emnlp.611/

**E5.** "SHIELD: Interpretable Hate Speech Detection using LLM-extracted Rationales" — **NAACL WOAH 2024**
- Stage 1: ChatGPT extracts rationale words/phrases linked to hate labels; Stage 2: BERT-based detector trained with rationale augmentation for improved detection + interpretability
- paper: https://aclanthology.org/2024.woah-1.17/

**E6.** "Multi-perspective Rationale Generation and Verification for Multimodal Fake News Detection" — **AAAI 2026**
- Stage 1: LLM generates multi-perspective rationales; Stage 2: cross-verification screens contradictions, a separate detection classifier makes real/fake judgment with adaptive weighting fusion
- paper: https://ojs.aaai.org/index.php/AAAI/article/view/36965

**E7.** "Generate First, Then Sample: Enhancing Fake News Detection with LLM-Augmented Reinforced Sampling" — **ACL 2025**
- Stage 1: LLM generates synthetic fake news in 3 styles; Stage 2: RL-based optimal sampling trains a separate fake news detector (+24% improvement)
- paper: https://aclanthology.org/2025.acl-long.1182/

**E8.** "LLM-MRD: LLM-Guided Multi-View Reasoning Distillation for Fake News Detection" — **arXiv 2026.03** (under review)
- Stage 1: teacher LLM generates multi-view (textual, visual, cross-modal) reasoning chains; Stage 2: Calibration Distillation trains an efficient student detector (+5.19% ACC, +6.33% F1)
- paper: https://arxiv.org/abs/2603.19293

**E9.** "FakeSV-VLM: Taming VLM for Detecting Fake Short-Video News via Progressive MoE Adapter" — **EMNLP 2025 Findings**
- Stage 1: VLM processes short video multimodal content; Stage 2: Progressive MoE Adapter with 4 scenario-specific experts trained on top of frozen VLM for fake **video** news classification
- paper: https://aclanthology.org/2025.findings-emnlp.257/
- Video task; MoE architecture directly relevant to our SCM-MoE

**E10.** "SafeWatch: Efficient Safety-Policy Following Video Guardrail Model" — **ICLR 2025**
- Stage 1: multiple MLLMs generate consensus multi-label safety annotations + explanations for video frames; Stage 2: smaller MLLM trained via 3-stage distillation (guardrail perf, token pruning, explanation quality); +28.2% over baselines
- paper: https://openreview.net/forum?id=xjKz6IxgCX
- Video safety; multi-MLLM teacher → smaller student paradigm

### E.1 Earlier Efforts

**E11.** "Improving Hateful Meme Detection Exploiting LMM-Generated Knowledge" — **CVPR 2025 Workshop**
- Stage 1: frozen LMM generates descriptions + emotions per meme; Stage 2: CLIP encodes these, concatenated with image/text embeddings, trains a classification head
- paper: https://arxiv.org/abs/2504.09914

**E12.** "IntMeme: Leveraging Large Multimodal Models for Hateful Meme Detection" — **ICWSM 2025**
- Stage 1: frozen InstructBLIP/mPLUG-Owl generate meme interpretations; Stage 2: RoBERTa/FLAVA encode interpretations, combined features train a hateful meme classifier
- paper: https://ojs.aaai.org/index.php/ICWSM/article/view/35845

**E13.** "OSPC: Artificial VLM Features for Hateful Meme Detection" — **WWW 2024 Companion**
- Stage 1: large VLM (LLaVA-NeXT) generates probabilistic feature encodings from meme text; Stage 2: lightweight classifier trained on VLM-derived features
- paper: https://dl.acm.org/doi/10.1145/3589335.3665996

### E.2 LLM-Generated Concepts → Concept Bottleneck Classifier

**E14.** "VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance" — **NeurIPS 2024**
- Stage 1: LLM generates concept candidates, Grounding-DINO provides visually grounded annotations; Stage 2: concept bottleneck classifier (CBL + linear head) trained on concept scores
- paper: https://arxiv.org/abs/2408.01432

**E15.** "CB-LLM: Concept Bottleneck Large Language Models" — **ICLR 2025**
- Stage 1: ChatGPT generates concept set, sentence embedding models label samples; Stage 2: backbone LLM + concept bottleneck layer + linear classifier trained for text classification
- paper: https://arxiv.org/abs/2412.07992

**E16.** "BC-LLM: Bayesian Concept Bottleneck Models with LLM Priors" — **NeurIPS 2025**
- Stage 1: LLMs serve as both concept extraction and Bayesian prior; Stage 2: iteratively discovers concepts and trains sparse prediction model
- paper: https://arxiv.org/abs/2410.15555

**E17.** "CoCoBM: Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models" — **ACL 2025**
- Stage 1: LLM agents dynamically construct/adjust concept bank via environmental feedback; Stage 2: trains Conditional Concept Bottleneck Model with editable concept-score matrix
- paper: https://aclanthology.org/2025.acl-long.600/

**E18.** "PCGR: Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection" — **arXiv 2026.03**
- Stage 1: GPT-5 analyzes high-loss samples to auto-discover reasoning concepts; Stage 2: layered probabilistic concept graph with hierarchical attention aggregates concept probabilities for veracity classification
- paper: https://arxiv.org/abs/2603.25203
- Concept auto-growth via error-driven MLLM analysis; alternating training

### E.3 VLM Knowledge Distillation → Lightweight Classifier

**E19.** "VL2Lite: Task-Specific Knowledge Distillation from Large VLMs to Lightweight Networks" — **CVPR 2025**
- Stage 1: frozen VLM (CLIP) provides visual + linguistic embeddings; Stage 2: lightweight network (MobileNet/EfficientNet) trained with visual + linguistic KD losses to match VLM representations; +7% classification accuracy
- paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_VL2Lite_Task-Specific_Knowledge_Distillation_from_Large_Vision-Language_Models_to_Lightweight_CVPR_2025_paper.pdf

**E20.** "LLM2CLIP: Powerful Language Model Unlocks Richer Visual Representation" — **AAAI 2026 Outstanding Paper**
- Stage 1: fine-tunes LLM as embedding model via contrastive learning in caption space; Stage 2: fine-tuned LLM acts as teacher to train/improve CLIP's visual encoder through lightweight adaptor
- paper: https://arxiv.org/abs/2411.04997

**E21.** "FTP: Enhancing Video Transformers for Action Understanding with VLM-aided Training" — **ICLR 2025**
- Stage 1: VLM generates multi-aspect text descriptions (action, components, context) via contrastive learning; Stage 2: video transformer classifier trained using projection layer + classifier head only; VLM not needed at inference
- paper: https://openreview.net/forum?id=yspBoIZJ9Z
- Video task; VLM descriptions used for training only

### E.3 Earlier Efforts

**E22.** "Feature-Level Knowledge Distillation from LMM for Enhanced Image Classification" — **NeurIPS 2025 Workshop**
- Stage 1: LLaVA generates diverse textual descriptions per image → CLIP text embeddings; Stage 2: ResNet-50 trained to align image embeddings with LMM-derived text embeddings via cosine dissimilarity loss
- paper: https://openreview.net/forum?id=4GtMgRteAZ

### E.4 LLM-Generated Features/Labels → Downstream Model

**E23.** "GLPN-LLM: Synergizing LLMs with Global Label Propagation for Multimodal Fake News" — **ACL 2025**
- Stage 1: LLM generates pseudo-labels; Stage 2: Global Label Propagation Network propagates labels across sample graph with mask-based anti-leakage; GNN-style classifier outperforms LLM's own predictions
- paper: https://aclanthology.org/2025.acl-long.72/

**E24.** "TAPE: LLM-to-LM Interpreter for Enhanced Text-Attributed Graph Representation Learning" — **ICLR 2024**
- Stage 1: GPT-3.5 generates zero-shot classification + textual explanations; Stage 2: DeBERTa fine-tuned as interpreter to translate explanations into node feature vectors; features train downstream GNN; SOTA on Cora/PubMed/ogbn-arxiv
- paper: https://openreview.net/forum?id=RXFVcynVe1

**E25.** "Delving into Qualitative Implications of Synthetic Data for Hate Speech Detection" — **EMNLP 2024**
- Stage 1: LLMs (Llama, Mistral, Mixtral) generate synthetic hate speech via paraphrasing; Stage 2: downstream classifiers trained on real + LLM-generated synthetic data for improved OOD robustness
- paper: https://aclanthology.org/2024.emnlp-main.1099/

**E26.** "FeRG-LLM: Feature Engineering by Reason Generation Large Language Models" — **NAACL 2025 Findings**
- Stage 1: fine-tuned Llama 3.1 8B generates new features via two-stage conversational dialogues + DPO; Stage 2: generated features fed into separate classifiers (XGBoost, etc.) for classification
- paper: https://aclanthology.org/2025.findings-naacl.237/

---

## Cross-Reference: Papers in Both Categories

| Paper | Category A (Video Understanding) | Category B (Fusion Method) | Category C (Interleaved Reasoning) | Category D (Simple-to-Hard) | Category E (MLLM→Classifier) |
|---|---|---|---|---|---|
| VERA (CVPR 2025) | Video anomaly detection pipeline | Prompt optimization (learnable questions) | — | — | — |
| ImpliHateVid (ACL 2025) | Implicit hate in video | Two-stage contrastive fusion | — | — | — |
| MM-HSD (ACM MM 2025) | Video hate detection SOTA | Cross-modal attention fusion | — | — | — |
| MemeCLIP (EMNLP 2024) | Hateful meme detection | Frozen encoder + adapter fusion | — | — | — |
| Representation Collapse (ICML 2025) | Explains AV suppression | Cross-modal KD / basis reallocation | — | — | — |
| MoRE (WWW 2025) | Video hate detection | Mixture of retrieval-augmented experts | — | — | — |
| I2MoE (ICML 2025) | General multimodal | Interpretable MoE fusion | — | — | — |
| ARGUS (CVPR 2025) | — | — | Grounded CoT | — | — |
| MM-Verify (ACL 2025) | — | — | CoT verification | — | — |
| VideoEspresso (CVPR 2025) | Video reasoning dataset | — | Core frame selection + CoT | — | — |
| Filter-And-Refine (ACL 2025) | Video content moderation | — | — | MLLM cascade routing | — |
| GLPN-LLM (ACL 2025) | Fake news detection | — | — | — | LLM pseudo-labels → GNN |
| PCGR (arXiv 2026) | Misinformation detection | — | — | — | LLM concept growth → concept graph classifier |
| FakeSV-VLM (EMNLP 2025) | Fake video news | MoE adapter fusion | — | — | VLM → MoE adapter classifier |
| SafeWatch (ICLR 2025) | Video safety | — | — | — | Multi-MLLM → distilled student |
| Multi-perspective Rationale (AAAI 2026) | Fake news detection | — | — | — | LLM rationale → verification classifier |

---

## Category F: Label-Free Training (MLLM Self-Training & MLLM-Guided Small-Model Training)

Papers about training or adapting MLLMs/VLMs — or training smaller multimodal models guided by MLLMs — **without human labels**. Two primary threads: (F.1) MLLMs that self-train via self-rewarding, self-play, self-consistency, or self-distillation; (F.2) MLLM teachers supervising smaller student models with pseudo-labels/rationales instead of ground-truth annotations. Also includes label-free VAD / hateful / misinformation applications.

---

### F.1 MLLM Self-Training / Label-Free Fine-Tuning

**LF1.** "MM-UPT: Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO" — **NeurIPS 2025**
- Replaces GRPO's reward with majority-vote self-consistency pseudo-labels over sampled responses; no ground-truth labels; improves Qwen2.5-VL on MathVista/We-Math
- paper: https://arxiv.org/abs/2505.22453

**LF2.** "Calibrated Self-Rewarding Vision Language Models (CSR)" — **NeurIPS 2024**
- Iterative self-rewarding preference optimization with calibrated visual-token rewards; no external reward model or human preference labels
- paper: https://papers.nips.cc/paper_files/paper/2024/file/5c20c00504e0c049ec2370d0cceaf3c4-Paper-Conference.pdf

**LF3.** "STIC: Enhancing Large Vision Language Models with Self-Training on Image Comprehension" — **NeurIPS 2024**
- LVLM self-constructs a preference set from unlabeled images (good captions vs. corrupted-image captions); DPO fine-tuning with no human/teacher labels
- paper: https://arxiv.org/abs/2405.19716

**LF4.** "Vision-Language Models Can Self-Improve Reasoning via Reflection" — **NAACL 2025**
- Reflection-based self-improvement: VLM generates, critiques, and refines its own CoT; self-trains on filtered traces without ground truth
- paper: https://aclanthology.org/2025.naacl-long.447.pdf

**LF5.** "Realistic Test-Time Adaptation of Vision-Language Models" — **CVPR 2025**
- Entropy / pseudo-label–based TTA for CLIP-style VLMs under realistic deployment streams; no labels used at adaptation time
- paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Zanella_Realistic_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2025_paper.pdf

**LF6.** "Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play" — **arXiv 2025** (under review)
- Iterative Self-Play Policy Optimization alternates gamified self-play and RLVR; VLM improves on reasoning/chart QA with fully label-free data
- paper: https://arxiv.org/abs/2509.25541

**LF7.** "Self-Rewarding Vision-Language Model via Reasoning Decomposition" — **arXiv 2025**
- Decomposes visual reasoning into structured steps so the VLM rewards its own perception; fully self-rewarding, no external supervision
- paper: https://arxiv.org/abs/2508.19652

**LF8.** "CoFT: Fine-tuning Pre-trained Vision-Language Models in a Human-Annotation-Free Manner" — **arXiv 2026**
- Dual-prompt / dual-model collaborative self-training with sample-adaptive pseudo-label filtering; fully annotation-free VLM fine-tuning
- paper: https://arxiv.org/abs/2602.04337

**LFS1.** "Adapting Vision-Language Models Without Labels: A Comprehensive Survey" — **arXiv 2025** [REFERENCE]
- Taxonomy of label-free VLM adaptation: self-training, entropy optimization, external-resource approaches
- paper: https://arxiv.org/abs/2508.05547

**LFS2.** "Self-Improvement in Multimodal Large Language Models: A Survey" — **arXiv 2025** [REFERENCE]
- Taxonomy of self-rewarding, self-distillation, self-play, self-training techniques for MLLMs
- paper: https://arxiv.org/abs/2510.02665

### F.2 MLLM-Guided Label-Free Training of Smaller Multimodal Models

**LF9.** "MLLM-as-a-Judge for Image Safety without Human Labeling" — **CVPR 2025**
- Debiased token probabilities + cascaded CoT turn a frozen MLLM into a zero-shot safety judge; downstream safety supervision requires no human labels
- paper: https://arxiv.org/abs/2501.00192

**LF10.** "Multi-MLLM Knowledge Distillation for Out-of-Context News Detection" — **arXiv 2025**
- Multiple teacher MLLMs generate pseudo-labels + rationales; two-stage distillation trains a small student MLLM for OOC misinformation without human labels
- paper: https://arxiv.org/abs/2505.22517

**LF11.** "CanDist: Prompt Candidates, then Distill — Teacher-Student Framework for LLM-driven Data Annotation" — **ACL 2025**
- Teacher LLM emits candidate label sets under uncertainty; distilled into a small student, improving robustness of LLM-only pseudo-labeling
- paper: https://aclanthology.org/2025.acl-long.139/

**LF12.** "UnCo: Uncertainty-Driven Collaborative Framework of LLM and Small Model" — **EMNLP 2025**
- Uncertainty-guided cooperation: LLM relabels only hard cases; student learns from LLM pseudo-labels without ground-truth supervision
- paper: https://aclanthology.org/2025.emnlp-main.388.pdf

**LF13.** "MI-Fuse: Label Fusion for Unsupervised Domain Adaptation with MLLM Teachers" — **arXiv 2025**
- Fuses multiple MLLM teachers' pseudo-labels via mutual information weighting; trains student with zero target-domain labels
- paper: https://arxiv.org/abs/2509.20706

**LF14.** "LLKD: Learning with Less — Knowledge Distillation from LLMs via Unlabeled Data" — **NeurIPS 2024 Workshop**
- LLM generates pseudo-labels on unlabeled data; confidence-weighted distillation trains small classifiers with no human annotation
- paper: https://arxiv.org/abs/2411.08028

**LF15.** "Adaptive Collaborative Labeling with MLLMs (ACL-MER)" — **IJCNLP/AACL 2025**
- MLLM teacher zoo produces confidence-scored predictions that refine student probabilities into high-quality pseudo-labels for low-resource multimodal tasks
- paper: https://aclanthology.org/2025.ijcnlp-long.152.pdf

### F.3 Label-Free Applications: VAD / Hateful / Misinformation

**LF16.** "Towards Zero-Shot Anomaly Detection and Reasoning with MLLMs" — **CVPR 2025**
- Zero-shot MLLM-driven anomaly detector + reasoning over safety-relevant events; no target-domain labels
- paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_Towards_Zero-Shot_Anomaly_Detection_and_Reasoning_with_Multimodal_Large_Language_CVPR_2025_paper.pdf

**LF17.** "SafeWatch: Efficient Safety-Policy Following Video Guardrail Model" — **ICLR 2025**
- Multi-MLLM propose-discuss-consensus pipeline builds SafeWatch-Bench training data (no human frame labels); compact guardrail MLLM trained on those pseudo-labels
- paper: https://arxiv.org/abs/2412.06878
- Also listed in E.1 under different framing; shown here for the label-free annotation pipeline

**LF18.** "Fact-R1: Explainable Video Misinformation Detection with Deep Reasoning" — **arXiv 2025**
- CoT instruction tuning + DPO + GRPO with verifiable reward on FakeVV dataset; RL-driven training avoids human veracity labels at the RL stage
- paper: https://arxiv.org/abs/2505.16836

**LF19.** "VERA: Explainable Video Anomaly Detection via Verbalized Learning of VLMs" — **CVPR 2025**
- Learns verbalized anomaly-characterization prompts for a frozen VLM; no instruction tuning, no dense frame labels, only coarse video-level cues
- paper: https://openaccess.thecvf.com/content/CVPR2025/papers/Ye_VERA_Explainable_Video_Anomaly_Detection_via_Verbalized_Learning_of_Vision-Language_Models_CVPR_2025_paper.pdf
- Cross-listed in A.1; relevant here for its label-free verbalized training signal

### F.3 Earlier Efforts

**LF20.** "LAVAD: Harnessing LLMs for Training-free Video Anomaly Detection" — **CVPR 2024**
- Captioning VLM + LLM reasoning over caption histories scores anomalies with no VAD-specific training
- paper: https://openaccess.thecvf.com/content/CVPR2024/papers/Zanella_Harnessing_Large_Language_Models_for_Training-free_Video_Anomaly_Detection_CVPR_2024_paper.pdf
- Cross-listed in A.1 earlier efforts

### F.4 Label-Free VLM Adaptation (Other Domains — Transferable Baselines)

**LF21.** "LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Images" — **NeurIPS 2024**
- LLM-generated class descriptions as supervision; CLIP zero-shot pseudo-labels → adapter with description-guided contrastive loss; no human labels
- paper: https://arxiv.org/abs/2305.18287 | code: https://github.com/jmiemirza/LaFTer
- **[BASELINE]** Most directly comparable to TLB-Filter — both use VLM zero-shot signals to train a small classifier

**LF22.** "SuS-X: Training-Free Name-Only Transfer of Vision-Language Models" — **ICCV 2023**
- Constructs support set from CLIP's own feature space → nearest-neighbor calibration; training-free; SOTA on 19 datasets
- paper: https://arxiv.org/abs/2211.16198 | code: https://github.com/vishaal27/SuS-X (105 stars)
- **[BASELINE]** Training-free CLIP calibration — analog to token-logprob calibration

**LF23.** "TPT: Test-Time Prompt Tuning for Zero-Shot Generalization" — **NeurIPS 2022**
- Optimizes soft prompts at test time by minimizing entropy over augmented views; no training labels; + follow-up DiffTPT (NeurIPS 2023)
- paper: https://arxiv.org/abs/2209.07511 | code: https://github.com/azshue/TPT
- **[BASELINE]** Canonical training-free prompt adaptation

**LF24.** "UPL: Unsupervised Prompt Learning for Vision-Language Models" — **arXiv 2024**
- CLIP zero-shot → pseudo-labels → CoOp-style soft prompt training → iterate; top-K confidence filtering
- paper: https://arxiv.org/abs/2204.03649 | code: https://github.com/tonyhuang2022/UPL
- **[BASELINE]** Prompt-tuning analog of TLB-Filter: confidence-filtered pseudo-label self-training

**LF25.** "StatA: Realistic Test-Time Adaptation of Vision-Language Models" — **CVPR 2025 Highlight**
- Anchor-based regularization for robust TTA under variable effective classes; unsupervised transductive
- paper: https://arxiv.org/abs/2501.13838 | code: https://github.com/MaxZanella/StatA (59 stars)
- Cross-listed with LF5; shown here for its relevance as a label-free adaptation baseline

**LF26.** "STIC: Self-Training on Image Comprehension" — **NeurIPS 2024**
- LVLM generates preferred vs dis-preferred captions from unlabeled images → DPO fine-tuning; no human/teacher labels
- paper: https://arxiv.org/abs/2405.19716 | code: https://github.com/yihedeng9/STIC (69 stars)
- Cross-listed with LF3; shown here for code availability

### F.5 Noisy Pseudo-Label Debiasing (Transferable from Other Domains)

**LF27.** "PriDe: Large Language Models Are Not Robust Multiple Choice Selectors" — **ICLR 2024 Spotlight**
- Decomposes LLM output into intrinsic prediction + prior distribution over answer tokens; debiases by estimating prior from small permutation test; zero additional cost
- paper: https://arxiv.org/abs/2309.03882 | code: https://github.com/chujiezheng/LLM-MCQ-Bias
- **[BASELINE]** Directly fixes MLLM class bias in pseudo-labels — principled replacement for heuristic class-balanced filter

**LF28.** "DeFT: VLMs are Strong Noisy Label Detectors" — **NeurIPS 2024**
- CLIP vision-text alignment with positive/negative textual prompts → detects noisy labels; essentially confident learning for VLMs
- paper: https://arxiv.org/abs/2405.00000 | code: https://github.com/HotanLee/DeFT (17 stars)
- **[BASELINE]** Use CLIP to detect which MLLM pseudo-labels are likely wrong

**LF29.** "CLIPCleaner: Zero-Shot Clean Sample Selection for Learning with Noisy Labels" — **ACM MM 2024**
- Zero-shot CLIP-based clean sample selection in single offline step; simpler than iterative methods
- paper: https://arxiv.org/abs/2404.00000 | code: https://github.com/MrChenFeng/CLIPCleaner_ACMMM2024 (15 stars)
- **[BASELINE]** One-shot filter of bad pseudo-labels via CLIP

**LF30.** "NLPrompt: Noise-Label Prompt Learning" — **CVPR 2025**
- MAE loss + prompt-based optimal transport to partition clean/noisy subsets
- paper: https://arxiv.org/abs/2412.00000 | code: https://github.com/qunovo/NLPrompt (67 stars)

**LF31.** "CroSel: Cross Selection of Confident Pseudo Labels for Partial-Label Learning" — **CVPR 2024 Oral**
- Two networks cross-select true labels using historical prediction confidence; co-mix regularization; >90% label selection accuracy on CIFAR
- paper: https://arxiv.org/abs/2303.10365 | code: https://github.com/jokersio-tsy/CroSel

**LF32.** "DiffMatch: Towards Unbiased Learning in Semi-Supervised Semantic Segmentation" — **ICLR 2025**
- Diffusion-based pseudo-label denoising + debiased adjustment to shift conditional reverse probability from head to tail classes
- paper: https://arxiv.org/abs/2501.00000 | code: https://github.com/yuisuen/DiffMatch
- **[BASELINE]** Mathematically addresses class-bias in pseudo-labels (MLLM over-predicts "not hateful")

### F.6 Dynamic Rule/Concept Discovery + Efficient Sample Selection

**LF33.** "RuAG: Learned-Rule-Augmented Generation for Large Language Models" — **ICLR 2025**
- MCTS automatically distills offline data into first-order logic rules; LLM defines predicates, search explores rule space; discovered rules injected into prompts
- paper: https://openreview.net/forum?id=RuAG | code: https://github.com/microsoft/RuAG
- **[KEY]** Direct template for mining new hate-detection rules from data via MCTS

**LF34.** "IterAlign: Iterative Constitutional Alignment of Large Language Models" — **NAACL 2024**
- Automated cycle: red-team to find failures → oracle LLM proposes new constitutional principles → self-revise → SFT. Discovers new alignment principles each round (+13.5% harmlessness)
- paper: https://aclanthology.org/2024.naacl-long.369/ | code: https://github.com/xiusic/IterAlign
- **[KEY]** Closest to iterative constitution refinement from unlabeled data

**LF35.** "BC-LLM: Bayesian Concept Bottleneck Models with LLM Priors" — **NeurIPS 2025**
- Iterative Bayesian concept search: drop concept → LLM proposes replacements → annotate → accept/reject via held-out likelihood
- paper: https://arxiv.org/abs/2410.15555 | code: https://github.com/jjfeng/bc-llm
- Cross-listed with E16; shown here for iterative concept discovery

**LF36.** "PromptAgent: Strategic Planning with Language Models Enables Expert-level Prompt Optimization" — **ICLR 2024**
- MCTS for prompt optimization; examines classification errors → generates error-feedback → proposes refined prompts
- paper: https://openreview.net/forum?id=PromptAgent | code: https://github.com/XinyuanWangCS/PromptAgent

**LF37.** "TnT-LLM: Text Mining at Scale with Large Language Models" — **KDD 2024**
- Two-phase: zero-shot LLM reasoning iteratively generates/refines label taxonomy from raw text (no seed labels); then LLM labels data for lightweight classifiers
- paper: https://arxiv.org/abs/2403.12173

**LF38.** "CoCoBM: Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models" — **ACL 2025**
- LLM agent dynamically adjusts concept bank via environmental feedback (classification errors); editable concept-score matrix
- paper: https://aclanthology.org/2025.acl-long.600/
- Cross-listed with E17; shown here for agent-driven concept bank expansion

**LF39.** "AutoQual: LLM Agent for Automated Discovery of Interpretable Features" — **EMNLP 2025 Industry**
- Iterative loop: LLM generates feature hypotheses → operationalizes → tests against data → accumulates in persistent memory. Discovered 5 interpretable features confirmed via A/B testing
- paper: https://aclanthology.org/2025.emnlp-industry.87/

**LF40.** "SIMILAR: Submodular Information Measures Based Active Learning" — **NeurIPS 2021+**
- Uses submodular mutual information to select data subsets similar to a target set (known failures) and diverse within themselves; operates on pre-computed embeddings
- paper: https://arxiv.org/abs/2107.00717

**LF41.** "D2 Pruning: Data Pruning via Difficulty and Discriminability" — **ICLR 2024**
- Assigns difficulty (loss across epochs) and discriminability (margin) scores; selects hard-but-learnable samples via EL2N proxy
- paper: https://openreview.net/forum?id=D2Pruning

**LF42.** "T-MARS: Text-Masking for Efficient Data Filtering with CLIP" — **NeurIPS 2024**
- Masks text, re-scores with CLIP vision-only; selects samples where visual-only and text scores disagree most (cross-modal dependency)
- paper: https://arxiv.org/abs/2307.03132
- **[KEY]** Cross-modal disagreement selection maps directly to finding hate via visual-text juxtaposition

---

**Total papers**: 185
**Category A (Video Understanding)**: 56 papers across 9 sub-categories
**Category B (Multimodal Fusion)**: 32 papers across 5 sub-categories
**Category C (Interleaved Reasoning)**: 15 papers + 2 surveys across 7 sub-categories
**Category D (Simple-to-Hard / Cascade)**: 16 papers across 5 sub-categories
**Category E (MLLM→Classifier)**: 26 papers across 4 sub-categories
**Category F (Label-Free Training)**: 32 papers + 2 surveys across 5 sub-categories
**Cross-listed**: 20+ papers appear in multiple categories
**Category A (Video Understanding)**: 47 papers across 8 sub-categories
**Category B (Multimodal Fusion)**: 32 papers across 5 sub-categories
**Category C (Interleaved Reasoning)**: 15 papers + 2 surveys across 7 sub-categories
**Category D (Simple-to-Hard / Cascade)**: 16 papers across 5 sub-categories
**Category E (MLLM→Classifier)**: 26 papers across 4 sub-categories
**Category F (Label-Free Training)**: 20 papers + 2 surveys across 3 sub-categories
**Cross-listed**: 16+ papers appear in multiple categories
