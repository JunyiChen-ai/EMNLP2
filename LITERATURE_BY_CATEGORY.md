# Literature Organized by Two Categories

**Date**: 2026-04-06
**Source**: Refreshed from prior 44-paper collection + new search (2025-2026 top venues)

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

### A.7 Earlier Efforts

**N3.** "VadCLIP: Frozen CLIP for Weakly Supervised Video Anomaly Detection" — **arXiv 2024**
- Frozen CLIP -> fine-grained temporal text-video matching -> anomaly scoring
- paper: https://arxiv.org/abs/2308.11681 | code: https://github.com/nwpu-zxr/VadCLIP

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

**Total papers**: 72
**Category A (Video Understanding)**: 47 papers across 8 sub-categories
**Category B (Multimodal Fusion)**: 32 papers across 5 sub-categories
**Cross-listed**: 7 papers appear in both categories
**New papers added (this update)**: 28
