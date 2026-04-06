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

## VI. Open Insights Worth Building On

These insights from 150+ rounds of discussion are still valid and usable:

1. **Stability is an underexplored evaluation axis** — seed variance 5-8pp is huge, nobody studies this for hate video
2. **The bottleneck is boundary fitting, not representation** — MLP ≈ complex model, generic ≈ theory prompt
3. **Hate is relational** — target-act-stance structure, contextual polarity flips (quotation, counterspeech, satire, reclaimed language)
4. **MLLM textualization as semantic interface** — video multimodal content becomes compressible into text, but something is lost in that compression
5. **Paradigm shifts 2025-2026**: static classifiers → adaptive systems, passive encoding → active evidence gathering
6. **Cross-dataset transfer** as high-value evaluation dimension
7. **Legitimate uncertainty vs arbitrary instability** — important nuance for moderation systems
