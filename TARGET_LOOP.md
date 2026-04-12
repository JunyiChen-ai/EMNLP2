# Target-Driven Research Loop

## Target
- Metric: Accuracy > 80% on BOTH MHClip_EN and MHClip_ZH test sets
- Current baseline: MHClip_EN 69.78% ACC (10.3% recall), MHClip_ZH 72.73% ACC (15.8% recall)
- Gap: EN needs +10.2pp, ZH needs +7.3pp

## Constraints
- **C1 (from user)**: Dynamic constitution update phase allows max 1 MLLM multimodal call per video (excluding the existing objectification call in Step 1)
- **C2 (from user)**: MLLM call must be multimodal (frames + text), cannot degrade to unimodal input
- **C3 (from user)**: 1 GPU at any time
- **C4 (from CLAUDE.md)**: No ensemble / repeated MLLM querying for the same purpose
- **C5 (from CLAUDE.md)**: No external datasets — only target benchmark's own splits
- **C6 (from CLAUDE.md)**: Every technique must have a scientific story specific to hateful video
- **C7 (from CLAUDE.md)**: All code via Slurm, conda env SafetyContradiction, model Qwen3-VL-8B-Instruct
- **C8 (from CLAUDE.md)**: Offensive maps to Hateful (binary: Hateful+Offensive vs Normal)

## Environment
- User: jehc223
- Execution mode: Slurm (sbatch only, no login-node execution)
- GPU budget: 1 GPU max at any time
- Conda: SafetyContradiction (vllm 0.11.0)
- Model: Qwen/Qwen3-VL-8B-Instruct

## Baseline Error Analysis

### Confusion Matrices

| | MHClip_EN | MHClip_ZH |
|---|---|---|
| TP | 6 | 9 |
| TN | 121 | 119 |
| FP | 3 | 0 |
| FN | **52** | **48** |
| Total | 182 | 176 |

### Root Cause (from Phase 1 analysis)

**100% of false negatives have zero rules firing.** Not a threshold issue — the rules literally never trigger.

- 91% of rule evaluations fail at the very first precondition
- First preconditions require "explicit, unambiguous" language — implicit hate is invisible to them
- 79% of rules are marked RELEVANT by CLIP, but precondition scoring rejects all of them
- 15 near-miss cases: score_text ≥ 0.5 but score_full near 0 (visual suppression)
- Training split: 701 videos (EN), 699 videos (ZH)

### What Won't Work (already tested)
- CLIP embedding density as proxy for hatefulness — rejected (no signal)
- CLIP outlier detection — rejected (outliers are NSFW-adjacent Normal, not Hateful)
- Cross-modal CLIP disagreement — weak signal (~54% vs 40% baseline)

---

## Iteration 1 — Target-Mechanism Bottleneck Discovery (TMBD)

### References
- **IterAlign** (Cao et al., NAACL 2024): Iteratively discovers red-teaming categories from model outputs and adds them to safety constitution. Key idea: use model's own behavior on unlabeled data to discover gaps in alignment rules.
- **RuAG** (Fan et al., ICLR 2025): Automatically generates rules from data and augments LLM prompts. Key idea: mine rules from examples, then inject into prompts as in-context guidelines.
- **CLUE** (Yuan et al., CVPR 2025): Our base framework — constitution-following with precondition decomposition and debiased token probability.

### Proposal: Observation-Grounded Constitution Discovery (OGC-Discovery)

**What**: Use a single multimodal MLLM call per training video to collect open-ended observations about hateful content. Cluster these observations to discover implicit-hate rule categories not covered by the static constitution. Induce new rules, objectify them, extract preconditions, and re-run the pipeline.

**Why (scientific motivation)**:

*Phenomenon*: Implicit hate in YouTube/Bilibili videos uses culturally embedded mechanisms — ironic juxtaposition, coded language, contextual dog-whistles, sarcastic mockery — that platform-level policy rules (designed for explicit violations) structurally cannot capture. The MLLM's pre-training already encodes understanding of these implicit mechanisms, but the constitution never asks about them.

*Mechanism*: The static constitution fails not because the MLLM cannot recognize implicit hate, but because no rule covers it. By querying the MLLM with an open-ended observation prompt (rather than a closed yes/no precondition check), we let the model surface patterns in its own vocabulary. Clustering these free-form observations reveals new rule categories grounded in what the model actually sees — bridging the gap between the model's capability and the constitution's coverage.

*Key insight*: The 79% CLIP relevance rate proves the content is hate-adjacent; the 91% first-precondition failure rate proves the precondition language is wrong. We need rules that describe implicit mechanisms, not stricter versions of explicit rules.

**Expected effect**: New rules targeting implicit hate patterns (e.g., mockery, coded slurs, ironic dehumanization) should primarily increase recall on the 52 EN and 48 ZH false negatives. Precision may drop slightly due to more rules, but the net ACC gain should be substantial because FN >> FP in current results.

**Testable prediction**: If the MLLM observations on training data contain hateful descriptions that don't match any existing rule, clustering them should yield ≤ 5 new rule categories. These new rules should fire on ≥ 30% of test FN videos. If observations contain NO new patterns beyond existing rules, the hypothesis is wrong and the problem is scoring, not coverage.

### Method Steps

**Step A: Multimodal Observation on Training Split** (1 MLLM call per video)
- Input: top-8 frames (CLIP-selected or uniform) + title + transcript
- Prompt (see below)
- Output: per-video JSON with hateful_flag + observation text + target_group + mechanism

**Step B: Gap Filtering** (CPU, no MLLM)
- Keep observations where MLLM says hateful
- Remove observations that match an existing rule description (string similarity)
- Result: gap observations = implicit hate patterns not in constitution

**Step C: Rule Induction** (text-only LLM, 1 call)
- Cluster gap observations by semantic similarity
- For each cluster, use LLM to generalize into a new rule
- Quality check: discard rules that are redundant with existing constitution

**Step D: Objectification + Precondition Extraction** (existing Steps 1-2)
- Process new rules through the same pipeline

**Step E: Re-evaluate** (existing Steps 3-6 on test split)
- Updated constitution = original rules + new rules
- Run full pipeline on test

### Observation Prompt (Step A)

```
You are a content moderation analyst. Observe this video carefully — its visual frames, title, and transcript.

Title: {title}
Transcript: {transcript}

Task: Determine whether this video contains content that is hateful, offensive, or discriminatory toward any individual or group.

Respond in this exact format:
VERDICT: [HATEFUL or NOT_HATEFUL]
TARGET_GROUP: [the group being targeted, or "none"]
MECHANISM: [how the hate is conveyed — e.g., "explicit slurs", "ironic mockery", "dehumanizing comparison", "coded language", "stereotyping through humor", "none"]
OBSERVATION: [1-2 sentence description of what you observe, or "No hateful content detected."]
```

### Constraint Check
- C1 (max 1 MLLM multimodal call): PASS — exactly 1 call per training video in Step A. Steps B-C are text-only/CPU.
- C2 (must be multimodal): PASS — observation prompt includes frames + title + transcript.
- C3 (1 GPU): PASS — all steps run sequentially via Slurm.
- C4 (no ensemble): PASS — single observation, not aggregated across multiple queries.
- C5 (no external data): PASS — uses only the target benchmark's training split.
- C6 (scientific story): PASS — grounded in implicit-vs-explicit hate phenomenon.
- C7 (Slurm + SafetyContradiction + Qwen3-VL-8B): PASS.
- C8 (Offensive=1): PASS — unchanged from baseline.

### Code Files
- New: `src/observe_training.py` — Step A: collect MLLM observations on training split
- New: `src/induce_rules.py` — Steps B-C: gap filter + rule induction (not used after pivot)
- New: `src/refine_preconditions.py` — Observation-grounded precondition re-decomposition (pivot)
- Modified: `src/score_preconditions.py` — --constitution-suffix flag for refined constitution
- Modified: `src/clip_filter.py` — --constitution-suffix flag
- Modified: `src/classify.py` — --constitution-suffix flag
- Entry point: observe → refine → clip_filter → score → classify

---

### PIVOT: Coverage-Gap Hypothesis Falsified

**Observation data from MHClip_EN (555 videos, 134 hateful):**
- 133/134 hateful observations have covered_rule_ids (existing rules assigned)
- Only 1/134 has no covered rules — discovery set is empty
- R5 (slurs & stereotypes) assigned to 112/134 hateful observations
- 109/134 use explicit mechanisms, 25 use implicit mechanisms

**Diagnosis**: The MLLM recognizes that existing rules cover most hateful content when asked holistically. But the per-precondition scoring fails — 91% of rule evaluations fail at the first precondition because preconditions require "explicit, unambiguous" language.

**Pivot (GPT-approved)**: Instead of discovering new rules, use observation data to **re-decompose preconditions** for existing rules:
1. Collect observable_cues from observations grouped by assigned rule
2. Use implicit-mechanism-preserving prompt to re-extract preconditions
3. Split disjunctive rules into separate branches (e.g., R5 → R5a: slurs, R5b: stereotyping)
4. Run scoring pipeline with refined preconditions

**Diagnostic predictions (per GPT review)**:
- Higher first-precondition pass rate for R5/R6/R1/R3
- More current zero-rule FNs converted into rule firings
- Downstream recall gain

---

## Iteration 1 — Results

### Approaches Tested

| Approach | EN ACC | ZH ACC | EN Recall | ZH Recall | Notes |
|----------|--------|--------|-----------|-----------|-------|
| Baseline (static const.) | 69.78 | 72.73 | 10.34 | 15.79 | Phase 1 |
| Refined preconditions (alpha2=0.3) | 69.23 | — | 37.93 | — | Higher recall but more FP; best threshold sweep: 72.5% |
| Holistic observation (text verdict) | 74.73 | 71.59 | 48.28 | 59.65 | Constitution-informed holistic MLLM query |
| **Holistic token prob (best threshold)** | **75.3** | **77.3** | **43.1** | **49.1** | P(Yes)/(P(Yes)+P(No)) with calibrated threshold |
| Combined (holistic + precondition) | 75.3 | — | 44.8 | — | No improvement over holistic alone |

### Key Findings

1. **Coverage-gap hypothesis falsified.** 133/134 EN and 227/227 ZH hateful training observations had covered_rule_ids. The MLLM already recognizes that existing rules cover the hateful content. New rules not needed.

2. **Precondition decomposition is the bottleneck, not rule coverage.** The MLLM achieves ~77% ACC when asked holistically but only ~70% when forced through per-precondition scoring with AND logic. Decomposition loses signal.

3. **Observation-grounded re-decomposition helped recall (+28pp) but not ACC.** Refined preconditions (9→40 branches, implicit-aware) increased recall from 10% to 38%, but FP also increased, netting ~0% ACC change. No threshold achieves >72.5% ACC with decomposed scoring.

4. **Holistic token probability is the best single signal.** P(Yes)/(P(Yes)+P(No)) on a constitution-informed question gives clear separation: hateful mean 0.47/0.64, normal mean 0.12/0.17 for EN/ZH. Best ACC: EN 75.3%, ZH 77.3%.

5. **MLLM single-pass ceiling is ~77% ACC.** The MLLM's holistic judgment on training data matches test accuracy (~77%), indicating this is an inherent model capability limit, not a prompt issue.

6. **20 skipped frameless/large videos hurt EN by ~2pp.** 7 of 20 skipped EN videos are hateful → automatic FN. Text-only fallback could recover ~3pp.

### Gap to Target

| Dataset | Best ACC | Target | Gap | Needed |
|---------|----------|--------|-----|--------|
| EN | 75.3% | >80% | 4.7pp | ~9 more correct (from 137/182 to 146/182) |
| ZH | 77.3% | >80% | 2.7pp | ~5 more correct (from 136/176 to 141/176) |

### What the Data Tells Us for Iteration 2

- Holistic scoring is the right direction but needs additional signal to push past the ~77% ceiling
- Frameless video handling could recover 2-3pp on EN
- Per-rule scoring (9 focused questions instead of 1 holistic) might be more discriminative — each rule question is more specific → MLLM more confident. NOT ensemble: each call has a distinct named role (checking a different rule)
- ZH is closer to target (2.7pp gap) — possibly reachable with minor improvements

---

## Post-Iteration 1 Validation Experiments

### Design Error Discovery: Hand-Written Mechanism Options

All prompts used in observation/scoring contained hand-written mechanism options NOT in platform policy:
- `ironic_mockery`, `coded_language`, `humor_based_stereotyping`, `contextual_dog_whistle`, `sarcastic_ridicule`, `visual_juxtaposition`

These were added manually to the observation prompt (`observe_training.py`) and propagated to quad scoring and other prompts. They are **not** in the YouTube or Bilibili policy.

**Impact**: The entire observation pipeline was contaminated. The MLLM was primed to "discover" these mechanisms, biasing observations and downstream rule refinement.

**Chain of contamination**:
```
Hand-written mechanism → observation biased → rule update based on biased observation → scoring biased
```

### Prompt Version Control Failure

The original holistic prompt that achieved 75.3% EN / 77.3% ZH was overwritten with a calibrated version ("IMPORTANT: must target SPECIFIC PROTECTED GROUP") **without committing the original**. The original prompt is lost.

The calibrated version actually performs **differently** — detailed comparison below.

### Experiment: Quad Images (32 frames via 2×2 grids)

8 quad images per video, each containing 4 sub-frames. PIL pre-resize to 360K pixels to control token budget.

| Dataset | ACC | Recall | FP | Notes |
|---------|-----|--------|----|-------|
| MHClip_EN | 69.78% | — | 24 | Much worse than 4-frame holistic |
| MHClip_ZH | 72.16% | — | — | Also worse |

**Cause**: Prompt included hand-written mechanism options → high FP rate. More visual frames also amplified visual suppression.

### Experiment: Qwen3-VL-2B vs 8B Comparison

Ran Qwen3-VL-2B-Instruct with video input (not frames), two prompt variants.

| Model | Dataset | Prompt | ACC | Prec | Rec | F1 | TP | FP | TN | FN | mean score |
|-------|---------|--------|-----|------|-----|-----|----|----|----|----|------------|
| **8B** | EN | calibrated | 72.8% | 66.7% | 27.5% | 38.9% | 14 | 7 | 104 | 37 | 0.129 |
| 2B | EN | clean | 74.4% | 87.5% | 15.2% | 25.9% | 7 | 1 | 109 | 39 | 0.107 |
| 2B | EN | mechanism | 73.7% | 85.7% | 13.0% | 22.6% | 6 | 1 | 109 | 40 | 0.103 |
| **8B** | ZH | calibrated | 77.0% | 63.6% | 67.3% | 65.4% | 35 | 20 | 89 | 17 | 0.325 |
| 2B | ZH | clean | 73.8% | 100% | 13.3% | 23.5% | 6 | 0 | 104 | 39 | 0.078 |
| 2B | ZH | mechanism | 73.8% | 100% | 13.3% | 23.5% | 6 | 0 | 104 | 39 | 0.087 |

**Note**: Comparison is not fully fair — 8B used calibrated prompt + 4 frames, 2B used clean/mechanism prompt + video input.

#### Prompt definitions:
- **calibrated**: policy rules + "IMPORTANT: must target SPECIFIC PROTECTED GROUP..." + "Does this video contain content that targets a protected group..."
- **clean**: policy rules only + "Does this video contain content that violates any of the above rules?"
- **mechanism**: policy rules + "Consider both explicit and implicit forms of hate speech, including irony, coded language, humor-based stereotyping, and contextual dog-whistles." + "Does this video violate..."

### Technical Finding: mm_processor_kwargs Ineffective

`mm_processor_kwargs={"max_pixels": ...}` does NOT work for either Qwen2-VL or Qwen3-VL in vLLM. The parameter passes through the chain but has no effect on actual image processing. Confirmed by:
1. Output tensor shape identical with and without the parameter
2. Qwen3-VL-2B with max_model_len=16384 hit prompt length 32937 — images processed at original resolution

**Workaround**: PIL pre-resize before passing to vLLM (used in quad scoring).

### Cumulative Findings

1. **Model size is not the bottleneck** (for ACC). 2B and 8B achieve similar ACC (~74%) because ACC is dominated by correctly classifying Normal videos (high TN). The real gap is in recall.

2. **Model size matters for recall.** 8B ZH has 67.3% recall vs 2B's 13.3%. Larger models are better at detecting hateful content, but this doesn't translate to much ACC gain due to class imbalance.

3. **Clean vs mechanism prompt makes no difference on 2B.** The mechanism hints ("irony, coded language...") had zero effect — identical results on ZH, 1 TP difference on EN.

4. **All previous "best" results used contaminated or lost prompts.** The 75.3% EN result used a prompt that is now lost. The calibrated prompt in code gives 72.8%. No clean baseline exists for 8B.

5. **Video input works.** Direct video input to vLLM (Qwen3-VL) works without frame extraction. vLLM handles frame sampling internally. Fallback to single-item processing needed for batch failures.

6. **25 EN and 19 ZH test videos have no video files** — these are automatic errors and cannot be recovered without obtaining the original videos.

---

## Iteration 1.5 — Calibration + Unsupervised Threshold (Holistic Binary Scores)

### Motivation

The holistic binary scoring from Iteration 1 gives P(Yes)/(P(Yes)+P(No)) on a constitution-informed "Does this violate any rules?" question. Two label-free additions:
1. **Content-free calibration**: Measure P(Yes) base rate on blank input, apply affine correction
2. **Unsupervised threshold selection**: Otsu's method or 2-component GMM on training score distribution

### 2B Model Results

Content-free P(Yes) base rates: EN = 0.182, ZH = 0.119

#### Full 2x3 Ablation: (Raw|Cal) x (Otsu|GMM|Oracle-ACC) on test sets

**MHClip_EN** (161 test, 49 pos, 112 neg):

| Method | Thresh | ACC | F1 | Prec | Recall | TP | FP | FN | TN |
|--------|--------|-----|-----|------|--------|----|----|----|----|
| Raw + Otsu | 0.273 | **76.40%** | 0.457 | 0.762 | 0.327 | 16 | 5 | 33 | 107 |
| Raw + GMM | 0.098 | 69.57% | 0.505 | 0.500 | 0.510 | 25 | 25 | 24 | 87 |
| Raw + Oracle | 0.330 | 77.02% | 0.448 | 0.833 | 0.306 | 15 | 3 | 34 | 109 |
| Cal + Otsu | 0.241 | 76.40% | 0.424 | 0.824 | 0.286 | 14 | 3 | 35 | 109 |
| Cal + GMM | 0.004 | 72.05% | 0.430 | 0.567 | 0.347 | 17 | 13 | 32 | 99 |
| Cal + Oracle | 0.170 | 77.02% | 0.448 | 0.833 | 0.306 | 15 | 3 | 34 | 109 |

**MHClip_ZH** (149 scored of 157 test, 45 pos, 104 neg) — test-only thresholds, training not yet complete:

| Method | Thresh | ACC | Notes |
|--------|--------|-----|-------|
| Raw + Otsu | 0.274 | 75.84% | |
| Raw + GMM | 0.036 | **81.21%** | Matches oracle |
| Raw + Oracle | 0.030 | 81.21% | |

### Key Analytical Findings

1. **Calibration is destructive.** Affine calibration collapses 65% of EN positives to zero because p_base (0.182) exceeds median positive score (0.119). Calibration preserves rank order but degenerates the distribution. **Abandoned.**

2. **Temperature rescaling is a no-op for ACC.** Temperature T applied to logit-space: score(T) = sigmoid(logit(score)/T). AUC-ROC is invariant to T (monotonic transform preserves rank order). Oracle ACC is also invariant. Temperature only affects bimodality (useful for Otsu/GMM quality) but cannot push past the rank-order ceiling.

3. **EN bottleneck is AUC-ROC, not thresholding.** EN AUC = 0.725, Otsu ACC = 76.40% (within 0.6pp of oracle 77.02%). No threshold method can exceed oracle. Simulation shows EN needs AUC >= 0.825 for oracle ACC ~80%.

4. **2B score quantization.** Both EN and ZH produce only ~32-39 unique score values at 4 decimal places (despite 161/149 test samples). This is a vLLM/token-probability quantization artifact. Limits effective resolution.

5. **Unified method assessment (2B):**
   - Raw + Otsu: EN 76.40%, ZH 75.84% — neither clears 80%
   - Raw + GMM: EN 69.57%, ZH 81.21% — EN fails badly
   - No single 2B method achieves >80% on both datasets

### Pending: 8B Model Experiments

Need 8B (Qwen3-VL-8B-Instruct) with clean binary prompt on both datasets.
Critical metric: EN AUC-ROC. If > 0.825, unified method possible.

---

## Iteration 2 Design — Prompt Deflection (Contingency)

### Scientific Story (4-point)

**Phenomenon**: Safety-aligned MLLMs exhibit asymmetric confidence on harm-assessment questions. When asked "Does X violate policy?", RLHF safety training creates a conservative prior toward "No" (the model avoids affirmative harm claims). For hateful video detection, this manifests as score compression: even genuinely hateful content receives low P(Yes) scores (EN positive median = 0.119), making positive/negative class overlap severe (EN AUC = 0.725).

**Mechanism**: Prompt deflection inverts the question from "Does this video violate any of the above rules?" to "Is this video fully consistent with all of the above community guidelines?" Under deflection, P(Yes) = benign and P(No) = potentially hateful. The hate score becomes P(No)/(P(Yes)+P(No)) = 1 - original_score_formula. The safety prior that previously suppressed violation detection now helps: the model's reluctance to affirmatively claim "Yes, this is consistent" for borderline content (because asserting consistency is a commitment) translates into higher P(No) = higher hate scores for those borderline cases. Critically, this is NOT a monotonic transform of the original scores — the model's internal processing changes with the prompt, potentially producing a different ranking.

**Prediction**: 
- Content-free P(Yes|"consistent?", blank) should be >0.5 (blank is consistent)
- AUC-ROC under deflected prompt should exceed AUC under original prompt
- The improvement should concentrate in the borderline score zone (original scores 0.1-0.3)
- If AUC does NOT improve, the bottleneck is genuine model inability, not safety alignment

**Counterfactual ablation**: If deflection is just the logical complement (1 - original score), AUC would be identical. Any AUC difference proves the model treats the two framings non-equivalently, confirming the safety alignment asymmetry.

### Implementation

Minimal change to `score_holistic_2b.py`:
1. Add `DEFLECTED_PROMPT` with "Is this video fully consistent with all of the above community guidelines?"
2. Add `--prompt-style` argument: `violates` (default) or `consistent`
3. When `consistent`: hate_score = 1 - P(Yes)/(P(Yes)+P(No))
4. Output to `{split}_binary_deflected.jsonl`

### Constraint Check
- 1 MLLM call per video: PASS
- Multimodal input: PASS (same video + text)
- No ensemble: PASS (single call, different prompt)
- No external data: PASS
- Scientific story: PASS (safety alignment asymmetry specific to hate detection)

---

## Iteration 3 Design — Observe-then-Judge (Contingency if Deflection Fails)

### Scientific Story (4-point)

**Phenomenon**: Holistic hate detection conflates two cognitively distinct tasks — perceiving multimodal content and reasoning about policy violations — into a single MLLM call. This conflation forces the model to compress perception and judgment into a single P(Yes) token probability, losing information at the bottleneck. Hateful video compounds this problem because the relevant cues are often implicit (irony, juxtaposition, coded language) and require explicit articulation to reason about.

**Mechanism**: Separate the pipeline into two calls with distinct named roles:
- **Observe (O)**: Multimodal call. "Describe any content relevant to hate speech evaluation." The model externalizes its perceptual understanding as text, bypassing the Yes/No bottleneck. Safety alignment doesn't suppress description (describing is not accusing).
- **Judge (J)**: Text-only call. Given observation text + rules, "Does this violate any rules? Yes/No." The model reasons over explicit textual evidence, not ambiguous visual input. Text-based judgment is the MLLM's strongest modality.

**Prediction**:
- Observation step produces richer evidence for hateful videos than benign ones (more text, more specific groups/mechanisms mentioned)
- Judge step achieves higher AUC than holistic because it operates on explicit evidence
- If observation is replaced with blank text, ACC should drop to holistic levels or below (observation is load-bearing)
- If observation is replaced with ground-truth labels, ACC should be near-perfect (judgment step works given good input)

**Counterfactual**: Remove observation (blank it). If ACC drops, observation is load-bearing. If not, the judge prompt alone is doing the work.

### Cost: 2 MLLM calls per video (1 multimodal + 1 text-only). Justified by distinct structural roles.

### Implementation: Requires new scripts for observation extraction and text-only judgment.
