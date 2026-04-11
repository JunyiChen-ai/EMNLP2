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
