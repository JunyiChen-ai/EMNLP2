# Experiment Plan: Constitution-Following Hateful Video Detection

**Target**: ACC ≥ 85 on HateMM, MHClip_EN, MHClip_ZH (label-free)

---

## Phase 1 Results (Static Constitution) — COMPLETED

### Results

| Dataset | N | ACC | F1 | F1-macro | Precision | Recall | Pred H/NH | GT H/NH |
|---------|---|-----|-----|----------|-----------|--------|-----------|---------|
| HateMM | 215 | **85.12** | 82.61 | 84.80 | 77.55 | 88.37 | 98/117 | 86/129 |
| MHClip_EN | 182 | 69.78 | 17.91 | 49.70 | 66.67 | 10.34 | 9/173 | 58/124 |
| MHClip_ZH | 176 | 72.73 | 27.27 | 55.24 | 100.00 | 15.79 | 9/167 | 57/119 |

HateMM meets the 85% target. MHClip_EN and MHClip_ZH fail badly — near-zero recall indicates the static constitution misses most hateful videos.

### Key Findings

1. **Explicit hate is captured, implicit hate is not.** HateMM (BitChute, explicit hate speech) works well. MHClip (YouTube/Bilibili, implicit/contextual hate) has ~10-16% recall. The YouTube/Bilibili policy rules are designed for explicit violations; implicit hate (irony, coded language, juxtaposition) falls through the cracks.

2. **Only 2-3 rules fire at all on MHClip.** On MHClip_EN, only YT-R2 (incite hatred, 6 videos) and YT-R3 (dehumanization, 2 videos) trigger. 6 of 9 rules never fire. On MHClip_ZH, only BL-R2 (attacks, 7 videos) and BL-R3 (verbal abuse, 4 videos) trigger. The constitution is severely under-specified for these datasets.

3. **Prior ≈ 0 collapses debiasing.** The prior prompt ("no content available") causes the MLLM to always say "No", producing score_prior ≈ 0 for all preconditions. This makes alpha2 = 0.8*(1-0) = 0.8, requiring score_full > 0.8 for confident "satisfied" — an unreachably high bar for implicit content.

4. **Visual suppression.** When frames are visually benign (common in implicit hate), they actively push score_full down even when text signal is strong (e.g., score_text=0.95 but score_full=0.01). The MLLM treats "benign frames + hateful text" as contradictory and sides with frames.

5. **CLIP embedding space is not a useful proxy for hate.** Tested two hypotheses for cheap sample selection:
   - Low-density regions have more hateful videos → **rejected** (Q4 sparsest has 0 Hateful in MHClip_EN)
   - Outliers are hateful → **rejected** (top-5% sparse are NSFW-adjacent Normal videos)

### Root Cause Diagnosis

The core problem is **constitution gap**: the static rules derived from platform policies only cover *explicit* hate patterns. Real-world hateful videos on YouTube/Bilibili use implicit mechanisms (contextual irony, coded slurs, dog-whistles, juxtaposition of benign visuals with hateful narration) that no explicit rule can anticipate a priori. Fixing the prior or debiasing thresholds alone will not solve this — the rules themselves need to be expanded.

---

## Phase 2: Dynamic Constitution Update — COMPLETED (Iteration 1)

### Goal

Improve ACC on MHClip_EN (69.78%) and MHClip_ZH (72.73%) toward >80% target.

### What Was Tried

#### 2A. Target-Mechanism Bottleneck Discovery (TMBD)

Collected structured MLLM observations on training data (1 multimodal call per video):
- MHClip_EN: 555 videos → 134 flagged HATEFUL_OR_OFFENSIVE
- MHClip_ZH: 579 videos → 227 flagged HATEFUL_OR_OFFENSIVE

**Key finding: Coverage-gap hypothesis falsified.**
- EN: 133/134 hateful observations had `covered_rule_ids` (existing rules assigned)
- ZH: 227/227 hateful observations had `covered_rule_ids`
- Discovery set (no covered rules) was empty → no new rules needed

**Root cause**: The MLLM recognizes that existing rules cover hateful content when asked holistically. The problem is precondition decomposition, not rule coverage.

#### 2B. Observation-Grounded Precondition Re-Decomposition

Used observation data to rewrite preconditions with implicit-mechanism-aware language:
- YouTube: 9 rules → 40 branches (explicit + implicit variants)
- Bilibili: 8 rules → 35 branches

Results with refined preconditions + threshold sweep: **best 72.5% ACC** — higher recall (38% vs 10%) but more FP, net ACC unchanged.

#### 2C. Holistic Constitution-Informed Scoring

Bypassed precondition decomposition entirely. Single MLLM call per video:
- Input: 4 uniformly-sampled frames + title + transcript + platform rule list
- Output: P(Yes)/(P(Yes)+P(No)) via constrained token generation

**Best results:**

| Dataset | ACC | F1 | Precision | Recall | Method |
|---------|-----|-----|-----------|--------|--------|
| MHClip_EN | **75.3** | 52.6 | 67.6 | 43.1 | Holistic token prob, t=0.6 |
| MHClip_ZH | **77.3** | 58.3 | 71.8 | 49.1 | Holistic token prob, t=0.9 |

#### 2D. Other Approaches Tested (no improvement)

- Evidence-grounded calibration (require rule+target+mechanism for positive) → 75.3% EN
- Calibrated prompt (explicit "must target protected group") → 72.0% EN (too strict)
- Combined holistic + precondition AND/OR → no gain over holistic alone
- Various threshold sweeps on score_full, score_text, max(full,text) → max 75.3%

### Findings

1. **Precondition decomposition is the bottleneck**, not rule coverage. Holistic scoring beats decomposed scoring by 5pp.
2. **MLLM single-call ceiling is ~77% ACC** on MHClip. Train/test accuracy matches (~77%), confirming model capability limit.
3. **Score separation exists** but isn't sharp enough: hateful mean 0.47/0.64, normal mean 0.12/0.17 (EN/ZH).
4. **20 frameless EN test videos** are automatic errors (7 hateful → FN, ~2pp ACC loss).

### Code Files (Phase 2)

| File | Purpose |
|------|---------|
| `src/observe_training.py` | TMBD Step A: structured MLLM observations on any split |
| `src/induce_rules.py` | TMBD Steps B-C: gap filter + rule induction (unused after pivot) |
| `src/refine_preconditions.py` | Observation-grounded precondition re-decomposition |
| `src/score_holistic.py` | Holistic constitution-informed token probability scoring |

### Status

**Gap to target: EN 4.7pp, ZH 2.7pp.** Single-call Qwen3-VL-8B ceiling reached. Next iteration needs a fundamentally different signal source or relaxed constraints.

---

## Method Overview

```
Step 0: Define constitution    — rules from platform hate speech policies (offline)
Step 1: Rules objectification  — LLM rewrites subjective rules → objective (offline)
Step 2: Precondition extraction — LLM decomposes each rule → precondition chain (offline)
Step 3: Relevance scanning     — CLIP filters irrelevant rules per video (fast, no MLLM)
Step 4: Token prob judgment    — MLLM scores each precondition via debiased P(Yes)/(P(Yes)+P(No))
Step 5: Cascaded reasoning     — if token prob uncertain → fallback to CoT reasoning
Step 6: Aggregate              — all preconditions satisfied → rule violated → hateful

Output: Hateful/Not-Hateful + list of violated rules
```

---

## Step 0: Constitution (the starting rules)

Platform hate speech policies are saved in `constitution/` as JSON files:
- `constitution/youtube.json` — 9 rules, used by HateMM and MHClip_EN
- `constitution/bilibili.json` — 8 rules, used by MHClip_ZH

HateMM and MHClip_EN use YouTube's policy. MHClip_ZH uses Bilibili's policy.

YouTube hate speech policy (https://support.google.com/youtube/answer/2801939) defines 9 prohibition types:

| Rule ID | Name | Original rule (from YouTube policy) |
|---------|------|-------------------------------------|
| R1 | Encourage violence | Encourage violence against individuals or groups based on their protected group status. Implied calls for violence are treated as real threats. |
| R2 | Incite hatred | Incite hatred against individuals or groups based on their protected group status. |
| R3 | Dehumanization | Dehumanization of individuals or groups by calling them subhuman, comparing them to animals, insects, pests, disease, or any other non-human entity based on their protected group status. |
| R4 | Praise/glorify violence | Praise or glorification of violence against individuals or groups based on their protected group status. |
| R5 | Slurs and stereotypes | Use of racial, religious, or other slurs and stereotypes that incite or promote hatred based on protected group status. This can take the form of speech, text, or imagery promoting these stereotypes or treating them as factual. |
| R6 | Claims of inferiority | Claim that individuals or groups are physically or mentally inferior, deficient, or diseased based on their protected group status. This includes calls for the subjugation or domination over individuals or groups. |
| R7 | Hateful supremacism | Promotion of hateful supremacism by alleging the superiority of a group over those with protected group status to justify violence, discrimination, segregation, or exclusion. |
| R8 | Conspiratorial claims | Conspiratorial claims that individuals or groups are evil, corrupt, or malicious based on their protected group status. |
| R9 | Denial of violent events | Denial or minimization of a well-documented, major violent event or the victimhood of such an event. |

Protected group status: Age; Caste, Ethnicity, or Race; Disability; Immigration Status; Nationality; Religion; Sex, Gender, or Sexual Orientation; Veteran Status; Victims of a major violent event and their kin.

---

## Step 1: Rules Objectification

**Goal**: Rewrite each rule until objectiveness score ≥ 9/10, so the MLLM can evaluate them without subjective judgment.

**Code file**: `src/objectify_rules.py`

**Prompt for measuring objectiveness:**
```
Please act as an impartial judge and evaluate the objectiveness for the provided guideline. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, please rate the response on a scale of 1 to 10 (10 is the most objective).

Guideline: {rule_text}
```

**Prompt for rewriting** (LLM-as-Optimizer):
```
The following content moderation rule scored {score}/10 on objectiveness. Rewrite it to be more objective and actionable, while preserving its original intent. The rule should be verifiable from observing a video's visual content, audio, and text.

Original rule: {rule_text}
Score: {score}/10
Feedback: {explanation}

Rewritten rule:
```

**Loop**: Rewrite → score → if < 9, rewrite again. Max 10 iterations per rule.

**LLM**: Qwen2-VL-7B or Qwen3-VL-8B as text-only. No images needed.

Run separately for each platform constitution (`--constitution youtube` and `--constitution bilibili`).

**Output**: `constitution/objectified_youtube.json`, `constitution/objectified_bilibili.json`
```json
[
  {
    "rule_id": "R1",
    "original": "...",
    "objectified": "...(LLM will generate)...",
    "objectiveness_score": 9
  }
]
```

---

## Step 2: Precondition Extraction

**Goal**: Decompose each objectified rule into a chain of independently verifiable preconditions.

**Code file**: `src/extract_preconditions.py`

**Prompt:**
```
Given a content moderation policy for videos, extract the pre-conditions for violating the policy. A video violates the policy only when ALL pre-conditions are satisfied. Each pre-condition should be a single verifiable statement that can be checked by observing the video's visual frames, title text, and spoken transcript.

##Example 1:
##Policy: Content that dehumanizes individuals or groups by calling them subhuman, comparing them to animals, insects, pests, disease, or any other non-human entity based on protected group status.
##Extracted pre-conditions:
1. A specific identity group (defined by race, ethnicity, religion, gender, sexual orientation, disability, or nationality) is referenced or depicted in the video.
2. The video contains language or imagery that compares members of that group to animals, insects, pests, disease, subhuman entities, or other non-human entities.

##Example 2:
##Policy: Content that uses racial, religious, or other slurs and stereotypes that incite or promote hatred, in speech, text, or imagery.
##Extracted pre-conditions:
1. The video contains racial, religious, or other identity-based slurs or stereotypes in its speech, on-screen text, or imagery.
2. The slurs or stereotypes are used in a way that incites or promotes hatred rather than quoting, condemning, or educating.

Now given a new policy, perform this task.
Policy: {objectified_rule}
```

**LLM**: Same MLLM as text-only. Offline, one-time.

**Output**: `constitution/preconditions_youtube.json`, `constitution/preconditions_bilibili.json`

Example (`preconditions_youtube.json`):
```json
[
  {
    "rule_id": "R1",
    "preconditions": [
      "A specific identity group is referenced or depicted in the video.",
      "The video contains language that promotes violence against that group.",
      "The promotion encourages real-world action rather than fictional or educational discussion."
    ]
  }
]
```

Inter-precondition logic is always AND — a rule is violated only when all its preconditions are satisfied.

---

## Step 3: Relevance Scanning + Frame Selection

Two-stage filtering using CLIP: first decide whether a rule is relevant to the video at all, then select which frames to send to the MLLM.

**Code file**: `src/clip_filter.py` (separate process from MLLM scoring — CLIP and vLLM never share a GPU)

### 3a. Rule-level relevance filtering

Decide whether a rule is relevant to this video. Check both visual and textual similarity — a rule can be triggered by either modality.

```python
rule_emb = clip.encode_text(objectified_rule)        # [768]
frame_embs = clip.encode_image(frames)                # [32, 768]
video_emb = frame_embs.mean(dim=0)                    # [768]
text_emb = clip.encode_text(title + " " + transcript) # [768]

relevance = max(cos(video_emb, rule_emb), cos(text_emb, rule_emb))

if relevance < threshold:  # default t=0.22
    skip this rule
```

If either modality shows relevance, the rule passes. This ensures text-dominant rules (e.g., R3 slurs in transcript) are not filtered out when frames are visually benign.

### 3b. Frame selection for relevant rules

For rules that pass 3a, select the top-K most relevant frames to send to the MLLM. This reduces visual token count and improves signal-to-noise ratio (hate signals are temporally sparse).

```python
# Per-frame relevance to this rule
frame_scores = [cos(frame_embs[i], rule_emb) for i in range(num_frames)]

# Select top-K frames
top_k_indices = sorted(range(num_frames), key=lambda i: -frame_scores[i])[:K]
selected_frames = [frames[i] for i in sorted(top_k_indices)]  # keep temporal order

# Edge case: if ALL frame scores are very low (max < 0.1),
# the signal is likely in text/transcript only → send no frames, text only
if max(frame_scores) < frame_threshold:
    selected_frames = []  # text-only mode for this rule
```

Default K=8. All preconditions under the same rule share the same selected frames.

**Why rule-level, not precondition-level**: frame selection is done per rule, not per precondition. This ensures all preconditions of a rule see the same visual context, avoiding the problem of a policy losing half its conditions.

---

## Step 4: Debiased Token Probability Judgment

**Goal**: For each relevant precondition, score whether it is satisfied via debiased MLLM token probability.

**Code file**: `src/score_preconditions.py` (reads CLIP filter output, loads vLLM only — no CLIP on GPU)

### Prompt for scoring each precondition

```
{preamble}

Determine if the following content is present in this video based on what you can observe. No external context or intent of the video should be considered when determining presence.

Content: {precondition_text}

Answer "Yes" or "No".
```

Where `{preamble}` depends on input mode:
- **Full mode**: frames + `"Title: {title}\nTranscript: {transcript}"`
- **Text-only mode** (for debiasing): no frames, just `"Title: {title}\nTranscript: {transcript}"`

### Token probability extraction

```python
score = P("Yes") / (P("Yes") + P("No"))  # from position-0 logprobs
```

Single token output. `max_tokens=1`, `temperature=0`, `logprobs=50`.

### Debiasing

Compare score with input vs score without input to remove the MLLM's content-independent prior bias.

```python
score_full = M(frames + text, precondition)
score_text = M(text_only, precondition)     # no frames
score_prior = M(None, precondition)         # no frames, no text

alpha1 = -0.3 * score_prior
alpha2 = 0.8 * (1 - score_prior)
beta   = 0.6

if score_full - score_prior < alpha1:
    # Input actively contradicts precondition → NOT satisfied
    satisfied = False
elif score_full - score_prior > alpha2:
    # Strong evidence → satisfied
    satisfied = True
elif score_full - score_text > beta:
    # Visual adds significant evidence beyond text → satisfied
    satisfied = True
else:
    # Uncertain → fall through to cascaded reasoning (Step 5)
    satisfied = None
```

`score_prior` is constant per precondition — compute once, reuse for all videos.

---

## Step 5: Cascaded Reasoning (Fallback)

**Goal**: When token probability is uncertain, fall back to CoT reasoning for a more reliable judgment.

**Code file**: `src/score_preconditions.py` (integrated as fallback)

**Step 1 — CoT reasoning:**
```
Determine if the following content is present in this video based on what you can observe. No external context or intent of the video should be considered when determining a visible state. Think step by step.

Content: {precondition_text}
```

**Step 2 — Summarize to verdict:**
```
Based on the answer, summarize the results. Steps:
1. Select a "rating". This should be "Yes" or "No".
2. Provide a "rationale". Explain the reason for your decision.
Provide your assessment using the following JSON template:
{"rating": "Yes" | "No", "rationale": str}
```

`max_tokens=256` for reasoning. The two steps are a single multi-turn conversation, not two separate MLLM calls. Parse JSON from the final response.

---

## Step 6: Aggregate → Final Verdict

**Code file**: `src/classify.py`

```python
def classify(video_scores, rules):
    violated_rules = []
    for rule in rules:
        all_satisfied = all(
            video_scores[c]["satisfied"]
            for c in rule["preconditions"]
        )
        if all_satisfied:
            violated_rules.append(rule["rule_id"])

    hateful = len(violated_rules) > 0
    return hateful, violated_rules
```

**Training-free**. No classifier, no pseudo-labels, no student model.

---

## File Summary

### `src/objectify_rules.py` — Step 1

| | |
|---|---|
| **Input** | `constitution/youtube.json` or `constitution/bilibili.json` (raw platform rules) |
| **Output** | `constitution/objectified_youtube.json` or `constitution/objectified_bilibili.json` |
| **Args** | `--constitution youtube\|bilibili` `--model Qwen/Qwen3-VL-8B-Instruct` |
| **Log** | `logs/objectify_{platform}.log` |
| **Execution** | Slurm, 1 GPU (vLLM text-only inference) |

### `src/extract_preconditions.py` — Step 2

| | |
|---|---|
| **Input** | `constitution/objectified_youtube.json` or `constitution/objectified_bilibili.json` |
| **Output** | `constitution/preconditions_youtube.json` or `constitution/preconditions_bilibili.json` |
| **Args** | `--constitution youtube\|bilibili` `--model Qwen/Qwen3-VL-8B-Instruct` |
| **Log** | `logs/extract_preconditions_{platform}.log` |
| **Execution** | Slurm, 1 GPU (vLLM text-only inference) |

### `src/clip_filter.py` — Step 3

| | |
|---|---|
| **Input (constitution)** | `constitution/preconditions_{platform}.json` (objectified rule texts for CLIP encoding) |
| **Input (video data)** | `datasets/{dataset}/frames/{video_id}/frame_*.jpg` (pre-extracted frames) |
| **Input (metadata)** | `datasets/{dataset}/annotation(new).json` (title, transcript per video) |
| **Input (splits)** | `datasets/{dataset}/splits/{split}.csv` (video IDs) |
| **Output** | `results/clip_filter/{dataset}/{split}.json` — per-video, per-rule: relevance score, relevant flag, selected frame indices (top-K), text-only flag |
| **Args** | `--dataset HateMM\|MHClip_EN\|MHClip_ZH` `--split test` `--K 8` `--relevance-threshold 0.22` `--frame-threshold 0.1` |
| **Log** | `logs/clip_filter_{dataset}_{split}.log` |
| **Execution** | Slurm, 1 GPU (CLIP only) |

### `src/score_preconditions.py` — Steps 4-5

| | |
|---|---|
| **Input (constitution)** | `constitution/preconditions_{platform}.json` |
| **Input (clip filter)** | `results/clip_filter/{dataset}/{split}.json` (from Step 3 — relevant rules + selected frame indices per video) |
| **Input (video data)** | `datasets/{dataset}/frames/{video_id}/frame_*.jpg` (only reads frames selected by CLIP filter) |
| **Input (metadata)** | `datasets/{dataset}/annotation(new).json` (title, transcript per video) |
| **Input (splits)** | `datasets/{dataset}/splits/{split}.csv` |
| **Output (scores)** | `results/scores/{dataset}/{split}.jsonl` — per-video, per-rule, per-precondition scores (full/text_only/prior) + satisfied flag |
| **Output (prior cache)** | `results/scores/prior_cache_{platform}.json` — score_prior per precondition (computed once, reused) |
| **Args** | `--dataset HateMM\|MHClip_EN\|MHClip_ZH` `--split test` `--model Qwen/Qwen3-VL-8B-Instruct` |
| **Log** | `logs/score_{dataset}_{split}.log` |
| **Execution** | Slurm, 1 GPU (vLLM only, no CLIP) |

### `src/classify.py` — Step 6

| | |
|---|---|
| **Input (scores)** | `results/scores/{dataset}/test.jsonl` |
| **Input (GT labels)** | `datasets/{dataset}/annotation(new).json` + `datasets/{dataset}/splits/test.csv` |
| **Input (constitution)** | `constitution/preconditions_youtube.json` or `constitution/preconditions_bilibili.json` (for rule structure / AND logic) |
| **Output** | `results/eval/{dataset}/results.json` — ACC, F1, precision, recall, per-rule violation counts, per-rule ablation |
| **Args** | `--dataset HateMM\|MHClip_EN\|MHClip_ZH` |
| **Log** | `logs/classify_{dataset}.log` |
| **Execution** | Slurm, CPU only (no GPU needed) |

### `src/infer_vllm.py` — Shared skeleton (exists)

Provides `build_messages()`, `extract_token_probs()`, `build_label_token_ids()`, MLLM init, OOM handling.

---

## Execution Order

All jobs submitted via Slurm, one at a time (wait for completion before submitting next). Conda env: `SafetyContradiction` (has vllm 0.11.0).

```bash
# Step 1: Objectify rules (1 GPU each, ~10 min)
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/objectify_rules.py --constitution youtube"
# wait for completion
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/objectify_rules.py --constitution bilibili"
# wait for completion

# Step 2: Extract preconditions (1 GPU each, ~10 min)
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/extract_preconditions.py --constitution youtube"
# wait for completion
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/extract_preconditions.py --constitution bilibili"
# wait for completion

# Step 3: CLIP relevance scanning + frame selection (1 GPU each, ~20 min)
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/clip_filter.py --dataset HateMM --split test"
# wait for completion
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/clip_filter.py --dataset MHClip_EN --split test"
# wait for completion
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/clip_filter.py --dataset MHClip_ZH --split test"
# wait for completion

# Steps 4-5: MLLM precondition scoring (1 GPU each, ~2h)
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/score_preconditions.py --dataset HateMM --split test"
# wait for completion
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/score_preconditions.py --dataset MHClip_EN --split test"
# wait for completion
sbatch --gres=gpu:1 --wrap "conda activate SafetyContradiction && python src/score_preconditions.py --dataset MHClip_ZH --split test"
# wait for completion

# Step 6: Classify + evaluate (CPU only, seconds)
sbatch --wrap "conda activate SafetyContradiction && python src/classify.py --dataset HateMM"
# wait for completion
sbatch --wrap "conda activate SafetyContradiction && python src/classify.py --dataset MHClip_EN"
# wait for completion
sbatch --wrap "conda activate SafetyContradiction && python src/classify.py --dataset MHClip_ZH"
```

---

## Ablations (Phase 1)

1. **No objectification** — use original YouTube rules directly, skip Step 1
2. **No relevance scanning** — evaluate all rules for every video, skip Step 3
3. **No debiasing** — use raw token prob with fixed 0.5 threshold, skip debiasing logic
4. **No cascaded reasoning** — binarize uncertain cases at 0.5 instead of CoT fallback
5. **Per-rule contribution** — remove one rule at a time, measure ACC drop (completed, see Phase 1 Results)
