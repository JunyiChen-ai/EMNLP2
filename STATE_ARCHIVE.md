# State Archive — Label-Free Hateful Video Detection
**Archived**: 2026-04-12
**Branch**: label-free
**Target**: ACC >80% on BOTH MHClip_EN and MHClip_ZH (unified method)

---

## Current Status: EN BLOCKED at 75% (unified method) — NOT ACCEPTED

**Best unified method so far**: 8B + Triclass broad + transcript 1000 chars
- EN oracle ACC: **75.16%** (clean 161 test) — FAIL
- ZH oracle ACC: **81.94%** (clean 157 test) — PASS

**Target NOT met**. EN still blocked after 4 iterations (0: baselines, 1: calibration, 2: deflection, 3: observe-then-judge, 4: triclass + context length).

Best single-dataset results (NOT unified):
| Dataset | Model | Method | ACC | Notes |
|---------|-------|--------|-----|-------|
| EN | 2B | Binary Raw+Otsu | 77.02% | Oracle threshold — not valid |
| EN | 8B | Triclass broad t1000 | 75.16% | Oracle threshold — not valid |
| ZH | 2B | Binary Raw+GMM | 81.21% | Test-fit threshold — distribution leakage |
| ZH | 2B | Binary Raw+GMM train | 79.19% | Proper label-free — borderline |
| ZH | 8B | Triclass broad t1000 | 81.94% | Oracle threshold — not valid |

**Important**: All ACC numbers above use oracle or test-fit thresholds. No config has been evaluated with proper train-derived unsupervised threshold on the best (Triclass broad t1000) setup. Actual label-free ACC is unknown.

---

## Findings

### Iteration 0: Baselines (2B, clean binary + triclass prompts)

| Config | N scored | Best ACC | Thresh | Best F1 |
|--------|----------|----------|--------|---------|
| EN binary | 161 | 77.02% | 0.33 | 44.78% |
| EN triclass | 161 | 75.78% | 0.95 | 59.46% |
| ZH binary | 149 | 81.21% | 0.03 | 71.43% |
| ZH triclass | 149 | 79.87% | 0.77 | 67.37% |

**Key findings**:
1. Binary > triclass on ACC for both datasets
2. ZH binary already exceeds 80% with oracle threshold
3. EN bottleneck: FN=34, FP=3 — model is too conservative (high precision, terrible recall)
4. "Offensive" class is hardest: EN 30.6% accuracy on Offensive, ZH 64.3%

### Iteration 1: Content-Free Calibration + Unsupervised Threshold

**Content-free P(Yes) base rates** (8B model):
- EN: p_base = 0.0004 (near zero — 8B extremely reluctant to say "Yes" to hate questions)
- ZH: p_base = 0.0015

**Content-free P(Yes) base rates** (2B model, from training score distributions):
- EN: p_base = 0.1824
- ZH: p_base = 0.1192

**Unified method results (2B, train-derived thresholds on test)**:

| Method | EN ACC | ZH ACC | Unified? |
|--------|--------|--------|----------|
| Raw + Otsu | 76.40% | 77.85% | Same code path |
| Raw + GMM | 69.57% | **79.19%** | Same code path |
| Cal + Otsu | 76.40% | 74.50% | Same code path |
| Cal + GMM | 72.05% | 78.52% | Same code path |

**Best unified label-free**: Raw + GMM from training → EN 69.6%, ZH 79.2% (neither hits 80% unified)
**Best per-dataset**: EN=Raw+Otsu 76.4%, ZH=Raw+GMM 81.2% (but NOT allowed — must be unified)

**Key finding**: No single unsupervised method achieves >80% on BOTH datasets simultaneously. The EN discrimination ceiling is the bottleneck.

### Iteration 1b: 8B Model Scoring

| Model | Dataset | AUC-ROC | Oracle ACC | Best Unsupervised |
|-------|---------|---------|------------|-------------------|
| 2B | EN | 0.7254 | 77.02% | 76.40% (Otsu) |
| 8B | EN | 0.7482 | 74.53% | 72.67% (Otsu) |
| 2B | ZH | 0.8479 | 81.21% | 81.21% (GMM) |
| 8B | ZH | 0.8750 | 81.88% | 81.88% (Otsu) |

**Key finding**: 8B is WORSE than 2B on EN despite marginally better AUC. The 8B model's RLHF alignment makes it overconfident in both directions — pushing ambiguous scores to extremes. 25/49 EN positives score <0.01 on 8B.

### Root Cause Analysis: Why EN Fails

1. **Safety alignment token suppression**: The "Does this video violate rules?" framing triggers RLHF safety training, systematically suppressing P(Yes). Stronger on 8B than 2B.
2. **"Sensitive ≠ Hateful" conflation**: 8B flags educational/news content about LGBTQ, disability as hateful (FP), while missing implicit hate/mockery (FN).
3. **Offensive class is the gap**: 25/34 EN false negatives are Offensive (not Hateful). The model recognizes Hateful content but misses borderline Offensive content.
4. **Score compression**: EN positive mean P(Yes) = 0.22 (2B) / 0.15 (8B) — far too low. ZH positive mean = 0.19 (2B) / 0.27 (8B) — higher because ZH hate is more explicit.

---

## Literature Findings (Scout)

### Three-Component Label-Free Pipeline
1. **Contextual Calibration** (Zhao et al. ICML 2021): Content-free input to measure P(Yes) bias → affine correction
2. **Distributional Proxy** (bimodality coefficient): Label-free quality metric for prompt selection
3. **Unsupervised Threshold** (Otsu 1979 / GMM): Data-driven decision boundary from score distribution

### Additional Relevant Papers
- **OTTER** (NeurIPS 2024): Optimal transport label distribution adaptation for zero-shot models
- **DACA** (NeurIPS 2025): Disagreement between base/instruct models for unsupervised calibration
- **OPRO** (ICLR 2024): LLM as optimizer for prompt search
- **Entropy-guided prompt weighting** (ICASSP 2026): Low-entropy prompts ranked higher
- **Safety alignment asymmetry literature**: RLHF overrefusal, prompt polarity effects

---

## Iteration 2: Prompt Deflection — FAILED

| Config | EN AUC | EN Oracle ACC | ZH AUC | ZH Oracle ACC |
|--------|--------|-------------|--------|-------------|
| 8B Original | 0.7482 | 74.53% | 0.8750 | 81.88% |
| 8B Deflected | 0.7589 | 72.67% | 0.8722 | 81.88% |

**Conclusion**: Deflection rescues collapsed positives but inflates FP. AUC unchanged. Single Yes/No token prob cannot separate EN at >80% regardless of prompt framing. Bottleneck is discriminative ability, not safety suppression.

**Result files:**
- `results/holistic_8b/MHClip_EN/test_binary_deflected.jsonl` (161 records)
- `results/holistic_8b/MHClip_ZH/test_binary_deflected.jsonl` (157 records)

---

## Iteration 3: Observe-then-Judge — FAILED

Two MLLM calls with distinct roles: (1) multimodal free-text observation, (2) text-only binary judgment.

| Dataset | OTJ AUC | Holistic AUC | Delta |
|---------|---------|-------------|-------|
| EN | 0.746 | 0.748 | -0.002 |
| ZH | 0.845 | 0.875 | -0.030 |

**Diagnosis**: Observer self-censors (sanitizes borderline content due to safety alignment), judge inherits same conservative bias. Observation text features (length, hate keyword count) have AUC ~0.55 — essentially random. The bottleneck is NOT perception/judgment conflation — it's the MLLM's fundamental reluctance to flag content.

**Result files**: `results/otj_8b/`

---

## Iteration 4: Triclass + Transcript Length — PROMISING (best unified candidate)

Three independently-validated improvements:

### Improvement 1: Binary → Triclass label space (+0.034 AUC on EN)
Replacing Yes/No with Hateful/Offensive/Normal. The "Offensive" middle category reduces activation energy for flagging borderline content — model is more willing to say "Offensive" (descriptive) than "Yes this violates rules" (accusatory).

### Improvement 2: Transcript 300 → 1000 chars (+0.026 AUC on EN only)
EN hateful transcripts average 370 chars, 57% exceed 300. The hate signal often appears mid-speech. ZH transcripts average 78 chars — no benefit from longer window. This is a genuine cross-lingual structural difference.

### Improvement 3: Narrow → Broad triclass definitions (+0.004 AUC on EN)
"Hateful" = "contains hate speech... including through humor, mockery, or coded language". EN hate is more implicit; broader definitions help. ZH hate is more explicit; broader definitions add noise.

### EN Ablation (clean test, 161 videos, 8B model)

| Config | AUC | Oracle ACC |
|--------|-----|-----------|
| Binary t300 | 0.748 | 74.53% |
| Binary t1000 | 0.775 | 74.53% |
| Triclass narrow t300 | 0.782 | 75.78% |
| Triclass narrow t1000 | 0.808 | 74.53% |
| **Triclass broad t1000** | **0.812** | **75.16%** |
| Triclass norules t1000 | 0.806 | 75.78% |

### Unified method table (same config on both datasets)

| Config | EN AUC | EN Oracle | ZH AUC | ZH Oracle |
|--------|--------|-----------|--------|-----------|
| Binary t300 | 0.748 | 74.53% | 0.875 | 81.88% |
| Triclass narrow t1000 | 0.808 | 74.53% | 0.870 | 81.21% |
| **Triclass broad t1000** | **0.812** | **75.16%** | 0.860 | **81.94%** |

### Target check

| Dataset | Best unified method | Oracle ACC | Target 80%? |
|---------|---------------------|------------|-------------|
| MHClip_EN (clean 161) | Triclass broad t1000 | 75.16% | **NO** (-4.84pp) |
| MHClip_ZH (clean 157) | Triclass broad t1000 | 81.94% | YES |

**Status**: EN still blocked. Best oracle is 75.16% on clean test. AUC 0.812 needs to reach ~0.825 to give 80% ACC. Unsupervised threshold selection not yet evaluated on this config.

### Data issue uncovered
EN full test set has 39/182 videos with missing media files (12 Normal, 7 Offensive, 2 Hateful). With default=Normal for these, oracle ACC on full 182 drops to 73.08%. Clean test (161) is the correct evaluation set per user's "exclude if no media" constraint.

### Result files
- `results/holistic_8b/MHClip_EN/test_triclass*.jsonl` (multiple variants)
- `results/holistic_8b/MHClip_ZH/test_triclass*.jsonl` (multiple variants)

### Scripts modified
- `src/score_holistic_2b.py` — Added `--transcript-limit` and `--triclass-style {narrow, broad, norules}` args

---

## Remaining Gap (unified method, 2026-04-12)

**Best unified method**: 8B + Triclass broad + transcript 1000 chars
**EN**: 75.16% oracle ACC (clean 161 test) — **4.84pp below target**
**ZH**: 81.94% oracle ACC (clean 157 test) — passes target

**Key unknowns**:
1. Train-derived unsupervised threshold has NOT been run on triclass broad t1000 — actual label-free ACC unknown (could be lower than oracle)
2. Oracle ACC uses test set labels — not a valid method per EMNLP standards

**Possible next steps** (not yet tried, for future session):
- Run train split on triclass broad t1000 for label-free threshold
- Loosen system prompt (currently "answer strictly on observable evidence")
- Try different triclass label wordings
- Accept ceiling and reframe the paper

---

## Original planned Iteration 2 (now superseded by findings above)

**Hypothesis**: The "violates rules?" framing triggers safety alignment token suppression. Inverting the polarity to "Is this video fully consistent with community guidelines?" (where hate_score = P(No)) should sidestep this compression.

**4-point story**:
- *Phenomenon*: RLHF-aligned MLLMs suppress P(Yes) for harm-affirming questions. This is measurable: p_base for "violates rules?" is 0.0004 (8B), meaning the model defaults to "No" with 99.96% confidence before seeing any content.
- *Mechanism*: Deflected prompt inverts the question polarity. P(No|"consistent with guidelines?") extracts the same signal but from the opposite token, which is NOT suppressed by safety alignment.
- *Prediction*: Deflected prompt should increase P(positive) for hateful videos while maintaining low P(positive) for normal videos, improving AUC and enabling unsupervised threshold to reach >80%.
- *Counterfactual*: If deflected prompt has same AUC as original, the bottleneck is not safety alignment but the model's actual inability to distinguish hateful content.

**Backup: Iteration 3 — Observe-then-Judge**
Two calls per video with distinct roles:
- Call 1 (multimodal): Open-ended observation — describe hateful signals
- Call 2 (text-only): Given observation, does this violate rules?

---

## Result Files

### Scores
- `results/holistic_2b/MHClip_EN/test_binary.jsonl` (161 records)
- `results/holistic_2b/MHClip_EN/test_triclass.jsonl` (161 records)
- `results/holistic_2b/MHClip_EN/train_binary.jsonl` (550 records)
- `results/holistic_2b/MHClip_ZH/test_binary.jsonl` (157→149 valid records)
- `results/holistic_2b/MHClip_ZH/test_triclass.jsonl` (149 records)
- `results/holistic_2b/MHClip_ZH/train_binary.jsonl` (579 records)
- `results/holistic_8b/MHClip_EN/test_binary.jsonl` (161 records)
- `results/holistic_8b/MHClip_ZH/test_binary.jsonl` (157 records)
- `results/holistic_8b/MHClip_EN/test_binary_deflected.jsonl` (4 records — incomplete)
- `results/holistic_8b/content_free.json` (8B p_base: EN=0.00043, ZH=0.0015)

### Analysis
- `results/analysis/iteration_0.json` (full Iteration 0 metrics)
- `results/analysis/iteration_1.json` (calibration + threshold ablation)

### Calibrated
- `results/calibrated/MHClip_EN/` (calibrated scores)
- `results/calibrated/MHClip_ZH/` (calibrated scores)

### Scripts (new)
- `src/analyze_iteration0.py` — Iteration 0 analysis (metrics, distributions, unsupervised thresholds)
- `src/calibrate_and_threshold.py` — Calibration + Otsu/GMM threshold pipeline
- `src/content_free_calibration.py` — Content-free P(Yes) measurement

---

## Reproduction Commands

### 2B binary scoring (test)
```bash
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/score_holistic_2b.py --dataset MHClip_EN --mode binary --split test"
```

### 2B triclass scoring (test)
```bash
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/score_holistic_2b.py --dataset MHClip_EN --mode triclass --split test"
```

### 2B training split scoring
```bash
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/score_holistic_2b.py --dataset MHClip_EN --mode binary --split train"
```

### 8B binary scoring (test)
```bash
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/score_holistic_2b.py --dataset MHClip_EN --mode binary --split test --model Qwen/Qwen3-VL-8B-Instruct"
```

### Content-free calibration
```bash
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/content_free_calibration.py --model Qwen/Qwen3-VL-8B-Instruct"
```

### Calibration + threshold evaluation (CPU only)
```bash
sbatch --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/calibrate_and_threshold.py --dataset MHClip_EN --mode binary"
```

---

## Environment
- Conda: SafetyContradiction (vllm 0.11.0)
- Models: Qwen/Qwen3-VL-2B-Instruct, Qwen/Qwen3-VL-8B-Instruct
- GPU: 1x (via Slurm), bf16
- vLLM params: temperature=0, max_tokens=1, logprobs=20
- Media priority: mp4 > frames > exclude
- Label mapping: Hateful+Offensive → 1, Normal → 0
