# prompt_paradigm — Slurm run log

> **CORRECTION NOTICE (2026-04-13 later session, root-cause pass)**
>
> Multiple entries below (v4, v5, v6 in particular) cite "v3 p_evidence ZH oracle 0.8188"
> as a prior-art strict-beat target. **The root cause is NOT a sub-atom FP phantom** (my
> first correction was wrong). The real root cause: v3's `polarity_calibration.py:83`
> uses a user message that has the sentence `"You are a content moderation analyst."`
> DELETED from the start. Baseline's `BINARY_PROMPT` in `score_holistic_2b.py:52` starts
> with that sentence. The 45-character deletion shifts 78% of EN videos and 73% of ZH
> videos, max single-video score drift 0.2151, baseline-vs-p_evidence correlation 0.96/0.98.
> v3 p_evidence is therefore **not a re-scoring of baseline** — it is a **different
> prompt**. All "v3 p_evidence vs baseline" comparisons in v4/v5/v6 were comparing two
> different prompts without noticing. Under classical label-free methods on the shifted-
> prompt score file itself: EN tf_otsu 0.7516, tf_gmm 0.7081; ZH tf_otsu 0.7517, tf_gmm
> 0.7987 — **all below baseline on both datasets**. v3's own Ablation A integrity check
> ("A reproduces baseline within FP tolerance") was silently violated. Side-finding: v3
> p_evidence is effectively an unregistered 7th prompt config. See STATE_ARCHIVE
> §"v3 p_evidence row correction" for the full numbers.

Owner: prompt-paradigm (Teammate A). All jobs submitted one-at-a-time, max 2 concurrent,
no --dependency, no &, no scancel of foreign jobs. Each entry: job ID, command, expected
runtime, status, observed runtime, output path.

## v1 — Observe-then-Judge (gate-1 approved 2026-04-13)

| # | Job ID | Kind | Dataset | Split | Cmd summary | Expected | Status | Observed | Output |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 7949 | GPU | MHClip_EN | test  | observe_then_judge.py --dataset MHClip_EN --split test  | ~15-25 min | completed                      | ~20 min | results/prompt_paradigm/MHClip_EN/test_obsjudge.jsonl (161 records) |
| 2 | 7953 | GPU | MHClip_EN | train | observe_then_judge.py --dataset MHClip_EN --split train | ~1-2 h     | completed                      | ~1 h    | results/prompt_paradigm/MHClip_EN/train_obsjudge.jsonl (550 records) |
| 3 | 7954 | GPU | MHClip_ZH | test  | observe_then_judge.py --dataset MHClip_ZH --split test  | ~15-25 min | completed                      | ~20 min | results/prompt_paradigm/MHClip_ZH/test_obsjudge.jsonl (157 records) |
| 4 | 7959 | GPU | MHClip_ZH | train | observe_then_judge.py --dataset MHClip_ZH --split train | ~1-2 h     | completed                      | ~1 h    | results/prompt_paradigm/MHClip_ZH/train_obsjudge.jsonl (579 records) |
| 5 | 8014 | CPU | —         | —     | eval_with_frozen_thresholds.py                           | ~1 min     | completed (MISS)               | ~5 s    | results/prompt_paradigm/report.json (overwritten by v2) |

v1 Gate 2 verdict: **MISS**. EN oracle 0.7391 / ZH oracle 0.7785, both below baseline oracles (0.7764 / 0.8121). Rank-preserving calibration drift from text-only Judge call. v1 retired.

## v2 — Factored Verdict (gate-1 approved 2026-04-13)

Two video-grounded calls per video: Call 1 Target-Detector, Call 2 Stance-Judge. Final score = P_T × P_S.

| # | Job ID | Kind | Dataset | Split | Cmd summary | Expected | Status | Observed | Output |
|---|---|---|---|---|---|---|---|---|---|
| v2-1 | 8027 | GPU | MHClip_EN | test  | factored_verdict.py --dataset MHClip_EN --split test  | ~20 min | completed                       | ~20 min | results/prompt_paradigm/MHClip_EN/test_factored.jsonl |
| v2-2 | 8048 | GPU | MHClip_EN | train | factored_verdict.py --dataset MHClip_EN --split train | ~1 h    | completed                       | ~1 h    | results/prompt_paradigm/MHClip_EN/train_factored.jsonl (550) |
| v2-3 | 8028 | GPU | MHClip_ZH | test  | factored_verdict.py --dataset MHClip_ZH --split test  | ~20 min | completed                       | ~20 min | results/prompt_paradigm/MHClip_ZH/test_factored.jsonl (157) |
| v2-4 | 8052 | GPU | MHClip_ZH | train | factored_verdict.py --dataset MHClip_ZH --split train | ~1 h    | completed                       | ~1 h    | results/prompt_paradigm/MHClip_ZH/train_factored.jsonl (579) |
| v2-5 | 8120 | CPU | —         | —     | eval_factored.py                                      | ~1 min  | running (submitted 2026-04-13)  | - | results/prompt_paradigm/report_v2.json, results/analysis/prompt_paradigm_report.md |

Full sbatch commands:

```
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_EN --split test"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_EN --split train"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_ZH --split test"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/prompt_paradigm/observe_then_judge.py --dataset MHClip_ZH --split train"
sbatch --cpus-per-task=2 --mem=4G --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/prompt_paradigm/eval_with_frozen_thresholds.py"
```

## v3 — Polarity-Calibrated Probes (gate-1 approved 2026-04-13)

Two video-grounded calls: Call 1 Evidence-Probe (violates framing, FN-biased), Call 2 Compliance-Probe (consistent framing, FP-biased after negation). Fusion: logit-space bias-cancellation `score = sigmoid(0.5 * (logit(p_E) - logit(p_C)))`. Both prompts copied verbatim from frozen `src/score_holistic_2b.py` (BINARY_PROMPT + DEFLECTED_BINARY_PROMPT). GPU budget: up to 2 concurrent jobs (per team-lead authorization).

| # | Job ID | Kind | Dataset | Split | Cmd summary | Expected | Status | Observed | Output |
|---|---|---|---|---|---|---|---|---|---|
| v3-1 | 8124 | GPU | MHClip_EN | test  | polarity_calibration.py --dataset MHClip_EN --split test  | ~30-40 min | completed | ~21 min | results/prompt_paradigm/MHClip_EN/test_polarity.jsonl (161) |
| v3-3 | 8126 | GPU | MHClip_ZH | test  | polarity_calibration.py --dataset MHClip_ZH --split test  | ~30-40 min | completed | ~22 min | results/prompt_paradigm/MHClip_ZH/test_polarity.jsonl (157) |
| v3-2 | 8130 | GPU | MHClip_EN | train | polarity_calibration.py --dataset MHClip_EN --split train | ~2 h       | completed | ~54 min | results/prompt_paradigm/MHClip_EN/train_polarity.jsonl (550) |
| v3-4 | 8131 | GPU | MHClip_ZH | train | polarity_calibration.py --dataset MHClip_ZH --split train | ~2 h       | completed | ~1 h 48 min | results/prompt_paradigm/MHClip_ZH/train_polarity.jsonl (579) |
| v3-5 | 8143 | CPU | —         | —     | eval_polarity.py                                          | ~1 min     | completed (MISS) | ~5 s | results/prompt_paradigm/report_v3.json, results/analysis/prompt_paradigm_report.md |

v3 Gate 2 verdict: **MISS**. Oracle-first failed (EN 0.77640 = baseline 0.7764; ZH 0.81208 = baseline 0.8121 — both tied, not strict beat). AP1 self-binding violation: prob_avg EN oracle = logit_fused EN oracle = 0.7764. Ablation A leak on ZH: evidence_only oracle = fused oracle = 0.8121. Ablation E sign check passed on both datasets (mechanism directionally correct but insufficient). Diagnostic: fused Bhattacharyya overlap (EN 0.883, ZH 0.695) is worse than evidence-only (EN 0.791, ZH 0.579) — Call 2 compliance probe is near-saturated (EN pos_mean 0.895/neg_mean 0.776) and adds noise, not orthogonal signal. v3 retired per AP1 clause.

Note: v3 runs 2 MLLM calls per video (vs v2's 2 calls), so expected per-split runtime is roughly 2× a single-call baseline — ~30-40 min for a test split, ~2 h for a train split.

Full sbatch commands:

```
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_EN --split test"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_EN --split train"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_ZH --split test"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/polarity_calibration.py --dataset MHClip_ZH --split train"
sbatch --cpus-per-task=2 --mem=4G --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/prompt_paradigm/eval_polarity.py"
```

## v4 — Modality-Split Evidence Probes (gate-1 approved 2026-04-13)

Two video-grounded calls on DISJOINT input supports: Call 1 Visual-Evidence-Probe (frames only, title/transcript stripped), Call 2 Text-Evidence-Probe (no frames, title+transcript only). Same "violates rules?" question on both. Fusion: rank-space noisy-OR against train-split reference: `score = 1 - (1 - rank_train(p_vis)) * (1 - rank_train(p_text))`. GPU budget: 2 concurrent (authorized for this track).

| # | Job ID | Kind | Dataset | Split | Cmd summary | Expected | Status | Observed | Output |
|---|---|---|---|---|---|---|---|---|---|
| v4-1 | 8149 | GPU | MHClip_EN | test  | modality_split.py --dataset MHClip_EN --split test  | ~30-40 min | completed | ~25 min | results/prompt_paradigm/MHClip_EN/test_modality.jsonl |
| v4-3 | 8150 | GPU | MHClip_ZH | test  | modality_split.py --dataset MHClip_ZH --split test  | ~30-40 min | completed | ~29 min | results/prompt_paradigm/MHClip_ZH/test_modality.jsonl |
| v4-2 | 8152 | GPU | MHClip_EN | train | modality_split.py --dataset MHClip_EN --split train | ~1-2 h     | completed | ~55 min | results/prompt_paradigm/MHClip_EN/train_modality.jsonl (550) |
| v4-4 | 8153 | GPU | MHClip_ZH | train | modality_split.py --dataset MHClip_ZH --split train | ~1-2 h     | completed | ~1 h 5 min | results/prompt_paradigm/MHClip_ZH/train_modality.jsonl (540) |
| v4-5 | 8162 | CPU | —         | —     | eval_modality.py                                   | ~1 min     | completed (MISS) | ~5 s | results/prompt_paradigm/report_v4.json |

v4 Gate 2 verdict: **MISS**. Clause 1 (oracle-first) FAIL (EN fused 0.7640 vs 0.7764; ZH fused 0.7987 vs 0.8121). Clause 2 (unified cell) FAIL (no aggregator satisfies clause 1). Clause 3 (A/B load-bearing) LEAK on EN (text-only oracle 0.7640 = fused 0.7640). Clause 4 (prior-art vs v3 p_evidence) LEAK on BOTH (v3 0.7702/0.8188 > v4 fused 0.7640/0.7987). Clause 5 (Ablation D AP1) VIOLATION (prob_avg matches rank-nor on both). Clause 6 (Ablation E rescue phenomenon) PASS (corr EN 0.295 / ZH 0.493 < 0.7; rescue rates 54-77% all ≥ 30%). Clause 7 (N_test) PASS. Diagnostic: variable-modality-carriage phenomenon is empirically present (Clause 6) but the holistic joint prompt (v3 p_evidence) already exploits it more effectively than any rank-space fusion of two disjoint-support calls. The MLLM's own cross-modal attention is a stronger fusion operator than post-hoc rank-noisy-OR. v4 retired. Returning to Gate 1 with v5.

Full sbatch commands:

```
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/modality_split.py --dataset MHClip_EN --split test"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/modality_split.py --dataset MHClip_ZH --split test"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/modality_split.py --dataset MHClip_EN --split train"
sbatch --gres=gpu:1 --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && python src/prompt_paradigm/modality_split.py --dataset MHClip_ZH --split train"
sbatch --cpus-per-task=2 --mem=4G --wrap "source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh && conda activate SafetyContradiction && cd /data/jehc223/EMNLP2 && python src/prompt_paradigm/eval_modality.py"
```

### v5 pipeline runner started 2026-04-13 12:34:53
- v5 runner W1 test: MHClip_EN/test job 8166
- v5 runner W1 test: MHClip_ZH/test job 8167
- v5 runner W2 train: MHClip_EN/train job 8170
- v5 runner W2 train: MHClip_ZH/train job 8171
- v5 runner W3 eval: job 8188
- v5 runner finished 2026-04-13 13:44:56

### v6 pipeline runner started 2026-04-13 14:35:50
- v6 runner W1 test/axes: MHClip_EN/test/axes job 8202
- v6 runner W1 test/axes: MHClip_ZH/test/axes job 8203
- v6 runner W2 test/control: MHClip_EN/test/control job 8204
- v6 runner W2 test/control: MHClip_ZH/test/control job 8205
- v6 runner W3 train/axes: MHClip_EN/train/axes job 8208
- v6 runner W3 train/axes: MHClip_ZH/train/axes job 8209
- v6 runner W4 train/control: MHClip_EN/train/control job 8211
- v6 runner W4 train/control: MHClip_ZH/train/control job 8212
- v6 runner W5 eval: job 8220
- v6 runner finished 2026-04-13 17:00:56
