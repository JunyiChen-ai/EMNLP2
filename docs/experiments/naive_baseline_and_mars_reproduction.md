# Naive 2B text + MARS faithful reproduction (comparison baselines)

**Date**: 2026-04-14
**Backbone**: Qwen3-VL-2B-Instruct (our fixed project model)
**Splits**: `test_clean.csv` with `SKIP_VIDEOS` applied — EN 161, ZH 149
(8 broken mp4s skipped), HateMM 215.

Two comparison baselines added alongside **our method**
(`src/our_method/score_holistic_2b.py` — 2B binary_nodef + TF-Otsu/GMM/li_lee)
to give the final paper an external comparison column under identical
backbone + splits. "Baseline" in this document refers to the two *external*
reference points; our own pipeline is called **our method** throughout.

### 8 broken ZH mp4s → `SKIP_VIDEOS`

Discovered during the MARS run: 8 MHClip_ZH videos have mp4 files that pass
the size check but fail vLLM's Qwen3-VL video decoder with `"Expected
reading N frames, but only loaded 0 frames from video."` They also failed
in the naive 2B run (empty-string responses) and in our method (dropped by
`quick_eval_all.py` because score=None). All three methods fail on the
same 8 IDs. We added `SKIP_VIDEOS` to `src/our_method/data_utils.py` so
`get_media_path` returns `None` for them and every scoring script
skips-at-top. `eval_generative_predictions.py` also honors `SKIP_VIDEOS`
so eval-time counts line up. Our method's ZH baseline was always 0.8121
on n=149 (the evaluator silently dropped those 8 long before this
session); naive and MARS ZH are now also reported at n=149 so all three
methods are apples-to-apples:
EN=161 / ZH=149 / HateMM=215.

## Part A — Naive 2B text baseline

### Pre-registration

**Question**: What accuracy does Qwen3-VL-2B-Instruct reach with the
simplest possible zero-shot prompt, with no policy / no role / no logprob
decoding — just "ask and parse the text answer"?

**Method**:
- Single user-only conversation turn (no system message).
- Prompt (verbatim):
  ```
  Is this video hateful?
  Title: {title}
  Transcript: {transcript}

  Only respond Yes or No.
  ```
- Sampling: `temperature=0, max_tokens=8`.
- Frames: up to 8 (reused `build_media_content` from the baseline script).
- Parse: `response.strip().lower().startswith("yes")` → 1,
  `startswith("no")` → 0, else → -1 (unparseable).

**Hypothesis** (before running): 2B will refuse to predict positive most
of the time due to safety alignment conservatism → high negative-class
bias → low mF1 but middling accuracy.

### Execution

- Script: `src/naive_baseline/score_naive_2b.py`
- Slurm job 8261 (1 GPU, --all, ~15 min wall)
- Output: `results/naive_2b/{dataset}/test_naive.jsonl`
- Eval: `src/naive_baseline/eval_generative_predictions.py --naive`

### Results

| Dataset | n | ACC | mF1 | n_pos_pred |
|---|---|---|---|---|
| MHClip_EN | 161 | 0.7143 | 0.5289 | 11 |
| MHClip_ZH | 149 | 0.7450 | 0.5822 | 11 |
| HateMM | 215 | 0.6744 | 0.5868 | 30 |

**Hypothesis confirmed**: the model predicts positive only 11/161 (EN) and
11/149 (ZH) times — extreme conservatism. ZH is evaluated at n=149 (the 8
broken mp4s are skipped at `get_media_path` via `SKIP_VIDEOS`).

## Part B — MARS faithful reproduction

### Paper

**Title**: Training-Free and Interpretable Hateful Video Detection via
Multi-stage Adversarial Reasoning
**Authors**: Multimodal Intelligence Lab (MIL), University of Exeter
**Date**: Jan 2026 (arxiv 2601.15115)
**Repo**: <https://github.com/Multimodal-Intelligence-Lab-MIL/MARS>
**Reference commit**: `72e1618943d36edb26ba0a24d0b4b417978d38e6`

### MARS pipeline (from `code/MARS/Qwen.py`)

4 independent VLM calls per video, each a fresh user-only conversation
containing the 16 frames + a stage-specific prompt:

| Stage | Prompt role | Output schema |
|---|---|---|
| 1 | Objective description (no interpretation) | `{objective_visual_description: str}` |
| 2 | Assume hateful, gather evidence | `{evidence, reasoning, strength}` |
| 3 | Assume non-hateful, gather evidence | `{evidence, reasoning, strength}` |
| 4 | Meta-synthesis, weigh alternatives | `{final_decision: {label, confidence, key_factors, reasoning}}` |

Stage 4 embeds stages 1/2/3 parsed responses via `json.dumps(...)` into
its prompt. The final prediction is `stage_4.final_decision.label` ∈ {0, 1}.

### Our reproduction

- Script: `src/mars_repro/reproduce_mars_2b.py`
- **Stage prompts**: vendored byte-exact from `code/MARS/Qwen.py:137-219`
  as Python string constants with a commit-SHA header comment.
- **Transcript append rule**: vendored from `Qwen.py:224-228` —
  `"\n\nTranscript: {transcript}"` if non-empty, else
  `"\n\nTranscript: No transcript available."`.
- **Sampling params**: `temperature=0.7, top_p=0.9, max_tokens=4096`
  (matching MARS's `do_sample=True, temperature=0.7, top_p=0.9,
  max_new_tokens=4096` on line 286-290). `seed=42` added for run-to-run
  reproducibility under vLLM.
- **Frames**: 16 per stage. Pre-extracted frames folders pass 16 file
  URLs; mp4 inputs pass `{"type": "video_url", ...}` and vLLM handles
  frame extraction internally for Qwen3-VL.
- **`limit_mm_per_prompt`**: `{image: 16, video: 1}`.
- **`max_model_len`**: 40960 (bumped from 32768 after first-pass
  discovered one EN video with 32896-token Stage 4 prompt).

### Faithfulness scope

Faithful:
- All 4 stage prompts (byte-exact)
- Transcript append rule
- JSON-cleaning parser (primary path is verbatim from `Qwen.py:316-333`)
- Stage 4 label extraction from `final_decision.label`
- Sampling params (temperature, top_p, max_tokens)
- Frame count (16)

Adapted / bumped:
- Backbone: Qwen3-VL-2B-Instruct (not Qwen2.5-VL-32B / Llama4 / GPT5-mini /
  Gemini2.5-Flash). This is our fixed project backbone.
- Inference: vLLM v0.11.0 (not HuggingFace transformers + int8 BnB quant).
- Split: our fixed `test_clean.csv` (not MARS's 5-fold CV).
- `max_model_len`: 40960 (not 32768) to accommodate 16 frames + Stage 4
  prompt + transcripts.

Added (NOT in original MARS):
- **Fallback JSON parser chain** (F1-F4 in `clean_json_response`) to
  recover 2B-scale format infidelities — see §"Stage-1 infidelity" below.
- **`extract_stage4_label_from_text()`** regex extractor to recover label
  from non-JSON Stage 4 outputs.

### Stage-1 infidelity discovered during first pass

First-pass run (strict JSON parser, no fallback) produced these failure
rates per stage:

| Dataset | n | Stage 1 parse fail | Stage 2 fail | Stage 3 fail | Stage 4 fail | pred=-1 |
|---|---|---|---|---|---|---|
| MHClip_EN | 161 | 75 (47%) | 1 | 2 | 1 | 2 |
| MHClip_ZH | 157 | 81 (52%) | 0 | 2 | 0 | 9 |
| HateMM | 215 | 161 (75%) | 3 | 4 | 1 | 1 |

**Root cause of the Stage-1 failures**: Qwen3-VL-2B frequently emits

```
["objective_visual_description": "The video shows a rooster standing..."]
```

instead of `{...}`. It is mis-reading the MARS prompt's format hint
`Return ONLY valid JSON with one key: ["objective_visual_description"]`
as the literal output delimiter. Strict `json.loads` fails because
the `[...]` container doesn't carry `"key": value` syntax. Stages 2-4
have different format hints (3 keys / nested dict) and are largely
unaffected.

### Fallback parser

Added to `clean_json_response()` as F1-F4:

| Fallback | Pattern | Recovery |
|---|---|---|
| F1 | outer `[` ⇄ `{` swap | `["k": v]` → `{"k": v}` |
| F2 | balanced `{...}` scan inside text | dangling prose with embedded JSON |
| F3 | balanced `[...]` scan then F1 swap | as F1 but embedded |
| F4 | wrap raw text as `{default_key: text}` | pure-prose outputs |

Plus `extract_stage4_label_from_text()` for Stage 4 verdicts — regex-based
label-token search in the raw text (`"label": 1`, `label is 0`, etc.).

### Retry pass

After the first-pass runs completed:
1. Backed up all 3 jsonls as `.preretry_20260414`.
2. Filtered each jsonl to keep only records where all 4 stages
   `parse_ok=True` AND `pred ∈ {0, 1}`.
3. Resubmitted `reproduce_mars_2b.py` with the new fallback parser active.
   Resume logic re-processed only the removed records.

Post-retry audit:

| Dataset | Residual bad | Stage fails | pred=-1 |
|---|---|---|---|
| MHClip_EN | **0** | all 0 | 0 |
| MHClip_ZH | 9 | all 0 | 9 (8 broken mp4 + 1 parser leak, see below) |
| HateMM | 4 | turn_4: 4 (regex rescued) | 0 |

**Breakdown of the 9 ZH `pred=-1`**:
- **8** are video-decode failures (see §"8 broken ZH mp4s" at the top) —
  the `mars_one_video` call threw an exception at the `llm.chat` step, the
  record only has `{video_id, pred:-1, error}`. These 8 are now skipped at
  `get_media_path` and excluded from eval via `SKIP_VIDEOS`.
- **1** is `BV1wP411z7P9`: model emitted a flat `"final_decision": "0"`
  (string) instead of the nested `{"final_decision": {"label": 0, ...}}`
  MARS schema. `extract_final_label` only handles the nested form; regex
  fallback's label-keyword patterns don't match `"final_decision": "0"`.
  Slipped through; counted as `pred=0` in the metric.

### Final results (fallback parser active, `SKIP_VIDEOS` applied)

| Dataset | n | ACC | mF1 | n_pos_pred | residual pred=-1 |
|---|---|---|---|---|---|
| MHClip_EN | 161 | 0.6584 | 0.6280 | 66 | 0 |
| MHClip_ZH | 149 | 0.7450 | 0.7047 | 49 | 1 |
| HateMM | 215 | 0.6977 | 0.6964 | 143 | 0 |

For reference, the first-pass (strict parser) EN number was
ACC 0.6335 / mF1 0.5978 — the fallback parser recovered ~2.5pp ACC on EN.

### Side-by-side comparison

| Method | EN ACC / mF1 (n=161) | ZH ACC / mF1 (n=149) | HateMM ACC / mF1 (n=215) |
|---|---|---|---|
| **Our method** (2B binary_nodef + TF-threshold) | **0.7640 / 0.6532** | **0.8121 / 0.7871** | **~0.8047** |
| Naive 2B (single-turn plain-text) | 0.7143 / 0.5289 | 0.7450 / 0.5822 | 0.6744 / 0.5868 |
| MARS 2B faithful (4-stage) | 0.6584 / 0.6280 | 0.7450 / 0.7047 | 0.6977 / 0.6964 |

### Observations

1. **Neither comparison baseline reaches our method on any dataset.** Our
   single-pass holistic-score + unsupervised-threshold pipeline outperforms
   both "ask the model directly" and "4-stage adversarial reasoning" on the
   same 2B backbone.
2. **MARS underperforms even naive 2B on ACC for EN and ties on ZH**
   (0.6584 vs 0.7143 EN; 0.7450 vs 0.7450 ZH). The 4-stage reasoning's
   "hate hypothesis" stage introduces a positive-class bias: MARS predicts
   66 positive on EN vs. naive's 11. Many of these are false positives,
   dragging ACC down.
3. **MARS dominates naive on mF1 for all 3 datasets** (0.6280 vs 0.5289
   EN; 0.7047 vs 0.5822 ZH; 0.6964 vs 0.5868 HateMM). The 4-stage prompt
   forces symmetric commitment across both classes, which helps macro-
   averaged F1 despite hurting raw ACC on imbalanced datasets.
4. **On HateMM, MARS > naive on both metrics** (0.6977/0.6964 vs
   0.6744/0.5868). HateMM's higher base rate for positive class means
   MARS's hate-bias aligns with ground truth, turning the bias into a
   feature rather than a bug.
5. **Stage-1 failures are a 2B-vs-32B capability gap finding**, not a bug
   in our reproduction. The MARS prompt's `one key: ["..."]` format hint
   is ambiguous to 2B-scale models. This is worth flagging in any future
   MARS study on smaller backbones — the strict parser baseline would
   report misleadingly low numbers for these samples otherwise.

### Artifacts

- `src/our_method/data_utils.py` — `SKIP_VIDEOS` set (8 broken ZH mp4s)
- `src/naive_baseline/score_naive_2b.py`, `eval_generative_predictions.py`
- `src/mars_repro/reproduce_mars_2b.py`
- `third_party/MARS/` at commit `72e1618` (gitignored)
- `results/naive_2b/{MHClip_EN,MHClip_ZH,HateMM}/test_naive.jsonl`
- `results/mars_2b/{MHClip_EN,MHClip_ZH,HateMM}/test_mars.jsonl`
- `results/mars_2b/*/test_mars.jsonl.preretry_20260414` (first-pass backups)
- `results/analysis/naive_2b_eval.json`, `mars_2b_eval.json`
- `logs/naive_2b_all.out`, `logs/mars_2b_enzh.out`, `logs/mars_2b_hatemm.out`,
  `logs/mars_2b_retry_enzh.out`, `logs/mars_2b_retry_hatemm.out`
