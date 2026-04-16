# Task brief — ALARM (adapted to hateful video detection)

## Why this brief exists

ALARM (KDD 2026, Lang et al.) is a per-image harmful-meme detection
framework. Our 4 benchmarks are video. Per user directive
2026-04-15 (`feedback_meme_to_video_8frames.md`):

- Sample **8 frames per video** uniformly
- **One LLM call per video** (not per-frame), passing all 8 frames
  as a single multi-image prompt
- **Rewrite the prompt text** from meme/image framing to video
  framing
- **Pairwise comparison stages** (ALARM's Experience stage compares
  two memes side-by-side) become **video-vs-video** pairs: 16
  frames in one call, prompt labels "Video A: frames 1-8; Video B:
  frames 9-16"

## Upstream

- Repo: `external_repos/alarm/`
- Paper: "From Shallow Humor to Metaphor: Towards Label-Free Harmful
  Meme Detection via LMM Agent Self-Improvement" — KDD 2026
- Backbone upstream: `Qwen2.5-VL-72B-Instruct` via HF transformers +
  `Qwen2_5_VLForConditionalGeneration` (file
  `src/model/utils/models/qwen2vl_model.py:76-85`)
- Framework upstream: **HF transformers** (not vLLM)

## Framework decision

Upstream is HF transformers. Per `feedback_baseline_framework_match.md`
we stay in HF transformers. **Do NOT substitute vLLM here** —
ALARM's Label stage uses direct logit access
(`Confidence-based Explicit Meme Identification` per upstream
README), which vLLM's `LLM.chat` API does not expose cleanly. HF
transformers is the faithful path.

## Model — Qwen2.5-VL-72B-Instruct-AWQ

72B bf16 ≈ 144 GB weights alone; will not fit on an 80 GB A100.
Community AWQ quant (`Qwen/Qwen2.5-VL-72B-Instruct-AWQ`) drops
weights to ~40 GB, fits with activations under `gpu_memory_utilization=0.92`.
Precedent: MARS 32B-AWQ was user-approved for the same reason.
Document this in `src/alarm_repro/README.md` as a hard-block
quantization substitution.

If AWQ load fails in HF transformers, fall back to
`load_in_4bit=True` via `bitsandbytes` (slower but reliable). Flag
if neither works — that's a real escalation.

## Pipeline — 5 stages verbatim, adapted for video input

Upstream has 5 runners that must be ported as-is, with only the
item-level input layer changed:

1. **Label** (`src/stages/1_label.py` or equivalent) — computes a
   confidence score via output logits on the question "is this
   content hateful?". **Per-video call** passing 8 frames; prompt
   rewritten from meme to video (see prompts section below).
2. **make_embeddings** — uses the LMM to produce per-item
   embeddings for retrieval. Adapt: per-video call with 8 frames,
   extract the final hidden state / logit-over-vocab as upstream
   does.
3. **conduct_retrieval** — builds a k-NN retrieval pool across
   items. Each pool element is a video (not a meme); similarity
   computed over video embeddings from stage 2.
4. **Experience (pairwise)** — upstream's *key* step: compares two
   items side-by-side to build positive/negative training pairs
   for self-improvement. **Adapt to video-vs-video**: one call with
   **16 frames** (8 from video A + 8 from video B), prompt text
   says "Video A: frames 1-8. Video B: frames 9-16. Compare the
   two videos and determine which one is more likely to contain
   hateful content, and why." The rest of upstream's Experience
   logic (pair selection, iterative refinement) operates over
   video pairs instead of meme pairs.
5. **Reference / InPredict** — final prediction stage; upstream
   uses retrieved pairs as in-context examples. Adapt: each
   in-context example is a video (8 frames) with a label; the
   target is a video (8 frames); one call per target video.

## Prompts — meme → video rewrite rules

For every `"This meme"`, `"the image"`, `"this image"` reference in
upstream's prompt strings, rewrite to:
- `"this video"`
- `"these 8 frames uniformly sampled from a video"`
- `"the video"`

For Experience-stage prompts that describe "two memes" / "meme A"
and "meme B", rewrite to `"Video A (frames 1-8)"` and
`"Video B (frames 9-16)"`, and add one sentence telling the model
which frames correspond to which video.

**Every upstream instruction other than the meme→video rewrite is
preserved byte-for-byte.** Do not rewrite analysis instructions,
confidence phrasing, label vocabulary, or length limits.

## Frames

- 8 frames per video via uniform sampling
- Source: `<dataset_root>/frames_16/<vid>/frame_NNN.jpg` (already
  extracted)
- Indices: `[0, 2, 4, 6, 8, 10, 12, 14]` (even spacing)
- For Experience-stage video pairs, concatenate two frame lists →
  16 frames total

## Data flow

- Input: 4 test splits from our `data_utils.load_annotations(dataset)`
  + `splits/test_clean.csv`
- Output: `results/alarm/<dataset>/test_alarm.jsonl` with schema
  `{"video_id": vid, "pred": 0|1, "confidence": float, "reasoning": str}`
- Evaluable via `eval_generative_predictions.eval_one()`

## Where to write

- `src/alarm_repro/` — new dir
  - `reproduce_alarm.py` — main driver, 5-stage pipeline
  - `stages/` — sub-modules per stage if keeping upstream's file
    structure (recommended for audit clarity)
  - `README.md` — upstream commit + every deviation listed
    explicitly
  - Dataset adapter: `alarm_video_dataset.py` that wraps our 4
    test splits in the ALARM `{id, image_list, text, label}`
    shape, where `image_list` is the 8 frames

Do NOT edit upstream `external_repos/alarm/` — read-only reference.

## Deviations list (for README)

1. Scope: meme → video, via 8-frame multi-image adaptation
   (user rule 2026-04-15)
2. Prompts: meme language replaced with video language in all
   strings; Experience-stage "2 memes" → "2 videos (video A frames
   1-8, video B frames 9-16)"
3. Model: 72B bf16 → 72B-AWQ (hard-block quantization)
4. Dataset: FHM/MAMI/ToxiCN → our 4 video benchmarks; retrieval
   pool is video-to-video

No other deviations. If you find a stage that genuinely can't be
adapted to video, flag it (don't silently skip) and Feishu back to
the director.

## Syntax check (per `feedback_no_smoke_test.md`)

```bash
conda activate SafetyContradiction
python -m py_compile src/alarm_repro/reproduce_alarm.py
python -m py_compile src/alarm_repro/stages/*.py
python -m py_compile src/alarm_repro/alarm_video_dataset.py
python -c "import sys; sys.path.insert(0,'src/alarm_repro'); import reproduce_alarm, alarm_video_dataset; print('imports ok')"
```

**No sbatch. No smoke. No touching `results/`.** Report back with:
1. Audit notes — which upstream file each stage ports from and which
   ones needed non-trivial prompt adaptation
2. File list (new under `src/alarm_repro/`)
3. Syntax check log
4. Any deviation flagged explicitly

Director reviews against upstream, approves, submits real sbatch.
