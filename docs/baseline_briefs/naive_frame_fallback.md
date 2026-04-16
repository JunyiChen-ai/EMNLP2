# Task brief — Naive scorer frame-based fallback (LLaVA-Next + Gemma-3)

## Why this brief exists

The current `src/naive_baseline/score_naive_2b.py` uses vLLM content
type `{"type": "video", "video": <mp4_path>}`. For Qwen-family and
InternVL this works. For LLaVA-Next-Mistral-7B and Gemma-3-12B-it,
vLLM 0.11.0 raises `"At most 0 video(s) may be provided in one prompt."`
Root cause: these model families don't have a vLLM video-input adapter.

**ALL 926 LLaVA-Next samples and 941 Gemma samples produced so far
are errored** — 100% failure across 4 datasets each. These results
are invalid and will be overwritten by this task.

User directive 2026-04-15: fall back to feeding **image frames** via
vLLM (since vLLM does support multi-image content for both model
families). If even image-frame vLLM doesn't work, fall back to HF
transformers for that specific model.

## Frames source

Use pre-extracted frames under `<dataset_root>/frames_16/<vid>/frame_NNN.jpg`
(NNN = 000..015). **All 4 datasets now have these**:
- `/data/jehc223/Multihateclip/English/frames_16/` (161 videos)
- `/data/jehc223/Multihateclip/Chinese/frames_16/` (149 videos)
- `/data/jehc223/HateMM/frames_16/` (215 videos)
- `/data/jehc223/ImpliHateVid/frames_16/` (401 videos — extracted by job 8426)

Originally produced by `src/match_repro/extract_frames.py` for MATCH;
we reuse them here to avoid re-extraction.

## Frame count per prompt

- **Primary**: 16 frames per video (matches the frame count used by
  our other naive baselines — consistency within the naive set)
- **Fallback**: if vLLM's `limit_mm_per_prompt` cannot be raised to
  16 for a given model, drop to 8 frames (indices `[0,2,4,6,8,10,12,14]`),
  then 4 (`[0,4,8,12]`), then 1 (middle). Document which tier was
  used in the output jsonl under a new `frame_count` field
- **Escalation**: if even 1-frame vLLM fails for a model → escalate
  to director for HF transformers fallback (per user directive
  2026-04-15: "如果它连frame vllm都support不了 那就hf")

## Output path (distinct from broken vLLM runs)

- `results/naive_frame_llava_next_7b/<dataset>/test_naive.jsonl`
- `results/naive_frame_gemma3_12b/<dataset>/test_naive.jsonl`

Do NOT overwrite existing `results/naive_llava_hf_llava_v1.6_mistral_7b_hf/`
or `results/naive_gemma3_vl_12b/` — leave those in place as broken-run
evidence; director will move them to an `_archive_broken/` subdir
after the fallback succeeds.

## Schema (unchanged from existing naive scorer)

```json
{"video_id": "<vid>", "response_text": "<raw>", "pred": 0|1|-1, "frame_count": 16, "error": "<optional>"}
```

Parse logic is unchanged (yes→1, no→0, else -1). The existing parser
in `src/naive_baseline/score_naive_2b.py` is fine.

## Where to write

- Either: add a `--frames` mode to `src/naive_baseline/score_naive_2b.py`
  that swaps the content-type construction (video vs image-list) based on
  model family
- Or: write a sibling file `src/naive_baseline/score_naive_frames.py`
  that shares the model-load + generate + parse logic

Engineer's call — whichever keeps the diff cleanest. **Do not break
the existing Qwen / InternVL video path** since those results
(`results/naive_2b/`, `results/naive_internvl3_8b/`) are valid and
depend on the video path.

## vLLM init hints per model family

- **LLaVA-Next-Mistral-7B** — set `limit_mm_per_prompt={"image": 16}`;
  the chat template accepts a list of image placeholders
- **Gemma-3-12B-it** — vLLM support as of 0.11.0 is newer; engineer
  verifies at smoke-test time. If init fails, try `trust_remote_code=True`
  or escalate to HF
- Keep everything else consistent with the existing `score_naive_2b.py`
  (same prompt, same parser, same output schema)

## Smoke test deliverable

Per model (LLaVA-Next + Gemma), run on **one video** each from
MHClip_EN with the new frame-mode code path. Report back:

1. File list (new or modified files)
2. Smoke log for LLaVA-Next: init + 1-video result (response_text + pred)
3. Smoke log for Gemma-3: init + 1-video result (response_text + pred)
4. Which frame-tier each model accepted (16 / 8 / 4 / 1)
5. If either model fails even at 1-frame vLLM → flag "NEEDS HF FALLBACK"
   explicitly — do NOT attempt HF fallback without director approval

**Do NOT submit sbatch.** Do not touch `results/`. Do not overwrite
the existing broken jsonls.

## Review gate (director applies)

1. Existing Qwen / InternVL video path untouched — verify by grepping
   the script
2. New frame-mode code uses `{"type": "image", "image": path}` list
3. Frame-tier fallback logic present (16 → 8 → 4 → 1)
4. Output paths distinct from broken runs
5. Smoke log shows parseable yes/no for at least one model
6. If NEEDS HF FALLBACK is flagged, director Feishu-escalates to user
   before approving HF rewrite
