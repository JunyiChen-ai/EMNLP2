# Task brief — MATCH-HVD stage 2b.5 (Jina-CLIP frame alignment)

## Why this brief exists

During the MATCH stage 2 audit you correctly identified upstream
`preprocess/time_unit.py` as a missing intermediate step between
stages 2a/2b (hate.json / nonhate.json) and stage 2c (judgement).
Your single-block substitution was a downgrade; per strict-review
rule downgrades are only accepted on hard blocks, and Jina-CLIP-v2
is implementable — it's not a hard block. So we're doing it
properly. Option A is chosen.

## Upstream source of truth

`/data/jehc223/EMNLP2/external_repos/match_hvd/preprocess/time_unit.py`

Read this file **line by line**. The new code mirrors it with three
documented deviations driven by available signal (frames_32
extraction + per-frame OCR are prerequisites we build / skip).

## Scope — three new files + one update

### 3a. `src/match_repro/extract_frames_32.py`
Parallel to `extract_frames.py` (which produces 16 frames) — this one
produces **32 uniformly-sampled frames** per video under
`<dataset_root>/frames_32/<vid>/frame_NNN.jpg` (NNN = 000..031).

- Reuse the same preference order as `extract_frames.py`: existing
  `frames/<vid>/` directory > mp4 via decord
- 32 uniformly-spaced indices via `np.linspace(0, n-1, 32, dtype=int)`
- Same ProcessPoolExecutor + resume-skip pattern
- CPU only
- All 4 datasets: MHClip_EN, MHClip_ZH, HateMM, ImpliHateVid

Our 4 datasets already have `frames/<vid>/` (EN/ZH/HateMM) or mp4
(ImpliHateVid) — same as for `frames_16`.

### 3b. `src/match_repro/time_unit_jina_clip.py`
Byte-for-byte port of upstream `time_unit.py`, with these documented
deviations:

1. **Upstream `./MLLM/jina-clip-v2` → HF `jinaai/jina-clip-v2`** (same
   weights; we load via `AutoModel.from_pretrained('jinaai/jina-clip-v2', trust_remote_code=True)`)
2. **Frames path**: `<dataset_root>/frames_32/<vid>/` (what 3a
   produces)
3. **OCR signal**: we don't have per-frame OCR. Upstream's
   `find_best_matching_frame` computes
   `frame_text = ocr_data.get(str(idx), "") + " " + transcript_splits[idx]`
   — substitute with `frame_text = transcript_splits[idx]` (ocr
   contribution dropped). Document this: upstream uses OCR+transcript
   per frame; we use transcript-only per frame. No OCR extractor is
   added as a prerequisite (separate scope).

Otherwise verbatim:
- `split_answer` — upstream lines 9-29, regex `[。！？?.]`, pad to 4
  slots, even distribution when > 4 sentences
- `load_transcripts` — upstream lines 38-71, 32-segment split logic
  (sentence-based with word-count interpolation)
- `find_best_matching_frame` — upstream lines 72-101; compute
  `image_emb @ answer_emb.T` + `text_emb @ answer_emb.T`, pick
  argmax; return `{id, answer, frame, ocr, transcript}` per best
  frame
- `clip_run` — upstream lines 103-126; resume logic via
  `processed_video_ids`, write to output json after each video
- `truncate_dim = 512` (upstream exact)
- Output files: `results/match_qwen2vl_7b/<dataset>/time_unit_hate.json`
  and `.../time_unit_nonhate.json`
- Our wrapper reads transcripts from `load_annotations(dataset)` in
  our `data_utils.py` (single string per video) and emits the same
  `transcript_dict[vid]` = list-of-32 structure upstream builds from
  a jsonl

### 3c. Update `src/match_repro/judgement_vllm.py`
Currently consumes `hate.json` / `nonhate.json` with single-block
entries. Update to:

1. Read `time_unit_hate.json` / `time_unit_nonhate.json` (the
   per-subclaim per-frame output from 3b)
2. `format_entries` — copy upstream `judgement.py:48-77` verbatim:
   loops over entries, formats `**Reason {i+1}:**` text blocks with
   `ocr` and `transcript` fields (will show `N/A` for videos where
   the upstream OCR extractor produced nothing — OK), builds the
   2×2 grid from each entry's `frame` field (not first-4 frames)
2. Uses `cv2.imread` + `cv2.resize((128,128))` + `np.hstack`/`vstack`
   grid construction exactly as upstream (`judgement.py:62-76`)
3. Frames path changes from `frames_16` to `frames_32` since the
   per-subclaim `frame` fields refer to entries from `frames_32/`
4. Same vLLM local substitution (`Qwen/Qwen2.5-VL-7B-Instruct`,
   `max_tokens=4096`) — the API→vLLM substitution rule still applies
5. Final output unchanged: `judge_timeunit.json` with
   `[{"id": vid, "summary": "<text>"}]`

### 3d. Pre-download jinaai/jina-clip-v2 on login node
Add a short commented helper at the top of
`src/match_repro/time_unit_jina_clip.py`'s docstring, and a
comment in the MATCH README, noting that the first run downloads
`jinaai/jina-clip-v2` from HF Hub. Compute nodes are air-gapped,
so the download must be warmed on the login node before the real
sbatch submission. **You do NOT run the download yourself** — just
document the command. I (director) will run it from the login node
as part of the pre-flight.

### 3e. `src/match_repro/finalize_labelfree.py` — no change
Still parses judge summary for yes/no. It doesn't know or care
about frames_32 vs frames_16.

### 3f. `src/match_repro/run_match_agents.py` — no change
Still reads from `frames_16` for stages 2a/2b. The Qwen2-VL-7B-Instruct
agents still see 8 frames as upstream does. Only the judge side
uses `frames_32` + Jina-CLIP alignment.

### 3g. `src/match_repro/README.md` — update
Document the full pipeline:
```
  frames_16/  ──► stage 2a (Qwen2-VL HF) ──► hate.json
  frames_16/  ──► stage 2b (Qwen2-VL HF) ──► nonhate.json
  frames_32/  ──► stage 2b.5 (Jina-CLIP-v2 HF) ──► time_unit_hate.json, time_unit_nonhate.json
  frames_32/  ──► stage 2c (Qwen2.5-VL-7B vLLM) ──► judge_timeunit.json
  judge_timeunit.json ──► stage 4 (label-free parse) ──► test_match.jsonl
```

List all three documented deviations:
1. `./MLLM/jina-clip-v2` → `jinaai/jina-clip-v2` (identical weights)
2. Per-frame OCR signal dropped (upstream uses OCR+transcript, we use transcript only)
3. API `openai.OpenAI(...)` → local vLLM (standing rule `feedback_api_to_vllm.md`)

Note upstream's `./MLLM/` path is a local modelscope cache; HF hub
equivalent is the same weights — zero weight deviation.

## Syntax check (no sbatch, no smoke)

Same as before:
```bash
conda activate SafetyContradiction
python -m py_compile src/match_repro/extract_frames_32.py
python -m py_compile src/match_repro/time_unit_jina_clip.py
python -m py_compile src/match_repro/judgement_vllm.py
python -c "import sys; sys.path.insert(0,'src/match_repro'); \
  import extract_frames_32, time_unit_jina_clip, judgement_vllm, \
         run_match_agents, finalize_labelfree; \
  print('imports ok')"
```

## Deliverable

Reply with:
1. File list — new + modified
2. Per-deviation summary — the three documented deviations verbatim
3. Syntax check output
4. A short note on any upstream line you found worth flagging (edge
   cases in `split_answer`, `load_transcripts`, etc.)

**No sbatch. No smoke. No touching `results/`.**

The director will then:
(a) pre-download `jinaai/jina-clip-v2` on login node
(b) run `extract_frames_32` as a CPU sbatch to produce all
    `frames_32/` directories (blocking dependency for 2b.5 and 2c)
(c) submit stages 2a + 2b + 2b.5 + 2c + finalize as a bundled GPU
    sbatch
