# Task brief — MATCH-HVD stage 2 audit + stage 2c substitution

## Why this brief exists

`src/match_repro/{extract_frames.py,run_match_agents.py}` was
written under the old solo-director model without a rigorous
upstream audit. Under the new faithful-repro gate, it needs to be
re-verified line-by-line against the three upstream runners
(`MATCH_HateMM_run.py`, `MATCH_Multi_en_run.py`, `MATCH_Multi_zh_run.py`)
and extended to include stage 2c (judgement).

Upstream stage 2c (`preprocess/judgement.py`) calls an external
OpenAI-compatible API (SiliconFlow) with
`"Pro/Qwen/Qwen2.5-VL-7B-Instruct"`. Per **user directive
2026-04-15** (`feedback_api_to_vllm.md`), when upstream calls a
cloud API, we substitute **local vLLM with the same model
variant**. This is the default rule, not a deviation requiring
escalation.

## Upstream

- **Paper**: MATCH-HVD (anonymous submission)
- **Source**: https://anonymous.4open.science/r/MATCH-HVD
- **Mirror**: `external_repos/match_hvd/`
- **Relevant files** (read these before coding):
  - `external_repos/match_hvd/preprocess/MATCH_HateMM_run.py` (stage 2a+2b for HateMM)
  - `external_repos/match_hvd/preprocess/MATCH_Multi_en_run.py` (stage 2a+2b for MHClip_EN)
  - `external_repos/match_hvd/preprocess/MATCH_Multi_zh_run.py` (stage 2a+2b for MHClip_ZH)
  - `external_repos/match_hvd/preprocess/judgement.py` (stage 2c)

## Scope

**IN scope (this brief)**:
- Stage 2a: hate-evidence agent (per video, HF transformers Qwen2-VL-7B-Instruct)
- Stage 2b: nonhate-evidence agent (same framework, flipped `ifhate`)
- Stage 2c: judgement (local vLLM substitution for SiliconFlow API call)
- Final label-free verdict: parse yes/no from the judge's summary

**OUT of scope** (do NOT implement):
- Stage 3 supervised training (MFCC + ViViT + BERT + classifier).
  Stage 3 is marked `conditional on user go/no-go` in the plan; a
  separate brief will cover it later.

## Part 1 — Audit stages 2a/2b (existing code)

Re-read `src/match_repro/run_match_agents.py` line-by-line against
**all three** upstream runners. Key items to verify:

### Model load (all 3 runners use these lines)
```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, device_map=device
)
processor = AutoProcessor.from_pretrained(model_dir)
```

Upstream `model_dir = "./MLLM/Qwen2-VL-7B-Instruct"` — pulled via
`modelscope.snapshot_download`. We substitute with the HF hub name
`Qwen/Qwen2-VL-7B-Instruct` (same weights).

### Prompt — **differs per dataset**

- **HateMM** (`MATCH_HateMM_run.py`):
  ```
  You are a hate video detector. Follow these steps:
  1. As a proposer claiming that the video {ifhate} contains hate signals, review the visual content.Your task is to identify and highlight any subtle elements that support this claim.
  2. As a proposer claiming that the video {ifhate} contains hate signals, review the textual content based on the observations from the visual content.Your task is to verify these findings and identify additional subtle elements that support the claim, providing comprehensive and convincing evidence.
  Audio Transcription: {transcription}
  OCR: {ocr}
  ```

- **MHClip_EN / MHClip_ZH** (`MATCH_Multi_en_run.py`, `MATCH_Multi_zh_run.py`) — **includes Title, omits OCR truncation separately**:
  ```
  You are a hate video detector. Follow these steps:
  1. As a proposer claiming that the video {ifhate} contains hate signals, review the visual content.Your task is to identify and highlight any subtle elements that support this claim.
  2. As a proposer claiming that the video {ifhate} contains hate signals, review the textual content based on the observations from the visual content.Your task is to verify these findings and identify additional subtle elements that support the claim, providing comprehensive and convincing evidence.
  Audio Transcription: {transcription}
  Title: {title}
  OCR: {ocr}
  ```

- `ifhate = ""` for the **hate-evidence** (stage 2a) agent;
  `ifhate = "does not"` for the **nonhate-evidence** (stage 2b) agent
- `MAX_TRANSCRIPT_LENGTH = 1000`, `MAX_OCR_LENGTH = 100` (HateMM) or
  `MAX_TITLE_LENGTH = 300` (MHClip; upstream MHClip code also uses
  MAX_TITLE_LENGTH for OCR truncation — treat that as upstream-exact,
  not a bug to fix)

**For ImpliHateVid** (no upstream runner exists — this is the 4th
dataset we added): reuse the HateMM prompt template (strict binary
group-hate, no title field — our `ImpliHateVid` annotations don't
have titles; transcripts + OCR-as-empty). Document this as an
intentional extension in the README.

### Frames
- 8 frames per video, indices `[0, 2, 4, 6, 8, 10, 12, 14]` (upstream
  `qwen_run`)
- OOM fallback: indices `[0, 4, 8, 12]` (upstream `err_run`)
- Source: `<dataset_root>/frames_16/<vid>/frame_NNN.jpg` (all 4
  datasets now have these; produced by `extract_frames.py` and
  `src/match_repro/extract_frames.py`)

### Generation
- `max_new_tokens` — verify upstream's value (grep for `max_new_tokens`
  in the runners) and match it in the code. If upstream doesn't set
  it explicitly, use `max_new_tokens=2048` per
  `feedback_baseline_integrity.md`
- No temperature / top_p / top_k specified upstream → use HF defaults

### Output format
- Stage 2a writes `results/match_qwen2vl_7b/<dataset>/hate.json`
- Stage 2b writes `results/match_qwen2vl_7b/<dataset>/nonhate.json`
- Schema: `[{"id": vid, "answer": "<rationale>"}, ...]` — matches
  upstream

### Audit deliverable (Part 1)
For each item above, report in the deliverable message:
- **PASS** (existing code matches upstream) OR
- **FIX** (what was wrong, what you changed, with upstream
  file:line reference)

## Part 2 — Add stage 2c (judgement) via local vLLM

Upstream `preprocess/judgement.py` builds a 2×2 grid PNG from the
first 4 frames of each perspective, sends it + a text prompt to:
```python
openai.OpenAI(api_key=..., base_url=...)
client.chat.completions.create(model="Pro/Qwen/Qwen2.5-VL-7B-Instruct",
                               messages=[{"role":"user","content":[...]}],
                               max_tokens=4096)
```

**Our substitution**: run `Qwen/Qwen2.5-VL-7B-Instruct` **locally via
vLLM** (not HF, not the API). Rationale per
`feedback_api_to_vllm.md`: cloud API → local vLLM default.

### Implementation pattern

Write a new file `src/match_repro/judgement_vllm.py` that:

1. Loads Qwen2.5-VL-7B-Instruct via vLLM (`SamplingParams(max_tokens=4096)`,
   `limit_mm_per_prompt={"image": 2}`, `gpu_memory_utilization=0.92`,
   `max_model_len=16384` — same conventions as our existing
   `src/our_method/score_holistic_2b.py` vLLM init but swapping the
   model id)
2. For each video, reads `hate.json` + `nonhate.json` (produced by
   stages 2a/2b)
3. Builds the same 2×2 grid PNG for each perspective (copy the
   `format_entries` function from upstream `judgement.py` verbatim —
   `cv2.imread` + `cv2.resize((128,128))` + `np.hstack/vstack` + PIL
   base64 encode)
4. Builds the **exact prompt** upstream uses (copy verbatim from
   `judgement.py` lines 81-96; includes the `### Hateful Perspective:` /
   `### Non-Hateful Perspective:` / `### Image Information:` /
   `### Instructions:` sections)
5. Sends a vLLM chat-completion request with:
   - text: the prompt above
   - image 1: hate_images base64-encoded 2×2 grid
   - image 2: nonhate_images base64-encoded 2×2 grid
   - `max_tokens=4096`
6. Writes `results/match_qwen2vl_7b/<dataset>/judge_timeunit.json`
   with schema `[{"id": vid, "summary": "<text>"}]` (upstream exact)

**Fidelity deviations to document in the README**:
- API `openai.OpenAI.chat.completions.create` → vLLM local
  `LLM.chat(...)` or `LLM.generate(...)` with multimodal inputs
- Network API call becomes local inference
- Model weights are identical (Qwen2.5-VL-7B-Instruct)
- Everything else (prompt, images, max_tokens) unchanged

## Part 3 — Final label-free verdict parse

Upstream uses the judgement summary as input to a supervised
classifier in stage 3 (which we're skipping). For **label-free**
evaluation, we parse the judgement summary text directly for a
hateful/non-hateful verdict.

Write `src/match_repro/finalize_labelfree.py` that:
1. Reads `judge_timeunit.json` for each dataset
2. For each video, applies a yes/no parser on the 350-word summary:
   - Search for patterns like "hateful", "hate content", "should be
     considered hateful", "classify as hateful" → pred=1
   - Search for "not hateful", "does not contain hate", "non-hateful",
     "should not be classified as hateful" → pred=0
   - On ambiguity (both or neither), fall back to a simple
     substring-counting heuristic: count occurrences of "hateful"
     (hate-indicative) vs "not hateful" / "non-hateful"
     (non-hate-indicative); majority wins
   - On total failure: pred=0 (conservative)
3. Writes `results/match_qwen2vl_7b/<dataset>/test_match.jsonl` with
   schema `{"video_id": vid, "pred": 0|1, "summary": "<full judge summary>"}`
4. Compatible with `eval_generative_predictions.eval_one()`

Document the parser in the README. This is the only **intentional**
adaptation of MATCH for label-free use (since upstream is supervised);
no hiding it.

## Framework summary
- Stages 2a/2b: **HF transformers** (upstream uses HF)
- Stage 2c: **local vLLM** (upstream uses external API; user rule →
  local vLLM substitution)
- Stage 3: **skipped** (conditional on user go/no-go)
- Stage 4 (our label-free parse): CPU Python, no MLLM

## Datasets

All 4: MHClip_EN, MHClip_ZH, HateMM, ImpliHateVid.

Per-dataset prompts (summary from Part 1):
- MHClip_EN, MHClip_ZH: upstream prompt with Title field
- HateMM: upstream prompt without Title field
- ImpliHateVid: reuse HateMM prompt (no title in our annotations)

## Where to write

- **Audit + fixes** to existing:
  - `src/match_repro/run_match_agents.py` — add per-dataset prompt
    dispatch; verify model load + generation + OOM fallback
  - `src/match_repro/extract_frames.py` — verify it's not broken
- **New files**:
  - `src/match_repro/judgement_vllm.py` — stage 2c local vLLM
    substitution
  - `src/match_repro/finalize_labelfree.py` — stage 4 yes/no parser
  - `src/match_repro/README.md` — document the full pipeline, upstream
    refs, all deviations

## Syntax check (per `feedback_no_smoke_test.md`)

After implementing, run these checks in `SafetyContradiction` env
(NOT lavis_baselines — HF + vLLM live in SafetyContradiction):

```bash
conda activate SafetyContradiction
python -m py_compile src/match_repro/run_match_agents.py
python -m py_compile src/match_repro/judgement_vllm.py
python -m py_compile src/match_repro/finalize_labelfree.py
python -c "import sys; sys.path.insert(0,'src/match_repro'); import run_match_agents, judgement_vllm, finalize_labelfree; print('imports ok')"
```

**Do NOT run any smoke test.** Do NOT submit sbatch. Do NOT touch
`results/`.

## Deliverable

Reply to director with:
1. **Audit report for Part 1** — per-item PASS/FIX list referencing
   upstream file:line for each check
2. **File list** — new + modified
3. **Syntax-check log** — output of the py_compile and import check
4. **Fidelity notes** — any deviation flagged explicitly (there
   should be none beyond the documented API→vLLM and the label-free
   parser)

Director will review against the faithful-repro checklist and
submit the mass sbatch run directly.
