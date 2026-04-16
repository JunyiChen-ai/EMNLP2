# MATCH-HVD reproduction

**MATCH-HVD is a supervised multimodal hate-video detector.** The
authoritative baseline number comes from stage 3 (a small classifier
head trained on MFCC audio features + ViViT video features + BERT-
encoded judge summary text). Stages 2a/2b/2b.5/2c are **preprocessing
prerequisites** that produce the per-video judge summary consumed by
stage 3.

This directory currently contains stages 2a / 2b / 2b.5 / 2c + an
intermediate label-free peek parser. **Stage 3 (training + test
prediction) is covered by a separate brief** and will land in a
future commit.

## Upstream

- **Paper**: MATCH-HVD (anonymous submission)
- **Source**: https://anonymous.4open.science/r/MATCH-HVD
- **Local mirror**: `external_repos/match_hvd/`
- **Files audited**:
  - Stage 2a/2b runners: `preprocess/MATCH_HateMM_run.py`, `MATCH_Multi_en_run.py`, `MATCH_Multi_zh_run.py`
  - Stage 2b.5 CLIP matcher: `preprocess/time_unit.py`
  - Stage 2c judgement: `preprocess/judgement.py`
  - Stage 3 data loaders (for filename contract): `src/model/MATCH/data/{HateMM_MATCH.py,MHClipEN_MATCH.py,MHClipZH_MATCH.py}` — all three read `./data/<dataset>/judge.json`

## Pipeline

```
  frames_16/  ──► stage 2a   (Qwen2-VL-7B HF)        ──► hate.json
  frames_16/  ──► stage 2b   (Qwen2-VL-7B HF)        ──► nonhate.json
  frames_32/  ──► stage 2b.5 (Jina-CLIP-v2 HF)       ──► time_unit_hate.json
                                                        time_unit_nonhate.json
  frames_32/  ──► stage 2c   (Qwen2.5-VL-7B vLLM)    ──► judge.json
  judge.json  ──► stage 3    (MATCH classifier, supervised, SEPARATE BRIEF) ──► test_match.jsonl  [AUTHORITATIVE]
  judge.json  ──► stage 4-peek (label-free parse, DEBUG-ONLY) ──► test_match_peek.jsonl  [preview]
```

Stage 3 is the MATCH baseline's published number. Stage 4-peek is a
quick sanity check on the judge summary — useful while waiting for
stage 3 to land, never the headline result.

| Stage | File (this dir) | Framework | Model | Role | Status |
|---|---|---|---|---|---|
| extract | `extract_frames.py` | CPU Python + decord | — | 16 uniform frames → `<root>/frames_16/<vid>/` | ready |
| extract | `extract_frames_32.py` | CPU Python + decord | — | 32 uniform frames → `<root>/frames_32/<vid>/` | ready |
| 2a | `run_match_agents.py --agent hate` | HF transformers | Qwen/Qwen2-VL-7B-Instruct | Hate-evidence proposer | ready |
| 2b | `run_match_agents.py --agent nonhate` | HF transformers | Qwen/Qwen2-VL-7B-Instruct | Non-hate-evidence proposer | ready |
| 2b.5 | `time_unit_jina_clip.py` | HF transformers (jina-clip) | jinaai/jina-clip-v2 | Split each agent answer into ≤4 sub-claims; CLIP-match each sub-claim to best of 32 frames | ready |
| 2c | `judgement_vllm.py` | **local vLLM** | Qwen/Qwen2.5-VL-7B-Instruct | Judge synthesises both perspectives over CLIP-aligned 2×2 grids | ready |
| 3  | *(separate brief)* | HF transformers + librosa + ViViT | MATCH classifier + ViViT-b + MFCC | Supervised training + test prediction | **pending** |
| peek | `finalize_labelfree.py` | CPU Python | — | Debug-only label-free yes/no parse on the judge summary | ready |

## Stages 2a/2b — per-dataset dispatch

Stage 2a/2b audited against the 3 upstream runners. 6 FIX items
applied in `run_match_agents.py`:

| Dataset | qwen_run frames | err_run frames | Prompt lang | Title | OCR | Upstream |
|---|---|---|---|---|---|---|
| HateMM | `[0,2,4,6,8,10,12,14]` | `[0,4,8,12]` | EN | — | yes | `MATCH_HateMM_run.py:39-141, 206-290, 295-296` |
| MHClip_EN | `[0,1,2,3,5,7,9,11,13,14,15]` | `[0,2,4,6,8,10,12,14]` | EN | yes | loaded but not in prompt (upstream-exact) | `MATCH_Multi_en_run.py:46-145, 209-293, 298-299` |
| MHClip_ZH | `[0,2,4,6,8,10,12,14]` | `[0,4,8,12]` | ZH (中文) | yes | not loaded | `MATCH_Multi_zh_run.py:39-134, 196-276, 280-281` |
| ImpliHateVid | `[0,2,4,6,8,10,12,14]` | `[0,4,8,12]` | EN | — | — | *extension; reuses HateMM prompt* |

Upstream quirks we kept intact:
- `MATCH_Multi_en_run.py:42-44` reads `ocr.jsonl` with `MAX_TITLE_LENGTH` truncation but never interpolates OCR into the prompt (`:97-98`) — preserved as `max_ocr=0` for MHClip_EN.
- `MATCH_Multi_zh_run.py` never loads OCR and its `ifhate` non-hate token is `"没有"` (`:11`).
- `max_new_tokens=512` in all 3 upstream runners.

## Stage 2b.5 — Jina-CLIP frame alignment

File: `time_unit_jina_clip.py`. Byte-for-byte port of upstream `preprocess/time_unit.py` with three documented deviations.

| Upstream | Our file |
|---|---|
| `split_answer` (`:9-29`) | verbatim |
| `load_hate_data` (`:32-37`) | verbatim |
| `load_transcripts` (`:39-71`) | `build_transcript_dict` — body byte-for-byte (`:49-69`), wrapped around `data_utils.load_annotations(dataset)` instead of upstream's jsonl reader |
| `find_best_matching_frame` (`:72-101`) | verbatim except for deviation #3 |
| `clip_run` (`:103-126`) | verbatim, no `ocr_dir` parameter |
| `truncate_dim=512` (`:84`) | `TRUNCATE_DIM=512` |
| Output schema `{id, answer, frame, ocr, transcript}` | same |

**Three documented deviations:**

1. **`./MLLM/jina-clip-v2` → `jinaai/jina-clip-v2`.** Upstream loads from a modelscope cache; we load from HF Hub via `AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)`. Weights identical.
2. **Frames path `<root>/frames_32/<vid>/`** — produced by our `extract_frames_32.py`. Same layout as upstream `./data/<dataset>/frames_32/`.
3. **Per-frame OCR signal dropped.** Upstream's `find_best_matching_frame` computes `frame_text = ocr_data.get(str(idx), "") + " " + transcript_splits[idx]` using a per-frame OCR JSON. We don't have that extractor; we substitute `frame_text = transcript_splits[idx]` and leave `"ocr": ""` in each emitted entry. Downstream `judgement.py:58` already falls back to `"N/A"` on empty ocr, so rendering is preserved. No OCR extractor added as prerequisite.

### Pre-flight (director action)

First run on a compute node will pull `jinaai/jina-clip-v2` from HF Hub (~1.2 GB including trust_remote_code .py + weights). Compute nodes are air-gapped — pre-warm on login node before the real sbatch:

```bash
conda activate SafetyContradiction
export HUGGING_FACE_HUB_TOKEN=hf_...
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('jinaai/jina-clip-v2', \
    allow_patterns=['*.json','*.txt','*.bin','*.safetensors','tokenizer*','*.py'])"
```

Engineer does NOT run this; director handles pre-flight per `feedback_no_smoke_test.md`.

## Stage 2c — local vLLM judgement

File: `judgement_vllm.py`. Upstream `preprocess/judgement.py:17-19, 108-112` calls SiliconFlow:

```python
client = openai.OpenAI(api_key=api_key, base_url=base_url)
response = client.chat.completions.create(
    model="Pro/Qwen/Qwen2.5-VL-7B-Instruct",
    messages=api_messages,
    max_tokens=4096,
)
```

Our substitution (`feedback_api_to_vllm.md`, standing rule):

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-VL-7B-Instruct", ...)
outputs = llm.chat(messages, sampling_params=SamplingParams(max_tokens=4096, temperature=0.0))
```

Everything else is byte-for-byte upstream:
- **`format_entries_upstream`** mirrors `judgement.py:48-77` — per-entry Reason/Frame/OCR/Transcript block, 2×2 grid from `frame_uris[:4]`, pad missing frames with black 128×128, `cv2.imread` → `cv2.resize((128,128))` → `hstack`/`vstack` → `cv2.cvtColor(BGR2RGB)` → PNG → base64. Frames are loaded from `<root>/frames_32/<vid>/<frame_name>` using the `frame` field emitted by stage 2b.5.
- **`build_prompt`** copies `judgement.py:81-96` character-for-character.
- **`max_tokens=4096`** matches `judgement.py:111`.

**Output filename**: `judge.json`. This matches upstream stage 3 loader expectations exactly — all three of `src/model/MATCH/data/HateMM_MATCH.py:19`, `MHClipEN_MATCH.py:20`, `MHClipZH_MATCH.py:20` read `./data/<dataset>/judge.json`, so stage 3 (separate brief) will find our output without any renaming step. The schema `[{"id": vid, "summary": "..."}]` is identical to upstream.

## Stage 3 — MATCH supervised classifier (SEPARATE BRIEF, pending)

Not yet implemented in this directory. Per the 2026-04-15 scope correction, stage 3 is the authoritative MATCH baseline number. A separate brief will cover:
- MFCC audio feature extractor (librosa) → `fea_audio_mfcc.pt`
- ViViT 32-frame features via `google/vivit-b-16x2-kinetics400` (uses the `frames_32/` directories that stage 2b.5 already depends on) → `fea_frames_32_google-vivit-b.pt`
- Per-dataset dataset modules mirroring `src/model/MATCH/data/{HateMM_MATCH.py,MHClipEN_MATCH.py,MHClipZH_MATCH.py}`
- MATCH classifier architecture from `src/model/MATCH/MATCH.py`
- Training loop + Hydra config (50 epochs, AdamW lr=5e-4, fea_dim=128, batch_size=128 per upstream `src/config/*.yaml`)
- Test-set prediction jsonl in our standard schema consumable by `eval_generative_predictions.eval_one()`

Stage 3 output will be the primary MATCH baseline number for the paper.

## Stage 4-peek — debug-only label-free parser

File: `finalize_labelfree.py`. **Not the primary deliverable.** This is a debug / preview tool that parses the stage 2c judge summary text for a yes/no verdict, so we can spot-check whether the vLLM-substituted judge is producing sensible rationales before committing to the stage 3 training run.

Input : `judge.json`
Output: `test_match_peek.jsonl` (renamed from the earlier `test_match.jsonl` to avoid confusion with the stage 3 output that will land later)

Parser rules unchanged from the earlier deliverable: phrase-priority (negatives before positives to avoid substring collisions) + word-count fallback + conservative default. No label data is used to tune the phrase list.

## Output layout

```
results/match_qwen2vl_7b/
    MHClip_EN/
        hate.json                 # stage 2a
        nonhate.json              # stage 2b
        time_unit_hate.json       # stage 2b.5
        time_unit_nonhate.json    # stage 2b.5
        judge.json                # stage 2c  (matches upstream stage 3 loader)
        test_match.jsonl          # stage 3   (AUTHORITATIVE — pending separate brief)
        test_match_peek.jsonl     # stage 4-peek (debug-only preview)
    MHClip_ZH/ …
    HateMM/ …
    ImpliHateVid/ …
```

## CLI (all in `SafetyContradiction` conda env)

```bash
conda activate SafetyContradiction

# Pre-flight (director, once):
python src/match_repro/extract_frames_32.py --all --split test
# + HF Hub pre-warm for jinaai/jina-clip-v2 and Qwen/Qwen2.5-VL-7B-Instruct

# Preprocessing pipeline:
python src/match_repro/run_match_agents.py --all --agent hate    --split test
python src/match_repro/run_match_agents.py --all --agent nonhate --split test
python src/match_repro/time_unit_jina_clip.py --all
python src/match_repro/judgement_vllm.py --all

# Authoritative (stage 3, pending separate brief):
# python src/match_repro/train_match_stage3.py --all --split test   <-- not yet implemented

# Optional debug peek (not the headline number):
python src/match_repro/finalize_labelfree.py --all
```

## Deviations summary

1. **API → local vLLM** at stage 2c. Per standing rule `feedback_api_to_vllm.md`. Prompt, 2×2 grid, and `max_tokens=4096` unchanged from upstream.
2. **`./MLLM/jina-clip-v2` → `jinaai/jina-clip-v2`** at stage 2b.5. Identical weights, HF Hub loader instead of modelscope path.
3. **Per-frame OCR signal dropped** at stage 2b.5. `frame_text = transcript_splits[idx]` only; upstream used `ocr + transcript_split`. No per-frame OCR extractor in scope.
4. **`finalize_labelfree.py` demoted to debug-only preview** per the 2026-04-15 scope correction (MATCH is supervised). Output renamed to `test_match_peek.jsonl`.
5. **ImpliHateVid extension**: reuses HateMM English prompt (no upstream runner exists).

No other deviations.
