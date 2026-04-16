# Task brief — Pro-Cap V3 (supervised, multi-image captioner) 2026-04-15 ~17:00

## Why this brief exists

Audit of `external_repos/procap/codes/scr/{pbm.py,train.py,dataset.py,config.py}`
shows upstream Pro-Cap is a **supervised text classifier**:
RoBERTa-large PromptHateModel (PBM) trained with AdamW lr=1e-5 +
BCELoss + in-context few-shot demos on 8 VQA captions + 1 general
caption. Captions are **pre-computed offline** (pickle file) and
loaded via `load_pkl(cap_path)` during training (`dataset.py:114`).

Our current `src/procap_repro/reproduce_procap_lavis.py` uses LAVIS
BLIP-2 at a single middle frame + a degenerate 9th-probe label-free
yes/no hack. This is **wrong** both on (a) the learning regime
(upstream is supervised, we do zero-shot) and (b) the frame input
(upstream's BLIP-2 captioner is single-image, so for video we need a
multi-image-capable model that can process 16 frames as a video).

User directive: swap the captioner to a multi-image video model and
add the full supervised training loop. User confirmed the captioner:
**Qwen2-VL-7B-Instruct** (reuses MATCH stage 2's already-cached HF
model; `qwen_vl_utils` has a native video path).

## Target directory

**New directory**: `src/procap_v3_repro/`

Leave `src/procap_repro/` frozen as archive. Do NOT edit it. Do NOT
delete the old LAVIS results.

## Part 1 — Multi-image video caption generation

**File**: `src/procap_v3_repro/generate_captions_qwen2vl.py`

- **Captioner**: `Qwen/Qwen2-VL-7B-Instruct` via **vLLM**
  - Model already cached locally at
    `/data/jehc223/home/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/`
  - Reuse the vLLM loader pattern from
    `src/match_repro/judgement_vllm.py` (same model family) —
    `SamplingParams(max_tokens=256, temperature=0.0)` or whatever
    matches the brief below
- **Input**: 16 frames per video from
  `<dataset_root>/frames_16/<vid>/frame_{i:03d}.jpg` (i in 0..15),
  passed as a single multi-image user message via the Qwen2-VL chat
  template:
  ```python
  content = [{"type": "image", "image": f"file://{abspath}"} for path in frame_paths]
  content.append({"type": "text", "text": probe_text})
  messages = [{"role": "user", "content": content}]
  ```
  Then build the prompt via `processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)` and pass
  `prompt=prompt_text, multi_modal_data={"image": [PIL_frames_list]}`
  to vLLM's `llm.generate(...)`. Model sees 16 frames at once per
  probe call.
- **8 VQA probes — byte-for-byte from upstream notebook
  `external_repos/procap/codes/Pro-Cap-Generation.ipynb`**:
  ```python
  PROCAP_PROBES = [
      ("race",         "what is the race of the person in the video?"),
      ("gender",       "what is the gender of the person in the video?"),
      ("animal",       "is there an animal in the video?"),
      ("person",       "is there a person in the video?"),
      ("country",      "which country does the person in the video come from?"),
      ("what_animal",  "what animal is in the video?"),
      ("disabled",     "are there disabled people in the video?"),
      ("religion",     "what is the religion of the person in the video?"),
  ]
  ```
  Only substring change from upstream: `"in the image"` →
  `"in the video"`. All other wording preserved.
- **Per video**: 8 multi-image calls (one per probe). Store the
  Qwen2-VL response text per probe in a dict. Also capture the
  general caption (upstream calls this the "gen" caption) by asking
  a 9th probe `"describe the video in one sentence."` (this is the
  equivalent of the upstream notebook's `"a photo of ..."` general
  caption cell).
- **Output**: Per-dataset, per-split pickle file:
  - Path: `results/procap_v3/<dataset>/captions_{split}.pkl`
  - Schema: `{vid: {"race": str, "gender": str, "animal": str,
    "person": str, "country": str, "what_animal": str,
    "disabled": str, "religion": str, "gen": str}}`
- **Both splits**: the script must support `--split train` and
  `--split test`. Both must run (train is the prerequisite for
  supervised training; test is the prerequisite for test-set
  prediction; Mod-HATE also consumes the train pkl as support set).
- **Integrity discipline**: fsync per video, resume support (skip
  vids already in the output dict), periodic printing of progress.
- **SKIP_VIDEOS**: respect `data_utils.SKIP_VIDEOS[dataset]` when
  iterating the split.

## Part 2 — Supervised RoBERTa-large PromptHateModel training

**Files**:
- `src/procap_v3_repro/pbm.py` — port of
  `external_repos/procap/codes/scr/pbm.py:5-55` (class
  `PromptHateModel`, loads `RobertaForMaskedLM.from_pretrained("roberta-large")`)
- `src/procap_v3_repro/dataset_procap.py` — port of
  `external_repos/procap/codes/scr/dataset.py` caption loader +
  prompt builder (in-context demo sampling, `<mask>` token
  placement)
- `src/procap_v3_repro/train_procap.py` — port of
  `external_repos/procap/codes/scr/train.py:48-170` training loop
  (AdamW lr=1e-5 wd=0.01, BCELoss, in-context demos via
  `select_context`, `USE_DEMO=True`, `NUM_SAMPLE=1`)

**Key upstream details**:
- Text encoder: `roberta-large` (upstream config.py default)
- Optimizer: `AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)`
  (upstream train.py:103-107)
- Loss: BCELoss via `bce_for_loss()` helper (upstream train.py:13-16)
- Demo sampling: `select_context` picks `NUM_SAMPLE=1` random
  in-context example from the training pool per batch item
- Input sequence: upstream dataset.py builds a prompt of the form
  `"[CLS] <demo_caption> this is a <mask> meme. [SEP] <target_caption> this is a <mask> meme. [SEP]"`
  and trains the `<mask>` token to be one of `{good, bad, love, hate, ...}` — check upstream dataset.py exact format and replicate. Use `"video"` in place of `"meme"`.
- Target label: masked LM target is `"good"` token for benign and
  `"bad"` token for hateful (read upstream to confirm exact token
  IDs) → BCE on logit diff

**Adaptations required**:
- Load captions from our pkl: `results/procap_v3/<dataset>/captions_{split}.pkl`
- Labels: `data_utils.load_annotations(dataset)[vid]["label"]` + `eval_generative_predictions.collapse_label()` on train split; binary {0, 1}
- Validation split: deterministic 80/20 stratified split of
  `splits/train_clean.csv` (seed=2025), use for early stopping
  (patience 8)
- No Hydra, no wandb. Plain argparse: `--dataset`,
  `--captions-dir`, `--text-encoder`, `--num-epoch`, `--batch-size`,
  `--lr`, `--weight-decay`, `--seed`, `--patience`, `--output-dir`
- Save best model state to
  `results/procap_v3/<dataset>/best_pbm.pth`
- Reload best model and run on test split → per-video predictions
  to `results/procap_v3/<dataset>/test_procap.jsonl` with schema
  `{video_id, pred, score}` where score is the sigmoid of the
  `P(mask="bad") - P(mask="good")` logit difference (or whatever
  upstream uses — verify in upstream train.py's inference path)

## Part 3 — Update Mod-HATE caption adapter

**File**: `src/mod_hate_repro/video_caption_adapter.py`

- Add a new `prefer` option: `"v3_qwen2vl"` pointing to
  `results/procap_v3/<dataset>/captions_{split}.pkl`
- Make `"v3_qwen2vl"` the new **default** for `load_procap_captions`.
  The previous 1-frame / 8-frame BLIP-2 paths remain as explicit
  fallbacks but are not default.
- The pkl schema is the same as Pro-Cap V3 (8 probes + gen), so the
  existing `build_examples` function should pick them up without
  field renaming — verify by checking field names read by the
  function. If the old fallback used different field names (like
  `per_frame_captions`), add a minimal translation shim inside
  `load_procap_captions` that surfaces the new schema under the old
  keys the downstream function expects.

## Syntax check (no sbatch, no smoke)

```bash
conda activate SafetyContradiction
python -m py_compile src/procap_v3_repro/*.py
python -m py_compile src/mod_hate_repro/video_caption_adapter.py
python -c "import sys; sys.path.insert(0,'src/procap_v3_repro'); import generate_captions_qwen2vl, pbm, dataset_procap, train_procap; print('procap v3 imports ok')"
python -c "import sys; sys.path.insert(0,'src/mod_hate_repro'); import video_caption_adapter; print('mod_hate adapter ok')"
```

## Deliverable report

Reply to the director with:
1. File list under `src/procap_v3_repro/`
2. Diff or summary of changes to `src/mod_hate_repro/video_caption_adapter.py`
3. Copy of the 8 PROCAP_PROBES list (to verify the `"in the image"` → `"in the video"` substring edit)
4. Copy of the RoBERTa prompt template string used in `dataset_procap.py`
5. Copy of the vLLM generation kwargs used in `generate_captions_qwen2vl.py`
6. Syntax check log
7. **Do NOT run any smoke test.** Syntax only.

## Ordering

This is the **second** brief. Do LoReHM rework first (brief
`docs/baseline_briefs/lorehm_rework.md`), report back, then switch
to this one.
