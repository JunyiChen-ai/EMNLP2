# Task brief — Mod-HATE (adapted to hateful video detection)

## Why this brief exists

Mod-HATE (WWW 2024, Cao et al.) is a few-shot supervised
harmful-meme detector. LLaMA-7B text-only processes
Pro-Cap-generated **captions** + meme text. It never sees pixels.
Our 4 benchmarks are video. Per user rule 2026-04-15
(`feedback_meme_to_video_8frames.md`):

- Sample **8 frames per video** uniformly
- **Text-only adaptation** — run Pro-Cap on each of the 8 frames
  per video, concatenate the 8 resulting captions into one
  multi-frame caption string; LLaMA sees that concatenated
  caption
- **One LLaMA call per video** — no per-frame aggregation
- Rewrite the caption-level prompt from meme → video
- **Few-shot supervision accepted** as a reference baseline (same
  row as MATCH stage 3 — user approved supervised reference
  methods 2026-04-15)

## Upstream

- Repo: `external_repos/mod_hate/`
- Paper: "Modularized Networks for Few-shot Hateful Meme
  Detection" — WWW 2024
- Backbone: `yahma/llama-7b-hf` via HF transformers + PEFT
- Pre-trained LoRA modules in
  `external_repos/mod_hate/LoRA_modules/`:
  - `hate-speech/` (DT+WZ+Gab)
  - `meme-captions/` (MemeCap)
  - `hate-exp/` (HatReD)
- Pipeline:
  1. `lora_learning.py` — LoRAHub-style Nevergrad black-box
     optimization over the 3 module weights, using a K-shot
     labeled support set (K=4 and K=8)
  2. `new_lora.sh` — applies the weighted-average LoRAs on the
     test set
  3. Input format: Pro-Cap 8-answer caption + meme OCR text → LLaMA
     generation with "yes" / "no" head

## Prerequisite — 8-frame Pro-Cap LAVIS variant (new)

The existing Pro-Cap LAVIS job (8431) runs with **1 middle frame**
per video. Mod-HATE needs **8 captions per video**. Create a
separate 8-frame variant of the Pro-Cap LAVIS script:

- New file: `src/procap_repro/reproduce_procap_lavis_8frame.py`
  (new, parallel to the 1-frame `reproduce_procap_lavis.py`)
- Reads 8 frames per video from `frames_16/<vid>/` indices
  `[0,2,4,6,8,10,12,14]`
- For each frame, runs the 8 VQA probes via the same upstream
  `generate_prompt_result` helper
- Output schema: `{"video_id": vid, "per_frame_captions":
  [caption_frame_0, ..., caption_frame_14]}` — 8 strings per
  video, one per frame, each a full 8-probe concatenation
- Output path:
  `results/procap_lavis_blip2_flan_t5_xl_8frame/<dataset>/test_procap.jsonl`
- Reuses all Pro-Cap LAVIS machinery (LAVIS `load_model_and_preprocess`,
  `generate_prompt_result` helper with `length_penalty=3.0`,
  the 8 probes verbatim)
- The 1-frame version (8431 currently running) is unchanged — this
  is an additive file

Do NOT modify `reproduce_procap_lavis.py`.

## Mod-HATE pipeline — video-adapted

1. **Feature builder (our adaptation)**
   - Read 8-frame Pro-Cap captions from
     `results/procap_lavis_blip2_flan_t5_xl_8frame/<dataset>/test_procap.jsonl`
   - Concatenate the 8 per-frame captions into a single "video
     caption" string with frame markers: `"Frame 1: {cap_0}.
     Frame 2: {cap_1}. ... Frame 8: {cap_14}."` — document
     verbatim in the README
   - Pair with our annotation's `transcript` field as the "text"
     input (upstream uses meme OCR; we substitute with transcript
     since our videos have transcripts, not OCR)
2. **LoRA composition (lora_learning.py, verbatim except input
   adapter)**
   - Support set = K=4 or K=8 **labeled videos** from our train
     split per dataset. Same few-shot setting as upstream's 4/8-shot
     protocol.
   - Nevergrad black-box optimization over the 3 LoRA module
     weights, upstream-verbatim
   - Run twice per dataset: once with K=4, once with K=8
3. **Inference (new_lora.sh equivalent, rewritten as Python)**
   - Apply the composed LoRA weights on the test set
   - Input: `video caption + transcript` for each test video
   - Prompt template: upstream's, with meme → video rewrite.
     Upstream template is in `prompts/` or similar — copy verbatim
     then rewrite only meme language
   - Output: `"yes"` / `"no"` → pred = 1 / 0

## Prompt-rewrite rules

- "meme" → "video"
- "this meme contains" → "this video contains"
- "the image" → "the frames sampled from the video"
- "meme caption" → "video caption"
- "caption" (when referring to the Pro-Cap output) → "video
  caption" or `"8-frame caption"`

Analysis instructions, few-shot template, LoRA composition logic,
generation kwargs, and "yes/no" label vocabulary — all verbatim.

## Frames & data flow

- 8 Pro-Cap captions per video (via the new 8-frame Pro-Cap
  variant described above)
- Train split used as the K-shot labeled support set (K=4, K=8)
- Test split used for prediction
- Output:
  - `results/mod_hate/<dataset>/test_mod_hate_4shot.jsonl`
  - `results/mod_hate/<dataset>/test_mod_hate_8shot.jsonl`
  - Schema: `{"video_id": vid, "pred": 0|1, "response": "yes|no"}`
- Eval: `eval_generative_predictions.eval_one()` compatible

## Framework

HF transformers + PEFT, matching upstream. No vLLM substitution
(no cloud API in upstream; no oversized backbone — LLaMA-7B fits
easily).

## Where to write

- `src/mod_hate_repro/`
  - `reproduce_mod_hate.py` — main driver
  - `lora_compose.py` — Nevergrad LoRA composition wrapper
  - `video_caption_adapter.py` — 8-frame Pro-Cap caption concat
  - `README.md` — upstream commit + deviations
- `src/procap_repro/reproduce_procap_lavis_8frame.py` — 8-frame
  Pro-Cap LAVIS variant (preferred upstream of Mod-HATE)

## Deviations list (for README)

1. Scope: meme → video via 8-frame Pro-Cap concatenation (text-
   level adaptation, no pixels)
2. Caption source: 1-frame Pro-Cap (FHM/MAMI) → 8-frame Pro-Cap
   (our videos), concatenated with explicit frame markers
3. Text field: meme OCR → video transcript (closest analog in our
   annotations)
4. Support set: 4 / 8 labeled videos per dataset (upstream's
   4-shot / 8-shot protocol; few-shot supervised reference in-bounds
   under 2026-04-15 user directive)
5. Datasets: FHM/MAMI → our 4 video benchmarks; LoRA modules
   (hate-speech, meme-captions, hate-exp) used verbatim
6. Prompts: meme → video language rewrite
7. Trials: K=4 and K=8 both reported (upstream protocol)

## Syntax check

```bash
conda activate SafetyContradiction
python -m py_compile src/procap_repro/reproduce_procap_lavis_8frame.py
python -m py_compile src/mod_hate_repro/*.py
python -c "import sys; sys.path.insert(0,'src/mod_hate_repro'); import reproduce_mod_hate, lora_compose, video_caption_adapter; print('imports ok')"
```

**No sbatch. No smoke.** Report back with file list, audit notes,
syntax log, deviation list.

## Ordering for director-side submission

1. 8-frame Pro-Cap LAVIS real run (director submits sbatch once
   engineer delivers the 8-frame variant script and current 1-frame
   8431 is done)
2. Mod-HATE 4-shot training + inference per dataset
3. Mod-HATE 8-shot training + inference per dataset
4. Final jsonl → director runs results walker
