# LoReHM reproduction (video-adapted, rework 2026-04-15)

LoReHM (EMNLP 2024, Huang et al., "Towards Low-Resource Harmful Meme Detection with LMM Agents") is a per-meme LLaVA-v1.6-34b detector with two optional augmentations: **MIA** (Meme Insight Augmentation, hand-written dataset hints appended to the prompt) and **RSA** (Relative Sample Augmentation, a precomputed top-k retrieval that flips the label when the model disagrees with a retrieval-based classifier). This repro adapts LoReHM to our 4 video benchmarks.

**Rework 2026-04-15**: the prior 2x4 grid composite path has been replaced with a **16-frame native multi-image** input via LLaVA-Next's `LlavaNextProcessor`. Frames are passed as a Python list of 16 PIL Images with 16 `<image>` chat-template entries at the video-reference position. Model loads at **bf16, no quantization**, sharded across **2 GPUs** via `device_map="auto"` (fidelity upgrade over the prior bnb-4bit plan; 2-GPU sbatch allocation approved).

## Upstream

- Repo: `external_repos/lorehm/`
- Paper: "Towards Low-Resource Harmful Meme Detection with LMM Agents", EMNLP 2024, Huang et al.
- Backbone: `liuhaotian/llava-v1.6-34b` via HF transformers + `utils/run_llava.py` `run_proxy`. HF-hub alias is `llava-hf/llava-v1.6-34b-hf` (same checkpoint).
- Datasets upstream: FHM, HarM, MAMI (single-image meme benchmarks).

## Upstream control flow (verified from `main.py:63-110`)

Per test item, upstream does **1 or 2** LLaVA calls on a **single image**:

1. **Basic call** (always): `BASIC_PROMPT.format(text)` with the meme image. Upstream text is the meme OCR/caption. Response is parsed via `parse_answer("Answer:")` for `harmful | harmless`.
2. **MIA** (optional): the hand-written insight block is appended to the query as `"\nNote:\n{insights}\n"` before step 1.
3. **RSA re-ask** (optional): if a precomputed `rel_sampl` tuple is available and `get_rsa_label(rel_sampl, k) != basic_predict`, the model is re-asked with `RSA_PROMPT.format(text, labels_str[rsa_label])`. The `rel_sampl` retrieval is a label-majority-vote over a top-50 pool — **the retrieved items are never shown as images to LLaVA**.

## Video-adapted pipeline (rework)

```
  frames_16/<vid>/frame_000.jpg .. frame_015.jpg   (16 frames)
      │
      ▼
  list[PIL.Image]  (each resized to 336x336 single-tile)
      │
      ├─► (optional, built once per dataset) jinaai/jina-clip-v2 pooled features
      │       │
      │       ▼
      │   rel_sampl.json  (top-50 by cosine on labeled train pool, persisted to disk)
      │
      ▼
  LLaVA-v1.6-34b (bf16, device_map="auto", 2 GPUs, sdpa attention)
     via LlavaNextProcessor.apply_chat_template with 16 {"type":"image"} entries
     ◄── BASIC_PROMPT.format(transcript[:500])  [+ MIA \nNote:\n{insights}\n]
      │
      ├─ basic_predict (harmful/harmless)
      │
      ├─ (if RSA && basic_predict != get_rsa_label(rel_sampl, k=5))
      │       re-ask  ◄── RSA_PROMPT.format(transcript[:500], labels_str[rsa_label])
      │
      ▼
  pred (final)  →  test_lorehm.jsonl
```

| Stage | File | Role |
|---|---|---|
| Dataset | `lorehm_video_dataset.py` | 16-frame loader from `frames_16/<vid>/frame_NNN.jpg` for NNN in 000..015, upstream-shaped item dicts with `(vid, frames, text, label, rel_sampl)` fields |
| Retrieval | `retrieval.py` | **jinaai/jina-clip-v2** `SentenceTransformer` loader (shared with MATCH stage 2b.5), mean-pooled features at `truncate_dim=512`, top-50 cosine search on labeled train pool, `get_rsa_label` byte-for-byte upstream, `rel_sampl.json` persistence |
| Prompts | `prompts.py` | Meme→video substring rewrites of `BASIC_PROMPT` / `RSA_PROMPT`; `LABELS_STR` verbatim; `MEME_INSIGHTS` dict with 4 per-dataset video-domain MIA hint blocks |
| LLaVA runner | `llava_runner.py` | `LlavaRunner` class over `llava-hf/llava-v1.6-34b-hf` via HF `LlavaNextForConditionalGeneration`. bf16 no quant, `device_map="auto"` with `max_memory={0:"75GiB",1:"75GiB","cpu":"64GiB"}`, `attn_implementation="sdpa"`. Accepts a list of 16 PIL Images, resizes each to 336x336, builds 16-image chat template, disables processor tiling via `image_grid_pinpoints=None` + `do_pad=False` + explicit `image_sizes=[(336,336)]*16` |
| Main | `reproduce_lorehm.py` | Upstream `main.py:63-99` control flow verbatim. 500-char transcript truncation, `rel_sampl.json` load-or-build, `_Args` upstream-compatible adapter, resume-skip, fsync per write |

## Upstream-verbatim bits

- **`parse_answer`** — byte-for-byte from `utils/utils.py:63-74`, including the `predict = 1` fallback on parse failure (returns "skip" as the answer string).
- **`get_rsa_label`** — byte-for-byte from `utils/utils.py:87-99`. The loop `count += 1 if example in harmful_examples`, `count -= 1 if example in harmless_examples`, `rsa_label = 1 if count >= 0 else 0` is unchanged.
- **`LABELS_STR = ["harmless", "harmful"]`** — upstream `main.py:61`, verbatim. This is what `RSA_PROMPT.format(text, labels_str[rsa_label])` plugs into its second slot.
- **`BASIC_PROMPT` / `RSA_PROMPT`** — upstream `constants.py:1-5` with the minimum necessary meme→video rewrites. The `Thought:` / `Answer:` format string, the `harmful / harmless` label vocabulary, "harmlessness and integrity of information on the Internet" phrasing, and the analysis instructions are preserved byte-for-byte.
- **`rsa_k = 5`** — upstream `main.py:20` default. Exposed as `--rsa-k` flag.
- **Top-50 retrieval pool** — upstream's shipped `rel_sampl` jsonls have ~50 entries per item, so `pool_topk=50` is the closest analog.
- **Control flow** — `reproduce_lorehm.score_one_video` mirrors `main.py:63-99` branch-by-branch: basic call → optional RSA re-ask only when `basic_predict != rsa_label`.

## Deviations (rework)

1. **Scope: meme → video** via LLaVA-Next's **native multi-image** path. 16 frames per video loaded from `frames_16/<vid>/frame_NNN.jpg` for NNN in 000..015 (same extraction used by MATCH / ALARM / Mod-HATE). Passed as a Python list of 16 PIL Images. The chat template is built with 16 `{"type":"image"}` entries followed by one `{"type":"text",...}` entry, so the prompt contains 16 `<image>` tokens at the video-reference position. Each frame is resized to 336×336 BICUBIC and the processor's image-tiling is disabled (`image_grid_pinpoints=None`, `do_pad=False`, explicit `image_sizes=[(336,336)]*16`) so the processor emits one 576-token tile per frame → 9216 visual tokens per call. No grid composite.
2. **Text field**: meme OCR → **video transcript, truncated to 500 chars** via `TRANSCRIPT_LIMIT = 500` in `reproduce_lorehm.py`.
3. **Substring prompt rewrites** of upstream `external_repos/lorehm/utils/constants.py:1-5`:
   - `"meme"` → `"video"`
   - `"the Text: \"{}\" embedded in the image"` → `"the transcript: \"{}\" over the 16 frames of the video"`
   
   Every other character — `"harmful or not"`, `"harmlessness and integrity of information on the Internet"`, `"classifier has labeled"` framing, `"If you disagree with the classifier's judgment, you must provide exceptionally thorough and persuasive reasons"`, the `Thought:` / `Answer:` / `harmful | harmless` format — is preserved byte-for-byte.
4. **`rel_sampl` computation**: upstream ships pre-built `rel_sampl` jsonls for FHM / HarM / MAMI. For our video benchmarks we build them ourselves via `retrieval.build_rel_sampl_for_dataset` using **`jinaai/jina-clip-v2`** (same `SentenceTransformer(..., trust_remote_code=True, truncate_dim=512, device='cuda')` loader used by our MATCH stage 2b.5 — cache shared). Per-video feature = mean of per-frame encode outputs → L2 renormalize → `[512]`. Per-test-video top-50 neighbors via cosine on the labeled train pool. `rel_sampl` 3-tuple `(example, harmful_example, harmless_example)` persisted to `results/lorehm/<dataset>/rel_sampl.json`. `get_rsa_label` is then applied byte-for-byte upstream.
5. **Model load: bf16, no quantization, 2-GPU sharded.** `LlavaNextForConditionalGeneration.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="auto", max_memory={0:"75GiB",1:"75GiB","cpu":"64GiB"}, low_cpu_mem_usage=True, attn_implementation="sdpa")`. No bitsandbytes, no AWQ — this is a **fidelity upgrade** over the prior bnb-4bit plan, made possible by the 2-GPU sbatch allocation approved for this baseline's priority run.
6. **MIA insights**: upstream `llava_harmC_insights` / `llava_FHM_insights` / `llava_MAMI_insights` (`constants.py:7-32`) are dataset-specific meme-domain hint strings. We wrote 4 new dataset-specific hint blocks matching upstream's tone and length, one per video benchmark (`MHCLIP_EN_INSIGHTS`, `MHCLIP_ZH_INSIGHTS`, `HATEMM_INSIGHTS`, `IMPLIHATEVID_INSIGHTS`). Keyed into `MEME_INSIGHTS["llava-v1.6-34b"][<dataset>]` in `prompts.py`.
7. **HF LLaVA-v1.6 path**: upstream uses `liuhaotian/llava-v1.6-34b` via `utils/run_llava.py`'s `run_proxy`. We use the HF canonical port `llava-hf/llava-v1.6-34b-hf` via `LlavaNextForConditionalGeneration` + `LlavaNextProcessor`. Same checkpoint weights, different loader.
8. **Generation kwargs**: `do_sample=False`, `num_beams=1`, `max_new_tokens=1024` — matches upstream `main.py:31-38` (`temperature=0, num_beams=1, max_new_tokens=1024`). Greedy decoding.
9. **Datasets**: FHM / HarM / MAMI → MHClip_EN / MHClip_ZH / HateMM / ImpliHateVid.

**NOT deviations** (preserved byte-for-byte upstream per v2 brief §182-188):
- `get_rsa_label` retrieval + majority vote (`utils/utils.py:87-99`)
- `parse_answer` harmful/harmless extraction (`utils/utils.py:63-74`)
- `LABELS_STR = ["harmless", "harmful"]` (`main.py:61`)
- `rsa_k = 5` default (`main.py:20`)
- Re-ask branch: `basic_predict != rsa_label` → `RSA_PROMPT` (`main.py:81-97`)
- Control flow in the main per-item loop (`main.py:63-99`)
- LLaVA forward via an upstream-shaped `args` object

## Output

```
results/lorehm/<dataset>/test_lorehm.jsonl
```

Schema per line:
```json
{
  "video_id": "...",
  "pred": 0 | 1,
  "label": 0 | 1,
  "basic_predict": 0 | 1,
  "rsa_label": 0 | 1 | null,
  "used_rsa_reask": true | false,
  "mia": true | false,
  "thought": "<model rationale>",
  "raw_response": "<full LLaVA output>"
}
```

Compatible with `src/naive_baseline/eval_generative_predictions.py --input ... --dataset ...`.

## CLI (SafetyContradiction env — HF + transformers + sentence-transformers/Jina-CLIP)

```bash
conda activate SafetyContradiction

# Step 1 — build rel_sampl.json per dataset (cheap, Jina-CLIP only).
# Reuses the already-cached jinaai/jina-clip-v2 from MATCH stage 2b.5.
python src/lorehm_repro/retrieval.py --all

# Step 2 — LLaVA-v1.6-34b mass run (2-GPU sbatch per dataset).
# Loads rel_sampl.json from disk; rebuilds via Jina-CLIP if missing.
# RSA + MIA enabled by default.
python src/lorehm_repro/reproduce_lorehm.py --all

# Ablations:
python src/lorehm_repro/reproduce_lorehm.py --all --no-rsa --no-mia   # basic only
python src/lorehm_repro/reproduce_lorehm.py --all --no-rsa            # MIA only
python src/lorehm_repro/reproduce_lorehm.py --all --no-mia            # RSA only

# Single-dataset:
python src/lorehm_repro/reproduce_lorehm.py --dataset MHClip_EN
```

## File list

Five `.py` files + this README under `src/lorehm_repro/` (grid_composite.py deleted in rework):

| File | Role |
|---|---|
| `lorehm_video_dataset.py` | 16-frame loader from `frames_16/<vid>/frame_NNN.jpg` for NNN in 000..015, per-video item dict with `(vid, frames, text, label, rel_sampl)` |
| `prompts.py` | `BASIC_PROMPT` / `RSA_PROMPT` (substring meme→video rewrites), `LABELS_STR`, `MEME_INSIGHTS` dict with 4 per-dataset hint blocks |
| `retrieval.py` | `jinaai/jina-clip-v2` SentenceTransformer loader, `build_rel_sampl` + `get_rsa_label` (upstream byte-for-byte), `save_rel_sampl` / `load_rel_sampl` disk persistence, `__main__` CLI |
| `llava_runner.py` | `LlavaRunner` class wrapping `llava-hf/llava-v1.6-34b-hf` (HF `LlavaNextForConditionalGeneration`). bf16 no quant, `device_map="auto"` 2-GPU, sdpa, 16-image multi-image chat template, single-tile 336x336 per frame |
| `reproduce_lorehm.py` | Upstream `main.py:63-99` control flow verbatim, `_Args` adapter, 500-char transcript truncation, `rel_sampl.json` load-or-build, resume-skip, fsync per write |

## Pre-flight (director action)

1. Pre-download `llava-hf/llava-v1.6-34b-hf` (~68 GB bf16). Fits sharded across 2×80GB A100 with `max_memory={0:"75GiB",1:"75GiB"}`.
2. **`jinaai/jina-clip-v2` — already cached** from MATCH stage 2b.5.
3. `frames_16/<vid>/frame_NNN.jpg` for all 4 datasets × both splits — extracted by `src/match_repro/extract_frames.py`.
