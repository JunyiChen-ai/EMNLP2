# Task brief — LoReHM REWORK (2026-04-15 ~17:00)

## Why this brief exists

User directive: drop the 2×4 grid composite workaround and pass **16
video frames directly as multi-image input** to LLaVA-v1.6-34b.
Upstream LoReHM uses a single image per LLaVA call; LLaVA-v1.6
natively supports multi-image via `LlavaNextProcessor` + a list of
PIL Images, and our existing grid composite is an unnecessary
degradation. Also: this baseline is now the **top-priority run** —
you get **2 GPUs** in bf16 (no quantization) to make it a fidelity
upgrade, not a downgrade.

Audit confirmed upstream regime is **zero-shot** single-image
LLaVA-v1.6-34b + RSA retrieval with label-majority-vote re-ask.
LLaVA never sees the retrieved examples — only their aggregate label
flips the prompt. Our regime matches upstream; only the frame-input
mechanism and the quantization are changing.

## Concrete edits required

**Target directory**: `src/lorehm_repro/` (rewrite in-place; do NOT
create a v2/v3 subdirectory)

### 1. DELETE the grid helper

- **Delete** `src/lorehm_repro/grid_composite.py` entirely. Also
  remove any `from .grid_composite import ...` references in
  `reproduce_lorehm.py` and `llava_runner.py`.

### 2. `src/lorehm_repro/llava_runner.py`

- Current implementation (around line 138-142) passes a single PIL
  image to `self.processor(images=image, text=prompt, ...)`. Change
  to accept a Python **list** of 16 PIL Images and pass
  `images=image_list` (list) to the processor.
- Ensure the generated prompt contains **16 `<image>` tokens** — one
  per frame — inserted at the video-reference location in
  BASIC_PROMPT / RSA_PROMPT before the `{text}` field. Use the
  LLaVA-Next chat template API
  (`processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, ..., {"type": "text", "text": prompt_body}]}], add_generation_prompt=True)`)
  for token correctness. That is: 16 `{"type": "image"}` entries
  followed by one `{"type": "text", "text": ...}`.
- Processor image pre-processing: force **single-tile 336×336** per
  frame (no multi-tile high-res expansion). Concretely:
  ```python
  processor.image_processor.image_grid_pinpoints = None  # or []
  # additionally if the processor still tries to tile:
  processor.image_processor.do_pad = False
  ```
  If the above doesn't disable tiling, explicitly resize every frame
  to `(336, 336)` with `PIL.Image.BICUBIC` BEFORE passing to the
  processor, and set `image_sizes` manually to `[(336, 336)] * 16`.
  Target visual-token count: 576 tokens/frame × 16 = 9216 tokens per
  call. Document the exact approach taken in the README.

### 3. Model load: 2-GPU bf16, no quantization

- Load `llava-hf/llava-v1.6-34b-hf` at **bf16**, NOT bnb 4-bit, via
  HF `LlavaNextForConditionalGeneration.from_pretrained`.
- Use **`device_map="auto"`** with `max_memory` set to allow
  ~75 GiB per GPU (leave headroom for activations):
  ```python
  max_memory = {0: "75GiB", 1: "75GiB", "cpu": "64GiB"}
  ```
- Set `torch_dtype=torch.bfloat16`, `low_cpu_mem_usage=True`,
  `trust_remote_code=False` (this model does not need it).
- You do NOT need to import bitsandbytes anymore. Remove any bnb
  quantization config from `llava_runner.py`.
- Drop any `attn_implementation="flash_attention_2"` references —
  use `sdpa` to stay compatible with the env.

### 4. `src/lorehm_repro/reproduce_lorehm.py`

- Load 16 frames per video from
  `<dataset_root>/frames_16/<vid>/frame_{i:03d}.jpg` for i in 0..15
  (all 16, no subsampling). Use `PIL.Image.open(...).convert("RGB")`.
- Build the `image_list` as `[PIL.Image, ...]` length 16.
- Pass the list + `text` to `LlavaRunner.run_model(image_list, text)`.
- Remove every reference to grid composites, frame hstack/vstack,
  and `to_grid_2x4`.
- Keep the control-flow structure from upstream `main.py:63-99`
  byte-for-byte (BASIC call → parse_answer → MIA branch (if
  enabled) → RSA re-ask if disagreement). RSA / `get_rsa_label` /
  `parse_answer` stay verbatim.
- Preserve output jsonl schema: `{video_id, pred, score,
  basic_predict, rsa_label (if computed), final_verdict_text}`.

### 5. `src/lorehm_repro/prompts.py`

- BASIC_PROMPT and RSA_PROMPT: rewrite the video-framing substring
  only. Specifically:
  - Replace any previous `"the 8 frames shown as a 2×4 grid"`
    phrasing with exactly: `"the 16 frames of the video"`.
  - Replace `"embedded in the image"` with
    `"over the 16 frames of the video"`.
  - Keep the `"meme" → "video"` substring substitution we already
    had.
  - All other upstream language preserved byte-for-byte from
    `external_repos/lorehm/utils/constants.py:1-5`.
- MEME_INSIGHTS dict and our 4 dataset hints stay unchanged.

### 6. Keep unchanged

- `src/lorehm_repro/retrieval.py` — jina-clip-v2 rel_sampl builder,
  byte-for-byte. Our own `rel_sampl` is already built in-house;
  upstream's precomputed jsonls don't exist for video benchmarks.
- `get_rsa_label(rel_sampl, k=5)` — upstream verbatim from
  `external_repos/lorehm/utils/utils.py:87-99`
- `parse_answer` — upstream verbatim
- Main loop control flow in `reproduce_lorehm.py`
- README structure (update it to list the new deviations — see below)

## Deviations list for updated README

1. Single image → **16 multi-image** (no grid, no composite). Uses
   LLaVA-Next's native multi-image processor path. Single-tile 336×336
   per frame to keep visual tokens manageable.
2. `text` field: meme OCR → video transcript (truncated 500 chars) —
   unchanged from prior LoReHM adaptation
3. Prompts: substring rewrite for `"16 frames of the video"` and
   `"meme" → "video"`; otherwise byte-for-byte from upstream
   `constants.py`
4. `rel_sampl` built by us via jinaai/jina-clip-v2 cosine on 8-frame
   pooled video features; upstream ships jsonls for FHM/HarM/MAMI
   only — unchanged
5. **Model loaded at bf16 (no quantization) sharded across 2 GPUs
   via `device_map="auto"`** — this is a FIDELITY UPGRADE versus
   our prior bnb-4bit plan (2-GPU sbatch allocation approved by
   director for this baseline's priority run)
6. 4 dataset-specific MEME_INSIGHTS blocks written by us — unchanged
7. Datasets: FHM/HarM/MAMI → our 4 video benchmarks

## Syntax check (no sbatch, no smoke)

```bash
conda activate SafetyContradiction
python -m py_compile src/lorehm_repro/*.py
python -c "import sys; sys.path.insert(0,'src/lorehm_repro'); import reproduce_lorehm, prompts, retrieval, llava_runner; print('imports ok')"
```

If `grid_composite` is still referenced anywhere, imports will fail
— the error will surface in this check.

## Deliverable report

Reply to the director with:
1. `git diff --stat src/lorehm_repro/` showing the rework
2. `ls src/lorehm_repro/` confirming `grid_composite.py` is deleted
3. Copy of the rewritten BASIC_PROMPT and RSA_PROMPT strings (so the
   director can verify the substring edits)
4. Syntax check log
5. Any deviation from this brief flagged explicitly (for example: if
   the processor tiling can't be disabled via attribute assignment
   and you had to manually pre-resize frames, say so)
6. **Do NOT run any smoke test.** Syntax check only.

## Diagnostic fallback protocol

If, at director-side sbatch submission time, the 2-GPU bf16 run OOMs
on 16 frames, the director will send a diagnostic brief asking you
to drop to 8 frames (still multi-image, still no grid, still bf16 on
2 GPUs). Do NOT preemptively code the fallback — only when the
director says so.
