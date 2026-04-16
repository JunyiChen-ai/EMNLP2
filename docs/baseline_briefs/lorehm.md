# Task brief — LoReHM (adapted to hateful video detection) — v2, CORRECTED

## Why v2

v1 of this brief (2026-04-15 05:18) described a "48-image
in-context retrieval" RSA mechanism that doesn't exist in upstream.
Engineer correctly flagged the mismatch after reading
`external_repos/lorehm/main.py:63-99`. **Upstream's RSA is a
pre-computed retrieval + label-majority-vote + re-ask pattern**
where LLaVA never sees retrieved examples. This v2 brief captures
the actually-faithful reproduction path (Option 1 from the
engineer's analysis).

## Upstream pipeline (verified)

- Repo: `external_repos/lorehm/`
- Paper: "Towards Low-Resource Harmful Meme Detection with LMM
  Agents" — EMNLP 2024
- Backbone: `llava-v1.6-34b` via HF transformers + `run_llava.py`
- Datasets: FHM, HarM, MAMI (single-image memes)

**Per test item, upstream LoReHM runs at most 2 LLaVA calls**, each
with a **single image** input:

1. **Basic call**: `BASIC_PROMPT.format(text)` — `text` is the meme
   caption/OCR. LLaVA sees 1 image. `parse_answer` extracts
   `harmful | harmless` from the `Answer:` line → `basic_predict`.
2. **MIA (optional)**: prepends a `Note:\n{insights}\n` block (hand-
   written dataset-specific hints in
   `utils/constants.py:MEME_INSIGHTS`) to the prompt before step 1.
3. **RSA (optional, re-ask when disagreement)**: uses a **pre-
   computed `rel_sampl` jsonl** (shipped with dataset). Each test
   item has `{example: [...], harmful_example: [...], harmless_example: [...]}`.
   `get_rsa_label(rel_sampl, k)` (upstream `utils/utils.py:87-99`):
   ```python
   examples = rel_sampl[0][:k]  # top-k
   count = sum(1 for e in examples if e in harmful_examples) - sum(1 for e in examples if e in harmless_examples)
   rsa_label = 1 if count >= 0 else 0
   ```
   If `rsa_label != basic_predict`, re-ask with
   `RSA_PROMPT.format(text, "harmful" if rsa_label == 1 else "harmless")`.
   LLaVA still sees **one image**. The retrieved examples are
   **never passed to LLaVA** — only their aggregate label flips the
   prompt.

Upstream default `rsa_k=5`. The "50-shot" phrase refers to the size
of the `rel_sampl` retrieval pool, not anything LLaVA sees.

## Our adaptation — 4 localized deviations from upstream

**Adaptation 1: single image → 2×4 grid composite of 8 frames.**
Upstream always passes 1 image per call. Our videos have 8 frames.
We build a 2×4 grid composite via cv2 (top row = hstack of frames
0,1,2,3; bottom row = hstack of frames 4,5,6,7; vstack top+bottom;
BGR→RGB; save as PNG; pass to `run_llava.py`). This extends
MATCH-HVD's upstream 2×2 grid for 4 frames (`judgement.py:62-75`)
to 2×4 for 8 frames — our adaptation, not a direct LoReHM precedent,
but consistent with the in-repo precedent for multi-frame single-
image reduction. Document explicitly in the README.

Frame indices: 8 frames via np.linspace from
`<root>/frames_8/` (if present) or uniform indices
`[0,2,4,6,8,10,12,14]` from `frames_16/<vid>/`. Reuse the existing
frames_16 extraction. For the composite, resize each frame to
336×336 (LLaVA-v1.6's `image_size`), giving a 1344×672 composite.

**Adaptation 2: text field = video transcript, not meme OCR.**
Upstream's `text` slot in `BASIC_PROMPT.format(text)` is the meme
caption/OCR. Our analog is the video transcript from
`data_utils.load_annotations(dataset)[vid]["transcript"]`, truncated
to 500 chars (to match the roughly-meme-text length; document the
choice).

**Adaptation 3: prompt meme→video rewrite, minimal.** `BASIC_PROMPT`
and `RSA_PROMPT` are copied verbatim from
`external_repos/lorehm/utils/constants.py:1-19` with only these
substring substitutions:
- `"meme"` → `"video"`
- `"the Text: \"{}\" embedded in the image"` → `"the transcript: \"{}\" from the video's audio and the 8 frames shown as a 2×4 grid"`

No other edits. The "harmful/harmless" vocabulary, the "classifier has
labeled" framing, the "please review/agree or disagree" instructions
— all preserved byte-for-byte.

**Adaptation 4: our own `rel_sampl` computation.** Upstream ships
precomputed `rel_sampl` jsonls per dataset. We don't have those for
video benchmarks, so build them at our end:

- For each (test_video, split=test), compute a retrieval list
  against the train_clean pool, ranked by cosine similarity over
  an 8-frame pooled video feature
- **Feature source**: reuse our already-cached `jinaai/jina-clip-v2`
  (was downloaded for MATCH stage 2b.5) — compute per-video
  embedding as the mean of 8 per-frame `encode_image` outputs with
  `truncate_dim=512`. Zero extra downloads.
- For each test video, return `(top_50_vids, harmful_vids_in_pool,
  harmless_vids_in_pool)` — matching upstream's 3-tuple `rel_sampl`
  shape (`[examples, harmful_examples, harmless_examples]`)
- `harmful_vids_in_pool` = all train videos with label=1;
  `harmless_vids_in_pool` = all train videos with label=0. Ground-
  truth train labels are in-bounds: few-shot retrieval pool
  construction is allowed for a few-shot supervised baseline
  (precedent: MATCH stage 3, Mod-HATE K=4/8 support set)
- Save precomputed `rel_sampl` per dataset to
  `results/lorehm/<dataset>/rel_sampl.json`
- `get_rsa_label` is then upstream byte-for-byte

## Model — llava-v1.6-34b quantization (hard-block)

34B bf16 ≈ 68 GB weights; 80 GB A100 at gpu_util 0.92 = ~73.6 GB
usable. Weights barely fit; activations will OOM on LLaVA-v1.6's
image processing path.

**Primary path**: community AWQ quant
(`liuhaotian/llava-v1.6-34b` → check HF for
`<org>/llava-v1.6-34b-awq` or equivalent). If unavailable, fall
back to **bitsandbytes 4-bit**: upstream `run_llava.py` → `builder.py`
supports `load_in_4bit=True` via `BitsAndBytesConfig`.

Document the quant choice in the README as hard-block VRAM
substitution, citing the MARS 32B-AWQ precedent.

## MIA hints — rewrite for 4 video datasets

Upstream `utils/constants.py` ships
`llava_FHM_insights`, `llava_MAMI_insights`, `llava_harmC_insights`
as dataset-specific hand-written hint strings. Pattern: 5-10
sentences, 2-3 sentences each, describing the dataset's content
domain and common hate patterns.

Write new 4 hint blocks matching the same tone and length, one per
dataset:
- **MHClip_EN**: short-form YouTube / TikTok English video clips;
  visual-textual memes over racial, gender, religious targets;
  often use irony and dog whistles
- **MHClip_ZH**: similar format for Chinese-language clips; may
  include anti-ethnic-minority, anti-Japanese/Korean, anti-feminist
  tropes
- **HateMM**: medium-length YouTube videos; explicit hate speech
  against racial and religious groups; includes dehumanizing
  language and symbols
- **ImpliHateVid**: implicit hate videos; subtle dog whistles,
  deniability, coded language; may require reasoning about context

Save as `MEME_INSIGHTS["llava-v1.6-34b"][<dataset>]` in your
`prompts.py`. Document the synthesis process in the README (we
wrote these, they're not from upstream).

## Where to write

- `src/lorehm_repro/`
  - `reproduce_lorehm.py` — main driver (ports upstream `main.py`
    control flow verbatim where possible)
  - `prompts.py` — BASIC_PROMPT, RSA_PROMPT (meme→video rewrite) +
    MEME_INSIGHTS dict (our 4 dataset hints)
  - `retrieval.py` — per-dataset `rel_sampl.json` builder using
    jinaai/jina-clip-v2
  - `grid_composite.py` — 8-frame 2×4 grid builder (cv2 + PIL)
  - `llava_runner.py` — thin wrapper around upstream
    `utils/run_llava.py`'s `run_proxy` to accept our composite PNG
    path + our video `text` field
  - `README.md` — all 4 deviations + quant choice + MIA hint origin
    + rel_sampl build process
  - `lorehm_video_dataset.py` — test/train iterator

## Deviations list (for README)

1. Single image input → 2×4 grid composite of 8 frames (extends
   MATCH-HVD's 2×2 precedent; documented)
2. `text` field: meme OCR → video transcript (truncated 500 chars)
3. Prompts: meme→video substring rewrite (BASIC_PROMPT, RSA_PROMPT
   otherwise byte-for-byte from upstream constants.py)
4. `rel_sampl` built at our end via jinaai/jina-clip-v2 cosine on
   8-frame pooled video features; upstream's pre-computed jsonls
   are dataset-specific and don't exist for our video benchmarks
5. Model: 34B bf16 → 34B AWQ (or bnb 4-bit); hard-block VRAM
6. MIA hints: 4 new dataset-specific blocks matching upstream tone,
   written by us for our video datasets (FHM/HarM/MAMI hints not
   usable for video content)
7. Datasets: FHM/HarM/MAMI → our 4 video benchmarks

**NOT deviations** (byte-for-byte upstream, preserved):
- `get_rsa_label` retrieval + majority vote
- `parse_answer` harmful/harmless extraction
- Re-ask branch: `basic_predict != rsa_label` → `RSA_PROMPT`
- Control flow in main loop
- LLaVA forward via `run_proxy`

## Order of operations

1. Engineer writes all 6 files + README
2. Syntax check (py_compile + imports)
3. Report deliverable
4. Director reviews against upstream verbatim bits
5. Director submits:
   - `retrieval.py` first (CPU sbatch or light GPU for jina-clip
     encoding — reuses cached model) to build `rel_sampl.json` per
     dataset (4 runs)
   - `reproduce_lorehm.py` per dataset (GPU sbatch, llava-34b-awq
     loaded once per dataset run; 4 runs)

## Syntax check

```bash
conda activate SafetyContradiction
python -m py_compile src/lorehm_repro/*.py
python -c "import sys; sys.path.insert(0,'src/lorehm_repro'); import reproduce_lorehm, prompts, retrieval, grid_composite, llava_runner, lorehm_video_dataset; print('imports ok')"
```

**No sbatch. No smoke.** Report file list, audit notes (per-item
PASS/FIX against upstream), syntax log, deviation list.
