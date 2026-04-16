# Task brief — Pro-Cap LAVIS faithful rewrite

## Why this brief exists

`src/procap_repro/reproduce_procap.py` was written under the old
solo-director model using HF transformers
(`Blip2ForConditionalGeneration`). Per `feedback_baseline_framework_match.md`
the upstream Pro-Cap paper loads the model via **LAVIS**
`load_model_and_preprocess`. Numbers from the HF port are NOT
faithful. Rewrite in LAVIS.

Existing HF results are under `results/procap_blip2_flan_t5_xl/`.
**Do not overwrite them** — write LAVIS outputs to a distinct path
so we can compare.

## Upstream

- **Paper**: Rui Cao et al., *Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection*, ACM MM 2023 (arxiv 2308.08088)
- **Repo**: https://github.com/Social-AI-Studio/Pro-Cap
- **Upstream copy**: `external_repos/procap/`
- **Source of truth**: `external_repos/procap/codes/Pro-Cap-Generation.ipynb`

## Framework — LAVIS (NOT HF transformers)

Verbatim upstream model load (notebook cell `ae5d43bd` + `29555f87`):

```python
from lavis.models import load_model_and_preprocess

device = torch.device("cuda:0")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_t5",
    model_type="caption_coco_flant5xl",
    is_eval=True,
    device=device,
)
model = model.float()
```

`.float()` upcast is upstream-exact — keep it.

## Probe invocation — verbatim

Upstream `generate_prompt_result` (notebook cell `1e529ebf`):

```python
def generate_prompt_result(im, ques):
    image = vis_processors["eval"](im).float().unsqueeze(0).to(device)
    ans = model.generate({"image": image,
                          "prompt": ("Question: %s Answer:" % (ques))},
                         length_penalty=3.0)
    return ans[0]
```

**Hard requirements** (any deviation = rejection):
- Prompt template: `"Question: %s Answer:"` (spaces + case exact)
- `length_penalty=3.0`
- `.float().unsqueeze(0)` after the vis processor
- `vis_processors["eval"]` (eval processor, not train)

## 8 VQA probes — verbatim

From the notebook cells, in upstream order:

1. `what is the race of the person in the image?`
2. `what is the gender of the person in the image?`
3. `is there an animal in the image?`
4. `is there a person in the image?`
5. `which country does the person in the image come from?`
6. `what animal is in the image?`
7. `are there disabled people in the image?`
8. `what is the religion of the person in the image?`

## Video adaptation (we are adapting a meme baseline to video)

Pro-Cap is image-only. For video:
- **1 frame per video** — the middle frame
- Use the already-existing `data_utils.get_media_path()` from
  `src/our_method/data_utils.py` to find either a pre-extracted
  `frames/<vid>/` directory or an mp4 to decode
- If the video has pre-extracted frames, grab the middle jpg
- If only mp4, use decord to pull frame at index `len(vr)//2`

## 9th classification probe — our label-free adaptation

Upstream Pro-Cap feeds the 8 captions into a supervised RoBERTa
classifier. We do NOT use their classifier — we're label-free. Our
adaptation: run a **9th BLIP-2 probe** using the same LAVIS
`model.generate` call, with the concatenated 8 answers + video
transcript as textual context, asking the model directly whether
the content is hateful. Parse yes/no.

**Classification prompt template** (we designed this — document it
clearly in the README):

```
Question: The image has been described as follows: {captions_concat}. The accompanying transcript says: "{transcript}". Is this content promoting hateful, discriminatory, or dehumanizing stereotypes against any identity group? Answer yes or no. Answer:
```

- `{captions_concat}` = the 8 VQA answers joined with `; ` in probe order
- `{transcript}` = first 1000 chars of `entry["transcript"]`, else empty string
- Parse the response: strip whitespace, lowercase, check if it starts
  with `yes` → pred=1, otherwise pred=0. Log the raw `verdict_text`
  for audit.

## Dataset adapter (our wrapper)

Use existing `src/our_method/data_utils.py` functions:
- `DATASET_ROOTS`, `load_annotations(dataset)`, `get_media_path(vid, dataset)`, `SKIP_VIDEOS`
- Iterate test split via `splits/test_clean.csv` (already generated)
- 4 datasets: MHClip_EN, MHClip_ZH, HateMM, ImpliHateVid

## Integrity discipline (per `feedback_baseline_integrity.md`)

- `fsync` after every per-video jsonl write
- Resume: skip video_ids already present in the output jsonl
- `max_new_tokens`: LAVIS `model.generate` defaults are fine for the
  probes, but if you see truncated outputs (ends without punctuation
  on a long answer), bump the `max_length` kwarg of
  `model.generate` and rerun from scratch
- Integrity audit every 50 videos: re-read the jsonl line-count
  and confirm each line has `video_id`, `pred`, `verdict_text`,
  `probe_answers`

## Output path + schema

- **Path**: `results/procap_lavis_blip2_flan_t5_xl/<dataset>/test_procap.jsonl`
  (distinct from the HF port's `results/procap_blip2_flan_t5_xl/`)
- **Schema** (one line per video):
  ```json
  {
    "video_id": "<vid>",
    "pred": 0 | 1,
    "verdict_text": "<raw response to 9th probe>",
    "probe_answers": {"race": "...", "gender": "...", ..., "religion": "..."},
    "caption": "<captions_concat passed into 9th probe>"
  }
  ```
- Evaluable via `src/naive_baseline/eval_generative_predictions.py --input ... --dataset ...`

## Where to write

- `src/procap_repro/reproduce_procap_lavis.py` — the new LAVIS script
- `src/procap_repro/README.md` — **update** to document both the HF
  port (legacy) and the LAVIS faithful version; note that the LAVIS
  version is the paper-faithful one

## Conda env

`lavis_baselines` (already built with `numpy<2`, `torch 2.4.1+cu121`,
`lavis` from pip). Smoke-test locally with:
```
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate lavis_baselines
HF_HUB_OFFLINE=1 python src/procap_repro/reproduce_procap_lavis.py --dataset MHClip_EN --split test --limit 1
```

Add a `--limit N` argparse flag for the smoke test (run only first N
videos) — purely to support smoke testing, not a feature.

## Smoke test deliverable

When done, reply to the director with:

1. File list (`src/procap_repro/reproduce_procap_lavis.py`, README diff)
2. Smoke log: the `--limit 1` run output including the 8 probe answers
   and the final yes/no verdict for a single video on MHClip_EN
3. Whether LAVIS loaded cleanly in the `lavis_baselines` env without
   errors
4. Any deviation from this brief — flagged explicitly for director
   review

**Do NOT submit sbatch.** Do not touch `results/`. Do not edit files
outside `src/procap_repro/`.

## Review gate (director applies)

On receiving the deliverable, director verifies:
1. Framework — `from lavis.models import load_model_and_preprocess` ✓
2. Model load line matches upstream verbatim (name, model_type, is_eval, `.float()`)
3. `generate_prompt_result` matches upstream's prompt template + length_penalty
4. All 8 probes are present in upstream order
5. 1 frame (middle) per video
6. Output path distinct from HF port
7. fsync + resume + integrity audit are in the code
8. README documents upstream commit + both versions
9. Smoke-log shows the 9th probe produced a parseable yes/no

Any miss → reject with a specific item list and return to engineer.
