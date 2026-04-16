# Pro-Cap reproduction (faithful probing half)

## Upstream
- **Paper**: Rui Cao et al., *Pro-Cap: Leveraging a Frozen Vision-Language Model for Hateful Meme Detection*, ACM MM 2023 (arxiv 2308.08088)
- **Repo**: https://github.com/Social-AI-Studio/Pro-Cap
- **Upstream commit**: main branch as of 2026-04-15 (cloned at `external_repos/procap/`)
- **Source of truth**: `external_repos/procap/codes/Pro-Cap-Generation.ipynb`
- **Backbone**: BLIP-2 FlanT5-XL, loaded via LAVIS `load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl")` (upstream notebook cells `ae5d43bd` / `29555f87`).

## Two versions in this directory

| File | Framework | Status | Notes |
|---|---|---|---|
| `reproduce_procap_lavis.py` | **LAVIS** (`load_model_and_preprocess`) | **Paper-faithful** — use this | Matches upstream notebook call verbatim, including `.float()` upcast, `vis_processors["eval"]`, `length_penalty=3.0`, and `"Question: %s Answer:"` prompt template. |
| `reproduce_procap.py` | HF transformers (`Blip2ForConditionalGeneration`) | Legacy port — kept for comparison only | Same checkpoint weights but routed through HF `processor + generate` instead of LAVIS. Not faithful per `feedback_baseline_framework_match.md`. Output lives under `results/procap_blip2_flan_t5_xl/` to keep it distinct from the LAVIS run. |

Results from the two versions are written to different directories so they can be compared side-by-side:
- LAVIS: `results/procap_lavis_blip2_flan_t5_xl/<dataset>/test_procap.jsonl`
- HF legacy: `results/procap_blip2_flan_t5_xl/<dataset>/test_procap.jsonl`

## Frames per video
**1 frame (middle frame).** Pro-Cap is image-only; video adaptation uses one representative frame (per `feedback_baseline_frame_fidelity.md`).

## 8 VQA probes (verbatim from upstream notebook)
Extracted from `external_repos/procap/codes/Pro-Cap-Generation.ipynb` cells, in upstream execution order:
1. `what is the race of the person in the image?`
2. `what is the gender of the person in the image?`
3. `is there an animal in the image?`
4. `is there a person in the image?`
5. `which country does the person in the image come from?`
6. `what animal is in the image?`
7. `are there disabled people in the image?`
8. `what is the religion of the person in the image?`

Each probe is invoked via the upstream helper `generate_prompt_result(im, ques)` (notebook cell `1e529ebf`):

```python
def generate_prompt_result(im, ques):
    image = vis_processors["eval"](im).float().unsqueeze(0).to(device)
    ans = model.generate({"image": image,
                          "prompt": ("Question: %s Answer:" % (ques))},
                         length_penalty=3.0)
    return ans[0]
```

The LAVIS script copies this helper byte-for-byte.

## Label-free classification head (our adaptation)

Upstream Pro-Cap feeds the 8 captions into a **supervised RoBERTa classifier**. We do not use that classifier because our setting is label-free. Instead, we run a **9th BLIP-2 probe** through the same LAVIS `generate_prompt_result` call, asking the model directly whether the content is hateful, with the concatenated 8 VQA answers + first 1000 chars of the video transcript as textual context.

Classification prompt (wrapped by `"Question: %s Answer:"` inside the helper, so the model sees `"Question: <below> Answer:"`):
```
The image has been described as follows: {captions_concat}. The accompanying
transcript says: "{transcript}". Is this content promoting hateful,
discriminatory, or dehumanizing stereotypes against any identity group?
Answer yes or no.
```
- `{captions_concat}` = 8 VQA answers joined with `; ` in upstream probe order.
- `{transcript}` = first 1000 chars of `entry["transcript"]`, with `"` → `'` to keep the prompt well-formed; empty string if the dataset has no transcript.
- Parsing: strip / lowercase the raw output; starts with `yes` → pred=1; starts with `no` → pred=0; otherwise pred=-1. The raw string is logged in `verdict_text` for audit.

This keeps the whole pipeline zero-shot through a single frozen MLLM (BLIP-2 FlanT5-XL) and matches the user's "probing-only half" requirement for label-free baselines.

## Total MLLM calls per video
**9** (8 VQA probes + 1 classification probe). Baselines are exempt from our method's 2-call cap; the mandate is faithful reproduction of the upstream pipeline.

## Integrity discipline (per `feedback_baseline_integrity.md`)
- `f.flush()` + `os.fsync()` after every per-video write.
- Resume via `resume_done_ids()` — re-running skips video_ids already in the output jsonl.
- Integrity audit every 50 videos: re-read line count, verify `video_id`, `pred`, `verdict_text`, `probe_answers` fields are present.
- LAVIS `model.generate` is left at its default `max_length`. If truncation is observed in a production run (long VQA answers ending mid-token), bump `max_length` in the helper and rerun from scratch.

## CLI

LAVIS (paper-faithful):
```bash
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate lavis_baselines
HF_HUB_OFFLINE=1 python src/procap_repro/reproduce_procap_lavis.py --dataset MHClip_EN --split test
# smoke test
HF_HUB_OFFLINE=1 python src/procap_repro/reproduce_procap_lavis.py --dataset MHClip_EN --split test --limit 1
# all datasets
HF_HUB_OFFLINE=1 python src/procap_repro/reproduce_procap_lavis.py --all --split test
```

HF legacy (comparison only):
```bash
conda activate SafetyContradiction
python src/procap_repro/reproduce_procap.py --all --split test
```

## Output schema (both versions)
One JSON line per video:
```json
{
  "video_id": "<vid>",
  "pred": 0 | 1 | -1,
  "verdict_text": "<raw response to 9th probe>",
  "probe_answers": {"race": "...", "gender": "...", "animal": "...", "person": "...", "country": "...", "what_animal": "...", "disabled": "...", "religion": "..."},
  "caption": "<captions_concat passed into the 9th probe>"
}
```

Evaluable via:
```bash
python src/naive_baseline/eval_generative_predictions.py \
  --input results/procap_lavis_blip2_flan_t5_xl/<dataset>/test_procap.jsonl \
  --dataset <dataset>
```
