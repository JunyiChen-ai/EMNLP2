# ALARM reproduction (video-adapted)

ALARM (KDD 2026, Lang et al., "From Shallow Humor to Metaphor: Towards Label-Free Harmful Meme Detection via LMM Agent Self-Improvement") is a 5-stage label-free pipeline that self-improves a learner LMM agent by contrasting confidence-identified "explicit" memes into pairwise training signals. This repro adapts ALARM to our 4 video benchmarks under the 2026-04-15 `feedback_meme_to_video_8frames.md` rule.

## Upstream

- Repo: `external_repos/alarm/`
- Paper: "From Shallow Humor to Metaphor: Towards Label-Free Harmful Meme Detection via LMM Agent Self-Improvement", KDD 2026, Lang et al.
- Backbone: `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` via HF transformers + `Qwen2_5_VLForConditionalGeneration.from_pretrained` (`src/model/utils/models/qwen2vl_model.py:76-86`).
- Retrieval model: **Jina-CLIP-v2** (`jinaai/jina-clip-v2`, `truncate_dim=512`) for per-item image + text embeddings (`make_embeddings.py:11`). Already cached for MATCH stage 2b.5.
- Datasets upstream: FHM, MAMI, ToxiCN (single-image meme benchmarks).

## Pipeline (5 stages + Jina-CLIP embedding stage)

Upstream's `run/run_FHM.sh` sequences 5 stages via Hydra dispatching on `cfg.task`:

```
  Stage 1 — Label           (Qwen2.5-VL-72B-AWQ, 1 image → logits[0]/logits[1], confidence)
  Stage 2 — make_embeddings (Jina-CLIP-v2, image + text → 512-d each)
  Stage 3 — conduct_retrieval (filter by confidence, greedy-match non-hateful ↔ hateful pairs)
  Stage 4 — Experience      (Qwen2.5-VL-72B-AWQ, 2 images → per-pair rationale)
  Stage 5 — Reference       (Qwen2.5-VL-72B-AWQ, text-only → distilled reference set via ADD/EDIT/UPVOTE/DOWNVOTE ops)
  Stage 6 — InPredict       (Qwen2.5-VL-72B-AWQ, 1 image + references → final harmful/harmless)
```

Our video-adapted pipeline follows the same order, reading from our project's `data_utils` + per-stage intermediate files:

```
  frames_16/<vid>/[0,2,4,6,8,10,12,14].jpg  (8 frames per video)
      │
      ├─► Stage 1 Label (8-frame multi-image call → logit 0/1)
      │       │
      │       ▼
      │   label.jsonl
      ├─► Stage 2 make_embeddings (Jina-CLIP per-frame mean-pooled)
      │       │
      │       ▼
      │   fea/{image,text,joint}_embed.pt
      ├─► Stage 3 conduct_retrieval (confidence filter + greedy pair match)
      │       │
      │       ▼
      │   retrieve/pairs.jsonl
      ├─► Stage 4 Experience (16-frame multi-image pair call)
      │       │
      │       ▼
      │   experience.jsonl
      ├─► Stage 5 Reference (text-only, distill ADD/EDIT/UPVOTE/DOWNVOTE ops)
      │       │
      │       ▼
      │   reference.json
      └─► Stage 6 InPredict (8-frame multi-image + references → test prediction)
              │
              ▼
          test_alarm.jsonl
```

Stages 1-5 run on the **train split** (upstream's self-improvement train loop). Stage 6 runs on the **test split** for the final prediction.

| Stage | File (this repro) | Upstream file | Call type | Frames per call |
|---|---|---|---|---|
| 1 | `stages.run_label` | `Label/label_runner.py` | `chat_label_video` | 8 |
| 2 | `stages.run_make_embeddings` | `Experience/make_embeddings.py` | Jina-CLIP encode | 8 (mean-pooled) |
| 3 | `stages.run_conduct_retrieval` | `Experience/conduct_retrieval.py` | cosine similarity + greedy match | — |
| 4 | `stages.run_experience` | `Experience/experience_runner.py` | `chat_multi_img_video` | 16 (8 from A + 8 from B) |
| 5 | `stages.run_reference` | `Reference/reference_runner.py` | `chat_text` | 0 (text-only) |
| 6 | `stages.run_inpredict` | `InPredict/inpredict_runner.py` | `chat_multi_img_video` | 8 |

## Upstream-verbatim bits

- **Qwen2.5-VL loader** (`qwen2vl_video_model.Qwen2VLVideoModel.__init__`): `Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="cuda", attn_implementation="flash_attention_2", low_cpu_mem_usage=True)` — byte-for-byte from upstream `qwen2vl_model.py:79-86`.
- **Token ids** `self.one_id = tokenizer.convert_tokens_to_ids("1")`, `self.zero_id = tokenizer.convert_tokens_to_ids("0")` — upstream `qwen2vl_model.py:91-92`.
- **`chat_label_video`** — mirrors upstream `chat_label` (`qwen2vl_model.py:186-243`) except the input is 8 frames instead of 1 image. The `generate(..., max_new_tokens=1, return_dict_in_generate=True, output_logits=True, return_legacy_cache=True)` logit-access trick + the softmax over `[logits[zero_id], logits[one_id]]` is byte-for-byte upstream.
- **`chat_multi_img_video`** — mirrors upstream `chat_multi_img` (`qwen2vl_model.py:246-286`); the upstream loop that builds `conversation[0]["content"]` with `{"type": "image", "image": f"data:image;base64,..."}` entries is identical, we just pass more frames. `max_new_tokens=1024` preserved.
- **`chat_text`** — byte-for-byte from `qwen2vl_model.py:288-322`.
- **Stage 1 Label prompt** — upstream `label_runner.py:164-169`, meme→video rewrite only (3 word replacements). All other phrasing (including "avoid overgeneralizing or being overly conclusive") byte-for-byte.
- **Stage 4 Experience prompt** — upstream `experience_runner.py:148-155`, meme→video rewrite. Step 1 / Step 2 instructions unchanged.
- **Stage 5 Reference prompt** — upstream `reference_runner.py:16-57`, meme→video rewrite. ADD/EDIT/UPVOTE/DOWNVOTE rules, `{size}` cap, JSON format, 7-step Processing Steps block all byte-for-byte upstream.
- **Stage 6 InPredict prompt** — upstream `inpredict_runner.py:21-28`, meme→video rewrite. `Thought: [Your analysis] Answer: [harmful/harmless]` format unchanged.
- **Stage 3 retrieval logic** — `run_conduct_retrieval` mirrors upstream `conduct_retrieval.py:95-197` line-by-line: confidence filter by `coverage_rate`, cosine similarity with `sims_img + sims_txt`, greedy pair matching keeping only `query.pred=0, base.pred=1` pairs, sorted by similarity descending, each id consumed at most once.
- **Jina-CLIP loader** — `SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True, truncate_dim=512, device='cuda')` — byte-for-byte from upstream `make_embeddings.py:11`.

## Deviations (all documented)

1. **Scope: meme → video** via 8-frame multi-image adaptation per `feedback_meme_to_video_8frames.md`. 8 frames via indices `[0,2,4,6,8,10,12,14]` from `frames_16/<vid>/` (same indices MATCH and LoReHM use; shared extracted jpgs).
2. **Prompt text rewrites** — meme → video, image → "these 8 frames uniformly sampled from the video". Experience stage's "two memes" becomes "Video A (frames 1-8)" / "Video B (frames 9-16)" with one added sentence telling the model which frames are which. Every other upstream instruction preserved byte-for-byte.
3. **Image embedding aggregation**: upstream encodes one meme image → 512-d. We encode 8 frames → 8 × 512-d → mean-pool → re-L2-normalize → 512-d video image embedding. Text embedding path unchanged (one transcript per video → 512-d).
4. **Model**: `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` (upstream-exact). 72B bf16 ≈ 144 GB weights, won't fit on 80 GB A100; AWQ 4-bit ≈ 40 GB, fits with activations at `gpu_memory_utilization=0.92`. Same hard-block VRAM substitution class as MARS 32B-AWQ.
5. **Framework**: HF transformers (no vLLM substitution) — upstream uses HF + `chat_label`'s `output_logits=True` logit trick, which vLLM's `LLM.chat` API doesn't expose cleanly. Keeping HF preserves Label-stage fidelity.
6. **Attention backend**: upstream `qwen2vl_model.py:84` uses `attn_implementation="flash_attention_2"`; we substitute `"sdpa"` (PyTorch scaled-dot-product attention). Rationale: `flash_attn` is not installed in our `SafetyContradiction` conda env and building from source is an engineering hazard on this cluster. `sdpa` is a documented drop-in replacement that Qwen2.5-VL officially supports via HF transformers, with near-identical throughput on modern A100 hardware. Mathematical result is identical; only the CUDA kernel differs.
7. **Text field**: meme OCR → video transcript (`ann["transcript"]`). Closest analog in our annotations.
8. **Experience 16-frame layout**: first 8 frames = Video A, last 8 frames = Video B. Frame order is the conversation-content order passed to the Qwen chat template; the prompt text explicitly tells the model "frames 1-8 are from Video A, frames 9-16 are from Video B."
9. **`coverage_rate` default = 0.5**: upstream iterates over 0.1..1.0 in 0.1 steps and evaluates at each step (`conduct_retrieval.py:98`). We pick the midpoint (`0.5`) as a sensible single operating point; exposed as a CLI flag for override.
10. **`reference_size` default = 20**: the upstream `cfg.size` field in `reference_qwen_FHM.yaml` sets the reference-set cap; our default matches the spirit of a small distilled set. Exposed as a CLI flag.
11. **Datasets**: FHM / MAMI / ToxiCN → MHClip_EN / MHClip_ZH / HateMM / ImpliHateVid.

## Files

- `alarm_video_dataset.py` — per-video item loader (`build_video_items`), frame index rule, `collapse_label` copy
- `qwen2vl_video_model.py` — `Qwen2VLVideoModel` class wrapping upstream's 72B-AWQ loader + `chat_label_video` / `chat_multi_img_video` / `chat_text`
- `stages.py` — 6 stage functions (`run_label`, `run_make_embeddings`, `run_conduct_retrieval`, `run_experience`, `run_reference`, `run_inpredict`) + `ReferenceSet` class + `_parse_reference_ops` / `_parse_answer` helpers + all 4 video-adapted prompt strings (`LABEL_PROMPT_VIDEO`, `EXPERIENCE_PROMPT_VIDEO`, `REFERENCE_PROMPT_VIDEO`, `INPREDICT_PROMPT_VIDEO`)
- `reproduce_alarm.py` — main driver wiring the 6 stages end-to-end, per-stage argparse toggles, per-dataset loop

## Output

```
results/alarm/<dataset>/
    label.jsonl                  # stage 1 — {id, pred, label, prob0, prob1, output_text}
    fea/
        image_embed.pt           # stage 2 — {id -> torch.Tensor[512]}
        text_embed.pt            # stage 2
        joint_embed.pt           # stage 2
    retrieve/
        pairs.jsonl              # stage 3 — {id1, id2, similarity}
    experience.jsonl             # stage 4 — {id1, id2, experience}
    reference.json               # stage 5 — list of {reference, importance}
    test_alarm.jsonl             # stage 6 — FINAL — {video_id, pred, label, thought, raw_response}
```

`test_alarm.jsonl` is compatible with `src/naive_baseline/eval_generative_predictions.py --input ... --dataset ...`.

## CLI (SafetyContradiction env — HF + transformers + Jina-CLIP + Qwen2.5-VL)

```bash
conda activate SafetyContradiction

# Full pipeline, all 4 datasets, all 6 stages:
python src/alarm_repro/reproduce_alarm.py --all

# Single dataset, full pipeline:
python src/alarm_repro/reproduce_alarm.py --dataset MHClip_EN

# Re-run only stages 3-6 (assuming label + embed already computed):
python src/alarm_repro/reproduce_alarm.py --all \
  --no-do-label --no-do-embed

# Re-run only the final InPredict stage:
python src/alarm_repro/reproduce_alarm.py --all \
  --no-do-label --no-do-embed --no-do-retrieve \
  --no-do-experience --no-do-reference
```

## Pre-flight (director-side)

1. `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` (~40 GB on disk) — login-node `snapshot_download` before the real sbatch
2. `jinaai/jina-clip-v2` — should already be warm from MATCH stage 2b.5
3. `frames_16/<vid>/frame_NNN.jpg` for all 4 datasets × both splits — extracted by `src/match_repro/extract_frames.py`, shared with MATCH and LoReHM
4. `flash_attention_2` — upstream `qwen2vl_model.py:84` requires `attn_implementation="flash_attention_2"`; ensure the Flash-Attention wheel is available in `SafetyContradiction`

## Anticipated runtime

- Stage 1 (Label on train split): ~N_train × 1 forward pass ≈ a few minutes per dataset at 72B-AWQ
- Stage 2 (Jina-CLIP embed train split): CPU/GPU-light, seconds per dataset
- Stage 3 (greedy matching): CPU, seconds
- Stage 4 (Experience on up to N_train/2 pairs): most expensive — 16-frame multi-image call per pair. Bound via `coverage_rate`.
- Stage 5 (Reference): one text-only call per experience entry, moderate
- Stage 6 (InPredict on test): ~N_test × 1 forward pass

Total: on the order of a few hours per dataset × 4 datasets ≈ half to full day for the full sweep, same ballpark as MATCH stage 2a/2b/2c combined.
