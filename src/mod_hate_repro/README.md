# Mod-HATE reproduction (video-adapted)

Mod-HATE (Cao et al., WWW 2024, "Modularized Networks for Few-shot Hateful Meme Detection") is a **few-shot supervised** text-only LLaMA-7B meme classifier. It never sees pixels directly — meme images are first turned into captions upstream (by Pro-Cap), then LLaMA reads `Image caption:<...>\nMeme text:<ocr>` and emits a `Yes` / `No` verdict.

This repro adapts Mod-HATE to our 4 video benchmarks under the 2026-04-15 user directive `feedback_meme_to_video_8frames.md` (meme→video via 8 uniformly sampled frames per video, one LLM call per video, text-only adaptation allowed here because Mod-HATE is already text-only).

## Upstream

- Repo: `external_repos/mod_hate/`
- Paper: "Modularized Networks for Few-shot Hateful Meme Detection", WWW 2024, Rui Cao, Roy Ka-Wei Lee, Jing Jiang.
- Backbone: `yahma/llama-7b-hf` via HF transformers + PEFT.
- Pre-trained LoRA modules in `external_repos/mod_hate/LoRA_modules/`:
  - `hate-exp/` — trained on HatReD (hateful-meme explanation)
  - `meme-captions/` — trained on MemeCap (meme comprehension)
  - `hate-speech/` — trained on DT+WZ+Gab text hate-speech corpora
- Pipeline:
  1. `lora_learning.py` — LoRAHub-style Nevergrad NGOpt optimization over the 3 LoRA-module weights on a labeled K-shot support set (K=4 or K=8 per class), `max_inference_step=40`, bounds `[-1.5, 1.5]`.
  2. `hfm_gen_eval.py` — evaluates the composed LoRA on the test set. Upstream reads yes/no logits at token ids `8241` (`Yes`) / `3782` (`No`) of the yahma/llama-7b-hf tokenizer.

## Pipeline (video-adapted)

```
 8-frame Pro-Cap captions  ─┐
                            ├─► video_caption_adapter.build_examples ─► {img, label, input, instruction, output}
 transcripts                ─┘                                                       │
                                                                                     │  (K-shot support + test)
                                                                                     ▼
                                                           lora_compose.lorahub_learning
                                                        (LLaMA-7B + 3 LoRAs, Nevergrad NGOpt)
                                                                                     │
                                                                                     ▼
                                                  composed LLaMA  →  per-video Yes/No logits
                                                                                     │
                                                                                     ▼
                                      test_mod_hate_{4,8}shot.jsonl (video_id, pred, response, yes/no logits)
```

| Stage | File | Framework | Role |
|---|---|---|---|
| Pre-req | `../procap_repro/reproduce_procap_lavis_8frame.py` | LAVIS | 8-frame Pro-Cap captions per video |
| 1 | `video_caption_adapter.py` | CPU Python | Stitch 8 per-frame captions → video caption + transcript → upstream-shaped example dict |
| 2 | `lora_compose.py` | HF + PEFT + Nevergrad | Load LLaMA-7B + 3 LoRAs, compose via NGOpt on K-shot support loss |
| 3 | `reproduce_mod_hate.py` | HF + PEFT | Run composed LLaMA on test split, read yes/no logits, write jsonl |

## Upstream-verbatim bits

- **Verbolizer**: upstream `few_hm_dataset.py:128-130` uses `{0: POS_WORD, 1: NEG_WORD}` = `{0: "No", 1: "Yes"}`. We keep this exactly — label=1 (hateful) → `"Yes"`.
- **Prompt templates**: `generate_train_prompt` / `generate_eval_prompt` in `reproduce_mod_hate.py` are byte-for-byte ports of upstream `few_hm_dataset.py:60-93` — same indentation, same `### Instruction:` / `### Input:` / `### Response:` section headers.
- **Tokenization rule**: `tokenize()` copies upstream `few_hm_dataset.py:95-117` verbatim (EOS append, label masking by user-prompt length, left-pad to `cutoff_len`).
- **LoRA module paths** resolved from `external_repos/mod_hate/LoRA_modules/<name>/` → `PeftModel.from_pretrained(base, <path>)`. Upstream's `load_base_model_and_lora_modules` (`lora_learning.py:48-96`) is reproduced in `lora_compose.py` with the same "first module is default" + arch-compatibility check pattern.
- **`get_score` / `default_get_loss` / `default_l1_regularization` / `get_final_weights`** — byte-for-byte copies of upstream `lora_learning.py:98-181`, logger calls dropped so the module is import-clean (our driver logs via `logging` instead).
- **Nevergrad setup**: `ng.p.Array(init=[0]*N, upper=[1.5]*N, lower=[-1.5]*N)` + `ng.optimizers.NGOpt(..., budget=max_inference_step)` — upstream `lora_learning.py:231-236`, verbatim.
- **Yes/No token ids `8241` / `3782`** — upstream `hfm_gen_eval.py:66-67`, hard-coded for yahma/llama-7b-hf tokenizer. We use the same ids. On the first real run we'll confirm these still decode correctly via `tokenizer.decode([8241])` / `[3782]`.

## Deviations (documented)

1. **Scope: meme → video** via 8-frame Pro-Cap concatenation. Upstream's meme-caption field becomes a video caption of the form `"Frame 1: <cap_0>. Frame 2: <cap_1>. ... Frame 8: <cap_7>."` (`video_caption_adapter.build_video_caption`).
2. **Caption source**: 1-frame Pro-Cap pickle (`BLIP-2/results/<dataset>-generic.pkl`) → 8-frame Pro-Cap LAVIS jsonl (`results/procap_lavis_blip2_flan_t5_xl_8frame/<dataset>/<split>_procap.jsonl`).
3. **Meme OCR → video transcript**. Our annotations have no per-meme OCR; `transcript` is the closest analog and is what Mod-HATE's `Meme text:` slot now receives. Truncated to `--cutoff-len` worth of tokens downstream.
4. **Support set**: 4 or 8 **labeled videos** per class sampled from `splits/train_clean.csv` (upstream's `Few_HM_Data.process_data` samples 4 or 8 per class from the meme train split via `counts[0]==num_shots and counts[1]==num_shots` — same stopping rule).
5. **`cutoff_len=512`** (default). Upstream default is 256; raised here because 8-frame concat + transcript is longer than a single meme caption + OCR. Still well within LLaMA-7B's 2048 context.
6. **Instruction rewrite**: "Please decide whether the **meme** is hateful according to its **image caption and meme text**" → "Please decide whether the **video** is hateful according to its **video caption and video transcript**". This is the only prompt-text rewrite, per `feedback_meme_to_video_8frames.md`'s rule "rewrite meme → video in prompt text".
7. **Eval**: upstream's `hfm_generation` runs its own meme-captions-pickle loader; we replace that with `reproduce_mod_hate.score_one_test_row`, which does exactly one `model.generate(..., max_new_tokens=1)` forward and reads `out["scores"][0][0, 8241]` / `[0, 3782]`. This is functionally equivalent to the upstream 1st-token scoring branch.
8. **Few-shot supervision is in-bounds**: the 4 / 8 labeled examples per dataset are accepted as a **supervised reference baseline** under the 2026-04-15 user directive (MATCH stage 3 precedent). Mod-HATE rows appear alongside MATCH in the "supervised reference" section of the comparison table.
9. **`load_8bit=True` by default** (upstream's `individual_module_infer.sh` pattern). LLaMA-7B int8 fits comfortably on a single A100; a `--no-load-8bit` flag is available if fp16 is preferred for a specific run.

## Datasets

All 4 (MHClip_EN, MHClip_ZH, HateMM, ImpliHateVid). Each dataset gets 2 runs: K=4 and K=8.

## Output

```
results/mod_hate/
    MHClip_EN/
        test_mod_hate_4shot.jsonl
        test_mod_hate_8shot.jsonl
    MHClip_ZH/ …
    HateMM/ …
    ImpliHateVid/ …
```

Schema per line:
```json
{
  "video_id": "...",
  "pred": 0 | 1,
  "response": "Yes" | "No",
  "yes_logit": ..., "no_logit": ...,
  "lora_weights": [w_hate_exp, w_meme_captions, w_hate_speech]
}
```

Evaluable via `src/naive_baseline/eval_generative_predictions.py --input ... --dataset ...`.

## CLI (in `SafetyContradiction` conda env — HF + PEFT + Nevergrad)

```bash
conda activate SafetyContradiction

# Pre-req: 8-frame Pro-Cap LAVIS output must exist for the dataset.
# (Director submits that run separately in the `lavis_baselines` env.)

# Both K=4 and K=8, all 4 datasets:
python src/mod_hate_repro/reproduce_mod_hate.py --all

# Single dataset, K=8 only:
python src/mod_hate_repro/reproduce_mod_hate.py --dataset MHClip_EN --shots 8

# Single dataset, custom Nevergrad budget:
python src/mod_hate_repro/reproduce_mod_hate.py --dataset HateMM --shots 4 --max-inference-step 80
```

## Files in this directory

- `video_caption_adapter.py` — Pro-Cap 8-frame → Mod-HATE example dict adapter (pure Python, no torch).
- `lora_compose.py` — upstream `lora_learning.py` port, LLaMA + 3 LoRAs loader + Nevergrad composer.
- `reproduce_mod_hate.py` — main driver, prompt templates, SupportDataset, test-set scoring, CLI.
- `README.md` — this file.

## Pre-req order for director-side mass submission

1. Wait for 8-frame Pro-Cap LAVIS run to land `results/procap_lavis_blip2_flan_t5_xl_8frame/<dataset>/test_procap.jsonl` (plus a `train_procap.jsonl` for each dataset to build the K-shot support).
2. Run `reproduce_mod_hate.py --all` — one GPU sbatch, processes all 4 datasets × 2 K values = 8 composed LoRAs + 8 test runs sequentially.
3. `scripts/eval_all_baselines.py` picks up `results/mod_hate/*/test_mod_hate_*shot.jsonl` automatically on its next walk.
