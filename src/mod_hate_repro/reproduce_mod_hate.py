"""Mod-HATE reproduction — video-adapted driver.

Upstream: `external_repos/mod_hate/` (Cao et al., WWW 2024,
"Modularized Networks for Few-shot Hateful Meme Detection"). Backbone
is `yahma/llama-7b-hf` via HF transformers + PEFT; 3 pre-trained LoRA
modules (`hate-exp`, `meme-captions`, `hate-speech`) are composed via
Nevergrad black-box optimization on a K=4 or K=8 labeled support set.

Video adaptation per `docs/baseline_briefs/mod_hate.md` + user rule
`feedback_meme_to_video_8frames.md`:

  1. Per-meme caption → 8-frame Pro-Cap caption concat (via
     `video_caption_adapter.build_video_caption`). Pro-Cap captions
     come from `results/procap_lavis_blip2_flan_t5_xl_8frame/...`.
  2. Per-meme OCR text → video transcript (closest analog in our
     annotations).
  3. K=4 and K=8 support sets are sampled from each dataset's train
     split (`splits/train_clean.csv`); upstream's balanced
     `counts[0]==K and counts[1]==K` stopping rule is preserved.
  4. Nevergrad NGOpt over the 3 LoRA-module weights (verbatim upstream
     `lorahub_learning`), budget `max_inference_step`.
  5. Inference on the test split: LLaMA `generate` with 1 new token
     max; we read the first-token logits for `Yes` / `No` (token ids
     `8241` / `3782` — upstream `hfm_gen_eval.py:66-67` hard-coded for
     the yahma/llama-7b-hf tokenizer) and pick the argmax as the
     binary label. Upstream's `POS_WORD="No"` / `NEG_WORD="Yes"` →
     `{0:"No", 1:"Yes"}` verbolizer is preserved so "Yes" means "yes,
     hateful."

Output: `results/mod_hate/<dataset>/test_mod_hate_<K>shot.jsonl` with
schema `{"video_id": ..., "pred": 0|1, "response": "Yes|No",
"yes_logit": ..., "no_logit": ...}`. Compatible with
`eval_generative_predictions.eval_one()`.

Framework: HF transformers + PEFT (no vLLM substitution — upstream
uses HF, no cloud API, no oversized backbone).
"""

import argparse
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))

import video_caption_adapter  # noqa: E402
import lora_compose  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"
ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]
DEFAULT_BASE_MODEL = "yahma/llama-7b-hf"

# yahma/llama-7b-hf tokenizer: "Yes"=8241, "No"=3782 (upstream
# `hfm_gen_eval.py:66-67`, verbatim token ids).
YES_TOKEN_ID = 8241
NO_TOKEN_ID = 3782


# Upstream `few_hm_dataset.py:60-69` — `generate_prompt_test` returns
# the prompt shape used for evaluation (response placeholder empty).
# We reuse the string byte-for-byte except for the `instruction`
# content which is meme→video rewritten inside
# `video_caption_adapter.build_examples`.
def generate_eval_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:\n"""  # noqa: E501


def generate_train_prompt(data_point):
    """Upstream `few_hm_dataset.py:78-93` `generate_prompt` — training
    version with the response slot filled."""
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:{data_point["instruction"]}

                ### Input:{data_point["input"]}

                ### Response:{data_point["output"]}"""  # noqa: E501


def tokenize(prompt, tokenizer, cutoff_len, add_eos_token=True):
    """Upstream `few_hm_dataset.py:95-117`, verbatim (with `chatglm`
    branch dropped since we target LLaMA only)."""
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


class SupportDataset:
    """Torch-style `Dataset` wrapping K-shot support rows, matching
    the tensor shapes upstream `Few_HM_Data` produces."""

    def __init__(self, rows, tokenizer, cutoff_len=256, train_on_inputs=False):
        import torch as _t
        self._torch = _t
        self.rows = rows
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.train_on_inputs = train_on_inputs
        self.entries = [self._prep(r) for r in rows]

    def _prep(self, data_point):
        """Upstream `Few_HM_Data.generate_and_tokenize_prompt`, verbatim
        except for the shape of `data_point` (which comes from our
        video-adapter, with the same keys upstream uses)."""
        full_prompt = generate_train_prompt(data_point)
        tok = tokenize(full_prompt, self.tokenizer, self.cutoff_len)
        if not self.train_on_inputs:
            user_prompt = generate_train_prompt({**data_point, "output": ""})
            tok_user = tokenize(
                user_prompt, self.tokenizer, self.cutoff_len,
                add_eos_token=False,
            )
            user_prompt_len = len(tok_user["input_ids"])
            tok["labels"] = (
                [-100] * user_prompt_len
                + tok["labels"][user_prompt_len:]
            )
        pad_len = self.cutoff_len - len(tok["input_ids"])
        if pad_len > 0:
            tok["input_ids"] = [0] * pad_len + tok["input_ids"]
            tok["labels"] = [-100] * pad_len + tok["labels"]
            tok["attention_mask"] = (
                [0] * pad_len + tok["attention_mask"]
            )
        return tok

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        t = self._torch
        e = self.entries[idx]
        return {
            "input_ids": t.LongTensor(e["input_ids"]),
            "labels": t.LongTensor(e["labels"]),
            "attention_mask": t.LongTensor(e["attention_mask"]),
        }


def score_one_test_row(row, model, tokenizer):
    """One LLaMA forward pass; return (pred, yes_logit, no_logit, raw).

    Upstream `hfm_gen_eval.evaluate` runs `model.generate(..., max_new_tokens=128,
    output_scores=True)` and reads `generation_output['scores'][0][0, 8241]`
    / `[0, 3782]`. We replicate that exactly: 1 new token is enough to
    decide the verdict since yes/no is always the first token after
    `### Response:`.
    """
    import torch as _t
    from transformers import GenerationConfig

    prompt = generate_eval_prompt(row)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    gen_cfg = GenerationConfig()
    with _t.no_grad():
        out = model.generate(
            input_ids=input_ids,
            generation_config=gen_cfg,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=1,
        )
    yes_logit = out["scores"][0][0, YES_TOKEN_ID].item()
    no_logit = out["scores"][0][0, NO_TOKEN_ID].item()
    if yes_logit < -1e10:
        yes_logit = -1e10
    if no_logit < -1e10:
        no_logit = -1e10
    pred = 1 if yes_logit >= no_logit else 0
    response_tok = _t.argmax(out["scores"][0][0]).item()
    # Decode just the first generated token for the raw response
    # (matches upstream's `output.split("### Response:")[1].strip()` shape).
    response = tokenizer.decode([response_tok], skip_special_tokens=True)
    return pred, yes_logit, no_logit, response


def run_one_dataset(dataset, num_shots, args, out_dir):
    logging.info(f"[{dataset}] building K={num_shots} support + test")
    support, test_rows, missing = (
        video_caption_adapter.build_support_and_test(
            dataset, num_shots=num_shots, seed=args.seed
        )
    )
    logging.info(
        f"[{dataset}]   support={len(support)}  test={len(test_rows)}  "
        f"missing_caps_train={len(missing['train'])}  "
        f"missing_caps_test={len(missing['test'])}  "
        f"support_counts={missing['support_counts']}"
    )
    if len(support) < 2:
        logging.warning(
            f"[{dataset}] support set too small ({len(support)}); skipping"
        )
        return

    lora_paths = lora_compose.build_lora_module_list(
        lora_root=args.lora_root
    )

    support_ds = None  # built after tokenizer is loaded

    # Load base + LoRAs, then wrap support rows in the tokenizer-aware
    # dataset so Nevergrad's `get_loss` can forward through the model.
    model, tokenizer, cache = lora_compose.load_base_model_and_lora_modules(
        args.base_model,
        lora_paths,
        load_8bit=args.load_8bit,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    support_ds = SupportDataset(
        support, tokenizer, cutoff_len=args.cutoff_len
    )

    # Run Nevergrad composition. `lorahub_learning` will itself call
    # `load_base_model_and_lora_modules` once, so instead of reusing
    # our already-loaded `model`, we simply free our local handle
    # before handing control over — this mirrors upstream's fresh-
    # loading pattern inside `lorahub_learning` and avoids two copies
    # of LLaMA-7B on the GPU.
    del model
    del cache
    import gc
    gc.collect()
    import torch as _t
    _t.cuda.empty_cache()

    weights, composed_model, tokenizer_out = lora_compose.lorahub_learning(
        lora_module_list=lora_paths,
        base_model=args.base_model,
        support_dataset=support_ds,
        max_inference_step=args.max_inference_step,
        batch_size=args.batch_size,
        load_8bit=args.load_8bit,
        seed=args.seed,
    )
    composed_model.eval()
    composed_model.config.pad_token_id = tokenizer_out.pad_token_id = 0
    composed_model.config.bos_token_id = 1
    composed_model.config.eos_token_id = 2

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"test_mod_hate_{num_shots}shot.jsonl")

    # Skip already-predicted ids for resume.
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("video_id"):
                        done.add(r["video_id"])
                except Exception:
                    pass
    logging.info(f"[{dataset}] {len(test_rows)} test rows, {len(done)} already scored")

    t0 = time.time()
    n_processed = 0
    with open(out_path, "a") as f:
        for row in test_rows:
            vid = row["img"]
            if vid in done:
                continue
            try:
                pred, yes_l, no_l, resp = score_one_test_row(
                    row, composed_model, tokenizer_out
                )
            except Exception as e:
                logging.error(f"  {vid}: score failed: {e}")
                pred, yes_l, no_l, resp = -1, 0.0, 0.0, ""
            rec = {
                "video_id": vid,
                "pred": pred,
                "response": resp,
                "yes_logit": yes_l,
                "no_logit": no_l,
                "lora_weights": list(weights),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            n_processed += 1
            if n_processed % 20 == 0:
                rate = n_processed / (time.time() - t0 + 1e-9)
                logging.info(
                    f"  [{dataset}] {n_processed}/{len(test_rows)}  "
                    f"{rate:.2f} vid/s"
                )

    logging.info(
        f"[{dataset}] done K={num_shots}, scored {n_processed} new, "
        f"output {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Mod-HATE video-adapted reproduction (LLaMA-7B + 3 LoRAs)"
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[4, 8],
        help="K-shot settings to run (default: both 4 and 8)",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--lora-root",
        default=lora_compose.UPSTREAM_LORA_ROOT,
        help="Directory containing hate-exp / meme-captions / hate-speech LoRA modules",
    )
    parser.add_argument("--max-inference-step", type=int, default=40,
                        help="Nevergrad budget (upstream default 40)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Support-set loss batch size (upstream default 16)")
    parser.add_argument("--cutoff-len", type=int, default=512,
                        help="Max tokens per prompt (upstream default 256; "
                             "raised to 512 to accommodate 8-frame concat + transcript)")
    parser.add_argument("--load-8bit", action="store_true", default=True)
    parser.add_argument("--no-load-8bit", dest="load_8bit", action="store_false")
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    datasets = ALL_DATASETS if args.all else [args.dataset]
    logging.info(
        f"Mod-HATE repro: datasets={datasets}  shots={args.shots}  "
        f"base_model={args.base_model}  load_8bit={args.load_8bit}"
    )

    for ds in datasets:
        out_dir = os.path.join(PROJECT_ROOT, "results", "mod_hate", ds)
        for K in args.shots:
            run_one_dataset(ds, num_shots=K, args=args, out_dir=out_dir)

    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
