"""Mod-HATE LoRA composition wrapper.

Ports upstream `src/lora_learning.py` from `external_repos/mod_hate/`
to our video-adapted setting. Core logic is kept verbatim:

- `load_base_model_and_lora_modules` — verbatim upstream except for
  the base-model HF id and the LoRA-module directory layout (we read
  from `external_repos/mod_hate/LoRA_modules/<name>/` rather than
  upstream's `<lora_dir>/LoRA/<name>/`).
- `get_score` / `default_get_loss` / `default_l1_regularization` /
  `get_final_weights` — byte-for-byte upstream (`lora_learning.py:98-181`).
- `lorahub_learning` body — upstream, with the `Few_HM_Data` support
  constructor replaced by a passthrough of a pre-built list of K-shot
  support examples produced by `video_caption_adapter.build_support_and_test`.

The weighted-average LoRA merge is preserved: upstream writes the
final composed state dict via `set_peft_model_state_dict` then
`merge_and_unload()`. Nevergrad `NGOpt` optimizer, budget
`max_inference_step`, bounds `[-1.5, 1.5]` — all verbatim.

Deviations (documented):
  1. **Support / test examples come from `video_caption_adapter`**,
     not from upstream's `Few_HM_Data`. Upstream reads pickled
     `BLIP-2/results/<dataset>-generic.pkl` meme captions plus
     `domain_splits/<dataset>_<split>.json` labels; we read the
     8-frame Pro-Cap jsonl + our project's `data_utils`.
  2. **Evaluation after composition** is handled by the main driver
     (`reproduce_mod_hate.py`), not by calling upstream's
     `hfm_generation`. The rationale is the same (loss on K-shot
     support → final-weights → test-set scoring), but the test
     loader is our video loader rather than the meme loader.
  3. **`load_8bit` defaults to True** because LLaMA-7B + 3 LoRAs fits
     comfortably in int8 on a single A100, matching upstream's
     `individual_module_infer.sh` pattern. A `--no-load-8bit` flag is
     exposed by `reproduce_mod_hate.py`.
"""

import copy
import logging
import os
import random

import numpy as np
import torch


UPSTREAM_LORA_ROOT = os.path.join(
    "/data/jehc223/EMNLP2", "external_repos", "mod_hate", "LoRA_modules"
)
DEFAULT_LORA_MODULES = ("hate-exp", "meme-captions", "hate-speech")


def set_seed(seed):
    """Upstream `lora_learning.py:28-35`, verbatim."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_base_model_and_lora_modules(
    base_model,
    lora_module_list,
    load_8bit=True,
    torch_dtype=None,
):
    """Upstream `lora_learning.py:48-96`, with:
      - `args.base_model` hoisted to an explicit parameter
      - `args.load_8bit` hoisted to an explicit parameter
      - the "first module is default" pattern preserved
    """
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import LlamaForCausalLM, LlamaTokenizer

    if torch_dtype is None:
        torch_dtype = torch.float16

    default_peft_model_id = lora_module_list[0]

    base = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # Upstream pad_token_id + padding_side (lora_learning.py:216-219)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    try:
        logging.info(f"Loading default peft model: {default_peft_model_id}")
        peft_model = PeftModel.from_pretrained(
            base, default_peft_model_id, torch_dtype=torch_dtype
        )
    except Exception as e:
        raise RuntimeError(
            f"{default_peft_model_id} failed to load into {base_model}: {e}"
        )

    peft_model.eval()
    logging.info("Begin to load LoRA modules")
    cache = {}
    first_dict = None
    for peft_id in lora_module_list:
        logging.info(f"  loading {peft_id}")
        cur = PeftModel.from_pretrained(
            base, peft_id, torch_dtype=torch_dtype
        )
        cache[peft_id] = get_peft_model_state_dict(cur)
        if first_dict is None:
            first_dict = cache[peft_id]
        # Upstream arch-compatibility check (lora_learning.py:89-94)
        for key in first_dict.keys():
            assert (
                first_dict[key].shape == cache[peft_id][key].shape
            ), f"LoRA {peft_id} has mismatched shape at key {key}"
    return peft_model, tokenizer, cache


def build_lora_module_list(lora_root=UPSTREAM_LORA_ROOT, names=None):
    """Resolve the 3 pre-trained module paths under
    `external_repos/mod_hate/LoRA_modules/<name>/`.
    """
    if names is None:
        names = list(DEFAULT_LORA_MODULES)
    paths = []
    for n in names:
        p = os.path.join(lora_root, n)
        if not os.path.isdir(p):
            raise FileNotFoundError(
                f"LoRA module dir not found: {p}"
            )
        paths.append(p)
    return paths


# -------------------- upstream get_score / loss / reg --------------------
# These are byte-for-byte copies of upstream `lora_learning.py:98-161`,
# reshaped only to accept explicit args (tokenizer, model, cache,
# example_dataset, batch_size, get_loss, get_regular).


def get_score(
    weights,
    tokenizer,
    model,
    cache,
    example_dataset,
    batch_size,
    get_loss,
    get_regular,
):
    from peft import set_peft_model_state_dict

    final_state_dict = {}
    lora_module_list = list(cache.keys())
    keys = list(cache[lora_module_list[0]].keys())
    for i, peft_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    set_peft_model_state_dict(model, final_state_dict)
    loss = get_loss(tokenizer, example_dataset, model, batch_size)
    metric_val = loss + get_regular(weights)
    return metric_val


def default_get_loss(tokenizer, example_dataset, model, batch_size):
    """Upstream `lora_learning.py:126-154`, verbatim (logger calls dropped)."""
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        example_dataset, batch_size=batch_size, shuffle=True
    )
    train_loss = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for _iter, batch in enumerate(train_dataloader):
            # In modern torch, dict collate returns a plain dict. Move each
            # tensor entry onto the device individually.
            batch = {
                k: (v.to(device) if hasattr(v, "to") else v)
                for k, v in batch.items()
            }
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.detach().float()
    loss = train_loss.float()
    return float(loss) / len(example_dataset)


def default_l1_regularization(weights):
    """Upstream `lora_learning.py:156-161`, verbatim."""
    sum_of_squares = sum([abs(x) for x in weights]) / len(weights)
    return 0.05 * sum_of_squares


def get_final_weights(weights, lora_module_list, cache):
    """Upstream `lora_learning.py:168-181`, verbatim."""
    final_state_dict = {}
    keys = cache[lora_module_list[0]].keys()
    for i, peft_id in enumerate(lora_module_list):
        lora_state_dict = cache[peft_id]
        if i == 0:
            for key in keys:
                final_state_dict[key] = weights[i] * lora_state_dict[key]
        else:
            for key in keys:
                final_state_dict[key] = (
                    final_state_dict[key] + weights[i] * lora_state_dict[key]
                )
    return final_state_dict


def lorahub_learning(
    lora_module_list,
    base_model,
    support_dataset,
    max_inference_step,
    batch_size=16,
    load_8bit=True,
    seed=42,
    get_loss=default_get_loss,
    get_regular=default_l1_regularization,
):
    """Upstream `lora_learning.py:183-251`, adapted.

    Differences vs upstream:
      1. `args` bag of config is hoisted into explicit params.
      2. `Few_HM_Data(args, tokenizer)` is replaced with `support_dataset`
         — a torch `Dataset` already built by the caller from our
         video-adapted K-shot rows.
      3. `hfm_generation` call at the end is NOT performed here;
         `reproduce_mod_hate.py` handles evaluation after composition,
         because our test loader is a video loader.
    """
    import nevergrad as ng
    from functools import partial
    from peft import set_peft_model_state_dict

    set_seed(seed)
    number_of_loras = len(lora_module_list)
    if number_of_loras == 0:
        raise ValueError("No LoRA modules provided.")

    logging.info(f"Composing {number_of_loras} LoRA modules: {lora_module_list}")
    model, tokenizer, cache = load_base_model_and_lora_modules(
        base_model, lora_module_list, load_8bit=load_8bit
    )
    model_copy = copy.deepcopy(model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    get_score_partial = partial(
        get_score,
        tokenizer=tokenizer,
        model=model_copy,
        cache=cache,
        example_dataset=support_dataset,
        batch_size=batch_size,
        get_loss=get_loss,
        get_regular=get_regular,
    )

    # Upstream `lora_learning.py:230-236`, verbatim bounds + budget.
    instrum = ng.p.Array(
        init=[0] * number_of_loras,
        upper=[1.5] * number_of_loras,
        lower=[-1.5] * number_of_loras,
    )
    optimizer = ng.optimizers.NGOpt(
        parametrization=instrum, budget=max_inference_step
    )
    logging.info("Begin gradient-free LoRA composition (Nevergrad NGOpt)")
    recommendation = optimizer.minimize(get_score_partial, verbosity=1)

    final_lora = get_final_weights(
        recommendation.value, lora_module_list, cache
    )
    set_peft_model_state_dict(model_copy, final_lora)
    model_copy = model_copy.merge_and_unload()
    for i, name in enumerate(lora_module_list):
        logging.info(
            f"  composed weight {name}: {recommendation.value[i]:.4f}"
        )
    return recommendation.value, model_copy, tokenizer
