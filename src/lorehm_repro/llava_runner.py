"""LoReHM multi-image runner — rewritten 2026-04-16 to use Qwen2-VL-7B.

Background: the original plan was to use `llava-hf/llava-v1.6-34b-hf`
as the LoReHM backbone at bf16 across 2 GPUs, with 16 multi-image
inputs per call. Multiple attempts (grid_pinpoints=None, =[], =[[336,336]],
pre-resized inputs, manual image_sizes) all failed because LLaVA-Next is
fundamentally a single-image anyres model — its processor always
produces a mismatched number of image features relative to the 16
`<image>` tokens in the prompt (split_sizes errors, grid_pinpoints
errors, unpack-None errors).

Pragmatic pivot: switch the backbone to **Qwen/Qwen2-VL-7B-Instruct**
via vLLM. Qwen2-VL natively supports multi-image input (Pro-Cap V3's
captioner and MATCH stage 2c use the same model successfully), greedy
decoding is fast, and the chat template expands image placeholders
correctly. This is a documented fidelity deviation from upstream
LoReHM (which used `llava-v1.6-34b` for meme images) and from our own
earlier plan. The RSA / MIA / parse_answer / control flow are
unchanged.

Generation: temperature=0 (greedy), max_tokens=1024.
"""

import logging
import os
from typing import List

DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
NUM_FRAMES = 16
MAX_TOKENS_DEFAULT = 1024


class LlavaRunner:
    """Qwen2-VL-7B multi-image runner (kept class name for callsite
    compatibility with reproduce_lorehm.py).

    Usage:
        runner = LlavaRunner()
        args = _Args(query=prompt, image_file=list_of_16_pil_images,
                     max_new_tokens=1024)
        response_text = runner.run_model(args)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        max_memory_per_gpu: str = "75GiB",  # unused; kept for CLI compat
        cpu_memory: str = "64GiB",           # unused; kept for CLI compat
    ):
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.model_id = model_id
        logging.info(f"Loading {model_id} via vLLM")
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.llm = LLM(
            model=model_id,
            max_model_len=16384,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": NUM_FRAMES},
            mm_processor_kwargs={"max_pixels": 200704},
            trust_remote_code=True,
        )
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS_DEFAULT,
        )
        logging.info("Qwen2-VL-7B loaded")

    def _coerce_image_list(self, image_file) -> List:
        from PIL import Image
        if isinstance(image_file, (list, tuple)):
            images = list(image_file)
        elif hasattr(image_file, "convert"):
            images = [image_file]
        elif isinstance(image_file, str):
            images = [Image.open(image_file)]
        else:
            raise TypeError(
                f"image_file must be list[PIL.Image], PIL.Image, or "
                f"path str; got {type(image_file)}"
            )
        out = []
        for im in images:
            if isinstance(im, str):
                im = Image.open(im)
            out.append(im.convert("RGB"))
        return out

    def run_model(self, args) -> str:
        image_list = self._coerce_image_list(args.image_file)
        n_img = len(image_list)

        content = [{"type": "image"} for _ in range(n_img)]
        content.append({"type": "text", "text": args.query})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        max_tokens = getattr(args, "max_new_tokens", MAX_TOKENS_DEFAULT)
        sampling = self.sampling
        if max_tokens != MAX_TOKENS_DEFAULT:
            from vllm import SamplingParams
            sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        outputs = self.llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image_list},
            },
            sampling_params=sampling,
        )
        return outputs[0].outputs[0].text.strip()
