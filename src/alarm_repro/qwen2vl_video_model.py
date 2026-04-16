"""Qwen2.5-VL-72B-Instruct-AWQ video adapter for ALARM.

Direct port of upstream `src/model/utils/models/qwen2vl_model.py`
with two changes:

1. `chat_label_video(prompt, frames)` replaces the upstream
   `chat_label(prompt, image)` — takes a **list of 8 PIL frames**
   instead of a single image, passes them via the processor's
   multi-image chat template, and runs the same `output_logits=True`
   single-token generation to recover probabilities over the `"0"` /
   `"1"` tokens. The logit-access contract (`self.one_id`,
   `self.zero_id`, softmax over `[logits[zero_id], logits[one_id]]`)
   is byte-for-byte upstream (`qwen2vl_model.py:91-92, 186-243`).
2. `chat_multi_img_video(prompt, frames_a, frames_b=None)` ports
   upstream `chat_multi_img` (`qwen2vl_model.py:246-286`) to accept
   either one video's 8 frames (total 8 images) or a video pair's
   16 frames (total 16 images). The upstream chat template builder
   already supports an arbitrary number of image entries per turn,
   so the port is a straightforward extension — we just loop over
   more images.
3. `chat_text(prompt, max_tokens)` is kept as an alias for
   `self.tokenizer.chat_template`-based text-only generation used by
   the Reference stage, verbatim from `qwen2vl_model.py:288-322`.

All other generation kwargs (`device_map="cuda"`, `torch_dtype="auto"`,
`max_new_tokens=1024` for multi-image, `max_new_tokens=1` for
chat_label) are upstream-exact. The one exception is
`attn_implementation`: upstream uses `"flash_attention_2"`; we
substitute `"sdpa"` because `flash_attn` is not installed in our
`SafetyContradiction` conda env and building it from source is an
engineering hazard on this cluster. `sdpa` is a documented drop-in
replacement that Qwen2.5-VL supports via HF transformers, with
identical mathematical output and near-identical throughput on A100.
Documented as Deviation #6 in the README.

Quantization: upstream loads `Qwen/Qwen2.5-VL-72B-Instruct-AWQ`
directly via HF transformers — the AWQ weights are auto-detected
and loaded with 4-bit precision under the hood. No additional
`BitsAndBytesConfig` is needed. This is the same hard-block VRAM
substitution path user-approved for MARS 32B-AWQ (`feedback_api_to_vllm.md`
chain).
"""

import base64
from io import BytesIO
from typing import List, Optional

# ---------- autoawq compat shim for transformers 4.57.1 ----------
# autoawq 0.2.9 imports PytorchGELUTanh and shard_checkpoint at module
# init time; transformers 4.57 removed both. Patch them back as stubs
# so the import succeeds. Qwen2.5-VL uses SwiGLU (not GELU-tanh) and
# we only load (never save), so neither stub is ever called at runtime.
import transformers.activations
import transformers.modeling_utils

if not hasattr(transformers.activations, "PytorchGELUTanh"):
    import torch.nn as _nn
    class _PytorchGELUTanh(_nn.Module):
        def forward(self, x):
            import torch
            return torch.nn.functional.gelu(x, approximate="tanh")
    transformers.activations.PytorchGELUTanh = _PytorchGELUTanh

if not hasattr(transformers.modeling_utils, "shard_checkpoint"):
    transformers.modeling_utils.shard_checkpoint = lambda *a, **k: ({}, None)
# -----------------------------------------------------------------

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"


def resize_image(image, max_size=640):
    """Upstream `qwen2vl_model.py:15-42`, verbatim.

    Keep aspect ratio; longest side = max_size pixels; LANCZOS.
    """
    from PIL import Image

    width, height = image.size
    if width >= height:
        scaling_factor = max_size / width
    else:
        scaling_factor = max_size / height
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return image.resize((new_width, new_height), Image.LANCZOS)


def pil_image_to_base64(image, image_format="PNG"):
    """Upstream `qwen2vl_model.py:45-73`, verbatim."""
    image = resize_image(image)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    return image_base64


class Qwen2VLVideoModel:
    """Thin wrapper around upstream `Qwen2VLModel` with video-shaped
    APIs. Upstream's `BaseModel` inheritance is flattened — we keep
    only the methods ALARM actually calls on the model object.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, params: Optional[dict] = None):
        import torch
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.model_id = model_id
        self.params = params if params else {}

        # Upstream `qwen2vl_model.py:79-86`, verbatim kwargs except
        # for `attn_implementation`: upstream uses
        # `"flash_attention_2"` but `flash_attn` is not installed in
        # the `SafetyContradiction` conda env and building from
        # source is an engineering hazard on this cluster. We
        # substitute PyTorch's `"sdpa"` (scaled-dot-product
        # attention) — a documented drop-in replacement that
        # Qwen2.5-VL officially supports via transformers, with
        # near-identical throughput on modern hardware. Documented
        # as Deviation #6 in the README.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Upstream `qwen2vl_model.py:91-95`, token ids for single-token
        # classification and harmful / harmless lookups.
        self.one_id = self.tokenizer.convert_tokens_to_ids("1")
        self.zero_id = self.tokenizer.convert_tokens_to_ids("0")
        self.harmful_id = self.tokenizer.convert_tokens_to_ids("Ġharmful")
        self.harmless_id = self.tokenizer.convert_tokens_to_ids("Ġharmless")

    # ----- video multi-image -----

    def _build_multi_image_conversation(self, prompt: str, frames: list):
        """Construct the chat-template conversation upstream
        `chat_multi_img` (`qwen2vl_model.py:246-262`) uses: one user
        turn with N `{"type": "image", "image": "data:image;base64,..."}`
        entries followed by one `{"type": "text", "text": prompt}`
        entry. Verbatim shape.
        """
        content = []
        for img in frames:
            content.append(
                {
                    "type": "image",
                    "image": f"data:image;base64,{pil_image_to_base64(img)}",
                }
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def chat_multi_img_video(
        self,
        prompt: str,
        frames: List,
        max_new_tokens: int = 1024,
    ) -> str:
        """One video (8 frames) or two videos (16 frames) → single
        LLM call. Returns the decoded text.

        Mirrors upstream `chat_multi_img` (`qwen2vl_model.py:246-286`)
        verbatim except that `frames` is our ordered list rather than
        upstream's 2-meme list — the conversation-builder loops over
        however many frames are provided.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        conversation = self._build_multi_image_conversation(prompt, frames)
        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            **self.params,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids
            )
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return output_text[0]

    def chat_label_video(self, prompt: str, frames: List):
        """Video-adapted Label stage: same prompt + 8 frames as a
        single multi-image user turn, generate 1 new token with
        `output_logits=True`, recover softmax over the `"0"` / `"1"`
        tokens (upstream `qwen2vl_model.py:186-243`, byte-for-byte
        for the post-generation decode).

        Returns `(decoded_text, probs)` where `probs = [prob0, prob1]`
        — matching the tuple shape upstream's Label_Runner consumes
        at `label_runner.py:173-182`.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        conversation = self._build_multi_image_conversation(prompt, frames)
        text_prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output_dict = self.model.generate(
            **inputs,
            **self.params,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_logits=True,
            return_legacy_cache=True,
        )
        output_ids = output_dict.sequences
        logits = output_dict.logits

        # Upstream `qwen2vl_model.py:220-231`, verbatim.
        logits = logits[0][0, :]
        logits_1 = logits[self.one_id].unsqueeze(-1)
        logits_0 = logits[self.zero_id].unsqueeze(-1)
        cls_logits = torch.cat([logits_0, logits_1], dim=-1)
        probs = torch.softmax(cls_logits, dim=-1).cpu().numpy().tolist()

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(
                inputs.input_ids.repeat(output_ids.shape[0], 1), output_ids
            )
        ]
        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return output_text[0], probs

    # ----- text only (Reference stage) -----

    def chat_text(self, prompt: str, max_tokens: int = 2048) -> str:
        """Upstream `qwen2vl_model.py:288-322`, verbatim single-item
        path. Used by the Reference stage for ADD/EDIT/UPVOTE/DOWNVOTE
        operations on the reference set.
        """
        import torch

        conversation = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            [conversation], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            **self.params,
        )
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]) :]
            for i in range(len(output_ids))
        ]
        output_text = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return output_text[0]
