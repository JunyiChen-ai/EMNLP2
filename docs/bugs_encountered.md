# Bugs encountered during baseline reproduction (2026-04-15 ~ 2026-04-16)

## Bug 1: vLLM 0.11.0 `llm.chat()` hangs on Qwen2.5-VL multi-image input

**Affected**: MARS reproduction (Qwen2.5-VL-32B-AWQ and Qwen2.5-VL-7B)

**Environment**:
- vLLM 0.11.0
- Qwen2.5-VL-32B-Instruct-AWQ or Qwen2.5-VL-7B-Instruct
- SafetyContradiction conda env (Python 3.11, CUDA, A100 80GB)

**Reproduction**:
```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-VL-7B-Instruct",  # or 32B-AWQ
    max_model_len=65536,
    gpu_memory_utilization=0.92,
    limit_mm_per_prompt={"image": 16, "video": 1},
    mm_processor_kwargs={"max_pixels": 32768},
    trust_remote_code=True,
    enforce_eager=True,
)
sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=4096)

# 16 frames as image_url content in a chat message
media_content = [
    {"type": "image_url", "image_url": {"url": f"file:///path/to/frame_{i:03d}.jpg"}}
    for i in range(16)
]
msg = [{"role": "user", "content": media_content + [{"type": "text", "text": "Describe this video."}]}]

out = llm.chat(messages=[msg], sampling_params=sampling)
# ^^^ HANGS HERE INDEFINITELY
```

**Symptom**:
- vLLM EngineCore process pegged at 99% CPU
- Python main process sleeps
- stderr shows `Adding requests: 100%` then `Processed prompts: 0%` — never advances
- No output, no error, no timeout — runs until scancel
- Tested for 2+ hours, zero tokens generated

**What was ruled out**:
- Not AWQ-specific: 7B bf16 (no quantization) also hangs
- Not max_pixels: reduced from 100352 → 32768, still hangs
- Not torch.compile: `enforce_eager=True` still hangs
- Not video_url vs image_url: passing `file://` paths to pre-extracted frames (not mp4) still hangs

**Key contrast**: `llm.generate()` with explicit `multi_modal_data={"image": pil_list}` works fine on the SAME model (Qwen2-VL-7B) with SAME 16 frames. This is how Pro-Cap V3 captioner runs successfully:
```python
outputs = llm.generate(
    {"prompt": prompt_text, "multi_modal_data": {"image": pil_frames}},
    sampling_params=sampling,
)
# ^^^ WORKS FINE, ~1 sec per call
```

**Root cause hypothesis**: `llm.chat()` triggers a different internal image loading / tokenization path in vLLM 0.11.0 that hangs on Qwen2.5-VL with multi-image `image_url` content. The `llm.generate()` path bypasses this by passing PIL images directly.

**Additional note**: Qwen3-VL-2B via `llm.chat()` with the same 16-image pattern works fine (existing mars_2b pilot). So the bug is specific to Qwen2.5-VL family, not all Qwen VL models.

**Workaround**: Use `llm.generate()` with explicit `multi_modal_data` instead of `llm.chat()`.

**Impact**: MARS 32B-AWQ reproduction failed; fell back to existing MARS 2B pilot (Qwen3-VL-2B).

---

## Bug 2: LLaVA-Next (llava-v1.6-34b-hf) multi-image anyres patch mismatch

**Affected**: LoReHM reproduction

**Environment**:
- transformers 4.57.1
- `llava-hf/llava-v1.6-34b-hf` via `LlavaNextForConditionalGeneration`
- 16 PIL images passed as multi-image input

**Reproduction**:
```python
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-34b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")

# Build prompt with 16 image tokens
content = [{"type": "image"} for _ in range(16)]
content.append({"type": "text", "text": "Is this video harmful?"})
conversation = [{"role": "user", "content": content}]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# 16 images pre-resized to 336x336
images = [Image.open(f"frame_{i:03d}.jpg").convert("RGB").resize((336, 336)) for i in range(16)]
inputs = processor(images=images, text=prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=1024)
```

**Symptoms** (depending on which workaround attempted):

### Attempt 1: `image_grid_pinpoints = None`
```
ValueError: grid_pinpoints must be a list of possible resolutions.
```

### Attempt 2: `image_grid_pinpoints = []`
```
TypeError: cannot unpack non-iterable NoneType object
```

### Attempt 3: `image_grid_pinpoints = [[336, 336]]`
```
RuntimeError: split_with_sizes expects split_sizes to sum exactly to 32
(input tensor's size at dimension 0), but got split_sizes=[3, 3, 3, 3,
3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
```
- 16 images × 3 patches each (1 base + 2 sub-tiles from closest pinpoint) = 48
- But image encoder produces 32 features (16 × 2)
- Mismatch crashes at `split_with_sizes`

### Attempt 4: Default pinpoints (no override)
Same `split_with_sizes` error with different numbers depending on image resolution.

**Root cause**: LLaVA-Next is fundamentally a **single-image anyres model**. Its image processor always applies anyres tiling (1 base tile + N sub-tiles based on closest `image_grid_pinpoints` match). With multiple images, the total patch count becomes `N_images × (1 + num_sub_tiles)` which doesn't match the `N_images` `<image>` tokens in the prompt. The model's forward pass expects these counts to align and crashes on the split.

There is no configuration of `image_grid_pinpoints` that cleanly maps to "1 tile per image" because:
- `None` → type error (must be list)
- `[]` → internal NoneType unpack
- `[[H, W]]` → still produces base + sub tiles
- Default → larger mismatch

**Conclusion**: `LlavaNextForConditionalGeneration` in transformers 4.57.1 does NOT support multi-image input. It's architecturally single-image with anyres expansion.

**Workaround**: Switched backbone to Qwen2-VL-7B-Instruct via vLLM, which natively supports multi-image.

**Impact**: LoReHM backbone changed from LLaVA-v1.6-34b to Qwen2-VL-7B (different model family + size).

---

## Bug 3: autoawq incompatible with transformers 4.57.1

**Affected**: ALARM reproduction (Qwen2.5-VL-72B-Instruct-AWQ)

**Environment**:
- transformers 4.57.1
- autoawq 0.2.5 or 0.2.9

**Symptom**:
```
# autoawq 0.2.9:
ImportError: cannot import name 'PytorchGELUTanh' from 'transformers.activations'

# autoawq 0.2.5:
ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils'
```

**Root cause**: transformers 4.57 removed `PytorchGELUTanh` from `activations.py` and `shard_checkpoint` from `modeling_utils.py`. autoawq imports these at module init time, crashing before any model load.

**Workaround**: Switched ALARM backbone from Qwen2.5-VL-72B-AWQ to Qwen2.5-VL-7B-Instruct (non-quantized bf16, fits on 1 A100).

**Impact**: ALARM backbone 72B → 7B.

---

## Bug 4: MATCH stage 3 uses bert-base-uncased for MHClip_ZH (should be bert-base-chinese)

**Affected**: MATCH-HVD stage 3 supervised training on MHClip_ZH

**Environment**:
- `src/match_repro/stage3/train_match.py`
- `DEFAULT_TEXT_ENCODER = "bert-base-uncased"` applied to ALL datasets

**Root cause**: Our `train_match.py` hardcodes `DEFAULT_TEXT_ENCODER = "bert-base-uncased"` and applies it uniformly to all 4 datasets. But upstream Hydra configs use per-dataset text encoders:

```
external_repos/match_hvd/src/config/HateMM_MATCH.yaml   → bert-base-uncased   ✓
external_repos/match_hvd/src/config/MHClipEN_MATCH.yaml  → bert-base-uncased   ✓
external_repos/match_hvd/src/config/MHClipZH_MATCH.yaml  → bert-base-chinese   ✗ MISMATCH
```

MHClip_ZH's 4 text streams (transcript, judge, hate agent, nonhate agent) contain Chinese text. `bert-base-uncased` tokenizer maps Chinese characters to `[UNK]` → BERT CLS embeddings carry no semantic information → classifier collapses to majority class.

**Symptom**: MHClip_ZH test results: acc=0.6779, macro_F1=0.4409. The mF1 near 0.44 is characteristic of majority-class prediction on a ~68/32 class split.

**Fix needed**: In `train_match.py`, use per-dataset text encoder dispatch:
```python
DATASET_TEXT_ENCODERS = {
    "MHClip_EN":    "google-bert/bert-base-uncased",
    "MHClip_ZH":    "google-bert/bert-base-chinese",
    "HateMM":       "google-bert/bert-base-uncased",
    "ImpliHateVid": "google-bert/bert-base-uncased",
}
```
Then rerun stage 3 for MHClip_ZH only.

**Impact**: MHClip_ZH MATCH accuracy artificially low. EN/HateMM/ImpliHateVid unaffected (already use correct encoder).

---

## Bug 5: vLLM 0.11.0 does not support video content type for LLaVA-Next and Gemma-3

**Affected**: Naive baselines — `llava-hf/llava-v1.6-mistral-7b-hf` and `google/gemma-3-12b-it`

**Environment**:
- vLLM 0.11.0
- SafetyContradiction conda env
- Both models loaded via `LLM()` with video input

**Reproduction**:
```python
from vllm import LLM

llm = LLM(model="llava-hf/llava-v1.6-mistral-7b-hf", ...)
# OR
llm = LLM(model="google/gemma-3-12b-it", ...)

# Pass video content:
content = [{"type": "video", "video": {"path": "/path/to/video.mp4"}}]
messages = [{"role": "user", "content": content + [{"type": "text", "text": "..."}]}]
out = llm.chat(messages=messages, ...)
```

**Symptom**:
```
At most 0 video(s) may be provided in one prompt.
```
Every single video in all 4 datasets produces this error. 100% failure rate — zero valid predictions.

**Root cause**: vLLM 0.11.0's model registry for LLaVA-Next-Mistral-7B and Gemma-3-12B does not register video as a supported modality. These models' vLLM adapters only support `image` content type, not `video`. The `limit_mm_per_prompt` for video is hard-capped at 0 by the model config.

**Workaround (not yet implemented)**: Frame-based fallback — extract N frames from each video as JPGs, pass as multi-image input instead of video. Requires rewriting the naive baseline script to use `{"type": "image"}` content entries instead of `{"type": "video"}`. Deprioritized per user directive (ablation sidebar, not headline comparison).

**Impact**: LLaVA-Next-Mistral-7B and Gemma-3-12B naive baselines are completely missing from the results table. Only Qwen3-VL-2B and InternVL3-8B naive baselines have valid results (these models support video natively in vLLM).

