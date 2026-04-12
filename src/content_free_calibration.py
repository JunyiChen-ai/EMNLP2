"""
Content-free calibration: measure MLLM base-rate P(Yes) bias.

Runs the binary prompt with empty title, empty transcript, and a solid
black frame to measure the model's prior bias toward Yes/No.

Saves {"en_p_base": ..., "zh_p_base": ...} to results/holistic_2b/content_free.json.
"""

import json
import logging
import math
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from score_holistic_2b import (
    BINARY_PROMPT, YOUTUBE_RULES, BILIBILI_RULES,
    build_binary_token_ids, extract_binary_score,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"


def create_black_frame(path, size=224):
    """Create a solid black image as content-free visual input."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    img.save(path)
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Content-free calibration")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    args = parser.parse_args()

    model_tag = "8b" if "8B" in args.model else "2b"
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", f"content_free_calibration_{model_tag}.log")),
            logging.StreamHandler(),
        ],
    )

    from vllm import LLM, SamplingParams

    model_name = args.model
    logging.info(f"Loading model: {model_name}")

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=0.92,
        max_model_len=32768,
        limit_mm_per_prompt={"video": 1, "image": 8},
        allowed_local_media_path="/data/jehc223",
        mm_processor_kwargs={"max_pixels": 100352},
    )

    tokenizer = llm.get_tokenizer()
    label_token_ids = build_binary_token_ids(tokenizer)

    all_constrained_ids = set()
    for tids in label_token_ids.values():
        all_constrained_ids.update(tids)

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=list(all_constrained_ids),
    )

    # Create content-free black frame
    tmpdir = tempfile.mkdtemp(dir="/data/jehc223")
    black_frame_path = create_black_frame(os.path.join(tmpdir, "black.jpg"))
    logging.info(f"Created black frame at {black_frame_path}")

    results = {}

    for platform, rules_text, key in [
        ("youtube", YOUTUBE_RULES, "en_p_base"),
        ("bilibili", BILIBILI_RULES, "zh_p_base"),
    ]:
        prompt_text = BINARY_PROMPT.format(
            title="",
            transcript="",
            rules=rules_text,
        )

        content = [
            {"type": "image_url", "image_url": {"url": f"file://{black_frame_path}"}},
            {"type": "text", "text": prompt_text},
        ]

        messages = [
            {"role": "system", "content": "You are a content moderation analyst. Answer based strictly on observable evidence."},
            {"role": "user", "content": content},
        ]

        logging.info(f"\nRunning content-free call for {platform} rules...")
        outputs = llm.chat(messages=[messages], sampling_params=sampling_params)
        score = extract_binary_score(outputs[0], label_token_ids)

        logging.info(f"  {platform} P(Yes) base rate: {score:.6f}")
        results[key] = score

    # Save
    out_dir = os.path.join(PROJECT_ROOT, "results", f"holistic_{model_tag}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "content_free.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nSaved to {out_path}")
    logging.info(f"  en_p_base = {results['en_p_base']:.6f}")
    logging.info(f"  zh_p_base = {results['zh_p_base']:.6f}")

    # Cleanup
    os.remove(black_frame_path)
    os.rmdir(tmpdir)


if __name__ == "__main__":
    main()
