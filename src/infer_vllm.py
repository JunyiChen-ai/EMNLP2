"""
vLLM inference skeleton for hateful video classification.

Runs a frozen VLM on pre-extracted video frames + metadata,
returns per-sample results with token-level logprobs.

Usage:
  python src/infer_vllm.py \
      --dataset HateMM --split test \
      --model Qwen/Qwen2-VL-7B-Instruct \
      --prompt-template prompt.txt \
      --num-frames 32 --batch-size 4
"""

import argparse
import json
import math
import os
import time
import numpy as np

# ── Dataset configs ──────────────────────────────────────────────────

DATASET_CONFIGS = {
    "HateMM": {
        "annotation": "datasets/HateMM/annotation(new).json",
        "splits_dir": "datasets/HateMM/splits",
        "frames_dir": "datasets/HateMM/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hate": 1, "Non Hate": 0},
    },
    "MHClip_EN": {
        "annotation": "datasets/MHClip_EN/annotation(new).json",
        "splits_dir": "datasets/MHClip_EN/splits",
        "frames_dir": "datasets/MHClip_EN/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
    "MHClip_ZH": {
        "annotation": "datasets/MHClip_ZH/annotation(new).json",
        "splits_dir": "datasets/MHClip_ZH/splits",
        "frames_dir": "datasets/MHClip_ZH/frames",
        "id_field": "Video_ID",
        "title_field": "Title",
        "transcript_field": "Transcript",
        "label_field": "Label",
        "label_map": {"Hateful": 1, "Offensive": 1, "Normal": 0},
    },
}

SYSTEM_PROMPT = "You are a content moderation expert. Classify video content precisely."

DEFAULT_USER_PROMPT = """You are given {num_frames} frames uniformly sampled from a video, along with its title and transcript.

Title: {title}
Transcript: {transcript}

Classify this video into ONE of three categories:
- hateful: targets a specific identity group with hate, dehumanization, or incitement
- offensive: rude, vulgar, aggressive, but does NOT target a specific identity group
- benign: neither hateful nor offensive

Answer with exactly one word: hateful, offensive, or benign."""


# ── Helpers ──────────────────────────────────────────────────────────

def load_split_ids(splits_dir, split_name, project_root):
    path = os.path.join(project_root, splits_dir, f"{split_name}.csv")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_messages(sample, cfg, project_root, user_prompt, max_frames=32):
    """Build vLLM chat messages with interleaved frame images."""
    vid_id = sample[cfg["id_field"]]
    title = sample.get(cfg["title_field"], "") or ""
    transcript = sample.get(cfg["transcript_field"], "") or ""
    if len(transcript) > 1500:
        transcript = transcript[:1500] + "..."

    frames_dir = os.path.join(project_root, cfg["frames_dir"], vid_id)
    if not os.path.isdir(frames_dir):
        return None

    frame_files = sorted(
        f for f in os.listdir(frames_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    )
    if not frame_files:
        return None

    # Uniform subsample if more frames than budget
    if len(frame_files) > max_frames:
        indices = np.linspace(0, len(frame_files) - 1, max_frames).round().astype(int)
        # deduplicate while preserving order
        seen = set()
        unique = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                unique.append(i)
        frame_files = [frame_files[i] for i in unique]

    text = user_prompt.format(
        title=title, transcript=transcript, num_frames=len(frame_files),
    )

    user_content = []
    for ff in frame_files:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"file://{os.path.join(frames_dir, ff)}"},
        })
    user_content.append({"type": "text", "text": text})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def extract_token_probs(output, label_token_ids):
    """Extract label probabilities from position-0 logprob distribution.

    Args:
        output: single vLLM RequestOutput
        label_token_ids: dict[label_name -> list[int]]  candidate first-token IDs

    Returns:
        dict with per-label probabilities, predicted label, confidence, margin,
        entropy, and raw logprobs.  None on failure.
    """
    if not output or not output.outputs:
        return None
    gen = output.outputs[0]
    if not gen.logprobs or len(gen.logprobs) == 0:
        return None

    pos0 = gen.logprobs[0]  # dict[token_id -> Logprob]

    FALLBACK = -30.0
    raw = {}
    for label, tids in label_token_ids.items():
        best = FALLBACK
        for tid in tids:
            if tid in pos0:
                best = max(best, pos0[tid].logprob)
        raw[label] = best

    # softmax
    labels = list(raw.keys())
    lps = [raw[l] for l in labels]
    mx = max(lps)
    exps = [math.exp(lp - mx) for lp in lps]
    total = sum(exps)
    if total <= 0:
        return None
    probs = {l: e / total for l, e in zip(labels, exps)}

    sorted_p = sorted(probs.values(), reverse=True)
    predicted = max(probs, key=probs.get)

    return {
        "predicted": predicted,
        "probs": probs,
        "confidence": sorted_p[0],
        "margin": sorted_p[0] - sorted_p[1],
        "entropy": -sum(p * math.log(p + 1e-12) for p in probs.values()),
        "raw_logprobs": raw,
        "in_top_k": {l: raw[l] > FALLBACK for l in labels},
        "generated_text": gen.text.strip(),
    }


def build_label_token_ids(tokenizer, label_names):
    """Map each label name to a set of candidate first-token IDs
    (covers BPE variants with/without leading space, capitalisation)."""
    def first_tok(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        return ids[0] if ids else None

    mapping = {}
    for name in label_names:
        tids = set()
        for variant in [name, f" {name}", name.capitalize(), f" {name.capitalize()}"]:
            tid = first_tok(variant)
            if tid is not None:
                tids.add(tid)
        mapping[name] = list(tids)
        decoded = [tokenizer.decode([t]) for t in mapping[name]]
        print(f"  label '{name}' -> token IDs {mapping[name]} ({decoded})")
    return mapping


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vLLM inference on video frames")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    parser.add_argument("--split", default="train",
                        choices=["train", "valid", "test", "all"])
    parser.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor-parallel size")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-logprobs", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=3)
    parser.add_argument("--labels", nargs="+",
                        default=["hateful", "offensive", "benign"],
                        help="Label names whose token logprobs to extract")
    parser.add_argument("--prompt-template", default=None,
                        help="Path to a text file with the user prompt "
                             "(use {title}, {transcript}, {num_frames} placeholders)")
    parser.add_argument("--project-root", default="/data/jehc223/EMNLP2")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = DATASET_CONFIGS[args.dataset]
    root = args.project_root

    # Load user prompt
    if args.prompt_template and os.path.isfile(args.prompt_template):
        with open(args.prompt_template) as f:
            user_prompt = f.read()
    else:
        user_prompt = DEFAULT_USER_PROMPT

    # Load annotations
    with open(os.path.join(root, cfg["annotation"])) as f:
        annotations = json.load(f)
    id2sample = {s[cfg["id_field"]]: s for s in annotations}

    # Load split
    if args.split == "all":
        split_ids = list(id2sample.keys())
    else:
        split_ids = load_split_ids(cfg["splits_dir"], args.split, root)
    print(f"Dataset={args.dataset}  split={args.split}  n={len(split_ids)}")

    # Output
    if args.output_dir is None:
        args.output_dir = os.path.join(root, "results", "infer", args.dataset)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.split}.jsonl")

    # Resume
    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    try:
                        done_ids.add(json.loads(line)["video_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    todo = [v for v in split_ids if v in id2sample and v not in done_ids]
    if not todo:
        print("All done.")
        return
    print(f"Resume: {len(done_ids)} done, {len(todo)} remaining")

    # Init vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        allowed_local_media_path="/data/jehc223",
        limit_mm_per_prompt={"image": args.num_frames},
        max_logprobs=max(20, args.num_logprobs),
    )
    tokenizer = llm.get_tokenizer()
    label_token_ids = build_label_token_ids(tokenizer, args.labels)

    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        logprobs=args.num_logprobs,
    )

    # Inference loop with adaptive OOM handling
    bs = args.batch_size
    max_pixels = 73728
    mm_kwargs = {"max_pixels": max_pixels}
    t0 = time.time()
    idx = 0
    n_done = 0

    while idx < len(todo):
        batch_ids = todo[idx : idx + bs]
        convs, meta = [], []
        for vid_id in batch_ids:
            msgs = build_messages(
                id2sample[vid_id], cfg, root, user_prompt, args.num_frames,
            )
            if msgs is not None:
                convs.append(msgs)
                gt = cfg["label_map"].get(
                    id2sample[vid_id].get(cfg["label_field"], ""), -1,
                )
                meta.append((vid_id, gt))

        if not convs:
            idx += len(batch_ids)
            continue

        try:
            outputs = llm.chat(
                messages=convs,
                sampling_params=sampling,
                mm_processor_kwargs=mm_kwargs,
            )
        except Exception as e:
            err = str(e).lower()
            if any(k in err for k in ("out of memory", "oom", "cuda")):
                if bs > 1:
                    bs = max(1, bs // 2)
                    print(f"  OOM → bs={bs}")
                    try:
                        import torch; torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                else:
                    mm_kwargs["max_pixels"] = max(
                        160 * 200, mm_kwargs["max_pixels"] // 2
                    )
                    print(f"  OOM bs=1 → pixels={mm_kwargs['max_pixels']}")
                    continue
            else:
                print(f"  Error: {e}")
                idx += len(batch_ids)
                continue

        # Write results
        with open(out_path, "a") as f:
            for i, (vid_id, gt) in enumerate(meta):
                out = outputs[i] if i < len(outputs) else None
                ext = extract_token_probs(out, label_token_ids)
                rec = {"video_id": vid_id, "gt_label": gt}
                if ext is None:
                    rec["success"] = False
                    rec["generated_text"] = (
                        out.outputs[0].text if out and out.outputs else ""
                    )
                else:
                    rec["success"] = True
                    rec.update(ext)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        n_done += len(meta)
        idx += len(batch_ids)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - n_done) / rate if rate > 0 else 0
        print(f"  [{n_done}/{len(todo)}] bs={bs}  "
              f"{rate:.1f} sam/s  ETA {eta/60:.0f}m")

    # Summary
    print(f"\nResults → {out_path}")
    recs = []
    with open(out_path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
    n_ok = sum(r.get("success", False) for r in recs)
    print(f"Total={len(recs)}  success={n_ok}  fail={len(recs)-n_ok}")


if __name__ == "__main__":
    main()
