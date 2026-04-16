"""ALARM 5-stage pipeline ports, video-adapted.

Each function below ports one of the upstream stages from
`external_repos/alarm/src/model/`:

  * `run_label(...)`         → `Label/label_runner.py`
  * `run_make_embeddings(...)` → `Experience/make_embeddings.py`
  * `run_conduct_retrieval(...)` → `Experience/conduct_retrieval.py`
  * `run_experience(...)`    → `Experience/experience_runner.py`
  * `run_reference(...)`     → `Reference/reference_runner.py`
  * `run_inpredict(...)`     → `InPredict/inpredict_runner.py`

Upstream prompt strings are reproduced byte-for-byte except for the
meme→video rewrite rule from `feedback_meme_to_video_8frames.md`.
The only other changes are:
  * `chat_label(prompt, image)` → `chat_label_video(prompt, frames_8)`
  * `chat_multi_img(prompt, [img1, img2])` → `chat_multi_img_video(prompt, frames_16)`
  * `chat_img(prompt, image)` → `chat_multi_img_video(prompt, frames_8)`
    (upstream Label and InPredict use a single image; we substitute
    the 8-frame multi-image call)

Retrieval features use **Jina-CLIP-v2** (`jinaai/jina-clip-v2`),
matching upstream `make_embeddings.py:11` exactly. Upstream encodes
one image + text per meme; we encode 8 frames + text per video and
mean-pool the 8 frame embeddings for the image side. This is the
straightforward 8-frame extension and is documented as a deviation.
"""

import json
import logging
import os
import re
from typing import Dict, List

import numpy as np


# ======================================================================
# Prompts — meme → video rewrites of upstream strings. Every upstream
# instruction other than the meme→video substitution is byte-for-byte.
# ======================================================================

# Upstream `label_runner.py:164-169`, verbatim meme→video rewrite.
LABEL_PROMPT_VIDEO = """
Given the video with these 8 frames uniformly sampled from it and the textual element '{text}' accompanying the video, your task is to carefully and critically assess whether this video is harmful or not, in order to maintain the benignness and integrity of information on the Internet.
Please leverage your extensive knowledge to deeply analyze and understand this video, and give your final judgment.
Please not that while your primary goal is to provide a judgment after thoughtful analysis, it's important to avoid overgeneralizing or being overly conclusive in cases where ambiguity exists.
You must only return `1' for harmful, or `0' for benign.
            """

# Upstream `experience_runner.py:148-155`, rewritten to reference two
# videos (A = first 8 frames, B = last 8 frames) instead of two memes.
EXPERIENCE_PROMPT_VIDEO = """
Video A (frames 1-8) Text: {text1}
Video B (frames 9-16) Text: {text2}
Given two videos that are visually or structurally similar but belong to distinct categories: Video i, which is harmful, and Video j, which is benign. The input contains 16 frames total: frames 1-8 are from Video A, frames 9-16 are from Video B. Please complete the following two steps:
Step 1:
Clearly summarize the content of each video by carefully analyzing its 8 sampled frames and textual element accompanying the video, and considering any implicit or explicit messages it conveys.
Step 2:
Based on the content of two videos, contrast the key differences between them to explain why Video i is classified as harmful content, while Video j remains benign.
    """

# Upstream `reference_runner.py:16-57`, meme→video rewrite applied
# only where upstream explicitly says "meme". The JSON-ops schema
# and the ADD/EDIT/UPVOTE/DOWNVOTE rules are byte-for-byte preserved.
REFERENCE_PROMPT_VIDEO = """
You have a set of experiences for identifying harmful videos, originally created by comparing similar but contradictory categories of two videos.
Now, a new experience arrives containing the description of one harmful and one similar but benign one, and a summary of the differences between the them.
Your task is to distill new references from the experience and update the existing references by choosing one operation: : ADD, EDIT, UPVOTE, and DOWNVOTE.

Strict Rules:

1. ADD only if:
   - add new references that are very different from exisitng references and relevatnt for other detection.
2. EDIT must:
   - if any existing reference is not general enough or can be enhanced, rewrite and improve it.
3. UPVOTE if:
   - if the existing reference is strongly relevant for current reference
4. DOWNVOTE if:
   - if one exisiing reference is contradictory or similar/duplicated to other existing reference.
5. Maximum {size} references preserved
6. Output only valid JSON

Context:
Current references Set (importance order):
{cur_set_str}

New Coming Experience:
{new_experience}

Processing Steps:

1. Ensure the added and edited references are concise, clear while keeping them 2 or 3 sentences.
2. Ensure the references are concise and easy to follow.
3. Actively downvote references that are vague or hard to understand, and maintain the reference set at {size} items.
4. Try to make every reference useful, make more upvotes, and downvotes.
5. Refine references to emphasize distinct signals that uniquely identify specific harmful patterns.
6. Generalize references to extract universal principles that capture common traits of harmful content.
7. Return only the JSON operations with the below format:
[
  {{
    "operation": "<ADD|EDIT|UPVOTE|DOWNVOTE>",
    "reference": "<index/none>",
    "insight": "<new/revised text>"
  }}
]
"""

# Upstream `inpredict_runner.py:21-28`, meme→video rewrite.
INPREDICT_PROMPT_VIDEO = """
Given a video with 8 frames uniformly sampled from it and textual element '{text}' accompanying the video, your task is to determine whether this video is harmful or benign by considering its multimodal content.
Moreover, a set of valuable detection references are also provided to serve as additional knowledge, which will help you in determine the class of the given video.
References set:
{reference_set}.
Please leverage your pre-trained knowledge and the references to analyze and understand this video, and give your final judgment.
Your output should strictly follow the format: Thought: [Your analysis] Answer: [harmful/harmless].
"""


# ======================================================================
# Stage 1 — Label
# ======================================================================


def run_label(model, items: List[Dict], out_path: str):
    """Port of `label_runner.py:122-198`.

    For each item (video), call `chat_label_video(prompt, frames)`
    and write `{id, pred, label, prob0, prob1}` to `out_path` jsonl.
    Resume-skip on already-present ids.
    """
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("id"):
                        done.add(r["id"])
                except Exception:
                    pass
    logging.info(
        f"Label stage: {len(items)} items, {len(done)} already done"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        for item in items:
            if item["id"] in done:
                continue
            prompt = LABEL_PROMPT_VIDEO.format(text=item["text"] or "")
            try:
                text, probs = model.chat_label_video(prompt, item["frames"])
            except Exception as e:
                logging.error(f"  {item['id']}: label failed: {e}")
                text, probs = "", [0.5, 0.5]
            pred = 1 if probs[1] >= 0.5 else 0
            rec = {
                "id": item["id"],
                "pred": int(pred),
                "label": int(item["label"]),
                "prob0": float(probs[0]),
                "prob1": float(probs[1]),
                "output_text": text,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


# ======================================================================
# Stage 2 — Jina-CLIP embeddings
# ======================================================================


def _load_jina_clip():
    """Upstream `make_embeddings.py:11`, verbatim loader.

    `SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True,
    truncate_dim=512, device='cuda')`. Lazy import so syntax checks
    don't require torch.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(
        "jinaai/jina-clip-v2",
        trust_remote_code=True,
        truncate_dim=512,
        device=device,
    )


def run_make_embeddings(items: List[Dict], out_dir: str):
    """Port of `make_embeddings.py`.

    For each video, Jina-CLIP encodes:
      * text → 512-d vector (upstream `make_embeddings.py:53`)
      * each of the 8 frames → 512-d, **mean-pooled** across frames
        → 512-d video image embedding (our video adaptation; upstream
        uses one meme image → one 512-d)

    Stores `image_embed.pt` / `text_embed.pt` / `joint_embed.pt`
    under `out_dir`, each as `{id -> torch.Tensor}`.
    """
    import torch

    model = _load_jina_clip()
    os.makedirs(out_dir, exist_ok=True)
    image_fea, text_fea, joint_fea = {}, {}, {}

    for item in items:
        # Per-item flush-as-you-go keeps failure damage bounded.
        frames = item["frames"]
        if frames is None:
            continue
        with torch.no_grad():
            frame_embs = model.encode(
                frames, normalize_embeddings=True, convert_to_tensor=True
            )
            text_emb = model.encode(
                [item["text"] or ""],
                normalize_embeddings=True,
                convert_to_tensor=True,
            )[0]
        img_emb = frame_embs.mean(dim=0)
        # Re-normalize after mean-pool (upstream keeps single-image
        # normalization; mean-pooling breaks L2 norm so we restore it).
        img_emb = img_emb / (img_emb.norm() + 1e-9)
        image_fea[item["id"]] = img_emb.cpu()
        text_fea[item["id"]] = text_emb.cpu()
        joint_fea[item["id"]] = torch.cat([img_emb, text_emb], dim=-1).cpu()

    torch.save(image_fea, os.path.join(out_dir, "image_embed.pt"))
    torch.save(text_fea, os.path.join(out_dir, "text_embed.pt"))
    torch.save(joint_fea, os.path.join(out_dir, "joint_embed.pt"))
    logging.info(
        f"  wrote {len(image_fea)} embeddings to {out_dir}"
    )


# ======================================================================
# Stage 3 — Conduct retrieval (build hateful/non-hateful pairs)
# ======================================================================


def run_conduct_retrieval(
    label_jsonl_path: str,
    fea_dir: str,
    out_pairs_path: str,
    coverage_rate: float = 0.5,
):
    """Port of `conduct_retrieval.py:95-197`, verbatim control flow.

    Filters the Label-stage predictions by confidence, keeps the
    top `coverage_rate` fraction, computes per-pair image+text cosine
    similarity on the kept videos, and greedy-matches non-hateful
    (`pred=0`) ↔ hateful (`pred=1`) pairs in descending similarity
    order. Writes `{id1, id2, similarity}` per line.

    Note: our adaptation uses the **test split** for the pair pool
    by default because ALARM's self-improvement loop is run on the
    train split in upstream, and we use the train split the same way.
    The caller passes the jsonl from whichever split was labeled.
    """
    import pandas as pd
    import torch
    from scipy.spatial.distance import cdist

    pred_df = pd.read_json(label_jsonl_path, lines=True)
    image_fea = torch.load(os.path.join(fea_dir, "image_embed.pt"))
    text_fea = torch.load(os.path.join(fea_dir, "text_embed.pt"))

    # Upstream `conduct_retrieval.py:123-135`, verbatim sort by
    # confidence and keep top-`coverage_rate` fraction.
    y_prob1 = pred_df["prob1"].astype(float)
    y_prob0 = pred_df["prob0"].astype(float)
    predicted_class = (y_prob1 >= 0.5).astype(int)
    pred_df = pred_df.assign(
        confidence=np.where(predicted_class == 1, y_prob1, y_prob0)
    )
    n_samples = int(len(pred_df) * coverage_rate)
    high_conf_df = pred_df.sort_values("confidence", ascending=False).head(
        n_samples
    )
    high_conf_df["pred"] = high_conf_df["pred"].astype(int)

    valid_ids = [
        i for i in high_conf_df["id"].tolist()
        if i in image_fea and i in text_fea
    ]
    if not valid_ids:
        logging.warning("  no valid high-confidence ids with embeddings")
        return
    img_mat = np.stack(
        [image_fea[i].cpu().numpy() for i in valid_ids], axis=0
    )
    txt_mat = np.stack(
        [text_fea[i].cpu().numpy() for i in valid_ids], axis=0
    )

    # Upstream `conduct_retrieval.py:30-46`, verbatim.
    sims_img = 1 - cdist(img_mat, img_mat, metric="cosine")
    sims_txt = 1 - cdist(txt_mat, txt_mat, metric="cosine")
    sims = sims_img + sims_txt

    id_to_label = dict(
        zip(high_conf_df["id"].tolist(), high_conf_df["pred"].tolist())
    )

    # Upstream `greedy_matching` (`conduct_retrieval.py:55-93`).
    all_pairs = []
    for i, qid in enumerate(valid_ids):
        qlabel = id_to_label[qid]
        for j, bid in enumerate(valid_ids):
            if i == j:
                continue
            blabel = id_to_label[bid]
            if qlabel == 0 and blabel == 1:
                all_pairs.append((qid, bid, float(sims[i, j])))
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    used = set()
    final_pairs = []
    for q_id, b_id, sim in all_pairs:
        if q_id in used or b_id in used:
            continue
        final_pairs.append((q_id, b_id, sim))
        used.add(q_id)
        used.add(b_id)

    os.makedirs(os.path.dirname(out_pairs_path), exist_ok=True)
    with open(out_pairs_path, "w") as f:
        for q, b, s in final_pairs:
            f.write(
                json.dumps({"id1": q, "id2": b, "similarity": s}) + "\n"
            )
        f.flush()
        os.fsync(f.fileno())
    logging.info(
        f"  wrote {len(final_pairs)} pairs to {out_pairs_path}"
    )


# ======================================================================
# Stage 4 — Experience (pairwise video-vs-video)
# ======================================================================


def run_experience(
    model,
    pairs_path: str,
    items_by_id: Dict[str, Dict],
    out_path: str,
):
    """Port of `experience_runner.py:117-164`, verbatim.

    For each `(id1, id2)` pair, build a 16-frame chat (8 frames from
    video A + 8 frames from video B, in that order) and ask the
    model to summarize then contrast them via the `EXPERIENCE_PROMPT_VIDEO`.
    Output is `{id1, id2, experience}` jsonl.
    """
    if not os.path.exists(pairs_path):
        logging.warning(f"  no pairs file at {pairs_path}")
        return
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r.get("id1"), r.get("id2"))
                    if key != (None, None):
                        done.add(key)
                except Exception:
                    pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n_processed = 0
    with open(pairs_path) as fp, open(out_path, "a") as fo:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            id1, id2 = row["id1"], row["id2"]
            if (id1, id2) in done:
                continue
            a = items_by_id.get(id1)
            b = items_by_id.get(id2)
            if a is None or b is None:
                continue
            if a.get("frames") is None or b.get("frames") is None:
                continue
            combined_frames = list(a["frames"]) + list(b["frames"])
            prompt = EXPERIENCE_PROMPT_VIDEO.format(
                text1=a["text"] or "", text2=b["text"] or ""
            )
            try:
                exp = model.chat_multi_img_video(prompt, combined_frames)
            except Exception as e:
                logging.error(
                    f"  pair ({id1}, {id2}): experience failed: {e}"
                )
                exp = ""
            rec = {"id1": id1, "id2": id2, "experience": exp}
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
            n_processed += 1
    logging.info(f"  wrote {n_processed} new experiences to {out_path}")


# ======================================================================
# Stage 5 — Reference set distillation
# ======================================================================


class ReferenceSet:
    """Simplified port of upstream `ReferenceSetManager`
    (`reference_runner.py:59-110` + continuation). We keep the
    ADD/EDIT/UPVOTE/DOWNVOTE op loop and the size cap verbatim.
    """

    def __init__(self, size: int = 20):
        self.size = size
        self.set: List[Dict] = [
            {"reference": "placeholder", "importance": 2}
        ]

    def cur_set_str(self, top_k: int = None):
        top = self.set if top_k is None else self.set[:top_k]
        return "\n".join(
            f"{i}: {r['reference']}" for i, r in enumerate(top)
        )

    def apply_ops(self, ops: List[Dict]):
        for op in ops:
            kind = (op.get("operation") or "").upper()
            ref = op.get("reference")
            insight = op.get("insight") or ""
            if kind == "ADD":
                if insight:
                    self.set.append(
                        {"reference": insight, "importance": 1}
                    )
            elif kind == "EDIT":
                try:
                    idx = int(ref)
                    if 0 <= idx < len(self.set) and insight:
                        self.set[idx]["reference"] = insight
                except (ValueError, TypeError):
                    pass
            elif kind == "UPVOTE":
                try:
                    idx = int(ref)
                    if 0 <= idx < len(self.set):
                        self.set[idx]["importance"] = (
                            self.set[idx].get("importance", 1) + 1
                        )
                except (ValueError, TypeError):
                    pass
            elif kind == "DOWNVOTE":
                try:
                    idx = int(ref)
                    if 0 <= idx < len(self.set):
                        self.set[idx]["importance"] = (
                            self.set[idx].get("importance", 1) - 1
                        )
                except (ValueError, TypeError):
                    pass
        # Cap at `size` by importance, keep the highest.
        self.set = sorted(
            self.set, key=lambda r: r.get("importance", 0), reverse=True
        )[: self.size]

    def to_list(self) -> List[Dict]:
        return list(self.set)


_JSON_BLOCK_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_reference_ops(text: str) -> List[Dict]:
    """Extract the first JSON array of op dicts from the model
    response. Matches upstream's `extract_instruction`
    (`reference_runner.py:73-`) behavior of taking the first valid
    JSON block.
    """
    if not text:
        return []
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for d in data:
        if isinstance(d, dict) and "operation" in d:
            out.append(d)
    return out


def run_reference(
    model,
    experience_path: str,
    reference_out_path: str,
    size: int = 20,
):
    """Port of `reference_runner.py:300-403` — iterate over the
    experiences jsonl, ask the model for ADD/EDIT/UPVOTE/DOWNVOTE
    ops via the `chat_text` API, apply them to the running
    `ReferenceSet`, and flush the final set to disk.
    """
    if not os.path.exists(experience_path):
        logging.warning(f"  no experience file at {experience_path}")
        return
    ref_set = ReferenceSet(size=size)
    with open(experience_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            new_exp = row.get("experience", "") or ""
            if not new_exp:
                continue
            prompt = REFERENCE_PROMPT_VIDEO.format(
                size=size,
                cur_set_str=ref_set.cur_set_str(),
                new_experience=new_exp,
            )
            try:
                resp = model.chat_text(prompt, max_tokens=2048)
            except Exception as e:
                logging.error(f"  reference chat_text failed: {e}")
                resp = ""
            ops = _parse_reference_ops(resp)
            ref_set.apply_ops(ops)

    os.makedirs(os.path.dirname(reference_out_path), exist_ok=True)
    with open(reference_out_path, "w") as f:
        json.dump(ref_set.to_list(), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    logging.info(
        f"  wrote reference set of {len(ref_set.set)} items to {reference_out_path}"
    )


# ======================================================================
# Stage 6 — InPredict (final test-split prediction with references)
# ======================================================================


def _parse_answer(text: str) -> int:
    """Parse `Thought: ... Answer: [harmful|harmless]` format.

    Matches upstream `inpredict_runner.py` post-processing: look for
    `Answer:` token, lowercase, strip punctuation, pick 0/1 based on
    `harmful | harmless`. Fallback = 0 (conservative).
    """
    if not text:
        return 0
    tail = text.split("Answer:")[-1].lower()
    tail = tail.replace(".", " ").replace(",", " ").replace("\n", " ")
    if "harmless" in tail:
        return 0
    if "harmful" in tail:
        return 1
    return 0


def run_inpredict(
    model,
    test_items: List[Dict],
    reference_path: str,
    out_path: str,
):
    """Port of `inpredict_runner.py:207-260`, video-adapted.

    For each test video:
      1. Load the distilled reference set.
      2. Format the `INPREDICT_PROMPT_VIDEO` with transcript + refs.
      3. Call `chat_multi_img_video(prompt, frames_8)` — upstream
         uses `chat_img(prompt, image)` with a single meme image;
         we substitute the 8-frame multi-image call per
         `feedback_meme_to_video_8frames.md`.
      4. Parse the `Thought: ... Answer:` format → binary pred.
      5. Write `{video_id, pred, label, thought, raw_response}` jsonl.
    """
    if not os.path.exists(reference_path):
        raise FileNotFoundError(
            f"reference set not found at {reference_path}"
        )
    with open(reference_path) as f:
        ref_list = json.load(f)
    reference_set_str = "\n".join(
        f"{i}: {r.get('reference', '')}" for i, r in enumerate(ref_list)
    )

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        for item in test_items:
            if item["id"] in done:
                continue
            prompt = INPREDICT_PROMPT_VIDEO.format(
                text=item["text"] or "",
                reference_set=reference_set_str,
            )
            try:
                response = model.chat_multi_img_video(
                    prompt, item["frames"], max_new_tokens=1024
                )
            except Exception as e:
                logging.error(f"  {item['id']}: inpredict failed: {e}")
                response = ""
            pred = _parse_answer(response)
            thought = ""
            if "Thought:" in response and "Answer:" in response:
                try:
                    thought = (
                        response.split("Thought:", 1)[1]
                        .split("Answer:", 1)[0]
                        .strip()
                    )
                except Exception:
                    thought = ""
            rec = {
                "video_id": item["id"],
                "pred": int(pred),
                "label": int(item["label"]),
                "thought": thought,
                "raw_response": response,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
