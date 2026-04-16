"""LoReHM RSA retrieval — video adaptation (v2 brief).

Upstream LoReHM's RSA (`utils/utils.py:87-99`, `get_rsa_label`) consumes
a pre-computed `rel_sampl` tuple of three lists:
    (example, harmful_example, harmless_example)
where `example` is the ranked list of retrieved items (top-50 per
upstream's shipped jsonls), `harmful_example` / `harmless_example`
are the subsets labeled harmful / harmless. `get_rsa_label(rel_sampl,
k)` majority-votes over the top-`k` entries (upstream default `k=5`).

Our task is to build `rel_sampl` tuples for our 4 video benchmarks,
then plug them into upstream's `get_rsa_label` verbatim.

Adaptation 4 per v2 brief:
  * Feature source: **jinaai/jina-clip-v2** (already cached for
    MATCH stage 2b.5). Zero extra downloads. Matches upstream ALARM's
    `make_embeddings.py:11` loader pattern.
  * Per-video image feature = mean of 8 per-frame `encode_image`
    outputs at `truncate_dim=512`, then L2-renormalized. This is the
    natural 8-frame video extension of the single-image meme
    encoding.
  * Retrieval pool = labeled train split (per-video ground-truth
    labels from `data_utils.load_annotations`). Ground-truth train
    labels are in-bounds under the MATCH stage 3 / Mod-HATE few-shot
    supervised precedent.
  * Top-50 retrieval, top-5 majority vote. `example` list holds 50
    train-video ids sorted by cosine descending; `harmful_example` /
    `harmless_example` are the label-1 and label-0 subsets.
  * `get_rsa_label` is then byte-for-byte upstream
    (`utils/utils.py:87-99`).
  * Persisted to `results/lorehm/<dataset>/rel_sampl.json` with
    schema `{test_vid -> [example, harmful, harmless]}`. Can be
    rebuilt by running `retrieval.py` as a script.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

_OUR_METHOD = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "our_method"
)
sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lorehm_video_dataset import (  # noqa: E402
    ALL_DATASETS,
    FRAME_INDICES,
    NUM_FRAMES,
    collapse_label,
    load_frames_16,
    load_split_video_ids,
)

PROJECT_ROOT = "/data/jehc223/EMNLP2"

# v2 brief adaptation 4: Jina-CLIP-v2, matching upstream ALARM
# `make_embeddings.py:11` loader (same model file, already cached).
DEFAULT_JINA_MODEL = "jinaai/jina-clip-v2"
JINA_TRUNCATE_DIM = 512

# Upstream `rel_sampl` jsonls have ~50 entries per item.
DEFAULT_POOL_TOPK = 50


def _load_jina_clip(model_id: str = DEFAULT_JINA_MODEL):
    """Lazy loader so the module is import-clean for syntax check.

    Matches upstream `make_embeddings.py:11`:
        SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True,
                            truncate_dim=512, device='cuda')
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(
        model_id,
        trust_remote_code=True,
        truncate_dim=JINA_TRUNCATE_DIM,
        device=device,
    )


def extract_video_features(dataset: str, split: str, model) -> Dict[str, "np.ndarray"]:
    """Return `{vid -> feature_vec}` with shape `(JINA_TRUNCATE_DIM,)`
    per video, computed as the L2-normalized mean of 8 per-frame Jina
    CLIP image embeddings.
    """
    import numpy as np
    import torch

    ann = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    vids = load_split_video_ids(dataset, split)

    out: Dict[str, np.ndarray] = {}
    for vid in vids:
        if vid in skip_set or vid not in ann:
            continue
        frames = load_frames_16(dataset, vid)
        if frames is None:
            continue
        with torch.no_grad():
            frame_embs = model.encode(
                frames,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
        pooled = frame_embs.mean(dim=0)
        pooled = pooled / (pooled.norm() + 1e-9)
        out[vid] = pooled.cpu().numpy()
    return out


def build_rel_sampl(
    test_features: Dict[str, "np.ndarray"],
    train_features: Dict[str, "np.ndarray"],
    train_labels: Dict[str, int],
    pool_topk: int = DEFAULT_POOL_TOPK,
) -> Dict[str, Tuple[List[str], List[str], List[str]]]:
    """Build one `rel_sampl` 3-tuple per test video.

    Matches upstream's schema `(example, harmful_example,
    harmless_example)` where `example` is ranked by cosine descending
    and `harmful_example` / `harmless_example` are the label-1 /
    label-0 subsets of the retrieval pool.
    """
    import numpy as np

    rel: Dict[str, Tuple[List[str], List[str], List[str]]] = {}
    train_vids = list(train_features.keys())
    if not train_vids:
        return rel
    train_mat = np.stack([train_features[v] for v in train_vids], axis=0)
    # train_mat: (N_train, D) already L2-normalized

    # Precompute harmful / harmless subsets from ground-truth labels.
    harmful_pool = [v for v in train_vids if train_labels.get(v, 0) == 1]
    harmless_pool = [v for v in train_vids if train_labels.get(v, 0) == 0]

    for test_vid, test_vec in test_features.items():
        tv = test_vec / (np.linalg.norm(test_vec) + 1e-9)
        sims = train_mat @ tv
        order = np.argsort(-sims)[:pool_topk]
        example = [train_vids[i] for i in order]
        # Upstream `rel_sampl[1]` / `rel_sampl[2]` contain the full
        # labeled pool of harmful / harmless items; `get_rsa_label`
        # checks `example in harmful_examples` by membership. The
        # subset intersection with `example` is equivalent to
        # "every example video's label" but matches upstream's
        # 3-list schema exactly.
        rel[test_vid] = (example, list(harmful_pool), list(harmless_pool))
    return rel


def get_rsa_label(rel_sampl, k):
    """Upstream `utils/utils.py:87-99`, byte-for-byte."""
    examples, harmful_examples, harmless_examples = rel_sampl
    examples = examples[:k]
    count = 0
    for l, example in enumerate(examples):
        if example in harmful_examples:
            count += 1
        elif example in harmless_examples:
            count -= 1
    rsa_label = 1 if count >= 0 else 0
    return rsa_label


def load_train_labels(dataset: str) -> Dict[str, int]:
    ann = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    vids = load_split_video_ids(dataset, "train")
    out: Dict[str, int] = {}
    for v in vids:
        if v in skip_set or v not in ann:
            continue
        out[v] = collapse_label(dataset, ann[v]["label"])
    return out


def rel_sampl_output_path(dataset: str) -> str:
    return os.path.join(
        PROJECT_ROOT, "results", "lorehm", dataset, "rel_sampl.json"
    )


def save_rel_sampl(dataset: str, rel_map, path: Optional[str] = None):
    """Persist `{test_vid -> [example, harmful, harmless]}` as JSON.

    Tuples are serialized as JSON lists — order is `(example,
    harmful, harmless)`, matching upstream `main.py:30`.
    """
    path = path or rel_sampl_output_path(dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        vid: [list(rs[0]), list(rs[1]), list(rs[2])]
        for vid, rs in rel_map.items()
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_rel_sampl(dataset: str, path: Optional[str] = None):
    """Load `{vid -> (example, harmful, harmless)}` tuples from disk.

    Returns an empty dict if the file doesn't exist.
    """
    path = path or rel_sampl_output_path(dataset)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        payload = json.load(f)
    return {
        vid: (lst[0], lst[1], lst[2]) for vid, lst in payload.items()
    }


def build_rel_sampl_for_dataset(
    dataset: str,
    pool_topk: int = DEFAULT_POOL_TOPK,
    model=None,
    persist: bool = True,
):
    """Top-level helper: load Jina-CLIP-v2, extract train + test
    features, build `rel_sampl` for the test split. Persists to disk
    if `persist=True`.
    """
    logging.info(
        f"[{dataset}] loading {DEFAULT_JINA_MODEL} (may be cached)"
    )
    if model is None:
        model = _load_jina_clip()
    logging.info(f"[{dataset}] extracting train features")
    train_feats = extract_video_features(dataset, "train", model)
    logging.info(
        f"[{dataset}]   extracted {len(train_feats)} train features"
    )
    logging.info(f"[{dataset}] extracting test features")
    test_feats = extract_video_features(dataset, "test", model)
    logging.info(
        f"[{dataset}]   extracted {len(test_feats)} test features"
    )
    train_labels = load_train_labels(dataset)
    rel_map = build_rel_sampl(
        test_feats, train_feats, train_labels, pool_topk
    )
    logging.info(
        f"[{dataset}]   built rel_sampl for {len(rel_map)} test videos"
    )
    if persist:
        out_path = rel_sampl_output_path(dataset)
        save_rel_sampl(dataset, rel_map, out_path)
        logging.info(f"[{dataset}]   wrote {out_path}")
    return rel_map, test_feats, train_feats, train_labels


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LoReHM retrieval — build rel_sampl.json per dataset using "
            "jinaai/jina-clip-v2 pooled 8-frame video features."
        )
    )
    parser.add_argument("--dataset", choices=ALL_DATASETS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pool-topk", type=int, default=DEFAULT_POOL_TOPK)
    args = parser.parse_args()
    if not args.dataset and not args.all:
        parser.error("Provide --dataset or --all")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load the Jina-CLIP model once so `--all` doesn't re-init per dataset.
    model = _load_jina_clip()
    datasets = ALL_DATASETS if args.all else [args.dataset]
    for ds in datasets:
        build_rel_sampl_for_dataset(
            ds, pool_topk=args.pool_topk, model=model, persist=True
        )
    logging.info("All datasets done.")


if __name__ == "__main__":
    main()
