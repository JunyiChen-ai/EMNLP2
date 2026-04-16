"""MATCH stage 3 — per-dataset torch Datasets + shared collator.

Ports upstream `HateMM_MATCH_Dataset` / `MHClipEN_MATCH_Dataset` /
`MHClipZH_MATCH_Dataset` from
`external_repos/match_hvd/src/model/MATCH/data/` and adds a new
`ImpliHateVid_MATCH_Dataset` with the same structure, swapping only
the per-dataset paths.

Adaptations vs upstream:
  1. Split membership comes from our `<root>/splits/{split}_clean.csv`
     (not upstream's `data/<dataset>/vids/{split}.csv`).
  2. Annotations come from our
     `data_utils.load_annotations(dataset)` (not upstream's
     `data/<dataset>/annotation.csv`).
  3. Label collapse uses `eval_generative_predictions.collapse_label`
     semantics (HateMM="Hate", ImpliHateVid="Hateful",
     MHClip Hateful+Offensive=1).
  4. Stage 2 text streams (`hate.json`, `nonhate.json`, `judge.json`)
     come from our `results/match_qwen2vl_7b/<dataset>/` (not
     upstream's `data/<dataset>/`).
  5. MFCC + ViViT features come from `<root>/fea/fea_audio_mfcc.pt`
     and `<root>/fea/fea_frames_32_google-vivit-b.pt` (our
     `extract_mfcc.py` / `extract_vivit.py` outputs).
  6. **Validation split is derived via a deterministic 80/20
     stratified split of `train_clean.csv` (seed=2025).** Upstream
     has a separate `valid` split file; our clean splits only have
     train + test. Seed matches upstream `HateMM_MATCH.yaml:seed:
     2025`.

Collator is upstream-verbatim in **output shape** — emits the exact
dict layout `{vids, trans_text_inputs, judge_answers_inputs,
hate_answers_inputs, nonhate_answers_inputs, mfcc_fea, vivit_fea,
labels}` consumed by upstream `main.py:220-225`.
"""

import json
import os
import random
import sys
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = "/data/jehc223/EMNLP2"
_OUR_METHOD = os.path.join(_PROJECT_ROOT, "src", "our_method")
if _OUR_METHOD not in sys.path:
    sys.path.insert(0, _OUR_METHOD)
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

ALL_DATASETS = ["MHClip_EN", "MHClip_ZH", "HateMM", "ImpliHateVid"]

# Upstream `HateMM_MATCH.yaml:seed: 2025` — kept for the 80/20
# stratified valid split derivation.
VALID_SPLIT_SEED = 2025
VALID_FRACTION = 0.20  # 80/20 stratified


def _collapse_label(dataset: str, gt_label: str) -> int:
    """Mirror of `eval_generative_predictions.collapse_label`."""
    if dataset == "HateMM":
        return 1 if gt_label == "Hate" else 0
    if dataset == "ImpliHateVid":
        return 1 if gt_label == "Hateful" else 0
    return 1 if gt_label in ("Hateful", "Offensive") else 0


def _stage2_dir(dataset: str) -> str:
    return os.path.join(_PROJECT_ROOT, "results", "match_qwen2vl_7b", dataset)


def _load_split_video_ids(dataset: str, split: str) -> List[str]:
    csv_path = os.path.join(
        DATASET_ROOTS[dataset], "splits", f"{split}_clean.csv"
    )
    if not os.path.isfile(csv_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(csv_path) as f:
        return [line.strip() for line in f if line.strip()]


def _stratified_train_valid_split(
    rows: List[Dict], seed: int = VALID_SPLIT_SEED, valid_frac: float = VALID_FRACTION
):
    """Deterministic 80/20 stratified split on `label`."""
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        by_label.setdefault(r["label"], []).append(i)
    train_idx, valid_idx = [], []
    for label, idxs in by_label.items():
        shuffled = list(idxs)
        rng.shuffle(shuffled)
        n_valid = max(1, int(round(len(shuffled) * valid_frac)))
        valid_idx.extend(shuffled[:n_valid])
        train_idx.extend(shuffled[n_valid:])
    train_rows = [rows[i] for i in sorted(train_idx)]
    valid_rows = [rows[i] for i in sorted(valid_idx)]
    return train_rows, valid_rows


def _load_stage2_answers(dataset: str) -> Dict[str, Dict[str, str]]:
    """Read `hate.json`, `nonhate.json`, `judge.json` for one dataset.

    Returns `{"hate": {vid -> answer}, "nonhate": {...}, "judge": {vid -> summary}}`.
    Missing files are surfaced as empty dicts; the caller fills `""`
    for any missing vid.
    """
    stage2 = _stage2_dir(dataset)
    out = {"hate": {}, "nonhate": {}, "judge": {}}
    for key in ("hate", "nonhate", "judge"):
        path = os.path.join(stage2, f"{key}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        if key == "judge":
            out[key] = {
                row["id"]: row.get("summary", "") for row in data
            }
        else:
            out[key] = {
                row["id"]: row.get("answer", "") for row in data
            }
    return out


def _load_feature_dict(root: str, fname: str):
    """Load `<root>/fea/<fname>` if present, else empty dict."""
    path = os.path.join(root, "fea", fname)
    if not os.path.exists(path):
        return {}
    try:
        return torch.load(path, weights_only=True)
    except Exception:
        return {}


class _BaseMatchDataset(Dataset):
    """Shared base that builds the per-video row list and loads the
    stage-2 + feature dicts once per instance. Subclasses only need
    to override `DATASET_NAME`.
    """

    DATASET_NAME: str = ""

    def __init__(self, split: str, transcript_limit: int = 1000):
        super().__init__()
        assert split in ("train", "valid", "test"), split
        self.split = split
        self.dataset = self.DATASET_NAME
        self.transcript_limit = transcript_limit
        self.root = DATASET_ROOTS[self.dataset]

        # Annotations
        self._ann = load_annotations(self.dataset)
        self._skip = SKIP_VIDEOS.get(self.dataset, set())

        # Stage 2 text streams
        self._stage2 = _load_stage2_answers(self.dataset)

        # Features
        self._mfcc = _load_feature_dict(self.root, "fea_audio_mfcc.pt")
        self._vivit = _load_feature_dict(
            self.root, "fea_frames_32_google-vivit-b.pt"
        )

        # Build per-video rows for the requested split.
        self.rows = self._get_data(split)

    def _get_data(self, split: str) -> List[Dict]:
        """Upstream `HateMM_MATCH.py:15` `_get_data` — returns a list
        of row dicts. We derive the `valid` split deterministically
        from train_clean.csv (v2 brief adaptation 6).
        """
        if split == "test":
            vids = _load_split_video_ids(self.dataset, "test")
            return self._rows_from_vids(vids)

        # train / valid share the same underlying csv; derive valid
        # via 80/20 stratified split, seed=2025.
        all_train_vids = _load_split_video_ids(self.dataset, "train")
        all_train_rows = self._rows_from_vids(all_train_vids)
        train_rows, valid_rows = _stratified_train_valid_split(
            all_train_rows, seed=VALID_SPLIT_SEED, valid_frac=VALID_FRACTION
        )
        if split == "train":
            return train_rows
        return valid_rows  # split == "valid"

    def _rows_from_vids(self, vids: List[str]) -> List[Dict]:
        rows = []
        for vid in vids:
            if vid in self._skip or vid not in self._ann:
                continue
            label = _collapse_label(
                self.dataset, self._ann[vid].get("label", "")
            )
            rows.append(
                {
                    "vid": vid,
                    "label": label,
                }
            )
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        vid = row["vid"]
        label = row["label"]

        transcript = (
            self._ann[vid].get("transcript", "") or ""
        )[: self.transcript_limit]
        judge_answers = self._stage2["judge"].get(vid, "") or ""
        hate_answers = self._stage2["hate"].get(vid, "") or ""
        nonhate_answers = self._stage2["nonhate"].get(vid, "") or ""

        mfcc_fea = self._mfcc.get(vid)
        if mfcc_fea is None:
            mfcc_fea = torch.zeros(40, dtype=torch.float32)
        else:
            mfcc_fea = mfcc_fea.float()
        vivit_fea = self._vivit.get(vid)
        if vivit_fea is None:
            vivit_fea = torch.zeros(768, dtype=torch.float32)
        else:
            vivit_fea = vivit_fea.float()

        return {
            "vid": vid,
            "trans_text": f"{transcript}",
            "judge_answers": f"{judge_answers}",
            "hate_answers": f"{hate_answers}",
            "nonhate_answers": f"{nonhate_answers}",
            "mfcc_fea": mfcc_fea,
            "vivit_fea": vivit_fea,
            "label": torch.tensor(int(label), dtype=torch.long),
        }


class HateMM_MATCH_Dataset(_BaseMatchDataset):
    DATASET_NAME = "HateMM"


class MHClipEN_MATCH_Dataset(_BaseMatchDataset):
    DATASET_NAME = "MHClip_EN"


class MHClipZH_MATCH_Dataset(_BaseMatchDataset):
    DATASET_NAME = "MHClip_ZH"


class ImpliHateVid_MATCH_Dataset(_BaseMatchDataset):
    """New loader for the 4th dataset. No upstream runner exists for
    ImpliHateVid; we reuse the HateMM structure — binary labels, no
    title field — since the upstream base class contract only depends
    on `{trans_text, judge_answers, hate_answers, nonhate_answers,
    mfcc_fea, vivit_fea, label}` which are all dataset-independent.
    """
    DATASET_NAME = "ImpliHateVid"


def get_dataset(dataset: str, split: str, **kwargs) -> _BaseMatchDataset:
    """Upstream `core_utils.get_dataset` equivalent — dataset factory."""
    if dataset == "HateMM":
        return HateMM_MATCH_Dataset(split=split, **kwargs)
    if dataset == "MHClip_EN":
        return MHClipEN_MATCH_Dataset(split=split, **kwargs)
    if dataset == "MHClip_ZH":
        return MHClipZH_MATCH_Dataset(split=split, **kwargs)
    if dataset == "ImpliHateVid":
        return ImpliHateVid_MATCH_Dataset(split=split, **kwargs)
    raise ValueError(f"Unknown dataset: {dataset}")


class MATCH_Collator:
    """Upstream `HateMM_MATCH_Collator` (`HateMM_MATCH.py:64-92`),
    byte-for-byte output shape. Tokenizes 4 text streams via
    `AutoTokenizer.from_pretrained(text_encoder)`, stacks MFCC + ViViT
    features, stacks labels.

    Dataset-independent — one collator serves all 4 datasets.
    """

    def __init__(self, text_encoder: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)

    def __call__(self, batch):
        vids = [item["vid"] for item in batch]
        labels = [item["label"] for item in batch]
        trans_texts = [item["trans_text"] for item in batch]
        judge_answers = [item["judge_answers"] for item in batch]
        hate_answers = [item["hate_answers"] for item in batch]
        nonhate_answers = [item["nonhate_answers"] for item in batch]
        mfcc_fea = torch.stack([item["mfcc_fea"] for item in batch])
        vivit_fea = torch.stack([item["vivit_fea"] for item in batch])

        trans_text_inputs = self.tokenizer(
            trans_texts, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        )
        judge_answers_inputs = self.tokenizer(
            judge_answers, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        )
        hate_answers_inputs = self.tokenizer(
            hate_answers, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        )
        nonhate_answers_inputs = self.tokenizer(
            nonhate_answers, padding=True, truncation=True,
            return_tensors="pt", max_length=512,
        )
        return {
            "vids": vids,
            "trans_text_inputs": trans_text_inputs,
            "judge_answers_inputs": judge_answers_inputs,
            "hate_answers_inputs": hate_answers_inputs,
            "nonhate_answers_inputs": nonhate_answers_inputs,
            "mfcc_fea": mfcc_fea,
            "vivit_fea": vivit_fea,
            "labels": torch.stack(labels),
        }
