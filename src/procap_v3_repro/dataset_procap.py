"""Pro-Cap V3 — Multimodal_Data port (caption + label entries +
in-context demo prompt builder).

Source: `external_repos/procap/codes/scr/dataset.py:49-294` (class
`Multimodal_Data`). The ported pieces are:

  * `load_pkl` — reads our Qwen2-VL caption pkl
    (`results/procap_v3/<dataset>/captions_<split>.pkl`)
  * Caption stitching: 8 VQA probe answers + general caption joined
    with `' . '` into a single `cap` string. Mirrors upstream's
    `cap=cap+' . '+ext` accumulation in `dataset.py:153-159`. Probe
    order matches upstream `--ASK_CAP race,gender,country,animal,
    valid_disable,religion` (`scr/config.py:43`) plus `person`,
    `what_animal`, and the gen caption for completeness — same
    fields the V3 captioner produces.
  * `select_context` — upstream `dataset.py:205-236` verbatim, picks
    one positive + one negative random demo from the train pool
    (`max_demo_per_label = 1`).
  * `process_prompt` — upstream `dataset.py:238-254` verbatim. Builds
    `<sent> . It was <mask> . <cap>` for the query and
    `<sent> . It was <label_word> . <cap>` for the demo, joined with
    ` . </s> `.

Key adaptations vs. upstream:

  1. **Video instead of meme**: `meme_text` → video transcript field
     (truncated to 500 chars, matching the LoReHM brief). The label
     mapping `{0: POS_WORD='good', 1: NEG_WORD='bad'}` is unchanged
     (`config.py:21-22`).
  2. **Train/valid/test splits** come from our
     `<dataset_root>/splits/{split}_clean.csv` instead of upstream's
     `domain_splits/<dataset>_<mode>.json`. Validation split is a
     deterministic 80/20 stratified split from the train pool
     (seed=2025 by default).
  3. **No `ADD_ENT` / `ADD_DEM`** — those upstream features pull
     external entity / demographic info we don't have for video.
     Disabled by default; the architecture is otherwise identical.
"""

import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "our_method"))
from data_utils import DATASET_ROOTS, SKIP_VIDEOS, load_annotations  # noqa: E402

PROJECT_ROOT = "/data/jehc223/EMNLP2"

# Upstream `config.py:21-22` — POS_WORD='good', NEG_WORD='bad'. The
# `{0: POS_WORD, 1: NEG_WORD}` mapping (label 0=harmless→good,
# label 1=harmful→bad) is unchanged from upstream.
LABEL_WORDS = ("good", "bad")
LABEL_MAPPING_WORD = {0: "good", 1: "bad"}

# Field order for caption stitching. Matches upstream's
# `--ASK_CAP race,gender,country,animal,valid_disable,religion` plus
# `person`, `what_animal`, and `gen` (the general caption) which the
# upstream notebook produces in addition to the ASK_CAP set.
PROBE_FIELDS = [
    "race",
    "gender",
    "country",
    "animal",
    "what_animal",
    "person",
    "disabled",
    "religion",
]
GEN_FIELD = "gen"

TRANSCRIPT_LIMIT = 500


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def collapse_label(dataset, gt_label):
    """Mirror of `eval_generative_predictions.collapse_label`."""
    if dataset == "HateMM":
        return 1 if gt_label == "Hate" else 0
    if dataset == "ImpliHateVid":
        return 1 if gt_label == "Hateful" else 0
    return 1 if gt_label in ("Hateful", "Offensive") else 0


def captions_path(dataset, split):
    return os.path.join(
        PROJECT_ROOT, "results", "procap_v3", dataset,
        f"captions_{split}.pkl",
    )


def load_split_video_ids(dataset, split):
    csv_path = os.path.join(
        DATASET_ROOTS[dataset], "splits", f"{split}_clean.csv"
    )
    if not os.path.isfile(csv_path):
        from data_utils import generate_clean_splits
        generate_clean_splits(dataset)
    with open(csv_path) as f:
        return [line.strip() for line in f if line.strip()]


def stitch_caption(record):
    """Stitch one V3 caption record into the upstream `cap` string.

    Mirrors upstream `dataset.py:153-159` accumulation:
        cap = gen
        ext = [probe_answers...]
        cap = cap + ' . ' + ' . '.join(ext)

    The gen caption is the spine; the 8 probe answers are appended.
    Empty probe answers are skipped (mirrors upstream's `if info` style
    guards, e.g. `if info.startswith('no'): continue` for valid_disable).
    """
    gen = (record.get(GEN_FIELD, "") or "").strip()
    parts = []
    for field in PROBE_FIELDS:
        val = (record.get(field, "") or "").strip()
        if val:
            parts.append(val)
    if not parts:
        return gen
    return gen + " . " + " . ".join(parts)


def build_entries(dataset, split):
    """Return `(entries, missing_caps_list)` for a given split.

    Each entry mirrors upstream `dataset.py:182-188`:
        {
          "cap":       <stitched caption>,
          "meme_text": <video transcript[:500]>,  # was meme OCR upstream
          "label":     0 | 1,
          "img":       <video_id>,                 # upstream field name
        }
    """
    cap_path = captions_path(dataset, split)
    if not os.path.exists(cap_path):
        raise FileNotFoundError(
            f"V3 captions missing for {dataset}/{split} at {cap_path}; "
            "run generate_captions_qwen2vl.py first"
        )
    captions = load_pkl(cap_path)
    annotations = load_annotations(dataset)
    skip_set = SKIP_VIDEOS.get(dataset, set())
    vids = load_split_video_ids(dataset, split)

    entries = []
    missing = []
    for vid in vids:
        if vid in skip_set:
            continue
        if vid not in annotations:
            continue
        if vid not in captions:
            missing.append(vid)
            continue
        cap = stitch_caption(captions[vid])
        transcript = (annotations[vid].get("transcript", "") or "")[
            :TRANSCRIPT_LIMIT
        ]
        label = collapse_label(dataset, annotations[vid]["label"])
        entries.append({
            "cap": cap.strip(),
            "meme_text": transcript,
            "label": int(label),
            "img": vid,
        })
    return entries, missing


def stratified_train_valid_split(train_entries, valid_frac=0.2, seed=2025):
    """Deterministic 80/20 stratified split per the brief.

    Returns `(train_subset, valid_subset)`. Within each label class we
    sort by `img` so the split is reproducible regardless of dict
    iteration order, then take the first `valid_frac` for valid.
    """
    by_label = {0: [], 1: []}
    for e in train_entries:
        by_label[e["label"]].append(e)
    rng = random.Random(seed)
    for lbl in (0, 1):
        by_label[lbl].sort(key=lambda e: e["img"])
        rng.shuffle(by_label[lbl])
    train_subset = []
    valid_subset = []
    for lbl in (0, 1):
        n_valid = max(1, int(round(len(by_label[lbl]) * valid_frac)))
        valid_subset.extend(by_label[lbl][:n_valid])
        train_subset.extend(by_label[lbl][n_valid:])
    rng.shuffle(train_subset)
    rng.shuffle(valid_subset)
    return train_subset, valid_subset


class Multimodal_Data(Dataset):
    """Port of upstream `dataset.py:49-293`.

    Constructor signature mirrors upstream: `(opt_like, dataset, mode)`.
    `opt_like` here is a tiny SimpleNamespace with the same field
    names upstream accesses (`NUM_LABELS`, `NUM_SAMPLE`, `POS_WORD`,
    `NEG_WORD`).
    """

    def __init__(
        self,
        dataset,
        entries,
        support_entries,
        num_labels=2,
        num_sample=1,
    ):
        super().__init__()
        self.dataset = dataset
        self.entries = entries
        self.support_examples = support_entries
        self.num_ans = num_labels
        self.num_sample = num_sample
        self.label_mapping_word = LABEL_MAPPING_WORD
        self._prepare_exp()

    def _prepare_exp(self):
        """Upstream `dataset.py:194-203` verbatim."""
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.entries)):
                context_indices = [
                    sidx for sidx in support_indices if sidx != query_idx
                ]
                self.example_idx.append(
                    (query_idx, context_indices, sample_idx)
                )

    def select_context(self, context_examples):
        """Upstream `dataset.py:205-236` verbatim. One positive + one
        negative demo per query (max_demo_per_label=1).
        """
        num_labels = self.num_ans
        max_demo_per_label = 1
        counts = {k: 0 for k in range(num_labels)}
        selection = []
        order = np.random.permutation(len(context_examples))
        for i in order:
            label = context_examples[i]["label"]
            if counts[label] < max_demo_per_label:
                selection.append(context_examples[i])
                counts[label] += 1
            if sum(counts.values()) == len(counts) * max_demo_per_label:
                break
        assert len(selection) > 0
        return selection

    def process_prompt(self, examples):
        """Upstream `dataset.py:238-254` adapted for longer video
        transcripts. Deviation #1 (2026-04-16): mask/label placed at
        the START of each segment instead of the middle. Upstream
        format was `{meme_text} . It was <mask> . {cap}` (mask at
        position ~35 for short meme OCR). Our transcript can be
        150+ tokens, pushing the mask past the truncation limit.
        New format: `It was <mask> . {meme_text} . {cap}` — mask at
        position ~3 regardless of transcript/cap length.
        """
        prompt_arch = "It was "
        concat_sent = []
        test_text = ""
        for segment_id, ent in enumerate(examples):
            if segment_id == 0:
                temp = prompt_arch + "<mask> . "
            else:
                label_word = self.label_mapping_word[ent["label"]]
                temp = prompt_arch + label_word + " . "
            whole_sent = temp + ent["meme_text"] + " . " + ent["cap"]
            concat_sent.append(whole_sent)
            if segment_id == 0:
                test_text = ent["meme_text"] + " . " + ent["cap"]
        return concat_sent, test_text

    def __getitem__(self, index):
        """Upstream `dataset.py:257-290` verbatim. Returns a dict with
        `prompt_all_text` (with demos) and `test_all_text` (no demos).
        """
        entry = self.entries[index]
        query_idx, context_indices, bootstrap_idx = self.example_idx[index]
        supports = self.select_context(
            [self.support_examples[i] for i in context_indices]
        )
        exps = [entry]
        exps.extend(supports)
        concat_sent, test_text = self.process_prompt(exps)
        prompt_texts = " . </s> ".join(concat_sent)

        vid = entry["img"]
        label = torch.tensor(entry["label"])
        target = torch.from_numpy(
            np.zeros((self.num_ans,), dtype=np.float32)
        )
        target[entry["label"]] = 1.0

        return {
            "img": vid,
            "target": target,
            "test_all_text": concat_sent[0] + " . </s> ",
            "test_text": test_text,
            "prompt_all_text": prompt_texts,
            "label": label,
        }

    def __len__(self):
        return len(self.entries)
