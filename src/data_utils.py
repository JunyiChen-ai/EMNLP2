"""
Data utilities for MHClip datasets.

Provides media path resolution, annotation loading, and clean split generation.
"""

import argparse
import csv
import json
import os
import glob as globmod

DATASET_ROOTS = {
    "MHClip_EN": "/data/jehc223/Multihateclip/English",
    "MHClip_ZH": "/data/jehc223/Multihateclip/Chinese",
    "HateMM": "/data/jehc223/HateMM",
}

MP4_SUBDIRS = {
    "MHClip_EN": "video_mp4",
    "MHClip_ZH": "video",
    "HateMM": "video",
}


def get_media_path(vid, dataset):
    """Return (path, media_type) or None.

    Checks mp4 first (must exist and be >1000 bytes), then frames dir
    (must contain at least 1 jpg).
    """
    root = DATASET_ROOTS[dataset]
    mp4_dir = MP4_SUBDIRS[dataset]

    # Check mp4
    mp4_path = os.path.join(root, mp4_dir, f"{vid}.mp4")
    if os.path.isfile(mp4_path) and os.path.getsize(mp4_path) > 1000:
        return (mp4_path, "video")

    # Check frames
    frames_dir = os.path.join(root, "frames", vid)
    if os.path.isdir(frames_dir):
        jpgs = globmod.glob(os.path.join(frames_dir, "*.jpg"))
        if len(jpgs) >= 1:
            return (frames_dir, "frames")

    return None


def load_annotations(dataset):
    """Load annotation(new).json and return dict[vid -> {title, transcript, label}]."""
    root = DATASET_ROOTS[dataset]
    ann_path = os.path.join(root, "annotation(new).json")
    with open(ann_path) as f:
        data = json.load(f)

    result = {}
    for entry in data:
        vid = entry["Video_ID"]
        result[vid] = {
            "title": entry.get("Title", ""),
            "transcript": entry.get("Transcript", ""),
            "label": entry.get("Label", ""),
        }
    return result


def generate_clean_splits(dataset):
    """Generate train_clean.csv and test_clean.csv excluding missing-media or
    not-in-annotation videos. Prints counts."""
    root = DATASET_ROOTS[dataset]
    splits_dir = os.path.join(root, "splits")
    annotations = load_annotations(dataset)

    for split_name in ["train", "test"]:
        src_path = os.path.join(splits_dir, f"{split_name}.csv")
        if not os.path.isfile(src_path):
            print(f"  [WARN] {src_path} not found, skipping")
            continue

        with open(src_path) as f:
            all_ids = [line.strip() for line in f if line.strip()]

        clean_ids = []
        excluded_no_annotation = 0
        excluded_no_media = 0

        for vid in all_ids:
            if vid not in annotations:
                excluded_no_annotation += 1
                continue
            if get_media_path(vid, dataset) is None:
                excluded_no_media += 1
                continue
            clean_ids.append(vid)

        out_path = os.path.join(splits_dir, f"{split_name}_clean.csv")
        with open(out_path, "w") as f:
            for vid in clean_ids:
                f.write(vid + "\n")

        # Count label distribution
        label_counts = {}
        for vid in clean_ids:
            lbl = annotations[vid]["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        print(f"  {dataset} {split_name}: {len(all_ids)} total -> {len(clean_ids)} clean "
              f"(excluded: {excluded_no_annotation} no-annotation, {excluded_no_media} no-media)")
        for lbl in sorted(label_counts.keys()):
            print(f"    {lbl}: {label_counts[lbl]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-splits", action="store_true",
                        help="Generate clean splits for both datasets")
    args = parser.parse_args()

    if args.generate_splits:
        for ds in ["MHClip_EN", "MHClip_ZH"]:
            print(f"\n=== {ds} ===")
            generate_clean_splits(ds)
