"""Convert ImpliHateVid xlsx split files to one-video-ID-per-line csvs."""

import os
import sys

import pandas as pd

SPLITS_DIR = "/data/jehc223/ImpliHateVid/splits"


def main():
    for src_name, dst_name in [
        ("Train_videos.xlsx", "train.csv"),
        ("Test_videos.xlsx", "test.csv"),
        ("Val_videos.xlsx", "val.csv"),
    ]:
        src = os.path.join(SPLITS_DIR, src_name)
        dst = os.path.join(SPLITS_DIR, dst_name)
        if not os.path.isfile(src):
            print(f"[skip] {src} not found")
            continue
        df = pd.read_excel(src)
        if "Video_ID" not in df.columns:
            print(f"[error] {src} missing Video_ID column; columns={list(df.columns)}")
            sys.exit(1)
        ids = [str(v).strip() for v in df["Video_ID"].tolist() if str(v).strip()]
        with open(dst, "w") as f:
            for vid in ids:
                f.write(vid + "\n")
        print(f"  {dst_name}: {len(ids)} ids")


if __name__ == "__main__":
    main()
