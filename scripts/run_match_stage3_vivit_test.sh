#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# MATCH stage 3 — ViViT-b-16x2-kinetics400 32-frame video features.
# TEST split first (frames_32 already extracted for test); train split
# will run after 8501 CPU extract finishes. Single GPU.
python src/match_repro/stage3/extract_vivit.py --all --split test
