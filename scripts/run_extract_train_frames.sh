#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# Extract 16 frames per video for TRAIN splits across all 4 datasets.
# Test splits already done. CPU-only; runs ffmpeg/opencv via the
# extract_frames.py helper which parallelizes via --workers.
# Blocks: LoReHM retrieval (needs train-pool features), Pro-Cap V3
# caption generation (needs train split), MATCH stage 3 ViViT (needs
# train frames_32 — different script, but same blocker pattern).
python src/match_repro/extract_frames.py --all --split train --workers 4
