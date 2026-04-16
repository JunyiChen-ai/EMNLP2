#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# Extract 32 frames per train video for MATCH stage 3 ViViT input.
# Blocks: MATCH stage 3 extract_vivit.py (needs frames_32/<vid>/).
# CPU-only.
python src/match_repro/extract_frames_32.py --all --split train --workers 4
