#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# MATCH stage 3 — MFCC audio feature extraction via librosa.
# CPU-only (no GPU), all 4 datasets, both splits.
# Output: <dataset_root>/fea/fea_audio_mfcc.pt dict keyed by vid.
python src/match_repro/stage3/extract_mfcc.py --all --split both
