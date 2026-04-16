#!/bin/bash
set -e
source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction
cd /data/jehc223/EMNLP2

# MATCH stage 3 — ViViT train-split feature extraction.
# EN/ZH/HateMM first (frames_32 train already extracted).
# Wait for ImpliHateVid frames_32 to finish (8501 CPU job) before the
# final dataset. ViViT test split already done (8502).
python src/match_repro/stage3/extract_vivit.py --dataset MHClip_EN --split train
python src/match_repro/stage3/extract_vivit.py --dataset MHClip_ZH --split train
python src/match_repro/stage3/extract_vivit.py --dataset HateMM --split train

# Wait for frames_32 extraction to ensure ImpliHateVid frames are ready.
while [ ! -f /data/jehc223/EMNLP2/logs/.extr_tr_32_done ]; do
  if ! squeue -u jehc223 -h -o "%j" 2>/dev/null | grep -q extr_tr_32; then
    echo "[vivit_train] extr_tr_32 no longer in queue; proceeding"
    break
  fi
  echo "[vivit_train] waiting for extr_tr_32 to finish..."
  sleep 60
done

python src/match_repro/stage3/extract_vivit.py --dataset ImpliHateVid --split train
