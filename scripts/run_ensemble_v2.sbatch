#!/bin/bash
#SBATCH --job-name=ens_v2
#SBATCH --output=/data/jehc223/EMNLP2/logs/ensemble_v2_%j.log
#SBATCH --error=/data/jehc223/EMNLP2/logs/ensemble_v2_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate implihate

export HF_HOME=/data/jehc223/.cache/huggingface
export HF_HUB_OFFLINE=0
export PYTHONUNBUFFERED=1
cd /data/jehc223/EMNLP2

echo "=== Ensemble v2: DeBERTa + Cross-dataset ==="
echo "Time: $(date)"

python src/ensemble_v2.py \
    --datasets HateMM MultiHateClip_CN MultiHateClip_EN \
    --model_name microsoft/deberta-v3-base \
    --seeds 42 123 456 789 1024 \
    --pretrain_epochs 5

echo "=== Done ==="
echo "Time: $(date)"
