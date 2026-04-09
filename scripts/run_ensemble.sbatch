#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --output=/data/jehc223/EMNLP2/logs/ensemble_%j.log
#SBATCH --error=/data/jehc223/EMNLP2/logs/ensemble_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate implihate

export HF_HOME=/data/jehc223/.cache/huggingface
export PYTHONUNBUFFERED=1
cd /data/jehc223/EMNLP2

echo "=== Ensemble Training ==="
echo "Time: $(date)"

python src/ensemble.py \
    --datasets HateMM MultiHateClip_CN MultiHateClip_EN \
    --seeds 42 123 456 789 1024

echo "=== Done ==="
echo "Time: $(date)"
