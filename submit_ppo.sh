#!/bin/bash
#SBATCH --job-name=emo-ppo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err

export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
# 减少 CUDA 显存碎片化，有助于避免 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/yukiwang/.cache/huggingface
export TRANSFORMERS_CACHE=/home/yukiwang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

module load python3.11-anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate emo

cd /home/yukiwang/Emo-Agent-RL-finetune

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "================================"

python scripts/run_rl.py --config configs/rl_default.yaml

echo "=== Job finished at $(date) ==="
