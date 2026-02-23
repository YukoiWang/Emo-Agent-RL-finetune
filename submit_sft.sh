#!/bin/bash
#SBATCH --job-name=emo-sft
#SBATCH --partition=RTX3090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/yukiwang/.cache/huggingface
export TRANSFORMERS_CACHE=/home/yukiwang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

module load python3.11-anaconda/2024.02
eval "$(conda shell.bash hook)"
conda activate emo

cd /home/yukiwang/Emo-Agent-RL-finetune
mkdir -p logs

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "================================"

python scripts/run_sft_empathetic.py --config configs/sft_empathetic.yaml

echo "=== Job finished at $(date) ==="
