#!/bin/bash
#SBATCH --job-name=emo-sft
#SBATCH --partition=RTX4090
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# 不依赖 module，直接使用 conda
# 若 conda 路径不同，请修改 CONDA_PATH（常见: ~/miniconda3 或 ~/anaconda3）
CONDA_PATH="${HOME}/miniconda3"
if [[ ! -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    CONDA_PATH="${HOME}/anaconda3"
fi
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate emo

export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/ytwonng/.cache/huggingface
export TRANSFORMERS_CACHE=/home/ytwong/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

cd /home/ytwong/Emo-Agent-RL-finetune

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "================================"

python scripts/sft/run_sft_empathetic.py --config configs/sft_empathetic.yaml

echo "=== Job finished at $(date) ==="
