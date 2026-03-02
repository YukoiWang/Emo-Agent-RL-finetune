#!/bin/bash
#SBATCH --job-name=emo-ppo-1
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/ppo_%j.out
#SBATCH --error=logs/ppo_%j.err

export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/ytwong/.cache/huggingface
export TRANSFORMERS_CACHE=/home/ytwong/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export DEEPSEEK_API_KEY="sk-8136694e31ae47098ed0fa350f5ea610"

# DDP 通信设置
export MASTER_ADDR=localhost
export MASTER_PORT=$((29501 + SLURM_JOB_ID % 1000))
export NCCL_DEBUG=WARN

CONDA_PATH="${HOME}/miniconda3"
if [[ ! -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    CONDA_PATH="${HOME}/anaconda3"
fi
source "${CONDA_PATH}/etc/profile.d/conda.sh"
conda activate emo

cd /home/ytwong/Emo-Agent-RL-finetune

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "MASTER_PORT=$MASTER_PORT"
echo "================================"

# ========== 4 卡训练，planning 走 DeepSeek API ==========
accelerate launch \
    --num_processes=1 \
    --multi_gpu \
    --main_process_port=$MASTER_PORT \
    --dynamo_backend=no \
    scripts/rl/run_rl.py --config configs/ppo.yaml

echo "=== Job finished at $(date) ==="