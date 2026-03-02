#!/bin/bash
#SBATCH --job-name=emo-grpo-mode1
#SBATCH --partition=ADA6000
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
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
export MASTER_PORT=$((29502 + SLURM_JOB_ID % 1000))
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

# 直接启动 GRPO 训练（planning 通过外部 API）
accelerate launch \
    --num_processes=4 \
    --multi_gpu \
    --main_process_port=$MASTER_PORT \
    --dynamo_backend=no \
    scripts/rl/run_rl.py --config configs/rl_grpo_emo_2.yaml

echo "=== Job finished at $(date) ==="