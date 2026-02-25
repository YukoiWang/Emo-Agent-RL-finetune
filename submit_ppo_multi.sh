#!/bin/bash
# 多卡 PPO/GRPO 训练 - 使用 accelerate launch
# 修改 --gres=gpu:N 来指定卡数
#SBATCH --job-name=emo-ppo-multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ppo_multi_%j.out
#SBATCH --error=logs/ppo_multi_%j.err

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

# 单卡可去掉 accelerate launch，直接用 python
# 多卡必须用 accelerate launch；SLURM 下可用 SLURM_GPUS_PER_NODE 或手动指定
NUM_GPUS=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-2}}
echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPUs: $NUM_GPUS"
echo "Python: $(python --version)"
echo "================================"

accelerate launch --num_processes=$NUM_GPUS scripts/rl/run_rl.py --config configs/rl_default.yaml

echo "=== Job finished at $(date) ==="
