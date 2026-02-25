#!/bin/bash
#SBATCH --job-name=emo-rm
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/reward_model_%j.out
#SBATCH --error=logs/reward_model_%j.err

# 用法:
#   sbatch submit_reward_model.sh              # 默认 empathetic 偏好数据
#   sbatch submit_reward_model.sh ipm          # 使用 data/ipm_prefdial_dpo.jsonl

CONDA_PATH="${HOME}/miniconda3"
[[ ! -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]] && CONDA_PATH="${HOME}/anaconda3"
if [[ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate emo
else
    source /home/yukiwang/Emo-Agent-RL-finetune/.venv/bin/activate
fi

export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/home/yukiwang/.cache/huggingface
export TRANSFORMERS_CACHE=/home/yukiwang/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

cd /home/yukiwang/Emo-Agent-RL-finetune
mkdir -p logs

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "================================"

if [[ "$1" == "ipm" ]]; then
    python static-rl/run_reward_model.py --config static-rl/configs/reward_model_ipm.yaml
else
    [[ ! -f static-rl/data/empathetic_preference.jsonl ]] && \
        python static-rl/build_empathetic_preference_dataset.py --output static-rl/data/empathetic_preference.jsonl
    python static-rl/run_reward_model.py --config static-rl/configs/reward_model.yaml
fi

echo "=== Job finished at $(date) ==="
