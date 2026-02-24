#!/bin/bash
#SBATCH --job-name=emo-dpo
#SBATCH --partition=gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dpo_%j.out
#SBATCH --error=logs/dpo_%j.err

# 环境：优先 conda，无则用项目 .venv
CONDA_PATH="${HOME}/miniconda3"
if [[ ! -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    CONDA_PATH="${HOME}/anaconda3"
fi
if [[ -f "${CONDA_PATH}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_PATH}/etc/profile.d/conda.sh"
    conda activate emo
else
    # 使用项目 venv
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

# 1. 若无偏好数据则先构建（使用本地 data/empathetic_dialogues/train.jsonl）
PREF_FILE="static-rl/data/empathetic_preference.jsonl"
if [[ ! -f "$PREF_FILE" ]]; then
    echo "Building preference dataset from local train.jsonl..."
    python static-rl/build_empathetic_preference_dataset.py --output "$PREF_FILE"
fi

# 2. 运行 DPO 训练（LoRA 默认；全量用 static-rl/configs/dpo_full.yaml）
python static-rl/run_dpo.py --config static-rl/configs/dpo.yaml

echo "=== Job finished at $(date) ==="
