#!/bin/bash
#SBATCH --job-name=emo-ppo-1gpu
#SBATCH --account=si568w26_class
#SBATCH -N 1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o logs/ppo_emo_1gpu_%j.out
#SBATCH -e logs/ppo_emo_1gpu_%j.err

set -euo pipefail

# ===== Environment =====
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
# Avoid TRANSFORMERS_CACHE deprecation warning
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export DEEPSEEK_API_KEY="sk-8136694e31ae47098ed0fa350f5ea610"
unset TRANSFORMERS_CACHE || true

mkdir -p logs "$HF_HOME"

# DDP/accelerate env (single GPU still uses a process group internally)
export MASTER_ADDR=localhost
export MASTER_PORT=$((29501 + SLURM_JOB_ID % 1000))
export NCCL_DEBUG=WARN

# Load user shell env; then conda OR project venv.
# NOTE: Some clusters' /etc/bashrc uses unset variables; keep nounset off.
set +u
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
# Optional: conda (with common install paths on compute nodes)
for _d in "$HOME/miniconda3" "$HOME/miniconda" "$HOME/anaconda3" "$HOME/anaconda"; do
  if [ -x "$_d/bin/conda" ]; then
    export PATH="$_d/bin:$PATH"
    break
  fi
done
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate emo
else
  # No conda: use project .venv if present
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo "Using project venv: $SCRIPT_DIR/.venv"
  else
    echo "No conda and no .venv; using PATH python: $(command -v python || echo 'python not found')"
  fi
fi
# Do not re-enable 'set -u' here; /etc/bashrc may be sourced again and use unset vars.

# ===== Project =====
cd /home/yukiwang/Emo-Agent-RL-finetune

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "HF_HOME=$HF_HOME"
echo "MASTER_PORT=$MASTER_PORT"
echo "================================"

# ===== Run (single GPU) =====
accelerate launch \
  --num_processes 1 \
  scripts/rl/run_rl.py \
  --config configs/ppo.yaml

echo "=== Job finished at $(date) ==="