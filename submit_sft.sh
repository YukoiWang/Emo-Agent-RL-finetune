#!/bin/bash
#SBATCH --job-name=emo-sft
#SBATCH --account=si568w26_class
#SBATCH -N 1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH -o logs/sft_%j.out
#SBATCH -e logs/sft_%j.err

set -euo pipefail

# ===== Environment =====
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
unset TRANSFORMERS_CACHE || true

mkdir -p logs "$HF_HOME"

# ===== Load Python env (conda or project .venv) =====
set +u
if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi
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
echo "================================"

python scripts/sft/run_sft_empathetic.py --config configs/sft_empathetic.yaml

echo "=== Job finished at $(date) ==="
