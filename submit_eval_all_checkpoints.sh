#!/bin/bash
# Run comprehensive evaluation for all RL emotion experiments (PPO / GRPO / GSPO) under outputs/.

#SBATCH --job-name=emo-eval-all
#SBATCH --account=si504f25_class
#SBATCH -N 1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH -o logs/eval_emo_all_%j.out
#SBATCH -e logs/eval_emo_all_%j.err

set -euo pipefail

# ===== Environment =====
export PYTHONNOUSERSITE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
unset TRANSFORMERS_CACHE || true

mkdir -p logs "$HF_HOME"

# DDP-style env (single GPU still forms a process group)
export MASTER_ADDR=localhost
export MASTER_PORT=$((29501 + SLURM_JOB_ID % 1000))
export NCCL_DEBUG=WARN
export DEEPSEEK_API_KEY="sk-8136694e31ae47098ed0fa350f5ea610"

# ===== Python / Conda =====
# NOTE: keep nounset off while sourcing user env
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

# ===== Project root =====
cd /home/yukiwang/Emo-Agent-RL-finetune

echo "=== Eval job started at $(date) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "HF_HOME=$HF_HOME"
echo "MASTER_PORT=$MASTER_PORT"
echo "===================================="

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "WARNING: DEEPSEEK_API_KEY is not set; evaluation that relies on DeepSeek may fail."
fi

# ===== Loop over all RL emotion training experiments (PPO + GRPO + GSPO) =====

for outdir in outputs/ppo_emo_train_* outputs/grpo_* outputs/gspo_*; do
  if [ ! -d "$outdir" ]; then
    continue
  fi

  # Require at least one checkpoint-* subdirectory
  if ! compgen -G "$outdir/checkpoint-*/*" > /dev/null; then
    echo "[SKIP] $outdir (no checkpoint-* found)"
    continue
  fi

  echo
  echo "===================================="
  echo " Evaluating experiment: $outdir"
  echo "===================================="

  # Run comprehensive evaluation on all checkpoints in this experiment.
  # - Levels 1,2,3: emotion outcome, strategy, capability
  # - Evaluate all checkpoints at interval 50
  # - Limit test samples for speed
  python scripts/eval/comprehensive/run_eval.py \
    --output-dir "$outdir" \
    --levels 1 2 3 \
    --max-samples 30 \
    --checkpoint-interval 50 \
    --device cuda \
    --api-key "${DEEPSEEK_API_KEY:-}" \
    --api-base "https://api.deepseek.com" \
    --judge-model "deepseek-chat" \
    || echo "[WARN] Evaluation failed for $outdir, continuing..."
done

echo
echo "=== Eval job finished at $(date) ==="

