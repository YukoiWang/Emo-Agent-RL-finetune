import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 若 /tmp 有空间，可将 HF 缓存放到 /tmp 避免磁盘满
_cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
if not _cache and os.path.exists("/tmp"):
    _tmp_cache = "/tmp/hf_cache"
    os.makedirs(_tmp_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", _tmp_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", _tmp_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _tmp_cache)

import yaml

from src.training.ppo_emo_trainer import run_ppo_emo_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL (PPO) training for counseling empathy model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to RL config YAML file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume training from (e.g. outputs/ppo_emo/checkpoint-500 or outputs/ppo_emo/final).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.resume:
        cfg.setdefault("training", {})["resume_from_checkpoint"] = args.resume

    algo = cfg.get("rl", {}).get("algo", "ppo")
    if algo == "ppo":
        run_ppo_emo_training(cfg)
    elif algo in ("grpo", "grpo_emo"):
        from src.training.grpo_training import run_grpo_emo_training
        run_grpo_emo_training(cfg)
    elif algo in ("gspo", "gspo_emo"):
        from src.training.gspo_training import run_gspo_emo_training
        run_gspo_emo_training(cfg)
    elif algo == "dpo_emo":
        from src.training.dpo_emo_trainer import run_dpo_emo_training
        run_dpo_emo_training(cfg)
    else:
        raise ValueError(f"不支持的 RL 算法: {algo}，当前支持 ppo, grpo, grpo_emo, gspo, gspo_emo, dpo_emo")


if __name__ == "__main__":
    main()

