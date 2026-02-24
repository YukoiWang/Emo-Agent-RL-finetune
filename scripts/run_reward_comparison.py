#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小规模实验：用 train_profile 作为模拟用户，分别用 mode1/mode2/mode3 三种 reward 进行 PPO 训练。

用法：
  python scripts/run_reward_comparison.py --config configs/rl_compare_rewards.yaml
  python scripts/run_reward_comparison.py --config configs/rl_compare_rewards.yaml --steps 100
  python scripts/run_reward_comparison.py --config configs/rl_compare_rewards.yaml --modes mode1 mode3  # 只跑指定模式
"""
import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
if not _cache and os.path.exists("/tmp"):
    _tmp_cache = "/tmp/hf_cache"
    os.makedirs(_tmp_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", _tmp_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", _tmp_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _tmp_cache)

import yaml

from src.training.ppo_emo_trainer import run_ppo_emo_training


def main() -> None:
    parser = argparse.ArgumentParser(description="小规模对比三种 reward 模式的 PPO 训练")
    parser.add_argument("--config", type=str, default="configs/rl_compare_rewards.yaml")
    parser.add_argument("--steps", type=int, default=None, help="覆盖 total_steps，默认用配置里的")
    parser.add_argument("--modes", nargs="+", default=["mode1", "mode2", "mode3"],
                        help="要跑的 reward 模式，默认全跑")
    parser.add_argument("--output-base", type=str, default="outputs/rl_compare",
                        help="输出根目录，各模式会存到 {base}_mode1 等")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    base_dir = args.output_base.rstrip("/")
    results = []

    for mode in args.modes:
        if mode not in ("mode1", "mode2", "mode3"):
            print(f"[WARN] 未知模式 {mode}，跳过")
            continue

        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("reward", {})["type"] = "emo"
        cfg.setdefault("reward", {})["reward_mode"] = mode
        cfg.setdefault("training", {})["output_dir"] = f"{base_dir}_{mode}"

        if args.steps is not None:
            cfg.setdefault("training", {})["total_steps"] = args.steps

        print("\n" + "=" * 70)
        print(f">>> 开始训练: reward_mode={mode}, output_dir={cfg['training']['output_dir']}")
        print(f"    total_steps={cfg['training'].get('total_steps', 'default')}")
        print("=" * 70)

        run_ppo_emo_training(cfg)
        results.append((mode, cfg["training"]["output_dir"]))

    print("\n" + "=" * 70)
    print("【实验完成】三种 reward 模式训练结果目录：")
    for mode, out_dir in results:
        final = os.path.join(out_dir, "final")
        print(f"  {mode}: {final}")
    print("=" * 70)
    print("建议：对各目录下的模型做推理/人工评估，或对比训练日志中的 reward 曲线。")


if __name__ == "__main__":
    main()
