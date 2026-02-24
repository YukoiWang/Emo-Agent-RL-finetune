#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证：非 static-rl 路径下的 PPO（三种 reward 模式）、GRPO。
跑完后可用 scripts/plot_rl_curves.py 画 KL / reward 曲线，并用 scripts/eval_rl_models.py 评估。

用法:
  # 默认每类小步数验证（PPO 每模式 50 步，GRPO 50 步）
  python scripts/run_quick_verify.py

  # 指定步数、只跑 PPO 三种模式
  python scripts/run_quick_verify.py --ppo-steps 30 --skip-grpo

  # 只跑 GRPO
  python scripts/run_quick_verify.py --skip-ppo
"""
import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 可选：HF 缓存到 /tmp
for env in ("HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"):
    if not os.environ.get(env) and os.path.exists("/tmp"):
        _tmp = "/tmp/hf_cache"
        os.makedirs(_tmp, exist_ok=True)
        os.environ.setdefault(env, _tmp)
        break

import yaml


def main():
    parser = argparse.ArgumentParser(description="快速验证 PPO(mode1/2/3) + GRPO")
    parser.add_argument("--output-base", type=str, default="outputs/quick_verify",
                        help="所有实验的根目录，下面会有 ppo_mode1, ppo_mode2, ppo_mode3, grpo")
    parser.add_argument("--ppo-steps", type=int, default=50, help="PPO 每种 reward 模式的总步数")
    parser.add_argument("--grpo-steps", type=int, default=50, help="GRPO 总步数")
    parser.add_argument("--skip-ppo", action="store_true", help="不跑 PPO 三种模式")
    parser.add_argument("--skip-grpo", action="store_true", help="不跑 GRPO")
    parser.add_argument("--ppo-config", type=str, default="configs/rl_compare_rewards.yaml")
    parser.add_argument("--grpo-config", type=str, default="configs/rl_grpo.yaml")
    args = parser.parse_args()

    base = Path(args.output_base)
    base.mkdir(parents=True, exist_ok=True)

    # ----- PPO：三种 reward 模式 -----
    if not args.skip_ppo:
        config_path = ROOT / args.ppo_config
        if not config_path.exists():
            print(f"[WARN] PPO 配置不存在: {config_path}，跳过 PPO")
        else:
            with config_path.open("r", encoding="utf-8") as f:
                ppo_base = yaml.safe_load(f)
            # 确保 reward 为 emo
            ppo_base.setdefault("reward", {})["type"] = "emo"
            ppo_base.setdefault("training", {})["total_steps"] = args.ppo_steps

            from src.training.ppo_emo_trainer import run_ppo_emo_training

            for mode in ("mode1", "mode2", "mode3"):
                cfg = copy.deepcopy(ppo_base)
                cfg.setdefault("reward", {})["reward_mode"] = mode
                out_dir = str(base / f"ppo_{mode}")
                cfg.setdefault("training", {})["output_dir"] = out_dir
                cfg["training"]["total_steps"] = args.ppo_steps
                print("\n" + "=" * 60)
                print(f">>> PPO reward_mode={mode}, total_steps={args.ppo_steps}, output_dir={out_dir}")
                print("=" * 60)
                run_ppo_emo_training(cfg)

    # ----- GRPO -----
    if not args.skip_grpo:
        config_path = ROOT / args.grpo_config
        if not config_path.exists():
            print(f"[WARN] GRPO 配置不存在: {config_path}，跳过 GRPO")
        else:
            with config_path.open("r", encoding="utf-8") as f:
                grpo_cfg = yaml.safe_load(f)
            out_dir = str(base / "grpo")
            grpo_cfg.setdefault("training", {})["output_dir"] = out_dir
            grpo_cfg.setdefault("training", {})["total_steps"] = args.grpo_steps
            from src.training.grpo_training import run_grpo_training
            print("\n" + "=" * 60)
            print(f">>> GRPO total_steps={args.grpo_steps}, output_dir={out_dir}")
            print("=" * 60)
            run_grpo_training(grpo_cfg)

    print("\n" + "=" * 60)
    print("【快速验证完成】")
    print(f"  输出根目录: {base.absolute()}")
    print("  画图:   python scripts/plot_rl_curves.py --log-dir " + str(base))
    print("  简单推理: python scripts/eval_rl_models.py --model-dir " + str(base))
    print("  全维度评估(Sentient+情绪+综合): python scripts/eval_all_models.py --quick-verify-dir " + str(base))
    print("=" * 60)


if __name__ == "__main__":
    main()
