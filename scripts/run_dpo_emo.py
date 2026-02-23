#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
On-policy DPO 训练入口：用 train_profile 模拟用户，每轮生成 k 个回复，
情感打分选 best/worst 构造偏好对，再 DPO 训练。

用法：
  python scripts/run_dpo_emo.py --config configs/rl_dpo_emo.yaml

需设置 DEEPSEEK_API_KEY 或 DASHSCOPE_API_KEY 供用户模拟器调用。
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_cache = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
if not _cache and os.path.exists("/tmp"):
    _tmp = "/tmp/hf_cache"
    os.makedirs(_tmp, exist_ok=True)
    os.environ.setdefault("HF_HOME", _tmp)
    os.environ.setdefault("TRANSFORMERS_CACHE", _tmp)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", _tmp)

import yaml

from src.training.dpo_emo_trainer import run_dpo_emo_training


def main() -> None:
    parser = argparse.ArgumentParser(description="On-policy DPO (train_profile 模拟用户)")
    parser.add_argument("--config", type=str, default="configs/rl_dpo_emo.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dpo_emo_training(cfg)


if __name__ == "__main__":
    main()
