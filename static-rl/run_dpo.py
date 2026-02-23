#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO 训练脚本 - 使用 EmpatheticDialogues 偏好对数据微调。

用法:
  1. 先生成偏好数据: python static-rl/build_empathetic_preference_dataset.py
  2. 运行 DPO: python static-rl/run_dpo.py --config static-rl/configs/dpo.yaml
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from src.training.dpo_trainer import run_dpo_training


def main():
    parser = argparse.ArgumentParser(description="DPO training on empathetic preference data")
    parser.add_argument("--config", type=str, default=str(ROOT / "static-rl/configs/dpo.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_dpo_training(cfg)


if __name__ == "__main__":
    main()
