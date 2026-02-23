# -*- coding: utf-8 -*-
"""
EmpatheticDialogues SFT 脚本，使用 Qwen2.5-7B-Instruct 进行共情对话微调。

用法:
    # 1. 先下载数据（若未下载）
    python scripts/download_empathetic_dialogues.py

    # 2. 运行 SFT
    python scripts/run_sft_empathetic.py
    python scripts/run_sft_empathetic.py --config configs/sft_empathetic.yaml
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

from src.training.sft_trainer import run_sft_training


def main():
    parser = argparse.ArgumentParser(description="EmpatheticDialogues SFT（Qwen2.5-7B + LoRA）")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(ROOT, "configs", "sft_empathetic.yaml"),
        help="Path to YAML config",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 解析数据路径（相对路径转为绝对路径）
    data_cfg = cfg.get("data", {})
    train_file = data_cfg.get("train_file", "")
    eval_file = data_cfg.get("eval_file", "")

    if train_file and not os.path.isabs(train_file):
        train_file = os.path.join(ROOT, train_file)
    if train_file and not os.path.isfile(train_file):
        print(f"警告: 训练数据不存在 {train_file}")
        print("请先运行: python scripts/download_empathetic_dialogues.py")
        sys.exit(1)
    cfg["data"]["train_file"] = train_file

    if eval_file and not os.path.isabs(eval_file):
        eval_file = os.path.join(ROOT, eval_file)
    cfg["data"]["eval_file"] = eval_file

    run_sft_training(cfg)
    out_dir = cfg.get("training", {}).get("output_dir", "outputs/sft_empathetic")
    print(f"SFT 完成。模型保存在 {out_dir}，可作为 RL 的 sft_model_path。")


if __name__ == "__main__":
    main()
