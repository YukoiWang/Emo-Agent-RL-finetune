# -*- coding: utf-8 -*-
"""
在 RL 微调之前，先用心理咨询对话数据集做 SFT。
用法：python scripts/sft/run_sft_counseling.py [--config configs/sft_counseling.yaml]
数据：config 中 train_file / eval_file 指向的 jsonl，每行需含 "user", "assistant"，可选 "system"。
     若使用 HuggingFace 数据集，可先将数据转为上述 jsonl 或修改 load_sft_dataset 支持 HF 名。
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml

from src.training.sft_trainer import run_sft_training


def main():
    parser = argparse.ArgumentParser(description="心理咨询对话 SFT（RL 前）")
    parser.add_argument("--config", type=str, default=os.path.join(ROOT, "configs", "sft_counseling.yaml"))
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_file = cfg.get("data", {}).get("train_file", "")
    if train_file and not os.path.isabs(train_file):
        train_file = os.path.join(ROOT, train_file)
    if train_file and not os.path.isfile(train_file):
        print(f"警告: 训练数据不存在 {train_file}，请准备 jsonl（每行含 user, assistant[, system]）或修改 config 中的 train_file。")
        print("可先创建占位目录: mkdir -p data/counseling_dialogue")
        print("或使用现有 SFT 数据路径修改 configs/sft_counseling.yaml 中的 data.train_file。")
    if train_file:
        cfg["data"]["train_file"] = train_file
    eval_file = cfg.get("data", {}).get("eval_file", "")
    if eval_file and not os.path.isabs(eval_file):
        eval_file = os.path.join(ROOT, eval_file)
    if eval_file:
        cfg["data"]["eval_file"] = eval_file

    run_sft_training(cfg)
    print("心理咨询 SFT 完成。将 outputs/sft_counseling（或 config 中 output_dir）作为 RL 的 sft_model_path。")


if __name__ == "__main__":
    main()
