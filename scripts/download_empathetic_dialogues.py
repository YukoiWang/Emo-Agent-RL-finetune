#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 EmpatheticDialogues 数据集并转换为 SFT 所需的 user/assistant 格式，
划分并保存为 train / validation / test jsonl 文件。

EmpatheticDialogues 来自 Facebook Research，包含共情对话，适合用于情感对话 SFT。
Hugging Face: facebook/empathetic_dialogues

用法:
    conda activate emo  # 或你的环境
    python scripts/download_empathetic_dialogues.py
    python scripts/download_empathetic_dialogues.py --output-dir data/empathetic_dialogues

依赖: datasets (pip install datasets)
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset


def _normalize_text(text: str) -> str:
    """EmpatheticDialogues 使用 people_comma_ 等占位符，需还原为正常标点"""
    if not isinstance(text, str):
        return ""
    return text.replace("_comma_", ",").replace("_period_", ".").replace("_exclamation_", "!").replace("_question_", "?").strip()


def convert_to_sft_format(example: dict, include_emotion_in_system: bool = True) -> dict:
    """
    将 EmpatheticDialogues 单行转为 SFT 格式。
    prompt -> user (说话者分享的情境)
    utterance -> assistant (共情回复)
    context -> system 可选 (情绪标签，如 sentimental, excited)
    """
    prompt = _normalize_text(example.get("prompt", ""))
    utterance = _normalize_text(example.get("utterance", ""))
    context = example.get("context", "")

    if not prompt or not utterance:
        return None

    out = {
        "user": prompt,
        "assistant": utterance,
    }
    if include_emotion_in_system and context:
        out["system"] = f"You are an empathetic listener. The user is expressing {context} emotions."
    else:
        out["system"] = ""

    return out


def main():
    parser = argparse.ArgumentParser(description="Download EmpatheticDialogues and split into train/val/test")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/empathetic_dialogues",
        help="Output directory for jsonl files (default: data/empathetic_dialogues)",
    )
    parser.add_argument(
        "--no-emotion-in-system",
        action="store_true",
        help="Do not include emotion context in system prompt",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading facebook/empathetic_dialogues from Hugging Face...")
    ds = load_dataset("facebook/empathetic_dialogues")

    # 数据集已自带 train / validation / test 划分
    splits = {"train": ds["train"], "validation": ds["validation"], "test": ds["test"]}

    for split_name, split_ds in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in split_ds:
                converted = convert_to_sft_format(ex, include_emotion_in_system=not args.no_emotion_in_system)
                if converted is not None:
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
        print(f"  {split_name}: {count} samples -> {out_path}")

    print("\nDone. SFT 配置示例:")
    print("  data:")
    print(f"    train_file: \"{output_dir / 'train.jsonl'}\"")
    print(f"    eval_file: \"{output_dir / 'validation.jsonl'}\"")


if __name__ == "__main__":
    main()
