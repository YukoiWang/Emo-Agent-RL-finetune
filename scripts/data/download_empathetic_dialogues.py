#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载 EmpatheticDialogues 数据集并转换为 SFT 所需的 user/assistant 格式，
划分并保存为 train / validation / test jsonl 文件。

EmpatheticDialogues 来自 Facebook Research，包含共情对话，适合用于情感对话 SFT。
数据源: https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz
(不依赖 Hugging Face datasets，兼容 datasets 3.0+ 及无外网环境)

用法:
    conda activate emo  # 或你的环境
    python scripts/data/download_empathetic_dialogues.py
    python scripts/data/download_empathetic_dialogues.py --output-dir data/empathetic_dialogues

依赖: 无额外依赖（仅标准库）
"""
import argparse
import csv
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve


# Facebook 官方数据源（与 HF empathetic_dialogues 脚本相同）
_DATA_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
_SPLIT_FILES = {
    "train": "empatheticdialogues/train.csv",
    "validation": "empatheticdialogues/valid.csv",
    "test": "empatheticdialogues/test.csv",
}


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


def _download_and_extract_archives() -> Path:
    """下载 tar.gz 并解压，返回解压后的目录路径"""
    print(f"Downloading from {_DATA_URL} ...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        urlretrieve(_DATA_URL, tmp.name)
        archive_path = tmp.name
    extract_dir = Path(tempfile.mkdtemp())
    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(extract_dir)
    Path(archive_path).unlink()
    return extract_dir


def _iter_csv_examples(csv_path: Path):
    """从 CSV 文件迭代示例，字段与 Hugging Face 版本一致"""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                "prompt": row.get("prompt", ""),
                "utterance": row.get("utterance", ""),
                "context": row.get("context", ""),
            }


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

    extract_dir = _download_and_extract_archives()

    for split_name, rel_path in _SPLIT_FILES.items():
        csv_path = extract_dir / rel_path
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found, skipping {split_name}")
            continue
        out_path = output_dir / f"{split_name}.jsonl"
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in _iter_csv_examples(csv_path):
                converted = convert_to_sft_format(ex, include_emotion_in_system=not args.no_emotion_in_system)
                if converted is not None:
                    f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    count += 1
        print(f"  {split_name}: {count} samples -> {out_path}")

    shutil.rmtree(extract_dir, ignore_errors=True)

    print("\nDone. SFT 配置示例:")
    print("  data:")
    print(f"    train_file: \"{output_dir / 'train.jsonl'}\"")
    print(f"    eval_file: \"{output_dir / 'validation.jsonl'}\"")


if __name__ == "__main__":
    main()
