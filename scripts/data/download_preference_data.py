#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载并处理 IPM-PrefDial 数据集（DecoupledESC 情感支持对话偏好对）。

数据来源: https://github.com/Zc0812/DecoupledESC
论文: DecoupledESC (EMNLP 2025 Findings) - Inferential Preference Mining 构建的偏好对

输出格式: jsonl 每行 {"user": "prompt (ChatML)", "chosen": "优选回复", "rejected": "劣选回复"}
用于 DPO 训练，如 dpo_trainer.run_dpo_training 或 static-rl/run_dpo.py

用法:
  python scripts/data/build_emo_senti_dataset_ch.py
  python scripts/data/build_emo_senti_dataset_ch.py --output data/ipm_prefdial_dpo.jsonl
  python scripts/data/build_emo_senti_dataset_ch.py --variant Qwen_RG --max-samples 5000
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path
from typing import Optional


# DecoupledESC 官方 DPO 数据 URL
DECOUPLEDESC_BASE = "https://raw.githubusercontent.com/Zc0812/DecoupledESC/main/data/Decoupled"
# RG=Response Generation (回复生成), SP=Strategy Planning (策略规划), VM=另一种变体
VARIANTS = {
    "Qwen_RG": f"{DECOUPLEDESC_BASE}/ESC_Qwen_RG_dpo.json",
    "Qwen_SP": f"{DECOUPLEDESC_BASE}/ESC_Qwen_SP_dpo.json",
    "Qwen_VM": f"{DECOUPLEDESC_BASE}/ESC_Qwen_VM_dpo.json",
    "Llama_RG": f"{DECOUPLEDESC_BASE}/ESC_Llama_RG_dpo.json",
    "Llama_SP": f"{DECOUPLEDESC_BASE}/ESC_Llama_SP_dpo.json",
    "Llama_VM": f"{DECOUPLEDESC_BASE}/ESC_Llama_VM_dpo.json",
}


def _conversations_to_chatml(conversations: list) -> str:
    """将 conversations 列表转为 ChatML 格式的 prompt 字符串。"""
    parts = []
    for turn in conversations:
        role = (turn.get("role") or "user").lower()
        content = (turn.get("content") or "").strip()
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>\n")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
    # 结尾加上 <|im_start|>assistant 表示模型需在此处生成
    parts.append("<|im_start|>assistant")
    return "".join(parts)


def _convert_item(item: dict) -> Optional[dict]:
    """将 DecoupledESC 单条记录转为 DPO 格式 (user, chosen, rejected)。"""
    conversations = item.get("conversations")
    chosen = item.get("chosen")
    rejected = item.get("rejected")
    if not conversations or not chosen or not rejected:
        return None
    chosen_content = chosen.get("content", "").strip() if isinstance(chosen, dict) else str(chosen).strip()
    rejected_content = rejected.get("content", "").strip() if isinstance(rejected, dict) else str(rejected).strip()
    if not chosen_content or not rejected_content:
        return None
    user = _conversations_to_chatml(conversations)
    return {"user": user, "chosen": chosen_content, "rejected": rejected_content}


def download_and_process(
    output_path: str = "data/ipm_prefdial_dpo.jsonl",
    variant: str = "Qwen_RG",
    max_samples: int = 0,
    seed: int = 42,
) -> None:
    random.seed(seed)
    url = VARIANTS.get(variant)
    if not url:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(VARIANTS.keys())}")

    print(f"Downloading IPM-PrefDial ({variant}) from DecoupledESC...")
    print(f"  URL: {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to download: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected list of items, got {type(data)}")

    print(f"  Loaded {len(data)} items, converting to DPO format...")
    out = []
    for item in data:
        converted = _convert_item(item)
        if converted:
            out.append(converted)

    if max_samples and len(out) > max_samples:
        out = random.sample(out, max_samples)
        print(f"  Sampled {max_samples} items")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Saved {len(out)} preference pairs to {output_path}")
    print("Format: user (ChatML prompt), chosen (preferred), rejected (non-preferred)")


def main():
    parser = argparse.ArgumentParser(
        description="Download and process IPM-PrefDial (DecoupledESC) for DPO training"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/ipm_prefdial_dpo.jsonl",
        help="Output jsonl path",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="Qwen_RG",
        choices=list(VARIANTS.keys()),
        help="DecoupledESC variant: Qwen_RG (Response Gen), Qwen_SP (Strategy Plan), etc.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to keep (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_and_process(
        output_path=args.output,
        variant=args.variant,
        max_samples=args.max_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
