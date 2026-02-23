#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 IPM-PrefDial (DecoupledESC) 数据集转换为 DPO 训练用的偏好对格式。

数据来源: https://github.com/Zc0812/DecoupledESC
格式: {"user": "prompt", "chosen": "preferred_response", "rejected": "rejected_response"}

用法:
  # 从网络下载并转换（默认使用 ESC_Qwen_RG_dpo.json - 共情回复生成）
  python scripts/convert_ipm_prefdial.py -o data/ipm_prefdial_dpo.jsonl

  # 指定本地 JSON 文件
  python scripts/convert_ipm_prefdial.py -i /path/to/ESC_Qwen_RG_dpo.json -o data/ipm_prefdial_dpo.jsonl

  # 指定模型以使用其 chat template
  python scripts/convert_ipm_prefdial.py -o data/ipm_prefdial_dpo.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
"""
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# DecoupledESC 数据 URL
DECOUPLED_ESC_BASE = "https://raw.githubusercontent.com/Zc0812/DecoupledESC/main/data/Decoupled"
AVAILABLE_FILES = [
    "ESC_Qwen_RG_dpo.json",   # Response Generation (推荐)
    "ESC_Qwen_SP_dpo.json",   # Strategy Planning
    "ESC_Qwen_VM_dpo.json",   # (另一子任务)
    "ESC_Llama_RG_dpo.json",
    "ESC_Llama_SP_dpo.json",
    "ESC_Llama_VM_dpo.json",
]


def _format_prompt_simple(conversations: List[Dict[str, str]]) -> str:
    """简单格式：不用 tokenizer，手动拼接为通用对话格式。"""
    parts = []
    for msg in conversations:
        role = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role == "system":
            parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        elif role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _format_prompt_with_tokenizer(conversations: List[Dict[str, str]], tokenizer) -> str:
    """使用 tokenizer 的 chat template 生成 prompt。"""
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(prompt, list):
            prompt = tokenizer.decode(prompt, skip_special_tokens=False)
        return prompt
    return _format_prompt_simple(conversations)


def convert_item(
    item: Dict[str, Any],
    tokenizer=None,
) -> Dict[str, str]:
    """
    将 DecoupledESC 单条样本转为 {"user", "chosen", "rejected"} 格式。
    """
    conversations = item.get("conversations", [])
    chosen_obj = item.get("chosen", {})
    rejected_obj = item.get("rejected", {})

    chosen_content = chosen_obj.get("content", "") if isinstance(chosen_obj, dict) else str(chosen_obj)
    rejected_content = rejected_obj.get("content", "") if isinstance(rejected_obj, dict) else str(rejected_obj)

    if not conversations or not chosen_content or not rejected_content:
        return None

    if tokenizer is not None:
        prompt = _format_prompt_with_tokenizer(conversations, tokenizer)
    else:
        prompt = _format_prompt_simple(conversations)

    return {
        "user": prompt.strip(),
        "chosen": chosen_content.strip(),
        "rejected": rejected_content.strip(),
    }


def load_json_source(path_or_url: str) -> List[Dict[str, Any]]:
    """从本地路径或 URL 加载 JSON。"""
    if path_or_url.startswith(("http://", "https://")):
        try:
            import urllib.request
            with urllib.request.urlopen(path_or_url, timeout=60) as resp:
                return json.load(resp)
        except Exception as e:
            raise RuntimeError(f"下载失败 {path_or_url}: {e}") from e
    path = Path(path_or_url)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 IPM-PrefDial (DecoupledESC) 转为 DPO 偏好对 jsonl"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help=f"输入 JSON 路径或 URL。不指定则下载 {AVAILABLE_FILES[0]}",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/ipm_prefdial_dpo.jsonl",
        help="输出 jsonl 路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="用于 chat template 的模型（可选，不指定则用简单格式）",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="不使用 tokenizer，用简单 Qwen 格式",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多转换条数，0 表示全部",
    )
    args = parser.parse_args()

    input_src = args.input
    if not input_src:
        input_src = f"{DECOUPLED_ESC_BASE}/{AVAILABLE_FILES[0]}"
        print(f"[INFO] 使用默认数据: {input_src}")

    print(f"[INFO] 加载数据: {input_src}")
    raw = load_json_source(input_src)
    if not isinstance(raw, list):
        raw = [raw]
    print(f"[INFO] 共 {len(raw)} 条原始样本")

    tokenizer = None
    if not args.no_tokenizer and AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            print(f"[INFO] 使用模型 chat template: {args.model}")
        except Exception as e:
            print(f"[WARN] 加载 tokenizer 失败，改用简单格式: {e}")
            tokenizer = None

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(raw):
            if args.max_samples and count >= args.max_samples:
                break
            converted = convert_item(item, tokenizer=tokenizer)
            if converted is None:
                skipped += 1
                continue
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"[INFO] 转换完成: {count} 条，跳过 {skipped} 条")
    print(f"[INFO] 输出: {out_path}")
    print()
    print("使用方式: 在 config 中设置 data.train_file 为该文件，rl.algo 为 dpo")


if __name__ == "__main__":
    main()
