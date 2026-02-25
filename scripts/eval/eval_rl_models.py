#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 quick_verify 或任意 RL 实验目录下的 final 模型做简单推理评估：用固定 prompt 生成回复并打印，便于人工对比。

用法:
  python scripts/eval/eval_rl_models.py --model-dir outputs/quick_verify
  python scripts/eval/eval_rl_models.py --model-dir outputs/quick_verify --prompts "我最近工作压力很大" "家里出了点事，心情不好"
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 若 final 为 LoRA 权重，需先加载 base 再挂 adapter
try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    _HAS_PEFT = False


def load_model_and_tokenizer(model_dir: Path, device: str = "cuda", base_model_name: str = None):
    """加载 final 目录下的模型（支持完整权重或 LoRA adapter）。"""
    model_dir = str(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    adapter_config = Path(model_dir) / "adapter_config.json"
    base_model_name = base_model_name or "Qwen/Qwen2.5-1.5B-Instruct"
    if _HAS_PEFT and adapter_config.exists():
        import json
        with open(adapter_config, "r") as f:
            base_name = json.load(f).get("base_model_name_or_path") or base_model_name
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
    if device == "cpu":
        model = model.to("cpu")
    return model, tokenizer


def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "cuda"):
    """单条 prompt 生成回复（对话格式按 Qwen 模板）。"""
    # 简单包装成对话
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    if device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=False)
    # 只取 assistant 部分
    if "<|im_start|>assistant" in full:
        full = full.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in full:
        full = full.split("<|im_end|>")[0]
    return full.strip()


def main():
    parser = argparse.ArgumentParser(description="对 RL 实验的 final 模型做简单推理评估")
    parser.add_argument("--model-dir", type=str, default="outputs/quick_verify",
                        help="实验根目录，下含 ppo_mode1/final, ppo_mode2/final, grpo/final 等")
    parser.add_argument("--prompts", nargs="+", default=["我最近工作压力很大，不知道该怎么办。"],
                        help="用于测试的 prompt 列表")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base-model", type=str, default=None,
                        help="LoRA 的 base 模型名，默认从 adapter_config 读或 Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    base = Path(args.model_dir)
    if not base.is_dir():
        print(f"目录不存在: {base}")
        return

    # 子目录名 -> 显示名
    candidates = [
        ("ppo_mode1", "PPO (mode1)"),
        ("ppo_mode2", "PPO (mode2)"),
        ("ppo_mode3", "PPO (mode3)"),
        ("grpo", "GRPO"),
    ]

    device = args.device if torch.cuda.is_available() else "cpu"
    for sub, label in candidates:
        final_dir = base / sub / "final"
        if not final_dir.is_dir():
            continue
        print("\n" + "=" * 60)
        print(f"模型: {label} ({final_dir})")
        print("=" * 60)
        try:
            model, tokenizer = load_model_and_tokenizer(final_dir, device, args.base_model)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue
        for p in args.prompts:
            reply = generate_reply(model, tokenizer, p, args.max_new_tokens, device)
            print(f"  Prompt: {p}")
            print(f"  Reply:  {reply[:400]}{'...' if len(reply) > 400 else ''}")
            print()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("评估结束。")


if __name__ == "__main__":
    main()
