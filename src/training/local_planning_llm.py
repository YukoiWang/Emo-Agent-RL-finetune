# -*- coding: utf-8 -*-
"""
用本地 SFT 大模型基座执行 planning（情感分析），供 PlayerSimulatorWithPlanning 使用。
其他部分（player_reply、首条消息）正常调 API。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import torch
import os

def build_local_planning_llm_fn(
    model_path: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.5,
) -> Callable[[List[Dict[str, str]]], str]:
    """
    从 SFT 模型路径构建 planning 用的 LLM 调用函数。
    输入: messages [{"role":"user","content":"..."}]
    输出: 生成的文本（用于 planning_reply 解析）
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    # device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     torch_dtype=_dtype,
    #     device_map="auto" if "cuda" in str(device) else None,
    #     trust_remote_code=True,
    # )
    # if "cuda" not in str(device) and hasattr(model, "to"):
    #     model = model.to(device)
    # model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank == 0:
        _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=_dtype,
            trust_remote_code=True,
        )

        model = model.to(device)
        model.eval()

        if world_size > 1 and torch.distributed.is_initialized():
            for param in model.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def fn(messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        content = messages[-1].get("content", "") if messages else ""
        if not content:
            return ""
        # 使用 chat template（Qwen2 等）
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    return fn
