# -*- coding: utf-8 -*-
"""
Planning（情感分析）LLM：优先 API（planning_service / deepseek 等），调用失败时回退到本地 SFT。
本地 planner 仅在 rank 0 加载，不占多卡显存；多卡且只用本地时非 rank 0 会报错提示用 API。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import torch


def build_local_planning_llm_fn(
    model_path: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: str = "bfloat16",
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 0.5,
    process_index: int = 0,
    world_size: int = 1,
) -> Callable[[List[Dict[str, str]]], str]:
    """
    从 SFT 模型路径构建 planning 用的 LLM 调用函数。
    **仅 rank 0 加载模型**；多卡时其他 rank 返回的 fn 会报错（提示使用 API 做 planning）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rank = process_index
    if world_size > 1 and rank != 0:
        # 多卡且非 rank 0：不加载模型，返回的 fn 调用时直接报错
        def _fn_no_model(_messages: List[Dict[str, str]]) -> str:
            raise RuntimeError(
                "Local planning model is only loaded on rank 0. "
                "For multi-GPU, set planning_service_url or planning_llm (e.g. deepseek) in config to use API."
            )
        return _fn_no_model

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    _device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(_device, str) and _device.startswith("cuda"):
        _device = torch.device(_device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype,
        trust_remote_code=True,
    )
    model = model.to(_device)
    model.eval()

    def fn(messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        content = messages[-1].get("content", "") if messages else ""
        if not content:
            return ""
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


def build_planning_llm_fn_prefer_api_then_local(
    rollout_cfg: dict,
    model_cfg: dict,
    device,
    process_index: int = 0,
    world_size: int = 1,
):
    """
    优先 API（planning_service_url / planning_llm=deepseek|qwen|openai），
    调用失败时再回退到本地 SFT（仅 rank 0 加载）。使用 API 时不会加载本地 planner。

    返回: planning_llm_fn 或 None。
    """
    planning_service_url = rollout_cfg.get("planning_service_url")
    planning_llm_kind = rollout_cfg.get("planning_llm")
    sft_model_path = model_cfg.get("sft_model_path")

    api_fn = None
    local_fn = None

    # 1) 优先 API：有 URL 或用 API 类 planning_llm 时只建 API，不加载本地
    if planning_service_url:
        from .planning_service_client import build_planning_service_llm_fn
        api_fn = build_planning_service_llm_fn(planning_service_url)
    elif planning_llm_kind in ("deepseek", "qwen", "openai"):
        from .qwen_user_simulator import build_qwen_user_llm_fn
        api_fn = build_qwen_user_llm_fn(
            model=rollout_cfg.get("planning_llm_model", "deepseek-chat"),
            temperature=rollout_cfg.get("planning_llm_temperature", 0.5),
        )

    # 2) 本地作为回退：仅在有 sft_model_path 时构建，且只让 rank 0 加载。
    #    多卡时非 rank 0 不建 local_fn，避免 API 失败时 fallback 到会报错的 stub。
    if sft_model_path and (world_size <= 1 or process_index == 0):
        local_fn = build_local_planning_llm_fn(
            sft_model_path,
            device=device,
            process_index=process_index,
            world_size=world_size,
        )

    if api_fn is not None and local_fn is not None:
        # 优先 API，失败再本地（仅 rank 0 有 local_fn，非 rank 0 会再抛错）
        def _wrapper(messages: List[Dict[str, str]]) -> str:
            try:
                return api_fn(messages)
            except Exception as e:
                if local_fn is not None:
                    return local_fn(messages)
                raise e
        return _wrapper
    if api_fn is not None:
        return api_fn
    if local_fn is not None:
        return local_fn
    return None
