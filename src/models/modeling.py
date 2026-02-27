from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass
class ModelAndTokenizer:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase


def load_base_model(
    base_model_name: str,
    dtype: str = "bfloat16",
    use_lora: bool = True,
    lora_config: Optional[dict] = None,
) -> ModelAndTokenizer:
    torch_dtype = {
        "float16": "auto",
        "bfloat16": "auto",
        "auto": "auto",
    }.get(dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if use_lora:
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)

    return ModelAndTokenizer(model=model, tokenizer=tokenizer)


def load_sft_model(
    sft_model_path: str,
    dtype: str = "bfloat16",
    use_lora: bool = True,
    lora_config: Optional[dict] = None,
    device_map: Optional[dict] = None,
) -> ModelAndTokenizer:
    """
    从 SFT 输出目录或 HF 模型 ID 加载模型（可继续用于 RL 微调）。
    use_lora=True 时在加载的模型上施加 LoRA，仅训练 LoRA 参数。
    device_map: 可选，多卡 DDP 时应传 {"": local_rank}，确保每 rank 加载到自己的 GPU。

    如果 sft_model_path 是一个 LoRA adapter checkpoint（含 adapter_config.json），
    会先合并 SFT adapter 到 base model，再施加新的 LoRA 用于 RL 训练。
    """
    import os
    from peft import PeftModel

    if device_map is None:
        device_map = {"": 0} if torch.cuda.is_available() else "auto"
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else (torch.float16 if dtype == "float16" else "auto")

    is_adapter_ckpt = os.path.isfile(os.path.join(sft_model_path, "adapter_config.json"))

    if is_adapter_ckpt:
        import json
        with open(os.path.join(sft_model_path, "adapter_config.json"), "r") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path", sft_model_path)
        print(f"[load_sft_model] LoRA adapter detected, base={base_model_name}")

        tokenizer = AutoTokenizer.from_pretrained(sft_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        model = PeftModel.from_pretrained(base_model, sft_model_path)
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(sft_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    if use_lora:
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)

    return ModelAndTokenizer(model=model, tokenizer=tokenizer)

