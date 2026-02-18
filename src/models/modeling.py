from dataclasses import dataclass
from typing import Tuple

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
    lora_config: dict | None = None,
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
) -> ModelAndTokenizer:
    """
    从 SFT 输出目录加载模型（可继续用于 RL 微调）。
    """
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype="auto" if dtype in ("bfloat16", "float16", "auto") else "auto",
        device_map="auto",
    )

    # 如果之前是 LoRA 训练，这里会自动加载 PEFT 权重
    if use_lora:
        # Hugging Face PEFT 通常在 from_pretrained 时自动处理
        pass

    return ModelAndTokenizer(model=model, tokenizer=tokenizer)

