from dataclasses import dataclass
from typing import Optional, Tuple

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
) -> ModelAndTokenizer:
    """
    从 SFT 输出目录或 HF 模型 ID 加载模型（可继续用于 RL 微调）。
    use_lora=True 时在加载的模型上施加 LoRA，仅训练 LoRA 参数。
    """
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        sft_model_path,
        torch_dtype="auto" if dtype in ("bfloat16", "float16", "auto") else "auto",
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
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            }
        peft_config = LoraConfig(**lora_config)
        model = get_peft_model(model, peft_config)

    return ModelAndTokenizer(model=model, tokenizer=tokenizer)

