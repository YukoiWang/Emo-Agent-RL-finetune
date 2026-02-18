from dataclasses import dataclass
from typing import Dict, Any

import torch
from transformers import TrainingArguments
from trl import SFTTrainer

from src.data.sft_dataset import load_sft_dataset
from src.models.modeling import load_base_model, ModelAndTokenizer


@dataclass
class SFTConfig:
    model: Dict[str, Any]
    data: Dict[str, Any]
    training: Dict[str, Any]


def build_training_arguments(training_cfg: Dict[str, Any]) -> TrainingArguments:
    return TrainingArguments(
        output_dir=training_cfg["output_dir"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", training_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=training_cfg.get("learning_rate", 1e-5),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.03),
        logging_steps=training_cfg.get("logging_steps", 50),
        evaluation_strategy="steps",
        eval_steps=training_cfg.get("eval_steps", 500),
        save_steps=training_cfg.get("save_steps", 1000),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        bf16=training_cfg.get("bf16", True),
        report_to=["none"],
    )


def run_sft_training(cfg: Dict[str, Any]) -> None:
    """
    入口函数：根据 YAML 配置执行一次 SFT 训练。
    """
    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]

    # 1. 加载基础模型（可选择 LoRA）
    lora_cfg = model_cfg.get("lora", None) if model_cfg.get("use_lora", True) else None
    mt: ModelAndTokenizer = load_base_model(
        base_model_name=model_cfg["base_model_name"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
    )

    # 2. 加载并 tokenization SFT 数据
    datasets = load_sft_dataset(
        train_file=data_cfg["train_file"],
        eval_file=data_cfg.get("eval_file"),
        tokenizer=mt.tokenizer,
        max_seq_length=data_cfg.get("max_seq_length", 2048),
        num_proc=data_cfg.get("num_proc", 4),
    )

    train_dataset = datasets["train"]
    eval_dataset = datasets.get("validation")

    # 3. 构建 TrainingArguments
    training_args = build_training_arguments(training_cfg)

    # 4. 构建 SFTTrainer
    trainer = SFTTrainer(
        model=mt.model,
        tokenizer=mt.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=None,  # 已经是 tokenized dataset
    )

    # 5. 训练与保存
    trainer.train()
    trainer.save_model(training_cfg["output_dir"])
    mt.tokenizer.save_pretrained(training_cfg["output_dir"])

