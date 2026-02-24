# -*- coding: utf-8 -*-
"""
DPO (Direct Preference Optimization) 训练。

使用偏好对数据 (user/prompt, chosen, rejected) 进行训练。
支持 IPM-PrefDial (DecoupledESC) 等情感支持对话偏好数据。
"""
from __future__ import annotations

import os
from typing import Any, Dict

import torch
from trl import DPOConfig, DPOTrainer

from src.data.rl_dataset import load_rl_dataset
from src.models.modeling import load_sft_model, ModelAndTokenizer


def run_dpo_training(cfg: Dict[str, Any]) -> None:
    """
    使用 DPO 在偏好对数据上微调 SFT 模型。

    数据格式（jsonl 每行）：
      {"user": "prompt（对话上下文）", "chosen": "优选回复", "rejected": "劣选回复"}
    """
    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    dpo_cfg = cfg.get("rl", {}).get("dpo", {})
    training_cfg = cfg.get("training", {})
    output_dir = training_cfg.get("output_dir", "outputs/dpo")
    num_train_epochs = training_cfg.get("num_train_epochs", 1)
    max_steps = training_cfg.get("max_steps", -1)
    save_steps = training_cfg.get("save_steps", 500)
    logging_steps = training_cfg.get("logging_steps", 50)

    # 1. 加载 SFT 模型（DPO 不需要 value head）
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
    )
    model = mt.model
    tokenizer = mt.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # 2. 加载偏好数据集
    dataset = load_rl_dataset(
        train_file=data_cfg["train_file"],
        num_proc=data_cfg.get("num_proc", 4),
        format="standard",
    )

    # TRL DPOTrainer 期望 "prompt" 列，我们的数据是 "user"
    if "user" in dataset.column_names and "prompt" not in dataset.column_names:
        dataset = dataset.rename_column("user", "prompt")

    # 检查必需列
    for col in ["prompt", "chosen", "rejected"]:
        if col not in dataset.column_names:
            raise ValueError(
                f"DPO 数据集需要 '{col}' 列。当前列: {dataset.column_names}。"
                "请提供 (prompt/user, chosen, rejected) 格式的偏好对 jsonl。"
            )

    # 3. DPO 配置
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg.get("beta", 0.1),
        learning_rate=dpo_cfg.get("learning_rate", 2.0e-6),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        save_steps=save_steps,
        logging_steps=logging_steps,
        bf16=model_cfg.get("dtype", "bfloat16") == "bfloat16",
        fp16=model_cfg.get("dtype") == "float16",
        remove_unused_columns=False,
    )

    # 4. DPOTrainer
    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=dataset,
    )

    # 5. 训练
    os.makedirs(output_dir, exist_ok=True)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DPO] 训练完成，模型已保存到 {output_dir}")
