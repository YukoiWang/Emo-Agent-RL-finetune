#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 EmpatheticDialogues 偏好对数据训练 Reward Model。

用法:
  1. 先生成偏好数据: python static-rl/build_empathetic_preference_dataset.py
  2. LoRA:  python static-rl/run_reward_model.py --config static-rl/configs/reward_model.yaml
  3. 全量:   python static-rl/run_reward_model.py --config static-rl/configs/reward_model_full.yaml
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer


def load_preference_dataset(train_file: str):
    """加载偏好对数据，格式: user, chosen, rejected -> TRL 需要 prompt, chosen, rejected"""
    ds = load_dataset("json", data_files={"train": train_file})["train"]
    prompts, chosens, rejecteds = [], [], []
    for ex in ds:
        user = (ex.get("user") or "").strip()
        chosen = (ex.get("chosen") or "").strip()
        rejected = (ex.get("rejected") or "").strip()
        if user and chosen and rejected:
            prompts.append(user)
            chosens.append(chosen)
            rejecteds.append(rejected)
    from datasets import Dataset
    return Dataset.from_dict({"prompt": prompts, "chosen": chosens, "rejected": rejecteds})


def main():
    parser = argparse.ArgumentParser(description="Train reward model on empathetic preference data")
    parser.add_argument("--config", type=str, default=str(ROOT / "static-rl/configs/reward_model.yaml"))
    args = parser.parse_args()

    config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rm_cfg = cfg.get("reward_model", {})
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    train_file = data_cfg.get("train_file", "static-rl/data/empathetic_preference.jsonl")
    full_train = Path(train_file) if Path(train_file).is_absolute() else ROOT / train_file
    if not full_train.exists():
        raise FileNotFoundError(
            f"Preference data not found: {full_train}. "
            "Run: python static-rl/build_empathetic_preference_dataset.py"
        )
    full_train = str(full_train)
    dataset = load_preference_dataset(full_train)
    print(f"Loaded {len(dataset)} preference pairs")

    model_name = rm_cfg.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")
    output_dir = training_cfg.get("output_dir", "static-rl/outputs/reward_model")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if rm_cfg.get("dtype", "bfloat16") == "bfloat16" else torch.float16,
        device_map="auto",
    )

    peft_config = None
    if rm_cfg.get("use_lora", True):  # use_lora: false 时为全量微调
        peft_config = LoraConfig(
            r=rm_cfg.get("lora_r", 16),
            lora_alpha=rm_cfg.get("lora_alpha", 32),
            lora_dropout=rm_cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["score"],  # 保存分类头
        )

    reward_config = RewardConfig(
        output_dir=output_dir,
        num_train_epochs=training_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=rm_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=rm_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=rm_cfg.get("learning_rate", 1e-5),
        logging_steps=training_cfg.get("logging_steps", 50),
        save_steps=training_cfg.get("save_steps", 500),
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    os.makedirs(output_dir, exist_ok=True)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[Reward Model] 训练完成，已保存到 {output_dir}")


if __name__ == "__main__":
    main()
