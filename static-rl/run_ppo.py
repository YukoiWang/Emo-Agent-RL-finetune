#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO 训练脚本 - 从 EmpatheticDialogues 采样 prompt，使用 Reward Model 作为奖励。

用法:
  1. 生成偏好数据: python static-rl/build_empathetic_preference_dataset.py
  2. 训练 Reward Model: python static-rl/run_reward_model.py
  3. 运行 PPO: python static-rl/run_ppo.py --config static-rl/configs/ppo.yaml

SFT 模型路径需先完成 empathetic SFT；未完成前可先用虚拟路径。
"""
import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.data.rl_dataset import load_rl_dataset
from src.models.modeling import load_sft_model, ModelAndTokenizer

# static-rl 不能作为包名，直接导入同目录模块
sys.path.insert(0, str(ROOT / "static-rl"))
from reward_model_scorer import build_reward_fn_from_model


def build_ppo_config(cfg: dict) -> PPOConfig:
    rl = cfg.get("rl", {}).get("ppo", {})
    return PPOConfig(
        batch_size=rl.get("batch_size", 8),
        mini_batch_size=rl.get("mini_batch_size", 2),
        ppo_epochs=rl.get("ppo_epochs", 4),
        learning_rate=rl.get("learning_rate", 1e-6),
        gamma=rl.get("gamma", 1.0),
        lam=rl.get("lam", 0.95),
        cliprange=rl.get("clip_range", 0.2),
        cliprange_value=rl.get("clip_range_value", 0.2),
        init_kl_coef=rl.get("kl_penalty_coef", 0.02),
        gradient_checkpointing=rl.get("gradient_checkpointing", False),
    )


def main():
    parser = argparse.ArgumentParser(description="PPO training with RM on empathetic prompts")
    parser.add_argument("--config", type=str, default="static-rl/configs/ppo.yaml")
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    training_cfg = cfg.get("training", {})
    output_dir = training_cfg.get("output_dir", "static-rl/outputs/ppo")
    total_steps = training_cfg.get("total_steps", 0)
    save_steps = training_cfg.get("save_steps", 500)
    save_total_limit = training_cfg.get("save_total_limit", 3)

    # Reward Model
    reward_cfg = cfg.get("reward", {})
    rm_path = reward_cfg.get("reward_model_path", "static-rl/outputs/reward_model")
    rm_full = ROOT / rm_path if not Path(rm_path).is_absolute() else Path(rm_path)
    if not rm_full.exists():
        raise FileNotFoundError(
            f"Reward model not found: {rm_full}. "
            "Run: python static-rl/run_reward_model.py"
        )
    reward_fn, _ = build_reward_fn_from_model(str(rm_full))

    # SFT 模型
    sft_path = model_cfg.get("sft_model_path", "outputs/sft_empathetic/final")
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=sft_path,
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
    )

    rl_cfg = cfg.get("rl", {}).get("ppo", {})
    if rl_cfg.get("gradient_checkpointing", False):
        if hasattr(mt.model, "gradient_checkpointing_enable"):
            mt.model.gradient_checkpointing_enable()

    model = AutoModelForCausalLMWithValueHead.from_pretrained(mt.model)
    tokenizer = mt.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # 数据集：使用偏好数据的 user 作为 prompt（与 EmpatheticDialogues 一致）
    train_file = data_cfg.get("train_file", "static-rl/data/empathetic_preference.jsonl")
    full_path = ROOT / train_file if not Path(train_file).is_absolute() else Path(train_file)
    dataset = load_rl_dataset(
        train_file=str(full_path),
        num_proc=data_cfg.get("num_proc", 4),
    )

    max_prompt_len = data_cfg.get("max_prompt_length", 512)
    max_resp_len = data_cfg.get("max_response_length", 256)

    def tokenize_sample(sample):
        enc = tokenizer(
            sample["user"],
            truncation=True,
            max_length=max_prompt_len,
            padding=False,
            return_tensors=None,
        )
        sample["input_ids"] = enc["input_ids"]
        sample["query"] = sample["user"]
        return sample

    dataset = dataset.map(tokenize_sample, num_proc=data_cfg.get("num_proc", 4), desc="tokenize")
    dataset.set_format(type=None, columns=["input_ids", "query"])

    def collator(data):
        return {k: [d[k] for d in data] for k in data[0]}

    ppo_config = build_ppo_config(cfg)
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    gen_kwargs = {
        "do_sample": True,
        "top_p": 1.0,
        "temperature": 1.0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "max_new_tokens": max_resp_len,
    }

    def _save():
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        model.pretrained_model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  [saved] {ckpt_dir}")

    def _prune():
        if save_total_limit <= 0:
            return
        ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")),
                      key=lambda p: int(p.split("-")[-1]))
        while len(ckpts) > save_total_limit:
            shutil.rmtree(ckpts.pop(0), ignore_errors=True)
            print(f"  [pruned] {ckpts[0] if ckpts else '...'}")

    os.makedirs(output_dir, exist_ok=True)
    global_step = 0

    for batch in ppo_trainer.dataloader:
        if total_steps and global_step >= total_steps:
            break

        reward_fn.set_prompts(batch["query"])
        query_tensors = [torch.tensor(ids).to(model.pretrained_model.device) for ids in batch["input_ids"]]
        responses = ppo_trainer.generate(query_tensors, **gen_kwargs)
        response_tensors = [r.squeeze() for r in responses]
        texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        reward_values = reward_fn(texts)
        reward_tensors = [torch.tensor(r, dtype=torch.float32).to(model.pretrained_model.device) for r in reward_values]
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        global_step += 1
        if save_steps and global_step % save_steps == 0:
            _save()
            _prune()

    final_dir = os.path.join(output_dir, "final")
    model.pretrained_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[PPO] 训练完成，模型已保存到 {final_dir}")


if __name__ == "__main__":
    main()
