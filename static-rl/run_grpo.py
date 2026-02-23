#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO 训练脚本 - 从 EmpatheticDialogues 采样 prompt，使用 Reward Model 作为奖励。

用法:
  1. 生成偏好数据: python static-rl/build_empathetic_preference_dataset.py
  2. 训练 Reward Model: python static-rl/run_reward_model.py
  3. 运行 GRPO: python static-rl/run_grpo.py --config static-rl/configs/grpo.yaml
"""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "static-rl"))
from reward_model_scorer import build_reward_fn_from_model

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.rl_dataset import load_rl_dataset
from src.models.modeling import load_sft_model, ModelAndTokenizer
from src.training.grpo_training import _generate_completions, _log_probs_for_response


def main():
    parser = argparse.ArgumentParser(description="GRPO training with RM on empathetic prompts")
    parser.add_argument("--config", type=str, default="static-rl/configs/grpo.yaml")
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    rl_cfg = cfg.get("rl", {}).get("grpo", {})
    training_cfg = cfg.get("training", {})
    output_dir = training_cfg.get("output_dir", "static-rl/outputs/grpo")

    # Reward Model
    reward_cfg = cfg.get("reward", {})
    rm_path = reward_cfg.get("reward_model_path", "static-rl/outputs/reward_model")
    rm_full = ROOT / rm_path if not Path(rm_path).is_absolute() else Path(rm_path)
    if not rm_full.exists():
        raise FileNotFoundError(
            f"Reward model not found: {rm_full}. Run: python static-rl/run_reward_model.py"
        )
    reward_fn, _ = build_reward_fn_from_model(str(rm_full))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[GRPO] 加载模型 ...")
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg.get("sft_model_path", "outputs/sft_empathetic/final"),
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
    )
    actor = mt.model
    tokenizer = mt.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    ref_model = copy.deepcopy(actor)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    train_file = data_cfg.get("train_file", "static-rl/data/empathetic_preference.jsonl")
    full_path = ROOT / train_file if not Path(train_file).is_absolute() else Path(train_file)
    dataset = load_rl_dataset(train_file=str(full_path), num_proc=data_cfg.get("num_proc", 4))
    max_prompt_len = data_cfg.get("max_prompt_length", 512)
    max_resp_len = data_cfg.get("max_response_length", 256)

    def collate_fn(batch):
        texts = [b["user"] for b in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_len,
        )
        enc["user"] = texts
        return enc

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    num_generations = rl_cfg.get("num_generations", 4)
    lr = rl_cfg.get("learning_rate", 1e-6)
    epsilon = rl_cfg.get("epsilon", 0.2)
    kl_coef = rl_cfg.get("beta", 0.02)
    temperature = rl_cfg.get("temperature", 1.0)
    top_p = rl_cfg.get("top_p", 1.0)
    total_steps = training_cfg.get("total_steps", 1000)
    logging_steps = training_cfg.get("logging_steps", 10)
    save_steps = training_cfg.get("save_steps", 500)

    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=lr,
    )

    print(f"[GRPO] 开始训练: total_steps={total_steps}, G={num_generations}")
    os.makedirs(output_dir, exist_ok=True)
    data_iter = iter(dataloader)
    global_step = 0
    total_loss_acc = 0.0
    total_reward_acc = 0.0
    t0 = time.time()

    while global_step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt_ids = batch["input_ids"][:1].to(device)
        prompt_mask = batch["attention_mask"][:1].to(device)
        user_texts = batch["user"][:1]
        q_len = prompt_ids.size(1)

        actor.eval()
        resp_ids_list, resp_texts = _generate_completions(
            actor, tokenizer, prompt_ids, prompt_mask,
            num_generations=num_generations, max_new_tokens=max_resp_len,
            temperature=temperature, top_p=top_p,
        )

        reward_fn.set_prompts(user_texts * num_generations)
        rewards = reward_fn(resp_texts)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        r_mean = rewards_t.mean()
        r_std = rewards_t.std().clamp(min=1e-6)
        advantages = (rewards_t - r_mean) / r_std

        actor.train()
        policy_loss = torch.tensor(0.0, device=device)
        kl_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        for i in range(num_generations):
            resp = resp_ids_list[i]
            if resp.numel() == 0:
                continue
            full_ids = torch.cat([prompt_ids[0], resp]).unsqueeze(0)
            full_mask = torch.ones_like(full_ids, dtype=torch.float32)

            actor_lp = _log_probs_for_response(actor, full_ids, full_mask, q_len)
            with torch.no_grad():
                ref_lp = _log_probs_for_response(ref_model, full_ids, full_mask, q_len)
                old_lp = actor_lp.detach()

            min_len = min(actor_lp.size(0), ref_lp.size(0), old_lp.size(0))
            actor_lp = actor_lp[:min_len]
            ref_lp = ref_lp[:min_len]
            old_lp = old_lp[:min_len]

            ratio = torch.exp(actor_lp - old_lp)
            adv = advantages[i]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
            policy_loss = policy_loss - torch.minimum(surr1, surr2).mean()
            kl_loss = kl_loss + (old_lp - ref_lp).mean()
            valid_count += 1

        if valid_count > 0:
            loss = (policy_loss + kl_coef * kl_loss) / valid_count
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in actor.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            total_loss_acc += loss.item()
            total_reward_acc += r_mean.item()

        global_step += 1

        if global_step % logging_steps == 0:
            avg_loss = total_loss_acc / logging_steps
            avg_reward = total_reward_acc / logging_steps
            print(f"  step {global_step}/{total_steps} | loss={avg_loss:.4f} | "
                  f"avg_reward={avg_reward:.4f} | elapsed={time.time()-t0:.0f}s")
            total_loss_acc = 0.0
            total_reward_acc = 0.0

        if save_steps and global_step % save_steps == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
            actor.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  [saved] {ckpt_dir}")

    final_dir = os.path.join(output_dir, "final")
    actor.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[GRPO] 训练完成，模型已保存到 {final_dir}")


if __name__ == "__main__":
    main()
