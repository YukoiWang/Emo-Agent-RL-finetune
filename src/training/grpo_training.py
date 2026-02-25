# -*- coding: utf-8 -*-
"""
GRPO (Group Relative Policy Optimization) 训练模块。

手动实现，不依赖 trl.GRPOTrainer（trl 0.11 无此类）。

核心思路：
- 每个 prompt 采样 G 个 completion
- 组内相对 advantage: A_i = (R_i - mean(R)) / (std(R) + eps)
- Clipped policy gradient + KL penalty (against frozen ref)
- 无需 Critic 网络

多卡训练：使用 `accelerate launch scripts/rl/run_rl.py --config xxx.yaml`
"""
from __future__ import annotations

import copy
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.data.rl_dataset import load_rl_dataset
from src.models.modeling import load_sft_model, ModelAndTokenizer


def _log_probs_for_response(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_start: int,
) -> torch.Tensor:
    """
    对 full sequence (query+response) 做 forward，返回 response 部分每个 token 的 log_prob。
    返回: (response_len,)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:].unsqueeze(-1)
    token_lp = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)
    return token_lp[0, response_start - 1:]


@torch.no_grad()
def _generate_completions(
    model, tokenizer, prompt_ids: torch.Tensor, prompt_mask: torch.Tensor,
    num_generations: int, max_new_tokens: int,
    temperature: float = 1.0, top_p: float = 1.0,
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    对单个 prompt 生成 num_generations 个 completion。
    返回 (response_ids_list, response_texts)。
    """
    pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)
    prompt_ids_batch = prompt_ids.expand(num_generations, -1)
    prompt_mask_batch = prompt_mask.expand(num_generations, -1)

    gen_out = model.generate(
        input_ids=prompt_ids_batch,
        attention_mask=prompt_mask_batch,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=pad_id,
    )
    generated = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
    q_len = prompt_ids.size(1)
    resp_ids_list = []
    resp_texts = []
    for i in range(num_generations):
        r = generated[i, q_len:]
        valid_mask = (r != pad_id)
        valid_len = valid_mask.sum().item()
        r = r[:max(1, valid_len)]
        resp_ids_list.append(r)
        resp_texts.append(tokenizer.decode(r, skip_special_tokens=True))
    return resp_ids_list, resp_texts


def run_grpo_training(
    cfg: Dict[str, Any],
    reward_fn: Optional[Callable[[List[str]], List[float]]] = None,
) -> None:
    """
    GRPO 训练主函数。

    每步：
    1. 取一个 prompt，生成 G 个 completion
    2. 用 reward_fn 打分
    3. 组内归一化得 advantage
    4. Clipped policy gradient loss + KL penalty
    5. 反向传播更新 actor
    """
    torch.manual_seed(cfg.get("seed", 42))

    accelerator = Accelerator()
    device = accelerator.device

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    rl_cfg = cfg.get("rl", {}).get("grpo", {})
    training_cfg = cfg.get("training", {})

    if accelerator.is_main_process:
        print(f"[GRPO] Accelerate: {accelerator.num_processes} process(es), device={device}")

    # -- Reward --
    if reward_fn is None:
        reward_cfg = cfg.get("reward", {}) or {}
        if reward_cfg.get("type") == "emo":
            from src.training.reward_emo import build_reward_fn_emo
            reward_fn = build_reward_fn_emo(
                emo_adapter_path=reward_cfg.get("emo_adapter_path"),
                reward_mode=reward_cfg.get("reward_mode", "mode1"),
            )
        else:
            from src.training.rl_trainer import simple_empathy_reward_fn
            reward_fn = simple_empathy_reward_fn

    # -- Model --
    print("[GRPO] 加载模型 ...")
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
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

    # -- Dataset --
    print("[GRPO] 加载数据集 ...")
    dataset = load_rl_dataset(
        train_file=data_cfg["train_file"],
        num_proc=data_cfg.get("num_proc", 4),
    )
    max_prompt_len = data_cfg.get("max_prompt_length", 512)
    max_resp_len = data_cfg.get("max_response_length", 64)

    def collate_fn(batch):
        texts = [b["user"] for b in batch]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_prompt_len,
        )
        return enc

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

    # -- Hyperparams --
    num_generations = rl_cfg.get("num_generations", 4)
    lr = rl_cfg.get("learning_rate", 1e-6)
    epsilon = rl_cfg.get("epsilon", 0.2)
    kl_coef = rl_cfg.get("beta", 0.02)
    temperature = rl_cfg.get("temperature", 1.0)
    top_p = rl_cfg.get("top_p", 1.0)
    total_steps = training_cfg.get("total_steps", 100)
    logging_steps = training_cfg.get("logging_steps", 10)
    save_steps = training_cfg.get("save_steps", 500)
    output_dir = training_cfg.get("output_dir", "outputs/grpo")

    # -- Optimizer & Accelerate prepare（多卡时自动 DDP）--
    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=lr,
    )
    actor, optimizer, dataloader = accelerator.prepare(actor, optimizer, dataloader)
    # ref_model 不参与 DDP，保持在各卡本地
    ref_model = ref_model.to(device)

    # -- Training loop --
    if accelerator.is_main_process:
        print(f"[GRPO] 开始训练: total_steps={total_steps}, G={num_generations}, lr={lr}, eps={epsilon}")
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    log_path = os.path.join(output_dir, "training_log.jsonl")
    log_file = open(log_path, "w", encoding="utf-8") if accelerator.is_main_process else None
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
        q_len = prompt_ids.size(1)

        # 1) 生成 G 个 completion
        actor.eval()
        resp_ids_list, resp_texts = _generate_completions(
            actor, tokenizer, prompt_ids, prompt_mask,
            num_generations=num_generations, max_new_tokens=max_resp_len,
            temperature=temperature, top_p=top_p,
        )

        # 2) 打分
        rewards = reward_fn(resp_texts)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # 3) 组内归一化 advantage
        r_mean = rewards_t.mean()
        r_std = rewards_t.std().clamp(min=1e-6)
        advantages = (rewards_t - r_mean) / r_std

        # 4) 计算 loss
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

            kl = (old_lp - ref_lp).mean()
            kl_loss = kl_loss + kl

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
            kl_avg = (kl_loss / valid_count).item()
            if accelerator.is_main_process and log_file is not None:
                log_record = {
                    "step": global_step,
                    "reward_mean": r_mean.item(),
                    "loss": loss.item(),
                    "kl_loss": kl_avg,
                }
                log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                log_file.flush()

        global_step += 1

        if accelerator.is_main_process and global_step % logging_steps == 0:
            avg_loss = total_loss_acc / logging_steps
            avg_reward = total_reward_acc / logging_steps
            elapsed = time.time() - t0
            print(f"  step {global_step}/{total_steps} | loss={avg_loss:.4f} | "
                  f"avg_reward={avg_reward:.4f} | elapsed={elapsed:.0f}s")
            total_loss_acc = 0.0
            total_reward_acc = 0.0

        if save_steps and global_step % save_steps == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(actor)
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
            unwrapped.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  [saved] {ckpt_dir}")

    if log_file is not None:
        log_file.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(actor)
        final_dir = os.path.join(output_dir, "final")
        unwrapped.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[GRPO] 训练完成，模型已保存到 {final_dir}，指标日志: {log_path}")
