# -*- coding: utf-8 -*-
"""
GSPO (Group Sequence Policy Optimization) 训练模块。

基于 Qwen 团队提出的 GSPO 算法 (arXiv:2507.18071)，核心改进：
- 使用序列级（length-normalized）重要性比率代替 GRPO 的 token 级比率
  s_i(θ) = exp( mean_t( log π_θ(y_{i,t}) - log π_old(y_{i,t}) ) )
- 序列级 clipping，对长序列/MoE 模型训练更稳定
- 支持非对称 clipping (epsilon / epsilon_high)
- 支持多步梯度更新 (steps_per_generation)，复用 rollout 数据提高样本效率
- 默认 KL 系数为 0，ε 远小于 GRPO (~3e-4 vs 0.2)

多卡训练：使用 `accelerate launch scripts/rl/run_rl.py --config xxx.yaml`
"""
from __future__ import annotations

import contextlib
import copy
import glob as glob_mod
import json
import os
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.data.rl_dataset import load_rl_dataset
from src.models.modeling import load_sft_model, ModelAndTokenizer
from .grpo_training import (
    _log_probs_for_response,
    _generate_completions,
    _masked_log_probs,
    _run_multiturn_grpo_rollout,
    _compute_emo_reward_scalar,
)


# ============================================================================
# Single-turn GSPO
# ============================================================================

def run_gspo_training(
    cfg: Dict[str, Any],
    reward_fn: Optional[Callable[[List[str]], List[float]]] = None,
) -> None:
    """
    单轮 GSPO 训练主函数。

    每步：
    1. 取 prompt，生成 G 个 completion
    2. 用 reward_fn 打分，组内归一化得 advantage
    3. 缓存 old/ref log probs
    4. 做 steps_per_generation 次梯度更新，每次使用序列级比率 + clipping
    """
    torch.manual_seed(cfg.get("seed", 42))

    accelerator = Accelerator()
    device = accelerator.device

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    rl_cfg = cfg.get("rl", {}).get("gspo", {})
    training_cfg = cfg.get("training", {})

    if accelerator.is_main_process:
        print(f"[GSPO] Accelerate: {accelerator.num_processes} process(es), device={device}")

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
    if accelerator.is_main_process:
        print("[GSPO] 加载模型 ...")
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_visible and accelerator.num_processes > 1 and n_visible >= accelerator.num_processes:
        dev_map = {"": accelerator.local_process_index}
    else:
        dev_map = {"": 0} if torch.cuda.is_available() else None
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
        device_map=dev_map,
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
    print("[GSPO] 加载数据集 ...")
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
    eps_low = rl_cfg.get("epsilon", 3e-4)
    eps_high = rl_cfg.get("epsilon_high", eps_low)
    kl_coef = rl_cfg.get("beta", 0.0)
    entropy_coeff = rl_cfg.get("entropy_coeff", 0.0)
    entropy_coeff = rl_cfg.get("entropy_coeff", 0.0)
    temperature = rl_cfg.get("temperature", 1.0)
    top_p = rl_cfg.get("top_p", 1.0)
    steps_per_gen = rl_cfg.get("steps_per_generation", 4)
    total_steps = training_cfg.get("total_steps", 100)
    logging_steps = training_cfg.get("logging_steps", 10)
    save_steps = training_cfg.get("save_steps", 500)
    output_dir = training_cfg.get("output_dir", "outputs/gspo")

    # -- Optimizer & Accelerate --
    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=lr,
    )
    actor, optimizer, dataloader = accelerator.prepare(actor, optimizer, dataloader)
    ref_model = ref_model.to(device)

    # -- Training loop --
    if accelerator.is_main_process:
        print(
            f"[GSPO] 开始训练: total_steps={total_steps}, G={num_generations}, lr={lr}, "
            f"eps={eps_low}/{eps_high}, steps_per_gen={steps_per_gen}"
        )
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

        # --- Phase 1: Generate completions ---
        actor.eval()
        resp_ids_list, resp_texts = _generate_completions(
            actor, tokenizer, prompt_ids, prompt_mask,
            num_generations=num_generations, max_new_tokens=max_resp_len,
            temperature=temperature, top_p=top_p,
        )

        rewards = reward_fn(resp_texts)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        r_mean = rewards_t.mean()
        r_std = rewards_t.std().clamp(min=1e-6)
        advantages = (rewards_t - r_mean) / r_std

        valid_indices = [i for i in range(num_generations) if resp_ids_list[i].numel() > 0]
        if not valid_indices:
            global_step += 1
            continue

        # --- Phase 2: Cache old & ref log probs ---
        cached_full: List[Tuple[torch.Tensor, torch.Tensor]] = []
        cached_old_lp: List[torch.Tensor] = []
        cached_ref_lp: List[torch.Tensor] = []
        with torch.no_grad():
            for i in valid_indices:
                full_ids = torch.cat([prompt_ids[0], resp_ids_list[i]]).unsqueeze(0)
                full_mask = torch.ones_like(full_ids, dtype=torch.float32)
                old_lp = _log_probs_for_response(actor, full_ids, full_mask, q_len)
                ref_lp = _log_probs_for_response(ref_model, full_ids, full_mask, q_len)
                cached_full.append((full_ids, full_mask))
                cached_old_lp.append(old_lp.detach())
                cached_ref_lp.append(ref_lp.detach())

        # --- Phase 3: K gradient steps with sequence-level ratio ---
        step_loss_total = 0.0
        step_kl_total = 0.0
        valid_count = len(valid_indices)
        actor.train()

        for _gstep in range(steps_per_gen):
            optimizer.zero_grad()
            policy_loss = torch.tensor(0.0, device=device)
            kl_loss_accum = torch.tensor(0.0, device=device)
            entropy_loss_accum = torch.tensor(0.0, device=device)

            for j, i in enumerate(valid_indices):
                full_ids, full_mask = cached_full[j]
                old_lp = cached_old_lp[j]
                ref_lp = cached_ref_lp[j]

                actor_lp = _log_probs_for_response(actor, full_ids, full_mask, q_len)
                min_len = min(actor_lp.size(0), old_lp.size(0), ref_lp.size(0))
                actor_lp = actor_lp[:min_len]

                # Sequence-level importance ratio (length-normalized)
                seq_log_ratio = (actor_lp - old_lp[:min_len]).mean()
                s_i = torch.exp(seq_log_ratio)

                adv = advantages[i]
                surr1 = s_i * adv
                surr2 = torch.clamp(s_i, 1.0 - eps_low, 1.0 + eps_high) * adv
                policy_loss = policy_loss - torch.minimum(surr1, surr2)

                # Monte Carlo entropy estimator over response tokens
                entropy_i = (-actor_lp).mean()
                entropy_loss_accum = entropy_loss_accum + entropy_i

                log_ratio_ref = actor_lp - ref_lp[:min_len]
                kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1).mean()
                kl_loss_accum = kl_loss_accum + kl

            loss = (
                policy_loss
                + kl_coef * kl_loss_accum
                - entropy_coeff * entropy_loss_accum
            ) / (valid_count * steps_per_gen)
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(
                [p for p in actor.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()

            step_loss_total += loss.item()
            step_kl_total += (kl_loss_accum / valid_count).item()

        agg_reward_mean = sum(rewards) / len(rewards)
        total_loss_acc += step_loss_total
        total_reward_acc += agg_reward_mean

        if accelerator.is_main_process and log_file is not None:
            log_record = {
                "step": global_step,
                "reward_mean": agg_reward_mean,
                "loss": step_loss_total,
                "kl_loss": step_kl_total / steps_per_gen,
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
        print(f"[GSPO] 训练完成，模型已保存到 {final_dir}，指标日志: {log_path}")


# ============================================================================
# Multi-turn GSPO with Emo Reward (same rollout/reward pipeline as GRPO-Emo)
# ============================================================================

def run_gspo_emo_training(cfg: Dict[str, Any]) -> None:
    """
    Multi-turn GSPO training with planning simulator + emo reward (3 modes).

    相比 GRPO-Emo 的关键改进：
    1. 序列级重要性比率 s_i = exp(mean_t(log π_θ - log π_old))，对长多轮对话更稳定
    2. 支持 steps_per_generation 多步梯度更新，同一批 rollout 数据复用 K 次
    3. 非对称 clipping (epsilon / epsilon_high)，默认 ε 更小

    Config layout mirrors ``rl_grpo_emo.yaml`` with ``rl.algo: "gspo_emo"``
    and an ``rl.gspo`` section for GSPO-specific hyper-parameters.
    """
    from .ppo_emo_trainer import _build_user_llm_fn
    from src.data.profile_dataset import ProfileDataset

    torch.manual_seed(cfg.get("seed", 42))

    accelerator = Accelerator()
    device = accelerator.device

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {}) or {}
    rollout_cfg = cfg.get("rollout", {}) or {}
    reward_cfg = cfg.get("reward", {}) or {}
    rl_cfg = (cfg.get("rl") or {}).get("gspo") or {}
    training_cfg = cfg.get("training", {}) or {}

    output_dir = training_cfg.get("output_dir", "outputs/gspo_emo")
    total_steps = training_cfg.get("total_steps", 500)
    logging_steps = training_cfg.get("logging_steps", 10)
    save_steps = training_cfg.get("save_steps", 500)
    save_total_limit = training_cfg.get("save_total_limit", 3)

    reward_mode = reward_cfg.get("reward_mode", "mode1")
    w1 = reward_cfg.get("w1", 1.0)
    w2 = reward_cfg.get("w2", 0.3)
    w3 = reward_cfg.get("w3", 0.2)
    trend_n = reward_cfg.get("trend_n", 5)
    S1 = reward_cfg.get("S1", 100)
    S2 = reward_cfg.get("S2", 300)
    warmup_steps_reward = reward_cfg.get("warmup_steps", 200)

    if accelerator.is_main_process:
        print(f"[GSPO-Emo] Accelerate: {accelerator.num_processes} process(es), device={device}")

    # ---------- 1. Model (actor + frozen ref, no Critic) ----------
    n_visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_visible and accelerator.num_processes > 1 and n_visible >= accelerator.num_processes:
        dev_map = {"": accelerator.local_process_index}
    else:
        dev_map = {"": 0} if torch.cuda.is_available() else None
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
        device_map=dev_map,
    )
    actor = mt.model
    tokenizer = mt.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    if rl_cfg.get("gradient_checkpointing", False) and hasattr(actor, "gradient_checkpointing_enable"):
        actor.gradient_checkpointing_enable()

    ref_model = copy.deepcopy(actor)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # ---------- 2. Dataset (ProfileDataset, same as PPO-Emo / GRPO-Emo) ----------
    data_dir = (
        data_cfg.get("data_dir")
        or os.path.dirname(data_cfg.get("train_file", "data/data/train_profile.jsonl"))
        or "data/data"
    )
    dataset = ProfileDataset(
        data_dir=data_dir,
        split="train",
        max_scene_len=data_cfg.get("max_scene_len", 1500),
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    # ---------- 3. User simulator & planning (same as PPO-Emo / GRPO-Emo) ----------
    # Planning: prefer API; on failure fallback to local (rank 0 only). API path does not load local planner.
    user_llm_fn = _build_user_llm_fn(cfg)
    from .local_planning_llm import build_planning_llm_fn_prefer_api_then_local
    planning_llm_fn = build_planning_llm_fn_prefer_api_then_local(
        rollout_cfg, model_cfg, device,
        process_index=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    sft_model_path = model_cfg.get("sft_model_path")
    target = rollout_cfg.get("target", "eq")

    # ---------- 4. GSPO hyper-params ----------
    num_generations = rl_cfg.get("num_generations", 4)
    lr = rl_cfg.get("learning_rate", 1e-6)
    eps_low = rl_cfg.get("epsilon", 3e-4)
    eps_high = rl_cfg.get("epsilon_high", eps_low)
    kl_coef = rl_cfg.get("beta", 0.0)
    entropy_coeff = rl_cfg.get("entropy_coeff", 0.0)
    steps_per_gen = rl_cfg.get("steps_per_generation", 4)
    temperature = rollout_cfg.get("temperature", 0.8)
    top_p = rollout_cfg.get("top_p", 0.95)
    max_turns = rollout_cfg.get("max_turns", 8)
    max_new_tokens = rollout_cfg.get("max_new_tokens_per_turn", 128)

    # ---------- 5. Optimizer & Accelerate ----------
    optimizer = torch.optim.AdamW(
        [p for p in actor.parameters() if p.requires_grad], lr=lr,
    )
    actor, optimizer, dataloader = accelerator.prepare(actor, optimizer, dataloader)
    ref_model = ref_model.to(device)

    # ---------- 6. Training loop ----------
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        print(
            f"[GSPO-Emo] 开始训练: total_steps={total_steps}, G={num_generations}, "
            f"lr={lr}, eps={eps_low}/{eps_high}, steps_per_gen={steps_per_gen}, "
            f"reward_mode={reward_mode}"
        )
    accelerator.wait_for_everyone()

    # ---------- 6a. Resume from checkpoint ----------
    resume_from = training_cfg.get("resume_from_checkpoint")
    global_step = 0

    if resume_from and os.path.isdir(resume_from):
        if accelerator.is_main_process:
            print(f"[GSPO-Emo] Resuming from {resume_from}")
        from peft import PeftModel
        unwrapped = accelerator.unwrap_model(actor)
        base_model = unwrapped.base_model.model if hasattr(unwrapped, "base_model") else unwrapped
        actor_loaded = PeftModel.from_pretrained(base_model, resume_from).to(device)
        unwrapped.load_state_dict(actor_loaded.state_dict(), strict=False)
        del actor_loaded
        meta_path = os.path.join(resume_from, "training_state.pt")
        if os.path.exists(meta_path):
            state = torch.load(meta_path, map_location=device)
            global_step = state.get("global_step", 0)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            if accelerator.is_main_process:
                print(f"[GSPO-Emo] Resumed: global_step={global_step}")

    from .monitor import TrainingMonitor
    monitor_cfg = cfg.get("monitor", {}) or {}
    monitor = TrainingMonitor(
        output_dir=output_dir,
        experiment_name=monitor_cfg.get("experiment_name", "gspo_emo"),
        use_tensorboard=monitor_cfg.get("use_tensorboard", True),
        use_wandb=monitor_cfg.get("use_wandb", False),
        wandb_project=monitor_cfg.get("wandb_project"),
        config=cfg,
        enabled=accelerator.is_main_process,
        resume=bool(resume_from),
    )

    grad_accum_steps = rl_cfg.get("gradient_accumulation_steps", 1)
    if accelerator.is_main_process and grad_accum_steps > 1:
        print(f"[GSPO-Emo] gradient_accumulation_steps={grad_accum_steps}, "
              f"effective prompts/step = {grad_accum_steps} × G={num_generations}")

    data_iter = iter(dataloader)
    total_loss_acc = 0.0
    total_reward_acc = 0.0
    t0 = time.time()

    while global_step < total_steps:
        # ================================================================
        # Phase 1: Collect rollouts for all grad_accum prompts
        # ================================================================
        actor_unwrapped = accelerator.unwrap_model(actor)
        actor_unwrapped.eval()

        all_rollouts_flat: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        all_advantages_flat: List[float] = []
        all_rewards_flat: List[float] = []
        prompts_done = 0

        for _accum_idx in range(grad_accum_steps):
            try:
                batch_raw = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_raw = next(data_iter)

            item = batch_raw[0]
            profile = item["profile"]
            prompt = item["prompt"]

            rollout_results: List[Tuple] = []
            for _g in range(num_generations):
                out = _run_multiturn_grpo_rollout(
                    profile=profile, prompt=prompt,
                    model=actor_unwrapped, tokenizer=tokenizer,
                    user_llm_fn=user_llm_fn, planning_llm_fn=planning_llm_fn,
                    device=device, max_turns=max_turns,
                    max_new_tokens_per_turn=max_new_tokens,
                    do_sample=True, temperature=temperature, top_p=top_p,
                    target=target, sft_model_path=sft_model_path,
                )
                if out is not None:
                    rollout_results.append(out)

            if not rollout_results:
                continue

            rewards = []
            for _, _, _, emo_pt, emo_turns in rollout_results:
                r = _compute_emo_reward_scalar(
                    emo_pt, emo_turns, reward_mode, w1, w2, w3, trend_n,
                    step=global_step, S1=S1, S2=S2, warmup_steps=warmup_steps_reward,
                )
                rewards.append(r)

            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            r_mean = rewards_t.mean()
            r_std = rewards_t.std().clamp(min=1e-6)
            advantages = (rewards_t - r_mean) / r_std

            for i, (full_ids, full_mask, resp_mask, _, _) in enumerate(rollout_results):
                all_rollouts_flat.append((full_ids, full_mask, resp_mask))
                all_advantages_flat.append(advantages[i].item())
                all_rewards_flat.append(rewards[i])

            prompts_done += 1

        if not all_rollouts_flat:
            global_step += 1
            continue

        # ================================================================
        # Phase 2: Cache old & ref log probs (before any gradient update)
        # ================================================================
        all_old_lps: List[torch.Tensor] = []
        all_ref_lps: List[torch.Tensor] = []
        with torch.no_grad():
            for full_ids, full_mask, resp_mask in all_rollouts_flat:
                old_lp = _masked_log_probs(actor_unwrapped, full_ids, full_mask, resp_mask)
                ref_lp = _masked_log_probs(ref_model, full_ids, full_mask, resp_mask)
                all_old_lps.append(old_lp.detach())
                all_ref_lps.append(ref_lp.detach())

        # ================================================================
        # Phase 3: K gradient steps with sequence-level ratio
        # ================================================================
        total_count = len(all_rollouts_flat)
        step_loss_total = 0.0
        step_kl_total = 0.0

        for _gstep in range(steps_per_gen):
            optimizer.zero_grad()
            actor.train()

            for idx in range(total_count):
                full_ids, full_mask, resp_mask = all_rollouts_flat[idx]
                old_lp = all_old_lps[idx]
                ref_lp = all_ref_lps[idx]
                adv = all_advantages_flat[idx]
                n_resp = resp_mask.sum().clamp(min=1)

                is_last = (idx == total_count - 1)
                sync_ctx = contextlib.nullcontext() if is_last else accelerator.no_sync(actor)

                with sync_ctx:
                    actor_lp = _masked_log_probs(actor, full_ids, full_mask, resp_mask)

                    # Sequence-level importance ratio (length-normalized)
                    log_ratio = actor_lp - old_lp
                    seq_log_ratio = log_ratio.sum() / n_resp
                    s_i = torch.exp(seq_log_ratio)

                    surr1 = s_i * adv
                    surr2 = torch.clamp(s_i, 1.0 - eps_low, 1.0 + eps_high) * adv
                    pg_loss = -torch.minimum(surr1, surr2)

                    # Monte Carlo entropy estimator over response tokens
                    entropy_i = (-(actor_lp * resp_mask)).sum() / n_resp

                    log_ratio_ref = actor_lp - ref_lp
                    kl = (torch.exp(log_ratio_ref) - log_ratio_ref - 1).sum() / n_resp

                    loss_i = (
                        pg_loss
                        + kl_coef * kl
                        - entropy_coeff * entropy_i
                    ) / (total_count * steps_per_gen)
                    accelerator.backward(loss_i)

                step_loss_total += loss_i.item()
                step_kl_total += kl.item()

            accelerator.clip_grad_norm_(
                [p for p in accelerator.unwrap_model(actor).parameters() if p.requires_grad],
                1.0,
            )
            optimizer.step()

        # ================================================================
        # Logging & Checkpointing
        # ================================================================
        agg_reward_mean = sum(all_rewards_flat) / len(all_rewards_flat)
        total_loss_acc += step_loss_total
        total_reward_acc += agg_reward_mean

        monitor.log(step=global_step, metrics={
            "reward_mean": agg_reward_mean,
            "rewards_max": max(all_rewards_flat),
            "rewards_min": min(all_rewards_flat),
            "loss": step_loss_total,
            "kl_loss": step_kl_total / max(total_count * steps_per_gen, 1),
        })

        global_step += 1

        if accelerator.is_main_process and global_step % logging_steps == 0:
            avg_loss = total_loss_acc / logging_steps
            avg_reward = total_reward_acc / logging_steps
            elapsed = time.time() - t0
            print(
                f"  step {global_step}/{total_steps} | loss={avg_loss:.4f} | "
                f"avg_reward={avg_reward:.4f} | elapsed={elapsed:.0f}s"
            )
            total_loss_acc = 0.0
            total_reward_acc = 0.0

        if save_steps and global_step % save_steps == 0:
            if accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(actor)
                ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                unwrapped.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                torch.save({
                    "global_step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(ckpt_dir, "training_state.pt"))
                print(f"  [saved] {ckpt_dir}")
            if accelerator.is_main_process and save_total_limit > 0:
                pattern = os.path.join(output_dir, "checkpoint-*")
                ckpts = sorted(glob_mod.glob(pattern), key=lambda p: int(p.split("-")[-1]))
                while len(ckpts) > save_total_limit:
                    shutil.rmtree(ckpts.pop(0), ignore_errors=True)

    monitor.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(actor)
        final_dir = os.path.join(output_dir, "final")
        unwrapped.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        torch.save({
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
        }, os.path.join(final_dir, "training_state.pt"))
        log_path = os.path.join(output_dir, "training_log.jsonl")
        print(f"[GSPO-Emo] 训练完成，模型已保存到 {final_dir}，指标日志: {log_path}")
