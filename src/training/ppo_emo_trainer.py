# -*- coding: utf-8 -*-
"""
多轮 PPO 训练：Profile 数据 + 用户模拟器多轮对话 → collect_rollouts_emo → GAE → PPO 更新。
完整流程入口：run_ppo_emo_training(cfg)。

多卡训练：使用 `accelerate launch scripts/rl/run_rl.py --config xxx.yaml`，
或在 config 中设置 training.use_accelerate: true 后由 run_rl 自动用 accelerate 启动。
"""
from __future__ import annotations

import copy
import glob
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch as T

torch = T  # 兼容直接使用 torch 的代码
from torch.utils.data import DataLoader
from accelerate import Accelerator

from src.data.profile_dataset import ProfileDataset, build_initial_prompt
from src.models.modeling import load_sft_model, ModelAndTokenizer
from src.training.ppo_emo_rollout import collect_rollouts_emo
from src.training.ppo_training import ActorRefRollout, Critic, PPOMemory, _pad_list_to_tensor


def _build_user_llm_fn(cfg: Dict[str, Any]) -> Callable[[List[Dict[str, str]]], str]:
    """从 config 构建 user_llm_fn。"""
    rollout_cfg = cfg.get("rollout", {}) or {}
    kind = rollout_cfg.get("user_llm", "mock")
    if kind == "mock":

        def mock_fn(messages):
            last = messages[-1] if messages else {}
            c = (last.get("content") or "").strip()
            if "请回复" in c or "NPC" in c:
                return "我最近工作压力很大，和家里人关系也紧张，不知道该怎么办。"
            if "建议" in c or "理解" in c:
                return "嗯，你说得对，我试试看。"
            return "谢谢你的建议，我再想想。再见。"

        return mock_fn
    if kind in ("deepseek", "qwen", "openai"):
        try:
            from src.training.qwen_user_simulator import build_qwen_user_llm_fn
        except ImportError:
            raise ImportError("user_llm=deepseek/qwen 需要 src.training.qwen_user_simulator")
        model_name = rollout_cfg.get("user_llm_model", "deepseek-chat")
        temp = rollout_cfg.get("user_llm_temperature", 0.7)
        return build_qwen_user_llm_fn(model=model_name, temperature=temp)
    raise ValueError(f"不支持的 user_llm: {kind}，可选 mock, deepseek, qwen, openai")


def run_ppo_emo_training(cfg: Dict[str, Any]) -> None:
    """
    多轮 PPO 完整训练：加载 Profile 数据与模型 → 每步 rollout → reward → GAE → PPO 更新 → 保存。

    cfg 结构示例：
      model:
        sft_model_path: "path_or_hf_id"
        dtype: "bfloat16"
        use_lora: true
        lora: { r: 16, ... }  # 可选
      data:
        data_dir: "data/data"
        batch_size: 2
      rollout:
        max_turns: 8
        max_new_tokens_per_turn: 128
        use_planning_emo: true
        user_llm: "mock"  # mock | deepseek | qwen
        target: "eq"
      reward:
        reward_mode: "mode1"
        w1, w2, w3, trend_n: ...
        S1, S2, warmup_steps: ...  # mode3
      rl:
        ppo:
          gamma: 1.0
          lam: 0.95
          learning_rate: 1e-6
          clip_range: 0.2
          kl_penalty_coef: 0.02
      training:
        output_dir: "outputs/ppo_emo"
        total_steps: 1000
        save_steps: 200
        save_total_limit: 3
        logging_steps: 10
    """
    import datetime
    nccl_timeout_min = cfg.get("training", {}).get("nccl_timeout_minutes", 30)
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    seed = cfg.get("seed", 42)
    T.manual_seed(seed)

    from accelerate.utils import InitProcessGroupKwargs
    kwargs = InitProcessGroupKwargs(
        timeout=datetime.timedelta(minutes=nccl_timeout_min),
    )
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    device = accelerator.device

    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {}) or {}
    rollout_cfg = cfg.get("rollout", {}) or {}
    reward_cfg = cfg.get("reward", {}) or {}
    rl_cfg = (cfg.get("rl") or {}).get("ppo") or {}
    training_cfg = cfg.get("training", {}) or {}

    data_dir = data_cfg.get("data_dir") or os.path.dirname(data_cfg.get("train_file", "data/data/train_profile.jsonl")) or "data/data"
    batch_size = data_cfg.get("batch_size", 2)
    output_dir = training_cfg.get("output_dir", "outputs/ppo_emo")
    total_steps = training_cfg.get("total_steps", 500)
    save_steps = training_cfg.get("save_steps", 200)
    save_total_limit = training_cfg.get("save_total_limit", 3)
    logging_steps = training_cfg.get("logging_steps", 10)

    if accelerator.is_main_process:
        print(f"[PPO-Emo] Accelerate: {accelerator.num_processes} process(es), device={device}")
    reward_mode = reward_cfg.get("reward_mode", "mode1")

    # ---------- 1. 模型 ----------
    # 设备映射：若每进程能看到多块 GPU，用 local_process_index 分配；否则用 0
    if T.cuda.is_available():
        n_visible = T.cuda.device_count()
        if accelerator.num_processes > 1 and n_visible >= accelerator.num_processes:
            dev_map = {"": accelerator.local_process_index}
        else:
            dev_map = {"": 0}
    else:
        dev_map = None
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
        device_map=dev_map,
    )
    if rl_cfg.get("gradient_checkpointing", False) and hasattr(mt.model, "gradient_checkpointing_enable"):
        mt.model.gradient_checkpointing_enable()

    tokenizer = mt.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    actor_ref = ActorRefRollout(
        causal_lm=mt.model,
        tokenizer=tokenizer,
        device=device,
    )
    hidden_size = mt.model.config.hidden_size
    # Critic 使用独立 backbone，不与 ref 共享
    critic_base = copy.deepcopy(mt.model)
    critic_base.config.use_cache = False 
    if rl_cfg.get("gradient_checkpointing", False) and hasattr(critic_base, "gradient_checkpointing_enable"):
        critic_base.gradient_checkpointing_enable()
    critic = Critic(critic_base, hidden_size, dropout=0.0).to(device)

    # ---------- 2. 数据 ----------
    dataset = ProfileDataset(
        data_dir=data_dir,
        split="train",
        max_scene_len=data_cfg.get("max_scene_len", 1500),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    # ---------- 3. 用户模拟与情感（使用 planning_reply） ----------
    # Planning：优先 API（planning_service_url / planning_llm），失败再回退本地；用 API 时不会加载本地 planner。
    user_llm_fn = _build_user_llm_fn(cfg)
    from src.training.local_planning_llm import build_planning_llm_fn_prefer_api_then_local
    planning_llm_fn = build_planning_llm_fn_prefer_api_then_local(
        rollout_cfg, model_cfg, device,
        process_index=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    sft_model_path = model_cfg.get("sft_model_path")
    target_default = rollout_cfg.get("target", "eq")

    # ---------- 4. Memory / ref_log_probs / optimizer ----------
    gamma = rl_cfg.get("gamma", 1.0)
    lam = rl_cfg.get("lam", 0.95)
    memory = PPOMemory(gamma=gamma, lam=lam, device="cpu")

    def get_ref_log_probs_fn(q_ids, q_mask, r_ids, r_mask):
        q_len = q_ids.size(1)
        full_ids = T.cat([q_ids, r_ids], dim=1)
        full_attn = T.ones_like(full_ids, dtype=T.float32, device=device)
        full_attn[:, :q_len] = q_mask.to(T.float32)
        full_attn[:, q_len:] = r_mask.to(T.float32)
        full_resp_mask = T.zeros_like(full_ids, dtype=T.float32, device=device)
        full_resp_mask[:, q_len:] = r_mask.to(T.float32)
        unwrapped = accelerator.unwrap_model(actor_ref)
        full_lp = unwrapped.get_ref_log_probs(full_ids, full_attn, full_resp_mask)
        return full_lp[:, q_len:]

    lr = rl_cfg.get("learning_rate", 1e-6)
    critic_lr = rl_cfg.get("critic_learning_rate", lr)
    actor_optimizer = T.optim.AdamW(
        actor_ref.parameters_for_optimizer(),
        lr=lr,
    )
    critic_optimizer = T.optim.AdamW(
        critic.parameters(),
        lr=critic_lr,
    )

    # ---------- 4b. DEBUG: 检查各 rank 模型参数（DDP 要求一致）----------
    n_actor = sum(p.numel() for p in actor_ref.parameters())
    n_actor_train = sum(p.numel() for p in actor_ref.parameters_for_optimizer())
    n_critic = sum(p.numel() for p in critic.parameters())
    n_critic_train = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(
        f"[PPO-Emo][rank={accelerator.process_index}] actor: total={n_actor}, trainable={n_actor_train} | "
        f"critic: total={n_critic}, trainable={n_critic_train}"
    )
    accelerator.wait_for_everyone()
    # ---------- 4c. Accelerate prepare（多卡时自动 DDP）----------
    actor_ref, critic, actor_optimizer, critic_optimizer, dataloader = accelerator.prepare(
        actor_ref, critic, actor_optimizer, critic_optimizer, dataloader
    )

    # ---------- 5. 训练循环 ----------
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # ---------- 5a. Resume from checkpoint ----------
    resume_from = training_cfg.get("resume_from_checkpoint")
    global_step = 0
    step_counter = [0]

    if resume_from and os.path.isdir(resume_from):
        if accelerator.is_main_process:
            print(f"[PPO-Emo] Resuming from {resume_from}")
        from peft import PeftModel
        unwrapped = accelerator.unwrap_model(actor_ref)
        unwrapped.actor = PeftModel.from_pretrained(
            unwrapped.actor.base_model.model, resume_from
        ).to(device)
        meta_path = os.path.join(resume_from, "training_state.pt")
        if os.path.exists(meta_path):
            state = T.load(meta_path, map_location=device)
            global_step = state.get("global_step", 0)
            if "actor_optimizer_state_dict" in state:
                actor_optimizer.load_state_dict(state["actor_optimizer_state_dict"])
                critic_optimizer.load_state_dict(state["critic_optimizer_state_dict"])
            elif "optimizer_state_dict" in state:
                if accelerator.is_main_process:
                    print("[PPO-Emo] Old checkpoint format (single optimizer), skipping optimizer state")
            if "critic_state_dict" in state:
                accelerator.unwrap_model(critic).load_state_dict(state["critic_state_dict"])
            if accelerator.is_main_process:
                print(f"[PPO-Emo] Resumed: global_step={global_step}")

    from .monitor import TrainingMonitor
    monitor_cfg = cfg.get("monitor", {}) or {}
    monitor = TrainingMonitor(
        output_dir=output_dir,
        experiment_name=monitor_cfg.get("experiment_name", "ppo_emo"),
        use_tensorboard=monitor_cfg.get("use_tensorboard", True),
        use_wandb=monitor_cfg.get("use_wandb", False),
        wandb_project=monitor_cfg.get("wandb_project"),
        config=cfg,
        enabled=accelerator.is_main_process,
        resume=bool(resume_from),
    )

    def _save():
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(actor_ref)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        unwrapped.actor.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        T.save({
            "global_step": global_step,
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "critic_state_dict": accelerator.unwrap_model(critic).state_dict(),
        }, os.path.join(ckpt_dir, "training_state.pt"))
        print(f"  [saved] {ckpt_dir}")

    def _prune():
        if not accelerator.is_main_process or save_total_limit <= 0:
            return
        pattern = os.path.join(output_dir, "checkpoint-*")
        ckpts = sorted(glob.glob(pattern), key=lambda p: int(p.split("-")[-1]))
        while len(ckpts) > save_total_limit:
            shutil.rmtree(ckpts.pop(0), ignore_errors=True)

    grad_accum_steps = rl_cfg.get("gradient_accumulation_steps") or 1
    if accelerator.is_main_process and grad_accum_steps > 1:
        print(f"[PPO-Emo] gradient_accumulation_steps={grad_accum_steps}, "
              f"effective batch = {batch_size} × {accelerator.num_processes} × {grad_accum_steps}")

    dataloader_iter = iter(dataloader)
    start_time = time.time()

    while global_step < total_steps:
        # --- Collect rollouts for grad_accum_steps iterations ---
        step_comp_stats: list[dict[str, float]] = []
        for _accum in range(grad_accum_steps):
            try:
                batch_raw = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch_raw = next(dataloader_iter)

            batch_items = [
                {
                    "profile": x["profile"],
                    "prompt": x["prompt"],
                    "idx": x["idx"],
                    "target": rollout_cfg.get("target", target_default),
                }
                for x in batch_raw
            ]

            step_counter[0] = global_step
            actor_ref_for_rollout = accelerator.unwrap_model(actor_ref)
            critic_for_rollout = accelerator.unwrap_model(critic)
            collect_kw = {
                "batch_items": batch_items,
                "actor_ref": actor_ref_for_rollout,
                "critic": critic_for_rollout,
                "tokenizer": tokenizer,
                "user_llm_fn": user_llm_fn,
                "memory": memory,
                "get_ref_log_probs_fn": get_ref_log_probs_fn,
                "device": device,
                "reward_mode": reward_mode,
                "w1": reward_cfg.get("w1", 1.0),
                "w2": reward_cfg.get("w2", 0.3),
                "w3": reward_cfg.get("w3", 0.2),
                "trend_n": reward_cfg.get("trend_n", 5),
                "max_turns": rollout_cfg.get("max_turns", 8),
                "max_new_tokens_per_turn": rollout_cfg.get("max_new_tokens_per_turn", 128),
                "do_sample": True,
                "temperature": rollout_cfg.get("temperature", 0.8),
                "top_p": rollout_cfg.get("top_p", 0.95),
                "target": target_default,
                "sft_model_path": sft_model_path,
                "planning_llm_fn": planning_llm_fn,
            }
            if reward_mode == "mode3":
                collect_kw["step"] = global_step
                collect_kw["S1"] = reward_cfg.get("S1", 100)
                collect_kw["S2"] = reward_cfg.get("S2", 300)
                collect_kw["warmup_steps"] = reward_cfg.get("warmup_steps", 200)

            _, _, comp_stats = collect_rollouts_emo(**collect_kw)
            step_comp_stats.append(comp_stats)

        # Barrier: all ranks must finish rollout before any enters DDP update,
        # otherwise the faster rank's DDP forward blocks on NCCL collectives
        # that the slower rank hasn't reached yet → NCCL timeout.
        accelerator.wait_for_everyone()

        # All ranks must agree: if any rank has empty buffer, all skip the update.
        # Otherwise one rank would continue to next iter and miss all_reduce / DDP
        # below → collective timeout (rank 0 in BROADCAST, others in ALLGATHER).
        if accelerator.num_processes > 1:
            has_data = T.tensor([1.0 if len(memory) > 0 else 0.0], device=device)
            T.distributed.all_reduce(has_data, op=T.distributed.ReduceOp.MIN)
            if has_data.item() == 0.0:
                global_step += 1
                continue
        elif len(memory) == 0:
            global_step += 1
            continue

        data = memory.get(compute_gae=True)

        ppo_epochs = rl_cfg.get("ppo_epochs", 2)
        mini_batch_size = rl_cfg.get("mini_batch_size", 1)
        clip_range = rl_cfg.get("clip_range", 0.2)
        clip_range_value = rl_cfg.get("clip_range_value", 0.2)
        kl_coef = rl_cfg.get("kl_penalty_coef", 0.02)
        entropy_coeff = rl_cfg.get("entropy_coeff", 0.01)
        max_grad_norm = rl_cfg.get("max_grad_norm", 1.0)

        local_buffer_size = len(data["queries"])
        # All ranks must iterate the same number of DDP forward passes; use the
        # max buffer size so no rank exits the loop early while others still call
        # DDP forward (which would deadlock on NCCL collectives).
        if accelerator.num_processes > 1:
            buf_tensor = T.tensor([local_buffer_size], device=device)
            T.distributed.all_reduce(buf_tensor, op=T.distributed.ReduceOp.MAX)
            total_buffer_size = int(buf_tensor.item())
        else:
            total_buffer_size = local_buffer_size
        indices = list(range(local_buffer_size))
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        all_stats: list[dict[str, float]] = []
        # Debug stats aggregated over all minibatches in this PPO step
        adv_count = 0
        adv_sum = 0.0
        adv_sumsq = 0.0
        ratio_count = 0
        ratio_sum = 0.0
        ratio_sumsq = 0.0

        for _epoch in range(ppo_epochs):
            random.shuffle(indices)

            for i in range(0, total_buffer_size, mini_batch_size):
                batch_indices = [idx % local_buffer_size for idx in range(i, min(i + mini_batch_size, total_buffer_size))]
                if not batch_indices:
                    continue

                queries = [data["queries"][idx] for idx in batch_indices]
                responses = [data["responses"][idx] for idx in batch_indices]
                query_masks = [data["query_masks"][idx] for idx in batch_indices]
                response_masks = [data["response_masks"][idx] for idx in batch_indices]

                old_log_probs_list = [data["log_probs"][idx] for idx in batch_indices]
                ref_log_probs_list = [data["ref_log_probs"][idx] for idx in batch_indices]
                advantages_list = [data["advantages"][idx] for idx in batch_indices]
                returns_list = [data["returns"][idx] for idx in batch_indices]
                old_values_list = [data["values"][idx] for idx in batch_indices]

                qlens = [q.size(0) for q in queries]
                resp_lens = [r.size(0) for r in responses]
                full_seqs = [T.cat([q, r], dim=0) for q, r in zip(queries, responses)]
                max_len = max(s.size(0) for s in full_seqs)
                curr_bs = len(full_seqs)

                full_ids = T.full((curr_bs, max_len), pad_id, dtype=T.long, device=device)
                full_attn = T.zeros(curr_bs, max_len, dtype=T.float32, device=device)
                full_resp_mask = T.zeros(curr_bs, max_len, dtype=T.float32, device=device)

                for j, seq in enumerate(full_seqs):
                    L = seq.size(0)
                    full_ids[j, :L] = seq.to(device)
                    qlen, rlen = qlens[j], resp_lens[j]
                    full_attn[j, :qlen] = query_masks[j].to(T.float32).to(device)
                    full_attn[j, qlen : qlen + rlen] = response_masks[j].to(T.float32).to(device)
                    full_resp_mask[j, qlen : qlen + rlen] = response_masks[j].to(T.float32).to(device)

                old_log_probs, _ = _pad_list_to_tensor(old_log_probs_list, 0.0, device)
                ref_log_probs, _ = _pad_list_to_tensor(ref_log_probs_list, 0.0, device)
                advantages, adv_mask = _pad_list_to_tensor(advantages_list, 0.0, device)
                returns, ret_mask = _pad_list_to_tensor(returns_list, 0.0, device)
                old_values, _ = _pad_list_to_tensor(old_values_list, 0.0, device)
                max_r = max(resp_lens)

                # --- Critic Update (separate forward/backward) ---
                values_out = critic(full_ids, full_attn)
                values_resp = T.zeros(curr_bs, max_r, dtype=T.float32, device=device)
                for j in range(curr_bs):
                    s, rlen = qlens[j], resp_lens[j]
                    values_resp[j, :rlen] = values_out[j, s : s + rlen]

                vpred_clipped = old_values + T.clamp(
                    values_resp - old_values, -clip_range_value, clip_range_value
                )
                vf_loss1 = (values_resp - returns) ** 2
                vf_loss2 = (vpred_clipped - returns) ** 2
                valid_ret = ret_mask.sum().clamp(min=1e-8)
                vf_loss = 0.5 * (T.maximum(vf_loss1, vf_loss2) * ret_mask).sum() / valid_ret

                critic_optimizer.zero_grad()
                accelerator.backward(vf_loss)
                critic_grad_norm = T.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optimizer.step()

                n_ret = ret_mask.sum().clamp(min=1)
                vf_clipfrac = ((vf_loss2 > vf_loss1).float() * ret_mask).sum().item() / n_ret.item()
                vpred_mean = (values_resp.detach() * ret_mask).sum().item() / n_ret.item()
                del values_out

                # --- Actor Update (separate forward/backward, with entropy bonus) ---
                new_log_probs, entropy_full = actor_ref(
                    full_ids, full_attn, full_resp_mask, return_entropy=True
                )

                new_lp_resp = T.zeros(curr_bs, max_r, dtype=T.float32, device=device)
                entropy_resp = T.zeros(curr_bs, max_r, dtype=T.float32, device=device)
                for j in range(curr_bs):
                    s, rlen = qlens[j], resp_lens[j]
                    new_lp_resp[j, :rlen] = new_log_probs[j, s : s + rlen]
                    entropy_resp[j, :rlen] = entropy_full[j, s : s + rlen]

                ratio = T.exp(new_lp_resp - old_log_probs)
                surr1 = ratio * advantages
                surr2 = T.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
                valid_count = adv_mask.sum().clamp(min=1e-8)

                # --- Debug aggregates: adv / ratio mean & std on valid tokens ---
                with T.no_grad():
                    valid = adv_mask.to(dtype=T.bool)
                    if valid.any():
                        adv_v = advantages.detach()[valid]
                        ratio_v = ratio.detach()[valid]
                        adv_count += int(adv_v.numel())
                        adv_sum += float(adv_v.sum().item())
                        adv_sumsq += float((adv_v * adv_v).sum().item())
                        ratio_count += int(ratio_v.numel())
                        ratio_sum += float(ratio_v.sum().item())
                        ratio_sumsq += float((ratio_v * ratio_v).sum().item())

                pg_loss = -(T.minimum(surr1, surr2) * adv_mask).sum() / valid_count
                entropy_loss = (entropy_resp * adv_mask).sum() / valid_count
                _kl_ratio = new_lp_resp - ref_log_probs
                kl_loss = ((T.exp(_kl_ratio) - _kl_ratio - 1) * adv_mask).sum() / valid_count
                policy_loss = pg_loss - entropy_coeff * entropy_loss + kl_coef * kl_loss

                actor_optimizer.zero_grad()
                accelerator.backward(policy_loss)
                actor_grad_norm = T.nn.utils.clip_grad_norm_(
                    list(accelerator.unwrap_model(actor_ref).parameters_for_optimizer()),
                    max_grad_norm,
                )
                actor_optimizer.step()

                n_adv = adv_mask.sum().clamp(min=1)
                pg_clipfrac = ((ratio.detach() - 1.0).abs() > clip_range).float()
                pg_clipfrac = (pg_clipfrac * adv_mask).sum().item() / n_adv.item()
                ppo_kl = 0.5 * ((new_lp_resp.detach() - old_log_probs) ** 2 * adv_mask).sum().item() / n_adv.item()

                all_stats.append({
                    "actor/pg_loss": pg_loss.item(),
                    "actor/entropy_loss": entropy_loss.item(),
                    "actor/kl_loss": kl_loss.item(),
                    "actor/policy_loss": policy_loss.item(),
                    "actor/pg_clipfrac": pg_clipfrac,
                    "actor/ppo_kl": ppo_kl,
                    "actor/grad_norm": actor_grad_norm.item(),
                    "critic/vf_loss": vf_loss.item(),
                    "critic/vf_clipfrac": vf_clipfrac,
                    "critic/vpred_mean": vpred_mean,
                    "critic/grad_norm": critic_grad_norm.item(),
                })

                del full_ids, full_attn, full_resp_mask, new_log_probs, entropy_full, policy_loss, vf_loss

        avg_stats = {k: sum(s[k] for s in all_stats) / len(all_stats) for k in all_stats[0]} if all_stats else {}
        # Token-level rewards are stored per sequence; reduce to scalar per sequence for logging
        reward_scalars = [
            float(r.sum().item()) for r in data.get("rewards", [])  # type: ignore[arg-type]
        ]
        if reward_scalars:
            rewards_t = T.tensor(reward_scalars, dtype=T.float32)
            reward_mean = float(rewards_t.mean().item())
            reward_std = float(rewards_t.std(unbiased=False).item()) if rewards_t.numel() > 1 else 0.0
            reward_max = float(rewards_t.max().item())
        else:
            reward_mean, reward_std, reward_max = 0.0, 0.0, 0.0

        # Finalize debug stats for this step
        if adv_count > 0:
            adv_mean_dbg = adv_sum / adv_count
            adv_var_dbg = max(0.0, adv_sumsq / adv_count - adv_mean_dbg * adv_mean_dbg)
            adv_std_dbg = adv_var_dbg ** 0.5
        else:
            adv_mean_dbg, adv_std_dbg = 0.0, 0.0
        if ratio_count > 0:
            ratio_mean_dbg = ratio_sum / ratio_count
            ratio_var_dbg = max(0.0, ratio_sumsq / ratio_count - ratio_mean_dbg * ratio_mean_dbg)
            ratio_std_dbg = ratio_var_dbg ** 0.5
        else:
            ratio_mean_dbg, ratio_std_dbg = 0.0, 0.0

        if T.cuda.is_available():
            T.cuda.empty_cache()
        memory.clear()
        elapsed = time.time() - start_time
        if accelerator.is_main_process and (global_step + 1) % logging_steps == 0:
            print(
                f"  step={global_step + 1}/{total_steps} | reward={reward_mean:.4f} | "
                f"pg={avg_stats.get('actor/pg_loss', 0):.4f} "
                f"vf={avg_stats.get('critic/vf_loss', 0):.4f} "
                f"ent={avg_stats.get('actor/entropy_loss', 0):.4f} "
                f"kl={avg_stats.get('actor/kl_loss', 0):.4f} "
                f"pg_clip={avg_stats.get('actor/pg_clipfrac', 0):.3f} "
                f"vf_clip={avg_stats.get('critic/vf_clipfrac', 0):.3f} | {elapsed:.0f}s"
            )
            print(
                f"    [dbg] adv mean/std={adv_mean_dbg:.4f}/{adv_std_dbg:.4f} | "
                f"ratio mean/std={ratio_mean_dbg:.4f}/{ratio_std_dbg:.4f} | "
                f"reward mean/std/max={reward_mean:.4f}/{reward_std:.4f}/{reward_max:.4f}"
            )
        avg_comp = {}
        if step_comp_stats:
            for k in step_comp_stats[0]:
                avg_comp[k] = sum(s[k] for s in step_comp_stats) / len(step_comp_stats)
        monitor.log(step=global_step, metrics={
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_max": reward_max,
            "dbg/adv_mean": adv_mean_dbg,
            "dbg/adv_std": adv_std_dbg,
            "dbg/ratio_mean": ratio_mean_dbg,
            "dbg/ratio_std": ratio_std_dbg,
            **avg_stats,
            **avg_comp,
            "elapsed": elapsed,
        })

        global_step += 1
        if save_steps and global_step % save_steps == 0:
            _save()
            _prune()

    monitor.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final")
        unwrapped = accelerator.unwrap_model(actor_ref)
        unwrapped.actor.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        T.save({
            "global_step": global_step,
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": critic_optimizer.state_dict(),
            "critic_state_dict": accelerator.unwrap_model(critic).state_dict(),
        }, os.path.join(final_dir, "training_state.pt"))
        log_path = os.path.join(output_dir, "training_log.jsonl")
        print(f"[PPO-Emo] 多轮训练完成，模型已保存到 {final_dir}，日志: {log_path}")
