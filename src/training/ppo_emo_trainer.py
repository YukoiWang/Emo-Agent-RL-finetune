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
    seed = cfg.get("seed", 42)
    T.manual_seed(seed)

    accelerator = Accelerator()
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
    user_llm_fn = _build_user_llm_fn(cfg)
    planning_service_url = rollout_cfg.get("planning_service_url")
    sft_model_path = model_cfg.get("sft_model_path")
    if planning_service_url:
        from src.training.planning_service_client import build_planning_service_llm_fn
        planning_llm_fn = build_planning_service_llm_fn(planning_service_url)
    elif sft_model_path:
        from src.training.local_planning_llm import build_local_planning_llm_fn
        planning_llm_fn = build_local_planning_llm_fn(sft_model_path, device=device)
    else:
        planning_llm_fn = None
    target_default = rollout_cfg.get("target", "eq")

    # ---------- 4. Memory / ref_log_probs / optimizer ----------
    gamma = rl_cfg.get("gamma", 1.0)
    lam = rl_cfg.get("lam", 0.95)
    memory = PPOMemory(gamma=gamma, lam=lam, device=device)

    def get_ref_log_probs_fn(q_ids, q_mask, r_ids, r_mask):
        q_len = q_ids.size(1)
        full_ids = T.cat([q_ids, r_ids], dim=1)
        full_attn = T.ones_like(full_ids, dtype=T.float32, device=device)
        full_attn[:, :q_len] = q_mask.to(T.float32)
        full_attn[:, q_len:] = r_mask.to(T.float32)
        full_resp_mask = T.zeros_like(full_ids, dtype=T.float32, device=device)
        full_resp_mask[:, q_len:] = r_mask.to(T.float32)
        full_lp = actor_ref.get_ref_log_probs(full_ids, full_attn, full_resp_mask)
        return full_lp[:, q_len:]

    lr = rl_cfg.get("learning_rate", 1e-6)
    optimizer = T.optim.AdamW(
        list(actor_ref.parameters_for_optimizer()) + list(critic.parameters()),
        lr=lr,
    )

    # ---------- 4b. Accelerate prepare（多卡时自动 DDP）----------
    actor_ref, critic, optimizer, dataloader = accelerator.prepare(
        actor_ref, critic, optimizer, dataloader
    )

    # ---------- 5. 训练循环 ----------
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    log_path = os.path.join(output_dir, "training_log.jsonl")
    log_file = open(log_path, "w", encoding="utf-8") if accelerator.is_main_process else None
    global_step = 0
    step_counter = [0]

    def _save():
        if not accelerator.is_main_process:
            return
        unwrapped = accelerator.unwrap_model(actor_ref)
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        unwrapped.actor.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  [saved] {ckpt_dir}")

    def _prune():
        if not accelerator.is_main_process or save_total_limit <= 0:
            return
        pattern = os.path.join(output_dir, "checkpoint-*")
        ckpts = sorted(glob.glob(pattern), key=lambda p: int(p.split("-")[-1]))
        while len(ckpts) > save_total_limit:
            shutil.rmtree(ckpts.pop(0), ignore_errors=True)

    dataloader_iter = iter(dataloader)
    start_time = time.time()

    while global_step < total_steps:
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
        # rollout 阶段用 unwrap 后的模型，DDP 包装器没有 .generate() 方法
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

        collect_rollouts_emo(**collect_kw)

        if len(memory) == 0:
            global_step += 1
            continue

        data = memory.get(compute_gae=True)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        queries = data["queries"]
        responses = data["responses"]
        qlens = [q.size(0) for q in queries]
        resp_lens = [r.size(0) for r in responses]
        full_seqs = [T.cat([q, r], dim=0) for q, r in zip(queries, responses)]
        max_len = max(s.size(0) for s in full_seqs)
        n_batch = len(full_seqs)

        full_ids = T.full((n_batch, max_len), pad_id, dtype=T.long, device=device)
        full_attn = T.zeros(n_batch, max_len, dtype=T.float32, device=device)
        full_resp_mask = T.zeros(n_batch, max_len, dtype=T.float32, device=device)
        for i, seq in enumerate(full_seqs):
            L = seq.size(0)
            full_ids[i, :L] = seq.to(device)
            qlen, rlen = qlens[i], resp_lens[i]
            full_attn[i, :qlen] = data["query_masks"][i].to(T.float32).to(device)
            full_attn[i, qlen : qlen + rlen] = data["response_masks"][i].to(T.float32).to(device)
            full_resp_mask[i, qlen : qlen + rlen] = data["response_masks"][i].to(T.float32).to(device)

        new_log_probs = actor_ref.get_actor_log_probs(full_ids, full_attn, full_resp_mask)
        values = critic(full_ids, full_attn)

        old_log_probs, _ = _pad_list_to_tensor(data["log_probs"], 0.0, device)
        ref_log_probs, _ = _pad_list_to_tensor(data["ref_log_probs"], 0.0, device)
        advantages, adv_mask = _pad_list_to_tensor(data["advantages"], 0.0, device)
        returns, ret_mask = _pad_list_to_tensor(data["returns"], 0.0, device)
        resp_mask_padded, _ = _pad_list_to_tensor(data["response_masks"], 0.0, device)
        n_valid = resp_mask_padded.sum().clamp(min=1e-8)

        max_r = max(resp_lens)
        new_lp_resp = T.zeros(n_batch, max_r, dtype=T.float32, device=device)
        values_resp = T.zeros(n_batch, max_r, dtype=T.float32, device=device)
        for i in range(n_batch):
            s, rlen = qlens[i], resp_lens[i]
            new_lp_resp[i, :rlen] = new_log_probs[i, s : s + rlen]
            values_resp[i, :rlen] = values[i, s : s + rlen]

        clip_range = rl_cfg.get("clip_range", 0.2)
        ratio = T.exp(new_lp_resp - old_log_probs)
        surr1 = ratio * advantages
        surr2 = T.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        policy_loss = -(T.minimum(surr1, surr2) * adv_mask).sum() / n_valid
        value_loss = ((values_resp - returns) ** 2 * ret_mask).sum() / n_valid
        kl_coef = rl_cfg.get("kl_penalty_coef", 0.02)
        kl_loss = ((old_log_probs - ref_log_probs) * adv_mask).sum() / n_valid
        total_loss = policy_loss + 0.5 * value_loss + kl_coef * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        T.nn.utils.clip_grad_norm_(
            list(actor_ref.parameters_for_optimizer()) + list(critic.parameters()),
            1.0,
        )
        optimizer.step()

        memory.clear()

        reward_mean = sum(data["rewards"]) / len(data["rewards"]) if data["rewards"] else 0.0
        elapsed = time.time() - start_time
        if accelerator.is_main_process and (global_step + 1) % logging_steps == 0:
            print(
                f"  step={global_step + 1}/{total_steps} | reward_mean={reward_mean:.4f} | "
                f"policy_loss={policy_loss.item():.4f} value_loss={value_loss.item():.4f} "
                f"kl_loss={kl_loss.item():.4f} | {elapsed:.0f}s"
            )
        if accelerator.is_main_process and log_file is not None:
            log_record = {
                "step": global_step,
                "reward_mean": reward_mean,
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),
                "kl_loss": kl_loss.item(),
                "total_loss": total_loss.item(),
                "elapsed": elapsed,
            }
            log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            log_file.flush()

        global_step += 1
        if save_steps and global_step % save_steps == 0:
            _save()
            _prune()

    if log_file is not None:
        log_file.close()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final")
        unwrapped = accelerator.unwrap_model(actor_ref)
        unwrapped.actor.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"[PPO-Emo] 多轮训练完成，模型已保存到 {final_dir}，日志: {log_path}")
