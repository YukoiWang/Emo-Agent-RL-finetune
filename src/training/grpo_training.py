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
    if accelerator.is_main_process:
        print("[GRPO] 加载模型 ...")
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
    entropy_coeff = rl_cfg.get("entropy_coeff", 0.0)
    temperature = rl_cfg.get("temperature", 1.0)
    top_p = rl_cfg.get("top_p", 1.0)
    total_steps = training_cfg.get("total_steps", 100)
    logging_steps = training_cfg.get("logging_steps", 10)
    save_steps = training_cfg.get("save_steps", 500)
    output_dir = training_cfg.get("output_dir", "outputs/grpo")
    max_grad_norm = rl_cfg.get("max_grad_norm", 1.0)

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
    grad_accum_steps = rl_cfg.get("gradient_accumulation_steps", 1)
    if accelerator.is_main_process and grad_accum_steps > 1:
        print(f"[GRPO] gradient_accumulation_steps={grad_accum_steps}")

    log_path = os.path.join(output_dir, "training_log.jsonl")
    log_file = open(log_path, "w", encoding="utf-8") if accelerator.is_main_process else None
    data_iter = iter(dataloader)
    global_step = 0
    total_loss_acc = 0.0
    total_reward_acc = 0.0
    t0 = time.time()

    while global_step < total_steps:
        optimizer.zero_grad()
        step_loss_val = 0.0
        step_kl_val = 0.0
        step_rewards: List[float] = []
        prompts_done = 0

        for _accum_idx in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            prompt_ids = batch["input_ids"][:1].to(device)
            prompt_mask = batch["attention_mask"][:1].to(device)
            q_len = prompt_ids.size(1)

            actor.eval()
            resp_ids_list, resp_texts = _generate_completions(
                actor, tokenizer, prompt_ids, prompt_mask,
                num_generations=num_generations, max_new_tokens=max_resp_len,
                temperature=temperature, top_p=top_p,
            )

            # 标量 reward -> token 级 reward 展开（只在最后一个 token 上放置 outcome reward），
            # 再按组做标准化，得到 GRPO advantage。
            rewards = reward_fn(resp_texts)
            rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
            if len(resp_ids_list) > 0:
                max_resp_len = max(r.size(0) for r in resp_ids_list)
            else:
                max_resp_len = 0
            token_level_rewards = torch.zeros(
                num_generations, max_resp_len, dtype=torch.float32, device=device,
            )
            for i, resp in enumerate(resp_ids_list):
                L = resp.size(0)
                if L > 0:
                    # 将该条回复的 outcome reward 放到最后一个有效 token 上
                    token_level_rewards[i, L - 1] = rewards_t[i]
            # 每条回复的总 reward（与 rewards_t 等价）做组内标准化
            scores = token_level_rewards.sum(dim=-1)
            r_mean = scores.mean()
            r_std = scores.std().clamp(min=1e-6)
            advantages = (scores - r_mean) / r_std

            actor.train()
            policy_loss = torch.tensor(0.0, device=device)
            kl_loss_accum = torch.tensor(0.0, device=device)
            entropy_loss_accum = torch.tensor(0.0, device=device)
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

                # Monte Carlo entropy estimator: E[-log π(a)] ≈ mean_t(-log π(a_t))
                entropy_i = (-actor_lp).mean()
                entropy_loss_accum = entropy_loss_accum + entropy_i

                _kl_ratio = actor_lp - ref_lp
                kl = (torch.exp(_kl_ratio) - _kl_ratio - 1).mean()
                kl_loss_accum = kl_loss_accum + kl

                valid_count += 1

            if valid_count > 0:
                loss = (
                    policy_loss
                    + kl_coef * kl_loss_accum
                    - entropy_coeff * entropy_loss_accum
                ) / (valid_count * grad_accum_steps)
                accelerator.backward(loss)
                step_loss_val += loss.item()
                step_kl_val += (kl_loss_accum / valid_count).item()
                step_rewards.extend(rewards)
                prompts_done += 1

        if prompts_done == 0:
            global_step += 1
            continue

        torch.nn.utils.clip_grad_norm_(
            [p for p in actor.parameters() if p.requires_grad], max_grad_norm,
        )
        optimizer.step()

        agg_reward_mean = sum(step_rewards) / len(step_rewards)
        total_loss_acc += step_loss_val
        total_reward_acc += agg_reward_mean

        if accelerator.is_main_process and log_file is not None:
            log_record = {
                "step": global_step,
                "reward_mean": agg_reward_mean,
                "loss": step_loss_val,
                "kl_loss": step_kl_val / prompts_done,
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


# ============================================================================
# Multi-turn GRPO with Emo Reward (same pipeline as PPO-Emo)
# ============================================================================

def _compute_emo_reward_scalar(
    emo_point: float,
    emo_point_turns: List[float],
    reward_mode: str = "mode1",
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 0.2,
    trend_n: int = 5,
    step: int = 0,
    S1: int = 100,
    S2: int = 300,
    warmup_steps: int = 200,
) -> float:
    """Same scalar reward computation as PPO's compute_reward_tensors."""
    from .reward_emo import _trend_reward, _volatility_penalty, _clamp01

    baseline = emo_point / 100.0
    turns = emo_point_turns if emo_point_turns else [emo_point]

    if reward_mode == "mode1":
        return baseline
    elif reward_mode == "mode3":
        alpha = _clamp01((step - S1) / warmup_steps) if warmup_steps > 0 else 0.0
        beta = _clamp01((step - S2) / warmup_steps) if warmup_steps > 0 else 0.0
        trend_r = _trend_reward(turns, n=trend_n)
        vol_p = _volatility_penalty(turns, n=trend_n)
        return min(1.0, baseline + alpha * w2 * trend_r - beta * w3 * vol_p)
    elif reward_mode == "mode4":
        trend_r = _trend_reward(turns, n=trend_n)
        if step < S1:
            return baseline
        return min(1.0, max(0.0, trend_r * w2))
    else:  # mode2
        trend_r = _trend_reward(turns, n=trend_n)
        vol_p = _volatility_penalty(turns, n=trend_n)
        return min(1.0, w1 * baseline + w2 * trend_r - w3 * vol_p)


@torch.no_grad()
def _run_multiturn_grpo_rollout(
    profile: Dict[str, Any],
    prompt: str,
    model,
    tokenizer,
    user_llm_fn: Callable,
    planning_llm_fn: Optional[Callable],
    device: torch.device,
    max_turns: int = 8,
    max_new_tokens_per_turn: int = 128,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.95,
    target: str = "eq",
    sft_model_path: Optional[str] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, List[float]]]:
    """
    Single multi-turn dialogue rollout for GRPO (no critic needed).

    Returns the full sequence with a mask indicating which positions are model
    responses, plus emo_point data.  Returns None if no valid response produced.

    Returns: (full_ids (1,L), full_mask (1,L), resp_mask (1,L), emo_point, emo_point_turns)
    """
    from .hard_player_simulator_dsv3 import build_player_simulator_with_planning

    pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)

    sim = build_player_simulator_with_planning(
        profile=profile,
        player_llm_fn=user_llm_fn,
        planning_llm_fn=planning_llm_fn,
        sft_model_path=sft_model_path,
        target=target,
        initial_emo_point=50.0,
        device=str(device) if device else None,
    )
    # user_llm is an external API; if it fails (e.g., insufficient balance),
    # the simulator will fall back to a default first message.
    first_user = sim.generate_first_message()
    sim.dialog.append({"role": "user", "content": first_user})
    sim.emo_point_turns = [sim.emo_point]

    initial_text = prompt.rstrip() + "\n\n用户：" + first_user + "\n\nNPC："
    query_enc = tokenizer(
        initial_text, return_tensors="pt", truncation=True,
        max_length=2048, padding=False,
    )
    current_ids = query_enc["input_ids"].to(device)
    current_mask = torch.ones_like(current_ids, dtype=torch.float32)

    response_ranges: List[Tuple[int, int]] = []

    for _turn in range(max_turns):
        gen_out = model.generate(
            input_ids=current_ids,
            attention_mask=current_mask,
            max_new_tokens=max_new_tokens_per_turn,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
        )
        generated = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out

        resp_start = current_ids.size(1)
        resp_tokens = generated[0, resp_start:]
        valid_len = int((resp_tokens != pad_id).sum().item())
        if valid_len == 0:
            break

        resp_end = resp_start + valid_len
        response_ranges.append((resp_start, resp_end))

        npc_reply = tokenizer.decode(resp_tokens[:valid_len], skip_special_tokens=True)
        # sim.step may terminate early when external user_llm API is unavailable.
        user_reply, done = sim.step(npc_reply)

        current_ids = generated[:, :resp_end]
        current_mask = torch.ones_like(current_ids, dtype=torch.float32)

        if done:
            break

        next_text = "\n\n用户：" + user_reply + "\n\nNPC："
        next_enc = tokenizer(
            next_text, return_tensors="pt", truncation=True,
            max_length=512, padding=False, add_special_tokens=False,
        )
        next_ids = next_enc["input_ids"].to(device)
        current_ids = torch.cat([current_ids, next_ids], dim=1)
        current_mask = torch.ones_like(current_ids, dtype=torch.float32)

    if not response_ranges:
        return None

    full_len = current_ids.size(1)
    resp_mask = torch.zeros(1, full_len, dtype=torch.float32, device=device)
    for start, end in response_ranges:
        resp_mask[0, start:end] = 1.0

    return current_ids, current_mask, resp_mask, sim.get_emo_point(), sim.get_emo_point_turns()


def _masked_log_probs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    response_mask: torch.Tensor, _chunk: int = 256,
) -> torch.Tensor:
    """
    Compute token-level log_probs on the full sequence, masked to response positions.

    Uses chunked cross-entropy to reduce peak memory on long multi-turn sequences.
    Returns: (1, seq_len) with non-zero only at response positions.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    del outputs

    batch_size, seq_len, vocab = logits.shape
    seq_m1 = seq_len - 1
    shift_labels = input_ids[:, 1:]
    token_lp = torch.zeros(batch_size, seq_m1, dtype=logits.dtype, device=logits.device)

    for start in range(0, seq_m1, _chunk):
        end = min(start + _chunk, seq_m1)
        chunk_logits = logits[:, start:end, :].contiguous()
        chunk_labels = shift_labels[:, start:end].contiguous()
        token_lp[:, start:end] = -F.cross_entropy(
            chunk_logits.view(-1, vocab),
            chunk_labels.view(-1),
            reduction="none",
        ).view(batch_size, end - start)
        del chunk_logits, chunk_labels
    del logits, shift_labels

    full_lp = torch.zeros(
        batch_size, seq_len, dtype=token_lp.dtype, device=token_lp.device,
    )
    full_lp[:, 1:] = token_lp
    return full_lp * response_mask


def run_grpo_emo_training(cfg: Dict[str, Any]) -> None:
    """
    Multi-turn GRPO training with planning simulator + emo reward (3 modes).

    Uses the same rollout / user-simulator / reward pipeline as
    ``run_ppo_emo_training`` but optimises with GRPO (group-relative policy
    optimisation) instead of PPO.  No Critic network is needed.

    Config layout mirrors ``rl_default.yaml`` with ``rl.algo: "grpo_emo"`` and
    an ``rl.grpo`` section for GRPO-specific hyper-parameters.
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
    rl_cfg = (cfg.get("rl") or {}).get("grpo") or {}
    training_cfg = cfg.get("training", {}) or {}

    output_dir = training_cfg.get("output_dir", "outputs/grpo_emo")
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
        print(f"[GRPO-Emo] Accelerate: {accelerator.num_processes} process(es), device={device}")

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

    # ---------- 2. Dataset (ProfileDataset, same as PPO-Emo) ----------
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

    # ---------- 3. User simulator & planning (same as PPO-Emo) ----------
    # Planning: prefer API (planning_service_url / planning_llm); on failure fallback to local (rank 0 only). API path does not load local planner.
    user_llm_fn = _build_user_llm_fn(cfg)
    from .local_planning_llm import build_planning_llm_fn_prefer_api_then_local
    planning_llm_fn = build_planning_llm_fn_prefer_api_then_local(
        rollout_cfg, model_cfg, device,
        process_index=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    sft_model_path = model_cfg.get("sft_model_path")
    target = rollout_cfg.get("target", "eq")

    # ---------- 4. GRPO hyper-params ----------
    num_generations = rl_cfg.get("num_generations", 4)
    lr = rl_cfg.get("learning_rate", 1e-6)
    epsilon = rl_cfg.get("epsilon", 0.2)
    kl_coef = rl_cfg.get("beta", 0.02)
    entropy_coeff = rl_cfg.get("entropy_coeff", 0.0)
    max_grad_norm = rl_cfg.get("max_grad_norm", 1.0)
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
            f"[GRPO-Emo] 开始训练: total_steps={total_steps}, G={num_generations}, "
            f"lr={lr}, eps={epsilon}, reward_mode={reward_mode}"
        )
    accelerator.wait_for_everyone()

    # ---------- 6a. Resume from checkpoint ----------
    resume_from = training_cfg.get("resume_from_checkpoint")
    global_step = 0

    if resume_from and os.path.isdir(resume_from):
        if accelerator.is_main_process:
            print(f"[GRPO-Emo] Resuming from {resume_from}")
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
                print(f"[GRPO-Emo] Resumed: global_step={global_step}")

    from .monitor import TrainingMonitor
    monitor_cfg = cfg.get("monitor", {}) or {}
    monitor = TrainingMonitor(
        output_dir=output_dir,
        experiment_name=monitor_cfg.get("experiment_name", "grpo_emo"),
        use_tensorboard=monitor_cfg.get("use_tensorboard", True),
        use_wandb=monitor_cfg.get("use_wandb", False),
        wandb_project=monitor_cfg.get("wandb_project"),
        config=cfg,
        enabled=accelerator.is_main_process,
        resume=bool(resume_from),
    )

    grad_accum_steps = rl_cfg.get("gradient_accumulation_steps", 1)
    if accelerator.is_main_process and grad_accum_steps > 1:
        print(f"[GRPO-Emo] gradient_accumulation_steps={grad_accum_steps}, "
              f"effective prompts/step = {grad_accum_steps} × G={num_generations}")

    data_iter = iter(dataloader)
    total_loss_acc = 0.0
    total_reward_acc = 0.0
    t0 = time.time()

    while global_step < total_steps:
        optimizer.zero_grad()
        step_loss_val = 0.0
        step_kl_val = 0.0
        step_rewards_all: List[float] = []
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

            # --- Generate G multi-turn dialogues ---
            actor_unwrapped = accelerator.unwrap_model(actor)
            actor_unwrapped.eval()

            rollout_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, List[float]]] = []
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

            valid_count = len(rollout_results)
            if valid_count == 0:
                continue

            # --- Compute rewards ---
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

            # --- Accumulate gradients (scale loss by 1/grad_accum_steps) ---
            actor.train()
            is_last_accum = (_accum_idx == grad_accum_steps - 1)

            for i, (full_ids, full_mask, resp_mask, _, _) in enumerate(rollout_results):
                n_resp = resp_mask.sum().clamp(min=1)

                is_last_in_group = (i == valid_count - 1)
                should_sync = is_last_accum and is_last_in_group
                sync_ctx = contextlib.nullcontext() if should_sync else accelerator.no_sync(actor)

                with sync_ctx:
                    actor_lp = _masked_log_probs(actor, full_ids, full_mask, resp_mask)
                    with torch.no_grad():
                        ref_lp = _masked_log_probs(ref_model, full_ids, full_mask, resp_mask)
                        old_lp = actor_lp.detach()

                    ratio = torch.exp(actor_lp - old_lp)
                    adv = advantages[i]
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv
                    pg_loss = -(torch.minimum(surr1, surr2) * resp_mask).sum() / n_resp
                    # Monte Carlo entropy estimator on response tokens
                    entropy_i = (-(actor_lp * resp_mask)).sum() / n_resp
                    _kl_ratio = actor_lp - ref_lp
                    kl = ((torch.exp(_kl_ratio) - _kl_ratio - 1) * resp_mask).sum() / n_resp

                    loss_i = (
                        pg_loss
                        + kl_coef * kl
                        - entropy_coeff * entropy_i
                    ) / (valid_count * grad_accum_steps)
                    accelerator.backward(loss_i)

                step_loss_val += loss_i.item()
                step_kl_val += kl.item()

            step_rewards_all.extend(rewards)
            prompts_done += 1

        if prompts_done == 0:
            global_step += 1
            continue

        accelerator.clip_grad_norm_(
            [p for p in accelerator.unwrap_model(actor).parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()

        agg_reward_mean = sum(step_rewards_all) / len(step_rewards_all)
        total_loss_acc += step_loss_val
        total_reward_acc += agg_reward_mean

        monitor.log(step=global_step, metrics={
            "reward_mean": agg_reward_mean,
            "rewards_max": max(step_rewards_all),
            "rewards_min": min(step_rewards_all),
            "loss": step_loss_val,
            "kl_loss": step_kl_val / max(len(step_rewards_all), 1),
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
        print(f"[GRPO-Emo] 训练完成，模型已保存到 {final_dir}，指标日志: {log_path}")
