# -*- coding: utf-8 -*-
"""
多轮对话 Rollout：用 data/data 下的用户形象，Actor 与 PlayerSimulatorWithPlanning 多轮对话，
收集每条的 response_ids、response_mask、log_probs、values、emo_point、emo_point_turns，
供 reward 函数与 PPO 使用。情感分析使用 planning_reply（LLM prompt）。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch as T

from .hard_player_simulator_dsv3 import build_player_simulator_with_planning


def run_multi_turn_rollout_single(
    profile: Dict[str, Any],
    prompt: str,
    actor_ref,
    critic,
    tokenizer,
    user_llm_fn: Callable[[List[Dict[str, str]]], str],
    device: T.device = None,
    max_turns: int = 15,
    max_new_tokens_per_turn: int = 256,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 1.0,
    target: str = "eq",
    sft_model_path: Optional[str] = None,
    planning_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor, float, List[float]]:
    """
    单条样本的多轮对话 rollout。使用 planning_reply（LLM prompt）做情感分析。
    sft_model_path: planning 用本地 SFT 基座时指定路径；或直接传 planning_llm_fn。
    返回: query_ids (1, q_len), response_ids (1, r_len), response_mask (1, r_len),
          log_probs (1, r_len), values (1, r_len), emo_point, emo_point_turns。
    """
    pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)
    device = device or (next(actor_ref.parameters(), T.tensor(0)).device if hasattr(actor_ref, "parameters") else T.device("cpu"))

    sim = build_player_simulator_with_planning(
        profile=profile,
        player_llm_fn=user_llm_fn,
        planning_llm_fn=planning_llm_fn,
        sft_model_path=sft_model_path,
        target=target,
        initial_emo_point=50.0,
        device=str(device) if device else None,
    )
    first_user = sim.generate_first_message()
    sim.dialog.append({"role": "user", "content": first_user})
    sim.emo_point_turns = [sim.emo_point]

    # 拼接成给 NPC 的初始 prompt：背景 + "用户：" + first_user + "\nNPC："
    initial_text = prompt.rstrip() + "\n\n用户：" + first_user + "\n\nNPC："
    query_enc = tokenizer(
        initial_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=False,
    )
    query_ids = query_enc["input_ids"].to(device)
    query_mask = query_enc.get("attention_mask")
    if query_mask is not None:
        query_mask = query_mask.to(device)
    else:
        query_mask = T.ones_like(query_ids, dtype=T.float32, device=device)

    all_resp_ids: List[T.Tensor] = []
    all_log_probs: List[T.Tensor] = []
    all_values: List[T.Tensor] = []
    turn = 0
    current_query_ids = query_ids
    current_query_mask = query_mask

    while turn < max_turns:
        resp_ids, resp_mask, log_probs = actor_ref.generate(
            current_query_ids,
            current_query_mask,
            max_new_tokens=max_new_tokens_per_turn,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        r_len = int(resp_mask[0].sum().item())
        if r_len == 0:
            break
        npc_reply = tokenizer.decode(resp_ids[0, :r_len], skip_special_tokens=True)
        user_reply, done = sim.step(npc_reply)

        all_resp_ids.append(resp_ids[0, :r_len])
        all_log_probs.append(log_probs[0, :r_len])
        # values: 需要 critic(full_ids, full_attn) 再取 response 段
        full_ids = T.cat([current_query_ids, resp_ids[:, :r_len]], dim=1)
        full_attn = T.cat([
            current_query_mask,
            resp_mask[:, :r_len],
        ], dim=1)
        with T.no_grad():
            values_full = critic(full_ids, full_attn)
        v_resp = values_full[0, -r_len:]
        all_values.append(v_resp)

        if done:
            break
        # 下一轮输入 = 当前 full + "用户：" + user_reply + "\nNPC："
        next_suffix = "\n\n用户：" + user_reply + "\n\nNPC："
        next_enc = tokenizer(
            next_suffix,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
            add_special_tokens=False,
        )
        next_ids = next_enc["input_ids"].to(device)
        next_mask = next_enc.get("attention_mask")
        if next_mask is None:
            next_mask = T.ones_like(next_ids, dtype=T.float32, device=device)
        else:
            next_mask = next_mask.to(device)
        current_query_ids = T.cat([full_ids, next_ids], dim=1)
        current_query_mask = T.cat([full_attn, next_mask], dim=1)
        turn += 1

    if not all_resp_ids:
        # 没有任何有效回复，造一个占位
        all_resp_ids = [T.zeros(1, dtype=T.long, device=device)]
        all_log_probs = [T.zeros(1, dtype=T.float32, device=device)]
        all_values = [T.zeros(1, dtype=T.float32, device=device)]

    response_ids = T.cat(all_resp_ids, dim=0).unsqueeze(0)
    log_probs_cat = T.cat(all_log_probs, dim=0).unsqueeze(0)
    values_cat = T.cat(all_values, dim=0).unsqueeze(0)
    response_mask = (response_ids != pad_id).to(T.float32)
    emo_point = sim.get_emo_point()
    emo_point_turns = sim.get_emo_point_turns()

    return (
        query_ids,
        response_ids,
        response_mask,
        log_probs_cat,
        values_cat,
        emo_point,
        emo_point_turns,
    )


def run_multi_turn_rollout_batch(
    batch_items: List[Dict[str, Any]],
    actor_ref,
    critic,
    tokenizer,
    user_llm_fn: Callable[[List[Dict[str, str]]], str],
    device: T.device = None,
    max_turns: int = 15,
    max_new_tokens_per_turn: int = 256,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 1.0,
    target: str = "eq",
    sft_model_path: Optional[str] = None,
    planning_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    """
    batch_items: 每项 {"profile", "prompt", "idx"}。
    并行多轮 rollout：一批对话共享 turn-loop，每一轮对所有未结束对话做一次 batched generate。

    返回 dict: query_ids, query_mask, response_ids, response_mask, log_probs, values,
               non_tensor_batch: { emo_point, emo_point_turns }。
    """
    pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)
    device = device or next(actor_ref.parameters(), T.tensor(0)).device  # type: ignore[arg-type]

    batch_size = len(batch_items)

    # 为每条对话构建独立的 PlayerSimulator，并初始化首轮 user 消息与初始 prompt
    sims = []
    init_query_list: List[T.Tensor] = []
    init_query_mask_list: List[T.Tensor] = []
    curr_query_list: List[T.Tensor] = []
    curr_query_mask_list: List[T.Tensor] = []
    turn_counts = [0 for _ in range(batch_size)]
    done_flags = [False for _ in range(batch_size)]

    for item in batch_items:
        profile = item["profile"]
        prompt = item["prompt"]

        sim = build_player_simulator_with_planning(
            profile=profile,
            player_llm_fn=user_llm_fn,
            planning_llm_fn=planning_llm_fn,
            sft_model_path=sft_model_path,
            target=target,
            initial_emo_point=50.0,
            device=str(device) if device else None,
        )
        first_user = sim.generate_first_message()
        sim.dialog.append({"role": "user", "content": first_user})
        sim.emo_point_turns = [sim.emo_point]
        sims.append(sim)

        initial_text = prompt.rstrip() + "\n\n用户：" + first_user + "\n\nNPC："
        enc = tokenizer(
            initial_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        q_ids = enc["input_ids"].to(device)
        q_mask = enc.get("attention_mask")
        if q_mask is not None:
            q_mask = q_mask.to(device)
        else:
            q_mask = T.ones_like(q_ids, dtype=T.float32, device=device)

        init_query_list.append(q_ids[0])
        init_query_mask_list.append(q_mask[0])
        curr_query_list.append(q_ids[0])
        curr_query_mask_list.append(q_mask[0])

    # 存储每条对话在所有轮次上的 NPC 回复与 value
    all_resp_ids_list: List[List[T.Tensor]] = [[] for _ in range(batch_size)]
    all_log_probs_list: List[List[T.Tensor]] = [[] for _ in range(batch_size)]
    all_values_list: List[List[T.Tensor]] = [[] for _ in range(batch_size)]

    # turn-level 并行 rollout
    while True:
        alive_indices = [
            i for i in range(batch_size)
            if (not done_flags[i]) and (turn_counts[i] < max_turns)
        ]
        if not alive_indices:
            break

        # 构建当前未结束对话的 batch 输入
        lens = [curr_query_list[i].size(0) for i in alive_indices]
        max_q = max(lens)
        bsz_alive = len(alive_indices)
        batch_q_ids = T.full(
            (bsz_alive, max_q), pad_id, dtype=T.long, device=device,
        )
        batch_q_mask = T.zeros(
            bsz_alive, max_q, dtype=T.float32, device=device,
        )
        for j, i in enumerate(alive_indices):
            L = curr_query_list[i].size(0)
            batch_q_ids[j, :L] = curr_query_list[i]
            batch_q_mask[j, :L] = curr_query_mask_list[i]

        # 一次 batched generate，为所有未结束对话生成 NPC 回复
        resp_ids_batch, resp_mask_batch, log_probs_batch = actor_ref.generate(
            batch_q_ids,
            batch_q_mask,
            max_new_tokens=max_new_tokens_per_turn,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
        )

        # 对每条对话分别更新 simulator / critic / query
        for j, i in enumerate(alive_indices):
            resp_ids = resp_ids_batch[j]
            resp_mask = resp_mask_batch[j]
            r_len = int(resp_mask.sum().item())

            if r_len == 0:
                done_flags[i] = True
                continue

            npc_reply = tokenizer.decode(resp_ids[:r_len], skip_special_tokens=True)
            user_reply, done = sims[i].step(npc_reply)

            all_resp_ids_list[i].append(resp_ids[:r_len])
            all_log_probs_list[i].append(log_probs_batch[j, :r_len])

            # critic 仍按单条对话 forward，生成 value 序列
            full_ids = T.cat(
                [curr_query_list[i].unsqueeze(0), resp_ids[:r_len].unsqueeze(0)],
                dim=1,
            )
            full_attn = T.ones_like(full_ids, dtype=T.float32, device=device)
            full_attn[:, : curr_query_list[i].size(0)] = curr_query_mask_list[i].unsqueeze(0)
            full_attn[:, curr_query_list[i].size(0) :] = resp_mask[:r_len].unsqueeze(0)
            with T.no_grad():
                values_full = critic(full_ids, full_attn)
            v_resp = values_full[0, -r_len:]
            all_values_list[i].append(v_resp)

            turn_counts[i] += 1
            if done or turn_counts[i] >= max_turns:
                done_flags[i] = True
                continue

            # 构建下一轮的 query：当前 full + 用户回复
            next_suffix = "\n\n用户：" + user_reply + "\n\nNPC："
            next_enc = tokenizer(
                next_suffix,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
                add_special_tokens=False,
            )
            next_ids = next_enc["input_ids"].to(device)
            next_mask = next_enc.get("attention_mask")
            if next_mask is None:
                next_mask = T.ones_like(next_ids, dtype=T.float32, device=device)
            else:
                next_mask = next_mask.to(device)

            curr_query_list[i] = T.cat(
                [full_ids[0], next_ids[0]],
                dim=0,
            )
            curr_query_mask_list[i] = T.cat(
                [full_attn[0], next_mask[0]],
                dim=0,
            )

    # 将每条对话的多轮回复拼接，并 pad 成 batch
    q_list = [q for q in init_query_list]
    q_mask_list = [m for m in init_query_mask_list]
    r_list: List[T.Tensor] = []
    lp_list: List[T.Tensor] = []
    v_list: List[T.Tensor] = []
    emo_points: List[float] = []
    emo_point_turns_list: List[List[float]] = []

    for i in range(batch_size):
        if not all_resp_ids_list[i]:
            resp = T.zeros(1, dtype=T.long, device=device)
            lp = T.zeros(1, dtype=T.float32, device=device)
            val = T.zeros(1, dtype=T.float32, device=device)
        else:
            resp = T.cat(all_resp_ids_list[i], dim=0)
            lp = T.cat(all_log_probs_list[i], dim=0)
            val = T.cat(all_values_list[i], dim=0)
        r_list.append(resp)
        lp_list.append(lp)
        v_list.append(val)
        emo_points.append(float(sims[i].get_emo_point()))
        emo_point_turns_list.append(list(sims[i].get_emo_point_turns()))

    max_q = max(t.size(0) for t in q_list)
    max_r = max(t.size(0) for t in r_list)

    def _pad_to(tensors: List[T.Tensor], max_len: int, pad_val: float) -> Tuple[T.Tensor, T.Tensor]:
        stacked = T.full(
            (len(tensors), max_len), pad_val, dtype=tensors[0].dtype, device=device,
        )
        mask = T.zeros(len(tensors), max_len, dtype=T.float32, device=device)
        for i, t in enumerate(tensors):
            L = t.size(0)
            stacked[i, :L] = t
            mask[i, :L] = 1.0
        return stacked, mask

    query_ids = T.full((len(q_list), max_q), pad_id, dtype=T.long, device=device)
    query_mask = T.zeros(len(q_list), max_q, dtype=T.float32, device=device)
    for i, t in enumerate(q_list):
        query_ids[i, : t.size(0)] = t
        query_mask[i, : t.size(0)] = q_mask_list[i]

    response_ids = T.full((len(r_list), max_r), pad_id, dtype=T.long, device=device)
    response_mask = T.zeros(len(r_list), max_r, dtype=T.float32, device=device)
    for i, t in enumerate(r_list):
        response_ids[i, : t.size(0)] = t
        response_mask[i, : t.size(0)] = (t != pad_id).to(T.float32)

    log_probs, _ = _pad_to(lp_list, max_r, 0.0)
    values, _ = _pad_to(v_list, max_r, 0.0)

    return {
        "query_ids": query_ids,
        "query_mask": query_mask,
        "response_ids": response_ids,
        "response_mask": response_mask,
        "log_probs": log_probs,
        "values": values,
        "non_tensor_batch": {
            "emo_point": emo_points,
            "emo_point_turns": emo_point_turns_list,
        },
    }


def collect_rollouts_emo(
    batch_items: List[Dict[str, Any]],
    actor_ref,
    critic,
    tokenizer,
    user_llm_fn,
    memory,
    get_ref_log_probs_fn,
    device: T.device,
    reward_mode: str = "mode1",
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 0.2,
    trend_n: int = 5,
    max_turns: int = 15,
    max_new_tokens_per_turn: int = 256,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 1.0,
    target: str = "eq",
    sft_model_path: Optional[str] = None,
    planning_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    # mode3 三段式
    step: Optional[int] = None,
    S1: Optional[int] = None,
    S2: Optional[int] = None,
    warmup_steps: Optional[int] = None,
) -> Tuple[T.Tensor, T.Tensor]:
    """
    多轮对话 rollout + reward 计算 + 写入 memory。使用 planning_reply（LLM prompt）做情感分析。
    get_ref_log_probs_fn: (query_ids, query_mask, response_ids, response_mask) -> ref_log_probs (batch, resp_len)。
    mode3 时需传入 step, S1, S2, warmup_steps。
    返回 (original_reward_tensor, penalized_reward_tensor, component_stats) 供 trainer 记录。
    component_stats: dict with baseline_mean, trend_mean, vol_mean。
    """
    from .reward_emo import compute_reward_tensors, _trend_reward, _volatility_penalty

    gen_batch = run_multi_turn_rollout_batch(
        batch_items=batch_items,
        actor_ref=actor_ref,
        critic=critic,
        tokenizer=tokenizer,
        user_llm_fn=user_llm_fn,
        device=device,
        max_turns=max_turns,
        max_new_tokens_per_turn=max_new_tokens_per_turn,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        target=target,
        sft_model_path=sft_model_path,
        planning_llm_fn=planning_llm_fn,
    )
    non_tensor = gen_batch.get("non_tensor_batch") or {}
    emo_points = non_tensor.get("emo_point", [0.0] * len(batch_items))
    emo_point_turns_list = non_tensor.get("emo_point_turns", [])
    use_turns = reward_mode in ("mode2", "mode3")

    original_reward_tensor, penalized_reward_tensor = compute_reward_tensors(
        response_ids=gen_batch["response_ids"],
        response_mask=gen_batch["response_mask"],
        emo_points=emo_points,
        emo_point_turns_list=emo_point_turns_list if use_turns else None,
        reward_mode=reward_mode,
        w1=w1,
        w2=w2,
        w3=w3,
        trend_n=trend_n,
        device=device,
        step=step,
        S1=S1,
        S2=S2,
        warmup_steps=warmup_steps,
    )

    # 标量 reward：取每条最后一个有效位置的值，用于 GAE
    batch_size = gen_batch["response_ids"].size(0)
    rewards_scalar = T.zeros(batch_size, dtype=T.float32, device=device)
    for i in range(batch_size):
        m = gen_batch["response_mask"][i]
        length = int(m.sum().item())
        if length > 0:
            rewards_scalar[i] = original_reward_tensor[i, length - 1]
        else:
            rewards_scalar[i] = 0.0

    import os
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    print(
        f"[rank{rank}] emo_points={emo_points} | "
        f"resp_lengths={[int(gen_batch['response_mask'][i].sum().item()) for i in range(batch_size)]} | "
        f"rewards={rewards_scalar.tolist()} | "
        f"turns={emo_point_turns_list}"
    )

    ref_log_probs = get_ref_log_probs_fn(
        gen_batch["query_ids"],
        gen_batch["query_mask"],
        gen_batch["response_ids"],
        gen_batch["response_mask"],
    )

    memory.store_batch(
        gen_batch["query_ids"],
        gen_batch["query_mask"],
        gen_batch["response_ids"],
        gen_batch["response_mask"],
        gen_batch["log_probs"],
        gen_batch["values"],
        rewards_scalar,
        ref_log_probs,
    )

    baselines = [emo / 100.0 for emo in emo_points]
    trends = []
    vols = []
    for i in range(len(emo_points)):
        turns = emo_point_turns_list[i] if emo_point_turns_list and i < len(emo_point_turns_list) else [emo_points[i]]
        trends.append(_trend_reward(turns, n=trend_n))
        vols.append(_volatility_penalty(turns, n=trend_n))
    component_stats = {
        "baseline_mean": sum(baselines) / max(len(baselines), 1),
        "trend_mean": sum(trends) / max(len(trends), 1),
        "vol_mean": sum(vols) / max(len(vols), 1),
    }

    return original_reward_tensor, penalized_reward_tensor, component_stats
