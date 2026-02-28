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
    逐条跑 run_multi_turn_rollout_single，再 pad 成 batch。
    返回 dict: query_ids, query_mask, response_ids, response_mask, log_probs, values,
               non_tensor_batch: { emo_point, emo_point_turns }。
    """
    results = []
    for item in batch_items:
        out = run_multi_turn_rollout_single(
            profile=item["profile"],
            prompt=item["prompt"],
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
            target=item.get("target", target),
            sft_model_path=sft_model_path,
            planning_llm_fn=planning_llm_fn,
        )
        results.append(out)

    # Pad 成 batch
    pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)
    q_list = [r[0].squeeze(0) for r in results]
    r_list = [r[1].squeeze(0) for r in results]
    m_list = [r[2].squeeze(0) for r in results]
    lp_list = [r[3].squeeze(0) for r in results]
    v_list = [r[4].squeeze(0) for r in results]

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
        query_mask[i, : t.size(0)] = 1.0

    response_ids = T.full((len(r_list), max_r), pad_id, dtype=T.long, device=device)
    response_mask = T.zeros(len(r_list), max_r, dtype=T.float32, device=device)
    for i, t in enumerate(r_list):
        response_ids[i, : t.size(0)] = t
        response_mask[i, : t.size(0)] = (t != pad_id).to(T.float32)

    log_probs, _ = _pad_to(lp_list, max_r, 0.0)
    values, _ = _pad_to(v_list, max_r, 0.0)

    emo_points = [float(r[5]) for r in results]
    emo_point_turns_list = [list(r[6]) for r in results]

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
