# -*- coding: utf-8 -*-
"""
On-policy DPO 多轮对话 Rollout：每轮生成 k 个回答，用 reward 选 best/worst，构造 DPO 偏好对。
流程：用户说话 → policy 生成 k 个回复 → 情感打分 → 选总分最高为 chosen、最低为 rejected →
下一轮从 chosen 继续。

打分支持两种模式：
- reward_emo: 对回复文本做情感分类/关键词打分（与 PPO 相同）
- planning: 用 emo_planning 在完整上下文中分析 NPC 回复对用户情绪的影响
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch as T


def run_dpo_rollout_single(
    profile: Dict[str, Any],
    prompt: str,
    model: T.nn.Module,
    tokenizer: Any,
    score_fn: Callable[[List[str], Dict[str, Any]], List[float]],
    user_sim,
    device: T.device,
    num_samples: int = 4,
    max_turns: int = 8,
    use_planning_score: bool = True,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 1.0,
) -> List[Dict[str, str]]:
    """
    单条 profile 的多轮 DPO rollout，返回 DPO 偏好对列表。

    user_sim: 需有 generate_first_message()、step(npc_reply) -> (user_reply, done)、dialog、emo_point
    score_fn: (responses, context) -> List[float]，context 含 profile, dialog, emo_point
              planning 模式会用到完整上下文，reward_emo 模式可忽略 context

    返回: [{"prompt": "...", "chosen": "...", "rejected": "..."}, ...]，每轮最多一个 pair
    """
    pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)

    first_user = user_sim.generate_first_message()
    first_user = (first_user or "").strip() or "我最近有些事想和你聊聊。"

    user_sim.dialog.append({"role": "user", "content": first_user})
    if hasattr(user_sim, "emo_point_turns"):
        user_sim.emo_point_turns = [getattr(user_sim, "emo_point", 50.0)]

    current_text = prompt.rstrip() + "\n\n用户：" + first_user + "\n\nNPC："
    cumulative_total = 0.0
    dpo_pairs: List[Dict[str, str]] = []
    turn = 0

    while turn < max_turns:
        enc = tokenizer(
            current_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc.get("attention_mask")
        if attn is None:
            attn = T.ones_like(input_ids, dtype=T.long, device=device)

        responses: List[str] = []
        for _ in range(num_samples):
            with T.no_grad():
                out = model.generate(
                    input_ids,
                    attention_mask=attn,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_id,
                )
            new_tokens = out[0, input_ids.size(1) :]
            r_len = (new_tokens != pad_id).sum().item()
            if r_len == 0:
                r_len = new_tokens.size(0)
            txt = tokenizer.decode(new_tokens[:r_len], skip_special_tokens=True).strip()
            responses.append(txt or "…")

        if not responses:
            break

        context = {
            "profile": profile,
            "dialog": getattr(user_sim, "dialog", []),
            "emo_point": getattr(user_sim, "emo_point", 50.0),
        }
        raw_scores = score_fn(responses, context)
        scores = [float(r) if isinstance(r, (int, float)) else 50.0 for r in raw_scores]
        for i, r in enumerate(scores):
            scores[i] = max(0.0, min(100.0, r)) if use_planning_score else max(0.0, min(1.0, r))

        if use_planning_score:
            totals = scores
        else:
            totals = [cumulative_total + r for r in scores]
        best_idx = int(T.tensor(totals).argmax().item())
        worst_idx = int(T.tensor(totals).argmin().item())

        chosen = responses[best_idx]
        rejected = responses[worst_idx]

        if chosen.strip() and rejected.strip() and chosen != rejected:
            dpo_pairs.append({
                "prompt": current_text,
                "chosen": chosen,
                "rejected": rejected,
            })

        cumulative_total = totals[best_idx]
        user_reply, done = user_sim.step(chosen)
        user_reply = (user_reply or "").strip() or "…"

        if done:
            break

        current_text = (
            current_text.rstrip()
            + "\n\n"
            + chosen
            + "\n\n用户："
            + user_reply
            + "\n\nNPC："
        )
        turn += 1

    return dpo_pairs


def run_dpo_rollout_batch(
    batch_items: List[Dict[str, Any]],
    model: T.nn.Module,
    tokenizer: Any,
    score_fn: Callable[[List[str], Dict[str, Any]], List[float]],
    build_user_sim: Callable[[Dict[str, Any]], Any],
    device: T.device,
    num_samples: int = 4,
    max_turns: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 1.0,
    use_planning_score: bool = True,
) -> List[Dict[str, str]]:
    """
    对一批 profile 做 DPO rollout，合并所有偏好对。
    build_user_sim(profile) 返回 user_sim 实例。
    """
    all_pairs: List[Dict[str, str]] = []
    for item in batch_items:
        profile = item["profile"]
        prompt = item["prompt"]
        user_sim = build_user_sim(profile)
        pairs = run_dpo_rollout_single(
            profile=profile,
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            score_fn=score_fn,
            user_sim=user_sim,
            device=device,
            num_samples=num_samples,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_planning_score=use_planning_score,
        )
        all_pairs.extend(pairs)
    return all_pairs
