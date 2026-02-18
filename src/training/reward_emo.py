# -*- coding: utf-8 -*-
"""
Reward 为函数（非 reward model）：根据对话结束时的 emo_point（及可选每轮 emo 序列）计算 reward。
支持三种模式（通过参数切换）：
- 模式1：reward = 最终 emo_point/100，sparse（只在序列最后一个有效 token 位置非零）。
- 模式2：r_total = w1*baseline_emotion + w2*trend_reward - w3*volatility_penalty。
- 模式3（三段式）：alpha/beta 随 step warmup，reward = baseline + alpha*w2*trend - beta*w3*volatility。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as T


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _trend_reward(emo_point_turns: List[float], n: int = 5) -> float:
    """最近 n 轮情绪变化趋势：线性斜率或平均差异。上升 -> 正，下降或无明显改善 -> 低或 0。"""
    if len(emo_point_turns) < 2 or n < 1:
        return 0.0
    recent = emo_point_turns[-n:] if len(emo_point_turns) >= n else emo_point_turns
    x = np.arange(len(recent), dtype=np.float32)
    y = np.array(recent, dtype=np.float32)
    if np.std(y) < 1e-6:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    # 归一化到约 [0, 1]：斜率每 1 点/轮 约 0.1 奖励
    return float(np.clip(slope * 0.1, 0.0, 1.0))


def _volatility_penalty(emo_point_turns: List[float], n: int = 5) -> float:
    """最近 n 轮情绪波动（方差或绝对差）。波动大 -> 惩罚大，波动小 -> 惩罚低。"""
    if len(emo_point_turns) < 2 or n < 1:
        return 0.0
    recent = emo_point_turns[-n:] if len(emo_point_turns) >= n else emo_point_turns
    y = np.array(recent, dtype=np.float32)
    variance = np.var(y)
    # 方差约 0-2500（100^2），归一化到 [0, 1] 惩罚
    return float(min(1.0, variance / 400.0))


def compute_reward_tensors(
    response_ids: T.Tensor,
    response_mask: T.Tensor,
    emo_points: List[float],
    emo_point_turns_list: Optional[List[List[float]]] = None,
    reward_mode: str = "mode1",
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 0.2,
    trend_n: int = 5,
    device: Optional[T.device] = None,
    # mode3 三段式 warmup
    step: Optional[int] = None,
    S1: Optional[int] = None,
    S2: Optional[int] = None,
    warmup_steps: Optional[int] = None,
) -> Tuple[T.Tensor, T.Tensor]:
    """
    根据 non_tensor_batch 中的 emo_point（及可选的 emo_point_turns）计算 reward，
    并填到每条回复的「最后一个有效 token」位置，得到 original_reward_tensor 与 penalized_reward_tensor。

    response_ids: (batch, resp_len)
    response_mask: (batch, resp_len)，1 表示有效 token
    emo_points: 长度为 batch 的列表，每个为对话结束时的 emo_point [0,100]
    emo_point_turns_list: mode2/mode3 时每条的每轮 emo_point 序列
    reward_mode: "mode1" / "mode2" / "mode3"
    w1, w2, w3: baseline / trend / volatility 权重
    trend_n: 计算趋势和波动使用的最近轮数
    step, S1, S2, warmup_steps: 仅 mode3。alpha = clamp((step-S1)/warmup_steps,0,1), beta = clamp((step-S2)/warmup_steps,0,1)，
        reward = baseline + alpha*w2*trend - beta*w3*volatility

    返回 (original_reward_tensor, penalized_reward_tensor)，形状均为 (batch, resp_len)。
    """
    batch_size = response_ids.size(0)
    resp_len = response_ids.size(1)
    if device is None:
        device = response_ids.device

    # mode3 的 alpha, beta
    alpha, beta = 0.0, 0.0
    if reward_mode == "mode3" and step is not None and S1 is not None and S2 is not None and warmup_steps is not None and warmup_steps > 0:
        alpha = _clamp01((step - S1) / warmup_steps)
        beta = _clamp01((step - S2) / warmup_steps)

    rewards_batched: List[float] = []
    for i in range(batch_size):
        emo = emo_points[i] if i < len(emo_points) else 0.0
        baseline = max(0.0, emo / 100.0)

        if reward_mode == "mode1":
            r = baseline
        elif reward_mode == "mode3":
            turns = (emo_point_turns_list or [])[i] if emo_point_turns_list and i < len(emo_point_turns_list) else [emo]
            trend_r = _trend_reward(turns, n=trend_n)
            vol_p = _volatility_penalty(turns, n=trend_n)
            r = baseline + alpha * w2 * trend_r - beta * w3 * vol_p
            r = max(0.0, min(1.0, r))
        else:
            # mode2
            turns = (emo_point_turns_list or [])[i] if emo_point_turns_list and i < len(emo_point_turns_list) else [emo]
            trend_r = _trend_reward(turns, n=trend_n)
            vol_p = _volatility_penalty(turns, n=trend_n)
            r = w1 * baseline + w2 * trend_r - w3 * vol_p
            r = max(0.0, min(1.0, r))
        rewards_batched.append(r)

    # 放到每条回复的最后一个有效 token 位置
    original_reward_tensor = T.zeros(batch_size, resp_len, dtype=T.float32, device=device)
    for i in range(batch_size):
        mask = response_mask[i]
        length = int(mask.sum().item())
        if length > 0:
            original_reward_tensor[i, length - 1] = rewards_batched[i]

    penalized_reward_tensor = original_reward_tensor.clone()
    return original_reward_tensor, penalized_reward_tensor


def reward_fn_from_batch(
    data: Dict[str, Any],
    reward_mode: str = "mode1",
    w1: float = 1.0,
    w2: float = 0.3,
    w3: float = 0.2,
    trend_n: int = 5,
    step: Optional[int] = None,
    S1: Optional[int] = None,
    S2: Optional[int] = None,
    warmup_steps: Optional[int] = None,
) -> Tuple[T.Tensor, T.Tensor]:
    """
    从已包含 response_ids, response_mask 和 non_tensor_batch['emo_point']（及可选的 'emo_point_turns'）的
    data 中计算 reward 张量。mode3 时需传入 step, S1, S2, warmup_steps。
    返回 (original_reward_tensor, penalized_reward_tensor)。
    """
    non_tensor = data.get("non_tensor_batch") or data
    emo_points = list(non_tensor.get("emo_point", [0.0]))
    if isinstance(emo_points, (int, float)):
        emo_points = [float(emo_points)]
    emo_point_turns_list = non_tensor.get("emo_point_turns")
    response_ids = data["response_ids"]
    response_mask = data["response_mask"]
    return compute_reward_tensors(
        response_ids=response_ids,
        response_mask=response_mask,
        emo_points=emo_points,
        emo_point_turns_list=emo_point_turns_list,
        reward_mode=reward_mode,
        w1=w1,
        w2=w2,
        w3=w3,
        trend_n=trend_n,
        device=response_ids.device,
        step=step,
        S1=S1,
        S2=S2,
        warmup_steps=warmup_steps,
    )
