#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 PPO 训练的 stdout 日志中解析 emo_point_turns，计算 baseline / trend / volatility
三个分量的分布统计，并用等方差贡献法推荐 w1, w2, w3 权重。

用法:
  python scripts/analyze_reward_components.py --log /path/to/ppo_xxx.out
  python scripts/analyze_reward_components.py --log /path/to/ppo_xxx.out --ratio 7:2:1
  python scripts/analyze_reward_components.py --log /path/to/ppo_xxx.out --trend-n 5
"""
import argparse
import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.reward_emo import _trend_reward, _volatility_penalty


def parse_turns_from_log(log_path: str) -> List[List[float]]:
    """从日志文件中提取所有 turns=[[...], [...]] 数据。"""
    pattern = re.compile(r"turns=(\[\[.*?\]\])")
    all_turns: List[List[float]] = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            try:
                parsed = ast.literal_eval(m.group(1))
                for t in parsed:
                    if isinstance(t, list) and len(t) >= 1:
                        all_turns.append([float(x) for x in t])
            except (ValueError, SyntaxError):
                continue

    return all_turns


def _compute_components(
    all_turns: List[List[float]], trend_n: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    baselines, trends, vols, turn_lengths = [], [], [], []
    for turns in all_turns:
        baselines.append(turns[-1] / 100.0)
        trends.append(_trend_reward(turns, n=trend_n))
        vols.append(_volatility_penalty(turns, n=trend_n))
        turn_lengths.append(len(turns))
    return np.array(baselines), np.array(trends), np.array(vols), np.array(turn_lengths)


def _scale_weights(
    raw_w1: float, raw_w2: float, raw_w3: float, ceiling: float = 0.9,
) -> Tuple[float, float, float]:
    """统一缩放使 w1 + w2 = ceiling（正向部分不超过 ceiling）。"""
    k = ceiling / (raw_w1 + raw_w2)
    return raw_w1 * k, raw_w2 * k, raw_w3 * k


def _print_weight_eval(
    label: str, w1: float, w2: float, w3: float,
    baselines: np.ndarray, trends: np.ndarray, vols: np.ndarray,
) -> None:
    """打印一组权重下的实际效果。"""
    b_mean, t_mean, v_mean = baselines.mean(), trends.mean(), vols.mean()
    c_b = w1 * b_mean
    c_t = w2 * t_mean
    c_v = w3 * v_mean
    total_pos = max(c_b + c_t, 1e-8)
    rewards = w1 * baselines + w2 * trends - w3 * vols
    clip_rate = np.mean((w1 * baselines + w2 * trends) > 1.0) * 100
    neg_rate = np.mean(rewards < 0) * 100

    print(f"\n  [{label}]  w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")
    print(f"    w1+w2={w1+w2:.4f}")
    print(f"    各分量 mean 贡献: baseline={c_b:.4f} ({c_b/total_pos*100:.1f}%), "
          f"trend={c_t:.4f} ({c_t/total_pos*100:.1f}%), vol={c_v:.4f} (penalty)")
    print(f"    reward: mean={rewards.mean():.4f}, std={rewards.std():.4f}, "
          f"min={rewards.min():.4f}, max={min(1.0, rewards.max()):.4f}")
    print(f"    被 1.0 截断: {clip_rate:.1f}%,  负 reward: {neg_rate:.1f}%")


def analyze(
    all_turns: List[List[float]], trend_n: int = 5,
    user_ratio: str = None,
) -> None:
    baselines, trends, vols, turn_lengths = _compute_components(all_turns, trend_n)

    print(f"\n{'='*65}")
    print(f"  Reward Component Analysis  (N={len(all_turns)} rollouts, trend_n={trend_n})")
    print(f"{'='*65}")

    print(f"\n对话轮数:")
    print(f"  mean={turn_lengths.mean():.1f}, min={turn_lengths.min()}, "
          f"max={turn_lengths.max()}, median={np.median(turn_lengths):.0f}")

    print(f"\n各分量原始值分布:")
    for name, arr in [("baseline", baselines), ("trend_r", trends), ("vol_p", vols)]:
        print(f"  {name:10s}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
              f"min={arr.min():.4f}, max={arr.max():.4f}")

    b_std = max(baselines.std(), 1e-8)
    t_std = max(trends.std(), 1e-8)
    v_std = max(vols.std(), 1e-8)

    # ================================================================
    # 方法 1：等方差贡献（数据驱动，无预设比例）
    # w1*std(b) = w2*std(t) = w3*std(v)  =>  wi ∝ 1/std(i)
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  推荐方案 1：等方差贡献（每个分量对 reward 方差贡献相等）")
    print(f"{'='*65}")
    print(f"  原理: w_i ∝ 1/std(component_i)，确保三个信号强度相同")

    raw_w1_ev = 1.0 / b_std
    raw_w2_ev = 1.0 / t_std
    raw_w3_ev = 1.0 / v_std
    w1_ev, w2_ev, w3_ev = _scale_weights(raw_w1_ev, raw_w2_ev, raw_w3_ev)
    _print_weight_eval("等方差", w1_ev, w2_ev, w3_ev, baselines, trends, vols)

    # ================================================================
    # 方法 2：baseline 主导（baseline 方差贡献 = trend + vol 之和）
    # w1*std(b) = 2 * w2*std(t) = 2 * w3*std(v)
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  推荐方案 2：baseline 主导（baseline 方差贡献 = 其余两项之和）")
    print(f"{'='*65}")
    print(f"  原理: baseline 占 50% 方差, trend 占 25%, vol 占 25%")

    raw_w1_bd = 2.0 / b_std
    raw_w2_bd = 1.0 / t_std
    raw_w3_bd = 1.0 / v_std
    w1_bd, w2_bd, w3_bd = _scale_weights(raw_w1_bd, raw_w2_bd, raw_w3_bd)
    _print_weight_eval("baseline主导", w1_bd, w2_bd, w3_bd, baselines, trends, vols)

    # ================================================================
    # 方法 3：用户自定义比例
    # ================================================================
    if user_ratio:
        parts = user_ratio.split(":")
        if len(parts) == 3:
            try:
                r1, r2, r3 = float(parts[0]), float(parts[1]), float(parts[2])
                print(f"\n{'='*65}")
                print(f"  用户指定比例 {user_ratio}（按 mean 贡献比分配）")
                print(f"{'='*65}")
                b_mean = max(baselines.mean(), 1e-8)
                t_mean = max(trends.mean(), 1e-8)
                v_mean = max(vols.mean(), 1e-8)
                raw_w1_u = r1 / b_mean
                raw_w2_u = r2 / t_mean
                raw_w3_u = r3 / v_mean
                w1_u, w2_u, w3_u = _scale_weights(raw_w1_u, raw_w2_u, raw_w3_u)
                _print_weight_eval(f"ratio={user_ratio}", w1_u, w2_u, w3_u,
                                   baselines, trends, vols)
            except ValueError:
                print(f"  无法解析比例: {user_ratio}，格式应为 a:b:c（如 7:2:1）")

    # ================================================================
    # 对比：当前默认权重
    # ================================================================
    print(f"\n{'='*65}")
    print(f"  对比：当前默认权重")
    print(f"{'='*65}")
    _print_weight_eval("当前默认", 1.0, 0.3, 0.2, baselines, trends, vols)

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze reward components from PPO training log",
    )
    parser.add_argument("--log", type=str, required=True,
                        help="Path to PPO stdout log file")
    parser.add_argument("--trend-n", type=int, default=5,
                        help="Window size for trend/volatility (default: 5)")
    parser.add_argument("--ratio", type=str, default=None,
                        help="Custom contribution ratio, e.g. '7:2:1' (baseline:trend:vol)")
    args = parser.parse_args()

    all_turns = parse_turns_from_log(args.log)
    if not all_turns:
        print(f"未找到 turns 数据，请确认日志中包含 'turns=[[...]]' 格式的输出。")
        return

    print(f"从 {args.log} 中解析到 {len(all_turns)} 条 rollout 的 turns 数据。")
    analyze(all_turns, trend_n=args.trend_n, user_ratio=args.ratio)


if __name__ == "__main__":
    main()
