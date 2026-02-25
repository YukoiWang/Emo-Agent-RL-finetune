# -*- coding: utf-8 -*-
"""
用 Optuna 自动搜索 mode2 的 reward 权重 w1、w2、w3（及 trend_n），在验证轨迹上最大化目标指标。
说明：w1/w2/w3 是超参数，不用 Adam 更新；Adam 只用于 PPO 的 policy/critic 参数。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np


def load_val_rollouts(path: str):
    """
    加载验证集 rollout 结果。每行一个 JSON：{"emo_point": float, "emo_point_turns": [float, ...]}。
    可由 run_ppo_emo 或 collect_rollouts_emo 跑完验证集后把 non_tensor_batch 按条写出。
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append({
                "emo_point": float(obj.get("emo_point", 0)),
                "emo_point_turns": list(obj.get("emo_point_turns", [])),
            })
    return out


def compute_r_total(emo_point: float, emo_point_turns: list, w1: float, w2: float, w3: float, trend_n: int) -> float:
    """单条样本的 mode2 reward 标量（与 reward_emo 中逻辑一致）。"""
    baseline = max(0.0, emo_point / 100.0)
    if len(emo_point_turns) < 2 or trend_n < 1:
        return baseline
    recent = emo_point_turns[-trend_n:] if len(emo_point_turns) >= trend_n else emo_point_turns
    x = np.arange(len(recent), dtype=np.float32)
    y = np.array(recent, dtype=np.float32)
    # trend
    if np.std(y) >= 1e-6:
        slope = np.polyfit(x, y, 1)[0]
        trend_r = float(np.clip(slope * 0.1, 0.0, 1.0))
    else:
        trend_r = 0.0
    # volatility
    variance = np.var(y)
    vol_p = float(min(1.0, variance / 400.0))
    r = w1 * baseline + w2 * trend_r - w3 * vol_p
    return max(0.0, min(1.0, r))


def objective_one(trial, val_rollouts, metric: str):
    """
    Optuna 单次 trial：采样 w1, w2, w3（及 trend_n），在 val_rollouts 上算目标指标。
    metric: "mean_reward" 最大化平均 r_total；"mean_emo" 最大化平均最终 emo_point（此时权重影响不大，可只做 sanity check）。
    """
    w1 = trial.suggest_float("w1", 0.2, 1.5)
    w2 = trial.suggest_float("w2", 0.0, 0.8)
    w3 = trial.suggest_float("w3", 0.0, 0.5)
    trend_n = trial.suggest_int("trend_n", 3, 10)

    rewards = []
    emos = []
    for item in val_rollouts:
        r = compute_r_total(
            item["emo_point"],
            item["emo_point_turns"],
            w1=w1, w2=w2, w3=w3, trend_n=trend_n,
        )
        rewards.append(r)
        emos.append(item["emo_point"])

    if metric == "mean_reward":
        return np.mean(rewards)
    if metric == "mean_emo":
        return np.mean(emos) / 100.0
    # 组合：例如 0.7 * mean_reward + 0.3 * (mean_emo)
    return 0.7 * np.mean(rewards) + 0.3 * (np.mean(emos) / 100.0)


def main():
    parser = argparse.ArgumentParser(description="Optuna 搜索 mode2 reward 权重 w1, w2, w3")
    parser.add_argument("--val_rollouts", type=str, default="", help="验证集 rollout 结果 jsonl，每行 {emo_point, emo_point_turns}")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--metric", type=str, default="mean_reward", choices=["mean_reward", "mean_emo", "combo"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.val_rollouts or not os.path.isfile(args.val_rollouts):
        # 生成少量 dummy 数据便于本地试跑
        print("未提供 --val_rollouts，使用示例数据（建议用真实验证 rollout 结果）")
        val_rollouts = [
            {"emo_point": 65, "emo_point_turns": [50, 52, 55, 58, 60, 65]},
            {"emo_point": 45, "emo_point_turns": [50, 48, 45, 46, 44, 45]},
            {"emo_point": 72, "emo_point_turns": [50, 55, 60, 65, 70, 72]},
            {"emo_point": 58, "emo_point_turns": [50, 60, 52, 62, 55, 58]},
        ] * 5
    else:
        val_rollouts = load_val_rollouts(args.val_rollouts)
    print(f"验证样本数: {len(val_rollouts)}")

    try:
        import optuna
    except ImportError:
        print("请安装: pip install optuna")
        sys.exit(1)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10))
    study.optimize(
        lambda t: objective_one(t, val_rollouts, args.metric),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\n最优权重 (best params):", study.best_params)
    print("最优目标值 (best value):", study.best_value)
    print("建议: 将上述 w1, w2, w3, trend_n 填入 run_ppo_emo.py 或 collect_rollouts_emo 的 mode2 参数。")


if __name__ == "__main__":
    main()
