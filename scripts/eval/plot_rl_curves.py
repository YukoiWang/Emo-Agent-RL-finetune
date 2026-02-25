#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据 quick_verify 或任意 RL 实验目录下的 training_log.jsonl / trainer_state.json 画 KL、reward、loss 曲线。

用法:
  python scripts/eval/plot_rl_curves.py --log-dir outputs/quick_verify
  python scripts/eval/plot_rl_curves.py --log-dir outputs/quick_verify --out-dir outputs/quick_verify/plots
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_training_log_jsonl(path: Path):
    """加载 training_log.jsonl，返回 list of dict."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_trainer_state(path: Path):
    """加载 HuggingFace trainer_state.json，从 log_history 取 step/loss 等."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("log_history", [])


def extract_series(records, key_step="step", key_reward="reward_mean", key_kl=None, key_loss="loss"):
    """从 records 里取出 step, reward, kl, loss 序列。key_kl 可尝试多个候选."""
    steps = []
    rewards = []
    kls = []
    losses = []
    for r in records:
        s = r.get(key_step)
        if s is None:
            continue
        steps.append(int(s))
        if key_reward and key_reward in r:
            rewards.append(float(r[key_reward]))
        else:
            rewards.append(np.nan)
        # KL: 兼容 objective/kl, kl_loss 等
        kl_val = None
        if key_kl and key_kl in r:
            kl_val = float(r[key_kl])
        else:
            for k in ("objective/kl", "kl_loss", "policy/approxkl_avg"):
                if k in r:
                    kl_val = float(r[k])
                    break
        kls.append(kl_val if kl_val is not None else np.nan)
        if key_loss in r:
            losses.append(float(r[key_loss]))
        else:
            losses.append(np.nan)
    return steps, rewards, kls, losses


def plot_curves(ax, steps, values, label, color=None):
    """在 ax 上画一条曲线，支持 nan."""
    steps = np.array(steps)
    values = np.array(values, dtype=float)
    mask = ~np.isnan(values)
    if not np.any(mask):
        return
    x, y = steps[mask], values[mask]
    ax.plot(x, y, label=label, color=color)


def main():
    parser = argparse.ArgumentParser(description="画 KL / reward / loss 曲线")
    parser.add_argument("--log-dir", type=str, default="outputs/quick_verify",
                        help="实验根目录，包含 ppo_mode1, ppo_mode2, ppo_mode3, grpo 等子目录")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="图片输出目录，默认等于 log-dir")
    parser.add_argument("--smooth", type=float, default=0,
                        help="指数滑动平均窗口 (0=不平滑)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir or args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 子目录名 -> 显示名
    subdirs = [
        ("ppo_mode1", "PPO (mode1)"),
        ("ppo_mode2", "PPO (mode2)"),
        ("ppo_mode3", "PPO (mode3)"),
        ("grpo", "GRPO"),
    ]
    colors = ["C0", "C1", "C2", "C3"]

    all_reward = {}
    all_kl = {}
    all_loss = {}

    for (sub, label), color in zip(subdirs, colors):
        d = log_dir / sub
        if not d.is_dir():
            continue
        # 优先 training_log.jsonl
        jsonl_path = d / "training_log.jsonl"
        state_path = d / "trainer_state.json"
        records = []
        if jsonl_path.exists():
            records = load_training_log_jsonl(jsonl_path)
            steps, rewards, kls, losses = extract_series(
                records,
                key_reward="reward_mean",
                key_kl=None,  # 自动尝试 objective/kl, kl_loss
                key_loss="loss",
            )
        elif state_path.exists():
            raw = load_trainer_state(state_path)
            # 只保留训练步（含 "loss" 的条目），不要 eval 步
            records = [r for r in raw if "loss" in r]
            steps, rewards, kls, losses = extract_series(
                records,
                key_step="step",
                key_reward=None,
                key_kl=None,
                key_loss="loss",
            )
            rewards = [np.nan] * len(steps) if not steps else rewards
            kls = [np.nan] * len(steps) if not steps else kls
        else:
            continue
        if not steps:
            continue
        all_reward[label] = (steps, rewards, color)
        all_kl[label] = (steps, kls, color)
        all_loss[label] = (steps, losses, color)

    def smooth_curve(x, y):
        if args.smooth <= 0 or len(y) < 2:
            return y
        out = []
        alpha = 1.0 / (1.0 + args.smooth)
        s = y[0]
        for v in y:
            if np.isnan(v):
                out.append(np.nan)
            else:
                s = alpha * s + (1 - alpha) * v
                out.append(s)
        return out

    # ---- Reward ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for label, (steps, rewards, color) in all_reward.items():
        rewards = smooth_curve(steps, rewards)
        if not all(np.isnan(rewards)):
            plot_curves(ax, steps, rewards, label, color)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward (mean)")
    ax.set_title("Reward")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'reward_curves.png'}")

    # ---- KL ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for label, (steps, kls, color) in all_kl.items():
        kls = smooth_curve(steps, kls)
        if not all(np.isnan(kls)):
            plot_curves(ax, steps, kls, label, color)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL divergence")
    ax.set_title("KL Divergence")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "kl_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'kl_curves.png'}")

    # ---- Loss ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for label, (steps, losses, color) in all_loss.items():
        losses = smooth_curve(steps, losses)
        if not all(np.isnan(losses)):
            plot_curves(ax, steps, losses, label, color)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'loss_curves.png'}")

    print("Done. Figures saved under", out_dir.absolute())


if __name__ == "__main__":
    main()
