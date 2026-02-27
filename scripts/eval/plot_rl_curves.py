#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot RL training curves from training_log.jsonl files.

Supports:
  - Auto-discovery: any sub-directory containing training_log.jsonl is treated
    as an experiment.
  - Single-experiment mode: point --log-dir directly at a dir with the jsonl.
  - Metrics: reward, loss, kl, policy_loss, value_loss, rewards_max/min.
  - Exponential smoothing (--smooth).
  - High-DPI export (--dpi, default 300).
  - Combined multi-panel figure + individual per-metric PNGs.

Usage:
  # Multi-experiment comparison (auto-discovers sub-dirs)
  python scripts/eval/plot_rl_curves.py --log-dir outputs/

  # Single experiment
  python scripts/eval/plot_rl_curves.py --log-dir outputs/grpo_emo

  # Custom output dir & smoothing
  python scripts/eval/plot_rl_curves.py --log-dir outputs/ --out-dir figures/ --smooth 10 --dpi 300
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

METRIC_GROUPS: List[Tuple[str, List[str], str]] = [
    ("Reward (mean)", ["reward_mean"], "Reward"),
    ("Loss", ["loss", "total_loss"], "Loss"),
    ("KL Divergence", ["kl_loss", "objective/kl", "policy/approxkl_avg"], "KL"),
    ("Policy Loss", ["policy_loss"], "Policy Loss"),
    ("Value Loss", ["value_loss"], "Value Loss"),
    ("Reward Range", ["rewards_max", "rewards_min"], "Reward Range"),
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_experiments(log_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Return {label: records} for every training_log.jsonl found."""
    experiments = {}

    direct = log_dir / "training_log.jsonl"
    if direct.exists():
        experiments[log_dir.name] = load_jsonl(direct)
        return experiments

    for jsonl in sorted(log_dir.rglob("training_log.jsonl")):
        rel = jsonl.parent.relative_to(log_dir)
        label = str(rel).replace("/", " / ") if str(rel) != "." else log_dir.name
        experiments[label] = load_jsonl(jsonl)

    return experiments


def extract(records: List[Dict], key_candidates: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    steps, vals = [], []
    for r in records:
        s = r.get("step")
        if s is None:
            continue
        v = None
        for k in key_candidates:
            if k in r:
                v = float(r[k])
                break
        if v is not None:
            steps.append(int(s))
            vals.append(v)
    return np.array(steps), np.array(vals)


def ema_smooth(values: np.ndarray, span: float) -> np.ndarray:
    if span <= 0 or len(values) < 2:
        return values
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def plot_single_metric(
    ax: plt.Axes,
    experiments: Dict[str, List[Dict]],
    key_candidates: List[str],
    ylabel: str,
    smooth: float = 0,
    colors: Optional[List[str]] = None,
):
    plotted = False
    for i, (label, records) in enumerate(experiments.items()):
        steps, vals = extract(records, key_candidates)
        if len(steps) == 0:
            continue
        color = colors[i % len(colors)] if colors else f"C{i}"
        if smooth > 0:
            raw_alpha = 0.2
            ax.plot(steps, vals, color=color, alpha=raw_alpha, linewidth=0.5)
            vals = ema_smooth(vals, smooth)
        ax.plot(steps, vals, label=label, color=color, linewidth=1.5)
        plotted = True

    if plotted:
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
    return plotted


def plot_reward_range(
    ax: plt.Axes,
    experiments: Dict[str, List[Dict]],
    smooth: float = 0,
    colors: Optional[List[str]] = None,
):
    plotted = False
    for i, (label, records) in enumerate(experiments.items()):
        steps_mean, vals_mean = extract(records, ["reward_mean"])
        steps_max, vals_max = extract(records, ["rewards_max"])
        steps_min, vals_min = extract(records, ["rewards_min"])

        if len(steps_mean) == 0 or len(steps_max) == 0 or len(steps_min) == 0:
            continue

        color = colors[i % len(colors)] if colors else f"C{i}"
        n = min(len(steps_mean), len(vals_max), len(vals_min))
        s, m, mx, mn = steps_mean[:n], vals_mean[:n], vals_max[:n], vals_min[:n]

        if smooth > 0:
            m = ema_smooth(m, smooth)
            mx = ema_smooth(mx, smooth)
            mn = ema_smooth(mn, smooth)

        ax.fill_between(s, mn, mx, alpha=0.15, color=color)
        ax.plot(s, m, label=f"{label} (mean)", color=color, linewidth=1.5)
        ax.plot(s, mx, color=color, linewidth=0.6, linestyle="--", alpha=0.5)
        ax.plot(s, mn, color=color, linewidth=0.6, linestyle="--", alpha=0.5)
        plotted = True

    if plotted:
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.legend(loc="best")
    return plotted


def main():
    parser = argparse.ArgumentParser(description="Plot RL training curves (high-quality)")
    parser.add_argument("--log-dir", type=str, default="outputs",
                        help="Root dir (auto-discovers sub-dirs) or single experiment dir")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir for figures (default: <log-dir>/plots)")
    parser.add_argument("--smooth", type=float, default=10,
                        help="EMA smoothing span (0=no smoothing)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI for saved figures")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"],
                        help="Output figure format")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = find_experiments(log_dir)
    if not experiments:
        print(f"No training_log.jsonl found under {log_dir}")
        return

    print(f"Found {len(experiments)} experiment(s): {list(experiments.keys())}")

    colors = [f"C{i}" for i in range(10)]

    # --- Individual metric plots ---
    for title, keys, ylabel in METRIC_GROUPS:
        fig, ax = plt.subplots(figsize=(9, 4.5))

        if "Range" in title:
            plotted = plot_reward_range(ax, experiments, args.smooth, colors)
        else:
            plotted = plot_single_metric(ax, experiments, keys, ylabel, args.smooth, colors)

        if plotted:
            ax.set_title(title)
            fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "")
            path = out_dir / f"{fname}.{args.format}"
            fig.savefig(path, dpi=args.dpi)
            print(f"  Saved {path}")
        plt.close(fig)

    # --- Combined multi-panel figure ---
    core_metrics = [
        ("Reward (mean)", ["reward_mean"], "Reward"),
        ("Loss", ["loss", "total_loss"], "Loss"),
        ("KL Divergence", ["kl_loss", "objective/kl"], "KL"),
    ]
    fig, axes = plt.subplots(1, len(core_metrics), figsize=(6 * len(core_metrics), 4.5))
    if len(core_metrics) == 1:
        axes = [axes]
    for ax, (title, keys, ylabel) in zip(axes, core_metrics):
        plot_single_metric(ax, experiments, keys, ylabel, args.smooth, colors)
        ax.set_title(title)

    fig.suptitle("RL Training Overview", fontsize=14, y=1.02)
    fig.tight_layout()
    combined_path = out_dir / f"training_overview.{args.format}"
    fig.savefig(combined_path, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved {combined_path}")

    print(f"\nAll figures saved to {out_dir.absolute()}")


if __name__ == "__main__":
    main()
