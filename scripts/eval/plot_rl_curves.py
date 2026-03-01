#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot RL training curves from training_log.jsonl files.

Supports both PPO-Emo and GRPO log formats:

  PPO-Emo fields:
    reward_mean, actor/pg_loss, actor/entropy_loss, actor/kl_loss,
    actor/policy_loss, actor/pg_clipfrac, actor/ppo_kl, actor/grad_norm,
    critic/vf_loss, critic/vf_clipfrac, critic/vpred_mean, critic/grad_norm,
    baseline_mean, trend_mean, vol_mean, elapsed

  GRPO fields:
    reward_mean, rewards_max, rewards_min, loss, kl_loss

Usage:
  python scripts/eval/plot_rl_curves.py --log-dir outputs/
  python scripts/eval/plot_rl_curves.py --log-dir outputs/ppo_emo --smooth 10
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

# (title, candidate_keys, ylabel)
# candidate_keys: the first matching key in a record is used.
METRIC_GROUPS: List[Tuple[str, List[str], str]] = [
    # ── shared ──
    ("Reward (mean)",       ["reward_mean"],                                        "Reward"),
    ("Reward Range",        ["rewards_max", "rewards_min"],                         "Reward Range"),
    # ── loss (covers both PPO-Emo and GRPO) ──
    ("Policy Loss",         ["actor/policy_loss", "actor/pg_loss", "loss"],          "Loss"),
    ("Value Loss",          ["critic/vf_loss"],                                     "Loss"),
    # ── KL ──
    ("KL Divergence",       ["actor/kl_loss", "kl_loss"],                           "KL"),
    ("PPO Approx KL",       ["actor/ppo_kl"],                                       "KL"),
    # ── PPO-Emo specific ──
    ("Entropy",             ["actor/entropy_loss"],                                 "Entropy"),
    ("PG Clip Fraction",    ["actor/pg_clipfrac"],                                  "Fraction"),
    ("VF Clip Fraction",    ["critic/vf_clipfrac"],                                 "Fraction"),
    ("Critic V-pred",       ["critic/vpred_mean"],                                  "Value"),
    # ── reward components (PPO-Emo mode2/3) ──
    ("Reward: Baseline",    ["baseline_mean"],                                      "Score"),
    ("Reward: Trend",       ["trend_mean"],                                         "Score"),
    ("Reward: Volatility",  ["vol_mean"],                                           "Score"),
    # ── grad norms ──
    ("Actor Grad Norm",     ["actor/grad_norm"],                                    "Norm"),
    ("Critic Grad Norm",    ["critic/grad_norm"],                                   "Norm"),
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


def detect_trainer_type(records: List[Dict]) -> str:
    """Guess 'ppo' or 'grpo' from logged keys."""
    if not records:
        return "unknown"
    sample = records[min(5, len(records) - 1)]
    if "actor/pg_loss" in sample or "actor/policy_loss" in sample:
        return "ppo"
    if "loss" in sample and "rewards_max" in sample:
        return "grpo"
    return "unknown"


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
            ax.plot(steps, vals, color=color, alpha=0.2, linewidth=0.5)
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


def plot_reward_components(
    ax: plt.Axes,
    experiments: Dict[str, List[Dict]],
    smooth: float = 0,
):
    """Overlay baseline / trend / volatility on a single axis."""
    component_keys = [
        ("baseline_mean", "Baseline", "#2196F3"),
        ("trend_mean", "Trend", "#4CAF50"),
        ("vol_mean", "Volatility", "#FF5722"),
    ]
    plotted = False
    for label, records in experiments.items():
        for key, comp_label, color in component_keys:
            steps, vals = extract(records, [key])
            if len(steps) == 0:
                continue
            if smooth > 0:
                ax.plot(steps, vals, color=color, alpha=0.15, linewidth=0.4)
                vals = ema_smooth(vals, smooth)
            suffix = f" ({label})" if len(experiments) > 1 else ""
            ax.plot(steps, vals, label=f"{comp_label}{suffix}",
                    color=color, linewidth=1.5)
            plotted = True

    if plotted:
        ax.set_xlabel("Step")
        ax.set_ylabel("Component Score")
        ax.legend(loc="best")
    return plotted


def has_any_key(records: List[Dict], keys: List[str]) -> bool:
    """Check if at least one record contains any of the keys."""
    for r in records[:20]:
        for k in keys:
            if k in r:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Plot RL training curves")
    parser.add_argument("--log-dir", type=str, default="outputs",
                        help="Root dir (auto-discovers sub-dirs) or single experiment dir")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir for figures (default: <log-dir>/plots)")
    parser.add_argument("--smooth", type=float, default=10,
                        help="EMA smoothing span (0=no smoothing)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = Path(args.out_dir) if args.out_dir else log_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = find_experiments(log_dir)
    if not experiments:
        print(f"No training_log.jsonl found under {log_dir}")
        return

    print(f"Found {len(experiments)} experiment(s): {list(experiments.keys())}")
    for label, records in experiments.items():
        ttype = detect_trainer_type(records)
        print(f"  {label}: {len(records)} steps, type={ttype}")

    colors = [f"C{i}" for i in range(10)]

    # Gather all keys present across experiments to decide what to plot
    all_records = []
    for records in experiments.values():
        all_records.extend(records[:20])

    # ── Individual metric plots ──
    saved_count = 0
    for title, keys, ylabel in METRIC_GROUPS:
        if not has_any_key(all_records, keys):
            continue

        fig, ax = plt.subplots(figsize=(9, 4.5))

        if "Range" in title:
            plotted = plot_reward_range(ax, experiments, args.smooth, colors)
        else:
            plotted = plot_single_metric(ax, experiments, keys, ylabel, args.smooth, colors)

        if plotted:
            ax.set_title(title)
            fname = title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("-", "")
            path = out_dir / f"{fname}.{args.format}"
            fig.savefig(path, dpi=args.dpi)
            print(f"  Saved {path}")
            saved_count += 1
        plt.close(fig)

    # ── Reward components combined plot (PPO-Emo) ──
    if has_any_key(all_records, ["baseline_mean"]):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plotted = plot_reward_components(ax, experiments, args.smooth)
        if plotted:
            ax.set_title("Reward Components (Baseline / Trend / Volatility)")
            path = out_dir / f"reward_components.{args.format}"
            fig.savefig(path, dpi=args.dpi)
            print(f"  Saved {path}")
            saved_count += 1
        plt.close(fig)

    # ── Combined multi-panel overview ──
    # Adaptive: pick the metrics that actually exist
    overview_candidates = [
        ("Reward",          ["reward_mean"],                               "Reward"),
        ("Policy Loss",     ["actor/policy_loss", "actor/pg_loss", "loss"],"Loss"),
        ("KL",              ["actor/kl_loss", "kl_loss"],                  "KL"),
        ("Value Loss",      ["critic/vf_loss"],                            "Loss"),
        ("Entropy",         ["actor/entropy_loss"],                        "Entropy"),
        ("Reward Components", ["baseline_mean"],                           ""),
    ]
    core = [(t, k, y) for t, k, y in overview_candidates if has_any_key(all_records, k)]
    if core:
        n_panels = min(len(core), 6)
        core = core[:n_panels]
        cols = min(n_panels, 3)
        rows = (n_panels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))
        if n_panels == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for idx, (title, keys, ylabel) in enumerate(core):
            ax = axes[idx]
            if title == "Reward Components":
                plot_reward_components(ax, experiments, args.smooth)
            else:
                plot_single_metric(ax, experiments, keys, ylabel, args.smooth, colors)
            ax.set_title(title)

        for idx in range(len(core), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle("RL Training Overview", fontsize=14, y=1.02)
        fig.tight_layout()
        combined_path = out_dir / f"training_overview.{args.format}"
        fig.savefig(combined_path, dpi=args.dpi)
        plt.close(fig)
        print(f"  Saved {combined_path}")
        saved_count += 1

    print(f"\n{saved_count} figures saved to {out_dir.absolute()}")


if __name__ == "__main__":
    main()
