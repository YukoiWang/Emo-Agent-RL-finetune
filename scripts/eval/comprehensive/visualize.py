# -*- coding: utf-8 -*-
"""Visualization for the 5-level evaluation framework."""
from __future__ import annotations

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

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800",
           "#00BCD4", "#E91E63", "#8BC34A", "#3F51B5", "#CDDC39"]


def _save(fig, out_dir, name, fmt="png", dpi=300):
    path = Path(out_dir) / f"{name}.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print(f"  Saved {path}")


def _ema(vals, span=5.0):
    if span <= 0 or len(vals) < 2:
        return vals
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(vals)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return out


# ====================================================================
# Level 1: Emotion curves
# ====================================================================

def plot_emotion_curves(
    checkpoint_results: List[Dict[str, Any]],
    out_dir: str,
    smooth: float = 5.0,
    fmt: str = "png",
    dpi: int = 300,
):
    """Plot Emotion Score and Avg EmoChange over training steps."""
    steps = np.array([r["step"] for r in checkpoint_results])
    scores = np.array([r["emotion_score"] for r in checkpoint_results])
    changes = np.array([r["avg_emo_change"] for r in checkpoint_results])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, scores, alpha=0.25, color=COLORS[0], linewidth=0.8)
    ax1.plot(steps, _ema(scores, smooth), color=COLORS[0], linewidth=2,
             label="Emotion Score")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Emotion Score (0-100)")
    ax1.set_title("Level 1: Emotion Score")
    ax1.legend()

    ax2.plot(steps, changes, alpha=0.25, color=COLORS[1], linewidth=0.8)
    ax2.plot(steps, _ema(changes, smooth), color=COLORS[1], linewidth=2,
             label="Avg EmoChange")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Avg EmoChange")
    ax2.set_title("Level 1: Average Emotion Change")
    ax2.legend()

    fig.suptitle("Level 1: Emotion Outcome Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "level1_emotion", fmt, dpi)


def plot_per_turn_emo(
    per_turn_stats: Dict[str, List[float]],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
):
    """Plot average per-turn emotion trajectory."""
    means = np.array(per_turn_stats["mean"])
    stds = np.array(per_turn_stats["std"])
    turns = np.arange(len(means))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(turns, means, color=COLORS[0], linewidth=2, label="Mean Emo")
    ax.fill_between(turns, means - stds, means + stds,
                    alpha=0.2, color=COLORS[0])
    ax.set_xlabel("Turn")
    ax.set_ylabel("Emotion Point")
    ax.set_title("Average Per-Turn Emotion Trajectory")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "level1_per_turn_emo", fmt, dpi)


# ====================================================================
# Level 2: Strategy
# ====================================================================

def plot_strategy_frequency(
    strategy_freq: Dict[str, float],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
):
    """Bar chart of strategy proportions."""
    strategies = [s for s in strategy_freq if strategy_freq[s] > 0]
    values = [strategy_freq[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(strategies, values, color=COLORS[:len(strategies)])
    ax.set_ylabel("Proportion")
    ax.set_title("Level 2: Strategy Frequency Distribution")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "level2_strategy_freq", fmt, dpi)


def plot_strategy_contribution(
    contribution: Dict[str, float],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
):
    """Bar chart: conditional mean EmoChange per strategy."""
    strategies = [s for s in contribution if s != "Unknown"]
    values = [contribution[s] for s in strategies]
    colors_map = [COLORS[1] if v < 0 else COLORS[2] for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(strategies, values, color=colors_map)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Avg EmoChange")
    ax.set_title("Level 2: Strategy Contribution to Emotion Change")
    fig.tight_layout()
    _save(fig, out_dir, "level2_strategy_contribution", fmt, dpi)


# ====================================================================
# Level 3: Capability
# ====================================================================

def plot_scc_coordinate(
    scc_data: Dict[str, Any],
    out_dir: str,
    label: str = "Model",
    fmt: str = "png",
    dpi: int = 300,
):
    """Scatter the SCC (x, y) coordinate."""
    x, y = scc_data["coordinate"]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.axvline(x=0, color="gray", linewidth=0.8)
    ax.scatter([x], [y], s=200, c=COLORS[0], zorder=5, edgecolors="black")
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(10, 10),
                fontsize=12, fontweight="bold")
    ax.set_xlabel("Solution  <-->  Empathy")
    ax.set_ylabel("Structured  <-->  Creative")
    ax.set_title("Level 3: Social Cognition Coordinate")
    fig.tight_layout()
    _save(fig, out_dir, "level3_scc", fmt, dpi)


def plot_scc_trajectory(
    scc_points: List[Tuple[int, float, float]],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
):
    """Plot SCC trajectory across checkpoints: list of (step, x, y)."""
    if not scc_points:
        return
    steps, xs, ys = zip(*scc_points)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axhline(y=0, color="gray", linewidth=0.8)
    ax.axvline(x=0, color="gray", linewidth=0.8)
    ax.plot(xs, ys, "o-", color=COLORS[0], markersize=6, alpha=0.7)
    ax.scatter([xs[0]], [ys[0]], s=150, c=COLORS[2], zorder=5,
               label=f"Start (step {steps[0]})", edgecolors="black")
    ax.scatter([xs[-1]], [ys[-1]], s=150, c=COLORS[1], zorder=5,
               label=f"End (step {steps[-1]})", edgecolors="black")
    ax.set_xlabel("Solution  <-->  Empathy")
    ax.set_ylabel("Structured  <-->  Creative")
    ax.set_title("Level 3: SCC Trajectory")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "level3_scc_trajectory", fmt, dpi)


def plot_capability_radar(
    capabilities: Dict[str, float],
    out_dir: str,
    label: str = "Model",
    fmt: str = "png",
    dpi: int = 300,
):
    """Radar / spider chart for 5 core capabilities."""
    keys = list(capabilities.keys())
    values = [capabilities[k] for k in keys]
    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]
    labels = [k.replace("_", " ").title() for k in keys]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, values_closed, "o-", linewidth=2, color=COLORS[0])
    ax.fill(angles_closed, values_closed, alpha=0.2, color=COLORS[0])
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_ylim(0, 5)
    ax.set_title(f"Level 3: Core Capability ({label})", pad=20)
    fig.tight_layout()
    _save(fig, out_dir, "level3_capability_radar", fmt, dpi)


# ====================================================================
# Level 4: Stability
# ====================================================================

def plot_stability(
    stability: Dict[str, Any],
    out_dir: str,
    smooth: float = 5.0,
    fmt: str = "png",
    dpi: int = 300,
):
    """Plot learning curves with stability annotations."""
    steps = np.array(stability["steps"])
    es = np.array(stability["emotion_scores"])
    ec = np.array(stability["emo_changes"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(steps, es, alpha=0.3, color=COLORS[0])
    ax1.plot(steps, _ema(es, smooth), color=COLORS[0], linewidth=2)
    ax1.axhline(y=stability["best_emotion_score"], color=COLORS[2],
                linestyle=":", alpha=0.6, label=f"Best={stability['best_emotion_score']:.1f}")
    ax1.set_ylabel("Emotion Score")
    ax1.set_title("Level 4: Learning Curve Stability")
    ax1.legend()

    info_parts = [
        f"Var={stability['score_variance']:.2f}",
        f"Std={stability['score_std']:.2f}",
        f"Mono={stability['monotonicity']:.2f}",
    ]
    if stability["oscillation_detected"]:
        info_parts.append("OSCILLATION")
    if stability["collapse_detected"]:
        info_parts.append("COLLAPSE")
    ax1.text(0.02, 0.95, " | ".join(info_parts), transform=ax1.transAxes,
             fontsize=9, va="top",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax2.plot(steps, ec, alpha=0.3, color=COLORS[1])
    ax2.plot(steps, _ema(ec, smooth), color=COLORS[1], linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Avg EmoChange")

    fig.tight_layout()
    _save(fig, out_dir, "level4_stability", fmt, dpi)


# ====================================================================
# Level 5: Environment comparison
# ====================================================================

def plot_environment_comparison(
    comparison: Dict[str, Any],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
):
    """Grouped bar chart comparing metrics across environments."""
    envs = comparison.get("environments", [])
    outcome = comparison.get("outcome_comparison", {})
    if not envs or not outcome:
        return

    metrics = list(outcome.keys())
    n_envs = len(envs)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, env in enumerate(envs):
        vals = [outcome[m].get(env, 0) or 0 for m in metrics]
        offset = (i - n_envs / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=env, color=COLORS[i])

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_title("Level 5: Environment Comparison")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "level5_env_comparison", fmt, dpi)


def plot_simulator_quality(
    sim_quality: Dict[str, Dict[str, float]],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
):
    """Grouped bar chart of simulator quality scores."""
    if not sim_quality:
        return

    envs = list(sim_quality.keys())
    sample_keys = list(next(iter(sim_quality.values())).keys())
    n_envs = len(envs)
    x = np.arange(len(sample_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, env in enumerate(envs):
        vals = [sim_quality[env].get(k, 0) for k in sample_keys]
        offset = (i - n_envs / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=env, color=COLORS[i])

    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", " ").title() for k in sample_keys],
                       rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Level 5: Simulator Quality by Environment")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir, "level5_simulator_quality", fmt, dpi)


# ====================================================================
# Combined overview
# ====================================================================

def plot_overview_dashboard(
    final_result: Dict[str, Any],
    out_dir: str,
    fmt: str = "png",
    dpi: int = 300,
):
    """Single-page dashboard summarising key metrics."""
    fig = plt.figure(figsize=(16, 10))

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Emotion Score gauge
    ax1 = fig.add_subplot(gs[0, 0])
    score = final_result.get("emotion_score", 0)
    change = final_result.get("avg_emo_change", 0)
    ax1.barh(["Emotion Score", "Avg EmoChange"], [score, change],
             color=[COLORS[0], COLORS[1]])
    ax1.set_xlim(-20, 100)
    ax1.set_title("Level 1: Emotion Outcome")

    # Panel 2: Strategy pie
    ax2 = fig.add_subplot(gs[0, 1])
    freq = final_result.get("strategy_frequency", {})
    if freq:
        labels = [k for k, v in freq.items() if v > 0]
        sizes = [freq[k] for k in labels]
        ax2.pie(sizes, labels=labels, autopct="%1.0f%%",
                colors=COLORS[:len(labels)])
    ax2.set_title("Level 2: Strategy Distribution")

    # Panel 3: Capability radar
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    caps = final_result.get("core_capabilities", {})
    if caps:
        keys = list(caps.keys())
        vals = [caps[k] for k in keys]
        n = len(keys)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        vals_c = vals + [vals[0]]
        angles_c = angles + [angles[0]]
        ax3.plot(angles_c, vals_c, "o-", color=COLORS[0])
        ax3.fill(angles_c, vals_c, alpha=0.2, color=COLORS[0])
        ax3.set_thetagrids(np.degrees(angles),
                           [k.replace("_", " ").title() for k in keys])
        ax3.set_ylim(0, 5)
    ax3.set_title("Level 3: Core Capability", pad=20)

    # Panel 4: SCC
    ax4 = fig.add_subplot(gs[1, 0])
    scc = final_result.get("scc", {})
    if scc and "coordinate" in scc:
        sx, sy = scc["coordinate"]
        ax4.set_xlim(-1.2, 1.2)
        ax4.set_ylim(-1.2, 1.2)
        ax4.axhline(y=0, color="gray", linewidth=0.6)
        ax4.axvline(x=0, color="gray", linewidth=0.6)
        ax4.scatter([sx], [sy], s=200, c=COLORS[0], zorder=5,
                    edgecolors="black")
        ax4.set_xlabel("Solution <-> Empathy")
        ax4.set_ylabel("Structured <-> Creative")
    ax4.set_title("Level 3: SCC")

    # Panel 5-6: summary text
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    summary_lines = [
        f"Emotion Score:  {final_result.get('emotion_score', 'N/A'):.1f}",
        f"Avg EmoChange:  {final_result.get('avg_emo_change', 'N/A'):.1f}",
        f"N Dialogues:    {final_result.get('n_dialogues', 'N/A')}",
    ]
    if caps:
        avg_cap = sum(caps.values()) / len(caps) if caps else 0
        summary_lines.append(f"Avg Capability: {avg_cap:.2f} / 5.0")
    if scc and "coordinate" in scc:
        summary_lines.append(f"SCC: ({scc['coordinate'][0]:.2f}, "
                             f"{scc['coordinate'][1]:.2f})")

    ax5.text(0.1, 0.8, "\n".join(summary_lines), transform=ax5.transAxes,
             fontsize=13, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Comprehensive Evaluation Dashboard", fontsize=16, y=1.01)
    _save(fig, out_dir, "overview_dashboard", fmt, dpi)
