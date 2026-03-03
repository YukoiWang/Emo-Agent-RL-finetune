# -*- coding: utf-8 -*-
"""
Level 1 -- Emotion Outcome Metrics.

  1. Emotion Score : average final emo_point across test samples.
  2. Avg EmoChange : average (final - initial) emo_point.

Both rely on the planner scoring that happens during dialogue generation.
"""
from __future__ import annotations

from typing import Dict, List

from .dialogue_generator import DialogueResult


def compute_emotion_score(dialogues: List[DialogueResult]) -> float:
    """Average final emotion point over all dialogues."""
    if not dialogues:
        return 0.0
    return sum(d.final_emo for d in dialogues) / len(dialogues)


def compute_avg_emo_change(dialogues: List[DialogueResult]) -> float:
    """Average (final - initial) emotion change."""
    if not dialogues:
        return 0.0
    return sum(d.emo_change for d in dialogues) / len(dialogues)


def compute_emo_change_distribution(
    dialogues: List[DialogueResult],
) -> Dict[str, float]:
    """Breakdown of emo-change into positive / zero / negative fractions."""
    if not dialogues:
        return {"positive_frac": 0.0, "zero_frac": 0.0, "negative_frac": 0.0}
    n = len(dialogues)
    pos = sum(1 for d in dialogues if d.emo_change > 0)
    neg = sum(1 for d in dialogues if d.emo_change < 0)
    zero = n - pos - neg
    return {
        "positive_frac": pos / n,
        "zero_frac": zero / n,
        "negative_frac": neg / n,
    }


def compute_per_turn_emo_stats(
    dialogues: List[DialogueResult],
) -> Dict[str, List[float]]:
    """Per-turn average emotion across dialogues (aligned by turn index)."""
    if not dialogues:
        return {"mean": [], "std": []}
    max_len = max(len(d.emo_point_trajectory) for d in dialogues)
    means, stds = [], []
    for t in range(max_len):
        vals = [
            d.emo_point_trajectory[t]
            for d in dialogues
            if t < len(d.emo_point_trajectory)
        ]
        if vals:
            m = sum(vals) / len(vals)
            v = sum((x - m) ** 2 for x in vals) / len(vals)
            means.append(m)
            stds.append(v ** 0.5)
    return {"mean": means, "std": stds}


def compute_emo_volatility(dialogues: List[DialogueResult]) -> float:
    """
    对话级情绪波动指标：
    - 对每个对话，计算相邻轮次 emo_point 差分的标准差；
    - 再对所有对话的该值取平均。
    """
    vols = []
    for d in dialogues:
        traj = d.emo_point_trajectory
        if len(traj) < 2:
            continue
        deltas = [traj[i + 1] - traj[i] for i in range(len(traj) - 1)]
        if not deltas:
            continue
        m = sum(deltas) / len(deltas)
        v = sum((x - m) ** 2 for x in deltas) / len(deltas)
        vols.append(v ** 0.5)
    if not vols:
        return 0.0
    return sum(vols) / len(vols)


def evaluate_level1(dialogues: List[DialogueResult]) -> Dict[str, object]:
    """Run all Level-1 metrics."""
    return {
        "emotion_score": compute_emotion_score(dialogues),
        "avg_emo_change": compute_avg_emo_change(dialogues),
        "emo_change_distribution": compute_emo_change_distribution(dialogues),
        "per_turn_emo_stats": compute_per_turn_emo_stats(dialogues),
        "emo_volatility": compute_emo_volatility(dialogues),
        "n_dialogues": len(dialogues),
    }
