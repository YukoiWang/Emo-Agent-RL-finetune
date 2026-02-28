# -*- coding: utf-8 -*-
"""Level 4: Training Dynamics / Learning Curve Stability."""
from __future__ import annotations

import math
from typing import Any, Dict, List


def _variance(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)


def _std(vals):
    return math.sqrt(_variance(vals))


def _deltas(vals):
    return [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]


def detect_oscillation(vals, thr=3.0):
    """True if consecutive deltas frequently reverse sign."""
    ds = _deltas(vals)
    if len(ds) < 3:
        return False
    sc = sum(1 for i in range(len(ds) - 1) if ds[i] * ds[i + 1] < 0)
    return sc / (len(ds) - 1) > 0.6 and _std(ds) > thr


def detect_collapse(vals, w=5, drop=10.0):
    """True if the metric drops by more than drop within a window."""
    if len(vals) < w:
        return False
    for i in range(len(vals) - w + 1):
        seg = vals[i:i + w]
        if max(seg) - min(seg) > drop and seg[-1] < seg[0]:
            return True
    return False


def monotonicity_score(vals):
    """Fraction of non-decreasing consecutive steps (1.0 = perfect)."""
    if len(vals) < 2:
        return 1.0
    ups = sum(1 for i in range(len(vals) - 1) if vals[i + 1] >= vals[i])
    return ups / (len(vals) - 1)


def evaluate_level4(checkpoint_results):
    """
    Analyse learning-curve stability.

    Each entry needs keys: step, emotion_score, avg_emo_change.
    """
    if not checkpoint_results:
        return {"error": "no checkpoint results"}

    steps = [r["step"] for r in checkpoint_results]
    es = [r["emotion_score"] for r in checkpoint_results]
    ec = [r["avg_emo_change"] for r in checkpoint_results]
    ds = _std(_deltas(es)) if len(es) > 1 else 0.0

    return {
        "steps": steps,
        "emotion_scores": es,
        "emo_changes": ec,
        "score_variance": _variance(es),
        "score_std": _std(es),
        "change_variance": _variance(ec),
        "change_std": _std(ec),
        "delta_std": ds,
        "oscillation_detected": detect_oscillation(es),
        "collapse_detected": detect_collapse(es),
        "monotonicity": monotonicity_score(es),
        "best_step": steps[es.index(max(es))],
        "best_emotion_score": max(es),
        "final_emotion_score": es[-1],
    }
