# -*- coding: utf-8 -*-
"""
Level 5: Environment Impact Analysis.

Compare the same model trained/evaluated under different user-simulator
environments (e.g. vanilla vs challenging).
"""
from __future__ import annotations

from typing import Any, Dict, List

from .dialogue_generator import DialogueResult
from .llm_client import DeepSeekClient
from .prompt_templates import (
    PROMPT_SIMULATOR_NEED_LEVEL,
    format_dialogue_history,
)


def evaluate_simulator_need_level(
    dialogue: DialogueResult,
    client: DeepSeekClient,
) -> Dict[str, float]:
    """Score the user-simulator's need-expression quality for one dialogue."""
    profile = dialogue.profile or {}
    prompt = PROMPT_SIMULATOR_NEED_LEVEL.format(
        dialogue_history=format_dialogue_history(dialogue.dialogue_history),
        user_profile=profile.get("player", "N/A"),
        hidden_topic=profile.get("task", "N/A"),
    )
    raw = client.evaluate(prompt)
    try:
        data = client.parse_json_block(raw)
        scores = {}
        for key in [
            "need_expression_clarity",
            "emotional_authenticity",
            "progressive_disclosure",
            "reactivity",
            "overall",
        ]:
            entry = data.get(key, {})
            if isinstance(entry, dict):
                scores[key] = float(entry.get("score", 0))
            else:
                scores[key] = float(entry)
        return scores
    except Exception:
        return {
            "need_expression_clarity": 0.0,
            "emotional_authenticity": 0.0,
            "progressive_disclosure": 0.0,
            "reactivity": 0.0,
            "overall": 0.0,
        }


def compare_environments(
    env_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare metrics across environments.

    Parameters
    ----------
    env_results : dict mapping env_name -> evaluation result dict, e.g.::

        {
            "vanilla": {"emotion_score": 65.2, "avg_emo_change": 15.2, ...},
            "challenging": {"emotion_score": 58.1, "avg_emo_change": 8.1, ...},
        }

    Returns
    -------
    Comparison table and deltas.
    """
    if len(env_results) < 2:
        return {"note": "need at least 2 environments to compare"}

    env_names = list(env_results.keys())
    comparison_keys = [
        "emotion_score", "avg_emo_change",
    ]
    capability_keys = [
        "empathy_depth", "core_insight", "solution_crafting",
        "dialogue_guidance", "style_adaptability",
    ]

    table = {}
    for key in comparison_keys:
        row = {}
        for env in env_names:
            row[env] = env_results[env].get(key, None)
        vals = [v for v in row.values() if v is not None]
        if len(vals) >= 2:
            row["delta"] = max(vals) - min(vals)
        table[key] = row

    cap_results = {}
    for env in env_names:
        caps = env_results[env].get("core_capabilities", {})
        for key in capability_keys:
            if key not in cap_results:
                cap_results[key] = {}
            cap_results[key][env] = caps.get(key, None)

    for key in capability_keys:
        vals = [v for v in cap_results.get(key, {}).values() if v is not None]
        if len(vals) >= 2:
            cap_results[key]["delta"] = max(vals) - min(vals)

    return {
        "environments": env_names,
        "outcome_comparison": table,
        "capability_comparison": cap_results,
    }


def evaluate_level5(
    env_dialogues: Dict[str, List[DialogueResult]],
    env_results: Dict[str, Dict[str, Any]],
    client: DeepSeekClient,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all Level-5 metrics.

    Parameters
    ----------
    env_dialogues : dict mapping env_name -> list of DialogueResults
    env_results   : dict mapping env_name -> Level-1 + Level-3 result dicts
    client        : DeepSeekClient for simulator evaluation
    """
    sim_scores_by_env: Dict[str, Dict[str, float]] = {}
    for env_name, dialogues in env_dialogues.items():
        if verbose:
            print(f"  [L5] evaluating simulator for env={env_name}")
        all_scores: Dict[str, List[float]] = {}
        for i, dlg in enumerate(dialogues[:10]):
            if verbose:
                print(f"    [{i+1}/10]")
            scores = evaluate_simulator_need_level(dlg, client)
            for k, v in scores.items():
                all_scores.setdefault(k, []).append(v)
        sim_scores_by_env[env_name] = {
            k: sum(v) / len(v) if v else 0.0 for k, v in all_scores.items()
        }

    comparison = compare_environments(env_results)

    return {
        "environment_comparison": comparison,
        "simulator_quality": sim_scores_by_env,
    }
