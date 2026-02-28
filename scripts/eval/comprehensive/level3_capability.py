# -*- coding: utf-8 -*-
"""
Level 3 -- Capability Structure Metrics.

  5. SCC (Social Cognition Coordinate): 2-D style coordinate.
  6. Core Capability Rubric: five 1-5 scores.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from .dialogue_generator import DialogueResult
from .llm_client import DeepSeekClient
from .prompt_templates import (
    PROMPT_CORE_CAPABILITY,
    PROMPT_SCC_CONVERSATION_ANALYSIS,
    PROMPT_SCC_COORDINATE_SCALING,
    PROMPT_SCC_MODEL_PROFILE,
    PROMPT_SCC_STRATEGY_DISTRIBUTION,
    format_dialogue_history,
    format_user_profile_summary,
)

CAPABILITY_KEYS = [
    "empathy_depth",
    "core_insight",
    "solution_crafting",
    "dialogue_guidance",
    "style_adaptability",
]


# ── SCC ─────────────────────────────────────────────────────────────────

def _scc_step1_analyse_conversations(
    dialogues: List[DialogueResult],
    client: DeepSeekClient,
    verbose: bool = True,
) -> List[str]:
    """Step 1: analyse success / failure of each conversation."""
    analyses: List[str] = []
    for i, dlg in enumerate(dialogues):
        if verbose:
            print(f"    [SCC step1] {i+1}/{len(dialogues)}")
        prompt = PROMPT_SCC_CONVERSATION_ANALYSIS.format(
            dialogue_history=format_dialogue_history(dlg.dialogue_history),
            initial_emo=dlg.initial_emo,
            final_emo=dlg.final_emo,
        )
        result = client.evaluate(prompt)
        analyses.append(result)
    return analyses


def _scc_step2_model_profile(
    analyses: List[str],
    client: DeepSeekClient,
) -> str:
    """Step 2: extract model profile from aggregated analyses."""
    combined = "\n\n---\n\n".join(
        f"Conversation {i+1}:\n{a}" for i, a in enumerate(analyses)
    )
    prompt = PROMPT_SCC_MODEL_PROFILE.format(analyses=combined)
    return client.evaluate(prompt)


def _scc_step3_strategy_distribution(
    dialogues: List[DialogueResult],
    client: DeepSeekClient,
    verbose: bool = True,
) -> Dict[str, float]:
    """Step 3: aggregate strategy distribution across conversations."""
    from collections import Counter
    all_strategies: List[str] = []

    for i, dlg in enumerate(dialogues):
        if verbose:
            print(f"    [SCC step3] {i+1}/{len(dialogues)}")
        prompt = PROMPT_SCC_STRATEGY_DISTRIBUTION.format(
            dialogue_history=format_dialogue_history(dlg.dialogue_history),
        )
        raw = client.evaluate(prompt)
        try:
            data = client.parse_json_block(raw)
            if isinstance(data, list):
                for entry in data:
                    s = entry.get("strategy", "Unknown")
                    all_strategies.append(s)
        except Exception:
            pass

    counts = Counter(all_strategies)
    total = max(len(all_strategies), 1)
    return {s: c / total for s, c in counts.most_common()}


def _scc_step4_coordinate(
    model_profile: str,
    strategy_distribution: Dict[str, float],
    client: DeepSeekClient,
) -> Tuple[float, float]:
    """Step 4: compute SCC (x, y) coordinate."""
    strat_str = json.dumps(strategy_distribution, indent=2)
    prompt = PROMPT_SCC_COORDINATE_SCALING.format(
        model_profile=model_profile,
        strategy_distribution=strat_str,
    )
    raw = client.evaluate(prompt)
    try:
        data = client.parse_json_block(raw)
        x = float(data.get("x", 0.0))
        y = float(data.get("y", 0.0))
    except Exception:
        x, y = 0.0, 0.0
    return (x, y)


def compute_scc(
    dialogues: List[DialogueResult],
    client: DeepSeekClient,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full SCC pipeline: 4 steps -> (x, y) coordinate + intermediates."""
    analyses = _scc_step1_analyse_conversations(dialogues, client, verbose)
    profile = _scc_step2_model_profile(analyses, client)
    strat_dist = _scc_step3_strategy_distribution(dialogues, client, verbose)
    coord = _scc_step4_coordinate(profile, strat_dist, client)

    return {
        "coordinate": coord,
        "model_profile": profile,
        "strategy_distribution": strat_dist,
    }


# ── Core Capability Rubric ──────────────────────────────────────────────

def evaluate_single_capability(
    dialogue: DialogueResult,
    client: DeepSeekClient,
) -> Dict[str, float]:
    """Score 5 capability dimensions for a single dialogue."""
    profile_summary = format_user_profile_summary(dialogue.profile)
    prompt = PROMPT_CORE_CAPABILITY.format(
        dialogue_history=format_dialogue_history(dialogue.dialogue_history),
        user_profile_summary=profile_summary,
    )
    raw = client.evaluate(prompt)
    scores: Dict[str, float] = {}
    try:
        data = client.parse_json_block(raw)
        for key in CAPABILITY_KEYS:
            entry = data.get(key, {})
            scores[key] = float(entry.get("score", 0)) if isinstance(entry, dict) else float(entry)
    except Exception:
        for key in CAPABILITY_KEYS:
            scores[key] = client.parse_score(raw, key.replace("_", " "))
    return scores


def compute_core_capabilities(
    dialogues: List[DialogueResult],
    client: DeepSeekClient,
    verbose: bool = True,
) -> Dict[str, float]:
    """Average capability scores across dialogues."""
    from collections import defaultdict
    accum: Dict[str, List[float]] = defaultdict(list)

    for i, dlg in enumerate(dialogues):
        if verbose:
            print(f"  [L3 capability] {i+1}/{len(dialogues)}")
        scores = evaluate_single_capability(dlg, client)
        for k, v in scores.items():
            accum[k].append(v)

    return {
        k: sum(vals) / len(vals) if vals else 0.0
        for k, vals in accum.items()
    }


# ── top-level ───────────────────────────────────────────────────────────

def evaluate_level3(
    dialogues: List[DialogueResult],
    client: DeepSeekClient,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run all Level-3 metrics."""
    scc_result = compute_scc(dialogues, client, verbose)
    capabilities = compute_core_capabilities(dialogues, client, verbose)

    return {
        "scc": scc_result,
        "core_capabilities": capabilities,
    }
