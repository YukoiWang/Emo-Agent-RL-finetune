# -*- coding: utf-8 -*-
"""
Level 2 -- Behavior Strategy Metrics.

  3. Strategy Frequency   : proportion of each strategy in NPC responses.
  4. Strategy Contribution: conditional mean EmoChange per strategy.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .dialogue_generator import DialogueResult
from .llm_client import DeepSeekClient
from .prompt_templates import (
    PROMPT_STRATEGY_LABEL,
    format_dialogue_history,
)

VALID_STRATEGIES = [
    "Praise", "Deep Empathy", "Emotional Venting", "Advice", "Problem Analysis",
]


def classify_single_response(
    dialogue_context: str,
    npc_response: str,
    client: DeepSeekClient,
) -> str:
    """Use the LLM judge to classify one NPC response into a strategy."""
    prompt = PROMPT_STRATEGY_LABEL.format(
        dialogue_context=dialogue_context,
        npc_response=npc_response,
    )
    raw = client.evaluate(prompt)
    try:
        data = client.parse_json_block(raw)
        strategy = data.get("strategy", "Unknown")
    except Exception:
        strategy = "Unknown"
        for s in VALID_STRATEGIES:
            if s.lower() in raw.lower():
                strategy = s
                break
    if strategy not in VALID_STRATEGIES:
        for s in VALID_STRATEGIES:
            if s.lower() in strategy.lower():
                strategy = s
                break
        else:
            strategy = "Unknown"
    return strategy


def classify_dialogue_strategies(
    dialogue: DialogueResult,
    client: DeepSeekClient,
) -> List[str]:
    """Classify all NPC turns in a single dialogue."""
    strategies: List[str] = []
    context_parts: List[str] = []

    for msg in dialogue.dialogue_history:
        if msg["role"] == "assistant":
            context_str = "\n".join(context_parts[-6:])
            strategy = classify_single_response(
                context_str, msg["content"], client,
            )
            strategies.append(strategy)
        context_parts.append(
            f"[{'User' if msg['role'] == 'user' else 'AI'}]: {msg['content']}"
        )
    return strategies


def compute_strategy_frequency(
    all_strategies: List[str],
) -> Dict[str, float]:
    """Compute fraction of each strategy label."""
    if not all_strategies:
        return {}
    counts = Counter(all_strategies)
    total = len(all_strategies)
    return {s: counts.get(s, 0) / total for s in VALID_STRATEGIES + ["Unknown"]}


def compute_strategy_contribution(
    dialogues: List[DialogueResult],
    per_dialogue_strategies: List[List[str]],
) -> Dict[str, float]:
    """
    Conditional mean EmoChange per strategy.

    For each dialogue we assign its emo_change to the *dominant strategy*
    (the most frequent label across its NPC turns).
    """
    strat_emo: Dict[str, List[float]] = defaultdict(list)

    for dlg, strats in zip(dialogues, per_dialogue_strategies):
        if not strats:
            continue
        dominant = Counter(strats).most_common(1)[0][0]
        strat_emo[dominant].append(dlg.emo_change)

    result: Dict[str, float] = {}
    for s in VALID_STRATEGIES + ["Unknown"]:
        vals = strat_emo.get(s, [])
        result[s] = sum(vals) / len(vals) if vals else 0.0
    return result


def evaluate_level2(
    dialogues: List[DialogueResult],
    client: DeepSeekClient,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run all Level-2 metrics."""
    all_strategies: List[str] = []
    per_dialogue_strategies: List[List[str]] = []

    for i, dlg in enumerate(dialogues):
        if verbose:
            print(f"  [L2 strategy] {i+1}/{len(dialogues)}")
        strats = classify_dialogue_strategies(dlg, client)
        per_dialogue_strategies.append(strats)
        all_strategies.extend(strats)

    freq = compute_strategy_frequency(all_strategies)
    contribution = compute_strategy_contribution(
        dialogues, per_dialogue_strategies,
    )

    return {
        "strategy_frequency": freq,
        "strategy_contribution": contribution,
        "per_dialogue_strategies": per_dialogue_strategies,
        "total_responses_classified": len(all_strategies),
    }
