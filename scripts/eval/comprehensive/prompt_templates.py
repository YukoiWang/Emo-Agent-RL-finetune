# -*- coding: utf-8 -*-
"""
Prompt templates for the comprehensive evaluation framework.

Four families of templates:
  1. Social Cognition Coordinate (SCC)  -- 4 sub-templates
  2. Empathetic Ability Evaluation       -- 5 rubric dimensions
  3. Simulator Need-Expression Level     -- 1 template
  4. Strategy Analysis & Labeling        -- 1 template

Convention: every template uses Python str.format() / {placeholder} syntax.
Rubric descriptions marked with TODO are intended to be filled in with the
detailed scoring definitions you design.
"""
from __future__ import annotations

# ====================================================================
# 1. Social Cognition Coordinate (SCC)
# ====================================================================

PROMPT_SCC_CONVERSATION_ANALYSIS = (
    "# Task\n"
    "Below is a conversation where a user shares their troubles with an AI "
    "assistant. Please analyze in detail why the user's mood improved or "
    "worsened by the end (i.e., why the AI assistant succeeded or failed). "
    "After your analysis, provide a one-paragraph summary.\n\n"
    "# Conversation\n{dialogue_history}\n\n"
    "# Initial Emotion Point: {initial_emo}\n"
    "# Final Emotion Point:   {final_emo}\n\n"
    "# Output format\n"
    "Analysis:\n[Your detailed analysis]\n\n"
    "Summary:\n[One-paragraph summary of key success/failure factors]\n"
)

PROMPT_SCC_MODEL_PROFILE = (
    "# Task\n"
    "The following are analyses of scenarios where the same AI assistant "
    "interacts with multiple users who confide their concerns. Based on the "
    "reasons for its successes or failures, please summarize the key "
    "characteristics of the AI assistant. You can anthropomorphize the AI by "
    "describing its traits in terms of social distance (its relationship "
    "with users), professional role, and personality.\n\n"
    "# Analyses\n{analyses}\n\n"
    "# Output format\n"
    "Model Profile:\n"
    "[A comprehensive paragraph describing the AI assistant's personality, "
    "social distance, professional role, communication style, strengths, "
    "and weaknesses]\n"
)

PROMPT_SCC_STRATEGY_DISTRIBUTION = (
    "# Task\n"
    "Below is a conversation between a user and an AI assistant. For each "
    "AI assistant turn, classify the primary support strategy used.\n\n"
    "Available strategies:\n"
    "- Praise: affirming the user's strengths or efforts\n"
    "- Deep Empathy: deeply understanding and reflecting the user's feelings\n"
    "- Emotional Venting: providing space for the user to express emotions\n"
    "- Advice: offering practical suggestions or guidance\n"
    "- Problem Analysis: analyzing the root cause of the problem\n\n"
    "# Conversation\n{dialogue_history}\n\n"
    "# Output format (JSON array, one entry per assistant turn)\n"
    "```json\n"
    '[{{"turn": 1, "strategy": "<strategy name>", "reason": "<brief>"}}]\n'
    "```\n"
)

PROMPT_SCC_COORDINATE_SCALING = (
    "# Task\n"
    "Given the AI assistant's Model Profile and its Strategy Distribution "
    "across conversations, rate the assistant on two axes (each -1.0 to 1.0):\n\n"
    'Axis X -- "Solution <-> Empathy"\n'
    "  -1.0 = purely solution-oriented (focuses on fixing problems)\n"
    "  +1.0 = purely empathy-oriented (focuses on understanding feelings)\n\n"
    'Axis Y -- "Structured <-> Creative"\n'
    "  -1.0 = highly structured (formulaic, follows templates)\n"
    "  +1.0 = highly creative (adaptive, personalized, flexible)\n\n"
    "# Model Profile\n{model_profile}\n\n"
    "# Strategy Distribution\n{strategy_distribution}\n\n"
    "# Output format (JSON)\n"
    "```json\n"
    '{{"x": <float>, "y": <float>, "reasoning": "<brief>"}}\n'
    "```\n"
)

# ====================================================================
# 2. Core Capability Rubric (Appendix C)
# ====================================================================

PROMPT_CORE_CAPABILITY = (
    "# Task\n"
    "Evaluate the AI assistant's empathetic counseling ability in the "
    "following conversation on five dimensions. Score each 1 to 5.\n\n"
    "# Conversation\n{dialogue_history}\n\n"
    "# User Profile Summary\n{user_profile_summary}\n\n"
    "# Scoring Rubric\n\n"
    "## 1. Empathy Depth (1-5)\n"
    "TODO: Fill in detailed scoring criteria.\n"
    "  1 - No empathy; generic or dismissive.\n"
    "  2 - Surface-level acknowledgment.\n"
    "  3 - Moderate empathy; reflects some emotions.\n"
    "  4 - Strong empathy; captures nuanced emotions.\n"
    "  5 - Exceptional; deeply resonates, captures unspoken feelings.\n\n"
    "## 2. Core Insight (1-5)\n"
    "TODO: Fill in detailed scoring criteria.\n"
    "  1 - No insight into core issue.\n"
    "  2 - Touches superficial aspects.\n"
    "  3 - Identifies some underlying issues.\n"
    "  4 - Meaningful insight into root causes.\n"
    "  5 - Penetrating insight; reveals hidden patterns.\n\n"
    "## 3. Solution Crafting (1-5)\n"
    "TODO: Fill in detailed scoring criteria.\n"
    "  1 - No actionable suggestions.\n"
    "  2 - Generic advice.\n"
    "  3 - Reasonable, somewhat personalized.\n"
    "  4 - Well-crafted, personalized solutions.\n"
    "  5 - Transformative, creative, deeply tailored.\n\n"
    "## 4. Dialogue Guidance (1-5)\n"
    "TODO: Fill in detailed scoring criteria.\n"
    "  1 - No guidance; drifts aimlessly.\n"
    "  2 - Minimal steering; mostly reactive.\n"
    "  3 - Some proactive guidance.\n"
    "  4 - Skillful; deepens conversation naturally.\n"
    "  5 - Masterful flow; seamless transitions.\n\n"
    "## 5. Style Adaptability (1-5)\n"
    "TODO: Fill in detailed scoring criteria.\n"
    "  1 - Rigid, uniform style.\n"
    "  2 - Slight adjustments.\n"
    "  3 - Noticeable adaptation.\n"
    "  4 - Strong adaptation; mirrors user.\n"
    "  5 - Exceptional; dynamic adjustment.\n\n"
    "# Output format (JSON)\n"
    "```json\n"
    "{{\n"
    '  "empathy_depth":      {{"score": <int>, "reason": "<brief>"}},\n'
    '  "core_insight":       {{"score": <int>, "reason": "<brief>"}},\n'
    '  "solution_crafting":  {{"score": <int>, "reason": "<brief>"}},\n'
    '  "dialogue_guidance":  {{"score": <int>, "reason": "<brief>"}},\n'
    '  "style_adaptability": {{"score": <int>, "reason": "<brief>"}}\n'
    "}}\n"
    "```\n"
)

# ====================================================================
# 3. Simulator Need-Expression Level
# ====================================================================

PROMPT_SIMULATOR_NEED_LEVEL = (
    "# Task\n"
    "Evaluate how well the user simulator expresses its emotional needs in "
    "the following conversation. The simulator plays a user seeking emotional "
    "support with a hidden topic.\n\n"
    "# Conversation\n{dialogue_history}\n\n"
    "# User Profile\n{user_profile}\n\n"
    "# Hidden Topic\n{hidden_topic}\n\n"
    "# Evaluation Criteria (rate 1-5 each)\n"
    "1. Need Expression Clarity\n"
    "2. Emotional Authenticity\n"
    "3. Progressive Disclosure\n"
    "4. Reactivity\n\n"
    "# Output format (JSON)\n"
    "```json\n"
    "{{\n"
    '  "need_expression_clarity": {{"score": <int>, "reason": "<brief>"}},\n'
    '  "emotional_authenticity":  {{"score": <int>, "reason": "<brief>"}},\n'
    '  "progressive_disclosure":  {{"score": <int>, "reason": "<brief>"}},\n'
    '  "reactivity":              {{"score": <int>, "reason": "<brief>"}},\n'
    '  "overall":                 {{"score": <float>, "reason": "<brief>"}}\n'
    "}}\n"
    "```\n"
)

# ====================================================================
# 4. Strategy Analysis & Labeling (per NPC response)
# ====================================================================

PROMPT_STRATEGY_LABEL = (
    "# Task\n"
    "Analyze the AI assistant's response strategy in the given dialogue "
    "context. Classify the response into ONE primary strategy.\n\n"
    "# Dialogue Context\n{dialogue_context}\n\n"
    "# Latest AI Assistant Response\n{npc_response}\n\n"
    "# Available Strategies\n"
    "- Praise: affirming strengths, efforts, or progress\n"
    "- Deep Empathy: deeply understanding and reflecting feelings\n"
    "- Emotional Venting: creating safe space for emotional release\n"
    "- Advice: offering practical suggestions or guidance\n"
    "- Problem Analysis: analyzing situation logically, root causes\n\n"
    "# Output format (JSON)\n"
    "```json\n"
    "{{\n"
    '  "strategy": "<Praise / Deep Empathy / Emotional Venting / Advice / Problem Analysis>",\n'
    '  "confidence": <float 0-1>,\n'
    '  "reason": "<brief>"\n'
    "}}\n"
    "```\n"
)


# ====================================================================
# Helpers
# ====================================================================

def format_dialogue_history(dialogue: list[dict[str, str]]) -> str:
    """Pretty-print a list of {role, content} messages."""
    lines = []
    role_map = {"user": "User", "assistant": "AI Assistant"}
    for msg in dialogue:
        role = role_map.get(msg["role"], msg["role"])
        lines.append(f"[{role}]: {msg['content']}")
    return "\n\n".join(lines)


def format_user_profile_summary(profile: dict) -> str:
    """One-paragraph summary of the user profile for LLM prompts."""
    player = profile.get("player", "")
    task = profile.get("task", "")
    main_cha = profile.get("main_cha", "")
    cha_group = profile.get("cha_group", [])
    parts = []
    if main_cha:
        parts.append(f"Character type: {main_cha}")
    if cha_group:
        parts.append(f"Traits: {', '.join(cha_group)}")
    if task:
        parts.append(f"Hidden need: {task}")
    summary = "; ".join(parts)
    first_line = player.split("\n")[0] if player else ""
    return f"{first_line} ({summary})" if summary else first_line
