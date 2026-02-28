# -*- coding: utf-8 -*-
"""Evaluation configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent.parent.parent


@dataclass
class EvalConfig:
    # ── paths ──────────────────────────────────────────────────────────
    output_dir: str = "outputs/ppo_emo"
    test_data_path: str = "data/data/test_profile.jsonl"
    results_dir: str = "eval_results/comprehensive"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # ── API ────────────────────────────────────────────────────────────
    api_key: str = ""
    api_base: str = "https://api.deepseek.com"
    judge_model: str = "deepseek-chat"
    planner_model: str = "deepseek-chat"
    simulator_model: str = "deepseek-chat"

    # ── dialogue generation ────────────────────────────────────────────
    max_turns: int = 8
    max_new_tokens: int = 256
    temperature: float = 0.7
    initial_emo_point: float = 50.0

    # ── evaluation scope ───────────────────────────────────────────────
    max_test_samples: int = 50
    checkpoint_interval: int = 50
    eval_levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    eval_final_only: bool = False

    # ── strategies (Level 2) ───────────────────────────────────────────
    strategies: List[str] = field(default_factory=lambda: [
        "Praise", "Deep Empathy", "Emotional Venting",
        "Advice", "Problem Analysis",
    ])

    # ── core capabilities (Level 3) ───────────────────────────────────
    core_capabilities: List[str] = field(default_factory=lambda: [
        "Empathy Depth", "Core Insight", "Solution Crafting",
        "Dialogue Guidance", "Style Adaptability",
    ])

    # ── SCC axes (Level 3) ────────────────────────────────────────────
    scc_x_label: str = "Solution ↔ Empathy"
    scc_y_label: str = "Structured ↔ Creative"

    # ── visualization ──────────────────────────────────────────────────
    dpi: int = 300
    fig_format: str = "png"
    smooth_span: float = 5.0

    # ── device ─────────────────────────────────────────────────────────
    device: str = "cuda"

    # ── environment comparison (Level 5) ──────────────────────────────
    env_configs: Dict[str, dict] = field(default_factory=lambda: {
        "vanilla": {"simulator_mode": "vanilla"},
        "challenging": {"simulator_mode": "challenging"},
    })

    def resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p

    @property
    def resolved_output_dir(self) -> Path:
        return self.resolve_path(self.output_dir)

    @property
    def resolved_test_data(self) -> Path:
        return self.resolve_path(self.test_data_path)

    @property
    def resolved_results_dir(self) -> Path:
        p = self.resolve_path(self.results_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_api_key(self) -> str:
        return (
            self.api_key
            or os.environ.get("DEEPSEEK_API_KEY", "")
            or os.environ.get("DASHSCOPE_API_KEY", "")
        )
