# -*- coding: utf-8 -*-
"""
从 /home/yukiwang/xlwy/data/data 下的 profile jsonl 加载用户形象，生成用于 PPO 多轮对话的样本。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional

from torch.utils.data import Dataset


def load_profiles(data_dir: str, split: str = "train") -> List[Dict[str, Any]]:
    """
    data_dir: 例如 /home/yukiwang/xlwy/data/data
    split: "train" -> train_profile.jsonl, "test" -> test_profile.jsonl
    """
    fname = "train_profile.jsonl" if split == "train" else "test_profile.jsonl"
    path = os.path.join(data_dir, fname)
    if not os.path.isfile(path):
        path = os.path.join(data_dir, "train_profile.jsonl")
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_initial_prompt(profile: Dict[str, Any], max_scene_len: int = 1500) -> str:
    """
    根据 profile 构建 NPC 看到的「当前用户形象 + 背景」，作为对话的初始 prompt。
    用于多轮对话的第一条输入给 actor。
    """
    player = profile.get("player", "")
    scene = profile.get("scene", "")
    task = profile.get("task", "") or ""
    if not task and "隐藏主题" in (scene or ""):
        for line in (scene or "").split("\n"):
            if "隐藏主题" in line or "hidden" in line.lower():
                task = line.split(":", 1)[-1].strip()
                break
    if scene and len(scene) > max_scene_len:
        scene = scene[:max_scene_len] + "…"
    parts = [
        "你是一位善于倾听、能给出共情与建议的 NPC。当前有一位用户来向你倾诉。",
        "",
        "【用户形象】",
        player,
        "",
        "【背景与当前处境】",
        scene,
        "",
    ]
    if task:
        parts.append("【用户内心的诉求（可据此给予贴合主题的回应）】")
        parts.append(task)
        parts.append("")
    parts.append("请用简洁、温暖的口吻回复用户，每次一段话。现在用户刚开口，请先回应并引导对话。")
    return "\n".join(parts)


class ProfileDataset(Dataset):
    """Dataset 返回 profile 与构建好的 prompt 文本。"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_scene_len: int = 1500,
    ):
        self.profiles = load_profiles(data_dir, split=split)
        self.max_scene_len = max_scene_len

    def __len__(self) -> int:
        return len(self.profiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        profile = self.profiles[idx]
        prompt = build_initial_prompt(profile, max_scene_len=self.max_scene_len)
        return {"profile": profile, "prompt": prompt, "idx": idx}
