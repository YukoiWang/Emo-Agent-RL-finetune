# -*- coding: utf-8 -*-
"""
用户模拟器（PlayerSimulator）：根据 NPC 回复与隐藏主题的贴合程度、用户情感分析更新 emo_point，
并与 actor 进行多轮对话。说「再见/拜拜」或 emo_point <= 0 时结束对话。
"""
from __future__ import annotations

import re
from typing import Callable, Dict, Any, List, Optional, Tuple

# 规划+回复：用户模拟器用外部 LLM API 生成下一句；用 emo_analyzer 得到 change_value 更新 emo_point


def _default_theme_fit_fn(npc_reply: str, hidden_theme: str) -> float:
    """
    简单规则：NPC 回复与隐藏主题的贴合程度 [-1, 1]。
    可被外部 API（如 LLM 打分）替代。
    """
    # 占位：实际可调用 LLM 判断 npc_reply 是否贴合 hidden_theme
    if not npc_reply.strip() or not hidden_theme:
        return 0.0
    theme_lower = hidden_theme.lower().strip()
    reply_lower = npc_reply.lower().strip()
    if theme_lower in reply_lower:
        return 0.5
    # 简单关键词重叠
    tw = set(theme_lower.replace("，", " ").replace("。", " ").split())
    rw = set(reply_lower.replace("，", " ").replace("。", " ").split())
    overlap = len(tw & rw) / max(len(tw), 1)
    return min(1.0, overlap * 2)  # 粗略映射到 [0,1]，再转为 [-1,1] 在下面


class PlayerSimulator:
    """
    多轮对话中的「用户」模拟器。
    - 用 user_llm_fn 根据当前对话历史生成用户回复；
    - 用 emo_analyzer_fn 根据「NPC 回复、用户回复、隐藏主题」得到情绪变化 change_value；
    - emo_point += change_value，clamp 在 [0, 100]；
    - 若用户说了再见/拜拜或 emo_point <= 0，则对话结束。
    """

    GOODBYE_PATTERN = re.compile(r"(再见|拜拜|再会|下次聊|先这样)")

    def __init__(
        self,
        profile: Dict[str, Any],
        user_llm_fn: Callable[[List[Dict[str, str]]], str],
        emo_analyzer_fn: Callable[[str, str, str], Dict[str, Any]],
        initial_emo_point: float = 50.0,
        theme_fit_fn: Optional[Callable[[str, str], float]] = None,
    ):
        """
        profile: 至少包含 "player", "scene", "task"（或 "hidden_theme"）的字典。
        user_llm_fn: 输入 messages [{"role":"user"/"assistant", "content": "..."}]，输出用户下一句文本。
        emo_analyzer_fn: 输入 (npc_reply, user_reply, hidden_theme)，返回 dict，需包含 "change_value" (float)。
                        可选 "sentiment" (0/1/2), "theme_fit" (float)。
        initial_emo_point: 初始情绪分 [0, 100]。
        theme_fit_fn: 可选，单独计算 NPC 与主题贴合度；若 None 则仅依赖 emo_analyzer_fn 的 change。
        """
        self.profile = profile
        self.user_llm_fn = user_llm_fn
        self.emo_analyzer_fn = emo_analyzer_fn
        self.theme_fit_fn = theme_fit_fn or _default_theme_fit_fn
        self.emo_point = max(0.0, min(100.0, float(initial_emo_point)))
        self.hidden_theme = (
            profile.get("task") or profile.get("hidden_theme") or ""
        ).strip()
        if not self.hidden_theme and "scene" in profile:
            # 从 scene 里解析 "####隐藏主题:"
            raw = profile["scene"] or ""
            for line in raw.split("\n"):
                if "隐藏主题" in line or "hidden" in line.lower():
                    self.hidden_theme = line.split(":", 1)[-1].strip()
                    break
        self.dialog: List[Dict[str, str]] = []  # 完整多轮 [{"role":"user"/"assistant","content":"..."}]
        self.emo_point_turns: List[float] = [self.emo_point]  # 每轮结束后的 emo_point，用于 mode2 reward

    def _build_system_and_start(self) -> List[Dict[str, str]]:
        """根据 profile 构建系统提示和首轮用户开场（若需要）。"""
        player = self.profile.get("player", "")
        scene = self.profile.get("scene", "")
        task = self.profile.get("task", "") or self.hidden_theme
        system = (
            "你正在扮演以下角色与一位 NPC 进行倾诉对话。请严格按人设和当前情绪状态回复，每次只回复一段话。\n\n"
            f"【角色设定】\n{player}\n\n"
            f"【背景与当前处境】\n{scene}\n\n"
        )
        if task:
            system += f"【你内心的诉求/隐藏主题】\n{task}\n\n"
        system += "请用第一人称、口语化地回复，不要重复 NPC 的话。若想结束对话可以说再见或拜拜。"
        return [{"role": "system", "content": system}]

    def step(self, npc_reply: str) -> Tuple[str, bool]:
        """
        给定 NPC 本轮的回复，生成用户回复并更新 emo_point。
        返回 (user_reply, done)。done=True 表示对话应结束。
        """
        npc_reply = (npc_reply or "").strip()
        if not npc_reply:
            return "（请继续说。）", False

        # 1) 把 NPC 回复加入对话，并调用 user_llm 得到用户回复
        self.dialog.append({"role": "assistant", "content": npc_reply})
        messages = self._build_system_and_start() + self.dialog
        user_reply = self.user_llm_fn(messages)
        user_reply = (user_reply or "").strip() or "…"
        self.dialog.append({"role": "user", "content": user_reply})

        # 2) 用 emo_analyzer 得到情绪变化
        out = self.emo_analyzer_fn(npc_reply, user_reply, self.hidden_theme)
        change_value = float(out.get("change_value", 0.0))
        self.emo_point = max(0.0, min(100.0, self.emo_point + change_value))
        self.emo_point_turns.append(self.emo_point)

        # 3) 是否结束：再见/拜拜 或 emo_point <= 0
        done = self.emo_point <= 0 or bool(self.GOODBYE_PATTERN.search(user_reply))
        return user_reply, done

    def get_emo_point(self) -> float:
        return self.emo_point

    def get_emo_point_turns(self) -> List[float]:
        return list(self.emo_point_turns)

    def reset(self, initial_emo_point: Optional[float] = None) -> None:
        if initial_emo_point is not None:
            self.emo_point = max(0.0, min(100.0, initial_emo_point))
        self.dialog = []
        self.emo_point_turns = [self.emo_point]
