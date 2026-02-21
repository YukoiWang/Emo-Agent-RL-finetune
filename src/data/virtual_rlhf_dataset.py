# -*- coding: utf-8 -*-
"""
VirtualRLHFDataset: system_prompt_trained + 首条用户消息（来自 player_simulator.reply(None)）
用于 RL 多轮对话训练的虚拟 HF 格式数据集。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from torch.utils.data import Dataset

from .profile_dataset import build_initial_prompt, load_profiles
from ..training.hard_player_simulator_dsv3 import build_player_simulator_with_planning


class VirtualRLHFDataset(Dataset):
    """
    虚拟 RLHF 数据集：
    - system_prompt_trained: NPC 的系统提示（用户形象 + 背景 + 诉求）
    - first_user_msg: 来自 player_simulator.reply(None) 的用户开场白
    - profile: 原始 profile，供多轮 rollout 使用
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_scene_len: int = 1500,
        player_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        planning_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        sft_model_path: Optional[str] = None,
        target: str = "eq",
        api_key: Optional[str] = None,
        qwen_model: str = "qwen-plus",
        cache_first_messages: bool = False,
    ):
        """
        data_dir: profile jsonl 目录（如 data/data）
        player_llm_fn: 用户回复生成（API），用于首条消息和 player_reply
        planning_llm_fn: 情感分析（本地 SFT），用于 planning_reply
        sft_model_path: 若提供，用本地 SFT 基座做 planning，否则与 player 共用
        target: 对话目的 "eq" 等
        cache_first_messages: 是否缓存首条消息（避免重复调用 API）
        """
        self.profiles = load_profiles(data_dir, split=split)
        self.max_scene_len = max_scene_len
        self.target = target
        self.cache_first_messages = cache_first_messages
        self._first_msg_cache: Dict[int, str] = {}

        self._player_llm_fn = player_llm_fn
        self._planning_llm_fn = planning_llm_fn
        self._sft_model_path = sft_model_path
        self._api_key = api_key
        self._qwen_model = qwen_model

    def _get_simulator(self, profile: Dict[str, Any]):
        return build_player_simulator_with_planning(
            profile=profile,
            player_llm_fn=self._player_llm_fn,
            planning_llm_fn=self._planning_llm_fn,
            sft_model_path=self._sft_model_path,
            api_key=self._api_key,
            model=self._qwen_model,
            target=self.target,
        )

    def _get_first_user_message(self, idx: int, profile: Dict[str, Any]) -> str:
        if self.cache_first_messages and idx in self._first_msg_cache:
            return self._first_msg_cache[idx]
        sim = self._get_simulator(profile)
        ret = sim.reply(None)
        msg = (ret.get("content") or "").strip() or "我最近有些事想和你聊聊。"
        if self.cache_first_messages:
            self._first_msg_cache[idx] = msg
        return msg

    def __len__(self) -> int:
        return len(self.profiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        profile = self.profiles[idx]
        system_prompt = build_initial_prompt(profile, max_scene_len=self.max_scene_len)
        first_user_msg = self._get_first_user_message(idx, profile)
        prompt = system_prompt.rstrip() + "\n\n用户：" + first_user_msg + "\n\nNPC："
        return {
            "profile": profile,
            "system_prompt_trained": system_prompt,
            "first_user_msg": first_user_msg,
            "prompt": prompt,
            "idx": idx,
            "target": self.target,
        }
