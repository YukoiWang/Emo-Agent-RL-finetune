# -*- coding: utf-8 -*-
"""
RL 相关工具函数。

- 单轮 PPO（静态 prompt + Reward Model 打分）：见 static-rl/run_ppo.py。
- 多轮 PPO（Profile + 用户模拟器）：见 src.training.ppo_emo_trainer.run_ppo_emo_training。
- simple_empathy_reward_fn：供 GRPO 等脚本使用的规则 reward 示例。
"""
from __future__ import annotations

from typing import List


def simple_empathy_reward_fn(texts: List[str]) -> List[float]:
    """
    非常简单的规则奖励示例：
    - 包含一些共情关键词 +1
    - 出现明显危险词 -1
    实际使用时建议替换为训练好的 reward model 或人工打分数据。
    """
    positive_keywords = ["我理解你", "听起来你", "能感受到你", "谢谢你愿意分享", "你一定很不容易"]
    negative_keywords = ["自杀", "杀人", "别在乎", "无所谓", "不重要"]

    rewards = []
    for t in texts:
        score = 0.0
        for k in positive_keywords:
            if k in t:
                score += 1.0
        for k in negative_keywords:
            if k in t:
                score -= 1.0
        rewards.append(score)
    return rewards
