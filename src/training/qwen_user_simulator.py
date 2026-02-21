# -*- coding: utf-8 -*-
"""
用户模拟器：仿照 DSV3 思路，情感分析用 EmoAnalyzer（emo_classifier_lora），
用户回复用 Qwen API 生成。支持 target_prompt、emo_state、多轮对话。
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 对话目的（与用户提供的 target_prompt 一致）
TARGET_PROMPT = {
    "no-target": "你的对话目的是根据人物画像和对话背景，和NPC进行闲聊，你要等待NPC提出话题，然后按兴趣进行回复，你不需要主动提出或者转移话题。你要根据对话背景内定义的对话有趣度来进行对话",
    "target": "你的对话目的是首先完成自己的短期目标，随后按照自己的兴趣爱好进行闲聊。你要根据对话背景内定义的对话有趣度来进行对话",
    "test": "你的对话目的是根据人物画像和对话背景，扮演测试员和NPC进行对话，你要等待NPC提出话题，然后进行回复，你不需要主动提出或者转移话题。",
    "eq": """你的对话目的是谈心，谈心是指深入、真诚的交流，通常涉及个人情感、内心想法或重要话题。谈心的目的是为了增进理解、解决问题或分享感受，参与者通常会敞开心扉，表达真实的想法和情感。
*你需要根据对话背景内的"玩家可能想向NPC倾诉的主题"开启并深入谈心。
*你的目标是按照对话背景内的隐藏主题进行倾诉，但是你不可以直白的泄露隐藏主题。
*你需要根据你的当前情绪，按照对话背景内的相关定义进行不一样的回复。
*你要从玩家画像和背景中提取相关信息，完成高质量的回复。
*你不应该一直表达抽象的感受，而是用具体事件倾诉。
*你不应该表达"我真的很绝望""我真的很痛苦"，而是应该将感情隐含在你的发言中。""",
}


def _call_qwen_api(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = "qwen-plus",
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """
    调用 LLM API，返回 assistant 回复文本。
    支持 DashScope（Qwen）和 OpenAI-compatible API（DeepSeek 等）。
    通过 base_url 自动判断：含 deepseek 或指定了 base_url 时走 OpenAI 兼容协议。
    """
    api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 API key，请设置 DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY 环境变量")

    use_openai_compat = bool(base_url) or "deepseek" in model.lower()
    if use_openai_compat:
        return _call_openai_compatible_api(
            messages=messages, api_key=api_key, model=model,
            base_url=base_url or "https://api.deepseek.com",
            temperature=temperature, max_retries=max_retries,
        )

    try:
        import dashscope
        from dashscope import Generation
    except ImportError:
        raise ImportError("请安装 dashscope: pip install dashscope")

    for attempt in range(max_retries):
        try:
            resp = Generation.call(
                api_key=api_key,
                model=model,
                messages=messages,
                result_format="message",
                temperature=temperature,
            )
            if resp.status_code == 200 and resp.output and resp.output.choices:
                return (resp.output.choices[0].message.content or "").strip()
            if hasattr(resp, "message") and resp.message:
                raise RuntimeError(f"Qwen API 错误: {resp.message}")
        except Exception as e:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(2 ** attempt)
    return ""


def _call_openai_compatible_api(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    temperature: float = 0.7,
    max_retries: int = 3,
) -> str:
    """通过 requests 调用 OpenAI-compatible API（DeepSeek 等）。"""
    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 2048,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
            raise RuntimeError(f"API 错误 ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(2 ** attempt)
    return ""


def build_qwen_user_llm_fn(
    api_key: Optional[str] = None,
    model: str = "qwen-plus",
    base_url: Optional[str] = None,
    temperature: float = 0.7,
) -> Callable[[List[Dict[str, str]]], str]:
    """
    构建基于 LLM API 的 user_llm_fn，支持 Qwen (DashScope) 和 DeepSeek 等 OpenAI-compatible API。
    输入: messages [{"role":"system"/"user"/"assistant","content":"..."}]
    输出: 用户下一句文本
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("DASHSCOPE_API_KEY", "")

    def fn(messages: List[Dict[str, str]]) -> str:
        return _call_qwen_api(
            messages=messages,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    return fn


def build_user_llm_with_emo_prompt(
    qwen_fn: Callable[[List[Dict[str, str]]], str],
    target: str = "eq",
) -> Callable[[List[Dict[str, str]], str, str], str]:
    """
    包装 qwen_fn，使其接收 (messages, emo_state, planning_hint)，
    并构建带有 emo_state、target、planning 的用户回复 prompt。
    """

    def fn(
        messages: List[Dict[str, str]],
        emo_state: str,
        planning_hint: str = "",
    ) -> str:
        # 在最后一条 user 消息后追加当前情绪和规划提示
        target_text = TARGET_PROMPT.get(target, TARGET_PROMPT["eq"])
        emo_desc = {
            "Emotion-S": "你的emotion已经达到了最高，你可以感谢NPC并用「再见」或「拜拜」告别，直接结束对话。",
            "Emotion-A": "高emotion，此时你的对话感受较为正面，你的反馈也比较正面。",
            "Emotion-B": "中emotion，此时你没有正面或者负面的感受。",
            "Emotion-C": "低emotion，此时你的对话感受较为负面，你的反馈也比较负面。",
            "Emotion-F": "你的情绪已到达最负面，不想继续对话。此时你要用「再见」或「拜拜」告别，直接结束对话。",
        }.get(emo_state, "请根据上下文自然回复。")

        extra = f"\n\n【当前情绪状态】{emo_state}: {emo_desc}\n【对话目的】{target_text[:100]}..."
        if planning_hint:
            extra += f"\n【对NPC回复的简要感受】{planning_hint}"

        # 复制并修改最后一条 user 消息，或追加新消息
        msgs = list(messages)
        if msgs and msgs[-1].get("role") == "user":
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + extra}
        else:
            msgs.append({"role": "user", "content": "请根据当前情绪和对话上下文，用第一人称简短回复。" + extra})

        return qwen_fn(msgs)

    return fn


class PlayerSimulatorDSV3:
    """
    用户模拟器：仿照用户提供的 PlayerSimulator 思路。
    - 情感分析：EmoAnalyzer（sft_emo_model / emo_classifier_lora）
    - 用户回复：Qwen API
    - 支持 target_prompt、emo_state、emo_count、多轮对话
    """

    GOODBYE_PATTERN = re.compile(r"(再见|拜拜|再会|下次聊|先这样)")

    EMO_COUNT = {"Emotion-S": 100, "Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}

    def __init__(
        self,
        profile: Dict[str, Any],
        emo_analyzer_fn: Callable[[str, str, str], Dict[str, Any]],
        qwen_user_llm_fn: Callable[[List[Dict[str, str]]], str],
        target: str = "eq",
        initial_emo_point: float = 50.0,
        api_key: Optional[str] = None,
    ):
        """
        profile: 至少包含 player, scene, task, topic 等（来自 train_profile.jsonl）
        emo_analyzer_fn: (npc_reply, user_reply, hidden_theme) -> {change_value, sentiment, theme_fit}
        qwen_user_llm_fn: messages -> str，用于生成用户回复
        target: "eq" | "no-target" | "target" | "test"
        initial_emo_point: 初始 emo_point [0, 100]
        """
        self.profile = profile
        self.emo_analyzer_fn = emo_analyzer_fn
        self.qwen_user_llm_fn = qwen_user_llm_fn
        self.target = target
        self.emo_point = max(0.0, min(100.0, float(initial_emo_point)))
        self.emo_state = self._emo_point_to_state(self.emo_point)
        self.emo_count = dict(self.EMO_COUNT)

        self.hidden_theme = (profile.get("task") or "").strip()
        if not self.hidden_theme and "scene" in profile:
            raw = profile.get("scene") or ""
            for line in raw.split("\n"):
                if "隐藏主题" in line or "hidden" in line.lower():
                    self.hidden_theme = line.split(":", 1)[-1].strip().strip("*").strip()
                    break

        self.dialog: List[Dict[str, str]] = []
        self.emo_point_turns: List[float] = [self.emo_point]
        self.planning_hint: str = ""  # 上轮 EmoAnalyzer 的简要结果，供下次 prompt 参考

    def _emo_point_to_state(self, point: float) -> str:
        for state, threshold in sorted(self.EMO_COUNT.items(), key=lambda x: -x[1]):
            if point >= threshold:
                return state
        return "Emotion-F"

    def _build_system_and_start(self) -> List[Dict[str, str]]:
        """兼容 ppo_emo_rollout：返回 [{"role":"system","content":"..."}]"""
        return [{"role": "system", "content": self._build_system_prompt()}]

    def _build_system_prompt(self) -> str:
        player = self.profile.get("player", "")
        scene = self.profile.get("scene", "")
        target_text = TARGET_PROMPT.get(self.target, TARGET_PROMPT["eq"])
        return f"""你是一个演员。你将根据剧本中的人物画像和对话背景扮演角色和一个NPC进行对话。

# 你的任务
*你目的是在对话中扮演好人物画像和对话背景构成的角色
*你需要根据你实时变换的emotion，结合人物画像和对话背景中的相关定义，选择不同的对话策略，完成符合角色特征的回复

# 你的对话目的
*{target_text}

# Emotion（你会收到当前情绪）
- Emotion-S: 可感谢NPC并用「再见」或「拜拜」告别，直接结束对话
- Emotion-A: 高emotion，反馈较为正面
- Emotion-B: 中emotion，无明显偏向
- Emotion-C: 低emotion，反馈较为负面
- Emotion-F: 情绪最负面，用「再见」或「拜拜」结束对话

# 发言准则
1. 发言必须简洁、随意、自然
2. 不许一次提问超过两个问题
3. 不允许重复之前说过的回复
4. 可以自然使用口语化词汇
5. 发言应该精简，不准过长
6. 不要直白陈述情绪（如「我真的很绝望」），将感情蕴含在发言中

# 人物画像
{player}

# 当前对话背景
{scene}
"""

    def _messages_for_user_reply(self, planning_hint: str = "") -> List[Dict[str, str]]:
        msgs = [{"role": "system", "content": self._build_system_prompt()}]
        for m in self.dialog:
            msgs.append({"role": m["role"], "content": m["content"]})
        if planning_hint:
            tail = f"【对NPC最新回复的简要感受】{planning_hint}\n\n请根据以上感受和当前情绪（{self.emo_state}）用第一人称简短回复。"
        else:
            tail = f"请根据对话上下文和当前情绪（{self.emo_state}）用第一人称简短回复。若尚未开始，请用简短的回复开启倾诉。"
        msgs.append({"role": "user", "content": tail})
        return msgs

    def step(self, npc_reply: str) -> Tuple[str, bool]:
        """
        给定 NPC 本轮的回复，生成用户回复并更新 emo_point。
        返回 (user_reply, done)。
        """
        npc_reply = (npc_reply or "").strip()
        if not npc_reply:
            return "（请继续说。）", False

        self.dialog.append({"role": "assistant", "content": npc_reply})
        messages = self._messages_for_user_reply(planning_hint=self.planning_hint)
        user_reply = self.qwen_user_llm_fn(messages)
        user_reply = (user_reply or "").strip() or "…"
        self.dialog.append({"role": "user", "content": user_reply})

        out = self.emo_analyzer_fn(npc_reply, user_reply, self.hidden_theme)
        change_value = float(out.get("change_value", 0.0))
        self.emo_point = max(0.0, min(100.0, self.emo_point + change_value))
        self.emo_state = self._emo_point_to_state(self.emo_point)
        self.emo_point_turns.append(self.emo_point)

        sentiment = out.get("sentiment", 1)
        theme_fit = out.get("theme_fit", 0.5)
        sent_map = {0: "消极", 1: "中性", 2: "积极"}
        self.planning_hint = f"情感{sent_map.get(sentiment, '中性')}，与主题贴合度{theme_fit:.2f}，情绪变化{change_value:+.1f}"

        done = self.emo_point <= 0 or bool(self.GOODBYE_PATTERN.search(user_reply))
        return user_reply, done

    def reply(self, query: Optional[str]) -> Dict[str, str]:
        """
        对外接口：接收 NPC 的 query，返回用户回复。
        - query 为 None：生成开场白（用户先开口）
        - query 不为 None：把 NPC 回复加入对话，生成用户回复并更新 emo_point
        """
        if query is None:
            if not self.dialog:
                messages = self._messages_for_user_reply(planning_hint="")
                first = self.qwen_user_llm_fn(messages)
                first = (first or "").strip() or "我最近有些事想和你聊聊。"
                self.dialog.append({"role": "user", "content": first})
                return {"role": "user", "content": first}
            return {"role": "user", "content": self.dialog[-1]["content"]}
        user_reply, _ = self.step(query)
        return {"role": "user", "content": user_reply}

    def get_emo_point(self) -> float:
        return self.emo_point

    def get_emo_point_turns(self) -> List[float]:
        return list(self.emo_point_turns)

    def reset(self, initial_emo_point: Optional[float] = None) -> None:
        if initial_emo_point is not None:
            self.emo_point = max(0.0, min(100.0, initial_emo_point))
            self.emo_state = self._emo_point_to_state(self.emo_point)
        self.dialog = []
        self.emo_point_turns = [self.emo_point]
        self.planning_hint = ""


def build_player_simulator_dsv3(
    profile: Dict[str, Any],
    emo_adapter_path: str,
    qwen_api_key: Optional[str] = None,
    qwen_model: str = "qwen-plus",
    target: str = "eq",
    initial_emo_point: float = 50.0,
    device: Optional[Union[str, "torch.device"]] = None,
) -> PlayerSimulatorDSV3:
    """
    一站式构建 PlayerSimulatorDSV3：
    - 情感分析：EmoAnalyzer（emo_classifier_lora）
    - 用户回复：Qwen API
    """
    from .emo_analyzer import build_emo_analyzer, emo_analyzer_fn_from_analyzer

    analyzer = build_emo_analyzer(emo_adapter_path, device=device)
    emo_analyzer_fn = emo_analyzer_fn_from_analyzer(analyzer)
    qwen_fn = build_qwen_user_llm_fn(
        api_key=qwen_api_key or os.environ.get("DASHSCOPE_API_KEY"),
        model=qwen_model,
        temperature=0.7,
    )
    return PlayerSimulatorDSV3(
        profile=profile,
        emo_analyzer_fn=emo_analyzer_fn,
        qwen_user_llm_fn=qwen_fn,
        target=target,
        initial_emo_point=initial_emo_point,
    )


def generate_role(
    role_file: str,
    target: str = "eq",
    topic: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    从 train_profile.jsonl 随机抽取一个 role，构建 player_data。
    """
    if seed is not None:
        random.seed(seed)
    with open(role_file, "r", encoding="utf-8") as f:
        data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if topic is None or obj.get("topic") == topic:
                data.append(obj)
    if not data:
        raise ValueError(f"未找到符合条件的 profile，topic={topic}")
    role = random.sample(data, 1)[0]
    return {
        "id": role.get("id", ""),
        "player": role.get("player", ""),
        "scene": role.get("scene", ""),
        "task": role.get("task", ""),
        "topic": role.get("topic", ""),
        "main_cha": role.get("main_cha", ""),
        "cha_group": role.get("cha_group", []),
        "target": target,
    }
