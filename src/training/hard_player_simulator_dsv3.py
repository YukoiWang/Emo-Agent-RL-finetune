# -*- coding: utf-8 -*-
"""
用户模拟器（PlayerSimulator）：根据 NPC 回复与隐藏主题的贴合程度、用户情感分析更新 emo_point，
并与 actor 进行多轮对话。说「再见/拜拜」或 emo_point <= 0 时结束对话。

支持两种情感分析方式：
- EmoAnalyzer（分类模型）：emo_analyzer_fn(npc_reply, user_reply, hidden_theme)
- Planning（LLM prompt）：先 planning_reply 分析情绪变化，再 player_reply 生成用户回复
"""
from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# 用户提供的 player_reply 模板（用于 planning 模式的用户回复生成）
# 输出格式：Thinking / Origin / Change / Response，需解析 Response 为最终回复
# ---------------------------------------------------------------------------
PLAYER_REPLY_TEMPLATE = """你是一个演员。你将根据剧本中的人物画像和对话背景扮演角色和一个NPC进行对话。

# 你的任务 
*你目的是在对话中扮演好人物画像和对话背景构成的角色
*你需要根据你实时变换的emotion，结合人物画像和对话背景中的相关定义，选择不同的对话策略，完成符合角色特征的回复

# 你的对话目的
*{{target}}

# Emotion
*你会收到当前你的emotion，你一共有5档emotion，越高代表此时你的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了演员是否享受、投入当前对话
*emotion影响你不同的说话风格和语气以及回应方式，根据不同emotion按照对话背景内定义的角色不同emotion下的反应进行回复：
Emotion-S: 你的emotion已经达到了最高，你可以感谢NPC并用「再见」或「拜拜」告别，直接结束对话。
Emotion-A：高emotion，此时你的对话感受较为正面，你的反馈也比较正面。
Emotion-B：中emotion，此时你没有正面或者负面的感受。
Emotion-C：低emotion，此时你的对话感受较为负面，你的反馈也比较负面。
Emotion-F：你的情绪已到达最负面，不想继续对话。此时你要用「再见」或「拜拜」告别，直接结束对话。

# 你应该区分Emotion和对NPC最新回复感受，Emotion代表你的当前的对话情绪，对NPC回复的感受代表你对NPC回复的即时感受，你需要结合两者生成回复。

# 回复思路
*你会收到当前你对NPC最新回复的详细感受，包含客观分析部分和主观分析部分，你要结合人物画像、对话背景、隐藏主题和详细感受来分析，并决定回复内容。
*分析内容，应该包含以下5个维度：
1.根据你的详细感受和当前Emotion，结合隐藏主题，当前的回复态度偏向应该是正面、无偏向还是负面？
2.根据你的详细感受和当前Emotion，结合隐藏主题，你的本次回复目标应该是？（注意，你不需要针对NPC的每一句话做出回应，你不可以主动泄露隐藏主题）
3.根据人物画像中说话风格的相关定义，结合对话背景内定义的角色不同emotion下的反应和你的回复态度以及回复目标，你的说话语气、风格应该是？
4.根据人物画像和对话背景以及隐藏主题，结合你的详细感受以及前三轮分析，你的说话方式和内容应该是？（注意：如果根据人设你是被动型，则你的说话方式应该是被动、不主动提问）
*回复内容，根据分析结果生成初始回复，回复内容要尽可能简洁，不要一次包含过多信息量。
*改造内容，你需要参照下述规则改造你的回复让其更真实，从而得到最终回复：
1.你需要说话简洁，真实的回复一般不会包含太长的句子
2.真实的回复不会直接陈述自己的情绪，而是将情绪蕴含在回复中，用语气表达自己的情绪
3.你绝对不可以使用「我真的觉得……」「我真的不知道……」「我真的快撑不住了」这些句子，你不应该用「真的」、「根本」来表述你的情绪
4.真实的回复不会重复自己在对话上下文中说过的信息
5.你不应该生成和对话上下文中相似的回复

# 输出内容：
*你需要按照回复思路中的分析版块，首先进行5个维度分析
*然后你需要**逐步**按照分析内容并遵循注意事项生成初始回复，回复中的信息量来源于对话背景和你的联想，你不应该一次性谈论太多事件或内容
*随后你需要根据改造内容分析你应该如何针对初始回复进行改造
*最后你需要根据分析改造初始回复生成最终回复

# 输出格式:
Thinking:
[分析内容]
Origin:
[初始回复]
Change:
[改造分析]
Response:
[最终回复]


# 发言风格
你的发言需要严格遵守「玩家画像」中描述的人物设定和背景。
你的性格和发言风格要遵循「习惯和行为特点」的描述
如果发言要符合你的人物形象，比如负面的人物形象需要你进行负面的发言。
你的语气要符合你的年龄

* 你的发言要遵守以下5条准则
1. 发言必须简洁、随意、自然,按照自然对话进行交流。
2. 不许一次提问超过两个问题。
3. 不允许重复之前说过的回复或者进行相似的回复。
4. 在发言时，可以自然的使用一些口语化词汇
5. 你的发言应该精简，不准过长


#人物画像：
{{player_type}}

#当前对话背景：
{{player_topic}}

**这是上下文内容
{{dialog_history}}

**这是你和NPC的最新对话
{{new_history}}

**这是你对NPC最新回复的详细感受
{{planning}}

**这是你当前的Emotion
{{emotion}}

你生成的[回复]部分不允许和历史记录过于相似，不许过长，不许主动转移话题。
"""


def _parse_player_reply_response(raw: str) -> str:
    """从 player_reply 的 LLM 输出中解析 Response 部分作为最终回复。"""
    if not raw or "Response:" not in raw:
        return (raw or "").strip() or "…"
    reply = raw.split("Response:")[-1].strip()
    reply = reply.strip("\n").strip("[]").strip(""").strip(""").strip()
    return reply or "…"


class PlayerSimulatorWithPlanning:
    """
    使用 planning_reply（LLM prompt）做情感分析的用户模拟器。
    流程：NPC 回复 → planning_reply（本地 SFT 基座）→ player_reply（API）→ 用户回复。
    支持分离：planning_llm_fn（情感分析，本地模型）与 player_llm_fn（用户回复，API）。
    与 PlayerSimulator 兼容：step() 返回 (user_reply, done)，get_emo_point() 等接口一致。
    """

    GOODBYE_PATTERN = re.compile(r"(再见|拜拜|再会|下次聊|先这样)")

    EMO_COUNT = {"Emotion-S": 100, "Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}

    def __init__(
        self,
        profile: Dict[str, Any],
        player_llm_fn: Callable[[List[Dict[str, str]]], str],
        planning_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        target: str = "eq",
        initial_emo_point: float = 50.0,
    ):
        """
        profile: 至少包含 player, scene, task
        player_llm_fn: 用于 player_reply 和 generate_first_message（通常为 API，如 Qwen）
        planning_llm_fn: 用于 planning_reply（情感分析，通常为本地 SFT 基座）
        llm_fn: 兼容旧接口，若提供则同时用于 planning 和 player（planning_llm_fn/player_llm_fn 优先）
        target: 对话目的，对应 TARGET_PROMPT 的 key
        """
        self.profile = profile
        self.player_llm_fn = player_llm_fn or llm_fn
        self.planning_llm_fn = planning_llm_fn or self.player_llm_fn
        self.target = target
        self.emo_point = max(0.0, min(100.0, float(initial_emo_point)))
        self.dialog: List[Dict[str, str]] = []
        self.emo_point_turns: List[float] = [self.emo_point]
        self._target_text = self._get_target_text()

    def _get_target_text(self) -> str:
        from .qwen_user_simulator import TARGET_PROMPT
        return TARGET_PROMPT.get(self.target, TARGET_PROMPT["eq"])

    def _emo_point_to_state(self, point: float) -> str:
        for state, threshold in sorted(self.EMO_COUNT.items(), key=lambda x: -x[1]):
            if point >= threshold:
                return state
        return "Emotion-F"

    def _build_player_reply_prompt(
        self,
        planning: Dict[str, Any],
    ) -> str:
        player_type = self.profile.get("player", "")
        player_topic = self.profile.get("scene", "")
        mapping = {"user": "你", "assistant": "NPC"}
        if not self.dialog:
            dialog_str = "对话开始，你是玩家，请你先发起话题，用简短的回复开启倾诉"
            new_str = ""
        else:
            history = self.dialog[:-2] if len(self.dialog) >= 2 else []
            new_hist = self.dialog[-2:] if len(self.dialog) >= 2 else self.dialog
            dialog_str = json.dumps(
                [{"role": mapping.get(m["role"], m["role"]), "content": m["content"]} for m in history],
                ensure_ascii=False,
                indent=2,
            )
            new_str = json.dumps(
                [{"role": mapping.get(m["role"], m["role"]), "content": m["content"]} for m in new_hist],
                ensure_ascii=False,
                indent=2,
            )
        if not planning or (not planning.get("reason") and not planning.get("analyse")):
            planning_str = planning.get("analyse", "请你以一个简短的回复开启倾诉")
        else:
            planning_str = (
                f"对NPC回复的客观分析：\n{planning.get('reason', '')}\n"
                f"对NPC回复的主观分析：\n{planning.get('analyse', '')}"
            )
        emo_state = self._emo_point_to_state(self.emo_point)
        return (
            PLAYER_REPLY_TEMPLATE.replace("{{player_type}}", player_type)
            .replace("{{player_topic}}", player_topic)
            .replace("{{target}}", self._target_text)
            .replace("{{dialog_history}}", dialog_str)
            .replace("{{new_history}}", new_str)
            .replace("{{planning}}", planning_str)
            .replace("{{emotion}}", emo_state)
        )

    def step(self, npc_reply: str) -> Tuple[str, bool]:
        """给定 NPC 回复，先 planning 分析情绪，再生成用户回复。"""
        from .emo_planning import planning_reply

        npc_reply = (npc_reply or "").strip()
        if not npc_reply:
            return "（请继续说。）", False

        self.dialog.append({"role": "assistant", "content": npc_reply})

        player_data = {
            "player": self.profile.get("player", ""),
            "scene": self.profile.get("scene", ""),
            "task": self.profile.get("task", ""),
            "history": list(self.dialog),
            "emo_point": self.emo_point,
            "target": self.target,
        }

        player_data, planning = planning_reply(player_data, self.planning_llm_fn, target_prompt=self.target)
        self.emo_point = player_data["emo_point"]
        self.emo_point_turns.append(self.emo_point)

        prompt = self._build_player_reply_prompt(planning)
        raw_reply = self.player_llm_fn([{"role": "user", "content": prompt}])
        user_reply = _parse_player_reply_response(raw_reply or "")
        self.dialog.append({"role": "user", "content": user_reply})

        done = self.emo_point <= 0 or bool(self.GOODBYE_PATTERN.search(user_reply))
        return user_reply, done

    def get_emo_point(self) -> float:
        return self.emo_point

    def get_emo_point_turns(self) -> List[float]:
        return list(self.emo_point_turns)

    def _build_system_and_start(self) -> List[Dict[str, str]]:
        """兼容 ppo_emo_rollout：返回系统提示。"""
        return [{"role": "system", "content": f"你是演员，按人设和情绪回复。\n【人设】\n{self.profile.get('player', '')}\n【背景】\n{self.profile.get('scene', '')}"}]

    def generate_first_message(self) -> str:
        """生成开场白（首轮用户发言）。"""
        planning = {"analyse": "请你以一个简短的回复开启倾诉"}
        prompt = self._build_player_reply_prompt(planning)
        raw = self.player_llm_fn([{"role": "user", "content": prompt}])
        return _parse_player_reply_response(raw or "") or "我最近有些事想和你聊聊。"

    def reply(self, query: Optional[str]) -> Dict[str, str]:
        """
        对外接口，与用户提供的 chat_player 流程一致。
        query=None: 生成开场白（用户先开口）
        query 不为空: 把 NPC 回复加入 history，执行 planning_reply + player_reply，返回用户回复
        """
        if query is None:
            if not self.dialog:
                first = self.generate_first_message()
                self.dialog.append({"role": "user", "content": first})
                return {"role": "user", "content": first}
            return {"role": "user", "content": self.dialog[-1]["content"]}
        user_reply, _ = self.step(query)
        return {"role": "user", "content": user_reply}

    def reset(self, initial_emo_point: Optional[float] = None) -> None:
        if initial_emo_point is not None:
            self.emo_point = max(0.0, min(100.0, initial_emo_point))
        self.dialog = []
        self.emo_point_turns = [self.emo_point]


def build_player_simulator_with_planning(
    profile: Dict[str, Any],
    player_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    planning_llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    sft_model_path: Optional[str] = None,
    llm_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    api_key: Optional[str] = None,
    model: str = "qwen-plus",
    target: str = "eq",
    initial_emo_point: float = 50.0,
    device: Optional[str] = None,
) -> PlayerSimulatorWithPlanning:
    """
    一站式构建 PlayerSimulatorWithPlanning。
    - planning（情感分析）：sft_model_path 指定时用本地 SFT 基座，否则用 player_llm_fn/llm_fn
    - player_reply / 首条消息：player_llm_fn 或 Qwen API（需 DASHSCOPE_API_KEY）
    """
    if player_llm_fn is None and llm_fn is None:
        from .qwen_user_simulator import build_qwen_user_llm_fn
        player_llm_fn = build_qwen_user_llm_fn(api_key=api_key, model=model)
    else:
        player_llm_fn = player_llm_fn or llm_fn
    if sft_model_path:
        from .local_planning_llm import build_local_planning_llm_fn
        planning_llm_fn = build_local_planning_llm_fn(sft_model_path, device=device)
    return PlayerSimulatorWithPlanning(
        profile=profile,
        player_llm_fn=player_llm_fn,
        planning_llm_fn=planning_llm_fn,
        target=target,
        initial_emo_point=initial_emo_point,
    )
