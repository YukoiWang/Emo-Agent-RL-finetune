# -*- coding: utf-8 -*-
"""
基于 LLM prompt 的情感分析：用 planning_reply 模板分析 NPC 回复对演员情绪的影响，
输出 change_value 及 content/reason/activity/analyse 供 player_reply 使用。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

PLANNING_TEMPLATE = """你是一个emotion分析器，你擅长根据演员的画像和性格特征，侧写演员在对话时的感受。
# 演员的任务
*你是一个演员。你将根据剧本中的人物画像和对话背景扮演一个角色和NPC进行对话。
*你目的是在对话中扮演好人物画像和对话背景构成的角色
*你需要根据你实时变换的emotion，结合人物画像和对话背景中的相关定义，选择不同的对话策略，完成符合角色特征的回复

# 演员的对话目的
*{{target}}

# 你的任务
根据演员的人物画像、对话背景，结合对话上下文和演员当前的emotion，分析并侧写演员此刻对NPC回复的感受以及导致的emotion变化。

# 角色性格特征
演员具有鲜明的性格特征，你要始终根据人物画像和对话背景，代入演员的性格特征进行分析。
性格特征应该体现在：说话语气和方式，思维方式，感受变化等方面。

# emotion
emotion是一个0-100的数值，越高代表此时演员的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了演员是否享受、投入当前对话
emotion较高时，演员的感受和行为会偏向于正面
emotion较低时，演员的感受和行为会偏向于负面
emotion非常低时，演员会直接结束对话
你要结合角色性格和对话背景内定义的角色可能的反应分析emotion

# 分析维度
你需要代入演员的心理，对以下几个维度进行分析
*对NPC回复的客观分析：
1.根据最新对话中NPC回复，结合上下文，分析NPC想要表达的内容。
2.根据最新对话中NPC回复和隐藏主题，结合上下文和NPC表达的内容，哪些内容贴合了人物的隐藏主题？哪些内容可能不贴合，甚至可能引起人物的情绪波动？
*对NPC回复的主观分析：
3.根据人物画像中的角色性格特征以及对话背景中定义的不同emotion时的反应和隐藏主题，结合演员当前emotion值和客观分析，侧写描述演员当前的心理活动
4.根据对话背景中定义的演员可能的反应和隐藏主题，结合侧写得到的心理活动以及对NPC回复的客观分析，详细地侧写演员此刻对NPC回复的感受（如果NPC的回复不是自然语言（如乱码，夹杂大量符号），则你的感受很负面）
5.结合前几步分析，并用一个正负值来表示演员的emotion变化

# 输出内容：
1.NPC想要表达的内容
2.NPC回复与隐藏主题的贴合程度分析
3.演员当前的心理活动
4.演员对NPC回复的感受
5.用一个正负值来表示演员的emotion变化(注意，你只用输出值，不用输出原因或者描述)

# 输出格式:
Content:
[NPC想要表达的内容]
Reason:
[NPC回复与隐藏主题的贴合程度分析]
Activity:
[心理活动]
Analyse:
[演员对NPC回复的感受]
Change:
[演员的emotion变化]


#人物画像
{{player_type}}

#当前对话背景：
{{player_topic}}

**演员当前的情绪是{{emotion}}

**这是上下文内容
{{dialog_history}}

**这是演员和NPC的最新对话
{{new_history}}
"""


def _parse_planning_reply(reply: str) -> Dict[str, Any]:
    """解析 LLM 返回的 planning 文本，提取 Content/Reason/Activity/Analyse/Change。"""
    reply = (reply or "").replace("：", ":").replace("*", "").strip()
    planning: Dict[str, Any] = {
        "content": "",
        "reason": "",
        "activity": "",
        "analyse": "",
        "change": 0,
    }
    try:
        if "Content:" in reply:
            planning["content"] = reply.split("Content:")[-1].split("Reason:")[0].strip().strip("[]").replace("\n\n", "\n")
        if "Reason:" in reply:
            planning["reason"] = reply.split("Reason:")[-1].split("Activity:")[0].strip().strip("[]").replace("\n\n", "\n")
        if "Activity:" in reply:
            planning["activity"] = reply.split("Activity:")[-1].split("Analyse:")[0].strip().strip("[]").replace("\n\n", "\n")
        if "Analyse:" in reply:
            planning["analyse"] = reply.split("Analyse:")[-1].split("Change:")[0].strip().strip("[]").replace("\n\n", "\n")
        if "Change:" in reply:
            raw = reply.split("Change:")[-1].strip()
            if "变化" in raw:
                raw = raw.split("\n")[-1].strip()
            raw = raw.split("\n")[0].strip().strip("[]").strip("""").strip(""").strip()
            # 提取数字（支持正负）
            nums = re.findall(r"-?\d+", raw)
            planning["change"] = int(nums[0]) if nums else 0
    except Exception:
        pass
    return planning


def call_planning_llm(
    prompt: str,
    llm_fn: Callable[[List[Dict[str, str]]], str],
) -> str:
    """调用 LLM 执行 planning 分析。"""
    messages = [{"role": "user", "content": prompt}]
    return llm_fn(messages)


def _get_target_text(target: str) -> str:
    from .qwen_user_simulator import TARGET_PROMPT
    return TARGET_PROMPT.get(target, TARGET_PROMPT["eq"])


def build_planning_prompt(
    player_type: str,
    player_topic: str,
    target: str,
    emotion: float,
    dialog_history: List[Dict[str, str]],
    new_history: List[Dict[str, str]],
    mapping: Optional[Dict[str, str]] = None,
) -> str:
    """
    根据模板构建 planning 的 prompt。
    target: "eq" | "no-target" | "target" | "test"，会替换为对应的完整描述
    dialog_history: 除最新两轮外的历史
    new_history: 最新两轮（含 NPC 回复）
    """
    mapping = mapping or {"user": "你", "assistant": "NPC"}
    history_str = json.dumps(
        [{"role": mapping.get(m["role"], m["role"]), "content": m["content"]} for m in dialog_history],
        ensure_ascii=False,
        indent=2,
    )
    new_str = json.dumps(
        [{"role": mapping.get(m["role"], m["role"]), "content": m["content"]} for m in new_history],
        ensure_ascii=False,
        indent=2,
    )
    target_text = _get_target_text(target)
    return PLANNING_TEMPLATE.replace("{{player_type}}", player_type) \
        .replace("{{player_topic}}", player_topic) \
        .replace("{{target}}", target_text) \
        .replace("{{emotion}}", str(int(emotion))) \
        .replace("{{dialog_history}}", history_str) \
        .replace("{{new_history}}", new_str)


def planning_reply(
    player_data: Dict[str, Any],
    llm_fn: Callable[[List[Dict[str, str]]], str],
    target_prompt: str = "eq",
    max_retries: int = 3,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    执行 planning_reply：分析 NPC 回复对演员情绪的影响。
    返回 (updated_player_data, planning_dict)。
    planning_dict 包含: content, reason, activity, analyse, change (int)。
    """
    history = player_data.get("history", [])
    emo_point = float(player_data.get("emo_point", 50))
    player_type = player_data.get("player", "")
    player_topic = player_data.get("scene", "")
    target = player_data.get("target", target_prompt)

    if len(history) < 2:
        return player_data, {"content": "", "reason": "", "activity": "", "analyse": "请你以一个简短的回复开启倾诉", "change": 0}

    dialog_history = history[:-2]
    new_history = history[-2:]

    prompt = build_planning_prompt(
        player_type=player_type,
        player_topic=player_topic,
        target=target,
        emotion=emo_point,
        dialog_history=dialog_history,
        new_history=new_history,
    )

    reply = None
    for attempt in range(max_retries):
        try:
            reply = call_planning_llm(prompt, llm_fn)
            if reply:
                break
        except Exception as e:
            if attempt + 1 >= max_retries:
                raise
            time.sleep(2 ** attempt)

    planning = _parse_planning_reply(reply or "")
    change = int(planning.get("change", 0))
    emo_point = max(0.0, min(100.0, emo_point + change))

    out = dict(player_data)
    out["emo_point"] = emo_point
    return out, planning


def build_planning_emo_analyzer_fn(
    llm_fn: Callable[[List[Dict[str, str]]], str],
    target_prompt: str = "eq",
) -> Callable[..., Dict[str, Any]]:
    """
    构建兼容 emo_analyzer_fn 接口的 planning 版本。
    注意：标准接口是 (npc_reply, user_reply, hidden_theme)，但 planning 需要 profile/dialog/emo_point。
    本函数返回一个需要 (profile, dialog, emo_point) 的 callable，供 PlayerSimulatorWithPlanning 使用。
    返回格式与 emo_analyzer_fn 一致：{change_value, sentiment?, theme_fit?, planning?}
    """

    def fn(
        profile: Dict[str, Any],
        dialog: List[Dict[str, str]],
        emo_point: float,
    ) -> Dict[str, Any]:
        player_data = {
            "player": profile.get("player", ""),
            "scene": profile.get("scene", ""),
            "task": profile.get("task", ""),
            "history": dialog,
            "emo_point": emo_point,
            "target": profile.get("target", target_prompt),
        }
        _, planning = planning_reply(player_data, llm_fn, target_prompt=target_prompt)
        change = int(planning.get("change", 0))
        return {
            "change_value": float(change),
            "planning": planning,
        }

    return fn
