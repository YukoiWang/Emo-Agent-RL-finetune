# -*- coding: utf-8 -*-
"""
Smoke test: 验证 PlayerSimulatorWithPlanning 能否跑通。
用 Qwen API 做 planning + player_reply（不需要 GPU）。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.qwen_user_simulator import build_qwen_user_llm_fn
from src.training.hard_player_simulator_dsv3 import build_player_simulator_with_planning


def load_one_profile(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError("empty profile file")


def main():
    data_path = os.path.join(ROOT, "data/data/train_profile.jsonl")
    profile = load_one_profile(data_path)
    print(f"[OK] 加载 profile: id={profile.get('id', '?')}")
    print(f"     task: {profile.get('task', '')[:60]}...")

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("[FAIL] DASHSCOPE_API_KEY 未设置")
        return

    llm_fn = build_qwen_user_llm_fn(api_key=api_key, model="qwen-plus")
    print("[OK] build_qwen_user_llm_fn 成功")

    sim = build_player_simulator_with_planning(
        profile=profile,
        player_llm_fn=llm_fn,
        target="eq",
        initial_emo_point=50.0,
    )
    print("[OK] build_player_simulator_with_planning 成功")

    print("\n--- 生成首条用户消息 ---")
    first = sim.reply(None)
    print(f"  首条: {first['content'][:120]}")
    print(f"  emo_point: {sim.get_emo_point()}")

    npc_replies = [
        "听起来你最近有些心事，能跟我说说吗？我在这里陪着你。",
        "我能理解你的感受，这种纠结确实让人很不好受。你有没有想过，什么才是你真正想要的？",
    ]
    for i, npc in enumerate(npc_replies, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"  NPC: {npc[:80]}")
        result = sim.reply(npc)
        print(f"  用户: {result['content'][:120]}")
        print(f"  emo_point: {sim.get_emo_point()}")
        print(f"  emo_turns: {sim.get_emo_point_turns()}")

    print("\n[DONE] smoke test 通过！")


if __name__ == "__main__":
    main()
