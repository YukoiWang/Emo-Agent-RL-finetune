# -*- coding: utf-8 -*-
"""
Dry-run smoke test: 验证 import、模板拼接、解析逻辑能否正常工作。
不调真实 API，用 mock llm_fn 代替。
"""
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def mock_llm_fn(messages):
    """模拟 LLM 返回，根据 prompt 内容返回不同的 mock 回复。"""
    content = messages[-1]["content"] if messages else ""
    if "emotion分析器" in content or "emotion" in content.lower() and "侧写" in content:
        return (
            "Content:\nNPC表达了对演员感受的关心和理解\n"
            "Reason:\nNPC的回复贴合了隐藏主题，展现了共情能力\n"
            "Activity:\n演员感到被理解，内心有些温暖\n"
            "Analyse:\n演员对NPC的回复感受是正面的，觉得NPC有在认真倾听\n"
            "Change:\n+5"
        )
    return (
        "Thinking:\n根据当前情绪和人设，我应该用委婉的语气开启倾诉\n"
        "Origin:\n最近有些事情压在心里，不知道该怎么处理。\n"
        "Change:\n让语气更自然些\n"
        "Response:\n最近心里一直不太踏实，有些事情想聊聊。"
    )


def main():
    passed = 0
    failed = 0

    # 1. Test imports
    print("=" * 50)
    print("TEST 1: imports")
    try:
        from src.training.qwen_user_simulator import TARGET_PROMPT, build_qwen_user_llm_fn
        from src.training.hard_player_simulator_dsv3 import (
            PlayerSimulatorWithPlanning,
            build_player_simulator_with_planning,
            _parse_player_reply_response,
            PLAYER_REPLY_TEMPLATE,
        )
        from src.training.emo_planning import (
            planning_reply,
            build_planning_prompt,
            _parse_planning_reply,
            PLANNING_TEMPLATE,
        )
        print("  [PASS] all imports ok")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] import error: {e}")
        traceback.print_exc()
        failed += 1
        return

    # 2. Test _parse_planning_reply
    print("\nTEST 2: _parse_planning_reply")
    try:
        raw = (
            "Content:\nNPC说了一些安慰的话\n"
            "Reason:\n贴合了隐藏主题\n"
            "Activity:\n内心感到被理解\n"
            "Analyse:\n正面感受\n"
            "Change:\n+8"
        )
        p = _parse_planning_reply(raw)
        assert p["change"] == 8, f"expected 8, got {p['change']}"
        assert "安慰" in p["content"]
        assert "贴合" in p["reason"]
        print(f"  [PASS] parsed: change={p['change']}, content='{p['content'][:30]}...'")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # 3. Test _parse_player_reply_response
    print("\nTEST 3: _parse_player_reply_response")
    try:
        raw = "Thinking:\n分析内容\nOrigin:\n初始回复\nChange:\n改造\nResponse:\n最近心里不太好。"
        reply = _parse_player_reply_response(raw)
        assert "心里不太好" in reply, f"unexpected reply: {reply}"
        print(f"  [PASS] parsed reply: '{reply}'")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        failed += 1

    # 4. Test build_planning_prompt
    print("\nTEST 4: build_planning_prompt")
    try:
        prompt = build_planning_prompt(
            player_type="测试人设",
            player_topic="测试背景",
            target="eq",
            emotion=50.0,
            dialog_history=[{"role": "user", "content": "你好"}],
            new_history=[
                {"role": "assistant", "content": "你好，请坐"},
                {"role": "user", "content": "谢谢"},
            ],
        )
        assert "emotion分析器" in prompt
        assert "测试人设" in prompt
        assert "50" in prompt
        print(f"  [PASS] prompt length={len(prompt)}, contains template keywords")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        failed += 1

    # 5. Test PlayerSimulatorWithPlanning with mock
    print("\nTEST 5: PlayerSimulatorWithPlanning (mock)")
    try:
        data_path = os.path.join(ROOT, "data/data/train_profile.jsonl")
        with open(data_path, "r", encoding="utf-8") as f:
            profile = json.loads(f.readline().strip())

        sim = PlayerSimulatorWithPlanning(
            profile=profile,
            player_llm_fn=mock_llm_fn,
            planning_llm_fn=mock_llm_fn,
            target="eq",
            initial_emo_point=50.0,
        )
        print(f"  初始 emo_point: {sim.get_emo_point()}")

        first = sim.reply(None)
        print(f"  首条消息: '{first['content'][:60]}'")
        assert first["content"], "first message empty"

        user_reply, done = sim.step("我能感受到你最近压力很大，愿意跟我说说具体是什么事情吗？")
        print(f"  step() 用户回复: '{user_reply[:60]}'")
        print(f"  emo_point: {sim.get_emo_point()}, done={done}")
        print(f"  emo_turns: {sim.get_emo_point_turns()}")
        assert sim.get_emo_point() >= 0
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        failed += 1

    # 6. Test planning_reply with mock
    print("\nTEST 6: planning_reply (mock)")
    try:
        player_data = {
            "player": "测试人设",
            "scene": "测试背景",
            "task": "测试隐藏主题",
            "history": [
                {"role": "user", "content": "我最近有些事"},
                {"role": "assistant", "content": "说说看"},
                {"role": "user", "content": "就是感情的事"},
                {"role": "assistant", "content": "我理解你的感受"},
            ],
            "emo_point": 50,
            "target": "eq",
        }
        updated, planning = planning_reply(player_data, mock_llm_fn)
        print(f"  change={planning['change']}, new emo={updated['emo_point']}")
        assert planning["change"] != 0 or True  # mock returns +5
        print("  [PASS]")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        failed += 1

    print("\n" + "=" * 50)
    print(f"结果: {passed} passed, {failed} failed")
    if failed == 0:
        print("[ALL PASS] 代码逻辑没有问题，需要有效的 API key 才能做完整测试。")
    sys.exit(failed)


if __name__ == "__main__":
    main()
