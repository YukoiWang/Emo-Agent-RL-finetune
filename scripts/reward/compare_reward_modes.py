#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速对比三种 reward 模式的效果。

用法：
  python scripts/reward/compare_reward_modes.py                    # 离线对比（秒级，无需 GPU）
  python scripts/reward/compare_reward_modes.py --quick-ppo        # 各跑 20 步 PPO 对比（需 GPU，约 2–5 分钟）
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def run_offline_comparison(emo_adapter_path: Optional[str] = None) -> None:
    """离线对比：对若干示例回复分别计算 mode1/mode2/mode3 的 reward，不打 PPO。"""
    from src.training.reward_emo import (
        build_reward_fn_emo,
        _trend_reward,
        _volatility_penalty,
    )

    # 示例回复（覆盖不同情绪倾向）
    samples = [
        ("高共情", "我能理解你的感受，说出来真的很不容易，谢谢你的信任。"),
        ("中等", "嗯，这种情况确实让人不太好受。"),
        ("低共情", "无所谓，别在乎这些小事。"),
        ("消极词", "不重要，你自己看着办吧。"),
        ("中性", "好的，我知道了。"),
    ]

    # 构建三种 reward 函数（无 emo_adapter 时用关键词规则，速度快）
    step_counter = [0]

    def step_fn():
        step_counter[0] += 1
        return step_counter[0]

    fn1 = build_reward_fn_emo(reward_mode="mode1", emo_adapter_path=emo_adapter_path)
    fn2 = build_reward_fn_emo(reward_mode="mode2", emo_adapter_path=emo_adapter_path)
    fn3_early = build_reward_fn_emo(
        reward_mode="mode3",
        emo_adapter_path=emo_adapter_path,
        step_fn=lambda: 50,   # step < S1=100, alpha=0, beta=0
        S1=100, S2=300, warmup_steps=200,
    )
    fn3_mid = build_reward_fn_emo(
        reward_mode="mode3",
        emo_adapter_path=emo_adapter_path,
        step_fn=lambda: 150,  # S1 < step < S2, alpha≈0.25, beta=0
        S1=100, S2=300, warmup_steps=200,
    )
    fn3_late = build_reward_fn_emo(
        reward_mode="mode3",
        emo_adapter_path=emo_adapter_path,
        step_fn=lambda: 400,  # step > S2, alpha=1, beta≈0.5
        S1=100, S2=300, warmup_steps=200,
    )

    texts = [t[1] for t in samples]
    r1 = fn1(texts)
    r2 = fn2(texts)
    r3e = fn3_early(texts)
    r3m = fn3_mid(texts)
    r3l = fn3_late(texts)

    print("\n" + "=" * 90)
    print("【1】单轮回复 reward 对比（当前 reward_fn 对每条回复独立打分）")
    print("=" * 90)
    print(f"{'标签':<10} {'mode1':<10} {'mode2':<10} {'mode3(早期)':<12} {'mode3(中期)':<12} {'mode3(后期)':<12} | 示例")
    print("-" * 90)
    for i, (label, _) in enumerate(samples):
        print(f"{label:<10} {_fmt(r1[i]):<10} {_fmt(r2[i]):<10} {_fmt(r3e[i]):<12} {_fmt(r3m[i]):<12} {_fmt(r3l[i]):<12} | {samples[i][1][:30]}...")
    print()

    # 多轮模拟：直接展示 trend/volatility 如何影响 mode2 vs mode3
    print("=" * 90)
    print("【2】多轮情绪序列模拟（trend/volatility 生效时的差异）")
    print("=" * 90)
    scenarios = [
        ("情绪改善", [30.0, 45.0, 55.0, 70.0, 80.0]),
        ("情绪下降", [75.0, 65.0, 50.0, 40.0, 35.0]),
        ("高波动", [50.0, 85.0, 35.0, 90.0, 55.0]),
        ("平稳", [55.0, 56.0, 54.0, 55.0, 56.0]),
    ]
    w1, w2, w3, trend_n = 1.0, 0.3, 0.2, 5
    print(f"权重: w1={w1}, w2={w2}, w3={w3}, trend_n={trend_n}")
    print()
    print(f"{'场景':<12} {'baseline':<10} {'trend':<10} {'vol_penalty':<12} {'mode1':<10} {'mode2':<10} {'mode3(S=400)':<12}")
    print("-" * 90)
    for name, turns in scenarios:
        baseline = max(0.0, turns[-1] / 100.0)
        tr = _trend_reward(turns, n=trend_n)
        vp = _volatility_penalty(turns, n=trend_n)
        r1_val = baseline
        r2_val = max(0.0, min(1.0, w1 * baseline + w2 * tr - w3 * vp))
        alpha = min(1.0, max(0.0, (400 - 100) / 200))
        beta = min(1.0, max(0.0, (400 - 300) / 200))
        r3_val = max(0.0, min(1.0, baseline + alpha * w2 * tr - beta * w3 * vp))
        print(f"{name:<12} {_fmt(baseline):<10} {_fmt(tr):<10} {_fmt(vp):<12} {_fmt(r1_val):<10} {_fmt(r2_val):<10} {_fmt(r3_val):<12}")
    print()
    print("说明: mode1 只看最终情绪; mode2 始终考虑 trend/volatility; mode3 随 step warmup 逐步加入。")
    print()


def run_quick_ppo_comparison(config_path: str) -> None:
    """各跑 20 步 PPO，对比三种 reward 的 loss/reward 曲线。"""
    import yaml
    from src.training.ppo_emo_trainer import run_ppo_emo_training

    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 强制小步数
    cfg.setdefault("training", {})["total_steps"] = 20
    cfg.setdefault("training", {})["logging_steps"] = 5
    cfg.setdefault("reward", {})["type"] = "emo"

    for mode in ["mode1", "mode2", "mode3"]:
        cfg["reward"]["reward_mode"] = mode
        out = cfg["training"].get("output_dir", "outputs/rl")
        cfg["training"]["output_dir"] = f"{out}_compare_{mode}"
        print(f"\n{'='*60}\n>>> 运行 PPO reward_mode={mode}, total_steps=20\n{'='*60}")
        run_ppo_emo_training(cfg)
    print("\n完成。查看各 output 目录下的 trainer_state.json 或日志对比 loss/reward。")


def main() -> None:
    parser = argparse.ArgumentParser(description="快速对比三种 reward 模式")
    parser.add_argument("--quick-ppo", action="store_true", help="各跑 20 步 PPO 对比（较慢，需 GPU）")
    parser.add_argument("--config", type=str, default="configs/rl_default.yaml", help="PPO 配置文件（仅 --quick-ppo 时用）")
    parser.add_argument("--emo-adapter", type=str, default=None, help="情感 adapter 路径（可选，用于更准确打分）")
    args = parser.parse_args()

    if args.quick_ppo:
        run_quick_ppo_comparison(args.config)
    else:
        run_offline_comparison(emo_adapter_path=args.emo_adapter or None)


if __name__ == "__main__":
    main()
