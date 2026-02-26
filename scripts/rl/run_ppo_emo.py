# -*- coding: utf-8 -*-
"""
PPO 多轮对话训练入口：
- 从 data/data 加载用户形象，生成多轮对话（Actor + PlayerSimulatorWithPlanning）
- Reward 为函数：模式1（仅最终 emo_point/100）或模式2（baseline + trend - volatility），通过参数切换
- 使用 planning_reply（LLM prompt）分析用户情感并更新 emo_point
"""
from __future__ import annotations

import argparse
import os
import sys

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch as T
from torch.utils.data import DataLoader

from src.data.profile_dataset import ProfileDataset
from src.training.ppo_emo_rollout import run_multi_turn_rollout_batch
from src.training.reward_emo import compute_reward_tensors


def get_user_llm_fn(api: str = "mock"):
    """
    返回 user_llm_fn(messages) -> str。
    api=="mock": 简单占位，返回固定短句（便于先跑通流程）。
    api=="openai" 或其它: 可从环境变量读 API key 并调用对应接口，这里仅预留。
    """
    if api == "mock":

        def mock_fn(messages):
            # 模拟用户回复
            last = messages[-1] if messages else {}
            c = (last.get("content") or "").strip()
            if "请回复" in c or "NPC" in c:
                return "我最近工作压力很大，和家里人关系也紧张，不知道该怎么办。"
            if "建议" in c or "理解" in c:
                return "嗯，你说得对，我试试看。"
            return "谢谢你的建议，我再想想。再见。"

        return mock_fn
    # 可扩展：从 os.environ 读 OPENAI_API_KEY 等，调用 OpenAI / 其他 API
    return get_user_llm_fn("mock")


def main():
    parser = argparse.ArgumentParser(description="PPO 多轮对话（用户形象 + emo_point reward）")
    parser.add_argument("--data_dir", type=str, default="/home/yukiwang/xlwy/data/data")
    parser.add_argument("--sft_model_path", type=str, default=None, help="本地 SFT 模型路径，用于 planning 情感分析")
    parser.add_argument("--reward_mode", type=str, default="mode1", choices=["mode1", "mode2", "mode3"])
    parser.add_argument("--w1", type=float, default=1.0, help="mode2/mode3 baseline weight")
    parser.add_argument("--w2", type=float, default=0.3, help="mode2/mode3 trend weight")
    parser.add_argument("--w3", type=float, default=0.2, help="mode2/mode3 volatility penalty weight")
    parser.add_argument("--trend_n", type=int, default=5, help="mode2/mode3 recent n turns for trend/volatility")
    parser.add_argument("--S1", type=int, default=100, help="mode3: step 后开始 alpha warmup")
    parser.add_argument("--S2", type=int, default=300, help="mode3: step 后开始 beta warmup")
    parser.add_argument("--warmup_steps", type=int, default=200, help="mode3: alpha/beta 的 warmup 步数")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_new_tokens_per_turn", type=int, default=128)
    parser.add_argument("--user_llm", type=str, default="mock", help="mock or openai")
    parser.add_argument("--device", type=str, default="cuda" if T.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = T.device(args.device)

    # 1) 用户 LLM（模拟器用）
    user_llm_fn = get_user_llm_fn(args.user_llm)

    # 3) 数据集与 DataLoader
    dataset = ProfileDataset(args.data_dir, split="train")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    # 4) 加载 Actor + Critic（此处用占位，实际需接你的 SFT 模型与 Critic）
    # 若你已有 actor_ref_rollout 和 critic，可替换为：
    # from src.models.modeling import load_sft_model
    # mt = load_sft_model(...); actor_ref = ActorRefRollout(mt.model, mt.tokenizer, ...); critic = Critic(...)
    print("请在此脚本中接入你的 ActorRefRollout 与 Critic（如从 ppo_training 与 modeling 加载）。")
    print("当前仅演示：多轮 rollout 调用方式与 reward 计算。")
    actor_ref = None
    critic = None
    tokenizer = None

    if actor_ref is None or critic is None or tokenizer is None:
        print("未加载模型，退出。请先实现模型加载后取消下方 return。")
        return

    # 5) 跑一个 batch 演示（使用 PlayerSimulatorWithPlanning + planning_reply）
    batch_items = next(iter(dataloader))
    gen_batch = run_multi_turn_rollout_batch(
        batch_items=batch_items,
        actor_ref=actor_ref,
        critic=critic,
        tokenizer=tokenizer,
        user_llm_fn=user_llm_fn,
        device=device,
        max_turns=args.max_turns,
        max_new_tokens_per_turn=args.max_new_tokens_per_turn,
        sft_model_path=args.sft_model_path,
    )

    gen_batch["non_tensor_batch"] = gen_batch.get("non_tensor_batch") or {}
    gen_batch["non_tensor_batch"]["emo_point"] = gen_batch["non_tensor_batch"].get("emo_point", [0.0] * len(batch_items))
    gen_batch["non_tensor_batch"]["emo_point_turns"] = gen_batch["non_tensor_batch"].get("emo_point_turns", [])

    # 6) Reward 计算（函数，非 reward model）；mode3 需传 step（演示用 step=0）
    current_step = 0
    original_reward_tensor, penalized_reward_tensor = compute_reward_tensors(
        response_ids=gen_batch["response_ids"],
        response_mask=gen_batch["response_mask"],
        emo_points=gen_batch["non_tensor_batch"]["emo_point"],
        emo_point_turns_list=gen_batch["non_tensor_batch"].get("emo_point_turns"),
        reward_mode=args.reward_mode,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        trend_n=args.trend_n,
        device=device,
        step=current_step if args.reward_mode == "mode3" else None,
        S1=args.S1 if args.reward_mode == "mode3" else None,
        S2=args.S2 if args.reward_mode == "mode3" else None,
        warmup_steps=args.warmup_steps if args.reward_mode == "mode3" else None,
    )

    print("reward_mode:", args.reward_mode)
    print("original_reward_tensor shape:", original_reward_tensor.shape)
    print("penalized_reward_tensor shape:", penalized_reward_tensor.shape)
    print("emo_points:", gen_batch["non_tensor_batch"]["emo_point"])
    print("演示完成。将 gen_batch 与 (original_reward_tensor, penalized_reward_tensor) 交给 PPO trainer 即可。")


if __name__ == "__main__":
    main()
