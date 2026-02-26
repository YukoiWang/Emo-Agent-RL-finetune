# -*- coding: utf-8 -*-
"""
On-policy DPO 训练：用 train_profile 模拟用户，每轮生成 k 个回复，情感打分选 best/worst 构造偏好对，再 DPO 训练。
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import torch
from datasets import Dataset
from trl import DPOConfig, DPOTrainer

from src.data.profile_dataset import load_profiles, build_initial_prompt
from src.models.modeling import load_sft_model, ModelAndTokenizer
from src.training.dpo_emo_rollout import run_dpo_rollout_batch
from src.training.emo_planning import score_responses_with_planning
from src.training.reward_emo import build_reward_fn_emo


class _MockUserSimulator:
    """无 API 调用的简单用户模拟器，用于快速测试。"""

    def __init__(self):
        self.dialog: List[Dict[str, str]] = []
        self._turn = 0
        self._canned = [
            "我最近有些事想和你聊聊。",
            "谢谢你的理解，我确实有这方面的困扰。",
            "嗯，你说得对，我再想想。",
            "好的，我明白了。谢谢你的建议，再见。",
        ]

    def generate_first_message(self) -> str:
        return self._canned[0]

    def step(self, npc_reply: str) -> tuple:
        self._turn += 1
        if self._turn >= len(self._canned) or "再见" in (npc_reply or ""):
            return self._canned[-1], True
        return self._canned[min(self._turn, len(self._canned) - 1)], False


def run_dpo_emo_training(cfg: Dict[str, Any]) -> None:
    """
    On-policy DPO：用当前 policy 生成多轮对话，每轮 k 个回复 → reward 选 best/worst → DPO 训练。
    """
    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    dpo_cfg = cfg.get("rl", {}).get("dpo", {})
    emo_cfg = cfg.get("reward", {}) or {}
    training_cfg = cfg.get("training", {})
    rollout_cfg = cfg.get("rollout", {}) or {}

    output_dir = training_cfg.get("output_dir", "outputs/dpo_emo")
    num_profiles = rollout_cfg.get("num_profiles", 20)
    batch_size = rollout_cfg.get("batch_size", 4)
    num_samples = rollout_cfg.get("num_samples", 4)
    max_turns = rollout_cfg.get("max_turns", 8)
    max_new_tokens = data_cfg.get("max_response_length", 256)
    num_train_epochs = training_cfg.get("num_train_epochs", 1)
    max_steps = training_cfg.get("max_steps", -1)
    save_steps = training_cfg.get("save_steps", 500)
    logging_steps = training_cfg.get("logging_steps", 50)

    train_file = data_cfg.get("train_file", "data/data/train_profile.jsonl")
    data_dir = data_cfg.get("data_dir") or os.path.dirname(train_file)
    if not os.path.isdir(data_dir):
        data_dir = os.path.dirname(os.path.abspath(train_file))

    profiles = load_profiles(data_dir, split="train")
    profiles = profiles[:num_profiles]
    if not profiles:
        raise ValueError(f"未找到 profile 数据，请检查 {data_dir}")

    batch_items = [
        {"profile": p, "prompt": build_initial_prompt(p)}
        for p in profiles
    ]

    use_planning_score = rollout_cfg.get("use_planning_score", True)
    use_mock_sim = rollout_cfg.get("use_mock_simulator", False)
    sft_model_path = rollout_cfg.get("sft_model_path") or model_cfg.get("sft_model_path")
    target = rollout_cfg.get("target", "eq")

    if use_planning_score and not use_mock_sim:
        if sft_model_path:
            from src.training.local_planning_llm import build_local_planning_llm_fn
            planning_llm_fn = build_local_planning_llm_fn(sft_model_path)
        else:
            from src.training.qwen_user_simulator import build_qwen_user_llm_fn
            planning_llm_fn = build_qwen_user_llm_fn(
                api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"),
                model=rollout_cfg.get("planning_model", "deepseek-chat"),
            )

        def score_fn(responses, context):
            return score_responses_with_planning(
                responses=responses,
                profile=context["profile"],
                dialog=context["dialog"],
                emo_point=context["emo_point"],
                llm_fn=planning_llm_fn,
                target_prompt=target,
            )
    else:
        reward_fn = build_reward_fn_emo(
            emo_adapter_path=emo_cfg.get("emo_adapter_path"),
            reward_mode=emo_cfg.get("reward_mode", "mode1"),
            w1=emo_cfg.get("w1", 1.0),
            w2=emo_cfg.get("w2", 0.3),
            w3=emo_cfg.get("w3", 0.2),
            trend_n=emo_cfg.get("trend_n", 5),
        )

        def score_fn(responses, context):
            raw = reward_fn(responses)
            return [float(r) if isinstance(r, (int, float)) else 0.5 for r in raw]

    def build_user_sim(profile: Dict[str, Any]):
        if use_mock_sim:
            return _MockUserSimulator()
        from src.training.hard_player_simulator_dsv3 import build_player_simulator_with_planning
        return build_player_simulator_with_planning(
            profile=profile,
            sft_model_path=sft_model_path,
            target=rollout_cfg.get("target", "eq"),
            initial_emo_point=50.0,
            api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"),
            model=rollout_cfg.get("user_model", "deepseek-chat"),
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
    )
    model = mt.model
    tokenizer = mt.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.eval()

    print(f"[DPO-Emo] 开始 rollout，profiles={len(batch_items)}, num_samples={num_samples}, max_turns={max_turns}, scoring={'planning' if use_planning_score and not use_mock_sim else 'reward_emo'}")
    all_pairs: List[Dict[str, str]] = []
    for i in range(0, len(batch_items), batch_size):
        batch = batch_items[i : i + batch_size]
        pairs = run_dpo_rollout_batch(
            batch_items=batch,
            model=model,
            tokenizer=tokenizer,
            score_fn=score_fn,
            build_user_sim=build_user_sim,
            device=device,
            num_samples=num_samples,
            max_turns=max_turns,
            max_new_tokens=max_new_tokens,
            temperature=rollout_cfg.get("temperature", 0.8),
            top_p=rollout_cfg.get("top_p", 1.0),
            use_planning_score=use_planning_score and not use_mock_sim,
        )
        all_pairs.extend(pairs)
        print(f"  已处理 {min(i + batch_size, len(batch_items))}/{len(batch_items)} profiles，收集 {len(pairs)} pairs，累计 {len(all_pairs)}")

    if not all_pairs:
        raise RuntimeError("Rollout 未收集到任何 DPO 偏好对，请检查 score_fn 或 num_samples")

    ds = Dataset.from_list(all_pairs)
    for col in ["prompt", "chosen", "rejected"]:
        if col not in ds.column_names:
            raise ValueError(f"DPO 数据需要 {col} 列，当前: {ds.column_names}")

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg.get("beta", 0.1),
        learning_rate=dpo_cfg.get("learning_rate", 2.0e-6),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps > 0 else -1,
        save_steps=save_steps,
        logging_steps=logging_steps,
        bf16=model_cfg.get("dtype", "bfloat16") == "bfloat16",
        fp16=model_cfg.get("dtype") == "float16",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        processing_class=tokenizer,
        train_dataset=ds,
    )

    os.makedirs(output_dir, exist_ok=True)
    print(f"[DPO-Emo] 开始训练，共 {len(all_pairs)} 条偏好对")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[DPO-Emo] 训练完成，模型已保存到 {output_dir}")
