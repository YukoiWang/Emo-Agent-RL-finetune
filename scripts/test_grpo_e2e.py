# -*- coding: utf-8 -*-
"""
GRPO 端到端测试：用 1.5B 模型跑几步验证完整流程。
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training.grpo_training import run_grpo_training

cfg = {
    "seed": 42,
    "model": {
        "sft_model_path": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        "dtype": "bfloat16",
        "use_lora": True,
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    },
    "data": {
        "train_file": os.path.join(ROOT, "data/data/train_profile.jsonl"),
        "max_prompt_length": 512,
        "max_response_length": 64,
        "num_proc": 1,
    },
    "rl": {
        "algo": "grpo",
        "grpo": {
            "num_generations": 4,
            "learning_rate": 1e-6,
            "epsilon": 0.2,
            "beta": 0.02,
            "temperature": 1.0,
            "top_p": 1.0,
        },
    },
    "training": {
        "output_dir": os.path.join(ROOT, "outputs/grpo_test"),
        "total_steps": 5,
        "logging_steps": 1,
        "save_steps": 0,
    },
}

if __name__ == "__main__":
    run_grpo_training(cfg)
