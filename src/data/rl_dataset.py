from typing import Dict, Any

from datasets import load_dataset, Dataset

from src.data.profile_dataset import build_initial_prompt


def _example_to_user(example: Dict[str, Any], max_scene_len: int = 1500) -> str:
    """将 profile 格式样本转为 user prompt（供 PPO 的 dataset_text_field 使用）。"""
    return build_initial_prompt(example, max_scene_len=max_scene_len)


def load_rl_dataset(
    train_file: str,
    num_proc: int = 4,
    format: str = "auto",
    max_scene_len: int = 1500,
) -> Dataset:
    """
    RL 数据集加载。

    支持格式（format）：
    - "auto": 自动检测。若样本有 "user" 则直接用；若有 "player"/"scene"/"task" 则按 profile 转成 user。
    - "standard": 期望有 "user" 字段。
    - "profile": 按 profile 格式（player, scene, task 等）构建 user prompt。

    1) 评分数据（用于 PPO 等）：
       {"user": "...", "assistant": "...", "score": 0.0-1.0}
    2) 偏好数据（用于 DPO/KTO 等）：
       {"user": "...", "chosen": "...", "rejected": "..."}
    3) Profile 数据（player/scene/task）：
       自动用 build_initial_prompt 生成 user。
    """
    dataset = load_dataset(
        "json",
        data_files={"train": train_file},
    )["train"]

    first = dataset[0]

    def _is_profile(ex: Dict[str, Any]) -> bool:
        return "player" in ex and ("scene" in ex or "task" in ex) and "user" not in ex

    use_profile = (
        (format == "profile") or
        (format == "auto" and _is_profile(first))
    )

    if use_profile:
        def add_user(example: Dict[str, Any]) -> Dict[str, Any]:
            example["user"] = _example_to_user(example, max_scene_len=max_scene_len)
            return example
        dataset = dataset.map(add_user, num_proc=num_proc, desc="build user from profile")
    else:
        def clean_example(example: Dict[str, Any]) -> Dict[str, Any]:
            example["user"] = (example.get("user") or "").strip()
            if "assistant" in example and isinstance(example["assistant"], str):
                example["assistant"] = example["assistant"].strip()
            if "chosen" in example and isinstance(example["chosen"], str):
                example["chosen"] = example["chosen"].strip()
            if "rejected" in example and isinstance(example["rejected"], str):
                example["rejected"] = example["rejected"].strip()
            return example
        dataset = dataset.map(clean_example, num_proc=num_proc)

    return dataset

