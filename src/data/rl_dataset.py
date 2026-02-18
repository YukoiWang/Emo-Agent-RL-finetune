from typing import Dict, Any

from datasets import load_dataset, Dataset


def load_rl_dataset(
    train_file: str,
    num_proc: int = 4,
) -> Dataset:
    """
    RL 数据集加载。

    支持两种典型格式（按需扩展）：
    1) 评分数据（用于 PPO 等）：
       {
         "user": "...",
         "assistant": "...（可选：已有回复，用作 warm-start）",
         "score": 0.0-1.0 或其他标量
       }
    2) 偏好数据（用于 DPO/KTO 等）：
       {
         "user": "...",
         "chosen": "更好的咨询师回复",
         "rejected": "较差的咨询师回复"
       }
    """
    dataset = load_dataset(
        "json",
        data_files={"train": train_file},
    )["train"]

    # 此处先不做复杂处理，只做基本清洗占位
    def clean_example(example: Dict[str, Any]) -> Dict[str, Any]:
        example["user"] = example["user"].strip()
        if "assistant" in example and isinstance(example["assistant"], str):
            example["assistant"] = example["assistant"].strip()
        if "chosen" in example and isinstance(example["chosen"], str):
            example["chosen"] = example["chosen"].strip()
        if "rejected" in example and isinstance(example["rejected"], str):
            example["rejected"] = example["rejected"].strip()
        return example

    dataset = dataset.map(clean_example, num_proc=num_proc)
    return dataset

