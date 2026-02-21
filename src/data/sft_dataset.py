from typing import Dict, List, Any, Optional

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def load_sft_dataset(
    train_file: str,
    eval_file: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
    num_proc: int = 4,
) -> Dict[str, Dataset]:
    """
    加载心理咨询 SFT 数据集。

    默认假设 jsonl 每行结构类似：
    {
      "user": "来访者的话",
      "assistant": "咨询师的回复",
      "system": "可选的系统提示（如角色设定）"
    }
    """

    data_files: Dict[str, str] = {"train": train_file}
    if eval_file:
        data_files["validation"] = eval_file

    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
    )

    def format_example(example: Dict[str, Any]) -> Dict[str, str]:
        system = example.get("system", "")
        user = example["user"]
        assistant = example["assistant"]

        if system:
            prompt = f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
        else:
            prompt = f"<|user|>\n{user}\n<|assistant|>\n"

        return {
            "prompt": prompt,
            "response": assistant,
        }

    raw_datasets = raw_datasets.map(
        format_example,
        num_proc=num_proc,
    )

    def tokenize_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        full_text = example["prompt"] + example["response"]
        tokenized = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )

        # 简单的因果语言模型标签：所有 token 预测下一个 token
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=False,
        num_proc=num_proc,
        remove_columns=[col for col in raw_datasets["train"].column_names if col not in ["prompt", "response"]],
    )

    return tokenized_datasets

