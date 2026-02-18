import os
# 强制仅用本地文件，避免把本地路径当 Hub repo 去请求 adapter_config.json
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 训练脚本 run_emo_senti_base.py 的保存路径（按脚本位置解析为绝对路径）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "emo_classifier_base"))


def _resolve_model_dir(output_dir: str) -> str:
    """若 output_dir 下有 checkpoint-* 且自身无 config.json，则用最新 checkpoint 路径。"""
    if os.path.isfile(os.path.join(output_dir, "config.json")):
        return output_dir
    import re
    checkpoints = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and re.match(r"checkpoint-\d+", d)
    ]
    if not checkpoints:
        return output_dir
    latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


MODEL_DIR = _resolve_model_dir(OUTPUT_DIR)
BASE_MODEL = "/home/yukiwang/models/Qwen2-7B-Instruct"  # tokenizer 兜底
TEST_FILE = "/home/yukiwang/xlwy/combined_sft_zh_test.jsonl"

label_map = {"消极": 0, "中性": 1, "积极": 2}

# ===== Dataset =====
dataset = load_dataset("json", data_files={"test": TEST_FILE})


def preprocess(example):
    return {
        "text": example["input"],
        "label": label_map[example["output"]],
    }


dataset = dataset.map(preprocess)

# ===== Tokenizer =====
# 若本地路径触发 HF repo_id 校验报错，则从基座加载（与训练时一致）
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True, local_files_only=True)
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ===== 加载 run_emo_senti_base.py 训练好的模型（基座冻结 + 分类头）=====
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(
        f"模型目录不存在: {MODEL_DIR}\n"
        "请先运行 run_emo_senti_base.py 训练并保存到该路径，或修改 MODEL_DIR 为已有 checkpoint 路径。"
    )
model = AutoModelForSequenceClassification.from_pretrained(
    os.path.abspath(MODEL_DIR),  # 使用绝对路径，避免被当作 Hub repo_id
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,
)

# 同步 padding 配置，避免 batch_size>1 报错
model.config.pad_token_id = tokenizer.pad_token_id
if model.config.eos_token_id is None:
    model.config.eos_token_id = tokenizer.eos_token_id


# ===== Metrics =====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ===== Trainer =====
args = TrainingArguments(
    output_dir="./eval_tmp_base",
    per_device_eval_batch_size=4,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# ===== Eval =====
results = trainer.evaluate()
print("===== Test Evaluation (run_emo_senti_base.py 训练结果：只训分类头) =====")
print(results)

