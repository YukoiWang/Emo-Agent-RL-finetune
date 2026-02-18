import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, # 核心改动：直接使用带分类头的模型
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ====================== 关键环境变量（解决CUDA非法访问） ======================
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # 同步执行，获取准确报错
os.environ["DISABLE_CUBLASLT_MATMUL"] = "1"  # 禁用cublasLt，解决4-bit矩阵乘法报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 避免tokenizer多线程冲突

# ====================== 基础配置 ======================
MODEL_NAME = "/home/yukiwang/models/Qwen2-7B-Instruct"
OUTPUT_DIR = "./emo_classifier_base"
NUM_GPUS = 2
NUM_LABELS = 3  # 积极 / 消极 / 中性

# ====================== 数据集 ======================
dataset = load_dataset(
    "json",
    data_files={
        "train": "/home/yukiwang/xlwy/combined_sft_zh_train.jsonl",
        "validation": "/home/yukiwang/xlwy/combined_sft_zh_val.jsonl",
        "test": "/home/yukiwang/xlwy/combined_sft_zh_test.jsonl",
    }
)

label_map = {"消极": 0, "中性": 1, "积极": 2}

def preprocess(example):
    text = example["input"]
    label = label_map[example["output"]]
    return {"text": text, "label": label}

dataset = dataset.map(preprocess)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right" # 对于分类任务 "right" 也可以，但 "left" 是更通用的推荐
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ===== 冻结大模型参数，只训练分类头 =====
# 先全部冻结
for name, param in model.named_parameters():
    param.requires_grad = False

# Qwen2ForSequenceClassification 的分类头通常叫做 "score"
if hasattr(model, "score"):
    for param in model.score.parameters():
        param.requires_grad = True
else:
    # 兜底：如果未来版本名字变了，你可以根据实际属性名调整这里
    raise ValueError("未找到分类头 `score`，请检查模型结构并修改冻结逻辑。")

model.gradient_checkpointing_enable()
model.config.use_cache = False

model.config.pad_token_id = tokenizer.pad_token_id
if model.config.eos_token_id is None:
    model.config.eos_token_id = tokenizer.eos_token_id

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # 准确率
    acc = accuracy_score(labels, preds)

    # 精确率、召回率、F1分数 (加权平均)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,   # 核心改动：开启fp16，适配RTX 3090
    bf16=False,  # 核心改动：关闭bf16
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    gradient_checkpointing=True,
    remove_unused_columns=True,
    save_total_limit=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

print("===== Starting Training =====")
# 首次训练不要从 checkpoint 恢复，训练完成后再用 resume_from_checkpoint 续训
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print(f"✅ 情感分类 Base 模型训练完成，已保存至: {OUTPUT_DIR}/final")

# ====================== 在测试集上进行最终评估 ======================
print("===== Test Set Evaluation Results =====")
test_results = trainer.evaluate(dataset["test"])
print(test_results)