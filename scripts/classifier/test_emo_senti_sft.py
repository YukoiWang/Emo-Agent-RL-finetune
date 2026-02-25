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
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_MODEL = "/home/yukiwang/models/Qwen2-7B-Instruct"
ADAPTER_DIR = "/home/yukiwang/xlwy/emo_classifier_lora/checkpoint-11025"
TEST_FILE = "/home/yukiwang/xlwy/combined_sft_zh_test.jsonl"

label_map = {"消极": 0, "中性": 1, "积极": 2}

# ===== Dataset =====
dataset = load_dataset("json", data_files={"test": TEST_FILE})

def preprocess(example):
    return {
        "text": example["input"],
        "label": label_map[example["output"]]
    }

dataset = dataset.map(preprocess)

# ===== Tokenizer =====
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# ===== Model =====
# 先用 8bit 量化加载底座模型，再加载 LoRA 适配器权重
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(label_map),
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
base_model.config.pad_token_id = tokenizer.pad_token_id
if base_model.config.eos_token_id is None:
    base_model.config.eos_token_id = tokenizer.eos_token_id

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.config.pad_token_id = tokenizer.pad_token_id
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
        "f1": f1
    }

# ===== Trainer =====
args = TrainingArguments(
    output_dir="./eval_tmp",
    per_device_eval_batch_size=4,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# ===== Eval =====
results = trainer.evaluate()
print("===== Test Evaluation =====")
print(results)
