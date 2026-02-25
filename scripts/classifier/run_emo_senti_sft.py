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
OUTPUT_DIR = "./emo_classifier_lora"
NUM_GPUS = 2
NUM_LABELS = 3  # 积极 / 消极 / 中性

# LoRA 参数
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

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

# ====================== Tokenizer ======================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right" # 对于分类任务 "right" 也可以，但 "left" 是更通用的推荐
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====================== 量化配置（修正） ======================
# 修正后的8-bit量化配置，移除了无效参数
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,  # RTX 3090 使用 float16 以获得最佳性能
)

# ====================== 模型加载（推荐方式） ======================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16  # 统一为 float16，与训练参数匹配
)

# 梯度检查点和缓存设置
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ====================== LoRA 配置 ======================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="SEQ_CLS",  # 明确任务类型为序列分类
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    inference_mode=False
)

# 应用PEFT LoRA
model = get_peft_model(model, lora_config)
# 确保 tokenizer 和 model 的配置一致
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
# 将 tokenizer 的 padding token id 同步到模型的配置中
model.config.pad_token_id = tokenizer.pad_token_id
# =========================================================================

# 打印可训练参数，验证LoRA是否配置成功
model.print_trainable_parameters()

# ====================== Tokenize 数据集 ======================
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

# ====================== 评估指标计算函数 ======================
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

# ====================== 训练参数（核心优化） ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    fp16=False,   # 核心改动：开启fp16，适配RTX 3090
    bf16=False,  # 核心改动：关闭bf16
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    optim="paged_adamw_8bit",  # 使用8-bit优化器节省显存
    gradient_checkpointing=True,
    remove_unused_columns=True,
    save_total_limit=3
)

# ====================== Trainer 初始化 ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics,
)

# ====================== 启动训练 ======================
print("===== Starting Training =====")
trainer.train(resume_from_checkpoint=True)
trainer.save_model(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

print(f"✅ 情感分类 LoRA 模型训练完成，已保存至: {OUTPUT_DIR}/final")

# ====================== 在测试集上进行最终评估 ======================
print("===== Test Set Evaluation Results =====")
test_results = trainer.evaluate(dataset["test"])
print(test_results)