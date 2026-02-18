import json
import torch
import numpy as np
from peft import PeftModel
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer

# ====================== 核心配置（改成你的路径） ======================
BASE_MODEL_PATH = "/home/yukiwang/models/Qwen2-7B-Instruct"  # 原生基座模型路径
LORA_MODEL_PATH = "./sft_emo_model/final"                    # SFT后的LoRA权重路径
EVAL_DATA_PATH = "./combined_eval.jsonl"                     # 测试集路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100
# 任务比例（和训练一致）
R = {"polarity":0.40, "emotion":0.30, "dynamics":0.20, "intensity":0.10}

# ====================== 全局标签映射（和之前一致） ======================
# 1. 极性（SST-2）
polarity_labels = ["negative", "positive"]
polarity_label2id = {l: i for i, l in enumerate(polarity_labels)}
# 2. 情感强度（SST-5）
intensity_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
intensity_label2id = {l: i for i, l in enumerate(intensity_labels)}
# 3. 对话动态（DailyDialog）
dialog_emotion_labels = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
dialog_label2id = {l: i for i, l in enumerate(dialog_emotion_labels)}
# 4. 多标签情感（GoEmotions）
def get_goemotions_labels():
    ds = load_dataset("go_emotions")
    return ds["train"].features["labels"].feature.names
goemotions_labels = get_goemotions_labels()
mlb = MultiLabelBinarizer(classes=goemotions_labels)
mlb.fit([goemotions_labels])

# ====================== 加载模型的通用函数 ======================
def load_model(model_type="base"):
    """
    加载模型：
    - model_type="base"：加载原生Qwen2模型（SFT前）
    - model_type="sft"：加载融合LoRA的SFT模型（SFT后）
    """
    print(f"\n===== Loading {model_type} model =====")
    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, trust_remote_code=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    # 融合LoRA（仅SFT模型）
    if model_type == "sft":
        model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
        model = model.merge_and_unload()
    
    model.eval()  # 评估模式
    print(f"{model_type} model loaded successfully!")
    return model, tokenizer

# ====================== 加载测试集 + 任务拆分 ======================
def load_and_split_eval_data(path):
    eval_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line.strip()))
    
    task_data = {
        "polarity": [], "emotion": [], "dynamics": [], "intensity": []
    }
    for item in eval_data:
        instr = item["instruction"].lower()
        if "sentiment polarity" in instr:
            task_data["polarity"].append(item)
        elif "identify the emotions" in instr and "multi-label" in instr:
            task_data["emotion"].append(item)
        elif "emotion changes between dialogue turns" in instr:
            task_data["dynamics"].append(item)
        elif "sentiment intensity level" in instr:
            task_data["intensity"].append(item)
    return task_data

# ====================== 通用预测函数 ======================
def predict_single_sample(model, tokenizer, item):
    # 构建Prompt（和训练一致）
    prompt = f"""
<|im_start|>system
You are an English emotion analysis assistant. Follow the instruction strictly and output only the required result, no extra words.
<|im_end|>
<|im_start|>user
{item["instruction"]}
{item["input"]}
<|im_end|>
<|im_start|>assistant
"""
    # 推理（无采样）
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    # 解析输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = response.split("assistant")[-1].strip().lower()
    # 按任务清洗
    instr = item["instruction"].lower()
    if "sentiment polarity" in instr:
        return "positive" if "positive" in pred else "negative" if "negative" in pred else "neutral"
    elif "multi-label" in instr:
        pred = pred.replace("[", "").replace("]", "").replace("'", "").replace('"', '')
        return [e.strip() for e in pred.split(",") if e.strip() in goemotions_labels] or ["neutral"]
    elif "emotion changes" in instr:
        prev = [e for e in dialog_emotion_labels if e in pred.split("prev_emotion")[1].split(",")[0]] if "prev_emotion" in pred else ["neutral"]
        curr = [e for e in dialog_emotion_labels if e in pred.split("curr_emotion")[1]] if "curr_emotion" in pred else ["neutral"]
        return {"prev_emotion": prev[0] if prev else "neutral", "curr_emotion": curr[0] if curr else "neutral"}
    elif "intensity level" in instr:
        for label in intensity_labels:
            if label in pred:
                return label
        return "neutral"

# ====================== 按任务评估的通用函数 ======================
def evaluate_task(task_name, task_data, model, tokenizer):
    """评估单个任务，返回核心指标"""
    if not task_data:
        return {}
    
    # 极性任务
    if task_name == "polarity":
        true = [polarity_label2id[item["output"]] for item in task_data]
        pred = [polarity_label2id.get(predict_single_sample(model, tokenizer, item), 0) for item in tqdm(task_data, desc=f"Eval {task_name}")]
        acc = accuracy_score(true, pred)
        f1 = precision_recall_fscore_support(true, pred, average="binary")[2]
        return {"acc": acc, "f1": f1}
    
    # 多标签情感
    elif task_name == "emotion":
        true = [item["output"] for item in task_data]
        pred = [predict_single_sample(model, tokenizer, item) for item in tqdm(task_data, desc=f"Eval {task_name}")]
        true_enc = mlb.transform(true)
        pred_enc = mlb.transform(pred)
        f1 = precision_recall_fscore_support(true_enc, pred_enc, average="macro")[2]
        return {"macro_f1": f1}
    
    # 对话动态
    elif task_name == "dynamics":
        true_prev = [dialog_label2id[item["output"]["prev_emotion"]] for item in task_data]
        true_curr = [dialog_label2id[item["output"]["curr_emotion"]] for item in task_data]
        pred_prev = []
        pred_curr = []
        for item in tqdm(task_data, desc=f"Eval {task_name}"):
            pred_dict = predict_single_sample(model, tokenizer, item)
            pred_prev.append(dialog_label2id.get(pred_dict["prev_emotion"], 0))
            pred_curr.append(dialog_label2id.get(pred_dict["curr_emotion"], 0))
        total_correct = sum([1 for tp, tc, pp, pc in zip(true_prev, true_curr, pred_prev, pred_curr) if tp == pp and tc == pc])
        acc = total_correct / len(task_data)
        return {"acc": acc}
    
    # 情感强度
    elif task_name == "intensity":
        true = [intensity_label2id[item["output"]] for item in task_data]
        pred = [intensity_label2id.get(predict_single_sample(model, tokenizer, item), 2) for item in tqdm(task_data, desc=f"Eval {task_name}")]
        acc = accuracy_score(true, pred)
        f1 = precision_recall_fscore_support(true, pred, average="weighted")[2]
        return {"acc": acc, "f1": f1}

# ====================== 整体评估 + 对比 ======================
def evaluate_and_compare():
    # 1. 加载测试集
    task_data = load_and_split_eval_data(EVAL_DATA_PATH)
    
    # 2. 评估原生模型（SFT前）
    base_model, base_tokenizer = load_model("base")
    base_results = {}
    for task in ["polarity", "emotion", "dynamics", "intensity"]:
        base_results[task] = evaluate_task(task, task_data[task], base_model, base_tokenizer)
    
    # 3. 评估SFT模型（SFT后）
    sft_model, sft_tokenizer = load_model("sft")
    sft_results = {}
    for task in ["polarity", "emotion", "dynamics", "intensity"]:
        sft_results[task] = evaluate_task(task, task_data[task], sft_model, sft_tokenizer)
    
    # 4. 计算加权平均分
    def calculate_weighted_acc(results):
        weighted = 0.0
        for task, metrics in results.items():
            if "acc" in metrics:
                weighted += metrics["acc"] * R[task]
            elif "macro_f1" in metrics:
                weighted += metrics["macro_f1"] * R[task]
        return weighted
    
    base_weighted = calculate_weighted_acc(base_results)
    sft_weighted = calculate_weighted_acc(sft_results)
    improvement = sft_weighted - base_weighted

    # 5. 输出对比结果
    print("\n" + "="*80)
    print("📊 SFT 前 vs SFT 后 效果对比")
    print("="*80)
    # 打印对比表格
    print(f"{'任务类型':<15} | {'SFT前指标':<12} | {'SFT后指标':<12} | {'提升幅度':<10}")
    print("-"*80)
    # 极性
    if base_results["polarity"]:
        base_acc = base_results["polarity"]["acc"]
        sft_acc = sft_results["polarity"]["acc"]
        print(f"{'Polarity (SST-2)':<15} | {base_acc:.4f}        | {sft_acc:.4f}        | {sft_acc-base_acc:+.4f}")
    # 多标签情感
    if base_results["emotion"]:
        base_f1 = base_results["emotion"]["macro_f1"]
        sft_f1 = sft_results["emotion"]["macro_f1"]
        print(f"{'Emotion (GoEmotions)':<15} | {base_f1:.4f}        | {sft_f1:.4f}        | {sft_f1-base_f1:+.4f}")
    # 对话动态
    if base_results["dynamics"]:
        base_acc = base_results["dynamics"]["acc"]
        sft_acc = sft_results["dynamics"]["acc"]
        print(f"{'Dynamics (DailyDialog)':<15} | {base_acc:.4f}        | {sft_acc:.4f}        | {sft_acc-base_acc:+.4f}")
    # 情感强度
    if base_results["intensity"]:
        base_acc = base_results["intensity"]["acc"]
        sft_acc = sft_results["intensity"]["acc"]
        print(f"{'Intensity (SST-5)':<15} | {base_acc:.4f}        | {sft_acc:.4f}        | {sft_acc-base_acc:+.4f}")
    # 整体加权
    print("-"*80)
    print(f"{'加权平均分':<15} | {base_weighted:.4f}        | {sft_weighted:.4f}        | {improvement:+.4f}")
    print("="*80)

    # 保存详细结果到文件
    with open("sft_compare_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "base_model": base_results,
            "sft_model": sft_results,
            "weighted_acc": {"base": base_weighted, "sft": sft_weighted, "improvement": improvement}
        }, f, indent=4, ensure_ascii=False)
    print("\n✅ 详细结果已保存到 sft_compare_results.json")

# ====================== 主函数 ======================
if __name__ == "__main__":
    evaluate_and_compare()