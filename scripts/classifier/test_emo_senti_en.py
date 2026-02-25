import json
import torch
import numpy as np
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
R = {"polarity":0.40, "emotion":0.30, "dynamics":0.20, "intensity":0.10}

# ====================== 配置项（改成你的路径） ======================
BASE_MODEL_PATH = "/home/yukiwang/models/Qwen2-7B-Instruct"  # 你的基座模型路径
LORA_MODEL_PATH = "./sft_emo_model/final"                    # 你的LoRA模型保存路径
EVAL_DATA_PATH = "./combined_eval.jsonl"                     # 生成的测试集路径
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100  # 足够容纳所有任务的输出长度

# ====================== 全局变量（适配4类任务） ======================
# 1. 极性任务标签映射（SST-2）
polarity_labels = ["negative", "positive"]
polarity_label2id = {l: i for i, l in enumerate(polarity_labels)}

# 2. 情感强度标签映射（SST-5）
intensity_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
intensity_label2id = {l: i for i, l in enumerate(intensity_labels)}

# 3. 对话动态标签映射（DailyDialog）
dialog_emotion_labels = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
dialog_label2id = {l: i for i, l in enumerate(dialog_emotion_labels)}

# 4. 多标签情感标签映射（GoEmotions，先加载训练集获取完整标签）
def get_goemotions_labels():
    ds = load_dataset("go_emotions")
    return ds["train"].features["labels"].feature.names
goemotions_labels = get_goemotions_labels()
mlb = MultiLabelBinarizer(classes=goemotions_labels)
mlb.fit([goemotions_labels])  # 初始化多标签编码器

# ====================== 加载模型和Tokenizer ======================
def load_model_and_tokenizer():
    print(f"Loading model from {BASE_MODEL_PATH}...")
    # 加载Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 加载基座模型 + 合并LoRA权重
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )

    # 合并LoRA权重（评估时合并更稳定）
    model = PeftModel.from_pretrained(model, LORA_MODEL_PATH)
    model = model.merge_and_unload()
    model.eval()  # 关闭dropout，进入评估模式
    print("Model loaded successfully!")
    return model, tokenizer

# ====================== 加载测试集并分类任务 ======================
def load_and_split_eval_data(path):
    """加载测试集，并按任务类型拆分（方便分类评估）"""
    eval_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            eval_data.append(json.loads(line.strip()))
    
    # 按instruction拆分任务
    task_data = {
        "polarity": [],       # SST-2: 极性判断
        "emotion": [],        # GoEmotions: 多标签情感
        "dynamics": [],       # DailyDialog: 对话动态
        "intensity": []       # SST-5: 情感强度
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
    
    print(f"✅ 测试集任务拆分完成：")
    for task, data in task_data.items():
        print(f"  - {task}: {len(data)} samples")
    return task_data

# ====================== 模型预测函数（按任务适配） ======================
def predict_single_sample(model, tokenizer, item):
    """单样本预测，返回清洗后的预测结果"""
    # 构建和训练时一致的prompt
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
    # 模型推理（关闭采样，保证结果稳定）
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.0,  # 确定性输出
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解析输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = response.split("assistant")[-1].strip()
    
    # 按任务清洗预测结果
    instr = item["instruction"].lower()
    if "sentiment polarity" in instr:  # 极性任务
        pred = pred.lower()
        if "positive" in pred:
            return "positive"
        elif "negative" in pred:
            return "negative"
        else:
            return "neutral"  # 兜底
    
    elif "multi-label" in instr:  # 多标签情感
        # 提取所有情感标签（兼容列表/字符串输出）
        pred = pred.lower().replace("[", "").replace("]", "").replace("'", "").replace('"', '')
        pred_emotions = [e.strip() for e in pred.split(",") if e.strip() in goemotions_labels]
        return pred_emotions if pred_emotions else ["neutral"]
    
    elif "emotion changes" in instr:  # 对话动态
        # 提取prev/curr情感（兼容字典/字符串输出）
        pred = pred.lower()
        prev_emotion = None
        curr_emotion = None
        # 适配常见输出格式："prev_emotion: neutral, curr_emotion: happiness"
        if "prev_emotion" in pred and "curr_emotion" in pred:
            prev_emotion = [e for e in dialog_emotion_labels if e in pred.split("prev_emotion")[1].split(",")[0]][0]
            curr_emotion = [e for e in dialog_emotion_labels if e in pred.split("curr_emotion")[1]][0]
        return {"prev_emotion": prev_emotion or "neutral", "curr_emotion": curr_emotion or "neutral"}
    
    elif "intensity level" in instr:  # 情感强度
        pred = pred.lower()
        # 匹配最接近的强度标签
        for label in intensity_labels:
            if label in pred:
                return label
        return "neutral"  # 兜底

# ====================== 按任务评估 ======================
def evaluate_polarity(task_data, model, tokenizer):
    """评估极性任务（SST-2）"""
    true = []
    pred = []
    for item in tqdm(task_data, desc="Evaluating polarity"):
        # 真实标签
        true_label = item["output"]
        true.append(polarity_label2id[true_label])
        # 预测标签
        pred_label = predict_single_sample(model, tokenizer, item)
        pred.append(polarity_label2id.get(pred_label, 0))  # 兜底为negative
    
    # 计算指标
    acc = accuracy_score(true, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average="binary")
    print("\n📊 Polarity (SST-2) Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(classification_report(true, pred, target_names=polarity_labels, digits=4))
    return {"polarity": {"acc": acc, "f1": f1}}

def evaluate_emotion(task_data, model, tokenizer):
    """评估多标签情感任务（GoEmotions）"""
    true = []
    pred = []
    for item in tqdm(task_data, desc="Evaluating multi-label emotion"):
        # 真实标签（多标签）
        true_labels = item["output"]
        true.append(true_labels)
        # 预测标签
        pred_labels = predict_single_sample(model, tokenizer, item)
        pred.append(pred_labels)
    
    # 多标签编码
    true_encoded = mlb.transform(true)
    pred_encoded = mlb.transform(pred)
    
    # 计算宏平均指标（多标签核心）
    precision, recall, f1, _ = precision_recall_fscore_support(true_encoded, pred_encoded, average="macro")
    print("\n📊 Multi-label Emotion (GoEmotions) Results:")
    print(f"  Macro Precision: {precision:.4f}")
    print(f"  Macro Recall: {recall:.4f}")
    print(f"  Macro F1-Score: {f1:.4f}")
    return {"emotion": {"macro_f1": f1}}

def evaluate_dynamics(task_data, model, tokenizer):
    """评估对话情感动态（DailyDialog）"""
    true_prev = []
    true_curr = []
    pred_prev = []
    pred_curr = []
    
    for item in tqdm(task_data, desc="Evaluating dialogue dynamics"):
        # 真实标签
        true_prev_label = item["output"]["prev_emotion"]
        true_curr_label = item["output"]["curr_emotion"]
        true_prev.append(dialog_label2id[true_prev_label])
        true_curr.append(dialog_label2id[true_curr_label])
        
        # 预测标签
        pred_dict = predict_single_sample(model, tokenizer, item)
        pred_prev_label = pred_dict["prev_emotion"]
        pred_curr_label = pred_dict["curr_emotion"]
        pred_prev.append(dialog_label2id.get(pred_prev_label, 0))
        pred_curr.append(dialog_label2id.get(pred_curr_label, 0))
    
    # 计算整体准确率（prev+curr都对才算对）
    total_correct = sum([1 for t_p, t_c, p_p, p_c in zip(true_prev, true_curr, pred_prev, pred_curr) if t_p == p_p and t_c == p_c])
    acc = total_correct / len(task_data)
    
    # 分别计算prev/curr的F1
    prev_f1 = precision_recall_fscore_support(true_prev, pred_prev, average="weighted")[2]
    curr_f1 = precision_recall_fscore_support(true_curr, pred_curr, average="weighted")[2]
    
    print("\n📊 Dialogue Emotion Dynamics (DailyDialog) Results:")
    print(f"  Overall Accuracy (prev+curr correct): {acc:.4f}")
    print(f"  Previous Emotion F1: {prev_f1:.4f}")
    print(f"  Current Emotion F1: {curr_f1:.4f}")
    return {"dynamics": {"acc": acc, "prev_f1": prev_f1, "curr_f1": curr_f1}}

def evaluate_intensity(task_data, model, tokenizer):
    """评估情感强度（SST-5）"""
    true = []
    pred = []
    for item in tqdm(task_data, desc="Evaluating sentiment intensity"):
        # 真实标签
        true_label = item["output"]
        true.append(intensity_label2id[true_label])
        # 预测标签
        pred_label = predict_single_sample(model, tokenizer, item)
        pred.append(intensity_label2id.get(pred_label, 2))  # 兜底为neutral
    
    # 计算指标
    acc = accuracy_score(true, pred)
    f1 = precision_recall_fscore_support(true, pred, average="weighted")[2]
    print("\n📊 Sentiment Intensity (SST-5) Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Weighted F1-Score: {f1:.4f}")
    print(classification_report(true, pred, target_names=intensity_labels, digits=4))
    return {"intensity": {"acc": acc, "f1": f1}}

# ====================== 主函数 ======================
def main():
    # 1. 加载模型
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 加载并拆分测试集
    task_data = load_and_split_eval_data(EVAL_DATA_PATH)
    
    # 3. 按任务评估
    results = {}
    if task_data["polarity"]:
        results.update(evaluate_polarity(task_data["polarity"], model, tokenizer))
    if task_data["emotion"]:
        results.update(evaluate_emotion(task_data["emotion"], model, tokenizer))
    if task_data["dynamics"]:
        results.update(evaluate_dynamics(task_data["dynamics"], model, tokenizer))
    if task_data["intensity"]:
        results.update(evaluate_intensity(task_data["intensity"], model, tokenizer))
    
    # 4. 输出整体汇总
    print("\n===================== 📈 Overall Evaluation Summary =====================")
    total_samples = sum([len(d) for d in task_data.values()])
    weighted_acc = 0.0
    for task, metrics in results.items():
        weight = R[task]
        if "acc" in metrics:
            weighted_acc += metrics["acc"] * weight
        elif "macro_f1" in metrics:  # 多标签用f1替代acc
            weighted_acc += metrics["macro_f1"] * weight
    print(f"Weighted Average Accuracy (by task ratio): {weighted_acc:.4f}")
    print("=========================================================================")

if __name__ == "__main__":
    # 补充加载go_emotions的依赖（如果没装）
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets"])
        from datasets import load_dataset
    main()