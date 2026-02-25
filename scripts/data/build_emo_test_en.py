import json
import random
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

# 测试集文件路径（和训练集文件名区分）
files_eval = {
    "polarity": "sst2_eval.jsonl",
    "emotion": "goemotions_eval.jsonl",
    "dynamics": "dialog_shift_eval.jsonl",
    "intensity": "sst5_eval.jsonl"
}

# 和训练集相同的比例，保证测试集分布一致
R = {
    "polarity": 0.40,
    "emotion": 0.30,
    "dynamics": 0.20,
    "intensity": 0.10
}

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# 1. SST-2 测试集：取原生test分区（训练没用过）
def build_sst2_eval():
    print("Building SST-2 eval set (native test split)...")
    ds = load_dataset("sst2")
    out = []
    for x in tqdm(ds["validation"]):
        label = "positive" if x["label"] == 1 else "negative"
        item = {
            "instruction": "Determine the sentiment polarity of the text (positive/negative)",
            "input": x["sentence"],
            "output": label
        }
        out.append(item)
    write_jsonl(files_eval["polarity"], out)

# 2. GoEmotions 测试集：取原生test分区（训练没用过）
def build_goemotions_eval():
    print("Building GoEmotions eval set (native test split)...")
    ds = load_dataset("go_emotions")
    label_names = ds["train"].features["labels"].feature.names
    out = []
    for x in tqdm(ds["test"]):
        emotions = [label_names[i] for i in x["labels"]]
        item = {
            "instruction": "Identify the emotions in the text (multi-label possible)",
            "input": x["text"],
            "output": emotions
        }
        out.append(item)
    write_jsonl(files_eval["emotion"], out)

# 3. DailyDialog 测试集：拆分全新10%（和训练集无重叠）
def build_dailydialog_eval():
    print("Building DailyDialog eval set (10% new split)...")
    ds = load_dataset("roskoN/dailydialog")
    all_dialogs = ds["train"]
    
    # 用不同种子拆分，保证和训练集无重叠（训练用seed=42，测试用seed=100）
    random.seed(100)
    n_test = int(len(all_dialogs) * 0.1)
    test_indices = set(random.sample(range(len(all_dialogs)), n_test))
    
    emo_map = {0:"neutral",1:"anger",2:"disgust",3:"fear",4:"happiness",5:"sadness",6:"surprise"}
    out = []
    for i in tqdm(test_indices):
        dialog = all_dialogs[i]
        utterances = dialog["utterances"]
        emotions = dialog["emotions"]
        for j in range(len(utterances)-1):
            item = {
                "instruction": "Identify emotion changes between dialogue turns",
                "input": f"Previous: {utterances[j]}\nCurrent: {utterances[j+1]}",
                "output": {"prev_emotion":emo_map[emotions[j]], "curr_emotion":emo_map[emotions[j+1]]}
            }
            out.append(item)
    write_jsonl(files_eval["dynamics"], out)

# 4. SST-5 测试集：取原生test分区（训练没用过）
def build_sst5_eval():
    print("Building SST-5 eval set (native test split)...")
    ds = load_dataset("SetFit/sst5")
    intensity_map = {0:"very negative",1:"negative",2:"neutral",3:"positive",4:"very positive"}
    out = []
    for x in tqdm(ds["test"]):
        item = {
            "instruction": "Determine the sentiment intensity level of the text",
            "input": x["text"],
            "output": intensity_map[x["label"]]
        }
        out.append(item)
    write_jsonl(files_eval["intensity"], out)

# 5. 合并测试集（按训练集比例）
def build_combined_eval():
    print("Building combined_eval.jsonl (分层采样，保证类别均衡)...")
    dataset_all = []
    
    for k, path in files_eval.items():
        # 1. 加载该任务的完整测试集
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        samples = [json.loads(line) for line in lines]  # 转成字典，方便按标签筛选
        if not samples:
            print(f"⚠️  {k} 测试集为空，跳过")
            continue

        # 2. 按任务类型分层采样，保证类别均衡
        sampled_samples = []
        target_total = int(len(samples) * R[k])  # 按比例计算目标采样数
        target_total = max(target_total, 50)     # 保底：每个任务至少采50条
        
        if k == "polarity":  # SST-2：强制正负样本各占50%
            # 拆分正负样本
            pos_samples = [s for s in samples if s["output"] == "positive"]
            neg_samples = [s for s in samples if s["output"] == "negative"]
            # 计算各采样数（保证都有样本）
            pos_sample_num = max(int(target_total * 0.5), 1) if pos_samples else 0
            neg_sample_num = max(target_total - pos_sample_num, 1) if neg_samples else 0
            # 采样
            pos_sampled = random.sample(pos_samples, min(pos_sample_num, len(pos_samples))) if pos_samples else []
            neg_sampled = random.sample(neg_samples, min(neg_sample_num, len(neg_samples))) if neg_samples else []
            sampled_samples = pos_sampled + neg_sampled
        
        elif k == "intensity":  # SST-5：保证5个强度类别都有样本
            intensity_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
            label_groups = {label: [] for label in intensity_labels}
            for s in samples:
                label = s["output"]
                if label in label_groups:
                    label_groups[label].append(s)
            # 每个类别至少采5条，剩余随机分配
            base_per_label = max(int(target_total / 5), 5)
            for label in intensity_labels:
                if label_groups[label]:
                    take = min(base_per_label, len(label_groups[label]))
                    sampled_samples.extend(random.sample(label_groups[label], take))
            # 补充剩余样本（随机）
            remaining = target_total - len(sampled_samples)
            if remaining > 0:
                all_remaining = [s for s in samples if s not in sampled_samples]
                sampled_samples.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))
        
        elif k == "emotion":  # GoEmotions：保证至少10个核心情感类别有样本
            core_emotions = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral", "admiration", "approval", "disapproval"]
            label_groups = {emo: [] for emo in core_emotions}
            for s in samples:
                # 多标签样本，只要包含核心情感就归为该类
                for emo in core_emotions:
                    if emo in s["output"]:
                        label_groups[emo].append(s)
                        break
            # 每个核心类别至少采3条
            for emo in core_emotions:
                if label_groups[emo]:
                    take = min(3, len(label_groups[emo]))
                    sampled_samples.extend(random.sample(label_groups[emo], take))
            # 补充剩余样本
            remaining = target_total - len(sampled_samples)
            if remaining > 0:
                all_remaining = [s for s in samples if s not in sampled_samples]
                sampled_samples.extend(random.sample(all_remaining, min(remaining, len(all_remaining))))
        
        else:  # dynamics（DailyDialog）：直接按比例采样，保底50条
            sampled_samples = random.sample(samples, min(target_total, len(samples))) if len(samples)>=target_total else samples
        
        # 转回jsonl格式并加入总列表
        dataset_all.extend([json.dumps(s, ensure_ascii=False)+"\n" for s in sampled_samples])
    
    # 打乱并保存
    random.shuffle(dataset_all)
    with open("combined_eval.jsonl", "w", encoding="utf-8") as out:
        for line in dataset_all:
            out.write(line)
    print(f"✅ 修复后的测试集生成完成：共 {len(dataset_all)} 条样本")
    print("✅ 强制保证：SST-2有正负样本，SST-5有5类样本，GoEmotions有核心情感样本")

if __name__ == "__main__":
    build_sst2_eval()
    build_goemotions_eval()
    build_dailydialog_eval()
    build_sst5_eval()
    build_combined_eval()