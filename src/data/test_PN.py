# 检查测试集分布
import json
eval_data = []
with open("combined_eval.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        eval_data.append(json.loads(line.strip()))

# 检查SST-2极性任务
polarity_samples = [s for s in eval_data if "sentiment polarity" in s["instruction"].lower()]
pos_count = len([s for s in polarity_samples if s["output"] == "positive"])
neg_count = len([s for s in polarity_samples if s["output"] == "negative"])
print(f"SST-2测试集：positive={pos_count}, negative={neg_count}")

# 检查SST-5强度任务
intensity_samples = [s for s in eval_data if "sentiment intensity" in s["instruction"].lower()]
intensity_counts = {}
for s in intensity_samples:
    label = s["output"]
    intensity_counts[label] = intensity_counts.get(label, 0) + 1
print("SST-5测试集分布：", intensity_counts)