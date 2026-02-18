from datasets import load_dataset

# 加载SST-2数据集的test分区
print("正在加载SST-2数据集的test分区...")
ds = load_dataset("sst2")
test_data = ds["test"]

# 统计标签分布（0=negative，1=positive）
label_counts = {0: 0, 1: 0}
all_sentences = []

print("\n开始统计test分区标签分布：")
for idx, sample in enumerate(test_data):
    label = sample["label"]
    label_counts[label] += 1
    all_sentences.append(sample["sentence"])
    
    # 打印前10条样本（直观验证）
    if idx < 10:
        sentiment = "negative" if label == 0 else "positive"
        print(f"样本{idx+1}：标签={label}({sentiment})，文本='{sample['sentence']}'")

# 输出最终统计结果
print("\n========== SST-2 test分区 标签分布 ==========")
print(f"negative样本数（label=0）：{label_counts[0]}")
print(f"positive样本数（label=1）：{label_counts[1]}")
print(f"test分区总样本数：{len(test_data)}")

# 关键验证：是否真的只有negative样本
if label_counts[1] == 0:
    print("\n❌ 确认问题：SST-2原生test分区确实只有negative样本！")
else:
    print("\n✅ SST-2原生test分区包含正负样本，之前的问题是采样逻辑导致的。")