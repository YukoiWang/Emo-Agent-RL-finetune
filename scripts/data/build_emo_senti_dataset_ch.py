# build_chinese_sft_dataset.py
import json
import random
from datasets import load_dataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random.seed(42)

# =========================
# 配置
# =========================

# 本地输出文件
OUT_WEIBO_TRAIN = "weibo_train.jsonl"
OUT_WEIBO_VAL   = "weibo_val.jsonl"
OUT_WEIBO_TEST  = "weibo_test.jsonl"

OUT_CHN_TRAIN   = "chnsenticorp_train.jsonl"
OUT_CHN_VAL     = "chnsenticorp_val.jsonl"
OUT_CHN_TEST    = "chnsenticorp_test.jsonl"

COMBINED_TRAIN  = "combined_sft_zh_train.jsonl"
COMBINED_VAL    = "combined_sft_zh_val.jsonl"
COMBINED_TEST   = "combined_sft_zh_test.jsonl"

# 各子集在 combined 中的采样比例
R_zh = {
    "weibo": 0.5,
    "chn": 0.5
}

# 验证/测试集比例
VALID_RATIO = 0.05
TEST_RATIO  = 0.05

# =========================
# 工具函数
# =========================

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# =========================
# 1. Weibo Senti 100k
# =========================

def build_weibo():
    print("下载 & 处理 Weibo Senti 100k ...")
    ds = load_dataset("dirtycomputer/weibo_senti_100k")

    label_map = {0: "消极", 1: "中性", 2: "积极"}
    out = []

    for x in tqdm(ds["train"]):
        out.append({
            "instruction": "判断文本的情感倾向（积极/消极/中性）",
            "input": x["review"],
            "output": label_map[x["label"]]
        })

    # 先拆 train + temp
    train_set, temp_set = train_test_split(out, test_size=(VALID_RATIO+TEST_RATIO), random_state=42)
    # 再拆 val 和 test
    val_set, test_set = train_test_split(temp_set, test_size=(TEST_RATIO/(VALID_RATIO+TEST_RATIO)), random_state=42)

    write_jsonl(OUT_WEIBO_TRAIN, train_set)
    write_jsonl(OUT_WEIBO_VAL,   val_set)
    write_jsonl(OUT_WEIBO_TEST,  test_set)

    print(f"Weibo Senti 划分完成: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

# =========================
# 2. ChnSentiCorp
# =========================

def build_chnsenticorp():
    print("下载 & 读取 ChnSentiCorp ...")
    ds = load_dataset("lansinuote/ChnSentiCorp", trust_remote_code=True)

    label_map = {0: "消极", 1: "积极"}

    def convert(split):
        out = []
        for x in tqdm(ds[split]):
            out.append({
                "instruction": "判断文本的情感倾向（积极/消极）",
                "input": x["text"],
                "output": label_map[x["label"]]
            })
        return out

    train_set = convert("train")
    val_set   = convert("validation")
    test_set  = convert("test")

    write_jsonl(OUT_CHN_TRAIN, train_set)
    write_jsonl(OUT_CHN_VAL,   val_set)
    write_jsonl(OUT_CHN_TEST,  test_set)

    print(f"ChnSentiCorp 读取完成: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

# =========================
# 3. 合并所有中文数据集
# =========================

def build_combined_zh():
    print("合并三类 train / val / test ...")

    # 读入 weibo
    with open(OUT_WEIBO_TRAIN, "r", encoding="utf-8") as f:
        weibo_train = [json.loads(x) for x in f.readlines()]
    with open(OUT_WEIBO_VAL, "r", encoding="utf-8") as f:
        weibo_val   = [json.loads(x) for x in f.readlines()]
    with open(OUT_WEIBO_TEST, "r", encoding="utf-8") as f:
        weibo_test  = [json.loads(x) for x in f.readlines()]

    # 读入 chnSenti
    with open(OUT_CHN_TRAIN, "r", encoding="utf-8") as f:
        chn_train = [json.loads(x) for x in f.readlines()]
    with open(OUT_CHN_VAL, "r", encoding="utf-8") as f:
        chn_val   = [json.loads(x) for x in f.readlines()]
    with open(OUT_CHN_TEST, "r", encoding="utf-8") as f:
        chn_test  = [json.loads(x) for x in f.readlines()]

    # 采样按比例融合
    train_all = []
    val_all   = []
    test_all  = []

    train_all.extend(random.sample(weibo_train, int(len(weibo_train)*R_zh["weibo"])))
    train_all.extend(random.sample(chn_train,  int(len(chn_train)*R_zh["chn"])))

    val_all.extend(random.sample(weibo_val, int(len(weibo_val)*R_zh["weibo"])))
    val_all.extend(random.sample(chn_val,  int(len(chn_val)*R_zh["chn"])))

    test_all.extend(random.sample(weibo_test, int(len(weibo_test)*R_zh["weibo"])))
    test_all.extend(random.sample(chn_test,  int(len(chn_test)*R_zh["chn"])))

    random.shuffle(train_all)
    random.shuffle(val_all)
    random.shuffle(test_all)

    write_jsonl(COMBINED_TRAIN, train_all)
    write_jsonl(COMBINED_VAL,   val_all)
    write_jsonl(COMBINED_TEST,  test_all)

    print(f"Combined 数据集构建完成: train={len(train_all)}, val={len(val_all)}, test={len(test_all)}")

# =========================
# 主入口
# =========================

if __name__ == "__main__":
    build_weibo()
    build_chnsenticorp()
    build_combined_zh()
