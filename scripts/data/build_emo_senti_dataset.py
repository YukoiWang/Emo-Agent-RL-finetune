# build_emotion_sft_dataset.py
import json
import random
from datasets import load_dataset
from tqdm import tqdm

random.seed(42)

# =========================
# Configuration
# =========================

files = { 
    "polarity": "sst2_sft.jsonl",
    "emotion": "goemotions_sft.jsonl",
    "dynamics": "dialog_shift_sft.jsonl",
    "intensity": "sst5_sft.jsonl"
}

R = {
    "polarity": 0.40,
    "emotion": 0.30,
    "dynamics": 0.20,
    "intensity": 0.10
}

# =========================
# Utility functions
# =========================

def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for x in data:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

# =========================
# 1. Polarity: SST-2
# =========================

def build_sst2():
    print("Downloading SST-2...")
    ds = load_dataset("sst2")

    out = []
    for x in tqdm(ds["train"]):
        label = "positive" if x["label"] == 1 else "negative"
        item = {
            "instruction": "Determine the sentiment polarity of the text (positive/negative)",
            "input": x["sentence"],
            "output": label
        }
        out.append(item)

    write_jsonl(files["polarity"], out)
    print(f"SST-2 saved -> {files['polarity']}")

# =========================
# 2. Emotion: GoEmotions
# =========================

def build_goemotions():
    print("Downloading GoEmotions...")
    ds = load_dataset("go_emotions")

    label_names = ds["train"].features["labels"].feature.names
    out = []

    for x in tqdm(ds["train"]):
        emotions = [label_names[i] for i in x["labels"]]
        item = {
            "instruction": "Identify the emotions in the text (multi-label possible)",
            "input": x["text"],
            "output": emotions
        }
        out.append(item)

    write_jsonl(files["emotion"], out)
    print(f"GoEmotions saved -> {files['emotion']}")

# =========================
# 3. Dynamics: DailyDialog Emotion (HF mirror)
# =========================

def build_dailydialog():
    print("Downloading DailyDialog from HF mirror...")
    ds = load_dataset("roskoN/dailydialog")

    emo_map = {
        0: "neutral",
        1: "anger",
        2: "disgust",
        3: "fear",
        4: "happiness",
        5: "sadness",
        6: "surprise"
    }

    out = []

    for dialog in tqdm(ds["train"]):
        utterances = dialog["utterances"]
        emotions = dialog["emotions"]

        for i in range(len(utterances)-1):
            item = {
                "instruction": "Identify emotion changes between dialogue turns",
                "input": f"Previous: {utterances[i]}\nCurrent: {utterances[i+1]}",
                "output": {
                    "prev_emotion": emo_map[emotions[i]],
                    "curr_emotion": emo_map[emotions[i+1]]
                }
            }
            out.append(item)

    write_jsonl(files["dynamics"], out)
    print(f"DailyDialog saved -> {files['dynamics']}")

# =========================
# 4. Intensity: SST-5
# =========================

def build_sst5():
    print("Downloading SST-5...")
    ds = load_dataset("SetFit/sst5")

    intensity_map = {
        0: "very negative",
        1: "negative",
        2: "neutral",
        3: "positive",
        4: "very positive"
    }

    out = []
    for x in tqdm(ds["train"]):
        item = {
            "instruction": "Determine the sentiment intensity level of the text",
            "input": x["text"],
            "output": intensity_map[x["label"]]
        }
        out.append(item)

    write_jsonl(files["intensity"], out)
    print(f"SST-5 saved -> {files['intensity']}")

# =========================
# 5. Combine all datasets into combined_sft.jsonl
# =========================

def build_combined():
    print("Building combined_sft.jsonl ...")
    dataset_all = []

    for k, path in files.items():
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        target_size = int(len(lines) * R[k])
        sampled = random.sample(lines, target_size)
        dataset_all.extend(sampled)

    random.shuffle(dataset_all)

    with open("combined_sft.jsonl", "w", encoding="utf-8") as out:
        for line in dataset_all:
            out.write(line)

    print("combined_sft.jsonl built successfully.")

# =========================
# Main
# =========================

if __name__ == "__main__":
    build_sst2()
    build_goemotions()
    build_dailydialog()
    build_sst5()
    build_combined()
