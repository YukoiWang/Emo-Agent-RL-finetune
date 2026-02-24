#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 EmpatheticDialogues 构建偏好对数据集。

构造方法（Plutchik 情绪轮）：
- preferred (chosen): 使用 EmpatheticDialogues 中真实的人类回复，对上下文最共情
- non-preferred (rejected): 从语料库中随机抽取带「对立情绪」标签的回复，
  这类回答在情绪上与当前 prompt 不匹配，不具有共情意义

Plutchik 情绪轮四组对立关系：
  Joy ↔ Sadness
  Trust ↔ Disgust
  Fear ↔ Anger
  Anticipation ↔ Surprise

用法:
  python static-rl/build_empathetic_preference_dataset.py --output static-rl/data/empathetic_preference.jsonl
  python static-rl/build_empathetic_preference_dataset.py --max-samples 10000 --output static-rl/data/empathetic_preference.jsonl
"""
import argparse
import json
import random
import re
from pathlib import Path
from collections import defaultdict


# EmpatheticDialogues 32 情绪 -> Plutchik 8 类映射
# Plutchik: Joy, Sadness, Trust, Disgust, Fear, Anger, Anticipation, Surprise
PLUTCHIK_OPPOSITES = {
    "joy": "sadness",
    "sadness": "joy",
    "trust": "disgust",
    "disgust": "trust",
    "fear": "anger",
    "anger": "fear",
    "anticipation": "surprise",
    "surprise": "anticipation",
}

# EmpatheticDialogues 32 情绪 -> Plutchik 8 类
# 根据情绪语义映射到 Plutchik 基本情绪
ED_TO_PLUTCHIK = {
    "afraid": "fear",
    "angry": "anger",
    "annoyed": "anger",
    "anticipating": "anticipation",
    "anxious": "fear",
    "appreciative": "joy",
    "ashamed": "sadness",
    "caring": "trust",
    "confident": "joy",
    "content": "joy",
    "devastated": "sadness",
    "disappointed": "sadness",
    "disgusted": "disgust",
    "embarrassed": "sadness",
    "excited": "joy",
    "faithful": "trust",
    "frightened": "fear",
    "grateful": "joy",
    "guilty": "sadness",
    "hopeful": "anticipation",
    "impressed": "surprise",
    "jealous": "anger",
    "joyful": "joy",
    "lonely": "sadness",
    "nostalgic": "sadness",
    "prepared": "anticipation",
    "proud": "joy",
    "sad": "sadness",
    "sentimental": "sadness",
    "stressed": "fear",
    "surprised": "surprise",
    "terrified": "fear",
    "trusting": "trust",
}


# 从 system 字符串提取情绪标签，例如 "expressing sentimental emotions" -> "sentimental"
_EMOTION_FROM_SYSTEM = re.compile(r"expressing\s+(\w+)\s+emotions?", re.I)


def _load_local_empathetic_jsonl(path: str) -> list[dict]:
    """从本地 jsonl（user/assistant/system）加载，返回与 HF 格式兼容的列表：prompt, utterance, context。"""
    out = []
    path = Path(path)
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = (obj.get("user") or "").strip()
            assistant = (obj.get("assistant") or "").strip()
            system = (obj.get("system") or "").strip()
            if not user or not assistant:
                continue
            match = _EMOTION_FROM_SYSTEM.search(system)
            context = match.group(1).lower() if match else "neutral"
            out.append({"prompt": user, "utterance": assistant, "context": context})
    return out


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return (
        text.replace("_comma_", ",")
        .replace("_period_", ".")
        .replace("_exclamation_", "!")
        .replace("_question_", "?")
        .strip()
    )


def get_opposite_plutchik(emotion: str, fallback_map: dict | None = None) -> str:
    """根据 Plutchik 情绪轮返回对立情绪（Plutchik 类名）。"""
    emo_lower = emotion.lower().strip()
    mapping = {**(fallback_map or {}), **ED_TO_PLUTCHIK}
    plutchik = mapping.get(emo_lower, "joy")  # 未知情绪默认 joy
    return PLUTCHIK_OPPOSITES.get(plutchik, "sadness")


def build_preference_dataset(
    output_path: str,
    max_samples: int = 0,
    seed: int = 42,
    include_system: bool = True,
    local_data_path: str | None = None,
) -> None:
    random.seed(seed)

    # 优先使用本地 jsonl（避免 HF dataset 脚本已废弃导致的加载失败）
    repo_root = Path(__file__).resolve().parent.parent
    default_local = repo_root / "data" / "empathetic_dialogues" / "train.jsonl"
    train: list[dict] = []

    if local_data_path or default_local.exists():
        path = Path(local_data_path) if local_data_path else default_local
        if not path.is_absolute():
            path = repo_root / path
        print(f"Loading from local data: {path}")
        train = _load_local_empathetic_jsonl(str(path))
        if not train:
            raise FileNotFoundError(f"No valid records in {path}")
    else:
        try:
            from datasets import load_dataset
            print("Loading facebook/empathetic_dialogues from Hugging Face...")
            ds = load_dataset("facebook/empathetic_dialogues")
            train = list(ds["train"])
        except RuntimeError as e:
            if "no longer supported" in str(e) or "Dataset scripts" in str(e):
                raise FileNotFoundError(
                    "Hugging Face no longer supports this dataset script. "
                    f"Please use local data: put train.jsonl at {default_local} or pass --local-data <path>"
                ) from e
            raise

    # 按情绪构建 utterance 索引：emotion -> [utterance1, utterance2, ...]
    emotion_to_utterances = defaultdict(list)
    for ex in train:
        ctx = (ex.get("context") or "").strip().lower()
        utt = _normalize_text(ex.get("utterance", ""))
        if ctx and utt:
            emotion_to_utterances[ctx].append(utt)

    # 未知情绪补充到映射（保持向后兼容），尝试相似匹配
    fallback = {}
    for emo in emotion_to_utterances:
        if emo not in ED_TO_PLUTCHIK:
            # 尝试部分匹配
            matched = next((k for k in ED_TO_PLUTCHIK if k in emo or emo in k), None)
            fallback[emo] = ED_TO_PLUTCHIK.get(matched, "joy") if matched else "joy"

    print(f"Found {len(emotion_to_utterances)} emotions, building preference pairs...")

    samples = []
    for ex in train:
        prompt = _normalize_text(ex.get("prompt", ""))
        utterance = _normalize_text(ex.get("utterance", ""))
        context = (ex.get("context") or "").strip()

        if not prompt or not utterance or not context:
            continue

        # 对立情绪（Plutchik）
        opposite_plutchik = get_opposite_plutchik(context, fallback)
        # 找出语料库中属于对立 Plutchik 类的所有情绪
        all_plutchik = {**ED_TO_PLUTCHIK, **fallback}
        opposite_emotions = [
            e for e, pl in all_plutchik.items()
            if pl == opposite_plutchik and e in emotion_to_utterances and emotion_to_utterances[e]
        ]

        if not opposite_emotions:
            continue

        # 随机选一个对立情绪，再从中随机选一条回复作为 rejected
        opp_emotion = random.choice(opposite_emotions)
        rejected_utterances = emotion_to_utterances[opp_emotion]
        rejected = random.choice(rejected_utterances)

        # 构造 user prompt（与 SFT 格式一致，方便后续训练）
        if include_system:
            user = (
                f"<|im_start|>system\n"
                f"You are an empathetic listener. The user is expressing {context} emotions.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            user = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        samples.append({
            "user": user,
            "chosen": utterance,
            "rejected": rejected,
            "emotion": context,
            "opposite_emotion": opp_emotion,
        })

        if max_samples and len(samples) >= max_samples:
            break

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            out = {"user": s["user"], "chosen": s["chosen"], "rejected": s["rejected"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved {len(samples)} preference pairs to {output_path}")
    print("Format: user (prompt), chosen (preferred), rejected (non-preferred)")


def main():
    parser = argparse.ArgumentParser(
        description="Build empathetic preference dataset from EmpatheticDialogues (Plutchik opposites)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="static-rl/data/empathetic_preference.jsonl",
        help="Output jsonl path",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to generate (0 = all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Do not include system prompt with emotion",
    )
    parser.add_argument(
        "--local-data",
        type=str,
        default=None,
        help="Path to local train.jsonl (user/assistant/system). Default: data/empathetic_dialogues/train.jsonl",
    )
    args = parser.parse_args()

    build_preference_dataset(
        output_path=args.output,
        max_samples=args.max_samples,
        seed=args.seed,
        include_system=not args.no_system,
        local_data_path=args.local_data,
    )


if __name__ == "__main__":
    main()
