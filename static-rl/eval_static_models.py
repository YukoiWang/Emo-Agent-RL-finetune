#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Static-RL 全模型评估脚本：对比 Base / SFT / DPO / PPO / GRPO 在同一测试集上的表现。

评估维度：
  1. RM 偏好准确率（RM 给 chosen 的分数 > rejected 的比例）
  2. RM Score（模型生成回复的 RM 打分）
  3. 多样性指标（Distinct-1/2、平均长度）
  4. LLM-as-Judge（可选，需配置 API）
  5. 各模型生成示例对比（人工审查）

用法:
  python static-rl/eval_static_models.py --config static-rl/configs/eval.yaml

  # 或直接指定关键路径
  python static-rl/eval_static_models.py \
    --data data/ipm_prefdial_dpo.jsonl \
    --rm-path static-rl/outputs/reward_model \
    --model-paths '{"dpo":"static-rl/outputs/dpo/final","ppo":"static-rl/outputs/ppo/final","grpo":"static-rl/outputs/grpo/final"}' \
    --base-model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "static-rl"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    PeftModel = None
    _HAS_PEFT = False


# =========================================================================
# 1. 数据加载
# =========================================================================

def load_test_data(
    data_path: str,
    test_ratio: float = 0.1,
    max_test: int = 200,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """从偏好对 JSONL 中划分 train/test；返回 (train_data, test_data)。"""
    path = Path(data_path) if Path(data_path).is_absolute() else ROOT / data_path
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.seed(seed)
    random.shuffle(records)
    n_test = min(max_test, max(1, int(len(records) * test_ratio)))
    test_data = records[:n_test]
    train_data = records[n_test:]
    print(f"[数据] 总计 {len(records)} 条, 测试集 {len(test_data)} 条")
    return train_data, test_data


# =========================================================================
# 2. 模型加载
# =========================================================================

def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> Tuple[Any, Any]:
    """加载 CausalLM 模型（支持完整权重或 LoRA adapter）。"""
    path = model_path if os.path.isabs(model_path) else str(ROOT / model_path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"模型路径不存在: {path}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    adapter_config = Path(path) / "adapter_config.json"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if device == "cuda" else None

    if _HAS_PEFT and adapter_config.exists():
        with open(adapter_config, "r") as f:
            base_name = json.load(f).get("base_model_name_or_path") or base_model_name
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch_dtype, device_map=device_map, trust_remote_code=True,
        )

    model.eval()
    return model, tokenizer


# =========================================================================
# 3. 生成回复
# =========================================================================

@torch.no_grad()
def generate_responses(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    batch_size: int = 4,
) -> List[str]:
    """批量生成回复。prompts 应该已经是带模板的完整 prompt。"""
    all_responses = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for i, ids in enumerate(out):
            prompt_len = enc["input_ids"][i].shape[0]
            resp_ids = ids[prompt_len:]
            text = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
            all_responses.append(text)
    return all_responses


# =========================================================================
# 4. RM 偏好准确率
# =========================================================================

def eval_rm_preference_accuracy(
    rm_scorer,
    test_data: List[Dict],
    batch_size: int = 8,
) -> Dict[str, float]:
    """计算 RM 在 held-out 测试集上的偏好准确率。"""
    correct = 0
    total = 0
    score_gaps = []

    for start in range(0, len(test_data), batch_size):
        batch = test_data[start:start + batch_size]
        prompts = [d["user"] for d in batch]
        chosens = [d["chosen"] for d in batch]
        rejecteds = [d["rejected"] for d in batch]

        scores_c = rm_scorer.score(prompts, chosens)
        scores_r = rm_scorer.score(prompts, rejecteds)

        for sc, sr in zip(scores_c, scores_r):
            total += 1
            if sc > sr:
                correct += 1
            score_gaps.append(sc - sr)

    accuracy = correct / total if total > 0 else 0.0
    avg_gap = sum(score_gaps) / len(score_gaps) if score_gaps else 0.0
    return {
        "rm_accuracy": round(accuracy, 4),
        "rm_avg_score_gap": round(avg_gap, 4),
        "rm_total_pairs": total,
    }


# =========================================================================
# 5. RM Score 评估
# =========================================================================

def eval_rm_scores(
    rm_scorer,
    prompts: List[str],
    responses: List[str],
    human_responses: Optional[List[str]] = None,
    batch_size: int = 8,
) -> Dict[str, float]:
    """用 RM 对模型生成的回复打分。"""
    all_scores = []
    for start in range(0, len(prompts), batch_size):
        p = prompts[start:start + batch_size]
        r = responses[start:start + batch_size]
        scores = rm_scorer.score(p, r)
        all_scores.extend(scores)

    result = {
        "rm_score_mean": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "rm_score_max": round(max(all_scores), 4) if all_scores else 0.0,
        "rm_score_min": round(min(all_scores), 4) if all_scores else 0.0,
    }

    if human_responses:
        human_scores = []
        for start in range(0, len(prompts), batch_size):
            p = prompts[start:start + batch_size]
            r = human_responses[start:start + batch_size]
            scores = rm_scorer.score(p, r)
            human_scores.extend(scores)
        result["rm_score_human_mean"] = round(
            sum(human_scores) / len(human_scores), 4
        ) if human_scores else 0.0

    return result


# =========================================================================
# 6. 多样性指标
# =========================================================================

def eval_diversity(responses: List[str]) -> Dict[str, float]:
    """计算 Distinct-1/2 和平均长度。"""
    if not responses:
        return {"distinct_1": 0, "distinct_2": 0, "avg_length": 0, "avg_length_chars": 0}

    all_unigrams = []
    all_bigrams = []
    total_chars = 0
    total_tokens = 0

    for r in responses:
        tokens = r.split()
        total_tokens += len(tokens)
        total_chars += len(r)
        all_unigrams.extend(tokens)
        all_bigrams.extend(zip(tokens[:-1], tokens[1:]))

    distinct_1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
    distinct_2 = len(set(all_bigrams)) / max(len(all_bigrams), 1)

    n = len(responses)
    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "avg_length_tokens": round(total_tokens / n, 1),
        "avg_length_chars": round(total_chars / n, 1),
    }


# =========================================================================
# 7. LLM-as-Judge（可选）
# =========================================================================

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for empathetic dialogue systems.

Given the user's message and an AI response, rate the response on 4 dimensions (1-5 each).

User: {user_message}
AI Response: {response}

Rate strictly as JSON:
{{"empathy": <1-5>, "relevance": <1-5>, "coherence": <1-5>, "helpfulness": <1-5>}}

Scoring guide:
- 5: Excellent  4: Good  3: Adequate  2: Poor  1: Very Poor
- empathy: Does the response show genuine understanding of the user's emotions?
- relevance: Is the response on-topic and addresses the user's concern?
- coherence: Is the response fluent, well-formed, and logically consistent?
- helpfulness: Does it provide meaningful emotional support or useful advice?
"""


def eval_llm_judge(
    prompts: List[str],
    responses: List[str],
    api_base: str = "",
    api_key: str = "",
    model: str = "deepseek-chat",
    max_samples: int = 50,
) -> Dict[str, float]:
    """用 LLM 对回复打分（需要 OpenAI-compatible API）。"""
    if not api_base or not api_key:
        return {"llm_judge": "skipped (no API configured)"}

    try:
        from openai import OpenAI
    except ImportError:
        return {"llm_judge": "skipped (openai not installed)"}

    client = OpenAI(base_url=api_base, api_key=api_key)
    scores_all = {"empathy": [], "relevance": [], "coherence": [], "helpfulness": []}
    n = min(max_samples, len(prompts))

    for i in range(n):
        user_msg = prompts[i]
        if "<|im_start|>user\n" in user_msg:
            user_msg = user_msg.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()

        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            user_message=user_msg, response=responses[i],
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            content = resp.choices[0].message.content.strip()
            parsed = json.loads(content)
            for k in scores_all:
                if k in parsed:
                    scores_all[k].append(float(parsed[k]))
        except Exception:
            continue

    result = {}
    for k, v in scores_all.items():
        result[f"judge_{k}"] = round(sum(v) / len(v), 2) if v else 0.0
    result["judge_samples"] = n
    result["judge_valid"] = min(len(v) for v in scores_all.values()) if scores_all else 0
    return result


# =========================================================================
# 8. 主评估流程
# =========================================================================

def run_single_model_eval(
    model_name: str,
    model_path: str,
    test_data: List[Dict],
    rm_scorer,
    base_model_name: str,
    device: str,
    max_new_tokens: int,
    api_base: str = "",
    api_key: str = "",
    judge_model: str = "deepseek-chat",
) -> Dict[str, Any]:
    """对单个模型运行全部评估。"""
    print(f"\n{'='*60}")
    print(f"  评估模型: {model_name} -> {model_path}")
    print(f"{'='*60}")

    t0 = time.time()

    # 加载模型
    try:
        model, tokenizer = load_model_and_tokenizer(model_path, device, base_model_name)
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return {"error": str(e)}

    prompts = [d["user"] for d in test_data]
    human_chosen = [d["chosen"] for d in test_data]

    # 生成回复
    print(f"  生成回复 ({len(prompts)} 条) ...")
    tokenizer.padding_side = "left"
    responses = generate_responses(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens,
    )

    # 释放模型显存
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    result: Dict[str, Any] = {"model_path": model_path}

    # RM Score
    if rm_scorer is not None:
        print("  计算 RM Score ...")
        rm_result = eval_rm_scores(rm_scorer, prompts, responses, human_chosen)
        result.update(rm_result)

    # 多样性
    print("  计算多样性指标 ...")
    div_result = eval_diversity(responses)
    result.update(div_result)

    # LLM-as-Judge
    if api_base and api_key:
        print("  运行 LLM-as-Judge ...")
        judge_result = eval_llm_judge(
            prompts, responses,
            api_base=api_base, api_key=api_key, model=judge_model,
        )
        result.update(judge_result)

    # 保存几个生成示例
    examples = []
    for i in range(min(5, len(prompts))):
        user_msg = prompts[i]
        if "<|im_start|>user\n" in user_msg:
            user_msg = user_msg.split("<|im_start|>user\n")[-1].split("<|im_end|>")[0].strip()
        examples.append({
            "prompt": user_msg[:200],
            "human": human_chosen[i][:300],
            "model": responses[i][:300],
        })
    result["examples"] = examples

    elapsed = time.time() - t0
    result["eval_time_sec"] = round(elapsed, 1)
    print(f"  ✅ 完成 ({elapsed:.1f}s)")
    return result


def print_summary_table(all_results: Dict[str, Dict]):
    """打印对比摘要表格。"""
    print("\n" + "=" * 90)
    print("  评估结果摘要")
    print("=" * 90)

    metrics = [
        ("rm_score_mean", "RM Score"),
        ("rm_score_human_mean", "RM (Human)"),
        ("distinct_1", "Distinct-1"),
        ("distinct_2", "Distinct-2"),
        ("avg_length_chars", "Avg Len"),
        ("judge_empathy", "J:Empathy"),
        ("judge_relevance", "J:Relevance"),
        ("judge_coherence", "J:Coherence"),
        ("judge_helpfulness", "J:Helpful"),
    ]

    header = f"{'Model':<15}"
    for _, label in metrics:
        header += f" {label:>12}"
    print(header)
    print("-" * len(header))

    for name, res in all_results.items():
        if "error" in res:
            print(f"{name:<15}  错误: {res['error']}")
            continue
        row = f"{name:<15}"
        for key, _ in metrics:
            val = res.get(key, "-")
            if isinstance(val, float):
                row += f" {val:>12.4f}"
            else:
                row += f" {str(val):>12}"
        print(row)

    print()


def print_examples(all_results: Dict[str, Dict]):
    """打印各模型生成示例对比。"""
    models_with_examples = {
        k: v for k, v in all_results.items()
        if "examples" in v and v["examples"]
    }
    if not models_with_examples:
        return

    first_model = next(iter(models_with_examples.values()))
    n = len(first_model["examples"])

    for i in range(min(3, n)):
        print(f"\n{'─'*70}")
        print(f"  示例 {i+1}")
        print(f"{'─'*70}")

        prompt = first_model["examples"][i]["prompt"]
        human = first_model["examples"][i]["human"]
        print(f"  Prompt: {prompt}")
        print(f"  Human:  {human}")

        for name, res in models_with_examples.items():
            if i < len(res["examples"]):
                model_resp = res["examples"][i]["model"]
                print(f"  {name}: {model_resp}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Static-RL 全模型评估")
    parser.add_argument("--config", type=str, default=None, help="评估配置 YAML")
    parser.add_argument("--data", type=str, default=None, help="偏好对数据 JSONL 路径")
    parser.add_argument("--rm-path", type=str, default=None, help="Reward Model 路径")
    parser.add_argument("--model-paths", type=str, default=None,
                        help='JSON dict: {"dpo":"path","ppo":"path","grpo":"path"}')
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="LoRA 的 base model 名称")
    parser.add_argument("--sft-model", type=str, default=None, help="SFT 模型路径（作为对比基线）")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--max-test", type=int, default=200, help="测试集最大条数")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="生成最大 token 数")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="static-rl/outputs/eval_results.json")
    parser.add_argument("--api-base", type=str, default="", help="LLM Judge API base URL")
    parser.add_argument("--api-key", type=str, default="", help="LLM Judge API key")
    parser.add_argument("--judge-model", type=str, default="deepseek-chat")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 从 YAML 配置加载（优先级低于命令行参数）
    cfg = {}
    if args.config:
        config_path = Path(args.config) if Path(args.config).is_absolute() else ROOT / args.config
        if config_path.exists():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

    data_path = args.data or cfg.get("data", {}).get("test_file") or cfg.get("data", {}).get("train_file", "")
    rm_path = args.rm_path or cfg.get("reward", {}).get("reward_model_path", "")
    base_model = args.base_model or cfg.get("model", {}).get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
    sft_model = args.sft_model or cfg.get("model", {}).get("sft_model_path", "")
    device = args.device if torch.cuda.is_available() else "cpu"
    api_base = args.api_base or cfg.get("judge", {}).get("api_base", "")
    api_key = args.api_key or cfg.get("judge", {}).get("api_key", "")
    judge_model = args.judge_model or cfg.get("judge", {}).get("model", "deepseek-chat")

    # 解析 model_paths
    model_paths: Dict[str, str] = {}
    if args.model_paths:
        if args.model_paths.strip().startswith("{"):
            model_paths = json.loads(args.model_paths)
        else:
            for part in args.model_paths.split(","):
                k, v = part.split("=", 1)
                model_paths[k.strip()] = v.strip()
    elif "models" in cfg:
        model_paths = cfg["models"]

    if not data_path:
        parser.error("需要指定 --data 或在 config 中配置 data.train_file")

    # 加载测试数据
    _, test_data = load_test_data(
        data_path, test_ratio=args.test_ratio,
        max_test=args.max_test, seed=args.seed,
    )

    # 加载 RM
    rm_scorer = None
    if rm_path:
        rm_full = Path(rm_path) if Path(rm_path).is_absolute() else ROOT / rm_path
        if rm_full.exists():
            print(f"[RM] 加载 Reward Model: {rm_full}")
            from reward_model_scorer import RewardModelScorer
            rm_scorer = RewardModelScorer(str(rm_full), device=device)

            print("[RM] 评估偏好准确率 ...")
            rm_acc = eval_rm_preference_accuracy(rm_scorer, test_data)
            print(f"  RM Accuracy: {rm_acc['rm_accuracy']:.4f} "
                  f"({rm_acc['rm_total_pairs']} pairs, avg gap: {rm_acc['rm_avg_score_gap']:.4f})")
        else:
            print(f"[RM] 路径不存在: {rm_full}, 跳过 RM 评估")

    # 评估各模型
    all_results: Dict[str, Dict] = {}

    if rm_scorer:
        all_results["_rm_accuracy"] = eval_rm_preference_accuracy(rm_scorer, test_data)

    # SFT 基线
    if sft_model:
        all_results["SFT"] = run_single_model_eval(
            "SFT", sft_model, test_data, rm_scorer, base_model, device,
            args.max_new_tokens, api_base, api_key, judge_model,
        )

    # RL 模型
    for name, path in model_paths.items():
        all_results[name] = run_single_model_eval(
            name, path, test_data, rm_scorer, base_model, device,
            args.max_new_tokens, api_base, api_key, judge_model,
        )

    # 输出结果
    output_path = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n完整结果已保存到: {output_path}")

    # 打印摘要
    display_results = {k: v for k, v in all_results.items() if not k.startswith("_")}
    print_summary_table(display_results)
    print_examples(display_results)

    if "_rm_accuracy" in all_results:
        rm_acc = all_results["_rm_accuracy"]
        print(f"[RM 偏好准确率] {rm_acc['rm_accuracy']:.2%} "
              f"(avg score gap: {rm_acc['rm_avg_score_gap']:.4f})")


if __name__ == "__main__":
    main()
