# -*- coding: utf-8 -*-
"""
全模型评估脚本：对比 base、SFT-only（心理咨询）、SFT+RL(mode1/2/3)、以及 mode3 各 stage。
评估维度：Sentient-Benchmark（情感/共情）、情绪改善指标、对话质量、综合能力（防遗忘）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# 1) 模型配置：名称 -> 路径（或通过命令行覆盖）
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATHS = {
    "base": "/home/yukiwang/models/Qwen2-7B-Instruct",
    "sft_only": "outputs/sft_counseling",
    "sft_rl_mode1": "outputs/ppo_emo_mode1",
    "sft_rl_mode2": "outputs/ppo_emo_mode2",
    "sft_rl_mode3": "outputs/ppo_emo_mode3",
    "sft_rl_mode3_stage1": "outputs/ppo_emo_mode3/checkpoint-S1",
    "sft_rl_mode3_stage2": "outputs/ppo_emo_mode3/checkpoint-S2",
    "sft_rl_mode3_stage3": "outputs/ppo_emo_mode3/final",
}


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """加载因果 LM 与 tokenizer。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = model_path if os.path.isabs(model_path) else os.path.join(ROOT, model_path)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 2) Sentient-Benchmark（情感智能基准）占位实现
# 论文指标：成功/失败对话率、共情深度、核心洞察、总体分。此处用虚拟数据或本地 jsonl。
# ---------------------------------------------------------------------------
SENTIENT_BENCHMARK_PLACEHOLDER = "data/eval/sentient_benchmark.jsonl"


def run_sentient_benchmark(
    model,
    tokenizer,
    eval_data_path: str,
    device: str,
    max_new_tokens: int = 256,
) -> Dict[str, float]:
    """
    在 Sentient-Benchmark 风格数据上评估：多轮对话 + 多维度打分。
    数据格式占位：每行 {"messages": [{"role":"user","content":...}, ...], "reference_insight": "..."}，
    或仅 {"prompt": "用户最后一句", "context": "历史"}，模型生成回复后由打分器/规则给出维度分。
    返回: success_rate, failure_rate, empathy_depth, core_insight, overall_score (0-100).
    """
    if not os.path.isfile(eval_data_path):
        # 虚拟结果，便于脚本跑通
        return {
            "success_rate": 0.0,
            "failure_rate": 0.0,
            "empathy_depth": 0.0,
            "core_insight": 0.0,
            "overall_score": 0.0,
        }
    results = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            # 占位：实际应对 item 做 prompt 构造、model.generate、再调用打分器
            results.append({
                "success": 1,
                "empathy_depth": 0.5,
                "core_insight": 0.5,
            })
    if not results:
        return {"success_rate": 0.0, "failure_rate": 0.0, "empathy_depth": 0.0, "core_insight": 0.0, "overall_score": 0.0}
    n = len(results)
    success_rate = sum(r.get("success", 0) for r in results) / n
    failure_rate = 1.0 - success_rate
    empathy_depth = sum(r.get("empathy_depth", 0) for r in results) / n
    core_insight = sum(r.get("core_insight", 0) for r in results) / n
    overall_score = (empathy_depth + core_insight) * 50.0  # 粗略映射到 0-100
    return {
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "empathy_depth": empathy_depth,
        "core_insight": core_insight,
        "overall_score": overall_score,
    }


# ---------------------------------------------------------------------------
# 3) 情绪改善指标：终端情绪分、情绪轨迹、成功/失败率
# 需要多轮对话 + 用户模拟器或固定测试集带 emo_point_turns。
# ---------------------------------------------------------------------------
def run_emotion_metrics(
    model,
    tokenizer,
    profile_data_path: str,
    device: str,
    emo_analyzer_fn=None,
    user_llm_fn=None,
    max_turns: int = 10,
    max_new_tokens: int = 128,
) -> Dict[str, float]:
    """
    在 profile 数据上跑多轮对话，收集终端情绪分、轨迹、是否成功/失败。
    成功：终端情绪 >= 阈值或相对起点改善；失败：终端 < 起点或显著恶化。
    返回: terminal_emotion_mean, trajectory_improvement_mean, success_rate, failure_rate.
    """
    profile_path = profile_data_path if os.path.isabs(profile_data_path) else os.path.join(ROOT, profile_data_path)
    profiles = []
    if os.path.isfile(profile_path):
        with open(profile_path, "r", encoding="utf-8") as f:
            profiles = [json.loads(line) for line in f if line.strip()][:5]
    else:
        try:
            from src.data.profile_dataset import load_profiles
            data_dir = profile_path if os.path.isdir(profile_path) else (os.path.dirname(profile_path) or ROOT)
            profiles = load_profiles(data_dir, "test")[:5]
        except Exception:
            pass
    if not profiles:
        return {
            "terminal_emotion_mean": 0.0,
            "trajectory_improvement_mean": 0.0,
            "success_rate": 0.0,
            "failure_rate": 0.0,
        }

    terminal_scores = []
    improvements = []
    success_count = 0
    failure_count = 0
    initial_default = 50.0
    success_threshold = 0.55  # 终端 emo/100 >= 0.55 视为成功
    failure_threshold = 0.45  # 终端 < 起点或 < 0.45 视为失败

    for pro in profiles:
        # 占位：若未接入 simulator，用固定数值
        end_emo = initial_default + 10.0
        start_emo = initial_default
        terminal_scores.append(end_emo / 100.0)
        improvements.append((end_emo - start_emo) / 100.0)
        if end_emo / 100.0 >= success_threshold:
            success_count += 1
        if end_emo < start_emo or end_emo / 100.0 < failure_threshold:
            failure_count += 1

    n = max(len(profiles), 1)
    return {
        "terminal_emotion_mean": sum(terminal_scores) / n,
        "trajectory_improvement_mean": sum(improvements) / n,
        "success_rate": success_count / n,
        "failure_rate": failure_count / n,
    }


# ---------------------------------------------------------------------------
# 4) 综合能力（防灾难性遗忘）：MATH500、LiveCodeBench、IFEval 占位
# ---------------------------------------------------------------------------
def run_general_capability(
    model,
    tokenizer,
    device: str,
    math_path: str = "",
    code_path: str = "",
    ifeval_path: str = "",
) -> Dict[str, float]:
    """
    逻辑/数学、代码、指令遵循。占位：可接 OpenCompass / 各 benchmark 的 run 脚本。
    返回: math_score (0-1), code_score (0-1), ifeval_score (0-1).
    """
    return {
        "math_score": 0.0,
        "code_score": 0.0,
        "ifeval_score": 0.0,
    }


# ---------------------------------------------------------------------------
# 5) 主流程：多模型 × 多指标，输出表格与 json
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="全模型评估：base / SFT / SFT+RL(mode1/2/3) / mode3 各 stage")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="要评估的模型名，如 base sft_only sft_rl_mode1 sft_rl_mode2 sft_rl_mode3 sft_rl_mode3_stage1 sft_rl_mode3_stage2 sft_rl_mode3_stage3")
    parser.add_argument("--model_paths", type=str, default=None,
                        help="JSON 或 key=value 覆盖默认路径，如 '{\"base\":\"/path/to/base\"}'")
    parser.add_argument("--sentient_data", type=str, default=os.path.join(ROOT, "data/eval/sentient_benchmark.jsonl"))
    parser.add_argument("--profile_data", type=str, default=os.path.join(ROOT, "data/data/test_profile.jsonl"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=os.path.join(ROOT, "outputs/eval_all_results.json"))
    parser.add_argument("--skip_load", action="store_true", help="不加载模型，仅用占位指标（用于测脚本）")
    args = parser.parse_args()

    model_names = args.models or list(DEFAULT_MODEL_PATHS.keys())
    paths = dict(DEFAULT_MODEL_PATHS)
    if args.model_paths:
        if args.model_paths.strip().startswith("{"):
            paths.update(json.loads(args.model_paths))
        else:
            for part in args.model_paths.split(","):
                k, v = part.split("=", 1)
                paths[k.strip()] = v.strip()

    sentient_path = args.sentient_data if os.path.isabs(args.sentient_data) else os.path.join(ROOT, args.sentient_data)
    profile_path = args.profile_data if os.path.isabs(args.profile_data) else os.path.join(ROOT, args.profile_data)

    all_results: Dict[str, Dict[str, Any]] = {}

    for name in model_names:
        if name not in paths:
            print(f"跳过未知模型: {name}")
            continue
        path = paths[name]
        print(f"评估模型: {name} -> {path}")

        if args.skip_load:
            model, tokenizer = None, None
        else:
            try:
                model, tokenizer = load_model_and_tokenizer(path, args.device)
            except Exception as e:
                print(f"  加载失败: {e}")
                all_results[name] = {"error": str(e)}
                continue

        out: Dict[str, Any] = {}

        # Sentient-Benchmark
        if model is not None and tokenizer is not None:
            out["sentient_benchmark"] = run_sentient_benchmark(model, tokenizer, sentient_path, args.device)
        else:
            out["sentient_benchmark"] = run_sentient_benchmark(None, None, sentient_path, args.device)

        # 情绪指标
        out["emotion_metrics"] = run_emotion_metrics(model, tokenizer, profile_path, args.device)

        # 综合能力
        out["general_capability"] = run_general_capability(model, tokenizer, args.device)

        all_results[name] = out

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # 打印简表
    print("\n============= 评估结果摘要 =============")
    for name, res in all_results.items():
        if "error" in res:
            print(f"{name}: 错误 - {res['error']}")
            continue
        sb = res.get("sentient_benchmark", {})
        em = res.get("emotion_metrics", {})
        gc = res.get("general_capability", {})
        print(f"{name}:")
        print(f"  Sentient: 总体分={sb.get('overall_score', 0):.2f}, 成功率={sb.get('success_rate', 0):.2f}, 失败率={sb.get('failure_rate', 0):.2f}, 共情={sb.get('empathy_depth', 0):.2f}, 洞察={sb.get('core_insight', 0):.2f}")
        print(f"  情绪: 终端情绪={em.get('terminal_emotion_mean', 0):.2f}, 轨迹改善={em.get('trajectory_improvement_mean', 0):.2f}, 成功率={em.get('success_rate', 0):.2f}, 失败率={em.get('failure_rate', 0):.2f}")
        print(f"  综合: math={gc.get('math_score', 0):.2f}, code={gc.get('code_score', 0):.2f}, ifeval={gc.get('ifeval_score', 0):.2f}")
    print(f"\n完整结果已写入: {args.output}")


if __name__ == "__main__":
    main()
