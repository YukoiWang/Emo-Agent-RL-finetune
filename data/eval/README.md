# 评估数据说明

## Sentient-Benchmark（情感智能基准）

用于 `scripts/eval_all_models.py` 的 `--sentient_data`（默认 `data/eval/sentient_benchmark.jsonl`）。

- **格式占位**：每行一个 JSON，例如  
  `{"messages": [{"role":"user","content":"..."}, ...], "reference_insight": "..."}`  
  或 `{"prompt": "用户最后一句", "context": "历史"}`。  
  脚本会据此构造输入、调用模型生成回复，再由打分器得到各维度分。
- **维度**：成功/失败对话率、共情深度（Empathy Depth）、核心洞察（Core Insight）、总体分（0–100）。
- 若文件不存在，脚本会使用占位指标（全 0）保证流程可跑通；替换为真实 benchmark 数据后即可得到真实分数。

## 情绪指标用 profile

`--profile_data` 默认指向 `data/data/test_profile.jsonl`（与训练用 profile 同格式）。  
评估时用这些 profile 做多轮对话（或使用已有对话结果），得到终端情绪分、轨迹改善、成功/失败率。

## 综合能力（MATH500 / LiveCodeBench / IFEval）

当前为占位。可对接 OpenCompass 或各 benchmark 官方脚本，在 `eval_all_models.py` 的 `run_general_capability()` 中调用并填入 `math_score`、`code_score`、`ifeval_score`。
