# static-rl: EmpatheticDialogues 静态 RL 流程

基于 **EmpatheticDialogues** 数据集和 **Plutchik 情绪轮** 构建的完整 RL 微调流程，包含：

1. 偏好对数据集构建（Plutchik 对立情绪）
2. DPO 训练
3. Reward Model 训练
4. PPO 训练（RM 作为奖励）
5. GRPO 训练（RM 作为奖励）

## 流程概览

```
EmpatheticDialogues (HF)
        │
        ▼
build_empathetic_preference_dataset.py  →  static-rl/data/empathetic_preference.jsonl
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
run_dpo.py                         run_reward_model.py
(DPO 微调)                          (训练 Reward Model)
        │                                  │
        ▼                                  ▼
static-rl/outputs/dpo              static-rl/outputs/reward_model
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                              ▼
              run_ppo.py                                    run_grpo.py
              (PPO + RM)                                    (GRPO + RM)
                    │                                              │
                    ▼                                              ▼
        static-rl/outputs/ppo                        static-rl/outputs/grpo
```

## 1. 偏好对数据集

基于 **Plutchik 情绪轮** 四组对立关系：
- Joy ↔ Sadness
- Trust ↔ Disgust
- Fear ↔ Anger
- Anticipation ↔ Surprise

- **preferred (chosen)**: EmpatheticDialogues 真实人类回复（共情回答）
- **non-preferred (rejected)**: 从语料库随机抽取的「对立情绪」标签回复

```bash
python static-rl/build_empathetic_preference_dataset.py --output static-rl/data/empathetic_preference.jsonl
# 可选: --max-samples 10000 限制样本数
```

## 2. DPO 训练

```bash
python static-rl/run_dpo.py --config static-rl/configs/dpo.yaml
```

## 3. Reward Model 训练

```bash
python static-rl/run_reward_model.py --config static-rl/configs/reward_model.yaml
```

## 4. PPO 训练

从偏好数据中采样 prompt，使用训练好的 Reward Model 打分：

```bash
python static-rl/run_ppo.py --config static-rl/configs/ppo.yaml
```

## 5. GRPO 训练

同样从 EmpatheticDialogues 衍生数据采样 prompt，使用 Reward Model：

```bash
python static-rl/run_grpo.py --config static-rl/configs/grpo.yaml
```

## Base Model 与 全量 / LoRA 模式

| 组件 | Base Model | LoRA 配置 | 全量配置 |
|------|------------|-----------|----------|
| **Reward Model** | Qwen/Qwen2.5-0.5B-Instruct | reward_model.yaml | reward_model_full.yaml |
| **DPO** | SFT: outputs/sft_empathetic/final | dpo.yaml | dpo_full.yaml |
| **PPO** | 同上 | ppo.yaml | ppo_full.yaml |
| **GRPO** | 同上 | grpo.yaml | grpo_full.yaml |

- **Reward Model**：RM 用 0.5B 作为 base，RL（DPO/PPO/GRPO）用 SFT 后的 1.5B/7B 作为 policy。
- **全量微调**：`use_lora: false`，学习率更小，显存占用更高。
- **LoRA 微调**：`use_lora: true`（默认），只训练 adapter，显存占用更低。

## 配置说明

- **SFT 模型路径**：RL 微调使用 EmpatheticDialogues SFT 后的模型。若尚未完成 SFT，配置中默认使用 `outputs/sft_empathetic/final` 作为占位路径，需先完成 SFT 或修改为实际路径。
- **数据路径**：所有脚本默认从 `static-rl/data/empathetic_preference.jsonl` 读取偏好数据。
- **输出目录**：DPO / RM / PPO / GRPO 的输出分别保存在 `static-rl/outputs/dpo`、`reward_model`、`ppo`、`grpo`；全量版本对应 `*_full` 子目录。

## 前置依赖

- 完成 EmpatheticDialogues SFT：`python scripts/sft/run_sft_empathetic.py`（或等价流程）
- 或修改配置中的 `sft_model_path` 为可用基础模型路径
