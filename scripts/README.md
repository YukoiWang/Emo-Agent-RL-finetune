# Scripts 目录说明

按**算法/功能**分类的脚本入口，从项目根目录执行时请使用对应子路径。

## 目录结构

| 子目录 | 作用 | 脚本 |
|--------|------|------|
| **sft/** | 监督微调 (SFT) | `run_sft_empathetic.py`（共情对话）、`run_sft_counseling.py`（心理咨询对话） |
| **rl/** | 强化学习 (PPO/DPO/GRPO) | `run_rl.py`、`run_ppo_emo.py`、`run_dpo_emo.py`、`run_quick_verify.py`、`run_reward_comparison.py` |
| **reward/** | 奖励设计 / 超参搜索 | `tune_reward_weights.py`（Optuna 搜权重）、`compare_reward_modes.py`（三种 reward 对比） |
| **eval/** | 评估与可视化 | `eval_rl_models.py`、`eval_all_models.py`、`plot_rl_curves.py` |
| **data/** | 数据下载与准备 | `download_empathetic_dialogues.py`、`build_emo_senti_dataset.py`、`build_emo_senti_dataset_ch.py`、`build_emo_test_en.py`、`test_sst2.py`、`test_PN.py` |
| **classifier/** | 情感分类器（训练与测试） | `run_emo_senti_base.py`、`run_emo_senti_sft.py`、`test_emo_senti_*.py`、`test_compare_sen.py` |
| **tests/** | 端到端与烟雾测试 | `test_ppo_e2e.py`、`test_grpo_e2e.py`、`smoke_test_simulator.py`、`smoke_test_dry.py` |

## 常用命令示例

```bash
# 数据
python scripts/data/download_empathetic_dialogues.py

# SFT
python scripts/sft/run_sft_empathetic.py --config configs/sft_empathetic.yaml
python scripts/sft/run_sft_counseling.py --config configs/sft_counseling.yaml

# RL
python scripts/rl/run_rl.py --config configs/rl_default.yaml
python scripts/rl/run_dpo_emo.py --config configs/rl_dpo_emo.yaml
python scripts/rl/run_quick_verify.py

# Reward
python scripts/reward/compare_reward_modes.py
python scripts/reward/tune_reward_weights.py --val_rollouts val_rollouts.jsonl

# 评估
python scripts/eval/plot_rl_curves.py --log-dir outputs/quick_verify
python scripts/eval/eval_rl_models.py --model-dir outputs/quick_verify
python scripts/eval/eval_all_models.py --quick-verify-dir outputs/quick_verify
```
