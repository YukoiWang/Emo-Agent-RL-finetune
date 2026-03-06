## 心理咨询共情大模型微调项目

本项目提供一个完整的代码框架，用于基于开源大模型进行：

- 监督微调（SFT）：让模型学会高质量的心理咨询/共情对话范式  
- 强化学习微调（RL，如 PPO/GRPO/GSPO 等）：在特定奖励函数或偏好数据上进一步优化模型的共情能力与安全性

### 目录结构

- `configs/`：训练配置（模型、数据、训练超参等）
- `data/`：原始数据与处理后数据（占位目录）
- `src/`
  - `data/`：数据加载与预处理（SFT & RL）
  - `models/`：模型加载、LoRA 配置等
  - `training/`：SFT 和 RL 训练脚本
  - `evaluation/`：共情和安全性评估脚本
- `scripts/`：命令行启动脚本（按功能分子目录：sft / rl / reward / eval / data / classifier / tests）

### 环境准备

```bash
pip install -r requirements.txt
```

### 示例用法

- 运行 SFT 训练（示例）：

```bash
python scripts/sft/run_sft_empathetic.py \
  --config configs/sft_empathetic.yaml
```

- 运行 RL 训练（示例）：

```bash
python scripts/rl/run_rl.py \
  --config configs/rl_default.yaml
```
