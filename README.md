## 心理咨询共情大模型微调项目

本项目提供一个完整的代码框架，用于基于开源大模型进行：

- 监督微调（SFT）：让模型学会高质量的心理咨询/共情对话范式  
- 强化学习微调（RL，如 PPO/DPO 等）：在特定奖励函数或偏好数据上进一步优化模型的共情能力与安全性

### 目录结构

- `configs/`：训练配置（模型、数据、训练超参等）
- `data/`：原始数据与处理后数据（占位目录）
- `src/`
  - `data/`：数据加载与预处理（SFT & RL）
  - `models/`：模型加载、LoRA 配置等
  - `training/`：SFT 和 RL 训练脚本
  - `evaluation/`：共情和安全性评估脚本（占位）
- `scripts/`：命令行启动脚本

### 环境准备

```bash
cd /home/yukiwang/xlwy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 示例用法

- 运行 SFT 训练（示例）：

```bash
python scripts/run_sft.py \
  --config configs/sft_default.yaml
```

- 运行 RL（PPO/DPO 风格）训练（示例）：

```bash
python scripts/run_rl.py \
  --config configs/rl_default.yaml
```

### 使用 IPM-PrefDial 进行 DPO 训练

[IPM-PrefDial](https://github.com/Zc0812/DecoupledESC) 是情感支持对话（ESC）的偏好对数据集，来自 DecoupledESC 论文。

**步骤：**

1. 转换数据为 DPO 格式（从网络自动下载）：

```bash
python scripts/convert_ipm_prefdial.py -o data/ipm_prefdial_dpo.jsonl
```

2. 运行 DPO 训练：

```bash
python scripts/run_rl.py --config configs/rl_dpo.yaml
```

可选：指定本地 JSON 或使用 tokenizer 的 chat template：

```bash
# 使用本地文件
python scripts/convert_ipm_prefdial.py -i /path/to/ESC_Qwen_RG_dpo.json -o data/ipm_prefdial_dpo.jsonl

# 使用 Qwen chat template 格式化 prompt
python scripts/convert_ipm_prefdial.py -o data/ipm_prefdial_dpo.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
```

