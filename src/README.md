# src — 核心代码

- **data/**：数据加载（profile / RL / SFT）与格式转换；数据构建脚本已移至 `scripts/data/`。
- **models/**：基座与 SFT 模型加载（LoRA 等）。
- **training/**：SFT、PPO、DPO、GRPO 训练入口，rollout，reward，用户模拟器。

各子目录见其下 `README.md`；完整文件说明与冗余分析见 **docs/src_overview.md**。
