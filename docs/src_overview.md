# src 目录说明

本目录是项目核心代码：**数据加载**、**模型封装**、**训练与 RL 流程**（SFT / PPO / DPO / GRPO、多轮对话与用户模拟）。

---

## 一、目录与文件作用概览

### 1. `data/` — 数据

| 文件 | 作用 | 被谁用 |
|------|------|--------|
| `profile_dataset.py` | 从 `data/data` 读 profile jsonl，`load_profiles` / `build_initial_prompt` 生成多轮对话的「用户设定」 | PPO/DPO 多轮、GRPO、eval |
| `rl_dataset.py` | `load_rl_dataset`：支持 profile / 偏好对(jsonl)，统一成 HF Dataset | dpo_trainer, grpo_training, static-rl |
| `sft_dataset.py` | `load_sft_dataset`：心理咨询等 SFT 的 user/assistant 格式化 | sft_trainer |
| `virtual_rlhf_dataset.py` | VirtualRLHFDataset：用 profile + 用户模拟器首条消息构造虚拟 RL 数据 | **未被引用**（冗余，见下） |
| `build_emo_senti_dataset.py` | 英文多任务情感 SFT 数据构建（SST-2/GoEmotions 等）→ jsonl | 独立脚本 |
| `build_emo_senti_dataset_ch.py` | 中文情感 SFT 数据构建（微博、ChnSentiCorp 等）→ jsonl | 独立脚本 |
| `build_emo_test_en.py` | 英文多任务情感**测试集**构建（eval 用） | 独立脚本 |
| `test_sst2.py` | 检查 SST-2 标签分布（一次性统计） | 独立脚本 |
| `test_PN.py` | 检查 combined_eval 极性/强度分布 | 独立脚本 |

### 2. `models/` — 模型

| 文件 | 作用 |
|------|------|
| `modeling.py` | `load_base_model` / `load_sft_model`（LoRA 等），返回 `ModelAndTokenizer` |

### 3. `training/` — 训练与环境

| 文件 | 作用 | 备注 |
|------|------|------|
| **训练入口** | | |
| `sft_trainer.py` | SFT 训练入口（TRL SFTTrainer） | |
| `dpo_trainer.py` | **Offline DPO**：静态偏好对 jsonl | |
| `dpo_emo_trainer.py` | **On-policy DPO**：profile 多轮 → 每轮 k 个回复 → 打分造偏好对 | |
| `ppo_emo_trainer.py` | **多轮 PPO**：profile + 用户模拟器 → rollout → reward_emo → GAE/PPO | |
| `grpo_training.py` | **GRPO**：每 prompt G 个生成，组内归一化 advantage，无 Critic | |
| `rl_trainer.py` | 仅含 `simple_empathy_reward_fn`（规则 reward 示例） | 名字易误导，实为工具函数 |
| **PPO 核心** | | |
| `ppo_training.py` | PPOMemory、Critic、ActorRefRollout、训练步等通用 PPO 组件 | |
| `ppo_emo_rollout.py` | 多轮 rollout：`collect_rollouts_emo`，与 PlayerSimulatorWithPlanning 交互 | |
| `reward_emo.py` | 三种 reward 模式（emo_point/100、trend、volatility、warmup） | |
| **DPO rollout** | | |
| `dpo_emo_rollout.py` | On-policy DPO 多轮 rollout：每轮 k 个回复 → 选 best/worst 造偏好对 | |
| **用户/环境模拟** | | |
| `hard_player_simulator_dsv3.py` | PlayerSimulatorWithPlanning：多轮对话、planning_reply 更新 emo_point | |
| `qwen_user_simulator.py` | 用户回复生成（DashScope/DeepSeek 等 API） | |
| `emo_planning.py` | 用 LLM planning 模板分析 NPC 回复对情绪的影响 | |
| `local_planning_llm.py` | 本地 SFT 模型做 planning（不调 API） | |

---

## 二、冗余与可清理项

1. **`data/virtual_rlhf_dataset.py`**  
   - 定义了 `VirtualRLHFDataset`，**全仓库无 import**。  
   - 建议：删除或改为在文件头注释中标注「已废弃，仅保留作参考」。

2. **`training/rl_trainer.py`**  
   - 仅包含一个规则 reward 示例 `simple_empathy_reward_fn`，名字像「RL 训练器」容易误解。  
   - 建议：改名为 `reward_utils.py` 或把函数迁到 `reward_emo.py` 后删除本文件（需改 GRPO 等处的 import）。

3. **`data/` 下的 `build_*.py`、`test_*.py`**  
   - 本质是**独立可执行脚本**（数据构建与分布检查），不是被其他模块 import 的数据层。  
   - 建议：移到 `scripts/data/`（与 `download_empathetic_dialogues.py` 同层），使 `src/data/` 只保留「数据集加载与格式转换」的模块。

---

## 三、分类建议（若要进一步分子目录）

- **保持现状**：当前 `data/`、`models/`、`training/` 三分已经清晰，仅做上述清理即可。
- **若希望 training 再细分**（会涉及较多 import 修改）可考虑：
  - `training/trainers/`：sft_trainer, dpo_trainer, dpo_emo_trainer, ppo_emo_trainer, grpo_training
  - `training/rollout/`：ppo_emo_rollout, dpo_emo_rollout
  - `training/reward/`：reward_emo（以及从 rl_trainer 迁入的 reward 工具）
  - `training/simulator/`：hard_player_simulator_dsv3, qwen_user_simulator, emo_planning, local_planning_llm
  - `training/core/`：ppo_training（PPO 内存、Critic、Actor 等）

如需我按上述方案直接改仓库（删除/移动/重命名），可以说明要做到哪一步（仅文档 / 只清理冗余 / 含 data 脚本迁移 / 含 training 子目录拆分）。
