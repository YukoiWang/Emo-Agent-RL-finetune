# PPO 多轮对话 + 情绪分 Reward 使用说明

## 概述

- **数据**：`/home/yukiwang/xlwy/data/data` 下的 `train_profile.jsonl` / `test_profile.jsonl`，每行为用户形象（player、scene、task/隐藏主题）。
- **用户模拟器**：`PlayerSimulatorWithPlanning`（`hard_player_simulator_dsv3.py`）用外部 LLM API 当「用户」，与 Actor 多轮对话；每轮用 planning_reply（LLM prompt）更新 `emo_point`（0–100）。
- **情绪分析**：`emo_classifier_lora` 微调的情感分类模型分析用户回复的积极/消极/中性，并结合 NPC 与隐藏主题贴合程度得到 `change_value`，用于更新 `emo_point`。
- **结束条件**：用户说「再见/拜拜」或 `emo_point <= 0` 时结束对话。
- **Reward**：为**函数**（非 reward model），支持三种模式，通过参数切换。
- **RL 前 SFT**：可先用心理咨询对话数据集做 SFT（`configs/sft_counseling.yaml` + `scripts/sft/run_sft_counseling.py`），再用该模型做 PPO。

## Reward 模式

### 模式 1（`reward_mode="mode1"`）

- **标量**：`reward_batched = data.non_tensor_batch['emo_point'] / 100`，再 `np.maximum(..., 0)`，即把情绪分归一化到 [0, 1]。
- **张量**：该标量放到每条回复的**最后一个有效 token** 位置，得到 `original_reward_tensor` 和 `penalized_reward_tensor`（当前实现两者相同）。
- 对应 trainer 中的 `original_reward` 与 `token_level_scores`；GAE 用标量在末位即可。

### 模式 2（`reward_mode="mode2"`）

- **基础**：模式 1 的最终情绪分作为 baseline。
- **趋势奖励**：最近 `n` 轮（默认 5）情绪变化趋势（线性斜率），上升 → 正向 reward，下降或无明显改善 → 低或 0。
- **波动惩罚**：最近 `n` 轮情绪方差，波动大 → 减少总体 reward。
- **组合**：  
  `r_total = w1 * baseline_emotion + w2 * trend_reward - w3 * volatility_penalty`  
  `w1`、`w2`、`w3` 可调（如 `--w1 1.0 --w2 0.3 --w3 0.2`），`trend_n` 控制用于趋势/波动的轮数。

### 模式 3（`reward_mode="mode3"`）三段式训练

- **公式**：  
  `alpha = clamp((step - S1) / warmup_steps, 0, 1)`  
  `beta  = clamp((step - S2) / warmup_steps, 0, 1)`  
  `reward = baseline + alpha * w2 * trend - beta * w3 * volatility`
- **含义**：随训练步数 `step` 先 warmup 趋势项（alpha），再 warmup 波动惩罚（beta），便于分阶段稳定训练。需传入 `--S1`、`--S2`、`--warmup_steps`。
- **阶段**：可在 step 到达 S1、S2 及训练结束时各存一次 checkpoint，用于「mode3 各 stage」的评估（见 `scripts/eval/eval_all_models.py`）。

## 脚本与调用

- **心理咨询 SFT（RL 前）**：`scripts/sft/run_sft_counseling.py --config configs/sft_counseling.yaml`  
  - 数据：`configs/sft_counseling.yaml` 中 `data.train_file` / `eval_file`（默认 `data/counseling_dialogue/train.jsonl`）。每行 jsonl 需含 `user`、`assistant`，可选 `system`。  
  - 输出：`outputs/sft_counseling`（可作 RL 的 sft_model_path）。

- **入口脚本**：`scripts/rl/run_ppo_emo.py`  
  - `--data_dir`：profile 数据目录（默认 `data/data`）。  
  - `--emo_adapter`：情绪分类 LoRA 路径（默认 `emo_classifier_lora/checkpoint-11025`）。  
  - `--reward_mode`：`mode1`、`mode2` 或 `mode3`。  
  - `--w1`, `--w2`, `--w3`, `--trend_n`：mode2/mode3 生效。  
  - `--S1`, `--S2`, `--warmup_steps`：仅 mode3 生效。  
  - `--user_llm mock`：用户模拟器用占位回复；可扩展为真实 LLM API。

- **多轮 Rollout**：`src/training/ppo_emo_rollout.py`  
  - `run_multi_turn_rollout_batch`：对一批 profile 做多轮对话，返回 `response_ids`、`response_mask`、`log_probs`、`values` 及 `non_tensor_batch['emo_point']`、`emo_point_turns`。  
  - `collect_rollouts_emo`：在上述结果上算 reward（调用 `reward_emo.compute_reward_tensors`），并把标量 reward 与序列写入 `PPOMemory`，返回 `(original_reward_tensor, penalized_reward_tensor)`。

- **Reward 计算**：`src/training/reward_emo.py`  
  - `compute_reward_tensors(...)`：根据 `emo_points` 与可选的 `emo_point_turns_list`、`reward_mode` 与权重，得到末位非零的 reward 张量。  
  - 返回 `(original_reward_tensor, penalized_reward_tensor)`。

## 与 PPO 训练器对接

1. 用 `ProfileDataset` + `DataLoader` 从 `data/data` 取 profile，构造每批 `batch_items`（含 `profile`、`prompt`）。
2. 每步调用 `collect_rollouts_emo(..., reward_mode=..., step=current_step, S1=..., S2=..., warmup_steps=...)`（mode3 时传入 step/S1/S2/warmup_steps），将 rollout 写入 `memory`，并得到 `(original_reward_tensor, penalized_reward_tensor)` 用于日志或 token 级 score。
3. 照常执行 `train_step()`（GAE + PPO 更新），然后 `memory.clear()`，进入下一步。

这样即可在现有 `ppo_training` 框架下，用「多轮对话 + 情绪分」的 reward 函数（模式 1 或 2）进行训练。

---

## OOM 调优与多卡训练

### 单卡仍 OOM 时可尝试的配置

在已有 `gradient_checkpointing: true`、`batch_size: 2`、`mini_batch_size: 1` 前提下，可继续减小显存占用：

| 参数 | 当前值 | 建议尝试 | 说明 |
|------|--------|----------|------|
| `data.batch_size` | 2 | 1 | 每步 rollout 样本数 |
| `data.max_scene_len` | 1500 | 800 或 500 | 场景文本最大长度 |
| `data.max_prompt_length` | 256 | 192 | prompt 截断 |
| `rollout.max_turns` | 8 | 4 | 减少对话轮数 |
| `rollout.max_new_tokens_per_turn` | 128 | 64 | 每轮生成长度 |
| `rl.ppo.ppo_epochs` | 4 | 2 | PPO 更新轮数 |

### 多卡训练（PPO / GRPO）

使用 HuggingFace Accelerate 做 DDP。**命令行启动**：

```bash
# 2 卡
accelerate launch --num_processes=2 scripts/rl/run_rl.py --config configs/rl_default.yaml

# 4 卡
accelerate launch --num_processes=4 scripts/rl/run_rl.py --config configs/rl_default.yaml
```

**SLURM 提交**：使用 `submit_ppo_multi.sh`（修改 `#SBATCH --gres=gpu:N` 指定卡数）：

```bash
sbatch submit_ppo_multi.sh
```

多卡时，DataLoader 自动按 rank 分片，每卡独立 rollout + 更新，梯度跨卡同步，有效 batch = `batch_size × num_gpus`。

---

## 权重 w1、w2、w3 怎么调优？

- **Adam 用在哪**：Adam 用来训练 PPO 的 **policy（actor）和 critic** 的参数，不是用来更新 w1、w2、w3。w1、w2、w3 是 **reward 公式里的超参数**，不参与反向传播。
- **怎么找最优权重**：把 w1、w2、w3（以及可选的 `trend_n`）当作超参数做搜索，常用两种方式：
  1. **自动搜索（推荐）**：用 **Optuna** 等做贝叶斯/TPE 搜索，在验证集上最大化你关心的指标（例如平均最终 emo_point、或「平均 reward」、或 PPO 若干步后的验证表现）。见下方脚本 `scripts/reward/tune_reward_weights.py`。
  2. **手动/网格**：先设 w1=1, w2=w3=0（等价 mode1），再逐步加大 w2、w3 做 ablation，看验证集上最终情绪分和训练稳定性。

目标指标可以选其一或组合使用：
- 验证集上 **平均最终 emo_point**（越高越好）；
- 验证集上 **平均 r_total**（在你定的 w1,w2,w3 下算出的 reward）；
- 或 PPO 训练若干步后 **验证集 mean emo_point**（更准但更耗算力）。

**自动搜索脚本**：`scripts/reward/tune_reward_weights.py`（需安装 `pip install optuna`）。

- 用法：先准备验证集 rollout 结果，每行一个 JSON：`{"emo_point": 65, "emo_point_turns": [50,52,...,65]}`，保存为 `val_rollouts.jsonl`。可在跑验证集多轮对话时，把每条样本的 `non_tensor_batch` 里 `emo_point` 和 `emo_point_turns` 写出。
- 运行：`python scripts/reward/tune_reward_weights.py --val_rollouts val_rollouts.jsonl --n_trials 50 --metric mean_reward`
- 输出：最优的 `w1, w2, w3, trend_n`；将这些参数填入 mode2 的 PPO 训练即可。
- 不提供 `--val_rollouts` 时会用示例数据试跑，仅作脚本测试用。

---

## 全模型评估（base / SFT / SFT+RL mode1/2/3 / mode3 各 stage）

**脚本**：`scripts/eval/eval_all_models.py`

- **对比模型**：base、sft_only（仅心理咨询 SFT）、sft_rl_mode1、sft_rl_mode2、sft_rl_mode3、sft_rl_mode3_stage1/2/3（三段式各阶段 checkpoint）。路径可在脚本内 `DEFAULT_MODEL_PATHS` 或通过 `--model_paths` 覆盖。
- **评估维度**：  
  1. **Sentient-Benchmark**（情感智能）：成功/失败对话率、共情深度、核心洞察、总体分（占位数据路径 `data/eval/sentient_benchmark.jsonl`，可替换为真实 benchmark）。  
  2. **情绪改善**：终端情绪分、情绪轨迹改善、成功/失败率。  
  3. **综合能力**（防遗忘）：MATH500、LiveCodeBench、IFEval（当前为占位，可接 OpenCompass 等）。
- **运行示例**：  
  `python scripts/eval/eval_all_models.py --models base sft_only sft_rl_mode1 sft_rl_mode2 sft_rl_mode3 sft_rl_mode3_stage1 sft_rl_mode3_stage2 sft_rl_mode3_stage3 --output outputs/eval_all_results.json`  
  若仅测脚本可加 `--skip_load`（不加载模型，仅输出占位指标）。
