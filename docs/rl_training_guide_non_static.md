# 非 static-rl 路径下的 RL 训练：原理、流程与用法

本文档说明 **static-rl 以外** 的 RL 训练：入口在 `scripts/rl/run_rl.py` 和 `scripts/rl/run_quick_verify.py`，实现分布在 `src/training/` 下。

---

## 一、整体架构：有哪些算法、用什么数据

| 算法 | 入口 / 实现 | 数据来源 | 训练方式 |
|------|-------------|----------|----------|
| **PPO（多轮）** | `run_rl.py --config ...` algo=ppo → `ppo_emo_trainer.run_ppo_emo_training` | Profile（`data/data/train_profile.jsonl` 或 `data_dir`） | 多轮对话 rollout → GAE → PPO 更新 |
| **GRPO** | `run_rl.py` algo=grpo → `grpo_training.run_grpo_training` | 同上 Profile，转成 `user` prompt | 每 prompt 采样 G 个回复 → 组内 reward 归一化 → clip + KL |
| **DPO（on-policy）** | `run_rl.py` algo=dpo_emo → `dpo_emo_trainer.run_dpo_emo_training` | 同上 Profile | 多轮对话每轮 k 个回复 → reward 选 best/worst → 构造偏好对 → DPO |
| **DPO（offline）** | 需单独调 `dpo_trainer.run_dpo_training`（如写小脚本或接 static-rl 配置） | 偏好对 jsonl：`user`, `chosen`, `rejected` | 静态数据，TRL DPOTrainer |

当前 **`scripts/rl/run_rl.py` 只支持三种 algo**：`ppo`、`grpo`、`dpo_emo`。  
Offline DPO 不在 `run_rl.py` 里，要用 `configs/rl_dpo.yaml` + 自己写一行调用 `run_dpo_training(cfg)`，或参考 `static-rl/run_dpo.py`。

---

## 二、数据：用什么、用多少

### 1. Profile 数据（PPO / GRPO / DPO-Emo）

- **路径**：由 `data.train_file` 或 `data.data_dir` 决定。  
  常见：`data/data/train_profile.jsonl` 或目录 `data/data`（`ProfileDataset` / `load_profiles` 会从该目录读 `train_profile.jsonl` 等）。
- **格式**：每行一个 profile（如 `player`、`scene`、`task` 等），由 `build_initial_prompt` 转成一段「用户开场」的 prompt，用于多轮对话或单轮生成。
- **用多少**：
  - **PPO**：`data.batch_size` 每步取多少个 profile 做 rollout；总步数由 `training.total_steps` 控制，数据会循环使用。
  - **GRPO**：同一套 Profile 经 `load_rl_dataset` 转成 `user` prompt，`DataLoader(dataset, batch_size=1)` 每步 1 个 prompt，总步数 `training.total_steps`。
  - **DPO-Emo**：`rollout.num_profiles` 限制参与 rollout 的 profile 数量（如 20），`rollout.batch_size` 每批几个 profile，多轮对话中每轮生成 `rollout.num_samples` 个回复用来造偏好对。

### 2. 偏好对数据（仅 Offline DPO）

- **路径**：`data.train_file`，例如 `data/ipm_prefdial_dpo.jsonl`（需先用 `scripts/convert_ipm_prefdial.py` 等生成）。
- **格式**：每行 `{"user": "prompt", "chosen": "优选回复", "rejected": "劣选回复"}`；DPO 读取时会改列名为 `prompt`（若原为 `user`）。
- **用多少**：由 `training.num_train_epochs` 和 `training.max_steps` 决定；`max_steps=-1` 表示跑完所有 epoch。

---

## 三、各算法原理与流程（结合代码）

### 1. PPO（多轮）：`ppo_emo_trainer.run_ppo_emo_training`

- **思路**：用 **Profile** 当「用户设定」，在 **用户模拟器**（mock 或 LLM）下和当前 policy 多轮对话，得到多条 (query, response) 轨迹；对每条轨迹用 **reward_emo**（mode1/2/3）算 reward，再做 **GAE**、**PPO** 更新。
- **流程概览**：
  1. 加载 SFT 模型 → 封装成 `ActorRefRollout`，单独加载 **Critic**（frozen backbone + 可训练 head）。
  2. `ProfileDataset` 从 `data_dir` 读 profile，`DataLoader` 按 `batch_size` 取一批 profile。
  3. 每步对这批 profile 调用 `collect_rollouts_emo`：
     - 用 `build_initial_prompt` 得到用户开场；
     - 多轮：policy 生成回复 → `user_llm_fn` 生成用户下一句 → 直到 `max_turns` 或结束；
     - 使用 planning LLM 给出每步的 emo 相关信号；
     - 用 `reward_emo`（mode1/2/3）算每条轨迹的 reward（可带 trend、volatility、warmup 等）；
     - 写入 `PPOMemory`（log_probs, ref_log_probs, values, rewards）。
  4. `memory.get(compute_gae=True)` 得到 advantages/returns，算 policy loss（clip）、value loss、KL penalty，反向传播更新 Actor + Critic。
  5. 每 `save_steps` 保存 checkpoint，最后保存到 `output_dir/final`；同时写 `training_log.jsonl`（step, reward_mean, policy/value/kl loss）。

- **Reward 三种模式**（`reward.reward_mode`）：
  - **mode1**：reward = 终点 emo_point / 100。
  - **mode2**：reward = w1×baseline + w2×trend_reward − w3×volatility_penalty。
  - **mode3**：在三段式 warmup 下，alpha/beta 随 step 变化，再组合 baseline + trend − volatility。

- **步数**：`training.total_steps`（例如 500、1000）；每步 = 一批 profile 的 rollout + 一次 PPO update。

### 2. GRPO：`grpo_training.run_grpo_training`

- **数据**：和 PPO 一样用 **Profile**。通过 `load_rl_dataset(train_file=data_cfg["train_file"], format="auto")` 加载；若检测到 profile 格式（`player` + `scene`/`task`），会用 `build_initial_prompt` 转成 **单条 `user` prompt**（开场/场景描述），因此 **GRPO 用的是 profile，只是用法是「一条 profile → 一条 prompt」**，不做多轮对话。
- **思路**：**无 Critic**；每个 prompt 采样 **G 个** completion，用 `reward_fn` 打分，组内做 **相对归一化** 得到 advantage，再用 **clipped policy gradient + KL penalty**（对 frozen ref）更新当前 policy。
- **Rollout**：有 rollout，但是 **单轮、多采样**——每步 1 个 prompt，模型生成 G 条 response，reward 对这 G 条打分；没有用户模拟器、没有多轮交互。
- **流程概览**：
  1. 加载 SFT 模型 → actor；`copy.deepcopy` 一份 **ref_model** 冻结。
  2. `load_rl_dataset` 读 Profile（或其它 jsonl），**若为 profile 则自动转成 `user`**；`DataLoader(dataset, batch_size=1)`。
  3. 每步：
     - 取 1 个 prompt（即一条 profile 对应的 user 文本），`_generate_completions(actor, ..., num_generations=G)` 得到 G 条 response 文本；
     - `reward_fn(resp_texts)` 得到 G 个标量；
     - advantage = (rewards - mean) / (std + eps)；
     - 对每条 response 算 actor log_probs、ref log_probs，ratio = exp(actor_lp - old_lp)，surr1/surr2 = ratio * adv 与 clip(ratio) * adv，policy_loss = -min(surr1,surr2)；kl_loss = (old_lp - ref_lp).mean()；
     - loss = policy_loss + kl_coef * kl_loss，反向传播只更新 actor。
  4. 每 `logging_steps` 打印 loss/reward；每 `save_steps` 保存 checkpoint；最后保存 `final`，并写 `training_log.jsonl`（step, reward_mean, loss, kl_loss）。

- **步数**：`training.total_steps`（如 100、5000）；每步 = 1 个 prompt × G 个生成 + 1 次梯度更新。

### 3. DPO（on-policy）：`dpo_emo_trainer.run_dpo_emo_training`

- **思路**：用 **Profile** 当用户设定，多轮对话；每轮对当前回复生成 **k 个** 候选，用 **score_fn**（planning 或 reward_emo）打分，选 best 和 worst 构成 (chosen, rejected) 偏好对，攒够一批后交给 **TRL DPOTrainer** 训练。
- **流程概览**：
  1. 加载 SFT 模型、从 `data_dir` 取 profile（数量 `rollout.num_profiles`）。
  2. 若 `use_planning_score` 且非 mock，构建 `planning_llm_fn`（或本地 planning 模型）；否则用 `build_reward_fn_emo` 当 score。
  3. 对 profile 做 `run_dpo_rollout_batch`：多轮对话，每轮生成 k 个回复，score 选 best/worst，得到 DPO 偏好对列表。
  4. 偏好对转成 `Dataset`（prompt, chosen, rejected），`DPOTrainer.train()`；保存到 `output_dir`。  
  DPO 的 step 数由「多少条偏好对」和 `per_device_train_batch_size`、`gradient_accumulation_steps` 等决定，也可用 `max_steps` 截断。

### 4. DPO（offline）：`dpo_trainer.run_dpo_training`

- **思路**：不 rollout，直接读已准备好的偏好对 jsonl，用 TRL 的 `DPOTrainer` 训练。
- **流程**：`load_rl_dataset(..., format="standard")` 得到 (user→prompt, chosen, rejected)，`DPOTrainer(model, args, processing_class=tokenizer, train_dataset=dataset)`，`trainer.train()`；步数由 epoch 和 `max_steps` 决定。

---

## 四、正式跑各算法的方法

### 1. 多轮 PPO（三种 reward 模式）

- **命令**：
  ```bash
  python scripts/rl/run_rl.py --config configs/rl_compare_rewards.yaml
  ```
  若要用不同 reward 模式，改配置里 `reward.reward_mode` 为 `mode1` / `mode2` / `mode3`，或分别用三个 yaml（如 `output_dir` 分别为 `outputs/ppo_emo_mode1` 等）。
- **配置要点**：
  - `rl.algo: "ppo"`（run_rl 里会调 `run_ppo_emo_training`）；
  - `data.data_dir` 或 `data.train_file` 指向 profile 数据；
  - `data.batch_size`、`rollout.max_turns`、`rollout.max_new_tokens_per_turn`、`rollout.user_llm`（mock/deepseek/qwen）；
  - `reward.reward_mode`、`reward.w1/w2/w3` 等；
  - `training.total_steps`、`save_steps`、`output_dir`。
- **数据量**：Profile 条数由数据文件决定；每步用 `batch_size` 个 profile，数据循环使用，总步数 = `total_steps`。

### 2. GRPO

- **命令**：
  ```bash
  python scripts/rl/run_rl.py --config configs/rl_grpo.yaml
  ```
- **配置要点**：
  - `rl.algo: "grpo"`；
  - `data.train_file`（Profile 路径）；
  - `rl.grpo.num_generations`（G）、`learning_rate`、`epsilon`、`beta`（KL 系数）；
  - `training.total_steps`、`save_steps`、`output_dir`。
- **数据量**：每步 1 个 prompt，总步数 = `total_steps`。

### 3. On-policy DPO（dpo_emo）

- **命令**：
  ```bash
  python scripts/rl/run_rl.py --config configs/rl_dpo_emo.yaml
  ```
  或专用脚本：
  ```bash
  python scripts/rl/run_dpo_emo.py --config configs/rl_dpo_emo.yaml
  ```
- **配置要点**：
  - `rl.algo: "dpo_emo"`（run_rl 时）；
  - `rollout.num_profiles`、`rollout.batch_size`、`rollout.num_samples`、`rollout.max_turns`；
  - `rollout.use_planning_score` / `use_planning_emo` / `use_mock_simulator`；
  - `training.output_dir`、`max_steps` 等。
- **数据量**：前 `num_profiles` 个 profile 参与 rollout；偏好对数量 = 多轮 × 每轮选出的 best/worst 对数，再按 batch 训练。

### 4. Offline DPO（偏好对 jsonl）

- **当前 run_rl.py 不包含**，需要单独调用。例如写一个脚本：
  ```python
  import yaml
  from src.training.dpo_trainer import run_dpo_training
  with open("configs/rl_dpo.yaml") as f:
      cfg = yaml.safe_load(f)
  run_dpo_training(cfg)
  ```
  或参考 `static-rl/run_dpo.py` 的逻辑，用 `configs/rl_dpo.yaml`（`data.train_file` 指向 `data/ipm_prefdial_dpo.jsonl` 等）。
- **数据量**：由 jsonl 行数和 `num_train_epochs` / `max_steps` 决定。

---

## 五、小实验（快速验证）：`run_quick_verify.py`

- **目的**：用较少步数快速跑通 **PPO（三种 reward 模式）** 和 **GRPO**，检查流程、画 KL/reward 曲线、做简单评估。
- **命令**：
  ```bash
  python scripts/rl/run_quick_verify.py
  ```
  可选：
  - `--ppo-steps 50`、`--grpo-steps 50`：各算法步数；
  - `--output-base outputs/quick_verify`：输出根目录；
  - `--skip-ppo` / `--skip-grpo`：只跑其中一种。
- **数据**：与正式跑相同，仍用 `configs/rl_compare_rewards.yaml` 和 `configs/rl_grpo.yaml` 里的 `data.train_file` / `data_dir`（Profile）；**只是把 `total_steps` 改小**（如 50）。
- **步数设计**：
  - 小实验里 **PPO** 的 `total_steps` 由 `--ppo-steps` 覆盖（默认 50），即 50 个 batch 的 rollout + 50 次 PPO 更新；
  - **GRPO** 的 `total_steps` 由 `--grpo-steps` 覆盖（默认 50），即 50 个 prompt × G 个生成 + 50 次更新。
  这样可以在几分钟到十几分钟内看到曲线、确认无报错。
- **流程**：
  1. 对 mode1 / mode2 / mode3 各跑一遍 PPO，输出到 `{output-base}/ppo_mode1`、`ppo_mode2`、`ppo_mode3`；
  2. 跑一遍 GRPO，输出到 `{output-base}/grpo`；
  3. 每个实验都会写 `training_log.jsonl`（及 PPO 的 `final`、GRPO 的 `final`）。
- **后续**：
  - 画图：`python scripts/eval/plot_rl_curves.py --log-dir outputs/quick_verify`
  - 简单推理：`python scripts/eval/eval_rl_models.py --model-dir outputs/quick_verify`
  - 全维度评估：`python scripts/eval/eval_all_models.py --quick-verify-dir outputs/quick_verify`

---

## 六、和 static-rl 的对比（简要）

- **static-rl**：单轮、静态 prompt（如 EmpatheticDialogues），用 **训练好的 Reward Model** 打分，PPO/GRPO/DPO 都围绕「RM 分数」；数据管线是 build 偏好 → 训 RM → 再 PPO/GRPO。
- **非 static-rl（本路径）**：  
  - **PPO** 是多轮 + **Profile + 用户模拟器** + **reward_emo**（mode1/2/3）；  
  - **GRPO** 用同一套 Profile 数据，reward 可用 reward_emo 或简单规则；  
  - **DPO** 分为 on-policy（dpo_emo，Profile + 多轮生成 + 打分造偏好对）和 offline（静态偏好对 jsonl）。

数据上，非 static-rl 以 **Profile 数据**（`data/data/train_profile.jsonl` 或目录）为主；offline DPO 单独用 **偏好对 jsonl**。步数由各配置里的 `total_steps` / `max_steps` / epoch 控制，小实验通过 `run_quick_verify.py` 的 `--ppo-steps` / `--grpo-steps` 把步数压小做快速验证。
