# PPO 微调 LLM 文档

本文档说明 `src/training/ppo_training.py` 中 PPO 相关类的用途，重点为 **ActorRefRollout** 各函数说明。

---

## ActorRefRollout 类

该类将 **reference**、**rollout**、**actor** 合并为一个类：

- **actor**：可训练的 policy（因果 LM），用于生成与 PPO 更新
- **ref**：冻结的 reference，仅用于计算 ref_log_probs（KL 惩罚）
- **rollout**：用 actor 做自回归生成（collect 阶段）

约定：传入的 causal_lm 需支持 `.generate(...)` 以及前向返回 `logits` / `hidden_states`。

---

### 各函数说明

| 函数 | 作用 |
|------|------|
| **`__init__(causal_lm, tokenizer, ref_sync_interval, device)`** | 初始化：把传入的 causal LM 作为 **actor**（可训练），深拷贝一份作为 **ref**（冻结），保存 tokenizer。`ref_sync_interval` 不为 None 时，每隔 N 步可将 ref 同步为当前 actor。 |
| **`sync_ref_from_actor()`** | 将 **ref** 的参数整体替换为当前 **actor** 的拷贝，并保持 ref 不可训练。用于周期性更新 reference，避免 policy 偏离过远。 |
| **`maybe_sync_ref()`** | 内部步数 +1；若设置了 `ref_sync_interval` 且步数是其倍数，则调用 `sync_ref_from_actor()`。训练循环中每步调用一次即可。 |
| **`_log_probs_from_model(model, input_ids, attention_mask, response_mask)`** | **内部用**：对任意 causal LM 做一次前向，按「logits 预测下一 token」计算每个位置的 log 概率，再按 `response_mask` 只保留 **response 段**，其余位置置 0。返回 `(batch, full_len)`。 |
| **`get_actor_log_probs(input_ids, attention_mask, response_mask)`** | 用**当前 actor** 对整段序列（query+response）前向，只计算 **response 段** 的 token log 概率。PPO 更新时用于计算「新 policy 的 log π(a)」。 |
| **`get_ref_log_probs(input_ids, attention_mask, response_mask)`** | 用**冻结的 ref** 对同一段序列前向，只计算 **response 段** 的 log 概率。用于 PPO 中的 KL(π∥π_ref) 项。 |
| **`generate(query_ids, query_mask, max_new_tokens, do_sample, top_p, temperature)`** | **Rollout**：用 **actor** 自回归生成 response（`actor.generate`），得到 `response_ids`、`response_mask`；再对「query+response」整段前向一次，用 `_log_probs_from_model` 取出 **response 段** 的 log_probs。返回 `(response_ids, response_mask, log_probs_response)`，供写入 buffer。 |
| **`parameters_for_optimizer()`** | 返回**需要训练的参数**（仅 actor 的 parameters），供优化器使用；ref 不参与训练。 |
| **`reward_collection(reward_model, query_ids, response_ids, response_mask)`** | 用 **reward_model** 对当前 batch 的 **response** 打分：按 `response_mask` 截掉 padding，decode 成文本，调用 `reward_model.compute_reward(texts)`，返回 `(batch,)` 的 reward tensor。 |

---

### 数据流对应关系

- **收集阶段**：`generate` 得到 response 与 log_probs → 外部用 critic 算 values、用 `get_ref_log_probs` 或 rollout 时存的 ref_log_probs → `reward_collection` 算 rewards → 写入 **PPOMemory**。
- **更新阶段**：从 memory 取出的 batch 中，用 **`get_actor_log_probs`** 得到「当前 policy 的 log_probs」用于 PPO 的 ratio；**ref** 仅通过 **`get_ref_log_probs`** 参与 KL 项，不更新。

---

## 其他相关类（简要）

| 类 / 模块 | 作用 |
|-----------|------|
| **PPOMemory** | Rollout buffer：存 query、response、log_probs、values、rewards、ref_log_probs，提供 `store`/`store_batch`、`get`/`sample`、`compute_gae`。 |
| **Critic** | Value head：输入 `(input_ids, attention_mask)`，经 base_model 取 hidden_states 再经 head 输出每位置 value `(batch, seq_len)`。 |
| **RewardModel / RuleRewardModel** | 奖励接口：`compute_reward(texts)` 或可学习的 `forward(input_ids, attention_mask)`。 |
| **PPOTrainer** | 训练循环：`collect_rollouts` 攒 buffer → `train_step` 内多次 `ppo_step` 更新 actor+critic → `memory.clear()`。 |

---

## PPOTrainer 与其他类的功能重合关系

下面说明 **PPOTrainer** 里哪些函数和 **ActorRefRollout / PPOMemory** 重合、谁该调谁，避免重复实现。

| PPOTrainer 函数 | 与谁重合 | 说明与建议 |
|----------------|----------|------------|
| **`generate`** | ActorRefRollout.generate | **不重复**。ActorRefRollout.generate 只返回 `(response_ids, response_mask, log_probs)`；PPOTrainer.generate 应在此基础上**再**用 Critic 对「整段 query+response」算 values，返回 4 个。即：PPOTrainer.generate = 调 `actor_ref.generate` + 拼 full 序列 + `critic(full_ids, full_attn)` 取 response 段 values。 |
| **`get_ref_log_probs`** | ActorRefRollout.get_ref_log_probs | **逻辑在 ActorRefRollout**。PPOTrainer 的入参是拆开的 `(query_ids, query_mask, response_ids, response_mask)`，应：拼成 `full_ids`、`full_attn`、`full_resp_mask`，再调 `actor_ref.get_ref_log_probs(full_ids, full_attn, full_resp_mask)`，返回的可以是整段或只取 response 段（与调用方约定一致即可）。**不要**在 PPOTrainer 里再写一遍算 ref log_probs。 |
| **`collect_rollouts`** | ActorRefRollout.generate / reward_collection | **应直接复用**。流程：dataloader 取 batch → 调 **PPOTrainer.generate**（内部用 actor_ref.generate + critic）得到 response_ids, response_mask, log_probs, values → 拼 full 序列后调 **actor_ref.get_ref_log_probs** 得 ref_log_probs → 调 **actor_ref.reward_collection(self.reward_model, ...)** 得 rewards → **memory.store_batch(...)**。算 reward 不要自己在 PPOTrainer 里 decode+reward_model，用 `actor_ref.reward_collection` 即可。 |
| **`train_step`** | PPOMemory.get / compute_gae | **不重复**。PPOTrainer 只负责「调 memory.get(compute_gae=True)」再按 mini_batch 调 ppo_step；GAE 的实现只在 **PPOMemory.compute_gae** 里。 |
| **advantage/return** | 仅在 PPOMemory.compute_gae 中实现 | 训练时用 **PPOMemory.get(compute_gae=True)** 即可；PPOTrainer 不要再实现一套 GAE。 |

### 小结

- **生成 + log_probs**：只在 ActorRefRollout.generate 里实现；PPOTrainer.generate = 该调用 + critic 算 values。
- **ref_log_probs**：只在 ActorRefRollout.get_ref_log_probs 里实现；PPOTrainer.get_ref_log_probs = 拼序列 + 调该函数。
- **reward**：只在 ActorRefRollout.reward_collection 里实现；collect_rollouts 里用 reward_model 时调它，不要在 PPOTrainer 里再写 decode+compute_reward。
- **advantage/return**：只在 PPOMemory.compute_gae 里实现（训练用）；ActorRefRollout 的简单版可选保留或删除。

这样 PPOTrainer 只做「拼数据、调子组件、跑循环」，不重复实现 log_probs / ref_log_probs / reward / GAE 的细节。

---

*文档对应代码：`src/training/ppo_training.py`*
