# ActorRefRollout 一步步实现指南

本文档面向想**自己动手写 PPO 里 Actor/Ref/Rollout** 的同学，按函数顺序讲清「要做什么」和「怎么写」，只给思路和关键点，不直接贴完整答案，方便你边学边实现。

---

## 0. 这个类在 PPO 里干什么

- **Actor**：当前要训练的 policy（因果语言模型），负责「根据 query 生成 response」和「算自己生成的那些 token 的 log 概率」。
- **Ref（Reference）**：一个**冻结**的拷贝，只用来算「在 ref 眼里这些 token 的 log 概率」，用于 PPO 的 **KL 惩罚**，防止新 policy 离旧 policy 太远。
- **Rollout**：用 actor 做**自回归生成**（一段段采样 response），并把生成时的 log_probs 等存进 buffer，供后面 PPO 更新用。

所以这个类 = 一个可训练的 LM（actor）+ 一个冻结的 LM（ref）+ 和它们相关的「生成、算 log_prob」的逻辑。

---

## 1. `__init__`：初始化 actor、ref、tokenizer

**要做什么**

- 把传入的 `causal_lm` 当作 **actor**（可训练）。
- 深拷贝一份当作 **ref**，并让 ref 的**所有参数** `requires_grad=False`。
- 保存 `tokenizer`、`ref_sync_interval`（每 N 步是否把 ref 同步成 actor）、`device`。
- 设一个内部计数器 `_step = 0`，后面 `maybe_sync_ref` 用。

**思路**

- `copy.deepcopy(causal_lm)` 得到 ref，注意 ref 要和 actor 结构一致、但参数独立。
- 遍历 `self.ref.parameters()`，逐个 `p.requires_grad_(False)`。
- `device` 若为 None，可用 `next(causal_lm.parameters()).device`。

**实现顺序**：先写 `__init__`，再写下面用到的 `sync_ref_from_actor`、`maybe_sync_ref`，这样类才能正常建起来。

---

## 2. `sync_ref_from_actor`：把 ref 同步成当前 actor

**要做什么**

- 用当前 **actor** 的参数字典覆盖 **ref** 的参数。
- 覆盖后再次保证 ref 所有参数 `requires_grad=False`。

**思路**

- `state_dict = copy.deepcopy(self.actor.state_dict())`，再 `self.ref.load_state_dict(state_dict)`。
- 再遍历一次 `self.ref.parameters()` 设 `requires_grad_(False)`。

**用途**：训练一段时间后，若希望「参考 policy」更新一下（例如防止 KL 爆炸），就调一次；也可以由 `maybe_sync_ref` 按步数自动调。

---

## 3. `maybe_sync_ref`：按步数决定是否 sync

**要做什么**

- 若 `ref_sync_interval` 为 None，直接 return。
- 否则：内部步数 `_step += 1`，若 `_step % ref_sync_interval == 0`，就调用 `sync_ref_from_actor()`。

**思路**：先判断 `self.ref_sync_interval is None`，再改 `_step`，再判断余数。训练循环里每步调用一次即可。

---

## 4. `_log_probs_from_model`：核心——从任意 LM 算 response 位置的 log 概率

**要做什么**

- 输入：一个 `model`（actor 或 ref）、`input_ids`（整段 query+response）、`attention_mask`、`response_mask`（只有 response 位置为 1）。
- 对 causal LM：`logits[:, t, :]` 预测的是**下一个 token**，即 `input_ids[:, t+1]`。
- 输出：形状 `(batch, full_len)` 的 log 概率，**只在 response 位置有值，其余为 0**。

**关键概念**

- Causal LM 前向：`out = model(input_ids=input_ids, attention_mask=attention_mask)`，`out.logits` 形状为 `(batch, seq_len, vocab_size)`。
- 位置 `t` 的 logits 预测的是位置 `t+1` 的 token，所以「位置 t+1 的 token 的 log 概率」要从 `logits[:, t, :]` 上取。
- 做法：对 `logits[:, :-1, :]` 做 `log_softmax`，得到每个位置预测下一个 token 的 log 概率分布；再用 `gather` 在最后一维上取出「实际出现的 token」对应的 log 概率，得到 `(batch, seq_len-1)`。
- 再把这列对齐到「位置 1 到 seq_len-1」的 log prob，位置 0 没有预测，填 0。
- 最后乘上 `response_mask`，把非 response 位置清零。

**思路**

1. `out = model(input_ids=..., attention_mask=...)`，取 `logits`。
2. `log_probs_next = log_softmax(logits[:, :-1, :], dim=-1)`。
3. `target = input_ids[:, 1:]`，用 `gather` 在 vocab 维上取 `target` 对应的 log 概率，得到 `(batch, seq_len-1)`。
4. 拼成 `(batch, seq_len)`：第 0 位为 0，第 1 到 end 为上面那列。
5. `return 结果 * response_mask`。

**实现顺序**：建议先单独写这个函数，用一个小 batch、已知的 input_ids 和 mask 打印 shape 和几个值，确认和「下一 token 预测」一致，再写 `get_actor_log_probs` / `get_ref_log_probs`。

---

## 5. `get_actor_log_probs` 和 `get_ref_log_probs`

**要做什么**

- `get_actor_log_probs`：用**当前 actor** 对「整段 query+response」算 log_probs，只保留 response 段（用 response_mask）。
- `get_ref_log_probs`：用 **ref** 对同一段序列做同样的事。

**思路**

- 直接调用 `_log_probs_from_model`，第一个参数分别传 `self.actor` 和 `self.ref`，其余参数相同。
- 这样避免重复写「算 log prob」的逻辑，只换模型。

---

## 6. `generate`：Rollout——生成 response 并拿到 log_probs

**要做什么**

- 用 **actor** 对 `query_ids` 做自回归生成，得到整段 `query + response` 的 token ids。
- 从生成结果里截出 **response 段**（去掉 query 长度），得到 `response_ids`、`response_mask`（非 pad 为 1）。
- 再对「整段 query+response」前向一次，用 `_log_probs_from_model` 取出 **response 段**的 log_probs。
- 返回：`(response_ids, response_mask, log_probs_response)`。

**思路**

1. 取 `pad_token_id`（tokenizer 的 `pad_token_id` 或 `eos_token_id`）。
2. `gen_out = self.actor.generate(query_ids, attention_mask=query_mask, max_new_tokens=..., do_sample=..., top_p=..., temperature=..., pad_token_id=pad_id)`，形状为 `(batch, query_len + new_tokens)`。
3. `qlen = query_ids.size(1)`，`response_ids = gen_out[:, qlen:]`，`response_mask = (response_ids != pad_id)` 转成 float。
4. 构造「整段」的 `full_ids = gen_out`，`full_attn`（query 部分用 query_mask，后面用 1），`full_resp_mask`（前 qlen 为 0，后面与 response_mask 一致）。
5. `full_log_probs = self._log_probs_from_model(self.actor, full_ids, full_attn, full_resp_mask)`，再 `log_probs_response = full_log_probs[:, qlen:]`。
6. 返回 `(response_ids, response_mask, log_probs_response)`。

**注意**：生成时用到的采样参数（do_sample、top_p、temperature）要和你的任务匹配；推理时可 `do_sample=False`。

---

## 7. `parameters_for_optimizer`

**要做什么**

- 返回「需要被优化器更新」的参数列表，这里**只有 actor** 的参数，ref 不训练。

**思路**：`return list(self.actor.parameters())`。

---

## 8. `reward_collection`

**要做什么**

- 输入：`reward_model`（有 `compute_reward(texts)` 的接口）、当前 batch 的 `response_ids`、`response_mask`。
- 对每条样本：按 `response_mask` 截掉 padding，把 response 的 token ids 解码成**文本**，得到 `texts`。
- 调用 `reward_model.compute_reward(texts)` 得到 list of float，再转成 `(batch,)` 的 tensor，device 与 `query_ids` 一致。

**思路**

- 遍历 batch，对第 i 条：`r_len = int(response_mask[i].sum().item())`，取 `response_ids[i, :r_len]`（若 r_len>0），用 `tokenizer.decode(..., skip_special_tokens=True)` 得到字符串，加入 list。
- `rewards = reward_model.compute_reward(texts)`，`return torch.tensor(rewards, dtype=torch.float32, device=query_ids.device)`。

---

## 建议实现顺序

1. **`__init__`** → **`sync_ref_from_actor`** → **`maybe_sync_ref`**：先把「两个模型 + 同步」搭好。
2. **`_log_probs_from_model`**：单独测通「给定 model + input_ids + mask，得到 response 段 log_probs」。
3. **`get_actor_log_probs`** / **`get_ref_log_probs`**：薄封装。
4. **`generate`**：先保证能生成、形状对，再接上 log_probs。
5. **`parameters_for_optimizer`**、**`reward_collection`**：按需实现。

每实现一个函数，建议用一个小 batch（例如 2 条、短 seq）打印 shape 和少量数值，确认和预期一致再往下写。这样你可以把 **ActorRefRollout** 当作「PPO 训练函数」的第一块积木，一步步搭完。

---

*对应代码骨架：`src/training/ppo_training.py` 中的 `ActorRefRollout` 类*
