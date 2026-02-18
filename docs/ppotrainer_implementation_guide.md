# PPOTrainer 一步步实现教程（入门版）

这份教程用**最简单的说法**带你一步步把 PPOTrainer 里的每个函数写出来。不用怕，我们像搭积木一样，一块一块来。

---

## 先搞懂：PPOTrainer 到底在干嘛？

可以把它想成**一个负责“考试 + 改卷”的老师**：

1. **收集答题卡**：让同学（模型）根据题目（query）写答案（response），并把答案、得分（reward）、以及“当时写的概率”（log_probs）等记在一张表里（memory）。
2. **改卷**：从表里拿出一批答题卡，根据“新算一遍的概率”和“当时的得分”，用 PPO 公式算 loss，更新模型（actor）和打分器（critic）。
3. **反复做**：重复“收集 → 改卷 → 清空这一批”，做很多轮（total_steps）。

所以 PPOTrainer 的活就是：**组织“谁去生成、谁去打分、谁去存、谁去更新”**，自己只写流程，具体算 log_prob、算 value 都交给 ActorRefRollout 和 Critic。

---

## 实现顺序建议

建议按这个顺序写，前面的会用到后面的：

1. **`__init__`**：先把“老师”手里要用的东西都接过来、存好（已经写好框架，只需确认都存了）。
2. **`generate`**：让模型生成答案，并拿到 log_probs 和 values。
3. **`get_ref_log_probs`**：用“参考模型”算 ref 的 log 概率（给 KL 用）。
4. **`collect_rollouts`**：用 dataloader 取题目 → 调 generate → 算 reward、ref_log_probs → 存进 memory。
5. **`_build_full_sequence_batch`**：把 memory 里的一小批“长短不一的序列”拼成“整齐的一摞”（方便一起送进模型）。
6. **`ppo_step`**：用这一小批数据算 PPO 的 loss，反传，更新参数。
7. **`train_step`**：从 memory 取出全部数据、算好 GAE，再按 mini_batch 多次调用 ppo_step。
8. **`run`**：大循环：收集 → 训练一步 → 清空 memory，重复 total_steps 次。

下面按这个顺序，每个函数用“要干啥 → 为啥 → 咋做”来说。

---

## 1. `__init__`：把要用到的东西都接过来

**要干啥**  
把“演员（actor_ref）”“打分器（critic）”“奖励规则（reward_model）”“小本本（memory）”“优化器（optimizer）”以及各种超参数（ppo_epochs、mini_batch_size、clip_range、kl_coef、max_new_tokens、device）都存到 `self.xxx` 里，后面每个函数直接用。

**为啥**  
这样 PPOTrainer 不用每次问别人要这些对象，自己就能调 actor_ref、critic、memory 等。

**咋做**  
代码框架里已经把这些参数接进来并赋给 `self` 了。你只需要确认：  
- `reward_model` 如果是普通函数（只接受 list[str] 返回 list[float]），会被包成 `RuleRewardModel`；  
- `device` 如果没传，就用 `actor_ref_rollout.actor` 的参数所在的 device。  

一般不用改，只要知道“这些 self.xxx 后面都会用到”就行。

---

## 2. `generate`：让模型写答案，并拿到 log_probs 和 values

**要干啥**  
给定一批题目 `query_ids`、`query_mask`，做两件事：  
1）用 **actor_ref.generate** 生成 response，得到 `response_ids`、`response_mask`、`log_probs`；  
2）用 **critic** 对“整段 query+response”算每个位置的 value，再只取出 **response 那段**的 values。  
最后返回 4 样东西：`response_ids`、`response_mask`、`log_probs`、`values`（都是 response 段的，方便直接存进 memory）。

**为啥**  
PPO 要存“生成时”的 log_probs 和 values，后面算 advantage、ratio 都要用。生成在 ActorRefRollout 里做，算 value 在 Critic 里做，Trainer 只负责“调它俩、拼好数据、只取 response 段”。

**咋做**（分步）

1. 调 `self.actor_ref.generate(query_ids, query_mask, max_new_tokens=self.max_new_tokens, ...)`，得到 `response_ids`、`response_mask`、`log_probs`（3 个）。
2. 拼出“整段”：`full_ids = query + response`（把 query_ids 和 response_ids 在序列长度那一维拼起来），同样拼出 `full_attn`（attention_mask）。
3. 用 critic 算整段的 value：`values_full = self.critic(full_ids, full_attn)`，形状是 (batch, 整段长度)。
4. 从 `values_full` 里把 **response 那一段**切出来（和 response 长度一致），得到 `values`。
5. `return response_ids, response_mask, log_probs, values`（4 个）。

注意：拼 full 时 query 和 response 的长度要对齐；response 段从 `query_len` 开始到结尾。

---

## 3. `get_ref_log_probs`：用参考模型算“参考概率”

**要干啥**  
PPO 里要算“当前策略和参考策略差多少”（KL），所以要有一个“参考模型对这段 response 的 log 概率”。  
这里 Trainer 的接口是：给你**已经拆好的** `query_ids`、`query_mask`、`response_ids`、`response_mask`，你拼成“整段”，然后调 **actor_ref.get_ref_log_probs**，返回 ref 对 response 段的 log 概率（可以是整段里只取 response 段，或按你约定）。

**为啥**  
算 ref log prob 的实现在 ActorRefRollout 里，Trainer 只做“拼 query+response → 调 actor_ref → 返回”，不重复写一遍前向。

**咋做**（分步）

1. 用 `query_ids` 和 `response_ids` 在“序列长度”那一维拼成 `full_ids`（注意 response 要截掉 padding，只拼有效长度，或先拼再和 mask 对齐）。
2. 同样拼出 `full_attn`、`full_resp_mask`（只有 response 那一段为 1，前面 query 为 0）。
3. `ref_log_probs_full = self.actor_ref.get_ref_log_probs(full_ids, full_attn, full_resp_mask)`。
4. 从结果里切出 response 段（从 query_len 到结尾），返回；或直接返回整段，让调用方自己取 response 段（和 collect_rollouts 的约定一致即可）。

---

## 4. `collect_rollouts`：收集一批“答题卡”进 memory

**要干啥**  
从 dataloader 里取若干 batch 的“题目”（prompt / query），对每一批：  
- 用 **generate** 得到 response、log_probs、values；  
- 用 **get_ref_log_probs** 得到 ref_log_probs；  
- 用 **actor_ref.reward_collection** 和 **self.reward_model** 得到 rewards；  
- 用 **memory.store_batch** 把这一批存进去。  
“若干 batch”可以定为 1 个 batch，或凑够某个数量再停，由你定（例如固定 1 次 collect 就 1 个 batch）。

**为啥**  
训练前必须先有一批“题目→答案→得分→概率”的数据，这些都在 memory 里，后面 train_step 才有的可更新。

**咋做**（分步）

1. 用 `for batch in dataloader` 取数据（或只取一个 batch，看你怎么设计）。
2. 从 batch 里拿出 `query_ids`、`query_mask`（没有的话要自己用 tokenizer 把 prompt 转成 id 和 mask）。
3. 调 `self.generate(query_ids, query_mask)`，得到 `response_ids, response_mask, log_probs, values`。
4. 拼 full_ids / full_attn / full_resp_mask（和 get_ref_log_probs 一样），调 `self.get_ref_log_probs(...)` 得到 ref_log_probs（若是整段就切 response 段）。
5. 调 `self.actor_ref.reward_collection(self.reward_model, query_ids, response_ids, response_mask)` 得到 `rewards`（一维 tensor）。
6. 调 `self.memory.store_batch(query_ids, query_mask, response_ids, response_mask, log_probs, values, rewards, ref_log_probs)`。
7. 若想多收集几个 batch 再停，就重复 2～6；否则一次就 break。

---

## 5. `_build_full_sequence_batch`：把长短不一的序列“对齐成一摞”

**要干啥**  
memory 里存的是 **list**：每条样本的 query、response 长度可能不一样。但送进模型时要一个 **tensor**，形状是 (batch, 最大长度)，所以要把每条“query+response”拼成一条，再 pad 到同一长度，并记下每条“多长是 query、多长是 response”，方便后面只取 response 段。

**为啥**  
ppo_step 里要一次前向算 new_log_probs、new_values，模型只认 (batch, seq_len) 的 input_ids，所以必须先“对齐”。

**咋做**（分步）

1. 从 `batch` 里取出 `queries`、`query_masks`、`responses`、`response_masks`（都是 list，每个元素是 1 维 tensor）。
2. 对每条样本 i：把 `query_i` 和 `response_i`（只取有效长度，用 response_mask 或 rlen）在 dim=0 上拼成 `full_i`，记下这条的 response 长度 `resp_lens[i]`。
3. 找到这批里最大的 `max_full = max(len(full_i))`，以及最大的 response 长度 `max_r`。
4. 建 3 个 tensor：`full_ids`、`full_attn`、`full_resp_mask`，形状都是 (batch, max_full)，先填 pad_token_id 或 0。
5. 对每条 i：把 `full_i` 拷到 `full_ids[i, :len(full_i)]`，`full_attn[i, :len(full_i)] = 1`；response 段对应的位置在 `full_resp_mask` 里标 1（query 段为 0）。
6. 再记下每条 query 的长度 `qlens[i]`（方便后面切 response 段：从 qlens[i] 到 qlens[i]+resp_lens[i]）。
7. 返回 `full_ids, full_attn, full_resp_mask, qlens, resp_lens, max_r`。

---

## 6. `ppo_step`：用一小批数据做一次 PPO 更新

**要干啥**  
从 memory 里已经取出的一个 **mini_batch**（字典，里面是 list 的 tensor），做三件事：  
1）用 **actor** 和 **critic** 重新算这批的 new_log_probs、new_values（response 段）；  
2）用 PPO 公式算 policy loss（clip）、value loss、KL loss，加在一起 total_loss；  
3）`optimizer.zero_grad()` → `total_loss.backward()` → 梯度裁剪 → `optimizer.step()`。  
最后返回一个字典，例如 `{"policy_loss": ..., "value_loss": ..., "kl_loss": ..., "total_loss": ...}`（标量）。

**为啥**  
PPO 的“改卷”就是：用当前模型再算一遍概率和 value，和“当时存的”old_log_probs、advantages、returns 一起算 loss，再反传更新。

**咋做**（分步）

1. **拼 batch**：调 `_build_full_sequence_batch(batch, pad_token_id)`，得到 `full_ids, full_attn, full_resp_mask, qlens, resp_lens, max_r`。
2. **算 new_log_probs、new_values**：  
   - `new_log_probs_full = self.actor_ref.get_actor_log_probs(full_ids, full_attn, full_resp_mask)`  
   - `new_values_full = self.critic(full_ids, full_attn)`  
   然后对每条 i，从 `qlens[i]` 到 `qlens[i]+resp_lens[i]` 切出 response 段，得到一列 list；再用 `_pad_list_to_tensor` 把它们 pad 成 (batch, max_r)，得到 `new_log_probs`、`new_values` 和 `mask_r`。
3. **把 batch 里的 old 数据也 pad 成 (batch, max_r)**：  
   `old_log_probs`、`ref_log_probs`、`advantages`、`returns` 都用 `_pad_list_to_tensor(..., max_r)` 得到 tensor 和 mask（可共用一个 mask_r，或再取一份）。
4. **Policy loss（PPO clip）**：  
   - `ratio = exp(new_log_probs - old_log_probs)`  
   - `ratio_clipped = clamp(ratio, 1-clip_range, 1+clip_range)`  
   - `policy_loss = -mean(min(ratio*adv, ratio_clipped*adv) * mask_r)`（只对有效位置求平均）
5. **Value loss**：`value_loss = mean((new_values - returns)^2 * mask_r)`。
6. **KL loss**：`kl_loss = mean((new_log_probs - ref_log_probs) * mask_r)`（当前策略相对 ref 的偏离）。
7. **总 loss**：`total_loss = policy_loss + 0.5*value_loss + kl_coef*kl_loss`。
8. **更新**：`optimizer.zero_grad()` → `total_loss.backward()` → `clip_grad_norm_(actor 和 critic 的参数, 1.0)` → `optimizer.step()`。
9. `return {"policy_loss": ..., "value_loss": ..., "kl_loss": ..., "total_loss": ...}`（用 .item() 转成 Python 标量）。

---

## 7. `train_step`：用整批 buffer 做“一整轮”更新

**要干啥**  
当前 memory 里已经有一批数据了。这一步要做的是：  
1）用 **memory.get(compute_gae=True)** 取出全部数据，并且已经算好 advantages、returns；  
2）在这批数据上做 **ppo_epochs** 个 epoch，每个 epoch 里按 **mini_batch_size** 切成很多小块，每块调一次 **ppo_step**；  
3）把每次 ppo_step 返回的 stats 求平均，最后返回一个汇总的 stats 字典。

**为啥**  
PPO 习惯“大 buffer、小 mini_batch、多 epoch”，这样既用得上整批的 GAE，又不会一次更新太大导致不稳定。

**咋做**（分步）

1. `data = self.memory.get(compute_gae=True)`，得到带 advantages、returns 的字典。
2. `n = len(self.memory)`（这批有多少条）。
3. 设 `all_stats = []`。
4. 循环 `for _ in range(self.ppo_epochs)`：  
   再一层循环 `for start in range(0, n, self.mini_batch_size)`：  
   - 算出这一段的索引 `indices = [start, start+1, ..., min(start+mini_batch_size, n)-1]`；  
   - 从 `data` 里按 indices 取出一个 mini_batch（每个 key 若是 list，就 `[v[i] for i in indices]`）；  
   - `stats = self.ppo_step(mini_batch)`，把 `stats` 放进 `all_stats`。
5. 对 `all_stats` 里每个 key 求平均（例如 `policy_loss` 就对所有 stats 的 policy_loss 取平均），返回一个字典（可用之前的 `_aggregate_stats(all_stats)` 若已有的话）。

---

## 8. `run`：大循环——“收集 → 训练 → 清空”

**要干啥**  
重复 **total_steps** 次：  
1）**collect_rollouts(dataloader)**：往 memory 里攒一批数据；  
2）**train_step()**：用这批数据做多 epoch、多 mini_batch 的 PPO 更新；  
3）**memory.clear()**：清空 buffer，准备下一轮收集。  
还可以在每步前后调 **actor_ref.maybe_sync_ref()**（若你设置了 ref_sync_interval）。

**为啥**  
这就是“老师”的完整工作流：收一批卷子 → 改完 → 收走，再收下一批。

**咋做**（分步）

1. `for step in range(total_steps):`
2. 调 `self.collect_rollouts(dataloader)`。
3. （可选）`self.actor_ref.maybe_sync_ref()`。
4. 调 `self.train_step()`，拿到 stats，可以打印或打 log。
5. 调 `self.memory.clear()`。
6. 继续下一步。

---

## 小结：谁调谁（避免重复）

- **generate**：调 `actor_ref.generate`，再自己拼 full 序列、调 `critic`，只取 response 段返回。
- **get_ref_log_probs**：拼 full → 调 `actor_ref.get_ref_log_probs`，返回 response 段（或整段）。
- **collect_rollouts**：调 `generate`、`get_ref_log_probs`、`actor_ref.reward_collection`、`memory.store_batch`，不要在 Trainer 里再写一遍 decode+reward。
- **train_step**：只调 `memory.get(compute_gae=True)` 和多次 `ppo_step`，不自己实现 GAE。

按上面顺序，先写 generate 和 get_ref_log_probs，再写 collect_rollouts，再 _build_full_sequence_batch 和 ppo_step，最后 train_step 和 run，就不会乱。每写一个函数，可以用很少的 batch、很短的序列打一下 shape 和几个值，确认对了再写下一个。加油～

---

*对应代码：`src/training/ppo_training.py` 里的 `PPOTrainer` 类（目前为框架，待你按本教程补全）。*
