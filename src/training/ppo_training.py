"""
PPO 微调 LLM 用的组件：
- PPOMemory：rollout buffer + GAE
- Critic：value head 框架
- RewardModel：奖励模型 / 规则奖励框架
- ActorRefRollout：reference + rollout + actor 合并为一个类
- PPOTrainer：训练循环骨架
"""
import copy
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable


class PPOMemory:
    """
    LLM PPO 的 rollout buffer。
    按「序列」存：每条是 (query, response, log_probs, values, reward, ref_log_probs)。
    """

    def __init__(
        self,
        gamma: float = 1.0,
        lam: float = 0.95,
        device: Optional[T.device] = None,
    ):
        self.gamma = gamma
        self.lam = lam
        self.device = device or T.device("cpu")

        self.queries: List[T.Tensor] = []           # 每个 [query_len]
        self.query_masks: List[T.Tensor] = []       # 每个 [query_len]
        self.responses: List[T.Tensor] = []         # 每个 [response_len]，仅生成的 token ids
        self.response_masks: List[T.Tensor] = []   # 每个 [response_len]
        self.log_probs: List[T.Tensor] = []         # 每个 [response_len]，当前 policy 的 log prob
        self.values: List[T.Tensor] = []            # 每个 [response_len]，critic 的 value
        self.rewards: List[float] = []              # 每个序列一个标量 reward
        self.ref_log_probs: List[T.Tensor] = []     # 每个 [response_len]，ref 模型的 log prob（KL 用）

    def store(
        self,
        query_ids: T.Tensor,
        query_mask: T.Tensor,
        response_ids: T.Tensor,
        response_mask: T.Tensor,
        log_probs: T.Tensor,
        values: T.Tensor,
        reward: float,
        ref_log_probs: T.Tensor,
    ) -> None:
        """存一条 rollout（可为一整批压成一条，或逐条调用）。"""
        self.queries.append(query_ids.detach().to(self.device))
        self.query_masks.append(query_mask.detach().to(self.device))
        self.responses.append(response_ids.detach().to(self.device))
        self.response_masks.append(response_mask.detach().to(self.device))
        self.log_probs.append(log_probs.detach().to(self.device))
        self.values.append(values.detach().to(self.device))
        self.rewards.append(float(reward))
        self.ref_log_probs.append(ref_log_probs.detach().to(self.device))

    def store_batch(
        self,
        query_ids: T.Tensor,
        query_mask: T.Tensor,
        response_ids: T.Tensor,
        response_mask: T.Tensor,
        log_probs: T.Tensor,
        values: T.Tensor,
        rewards: T.Tensor,
        ref_log_probs: T.Tensor,
    ) -> None:
        """存一整批 rollout（batch 中每条序列单独 append）。"""
        for i in range(query_ids.size(0)):
            self.store(
                query_ids[i],
                query_mask[i],
                response_ids[i],
                response_mask[i],
                log_probs[i],
                values[i],
                rewards[i].item(),
                ref_log_probs[i],
            )

    def clear(self) -> None:
        """清空 buffer，准备下一轮 rollout。"""
        self.queries.clear()
        self.query_masks.clear()
        self.responses.clear()
        self.response_masks.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.ref_log_probs.clear()

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(
        self,
        last_value: Optional[T.Tensor] = None,
    ) -> Tuple[List[T.Tensor], List[T.Tensor]]:
        """
        用 GAE 计算每条序列的 advantage 和 return。
        返回 (advantages, returns)，均为 list of 1d tensor，与 response 长度对齐。
        last_value: 可选，最后一条的 next value（用于 bootstrap）。
        """
        advantages: List[T.Tensor] = []
        returns: List[T.Tensor] = []

        for i in range(len(self.rewards)):
            r = self.rewards[i]
            v = self.values[i]   # [response_len]
            mask = self.response_masks[i]

            v_dev = v.detach()
            mask_dev = mask.detach()
            length = int(mask_dev.sum().item())
            adv = T.zeros_like(v_dev, device=v_dev.device)
            gae = 0.0
            for t in reversed(range(length)):
                r_t = float(r) if t == length - 1 else 0.0
                next_v_val = v_dev[t+1] if t+1 < length else 0.0
                adv[t] = r_t + self.gamma * next_v_val - v_dev[t]
                gae = adv[t] + self.gamma * self.lam * gae
                adv[t] = gae
            advantages.append(adv)
            returns.append(adv + v)

        return advantages, returns

    def get(
        self,
        compute_gae: bool = True,
        last_value: Optional[T.Tensor] = None,
    ) -> dict:
        """
        取出当前 buffer 的全部数据，用于 PPO 更新。
        若 compute_gae=True，会先算 advantages/returns，再返回。
        返回 dict，包含：
          - query_ids, query_masks, response_ids, response_masks
          - log_probs, ref_log_probs, values
          - rewards, advantages, returns（后两者在 compute_gae=True 时有）
        """
        out = {
            "queries": self.queries,
            "query_masks": self.query_masks,
            "responses": self.responses,
            "response_masks": self.response_masks,
            "log_probs": self.log_probs,
            "values": self.values,
            "rewards": self.rewards,
            "ref_log_probs": self.ref_log_probs,
        }
        if compute_gae:
            advantages, returns = self.compute_gae(last_value=last_value)
            out["advantages"] = advantages
            out["returns"] = returns
        return out

    def sample(
        self,
        mini_batch_size: Optional[int] = None,
        compute_gae: bool = True,
        last_value: Optional[T.Tensor] = None,
    ) -> dict:
        """
        与 get() 相同；若传 mini_batch_size，则随机取子集做 mini-batch 更新。
        不传则返回全部（等价 get()）。
        """
        full = self.get(compute_gae=compute_gae, last_value=last_value)
        if mini_batch_size is None or mini_batch_size >= len(self):
            return full
        indices = T.randperm(len(self), device=self.device)[:mini_batch_size].tolist()
        return {
            k: [v[i] for i in indices] if isinstance(v, list) else v
            for k, v in full.items()
        }


# ---------------------------------------------------------------------------
# Critic：value head，输入 hidden states，输出每位置的 value
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Value head：输入 (batch, seq_len, hidden_size)，输出 (batch, seq_len)。
    通常与 Actor 共享 backbone，只多这一层；或单独 backbone+head。
    """

    def __init__(self, base_model, hidden_size, dropout=0.0):
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )


    def forward(self, input_ids: T.Tensor, attention_mask: T.Tensor) -> T.Tensor:
        """
        input_ids: (batch, seq_len), attention_mask: (batch, seq_len)
        返回: (batch, seq_len) value  per position
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        values = self.head(hidden_states).squeeze(-1)
        return values


# ---------------------------------------------------------------------------
# RewardModel：奖励模型 / 规则奖励的统一接口
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    可学习的奖励模型：输入 (input_ids, attention_mask) 或 文本列表，
    输出每条一个标量 reward。子类实现 forward 或 compute_reward。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_ids: T.Tensor,
        attention_mask: T.Tensor,
    ) -> T.Tensor:
        """
        input_ids: (batch, seq_len)，一般为 prompt+response 拼接
        attention_mask: (batch, seq_len)
        返回: (batch,) 标量 reward
        """
        raise NotImplementedError("子类实现或使用 RuleRewardModel")

    def compute_reward(self, texts: List[str]) -> List[float]:
        """基于文本的接口：list[str] -> list[float]，便于规则奖励或外部模型打分。"""
        raise NotImplementedError("子类实现或使用 RuleRewardModel")


class RuleRewardModel(RewardModel):
    """
    规则奖励：不包含可学习参数，用 callable 或内置规则。
    """

    def __init__(self, reward_fn: Callable[[List[str]], List[float]]):
        super().__init__()
        self.reward_fn = reward_fn

    def forward(self, input_ids: T.Tensor, attention_mask: T.Tensor) -> T.Tensor:
        raise NotImplementedError("RuleRewardModel 请用 compute_reward(texts)")

    def compute_reward(self, texts: List[str]) -> List[float]:
        return self.reward_fn(texts)


# ---------------------------------------------------------------------------
# ActorRefRollout：reference + rollout + actor 合并为一个类（骨架，供学习实现）
# ---------------------------------------------------------------------------

class ActorRefRollout(nn.Module):
    """
    把 reference、rollout、actor 合并成一个类：
    - actor：可训练的 policy（因果 LM），用于生成和更新
    - ref：冻结的 reference，只用于算 ref_log_probs（KL 惩罚）
    - rollout：用 actor 做自回归生成（collect 阶段）

    约定：外部传入的 causal_lm 需支持：
    - .generate(input_ids, attention_mask, ...) 返回 generated_ids
    - 前向返回 logits / hidden_states，用于算 log_probs 和喂给 Critic
    """

    def __init__(
        self,
        causal_lm: nn.Module,
        tokenizer,
        ref_sync_interval: Optional[int] = None,
        device: Optional[T.device] = None,
    ):
        super().__init__()
        # TODO: 保存 actor、ref（冻结拷贝）、tokenizer、ref_sync_interval、device、_step
        self.actor = causal_lm
        self.ref = copy.deepcopy(causal_lm)
        for param in self.ref.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer
        self.ref_sync_interval = ref_sync_interval
        self.device = device or next(causal_lm.parameters()).device
        self._step = 0

    def sync_ref_from_actor(self) -> None:
        """把 ref 参数同步为当前 actor 的拷贝（可选，用于 periodic ref update）。"""
        state_dict = copy.deepcopy(self.actor.state_dict())
        self.ref.load_state_dict(state_dict)
        for param in self.ref.parameters():
            param.requires_grad = False

    def maybe_sync_ref(self) -> None:
        """若设置了 ref_sync_interval，到步数则 sync。"""
        if self.ref_sync_interval is None:
            return
        self._step += 1
        if self._step % self.ref_sync_interval == 0:
            self.sync_ref_from_actor()
        

    def _log_probs_from_model(
        self,
        model: nn.Module,
        input_ids: T.Tensor,
        attention_mask: T.Tensor,
        response_mask: T.Tensor,
    ) -> T.Tensor:
        """
        用任意 causal LM 做前向，算 response 位置的 token log_probs。
        logits[:, t, :] 预测的是 input_ids[:, t+1]，故 position t+1 的 log_prob 从 logits[:, t, :] 取。
        返回: (batch, full_len)，仅 response 位置非零。
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        log_probs_next = F.log_softmax(logits[:, :-1, :], dim=-1)  # (batch, seq_len-1, vocab_size)
        target_ids = input_ids[:, 1:].unsqueeze(-1)  # (batch, seq_len-1, 1)
        token_log_probs = log_probs_next.gather(dim=-1, index=target_ids).squeeze(-1)  # (batch, seq_len-1)
        full_len = input_ids.size(1)
        full_log_probs = T.zeros(
            input_ids.size(0), full_len,
            dtype=token_log_probs.dtype, device=token_log_probs.device,
        )
        full_log_probs[:, 1:] = token_log_probs
        return full_log_probs * response_mask

    def get_actor_log_probs(
        self,
        input_ids: T.Tensor,
        attention_mask: T.Tensor,
        response_mask: T.Tensor,
    ) -> T.Tensor:
        """
        用当前 actor 对已生成的序列算 response 部分的 log_probs。
        input_ids: (batch, full_len) = query + response 拼接
        attention_mask: (batch, full_len)
        response_mask: (batch, full_len)，仅 response 位置为 1
        返回: (batch, full_len)，仅 response 位置有效，其余为 0。
        """
        log_probs = self._log_probs_from_model(self.actor, input_ids, attention_mask, response_mask)
        return log_probs

    def get_ref_log_probs(
        self,
        input_ids: T.Tensor,
        attention_mask: T.Tensor,
        response_mask: T.Tensor,
    ) -> T.Tensor:
        """
        用 ref 对已生成的序列算 response 部分的 log_probs。
        形状约定同 get_actor_log_probs。
        """
        log_probs = self._log_probs_from_model(self.ref, input_ids, attention_mask, response_mask)
        return log_probs

    def generate(
        self,
        query_ids: T.Tensor,
        query_mask: T.Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ) -> Tuple[T.Tensor, T.Tensor, T.Tensor]:
        """
        Rollout：用 actor 自回归生成 response，再前向一次得到 response 的 log_probs。
        返回: response_ids (batch, resp_len), response_mask (batch, resp_len), log_probs (batch, resp_len)。
        """
        pad_token_id = getattr(self.tokenizer, "pad_token_id", self.tokenizer.eos_token_id)
        gen_outputs = self.actor.generate(
            input_ids=query_ids,
            attention_mask=query_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=pad_token_id,
        )
        # HF generate 返回 tensor 或带 .sequences 的对象
        generated = gen_outputs.sequences if hasattr(gen_outputs, "sequences") else gen_outputs
        query_len = query_ids.size(1)
        full_ids = generated
        response_ids = full_ids[:, query_len:]
        response_mask = (response_ids != pad_token_id).to(T.float32)
        full_attn = T.ones_like(full_ids, dtype=T.float32, device=full_ids.device)
        if query_mask is not None:
            full_attn[:, :query_len] = query_mask.to(T.float32)
        full_resp_mask = T.zeros_like(full_ids, dtype=T.float32, device=full_ids.device)
        full_resp_mask[:, query_len:] = (full_ids[:, query_len:] != pad_token_id).to(T.float32)
        full_log_probs = self.get_actor_log_probs(full_ids, full_attn, full_resp_mask)
        log_probs = full_log_probs[:, query_len:]
        return response_ids, response_mask, log_probs

    def parameters_for_optimizer(self) -> list:
        """返回需要训练的参数（仅 actor）。"""
        return list(self.actor.parameters())

    def reward_collection(
        self,
        reward_model: RewardModel,
        query_ids: T.Tensor,
        response_ids: T.Tensor,
        response_mask: T.Tensor,
    ) -> T.Tensor:
        """
        用 reward_model 根据 query+response 解码文本算 reward。
        返回: (batch,) 标量 reward。若 reward_model 只有 compute_reward(texts)，则先 decode 再调用。
        """
        texts = []
        for i in range(query_ids.size(0)):
            r_len = int(response_mask[i].sum().item())
            if r_len > 0:
                ids = response_ids[i, :r_len]
            else:
                ids = response_ids[i]
            t = self.tokenizer.decode(ids, skip_special_tokens=True)
            texts.append(t)
        rewards = reward_model.compute_reward(texts)
        return T.tensor(rewards, dtype=T.float32, device=query_ids.device)


class PPOTrainer:
    """
    手搓 LLM PPO 训练器。负责：采样 prompt → 生成 response → 算 reward / log_probs / values
    → 写入 buffer → 多 epoch mini-batch PPO 更新。
    依赖：ActorRefRollout（actor+ref+rollout）、Critic、RewardModel、PPOMemory、dataloader。
    """

    def __init__(
        self,
        actor_ref_rollout: ActorRefRollout,
        critic: Critic,
        reward_model: Union[RewardModel, Callable[[List[str]], List[float]]],
        memory: PPOMemory,
        optimizer: T.optim.Optimizer,
        ppo_epochs: int = 4,
        mini_batch_size: int = 8,
        clip_range: float = 0.2,
        kl_coef: float = 0.02,
        max_new_tokens: int = 256,
        device: Optional[T.device] = None,
    ):
        self.actor_ref = actor_ref_rollout
        self.critic = critic
        self.reward_model = reward_model if hasattr(reward_model, "compute_reward") else RuleRewardModel(reward_model)
        self.memory = memory
        self.optimizer = optimizer
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_range = clip_range
        self.kl_coef = kl_coef
        self.max_new_tokens = max_new_tokens
        self.device = device or next(actor_ref_rollout.actor.parameters()).device

    def generate(self, query_ids: T.Tensor, query_mask: T.Tensor) -> Tuple[T.Tensor, T.Tensor, T.Tensor, T.Tensor]:
        """
        用 ActorRefRollout 做 rollout，返回 response_ids, response_mask, log_probs, values。
        values 需用 critic + actor 的 hidden_states 在外部或内部算。
        """
        raise NotImplementedError("actor_ref.generate + critic 算 values")

    def get_ref_log_probs(self, query_ids: T.Tensor, query_mask: T.Tensor, response_ids: T.Tensor, response_mask: T.Tensor) -> T.Tensor:
        """用 ActorRefRollout.ref 算 response 的 log prob。"""
        raise NotImplementedError("actor_ref.get_ref_log_probs")

    def collect_rollouts(self, dataloader) -> None:
        """
        从 dataloader 取若干 batch 的 prompt，生成 response，算 reward / log_probs / values / ref_log_probs，
        写入 memory；凑够一个 rollout batch 后返回（由调用方决定取多少 batch）。
        """
        raise NotImplementedError("循环 dataloader → generate → reward_fn → get_ref_log_probs → memory.store_batch")

    def _build_full_sequence_batch(self, batch: dict, pad_token_id: int) -> Tuple[T.Tensor, T.Tensor, T.Tensor, List[int], List[int], int]:
        """
        把 memory 里取出的 list 形式的 mini_batch 拼成 padded 的 full 序列。
        返回: full_ids, full_attn, full_resp_mask, qlens, resp_lens, max_r。
        """
        raise NotImplementedError

    def ppo_step(self, batch: dict) -> dict:
        """
        用 buffer 里取出的一个 mini_batch 做一次 PPO 更新：policy clip loss + value loss + KL 惩罚，
        然后 backward + optimizer.step。
        返回 stats dict（policy_loss, value_loss, kl_loss, total_loss）。
        """
        raise NotImplementedError

    def train_step(self) -> dict:
        """
        一次完整训练步：1）用当前 buffer 数据 get()；2）多 epoch 内多次 sample(mini_batch_size) 调用 ppo_step。
        返回汇总的 stats（可平均各 mini_batch 的 loss / kl）。
        """
        raise NotImplementedError

    def run(self, dataloader, total_steps: int) -> None:
        """
        主循环：每一步 collect_rollouts 攒一 buffer → train_step 更新 → memory.clear()；
        共执行 total_steps 步（或按 dataloader 轮数）。
        """
        raise NotImplementedError("for step in range(total_steps): collect_rollouts → train_step → memory.clear()")


def _aggregate_stats(stats_list: list) -> dict:
    """把多个 ppo_step 返回的 stats 求平均。"""
    if not stats_list:
        return {}
    keys = stats_list[0].keys()
    return {k: sum(s[k] for s in stats_list) / len(stats_list) for k in keys}


def _pad_list_to_tensor(
    tensors: List[T.Tensor],
    pad_value: float,
    device: T.device,
    max_len: Optional[int] = None,
) -> Tuple[T.Tensor, T.Tensor]:
    """
    把 list of 1d tensor  pad 成 (batch, max_len)，并返回有效位置 mask (1=有效)。
    返回: (stacked, mask), 均为 (batch, max_len)。
    """
    if max_len is None:
        max_len = max(t.size(0) for t in tensors)
    batch = len(tensors)
    dtype = tensors[0].dtype
    stacked = T.full(
        (batch, max_len), pad_value, dtype=dtype, device=device,
    )
    mask = T.zeros(batch, max_len, dtype=T.float32, device=device)
    for i, t in enumerate(tensors):
        L = t.size(0)
        stacked[i, :L] = t.to(device)
        mask[i, :L] = 1.0
    return stacked, mask
