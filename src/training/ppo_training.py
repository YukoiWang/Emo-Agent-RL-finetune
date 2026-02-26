"""
PPO components for LLM fine-tuning:
- PPOMemory: rollout buffer + GAE
- Critic: value head
- RewardModel: learnable / rule-based reward interface
- ActorRefRollout: reference + rollout + actor in one class
- PPOTrainer: training loop
"""
import copy
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Callable


class PPOMemory:
    """
    Rollout buffer for LLM PPO.
    Stores per sequence: (query, response, log_probs, values, reward, ref_log_probs).
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

        self.queries: List[T.Tensor] = []           # [query_len] each
        self.query_masks: List[T.Tensor] = []       # [query_len] each
        self.responses: List[T.Tensor] = []         # [response_len] each, generated token ids only
        self.response_masks: List[T.Tensor] = []   # [response_len] each
        self.log_probs: List[T.Tensor] = []         # [response_len] each, current policy log prob
        self.values: List[T.Tensor] = []            # [response_len] each, critic value
        self.rewards: List[float] = []              # scalar reward per sequence
        self.ref_log_probs: List[T.Tensor] = []     # [response_len] each, ref log prob for KL

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
        """Store one rollout (can be a whole batch or per-item)."""
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
        """Store a batch of rollouts (each sequence appended separately)."""
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
        """Clear buffer for next rollout."""
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
        Compute advantage and return per sequence using GAE.
        Returns (advantages, returns), both list of 1d tensors aligned with response length.
        last_value: optional next value for bootstrap.
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
        Get all data from buffer for PPO update.
        If compute_gae=True, computes advantages/returns before returning.
        Returns dict with: query_ids, query_masks, response_ids, response_masks,
        log_probs, ref_log_probs, values, rewards, advantages, returns (when compute_gae=True).
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
        Same as get(); if mini_batch_size given, randomly sample a mini-batch.
        If not given, returns all (equivalent to get()).
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
# Critic: value head, maps hidden states to value per position
# ---------------------------------------------------------------------------

class Critic(nn.Module):
    """
    Value head: (batch, seq_len, hidden_size) -> (batch, seq_len).
    Typically shares backbone with Actor, or has separate backbone+head.
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
        Returns: (batch, seq_len) value per position
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True, output_hidden_states=True,
        )
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states.to(self.head[0].weight.dtype)
        values = self.head(hidden_states).squeeze(-1)
        return values


# ---------------------------------------------------------------------------
# RewardModel: learnable / rule-based reward interface
# ---------------------------------------------------------------------------

class RewardModel(nn.Module):
    """
    Learnable reward model: (input_ids, attention_mask) or text list -> scalar reward per item.
    Subclasses implement forward or compute_reward.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_ids: T.Tensor,
        attention_mask: T.Tensor,
    ) -> T.Tensor:
        """
        input_ids: (batch, seq_len), typically prompt+response
        attention_mask: (batch, seq_len)
        Returns: (batch,) scalar reward
        """
        raise NotImplementedError("Subclass or use RuleRewardModel")

    def compute_reward(self, texts: List[str]) -> List[float]:
        """Text-based interface: list[str] -> list[float]."""
        raise NotImplementedError("Subclass or use RuleRewardModel")


class RuleRewardModel(RewardModel):
    """
    Rule-based reward: no learnable params, uses callable.
    """

    def __init__(self, reward_fn: Callable[[List[str]], List[float]]):
        super().__init__()
        self.reward_fn = reward_fn

    def forward(self, input_ids: T.Tensor, attention_mask: T.Tensor) -> T.Tensor:
        raise NotImplementedError("RuleRewardModel: use compute_reward(texts)")

    def compute_reward(self, texts: List[str]) -> List[float]:
        return self.reward_fn(texts)


# ---------------------------------------------------------------------------
# ActorRefRollout: reference + rollout + actor in one class
# ---------------------------------------------------------------------------

class ActorRefRollout(nn.Module):
    """
    Combines reference, rollout, and actor:
    - actor: trainable policy (causal LM) for generation and update
    - ref: frozen reference for ref_log_probs (KL penalty)
    - rollout: actor does autoregressive generation (collect phase)

    causal_lm must support:
    - .generate(input_ids, attention_mask, ...) -> generated_ids
    - forward returns logits / hidden_states for log_probs and Critic
    """

    def __init__(
        self,
        causal_lm: nn.Module,
        tokenizer,
        ref_sync_interval: Optional[int] = None,
        device: Optional[T.device] = None,
    ):
        super().__init__()
        # actor, ref (frozen copy), tokenizer, ref_sync_interval, device, _step
        self.actor = causal_lm
        self.ref = copy.deepcopy(causal_lm)
        for param in self.ref.parameters():
            param.requires_grad = False
        self.tokenizer = tokenizer
        self.ref_sync_interval = ref_sync_interval
        self.device = device or next(causal_lm.parameters()).device
        self._step = 0

    def sync_ref_from_actor(self) -> None:
        """Sync ref params from current actor (optional periodic ref update)."""
        state_dict = copy.deepcopy(self.actor.state_dict())
        self.ref.load_state_dict(state_dict)
        for param in self.ref.parameters():
            param.requires_grad = False

    def maybe_sync_ref(self) -> None:
        """Sync ref when ref_sync_interval steps reached."""
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
        Compute response-position token log_probs from causal LM.
        logits[:, t, :] predicts input_ids[:, t+1], so log_prob at t+1 from logits[:, t, :].
        Returns: (batch, full_len), only response positions non-zero.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs.logits  # (batch, seq_len, vocab_size)
        del outputs

        # Use cross_entropy to avoid materializing full (batch, seq-1, vocab) log_softmax
        shift_logits = logits[:, :-1, :].contiguous()
        del logits
        shift_labels = input_ids[:, 1:].contiguous()
        batch_size, seq_minus1, vocab = shift_logits.shape
        token_log_probs = -F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            reduction="none",
        ).view(batch_size, seq_minus1)
        del shift_logits, shift_labels

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
        Get actor log_probs for response part of generated sequence.
        input_ids: (batch, full_len) = query + response
        attention_mask: (batch, full_len)
        response_mask: (batch, full_len), 1 at response positions
        Returns: (batch, full_len), only response positions valid.
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
        Get ref log_probs for response part.
        Same shape convention as get_actor_log_probs.
        """
        log_probs = self._log_probs_from_model(self.ref, input_ids, attention_mask, response_mask)
        return log_probs

    @T.no_grad()
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
        Rollout: actor autoregressively generates response, then forward for log_probs.
        Returns: response_ids, response_mask, log_probs (each batch, resp_len).
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
        generated = gen_outputs.sequences if hasattr(gen_outputs, "sequences") else gen_outputs
        del gen_outputs
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
        """Return trainable params (actor only)."""
        return list(self.actor.parameters())

    def reward_collection(
        self,
        reward_model: RewardModel,
        query_ids: T.Tensor,
        response_ids: T.Tensor,
        response_mask: T.Tensor,
    ) -> T.Tensor:
        """
        Compute reward via reward_model from query+response decoded text.
        Returns: (batch,) scalar reward. Decodes then calls compute_reward(texts).
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
    PPO trainer for LLM: sample prompt -> generate response -> reward / log_probs / values
    -> write to buffer -> multi-epoch mini-batch PPO update.
    Depends on: ActorRefRollout, Critic, RewardModel, PPOMemory, dataloader.
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
        Rollout via ActorRefRollout, return response_ids, response_mask, log_probs, values.
        values computed from critic + actor hidden states.
        """
        response_ids, response_mask, log_probs = self.actor_ref.generate(
            query_ids,
            query_mask,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=1.0,
            temperature=1.0,
        )
        pad_id = getattr(self.actor_ref.tokenizer, "pad_token_id", self.actor_ref.tokenizer.eos_token_id)
        query_len = query_ids.size(1)
        full_ids = T.cat([query_ids, response_ids], dim=1)
        full_attn = T.ones_like(full_ids, dtype=T.float32, device=full_ids.device)
        full_attn[:, :query_len] = query_mask.to(T.float32)
        full_attn[:, query_len:] = response_mask.to(T.float32)
        values_full = self.critic(full_ids, full_attn)
        values = values_full[:, query_len:]
        return response_ids, response_mask, log_probs, values

    def get_ref_log_probs(self, query_ids: T.Tensor, query_mask: T.Tensor, response_ids: T.Tensor, response_mask: T.Tensor) -> T.Tensor:
        """Get ref log_probs for response via ActorRefRollout.ref."""
        query_len = query_ids.size(1)
        full_ids = T.cat([query_ids, response_ids], dim=1)
        full_attn = T.ones_like(full_ids, dtype=T.float32, device=full_ids.device)
        full_attn[:, :query_len] = query_mask.to(T.float32)
        full_attn[:, query_len:] = response_mask.to(T.float32)
        full_resp_mask = T.zeros_like(full_ids, dtype=T.float32, device=full_ids.device)
        full_resp_mask[:, query_len:] = response_mask.to(T.float32)
        full_log_probs = self.actor_ref.get_ref_log_probs(full_ids, full_attn, full_resp_mask)
        return full_log_probs[:, query_len:]

    def collect_rollouts(
        self,
        dataloader,
        max_prompt_length: int = 1024,
        batches_per_rollout: int = 1,
    ) -> None:
        """
        Fetch batches from dataloader -> generate response -> reward / log_probs / values / ref_log_probs
        -> store in memory. Caller controls how many batches per rollout.
        """
        tokenizer = self.actor_ref.tokenizer
        pad_id = getattr(tokenizer, "pad_token_id", tokenizer.eos_token_id)
        it = iter(dataloader)
        for _ in range(batches_per_rollout):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dataloader)
                batch = next(it)
            if isinstance(batch, (list, tuple)):
                texts = [b["user"] if isinstance(b, dict) else b for b in batch]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length,
                    pad_to_multiple_of=1,
                    return_attention_mask=True,
                )
                query_ids = enc["input_ids"].to(self.device)
                query_mask = enc.get("attention_mask")
                if query_mask is not None:
                    query_mask = query_mask.to(self.device)
                else:
                    query_mask = (query_ids != pad_id).to(T.float32)
            elif isinstance(batch, dict) and "user" in batch:
                texts = batch["user"] if isinstance(batch["user"], list) else [batch["user"]]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_prompt_length,
                    pad_to_multiple_of=1,
                    return_attention_mask=True,
                )
                query_ids = enc["input_ids"].to(self.device)
                query_mask = enc.get("attention_mask")
                if query_mask is not None:
                    query_mask = query_mask.to(self.device)
                else:
                    query_mask = (query_ids != pad_id).to(T.float32)
            else:
                query_ids = batch["input_ids"].to(self.device)
                query_mask = batch.get("attention_mask")
                if query_mask is not None:
                    query_mask = query_mask.to(self.device)
                else:
                    query_mask = (query_ids != pad_id).to(T.float32)

            with T.no_grad():
                response_ids, response_mask, log_probs, values = self.generate(query_ids, query_mask)
                ref_log_probs = self.get_ref_log_probs(query_ids, query_mask, response_ids, response_mask)
                texts = []
                for i in range(query_ids.size(0)):
                    r_len = int(response_mask[i].sum().item())
                    ids = response_ids[i, :r_len] if r_len > 0 else response_ids[i]
                    t = tokenizer.decode(ids, skip_special_tokens=True)
                    texts.append(t)
                rewards = self.reward_model.compute_reward(texts)
                rewards_t = T.tensor(rewards, dtype=T.float32, device=self.device)
            self.memory.store_batch(
                query_ids, query_mask, response_ids, response_mask,
                log_probs, values, rewards_t, ref_log_probs,
            )

    def _build_full_sequence_batch(self, batch: dict, pad_token_id: int) -> Tuple[T.Tensor, T.Tensor, T.Tensor, List[int], List[int], int]:
        """
        Build padded full sequences from memory mini_batch lists.
        Returns: full_ids, full_attn, full_resp_mask, qlens, resp_lens, max_r.
        """
        queries = batch["queries"]
        query_masks = batch["query_masks"]
        responses = batch["responses"]
        response_masks = batch["response_masks"]
        qlens = [q.size(0) for q in queries]
        resp_lens = [r.size(0) for r in responses]
        full_seqs = [T.cat([q, r], dim=0) for q, r in zip(queries, responses)]
        max_len = max(s.size(0) for s in full_seqs)
        batch_size = len(full_seqs)
        full_ids = T.full(
            (batch_size, max_len), pad_token_id,
            dtype=queries[0].dtype, device=self.device,
        )
        full_attn = T.zeros(batch_size, max_len, dtype=T.float32, device=self.device)
        full_resp_mask = T.zeros(batch_size, max_len, dtype=T.float32, device=self.device)
        for i, seq in enumerate(full_seqs):
            L = seq.size(0)
            full_ids[i, :L] = seq.to(self.device)
            qlen, rlen = qlens[i], resp_lens[i]
            full_attn[i, :qlen] = query_masks[i].to(T.float32).to(self.device)
            full_attn[i, qlen:qlen + rlen] = response_masks[i].to(T.float32).to(self.device)
            full_resp_mask[i, qlen:qlen + rlen] = response_masks[i].to(T.float32).to(self.device)
        max_r = max(resp_lens)
        return full_ids, full_attn, full_resp_mask, qlens, resp_lens, max_r

    def ppo_step(self, batch: dict) -> dict:
        """
        One PPO update on mini_batch: policy clip loss + value loss + KL penalty,
        then backward + optimizer.step.
        Returns stats dict: policy_loss, value_loss, kl_loss, total_loss.
        """
        pad_id = getattr(self.actor_ref.tokenizer, "pad_token_id", self.actor_ref.tokenizer.eos_token_id)
        full_ids, full_attn, full_resp_mask, qlens, resp_lens, _ = self._build_full_sequence_batch(batch, pad_id)

        old_log_probs_list = batch["log_probs"]
        ref_log_probs_list = batch["ref_log_probs"]
        advantages_list = batch["advantages"]
        returns_list = batch["returns"]

        old_log_probs, _ = _pad_list_to_tensor(old_log_probs_list, 0.0, self.device)
        ref_log_probs, _ = _pad_list_to_tensor(ref_log_probs_list, 0.0, self.device)
        advantages, adv_mask = _pad_list_to_tensor(advantages_list, 0.0, self.device)
        returns, ret_mask = _pad_list_to_tensor(returns_list, 0.0, self.device)

        resp_mask_padded, _ = _pad_list_to_tensor(batch["response_masks"], 0.0, self.device)
        n_valid = resp_mask_padded.sum().clamp(min=1e-8)

        new_log_probs = self.actor_ref.get_actor_log_probs(full_ids, full_attn, full_resp_mask)
        batch_size = full_ids.size(0)
        new_log_probs_resp = T.zeros_like(old_log_probs, device=self.device)
        for i in range(batch_size):
            s, rlen = qlens[i], resp_lens[i]
            new_log_probs_resp[i, :rlen] = new_log_probs[i, s:s + rlen]

        values = self.critic(full_ids, full_attn)
        values_resp = T.zeros_like(returns, device=self.device)
        for i in range(batch_size):
            s, rlen = qlens[i], resp_lens[i]
            values_resp[i, :rlen] = values[i, s:s + rlen]

        ratio = T.exp(new_log_probs_resp - old_log_probs)
        surr1 = ratio * advantages
        surr2 = T.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -(T.minimum(surr1, surr2) * adv_mask).sum() / n_valid
        value_loss = ((values_resp - returns) ** 2 * ret_mask).sum() / n_valid
        kl_loss = ((old_log_probs - ref_log_probs) * adv_mask).sum() / n_valid
        total_loss = policy_loss + 0.5 * value_loss + self.kl_coef * kl_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        T.nn.utils.clip_grad_norm_(self.actor_ref.parameters_for_optimizer(), 1.0)
        self.optimizer.step()

        return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "kl_loss": kl_loss.item(), "total_loss": total_loss.item()}

    def train_step(self) -> dict:
        if len(self.memory) == 0:
            return {}
        data = self.memory.get(compute_gae=True)
        all_stats = []
        for _ in range(self.ppo_epochs):
            mini = self.memory.sample(mini_batch_size=self.mini_batch_size, compute_gae=False)
            mini["advantages"] = [data["advantages"][i] for i in [data["queries"].index(q) for q in mini["queries"]]]
            mini["returns"] = [data["returns"][i] for i in [data["queries"].index(q) for q in mini["queries"]]]
            # Sample indices from data
            indices = list(range(len(data["queries"])))
            if self.mini_batch_size and self.mini_batch_size < len(indices):
                import random
                idx = random.sample(indices, self.mini_batch_size)
            else:
                idx = indices
            mini = {k: [v[i] for i in idx] if isinstance(v, list) else v for k, v in data.items()}
            stats = self.ppo_step(mini)
            all_stats.append(stats)
        return _aggregate_stats(all_stats) if all_stats else {}

    def run(self, dataloader, total_steps: int, max_prompt_length: int = 1024) -> None:
        for step in range(total_steps):
            self.collect_rollouts(dataloader, max_prompt_length=max_prompt_length, batches_per_rollout=1)
            if len(self.memory) > 0:
                self.train_step()
            self.memory.clear()


def _aggregate_stats(stats_list: list) -> dict:
    """Average stats from multiple ppo_step calls."""
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
    """Pad list of 1d tensors to (batch, max_len), return valid mask (1=valid)."""
    if max_len is None:
        max_len = max(t.size(0) for t in tensors)
    batch = len(tensors)
    dtype = tensors[0].dtype
    stacked = T.full((batch, max_len), pad_value, dtype=dtype, device=device)
    mask = T.zeros(batch, max_len, dtype=T.float32, device=device)
    for i, t in enumerate(tensors):
        L = t.size(0)
        stacked[i, :L] = t.to(device)
        mask[i, :L] = 1.0
    return stacked, mask
