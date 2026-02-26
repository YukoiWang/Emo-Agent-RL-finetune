# -*- coding: utf-8 -*-
"""
端到端 PPO 训练测试：用 2-3 个 profile 跑通完整流程。
- Actor: Qwen2.5-0.5B-Instruct + LoRA
- 用户模拟: PlayerSimulatorWithPlanning + DeepSeek API
- Reward: emo_point based (mode1)
- PPO: 1 rollout + 1 update
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-8136694e31ae47098ed0fa350f5ea610")

import torch

from src.data.profile_dataset import load_profiles, build_initial_prompt
from src.training.ppo_training import ActorRefRollout, Critic, PPOMemory
from src.training.qwen_user_simulator import build_qwen_user_llm_fn
from src.training.ppo_emo_rollout import collect_rollouts_emo


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # ========================
    # 1. 加载模型
    # ========================
    print("\n[STEP 1] 加载 Qwen2.5-0.5B-Instruct + LoRA ...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    print(f"  模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    print(f"  模型加载完成 ({time.time()-t0:.1f}s)")
    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ========================
    # 2. 构建 ActorRefRollout + Critic
    # ========================
    print("\n[STEP 2] 构建 ActorRefRollout + Critic ...")
    t0 = time.time()
    actor_ref = ActorRefRollout(
        causal_lm=model,
        tokenizer=tokenizer,
        device=device,
    )

    hidden_size = base_model.config.hidden_size
    import copy
    critic_base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    for p in critic_base_model.parameters():
        p.requires_grad = False
    critic = Critic(critic_base_model, hidden_size, dropout=0.0).to(device)
    print(f"  构建完成 ({time.time()-t0:.1f}s), hidden_size={hidden_size}")

    # ========================
    # 3. 加载 2-3 个 profile
    # ========================
    print("\n[STEP 3] 加载 profile ...")
    profiles = load_profiles(os.path.join(ROOT, "data/data"), split="train")
    n_test = min(2, len(profiles))
    test_profiles = profiles[:n_test]
    batch_items = []
    for i, p in enumerate(test_profiles):
        prompt = build_initial_prompt(p)
        batch_items.append({
            "profile": p,
            "prompt": prompt,
            "idx": i,
            "target": "eq",
        })
    print(f"  选取 {n_test} 个 profile")
    for item in batch_items:
        print(f"    [{item['idx']}] id={item['profile'].get('id','')[:12]}... task={item['profile'].get('task','')[:40]}")

    # ========================
    # 4. 构建 user_llm_fn (DeepSeek API)
    # ========================
    print("\n[STEP 4] 构建 DeepSeek user_llm_fn ...")
    user_llm_fn = build_qwen_user_llm_fn(
        model="deepseek-chat",
        temperature=0.7,
    )

    # ========================
    # 5. PPOMemory + ref_log_probs_fn
    # ========================
    print("\n[STEP 5] 构建 PPOMemory + optimizer ...")
    memory = PPOMemory(gamma=1.0, lam=0.95, device=device)

    def get_ref_log_probs_fn(query_ids, query_mask, response_ids, response_mask):
        q_len = query_ids.size(1)
        full_ids = torch.cat([query_ids, response_ids], dim=1)
        full_attn = torch.ones_like(full_ids, dtype=torch.float32, device=device)
        full_attn[:, :q_len] = query_mask.to(torch.float32)
        full_attn[:, q_len:] = response_mask.to(torch.float32)
        full_resp_mask = torch.zeros_like(full_ids, dtype=torch.float32, device=device)
        full_resp_mask[:, q_len:] = response_mask.to(torch.float32)
        full_lp = actor_ref.get_ref_log_probs(full_ids, full_attn, full_resp_mask)
        return full_lp[:, q_len:]

    optimizer = torch.optim.AdamW(
        list(actor_ref.parameters_for_optimizer()) + list(critic.parameters()),
        lr=1e-6,
    )

    # ========================
    # 6. 多轮 Rollout
    # ========================
    print(f"\n[STEP 6] 多轮对话 rollout (max_turns=3, {n_test} profiles) ...")
    print("  （每个 profile 会调多次 DeepSeek API，请耐心等待）")
    t0 = time.time()
    try:
        orig_reward, pen_reward = collect_rollouts_emo(
            batch_items=batch_items,
            actor_ref=actor_ref,
            critic=critic,
            tokenizer=tokenizer,
            user_llm_fn=user_llm_fn,
            memory=memory,
            get_ref_log_probs_fn=get_ref_log_probs_fn,
            device=device,
            reward_mode="mode1",
            max_turns=3,
            max_new_tokens_per_turn=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            target="eq",
        )
        print(f"  Rollout 完成 ({time.time()-t0:.1f}s)")
        print(f"  memory size: {len(memory)}")
        print(f"  orig_reward shape: {orig_reward.shape}")
        print(f"  rewards: {[memory.rewards[i] for i in range(len(memory))]}")
    except Exception as e:
        print(f"  [FAIL] Rollout 失败: {e}")
        traceback.print_exc()
        return

    # ========================
    # 7. PPO Update
    # ========================
    print("\n[STEP 7] PPO update ...")
    t0 = time.time()
    try:
        if len(memory) == 0:
            print("  [WARN] memory 为空，跳过 PPO update")
            return

        data = memory.get(compute_gae=True)

        from src.training.ppo_training import _pad_list_to_tensor
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        queries = data["queries"]
        responses = data["responses"]
        qlens = [q.size(0) for q in queries]
        resp_lens = [r.size(0) for r in responses]
        full_seqs = [torch.cat([q, r], dim=0) for q, r in zip(queries, responses)]
        max_len = max(s.size(0) for s in full_seqs)
        batch_size = len(full_seqs)

        full_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        full_attn = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)
        full_resp_mask = torch.zeros(batch_size, max_len, dtype=torch.float32, device=device)

        for i, seq in enumerate(full_seqs):
            L = seq.size(0)
            full_ids[i, :L] = seq.to(device)
            qlen, rlen = qlens[i], resp_lens[i]
            full_attn[i, :qlen] = data["query_masks"][i].to(torch.float32).to(device)
            full_attn[i, qlen:qlen + rlen] = data["response_masks"][i].to(torch.float32).to(device)
            full_resp_mask[i, qlen:qlen + rlen] = data["response_masks"][i].to(torch.float32).to(device)

        new_log_probs = actor_ref.get_actor_log_probs(full_ids, full_attn, full_resp_mask)
        values = critic(full_ids, full_attn)

        old_log_probs, _ = _pad_list_to_tensor(data["log_probs"], 0.0, device)
        ref_log_probs, _ = _pad_list_to_tensor(data["ref_log_probs"], 0.0, device)
        advantages, adv_mask = _pad_list_to_tensor(data["advantages"], 0.0, device)
        returns, ret_mask = _pad_list_to_tensor(data["returns"], 0.0, device)
        resp_mask_padded, _ = _pad_list_to_tensor(data["response_masks"], 0.0, device)
        n_valid = resp_mask_padded.sum().clamp(min=1e-8)

        max_r = max(resp_lens)
        new_lp_resp = torch.zeros(batch_size, max_r, dtype=torch.float32, device=device)
        values_resp = torch.zeros(batch_size, max_r, dtype=torch.float32, device=device)
        for i in range(batch_size):
            s, rlen = qlens[i], resp_lens[i]
            new_lp_resp[i, :rlen] = new_log_probs[i, s:s + rlen]
            values_resp[i, :rlen] = values[i, s:s + rlen]

        ratio = torch.exp(new_lp_resp - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        policy_loss = -(torch.minimum(surr1, surr2) * adv_mask).sum() / n_valid
        value_loss = ((values_resp - returns) ** 2 * ret_mask).sum() / n_valid
        kl_loss = ((old_log_probs - ref_log_probs) * adv_mask).sum() / n_valid
        total_loss = policy_loss + 0.5 * value_loss + 0.02 * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(actor_ref.parameters_for_optimizer()) + list(critic.parameters()), 1.0,
        )
        optimizer.step()

        print(f"  PPO update 完成 ({time.time()-t0:.1f}s)")
        print(f"  policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, "
              f"kl_loss={kl_loss.item():.4f}, total_loss={total_loss.item():.4f}")
    except Exception as e:
        print(f"  [FAIL] PPO update 失败: {e}")
        traceback.print_exc()
        return

    print("\n" + "=" * 60)
    print("[ALL PASS] PPO 端到端测试通过！完整流程：")
    print("  模型加载 → ActorRefRollout + Critic → 多轮对话 Rollout → Reward → PPO Update")
    print("=" * 60)


if __name__ == "__main__":
    main()
