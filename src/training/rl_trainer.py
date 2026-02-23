import glob
import os
import shutil
from typing import Dict, Any, Callable, List, Optional

import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.data.rl_dataset import load_rl_dataset
from src.models.modeling import load_sft_model, ModelAndTokenizer
from src.training.reward_emo import build_reward_fn_emo

def build_ppo_config(cfg: Dict[str, Any]) -> PPOConfig:
    return PPOConfig(
        batch_size=cfg.get("batch_size", 64),
        mini_batch_size=cfg.get("mini_batch_size", 8),
        ppo_epochs=cfg.get("ppo_epochs", 4),
        learning_rate=cfg.get("learning_rate", 1e-6),
        gamma=cfg.get("gamma", 1.0),
        lam=cfg.get("lam", 0.95),
        cliprange=cfg.get("clip_range", 0.2),
        cliprange_value=cfg.get("clip_range_value", 0.2),
        init_kl_coef=cfg.get("kl_penalty_coef", 0.02),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    )


def simple_empathy_reward_fn(texts: List[str]) -> List[float]:
    """
    非常简单的规则奖励示例：
    - 包含一些共情关键词 +1
    - 出现明显危险词 -1
    实际使用时建议替换为训练好的 reward model 或人工打分数据。
    """
    positive_keywords = ["我理解你", "听起来你", "能感受到你", "谢谢你愿意分享", "你一定很不容易"]
    negative_keywords = ["自杀", "杀人", "别在乎", "无所谓", "不重要"]

    rewards = []
    for t in texts:
        score = 0.0
        for k in positive_keywords:
            if k in t:
                score += 1.0
        for k in negative_keywords:
            if k in t:
                score -= 1.0
        rewards.append(score)
    return rewards


def run_ppo_training(cfg: Dict[str, Any], reward_fn: Optional[Callable[[List[str]], List[float]]] = None) -> None:
    """
    使用 PPO 对 SFT 后模型做一次 RL 微调。
    reward_fn：输入生成的回答列表，输出对应的 reward 列表。
    """
    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    rl_cfg = cfg["rl"]["ppo"]
    training_cfg = cfg.get("training", {})
    output_dir = training_cfg.get("output_dir", "outputs/rl")
    total_steps = training_cfg.get("total_steps", 0)  # 0 = 跑完整个 dataset
    save_steps = training_cfg.get("save_steps", 500)
    save_total_limit = training_cfg.get("save_total_limit", 3)

    if reward_fn is None:
        reward_cfg = cfg.get("reward", {}) or {}
        if reward_cfg.get("type") == "emo":
            step_counter = [0]
            def step_fn():
                step_counter[0] += 1
                return step_counter[0]
            reward_fn = build_reward_fn_emo(
                emo_adapter_path=reward_cfg.get("emo_adapter_path"),
                reward_mode=reward_cfg.get("reward_mode", "mode1"),
                w1=reward_cfg.get("w1", 1.0),
                w2=reward_cfg.get("w2", 0.3),
                w3=reward_cfg.get("w3", 0.2),
                trend_n=reward_cfg.get("trend_n", 5),
                step_fn=step_fn if reward_cfg.get("reward_mode") == "mode3" else None,
                S1=reward_cfg.get("S1", 100),
                S2=reward_cfg.get("S2", 300),
                warmup_steps=reward_cfg.get("warmup_steps", 200),
            )
        else:
            reward_fn = simple_empathy_reward_fn

    # 1. 加载 SFT 模型（use_lora=True 时施加 LoRA，仅训练 LoRA 参数）
    lora_cfg = model_cfg.get("lora") if isinstance(model_cfg.get("lora"), dict) else None
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
        lora_config=lora_cfg,
    )

    # gradient checkpointing 在 wrap 之前启用，减少显存
    if rl_cfg.get("gradient_checkpointing", False):
        if hasattr(mt.model, "gradient_checkpointing_enable"):
            mt.model.gradient_checkpointing_enable()

    # 转为带 value head 的模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(mt.model)

    # 2. 加载 RL 数据集
    dataset = load_rl_dataset(
        train_file=data_cfg["train_file"],
        num_proc=data_cfg.get("num_proc", 4),
    )

    # 2b. 将 user 文本 tokenize，生成 input_ids 和 query（PPOTrainer 需要）
    max_prompt_len = data_cfg.get("max_prompt_length", 512)
    tokenizer = mt.tokenizer
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    def tokenize_sample(sample):
        enc = tokenizer(
            sample["user"],
            truncation=True,
            max_length=max_prompt_len,
            padding=False,
            return_tensors=None,
        )
        sample["input_ids"] = enc["input_ids"]
        sample["query"] = sample["user"]
        return sample

    dataset = dataset.map(tokenize_sample, num_proc=data_cfg.get("num_proc", 4), desc="tokenize")
    dataset.set_format(type=None, columns=["input_ids", "query"])

    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}

    # 3. PPO 配置
    ppo_config = build_ppo_config(rl_cfg)

    # 4. PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # 5. 训练循环
    os.makedirs(output_dir, exist_ok=True)
    global_step = 0

    # PPO 生成时必须 do_sample=True，否则默认贪婪解码会导致 KL 散度为负、训练失败
    max_new_tokens = data_cfg.get("max_response_length", 256)
    generation_kwargs = {
        "do_sample": True,  # 采样生成，允许探索
        "top_p": 1.0,       # nucleus sampling（1.0 即不截断）
        "temperature": 1.0, # 控制随机性
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "max_new_tokens": max_new_tokens,
    }

    def _save_checkpoint():
        """保存 Actor（pretrained_model，含 LoRA），推理时只需这部分。"""
        ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
        model.pretrained_model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        print(f"  [saved] {ckpt_dir}")

    def _prune_checkpoints():
        if save_total_limit <= 0:
            return
        pattern = os.path.join(output_dir, "checkpoint-*")
        ckpts = sorted(glob.glob(pattern), key=lambda p: int(p.split("-")[-1]))
        while len(ckpts) > save_total_limit:
            old = ckpts.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"  [pruned] {old}")

    for batch in ppo_trainer.dataloader:
        if total_steps and global_step >= total_steps:
            break

        query_tensors = [torch.tensor(ids).to(model.pretrained_model.device) for ids in batch["input_ids"]]
        responses = ppo_trainer.generate(query_tensors, **generation_kwargs)
        response_tensors = [r.squeeze() for r in responses]
        texts = mt.tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
        reward_values = reward_fn(texts)
        reward_tensors = [torch.tensor(r, dtype=torch.float32).to(model.pretrained_model.device) for r in reward_values]
        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
        ppo_trainer.log_stats(stats, batch, reward_tensors)

        global_step += 1

        # 定期保存 checkpoint
        if save_steps and global_step % save_steps == 0:
            _save_checkpoint()
            _prune_checkpoints()

    # 训练结束，保存最终模型
    final_dir = os.path.join(output_dir, "final")
    model.pretrained_model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[PPO] 训练完成，模型已保存到 {final_dir}")