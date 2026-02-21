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
    for batch in ppo_trainer.dataloader:
        queries = batch["input_ids"]
        responses = ppo_trainer.generate(
            queries,
            max_new_tokens=data_cfg.get("max_response_length", 512),
        )
        texts = mt.tokenizer.batch_decode(responses, skip_special_tokens=True)
        rewards = torch.tensor(reward_fn(texts), dtype=torch.float32).to(model.device)
        stats = ppo_trainer.step(queries, responses, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)