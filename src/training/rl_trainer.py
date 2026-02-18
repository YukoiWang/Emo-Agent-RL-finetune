from typing import Dict, Any, Callable

import torch
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

from src.data.rl_dataset import load_rl_dataset
from src.models.modeling import load_sft_model, ModelAndTokenizer


def build_ppo_config(cfg: Dict[str, Any]) -> PPOConfig:
    return PPOConfig(
        batch_size=cfg.get("batch_size", 64),
        mini_batch_size=cfg.get("mini_batch_size", 8),
        ppo_epochs=cfg.get("ppo_epochs", 4),
        learning_rate=cfg.get("learning_rate", 1e-6),
        gamma=cfg.get("gamma", 1.0),
        lam=cfg.get("lam", 0.95),
        clip_range=cfg.get("clip_range", 0.2),
        kl_penalty_cfg={
            "coef": cfg.get("kl_penalty_coef", 0.02),
        },
    )


def simple_empathy_reward_fn(texts: list[str]) -> list[float]:
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


def run_ppo_training(cfg: Dict[str, Any], reward_fn: Callable[[list[str]], list[float]] | None = None) -> None:
    """
    使用 PPO 对 SFT 后模型做一次 RL 微调。
    reward_fn：输入生成的回答列表，输出对应的 reward 列表。
    """
    torch.manual_seed(cfg.get("seed", 42))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    rl_cfg = cfg["rl"]["ppo"]

    if reward_fn is None:
        reward_fn = simple_empathy_reward_fn

    # 1. 加载 SFT 模型
    mt: ModelAndTokenizer = load_sft_model(
        sft_model_path=model_cfg["sft_model_path"],
        dtype=model_cfg.get("dtype", "bfloat16"),
        use_lora=model_cfg.get("use_lora", True),
    )

    # 转为带 value head 的模型
    model = AutoModelForCausalLMWithValueHead.from_pretrained(mt.model)

    # 2. 加载 RL 数据集
    dataset = load_rl_dataset(
        train_file=data_cfg["train_file"],
        num_proc=data_cfg.get("num_proc", 4),
    )

    # 3. PPO 配置
    ppo_config = build_ppo_config(rl_cfg)

    # 4. PPOTrainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=mt.tokenizer,
        dataset=dataset,
        dataset_text_field="user",
    )

    # 5. 训练循环（简化示例）
    for batch in ppo_trainer.dataloader:
        queries = batch["input_ids"]

        # 生成回复
        responses = ppo_trainer.generate(
            queries,
            max_new_tokens=data_cfg.get("max_response_length", 512),
        )

        # 解码文本，用 reward_fn 打分
        texts = mt.tokenizer.batch_decode(responses, skip_special_tokens=True)
        rewards = torch.tensor(reward_fn(texts), dtype=torch.float32).to(model.device)

        # PPO 更新
        stats = ppo_trainer.step(queries, responses, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

