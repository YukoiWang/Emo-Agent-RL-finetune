# 训练与 RL (training)

- **训练入口**：`sft_trainer`、`dpo_trainer`、`dpo_emo_trainer`、`ppo_emo_trainer`、`grpo_training`
- **PPO 核心**：`ppo_training`（Memory/Critic/Actor）、`ppo_emo_rollout`、`reward_emo`
- **DPO rollout**：`dpo_emo_rollout`
- **用户/环境模拟**：`hard_player_simulator_dsv3`、`qwen_user_simulator`、`emo_analyzer`、`emo_planning`、`local_planning_llm`
- **工具**：`rl_trainer`（仅含 `simple_empathy_reward_fn`，名字易误导）

详见 **docs/src_overview.md**。
