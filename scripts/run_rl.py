import argparse
from pathlib import Path

import yaml

from src.training.rl_trainer import run_ppo_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RL (PPO) training for counseling empathy model.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to RL config YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    algo = cfg.get("rl", {}).get("algo", "ppo")
    if algo != "ppo":
        raise ValueError(f"当前示例仅实现 PPO，收到 algo={algo}")

    run_ppo_training(cfg)


if __name__ == "__main__":
    main()

