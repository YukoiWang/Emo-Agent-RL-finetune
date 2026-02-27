# -*- coding: utf-8 -*-
"""
Unified training monitor: JSONL file + TensorBoard + wandb.

Usage:
    monitor = TrainingMonitor(
        output_dir="outputs/grpo_emo",
        experiment_name="grpo_emo_mode1",
        use_tensorboard=True,
        use_wandb=False,          # set True + configure wandb project to enable
        wandb_project="emo-rl",
        config=cfg,               # full config dict, logged as hyper-params
    )
    monitor.log(step=10, metrics={"reward_mean": 0.35, "loss": 0.12, "kl_loss": 0.01})
    monitor.log_text(step=10, tag="sample_dialogue", text="用户：... NPC：...")
    monitor.close()
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class TrainingMonitor:
    """Lightweight, dependency-safe logger that writes to three sinks."""

    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "train",
        use_tensorboard: bool = True,
        use_wandb: bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._tb_writer = None
        self._wandb_run = None
        self._log_file = None

        if not enabled:
            return

        os.makedirs(output_dir, exist_ok=True)

        # --- JSONL ---
        log_path = os.path.join(output_dir, "training_log.jsonl")
        self._log_file = open(log_path, "w", encoding="utf-8")

        # --- TensorBoard ---
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(output_dir, "tb_logs")
                self._tb_writer = SummaryWriter(log_dir=tb_dir)
                if config:
                    flat = _flatten_dict(config)
                    self._tb_writer.add_text(
                        "hyperparameters",
                        _dict_to_markdown_table(flat),
                        global_step=0,
                    )
                print(f"[Monitor] TensorBoard logs → {tb_dir}")
                print(f"[Monitor]   启动: tensorboard --logdir {tb_dir}")
            except ImportError:
                print("[Monitor] tensorboard not installed, skipping TensorBoard logging.")

        # --- wandb ---
        if use_wandb:
            try:
                import wandb
                if wandb.run is None:
                    wandb.init(
                        project=wandb_project or "emo-rl",
                        entity=wandb_entity,
                        name=wandb_run_name or experiment_name,
                        config=config or {},
                        dir=output_dir,
                        reinit=True,
                    )
                self._wandb_run = wandb.run
                print(f"[Monitor] wandb run → {wandb.run.url}")
            except ImportError:
                print("[Monitor] wandb not installed, skipping wandb logging.")
            except Exception as e:
                print(f"[Monitor] wandb init failed: {e}")

    # ------------------------------------------------------------------
    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        # JSONL
        if self._log_file is not None:
            record = {"step": step, **metrics}
            self._log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._log_file.flush()

        # TensorBoard
        if self._tb_writer is not None:
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    self._tb_writer.add_scalar(key, val, global_step=step)
            self._tb_writer.flush()

        # wandb
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log({**metrics, "step": step}, step=step)
            except Exception:
                pass

    def log_text(self, step: int, tag: str, text: str) -> None:
        if not self.enabled:
            return
        if self._tb_writer is not None:
            self._tb_writer.add_text(tag, text, global_step=step)
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)
            except Exception:
                pass

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None
        if self._wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
            self._wandb_run = None

    def __del__(self):
        self.close()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _dict_to_markdown_table(d: Dict[str, Any]) -> str:
    lines = ["| Key | Value |", "|---|---|"]
    for k, v in sorted(d.items()):
        lines.append(f"| `{k}` | `{v}` |")
    return "\n".join(lines)
