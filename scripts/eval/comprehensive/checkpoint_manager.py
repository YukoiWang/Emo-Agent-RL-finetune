# -*- coding: utf-8 -*-
"""Discover and load model checkpoints."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    _HAS_PEFT = True
except ImportError:
    PeftModel = None
    _HAS_PEFT = False


def discover_checkpoints(
    output_dir: str | Path,
    interval: int = 50,
) -> List[Dict[str, Any]]:
    """
    Scan *output_dir* for ``checkpoint-{step}/`` and ``final/``.

    Returns a sorted list of dicts: ``[{"step": int, "path": str}, ...]``
    The ``final/`` checkpoint is placed last with ``step = -1``.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output dir not found: {output_dir}")

    ckpts: List[Dict[str, Any]] = []
    pattern = re.compile(r"^checkpoint-(\d+)$")

    for child in sorted(output_dir.iterdir()):
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            step = int(m.group(1))
            ckpts.append({"step": step, "path": str(child)})
        elif child.name == "final":
            ckpts.append({"step": -1, "path": str(child)})

    ckpts.sort(key=lambda c: (c["step"] == -1, c["step"]))

    if interval > 0:
        kept = []
        for c in ckpts:
            if c["step"] == -1 or c["step"] % interval == 0:
                kept.append(c)
        ckpts = kept

    return ckpts


def get_final_checkpoint(output_dir: str | Path) -> Optional[str]:
    """Return the path of ``final/`` if it exists, else the last checkpoint."""
    output_dir = Path(output_dir)
    final = output_dir / "final"
    if final.exists():
        return str(final)
    ckpts = discover_checkpoints(output_dir, interval=0)
    if ckpts:
        return ckpts[-1]["path"]
    return None


def load_model_and_tokenizer(
    checkpoint_path: str,
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: str = "cuda",
) -> Tuple[Any, Any]:
    """Load a causal-LM checkpoint (supports full weights or LoRA adapter)."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    adapter_cfg = path / "adapter_config.json"
    if adapter_cfg.exists() and _HAS_PEFT:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(path))
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=dtype, trust_remote_code=True,
        )

    model.to(device).eval()

    tok_path = str(path) if (path / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def unload_model(model: Any) -> None:
    """Free GPU memory."""
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
