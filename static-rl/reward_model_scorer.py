# -*- coding: utf-8 -*-
"""
Reward Model 推理封装：将训练好的 RM 转为 reward_fn(prompts, responses) -> scores。
供 PPO/GRPO 使用。
"""
from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardModelScorer:
    """用训练好的 Reward Model 对 (prompt, response) 打分。"""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.model.eval()

    def _format_prompt_response(self, prompt: str, response: str) -> str:
        """拼接 prompt + assistant 回复（与训练时一致）"""
        prompt = prompt.rstrip()
        if not prompt.endswith("\n"):
            prompt += "\n"
        return prompt + response

    @torch.no_grad()
    def score(self, prompts: list[str], responses: list[str]) -> list[float]:
        """对多组 (prompt, response) 打分，返回标量列表"""
        if not prompts or not responses:
            return []
        assert len(prompts) == len(responses)
        texts = [
            self._format_prompt_response(p, r)
            for p, r in zip(prompts, responses)
        ]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.model.device)
        out = self.model(**enc)
        scores = out.logits.squeeze(-1).float().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores]
        return scores

    def as_reward_fn(self):
        """返回符合 reward_fn(texts) 接口的闭包；需配合 set_prompts 使用"""
        self._current_prompts: list[str] = []

        def set_prompts(prompts: list[str]):
            self._current_prompts = list(prompts)

        def fn(responses: list[str]) -> list[float]:
            if not self._current_prompts or len(self._current_prompts) != len(responses):
                return [0.0] * len(responses)  # fallback
            return self.score(self._current_prompts, responses)

        fn.set_prompts = set_prompts
        return fn


def build_reward_fn_from_model(model_path: str):
    """构建带 set_prompts 的 reward_fn，供 PPO 循环调用。"""
    scorer = RewardModelScorer(model_path)
    fn = scorer.as_reward_fn()
    return fn, scorer
