# -*- coding: utf-8 -*-
"""
情绪分析器：用 emo_classifier_lora 微调的情感分类模型分析「用户回复」的情感（积极/消极/中性），
并结合「NPC 回复与隐藏主题的贴合程度」计算 change_value，供 PlayerSimulator 更新 emo_point。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

import torch

# 积极/消极/中性 -> 对 emo_point 的影响：积极 +，消极 -，中性略偏主题贴合
LABEL_MAP = {"消极": 0, "中性": 1, "积极": 2}
ID_TO_LABEL = {0: "消极", 1: "中性", 2: "积极"}


def build_emo_analyzer(
    adapter_path: str,
    base_model_name: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    max_length: int = 256,
    theme_fit_weight: float = 0.5,
    sentiment_weight: float = 1.0,
) -> "EmoAnalyzer":
    """
    构建基于 emo_classifier_lora 的 EmoAnalyzer 实例。
    adapter_path: 例如 /home/yukiwang/xlwy/emo_classifier_lora/checkpoint-11025
    base_model_name: 底座模型，若 None 则从 adapter 的 adapter_config.json 读取。
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    if base_model_name is None:
        import json
        cfg_path = os.path.join(adapter_path, "adapter_config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_model_name = cfg.get("base_model_name_or_path") or "/home/yukiwang/models/Qwen2-7B-Instruct"
        else:
            base_model_name = "/home/yukiwang/models/Qwen2-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=len(LABEL_MAP),
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return EmoAnalyzer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        theme_fit_weight=theme_fit_weight,
        sentiment_weight=sentiment_weight,
    )


class EmoAnalyzer:
    """
    输入：npc_reply, user_reply, hidden_theme。
    用情感分类模型对 user_reply 打分（积极/消极/中性），
    再结合 theme_fit（NPC 与主题贴合度，可用规则或外部 API）计算 change_value。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: Union[str, torch.device],
        max_length: int = 256,
        theme_fit_weight: float = 0.5,
        sentiment_weight: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.theme_fit_weight = theme_fit_weight
        self.sentiment_weight = sentiment_weight

    def _predict_sentiment(self, texts: List[str]) -> List[int]:
        """返回每条文本的预测标签 id：0 消极，1 中性，2 积极。"""
        inp = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            logits = self.model(**inp).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        return preds

    def _theme_fit_rule(self, npc_reply: str, hidden_theme: str) -> float:
        """规则：NPC 回复与隐藏主题的贴合度 [0, 1]。可被 LLM API 替代。"""
        if not npc_reply.strip() or not hidden_theme:
            return 0.5
        theme_lower = hidden_theme.lower().strip()
        reply_lower = npc_reply.lower().strip()
        tw = set(theme_lower.replace("，", " ").replace("。", " ").replace("*", " ").split())
        rw = set(reply_lower.replace("，", " ").replace("。", " ").split())
        overlap = len(tw & rw) / max(len(tw), 1)
        return min(1.0, overlap * 2)

    def __call__(
        self,
        npc_reply: str,
        user_reply: str,
        hidden_theme: str,
    ) -> Dict[str, Any]:
        """
        返回 dict：change_value (float), sentiment (int 0/1/2), theme_fit (float [0,1])。
        change_value 规则示例：积极 +主题贴合高 -> 正；消极或主题偏离 -> 负。
        """
        theme_fit = self._theme_fit_rule(npc_reply, hidden_theme)
        preds = self._predict_sentiment([user_reply])
        sentiment = preds[0]  # 0 消极 1 中性 2 积极

        # 基础变化：积极 +，消极 -，中性略看 theme_fit
        if sentiment == 2:  # 积极
            base = 3.0 + 2.0 * theme_fit
        elif sentiment == 0:  # 消极
            base = -4.0 - 2.0 * (1.0 - theme_fit)
        else:  # 中性
            base = (theme_fit - 0.5) * 4.0  # 贴合则略正，偏离则略负

        change_value = (base * self.sentiment_weight) * 0.5
        change_value += (theme_fit - 0.5) * 2.0 * self.theme_fit_weight
        change_value = max(-10.0, min(10.0, change_value))  # 单轮变化不宜过大

        return {
            "change_value": change_value,
            "sentiment": sentiment,
            "theme_fit": theme_fit,
        }


def emo_analyzer_fn_from_analyzer(analyzer: EmoAnalyzer):
    """返回可供 PlayerSimulator 使用的 emo_analyzer_fn (npc_reply, user_reply, hidden_theme) -> dict。"""
    return lambda npc_reply, user_reply, hidden_theme: analyzer(
        npc_reply, user_reply, hidden_theme or ""
    )
