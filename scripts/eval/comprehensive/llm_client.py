# -*- coding: utf-8 -*-
"""Thin wrapper around the DeepSeek / OpenAI-compatible chat API."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import requests


class DeepSeekClient:
    """Call a DeepSeek-compatible chat-completions endpoint."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
    ) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=self.timeout,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"].strip()
                raise RuntimeError(
                    f"API error ({resp.status_code}): {resp.text[:300]}"
                )
            except Exception:
                if attempt + 1 >= self.max_retries:
                    raise
                time.sleep(2 ** attempt)
        return ""

    def evaluate(self, prompt: str, temperature: float = 0.1) -> str:
        return self.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    def as_llm_fn(self):
        """Return a ``Callable[[List[Dict]], str]`` compatible with planner / simulator."""
        def fn(messages: List[Dict[str, str]]) -> str:
            return self.chat(messages, temperature=self.temperature)
        return fn

    # ── helpers for structured outputs ──────────────────────────────

    @staticmethod
    def parse_json_block(text: str) -> Any:
        """Extract the first JSON object / array from *text*."""
        text = text.strip()
        for pattern in [r"```json\s*([\s\S]*?)```", r"```([\s\S]*?)```"]:
            m = re.search(pattern, text)
            if m:
                text = m.group(1).strip()
                break
        return json.loads(text)

    @staticmethod
    def parse_score(text: str, key: str = "score") -> float:
        """Extract a numeric score from LLM output."""
        text = (text or "").replace("：", ":")
        pattern = rf"{key}\s*[:=]\s*(-?\d+(?:\.\d+)?)"
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return float(m.group(1))
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return float(nums[0]) if nums else 0.0
