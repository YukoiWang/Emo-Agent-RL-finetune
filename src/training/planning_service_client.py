# -*- coding: utf-8 -*-
"""
Planning 服务 API 客户端：通过 HTTP 调用独立 planning 服务，
接口与 build_local_planning_llm_fn 兼容，供 PlayerSimulatorWithPlanning 使用。
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import requests


def build_planning_service_llm_fn(
    base_url: str,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> Callable[[List[Dict[str, str]]], str]:
    """
    从 planning 服务 URL 构建 planning 用的 LLM 调用函数。
    输入: messages [{"role":"user","content":"..."}]
    输出: 生成的文本（用于 planning_reply 解析）

    base_url: 服务地址，如 "http://localhost:8765" 或 "http://192.168.1.10:8765"
    """
    url = base_url.rstrip("/") + "/generate"

    def fn(messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        payload = {
            "messages": [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages],
            "max_new_tokens": 1024,
            "do_sample": True,
            "temperature": 0.5,
        }
        for attempt in range(max_retries):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()
                return data.get("text", "")
            except requests.RequestException as e:
                if attempt + 1 >= max_retries:
                    raise RuntimeError(f"Planning service request failed: {e}") from e
                time.sleep(2 ** attempt)
        return ""

    return fn
