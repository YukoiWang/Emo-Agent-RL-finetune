# -*- coding: utf-8 -*-
"""
独立 planning 服务：加载一份 SFT 模型，对外提供 HTTP 推理接口。
所有 rollout worker 通过 call_api 访问，不占用训练 GPU，延迟和隐私优于外部 API。

用法:
  python scripts/planning/run_planning_service.py \
    --model_path Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8765
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List


class ChatMessage(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.5


class GenerateResponse(BaseModel):
    text: str


def create_app(model_path: str, device: str = "cuda", dtype: str = "bfloat16"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    app = FastAPI(title="Emo Planning Service", version="0.1.0")

    _dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(dtype, torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_dtype,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()

    @app.post("/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        if not req.messages:
            return GenerateResponse(text="")
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        content = messages[-1].get("content", "") if messages else ""
        if not content:
            return GenerateResponse(text="")
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                temperature=req.temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        resp_text = tokenizer.decode(gen, skip_special_tokens=True).strip()
        return GenerateResponse(text=resp_text)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main():
    parser = argparse.ArgumentParser(description="Planning 情感分析 HTTP 服务")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    app = create_app(args.model_path, device=args.device, dtype=args.dtype)
    print(f"[Planning Service] 启动中: model={args.model_path} host={args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
