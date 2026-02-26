# Planning 情感分析服务

独立 HTTP 服务，只加载一份 planning 模型，供所有 rollout worker 通过 API 调用。不占用训练 GPU，延迟和隐私优于外部 API。

## 1. 启动服务

```bash
# 单机启动（默认 localhost:8765）
python scripts/planning/run_planning_service.py \
  --model_path Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8765

# 集群内多节点：将 --host 设为 0.0.0.0，训练节点用 <服务节点IP>:8765 访问
```

依赖：`pip install fastapi uvicorn`（已在 requirements.txt）

## 2. 训练配置

在 YAML 中设置 `rollout.planning_service_url`，优先于 `sft_model_path`：

```yaml
rollout:
  planning_service_url: "http://localhost:8765"   # 本机
  # planning_service_url: "http://192.168.1.10:8765"  # 集群内服务节点
  sft_model_path: null  # 使用服务时可不加载本地模型
```

## 3. 健康检查

```bash
curl http://localhost:8765/health
```
