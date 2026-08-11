#!/usr/bin/env bash
# S3 GRPO 的 **rollout server**（ms-swift 4.4.2 的 server 模式）。
#
# 为什么不用 colocate（vLLM 跟训练挤同一张卡）：4090D 只有 24G。S2 实测 4B LoRA 训练本身
# 就要 21G（batch 2、不开 gradient checkpointing），再塞一个 vLLM 引擎必 OOM。卡多是这台
# 机器的优势，rollout 独占一张卡才对得起 ROADMAP 里「六卡摆开」的分配。
#
# 独占 GPU 4。GPU 5(A100) 上跑着 M21/M22 留下的 embed / rerank 服务，GPU 6 是本轮 reward
# 检索用的 bge-m3 —— 都别动。
#
# 用法：bash run_grpo_rollout_server.sh   # 前台跑，日志直接看；训练脚本连 127.0.0.1:8010
set -euo pipefail

M23=${M23:-$HOME/m23}
VENV=${VENV:-$M23/.venv-rl}
BASE=${BASE:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
PORT=${PORT:-8010}

export CUDA_VISIBLE_DEVICES=${ROLLOUT_GPU:-4}
# vLLM 侧的显存比例：profile 实测 0.85 时 KV 够 256 条并发（一步 rollout 3.8s），
# 而 0.45 只慢 17%。这里给 0.8，留点余量防 LoRA 热更新时的峰值。
exec "$VENV/bin/swift" rollout \
    --model "$BASE" \
    --vllm_enable_lora true \
    --vllm_max_lora_rank 16 \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 2048 \
    --max_new_tokens 256 \
    --port "$PORT"
