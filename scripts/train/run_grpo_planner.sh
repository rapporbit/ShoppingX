#!/usr/bin/env bash
# S3：planner 的 GRPO 训练（ms-swift 4.4.2，4 卡 DDP LoRA + 独立 rollout server）。
#
# **从 SFT 的 LoRA 接着训，不是从基座重来**：S2 把格式正确率从 63% 拉到 100%，那是切 RL 的
# 前置条件（refdocs 08-2 §6.1）。丢掉它从基座起步，前几百步会全花在「学会吐 JSON」上，
# 而那件事监督学习几十分钟就能便宜地办完。
#
# **超参按 ROADMAP 定的初值**：group 8 / clip 0.2 / kl 0.03 / lr 1e-6。lr 比 SFT 小两个量级
# 是 RL 的常识性设定——policy 每步只该挪一点点，挪大了 KL 直接炸，格式率会先崩给你看。
#
# 前置：先起 rollout server（run_grpo_rollout_server.sh）、embed server(:8095)、Qdrant(:6333)。
# 用法：bash run_grpo_planner.sh [--max_steps 5 ...]   # 多余参数原样透传给 swift
set -euo pipefail

M23=${M23:-$HOME/m23}
VENV=${VENV:-$M23/.venv-rl}
BASE=${BASE:-$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554}
SFT_ADAPTER=${SFT_ADAPTER:-$M23/output/sft_r16/v2-20260812-010320/checkpoint-306}
OUT=${OUT:-$M23/output/grpo_r1}
PORT=${PORT:-8010}

export GLOBEX_REPO=${GLOBEX_REPO:-$M23/repo}          # reward 从这份仓库按文件加载
export ROLLOUT_EMBED_URL=${ROLLOUT_EMBED_URL:-http://127.0.0.1:8095/v1/embeddings}
export QDRANT_URL=${QDRANT_URL:-http://127.0.0.1:6333}
export QDRANT_COLLECTION=${QDRANT_COLLECTION:-globex_items}
export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1,2,3}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export PLANNER_REWARD_LOG_EVERY=${PLANNER_REWARD_LOG_EVERY:-5}

# generation_batch_size = per_device × world × grad_accum = 2×4×4 = 32 = 4 prompt × group 8。
# 组必须完整落在同一步里，否则组内优势算的是残缺的组。
"$VENV/bin/swift" rlhf \
    --rlhf_type grpo \
    --model "$BASE" \
    --adapters "$SFT_ADAPTER" \
    --tuner_type lora \
    --lora_rank 16 --lora_alpha 32 \
    --dataset "$M23/planner_grpo_train.jsonl" \
    --val_dataset "$M23/planner_grpo_dev.jsonl" \
    --external_plugins "$M23/grpo_planner_plugin.py" \
    --reward_funcs planner_reward \
    --num_generations 8 \
    --temperature 1.0 \
    --top_p 0.95 \
    --use_vllm true --vllm_mode server \
    --vllm_server_host 127.0.0.1 --vllm_server_port "$PORT" \
    --max_completion_length 256 \
    --max_length 1024 \
    --beta 0.03 \
    --epsilon 0.2 \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --gradient_checkpointing true \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --save_steps 50 --save_total_limit 4 \
    --eval_steps 50 \
    --output_dir "$OUT" \
    --report_to tensorboard \
    "$@"
