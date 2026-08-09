#!/bin/bash
# 单次 BGE-Reranker-v2-m3 精调。跑在 GPU 机器（huzhou / A100 5 号卡），不是本机。
#
# 用法: VER=r1 TAG=r1 EPOCHS=2 LR=1e-5 GROUP=8 ./train_reranker.sh
#
# 与 M21 的 train_one.sh（embedding）的差异，逐条都是有理由的：
#
#   - task_type=reranker + loss_type=listwise_reranker。listwise 是「1 个正例 + n 个负例算一次
#     组内 softmax CE」，正是 FlagEmbedding 训 reranker 的主流做法。pointwise（BCE 独立判每对
#     相关与否）也可选，但它学不到「这一组里哪个最该排第一」——而排序正是我们要治的病。
#
#   - max_length 160，实测定的不是拍的。用 bge-reranker 的 tokenizer 量过：query 中位 8 token
#     /p95 15，doc 中位 46/p95 71/p99 79/max 131，拼接后 p99 才约 100。体检脚本里那个 320 是
#     照「字符数 p95 224」估的，白烧了 3 倍算力。160 覆盖到 p99 还有富余，截断率近 0——**只要
#     不发生截断，长度参数就不影响结果**，所以与推理侧 320 并存也不构成 train/serve skew。
#
#   - batch 单位是「组」不是「样本」。一组 = 1 正 + 7 负 = 8 个 pair（FlagEmbedding 官方
#     train_group_size 主流值），BATCH=8 即每步 64 个 pair × 160 token。M21 那边
#     batch32×128token 用了 13GB，线性外推这里约 32GB，A100-40G 装得下。OOM 就先降 BATCH。
#
#   - lr 1e-5：与 M21 实测结论一致（5e-6 两组对照都更差，-1.01/-1.47pt），不照抄 refdocs 的
#     5e-6。FlagEmbedding 官方给 reranker 的 6e-5 是 base 尺寸模型的值，568M 的 large 不能用。
#
#   - 全参 + gradient_checkpointing，理由同 M21：BGE 系列官方就是全参，LoRA 是 LLM 那边的习惯；
#     checkpointing 拿时间换显存去撑大 batch。
set -euo pipefail
cd "$(dirname "$0")"

VER=${VER:?必须指定数据前缀，如 r1}
TAG=${TAG:-r1}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}
BATCH=${BATCH:-8}
SAVE_LIMIT=${SAVE_LIMIT:-3}
INIT_MODEL=${INIT_MODEL:-BAAI/bge-reranker-v2-m3}

export PATH=$HOME/.local/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
# listwise 组内 softmax 的温度。1.0 是 swift 默认，也是 CE 的标准形态；
# embedding 那边 τ=0.02 是 InfoNCE 的经验值，两者不是一回事，别照搬。
export LISTWISE_RERANKER_TEMPERATURE=${TEMP:-1.0}

echo "[train] ver=$VER tag=$TAG epochs=$EPOCHS lr=$LR batch=$BATCH init=$INIT_MODEL"

.venv/bin/swift sft \
  --model "$INIT_MODEL" \
  --task_type reranker \
  --tuner_type full \
  --loss_type listwise_reranker \
  --dataset swift_${VER}_train.jsonl \
  --val_dataset swift_${VER}_val.jsonl \
  --output_dir output/${TAG} \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH" \
  --per_device_eval_batch_size "$BATCH" \
  --gradient_accumulation_steps 1 \
  --learning_rate "$LR" \
  --max_length 160 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_steps 50 \
  --save_total_limit "$SAVE_LIMIT" \
  --dataloader_num_workers 4
