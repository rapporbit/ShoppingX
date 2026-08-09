#!/bin/bash
# 单次 BGE-M3 对比精调。跑在 GPU 机器（huzhou / A100），不是本机。
#
# 用法: VER=v1x TAG=e5 EPOCHS=2 LR=5e-6 TEMP=0.05 ./train_one.sh
#
# 首轮（2026-08-08 凌晨）踩出来的配置，保留不动的部分：
#   - 全参微调 --tuner_type full。ms-swift 默认 LoRA，但 BGE 系列官方（FlagEmbedding）就是全参，
#     568M 在 A100 上全参毫无压力，LoRA 是 LLM 那边的习惯。
#   - grad_accum 恒为 1。梯度累积对 in-batch negatives 完全无效——每个 micro-batch 独立算 loss，
#     累积只省显存不增负例，而 batch 就是对比学习的负例数量。想要大 batch 只能靠 GradCache。
#   - 开 gradient_checkpointing。曾以为显存够就该关掉，错了：BGE 官方默认开它，正是拿时间换显存
#     去撑大 batch。关掉后 batch 24 就 OOM，开着 batch 32 只用 13GB、GPU 97%、1.5 it/s。
#   - max_length 128。文本 p95 才 223 字符（约 56 token），512 是照抄来的浪费。
#
# 本轮（P0）改成可调的三个，都是对着 refdocs 04-2 调的：
#   - TEMP：首轮全局钉 0.02（BGE 官方值，但那是**跨语言对**的值）。§4.2 给的是动态温度——同语言
#     对该用 0.05。ESCI 的 query 与标题基本同语言，0.02 可能过锐，是首轮 hard 越多掉得越狠的
#     头号嫌疑。
#   - LR：首轮 1e-5，§5.2/§5.4 给 Stage2 SFT 的是 5e-6（防灾难遗忘）。我们用了官方建议的两倍，
#     而实测 2 epoch 就退化，两者大概率相关。
#   - SAVE_LIMIT：首轮 =1，撞上 best-checkpoint 逻辑——v1 因 epoch1 的 eval_loss 更低，epoch2 的
#     checkpoint 被自动删了，导致三组对照的 epoch 数不齐（v1 是 1ep，v2/v3 是 2ep）。默认改 3，
#     每个 epoch 都留下，评测时显式指定，不让框架替我们做选择。
set -e

VER=${VER:?必须指定数据前缀，如 v1x}
TAG=${TAG:-e}
EPOCHS=${EPOCHS:-2}
LR=${LR:-5e-6}
TEMP=${TEMP:-0.05}
BATCH=${BATCH:-32}
SAVE_LIMIT=${SAVE_LIMIT:-3}
INIT_MODEL=${INIT_MODEL:-BAAI/bge-m3}   # 换成本地 checkpoint 即为 curriculum 续训

export PATH=$HOME/.local/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
export INFONCE_TEMPERATURE=$TEMP
export INFONCE_USE_BATCH=True

echo "[train] ver=$VER tag=$TAG epochs=$EPOCHS lr=$LR temp=$TEMP batch=$BATCH init=$INIT_MODEL"

uv run swift sft \
  --model "$INIT_MODEL" \
  --task_type embedding \
  --tuner_type full \
  --loss_type infonce \
  --dataset swift_${VER}_train.jsonl \
  --val_dataset swift_${VER}_val.jsonl \
  --output_dir output/${TAG} \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH" \
  --gradient_accumulation_steps 1 \
  --learning_rate "$LR" \
  --max_length 128 \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --warmup_ratio 0.05 \
  --save_strategy epoch \
  --eval_strategy epoch \
  --logging_steps 50 \
  --save_total_limit "$SAVE_LIMIT" \
  --dataloader_num_workers 4
