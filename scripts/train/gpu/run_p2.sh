#!/bin/bash
# 第四轮：补第三轮的最后一个遗留 + 给合成数据留位置。
#
# 第三轮的 e13 证明 curriculum 能把 complement 从 .4207 压到 .3802（−9.6%），代价是主召回
# −0.56pt。但它接在 e6 上，而 e10（全展开）比 e6 更强 —— 所以「在最强底座上做课程」这一格
# 还是空的。这轮补上：如果 e14 能守住 e10 的主召回、同时吃到 e13 那样的 complement 收益，
# 那它就是最终交付版本。
set -u
cd ~/globex-train
mkdir -p logs results

export PATH=$HOME/.local/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
export HF_ENDPOINT=https://hf-mirror.com

stamp() { date "+%F %T"; }

ckpt_at() {
  local run
  run=$(find "$1" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort -V | tail -1)
  [ -z "$run" ] && return
  find "$run" -maxdepth 1 -name "checkpoint-*" -type d 2>/dev/null | sort -V | sed -n "$2p"
}

run() {   # run <标签> <数据前缀> <epochs> <lr> <temp> [初始模型]
  local TAG=$1 VER=$2 EP=$3 LR=$4 TEMP=$5 INIT=${6:-BAAI/bge-m3}
  echo "[$(stamp)] TRAIN $TAG (ver=$VER ep=$EP lr=$LR temp=$TEMP init=$INIT)"
  VER=$VER TAG=$TAG EPOCHS=$EP LR=$LR TEMP=$TEMP INIT_MODEL=$INIT \
    ./train_one.sh > logs/train_$TAG.log 2>&1
  local C
  C=$(ckpt_at output/$TAG "$EP")
  echo "[$(stamp)] TRAIN $TAG done ckpt=$C"
  [ -z "$C" ] && return
  uv run python eval_on_gpu.py --model "$C" --corpus corpus.jsonl \
    --qrels esci_eval_qrels.jsonl --out results/$TAG.json > logs/eval_$TAG.log 2>&1
  echo "[$(stamp)] EVAL $TAG rc=$?"
  uv run python eval_crosslingual.py --model "$C" --pairs xlingual_pairs.jsonl \
    --out results/xling_$TAG.json > logs/xling_$TAG.log 2>&1
  echo "[$(stamp)] XLING $TAG $(grep -o '"pass_rate": [0-9.]*' results/xling_$TAG.json 2>/dev/null)"
}

echo "[$(stamp)] === P2 开始 ==="

# E14：在最强底座 e10 上做 hard 课程（第三轮 e13 是在 e6 上做的）
CKPT_E10=$(ckpt_at output/e10_fullexp 2)
if [ -n "$CKPT_E10" ]; then
  run e14_curric3 v3 1 1e-5 0.02 "$CKPT_E10"
else
  echo "[$(stamp)] WARN: 找不到 e10 checkpoint"
fi

# E15：e10 再训一个 epoch（3ep）。首轮 v3 训 3ep 是退化的，但那是在 3.87 万条上；
#      现在数据量翻到 14.88 万，过拟合的临界点可能后移——一格便宜的消融。
run e15_ep3 v1f 3 1e-5 0.02

echo "[$(stamp)] === P2 完成 ==="
for f in results/e14*.json results/e15*.json; do echo "--- $f"; cat "$f"; echo; done
