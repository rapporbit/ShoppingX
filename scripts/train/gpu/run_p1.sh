#!/bin/bash
# 第三轮实验编排。跑在 GPU 机器，nohup 后台跑一夜。
#
# 第二轮结论（见 docs/interview/M21-检索侧微调与训练基建.md §5.4）：
#   ✅ 正例展开是唯一正收益：+1.25pt（e6 .4864 vs e9 .4739），最优配方 R@100 相对原版 +9.7%
#   ❌ 温度/lr 假说被证伪：τ0.05+lr5e-6 在两组独立对照里都更差（−1.58 / −2.53pt）
#   ⚠️ curriculum 没验出效果，但它接在"差配方"e5 后面，结论不作数
#
# 本轮三件事，全部是对第二轮遗留问题的直接回应：
#   1. 把「展开」推到头——第二轮只展开到 max-pos 3（8.6 万条），全展开是 14.88 万条
#   2. 拆开 τ 与 lr——第二轮把两个变量捆在一组改了，只知道组合更差，不知道是谁的锅
#   3. 基于 e6（真·最优配方）重做 curriculum，给首轮那个"hard negative 排序反常"的谜一个交代
set -u
cd ~/globex-train
mkdir -p logs results

export PATH=$HOME/.local/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-5}
export HF_ENDPOINT=https://hf-mirror.com   # 缺了会静默超时，表现为「在跑但 GPU 0%」

stamp() { date "+%F %T"; }

xling() {
  echo "[$(stamp)] XLING $2 <- $1"
  uv run python eval_crosslingual.py --model "$1" --pairs xlingual_pairs.jsonl \
    --out results/xling_$2.json > logs/xling_$2.log 2>&1
  echo "[$(stamp)] XLING $2 rc=$? $(grep -o '"pass_rate": [0-9.]*' results/xling_$2.json 2>/dev/null)"
}

evaluate() {
  echo "[$(stamp)] EVAL $2 <- $1"
  uv run python eval_on_gpu.py --model "$1" --corpus corpus.jsonl \
    --qrels esci_eval_qrels.jsonl --out results/$2.json > logs/eval_$2.log 2>&1
  echo "[$(stamp)] EVAL $2 rc=$?"
}

ckpt_at() {  # 先锁定最近一次 run 的子目录，否则重跑时新旧 checkpoint 会混排、静默串台
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
  [ -n "$C" ] && { evaluate "$C" "$TAG"; xling "$C" "$TAG"; }
}

echo "[$(stamp)] === P1 开始 ==="

# E10：正例全展开（14.88 万条，×3.94）。其余配置完全照搬第二轮最优 e6，单变量。
run e10_fullexp v1f 2 1e-5 0.02

# E13：基于 e6（真·最优）续训 hard 占比 71.2% 的数据 1 epoch。
#      第二轮的 e8 接在 e5 后面，起点就是歪的；这次才是 refdocs §5.2 课程学习的公平检验。
CKPT_E6=$(ckpt_at output/e6_expand 2)
if [ -n "$CKPT_E6" ]; then
  run e13_curric2 v3 1 1e-5 0.02 "$CKPT_E6"
else
  echo "[$(stamp)] WARN: 找不到 e6 checkpoint，跳过 E13"
fi

# E11 / E12：把第二轮捆在一起的两个变量拆开，各自单独对着 e6 变一个。
#            这两组纯粹是解释性的——不指望涨分，是要知道 −2.53pt 到底该记在谁头上。
run e11_temp v1x 2 1e-5 0.05    # 只改温度
run e12_lr   v1x 2 5e-6 0.02    # 只改学习率

echo "[$(stamp)] === 全部完成 ==="
for f in results/e1*.json results/xling_e1*.json; do echo "--- $f"; cat "$f"; echo; done
