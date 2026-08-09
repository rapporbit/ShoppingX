#!/bin/bash
# M22 的 2×2 析因队列：把「分级 label」和「ApproxNDCG loss」两个因素分开量。
#
#              CE loss              ApproxNDCG
#   二值 label  r1（已跑，ms-swift）  r4
#   分级 label  r3（ListNet）         r2 ← refdocs §10.3 Stage C
#
# 只跑 r2 的话，涨跌都归因不了。三组串行，每组训完立刻用同一份评测集打分，
# 分数落盘等本地跑 eval_rerank.py——GPU 只出分、口径在本地调，同第 1 步的分工。
set -uo pipefail
cd "$(dirname "$0")"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=${GPU:-5}
PY=.venv/bin/python

run() {
  local tag=$1 loss=$2 graded=$3
  echo "=== [$tag] loss=$loss graded=${graded:-no} $(date +%H:%M) ==="
  $PY train_reranker_graded.py \
    --train graded_r2_train.jsonl --val graded_r2_val.jsonl \
    --out output/$tag --loss $loss $graded || { echo "[$tag] 训练失败"; return 1; }
  $PY rerank_candidates.py \
    --input rerank_candidates_embed.jsonl \
    --output rerank_scores_$tag.jsonl \
    --no-category --model output/$tag/epoch-1 || { echo "[$tag] 打分失败"; return 1; }
  echo "=== [$tag] 完成 $(date +%H:%M) ==="
}

run r2 approxndcg --graded
run r3 ce --graded
run r4 approxndcg ""
echo "全部完成 $(date +%H:%M)"
