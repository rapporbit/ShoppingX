#!/bin/bash
# 端到端 A/B：自训 reranker 能不能在真实 Agent 链路上兑现离线的那 +19.5% ndcg@8。
#
# 姊妹脚本 run_ab_rubric.sh 问的是同一个问题（M21 那次答案是「离线 +12.6% → Rubric +7.66」），
# 这次换成精排腿。**不测就是空账**——离线尺子有 92% 的头部商品 ESCI 根本没标过，
# 那个 +19.5% 里有多少是真的、有多少是尺子噪声，只有端到端能回答。
#
# 两组唯一差异是 reranker 权重：
#   A 基线 = siliconflow 上的原版 BAAI/bge-reranker-v2-m3（线上现状）
#   B 实验 = GPU 机上 rerank_server.py 暴露的自训 checkpoint（SSH 转发到本地 8091）
# 两组的 embedding 侧完全一致（都用 e15 + globex_items_e15），否则变量就不止一个了。
#
# **一个已知的口径妥协，必须写在这**：线上 item_picker 传给 reranker 的 query 是 planner 判的
# 粗品类词，而模型是在完整意图句上训的——train/serve 形态不一致。离线实测同分母下意图句
# +3.10pt、品类词只有 +0.59pt，所以这一版 A/B 测的是「形态不改、只换权重」的收益，
# 是**保守下界**。改 query 形态要动 item_picker 的品类门/域反证那套逻辑，单独一步做。
#
# 前置：① GPU 上 rerank_server.py 已起并 ssh -N -L 8091:127.0.0.1:8091 huzhouet
#       ② embed_server 8090 隧道通着（两组都要用 e15）
#
# 用法：bash scripts/train/run_ab_reranker.sh
set -u
cd "$(dirname "$0")/../.."

OUT=data/eval
mkdir -p "$OUT"
STAMP=$(date +%m%d_%H%M)

# Agent 单条要跑 5-9 轮 LLM，默认超时会把慢的那几条判死（M21 踩过）
export LLM_REQUEST_TIMEOUT=300
export QDRANT_COLLECTION=globex_items_e15
export EMBED_BASE_URL=http://127.0.0.1:8090/v1

if ! curl -s -m 5 http://127.0.0.1:8091/health > /dev/null; then
  echo "8091 不通——先起 rerank_server 并做 SSH 端口转发"; exit 1
fi

echo "=== A 组：线上现状（siliconflow 原版 v2-m3） ==="
uv run python scripts/eval/run_rubric.py --concurrency 3 2>&1 | tail -25
cp "$OUT/rubric_report.json" "$OUT/rubric_A_baseCE_$STAMP.json"

echo
echo "=== B 组：自训 reranker（本地 8091） ==="
RERANKER_ENDPOINT=http://127.0.0.1:8091/v1/rerank \
RERANKER_MODEL=globex-reranker \
  uv run python scripts/eval/run_rubric.py --concurrency 3 2>&1 | tail -25
cp "$OUT/rubric_report.json" "$OUT/rubric_B_tuned_$STAMP.json"

echo
echo "=== 对照 ==="
uv run python - "$OUT/rubric_A_baseCE_$STAMP.json" "$OUT/rubric_B_tuned_$STAMP.json" <<'PY'
import json, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])


def by_query(rep):
    return {r["query_id"]: r for r in rep.get("results", []) if r.get("score") is not None}


qa, qb = by_query(a), by_query(b)
both = sorted(set(qa) & set(qb))
print(f"A 均分 {a.get('avg_score')}  B 均分 {b.get('avg_score')}  共同跑通 {len(both)} 条")
if both:
    da = sum(qa[q]["score"] for q in both) / len(both)
    db = sum(qb[q]["score"] for q in both) / len(both)
    print(f"共同子集：A {da:.2f} → B {db:.2f}（{db - da:+.2f}）")
    worse = [(qb[q]["score"] - qa[q]["score"], q) for q in both]
    worse.sort()
    print("跌得最多的 3 条（先看这些，别只看均分）：")
    for d, q in worse[:3]:
        print(f"  {d:+.0f}  {q}  {qa[q].get('query', '')[:50]}")
PY
