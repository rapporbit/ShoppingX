#!/bin/bash
# 端到端 A/B：微调后的 embedding 到底能不能在真实 Agent 链路上兑现离线的 +12.6%。
#
# 这是整个 M21 最该回答、也一直没回答的问题。离线 ESCI 涨了 12.6%，但 ESCI 是 3 词英文关键词
# 搜索，线上是"便宜又抗造的旅行三件套，预算 300"这种自然语言意图，中间还隔着 planner 拆解、
# reranker 精排、picker 精挑。refdocs 13-1 §6.4 给的经验比例是「离线涨 5% → 线上涨 1-2%」，
# 叠加分布偏移后完全可能归零。不测就永远只能说"我不知道值多少"。
#
# 两组唯一的差异是**向量空间**：
#   A 基线 = 原版 BGE-M3 编的商品向量 + 官方 embedding API 编的 query
#   B 实验 = e15_ep3 编的商品向量 + GPU 上 embed_server 暴露的同一个 e15 编的 query
# query 与商品必须同模型编码，否则不在一个空间里——召回不报错，只是静默变垃圾。
#
# 前置：① GPU 上 embed_server 已指向 e15 并做了 SSH 端口转发到本地 8090
#       ② globex_items_e15 已灌好（load_vectors_to_qdrant.py）
#
# 用法：bash scripts/train/run_ab_rubric.sh
set -u
cd "$(dirname "$0")/../.."

OUT=data/eval
mkdir -p "$OUT"
STAMP=$(date +%m%d_%H%M)

# LLM_REQUEST_TIMEOUT 必须放大：Agent 单条要跑 5-9 轮 LLM，默认超时会把慢的那几条判死
export LLM_REQUEST_TIMEOUT=300

echo "=== A 组：基线（原版 BGE-M3 向量） ==="
QDRANT_COLLECTION=globex_items \
  uv run python scripts/eval/run_rubric.py --concurrency 3 2>&1 | tail -25
cp "$OUT/rubric_report.json" "$OUT/rubric_A_base_$STAMP.json"
echo "→ 已存 $OUT/rubric_A_base_$STAMP.json"

echo
echo "=== B 组：e15_ep3（微调向量 + 本地 embed_server） ==="
QDRANT_COLLECTION=globex_items_e15 \
EMBED_BASE_URL=http://localhost:8090/v1 \
  uv run python scripts/eval/run_rubric.py --concurrency 3 2>&1 | tail -25
cp "$OUT/rubric_report.json" "$OUT/rubric_B_e15_$STAMP.json"
echo "→ 已存 $OUT/rubric_B_e15_$STAMP.json"

echo
echo "=== 对照 ==="
uv run python - "$OUT/rubric_A_base_$STAMP.json" "$OUT/rubric_B_e15_$STAMP.json" <<'PY'
import json, sys

a, b = (json.load(open(p, encoding="utf-8")) for p in sys.argv[1:3])
ma = {r["id"]: r for r in a}
mb = {r["id"]: r for r in b}
ids = [i for i in ma if i in mb]


def score(r):
    # 分数在 result.total（顶层没有 total，别被 r["ok"] 误导——那只是跑没跑通）
    return float((r.get("result") or {}).get("total") or 0)


def p0fail(r):
    return len((r.get("result") or {}).get("p0_failures") or [])


sa = sum(score(ma[i]) for i in ids) / max(1, len(ids))
sb = sum(score(mb[i]) for i in ids) / max(1, len(ids))
print(f"共同 query {len(ids)} 条")
print(f"A 基线均分 {sa:.2f}   B e15 均分 {sb:.2f}   差 {sb - sa:+.2f}")
print(f"P0 红线失败：A {sum(p0fail(ma[i]) > 0 for i in ids)} 条   "
      f"B {sum(p0fail(mb[i]) > 0 for i in ids)} 条")

moved = sorted(ids, key=lambda i: score(mb[i]) - score(ma[i]))
print("\n跌得最多的 3 条：")
for i in moved[:3]:
    print(f"  {i:<34} {score(ma[i]):5.1f} → {score(mb[i]):5.1f}")
print("涨得最多的 3 条：")
for i in moved[::-1][:3]:
    print(f"  {i:<34} {score(ma[i]):5.1f} → {score(mb[i]):5.1f}")
print("\n注意：judge 有单样本翻转的已知噪声，单条大幅变动先复跑再归因，别急着解释。")
PY
