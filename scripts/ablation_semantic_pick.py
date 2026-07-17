"""消融：跨语言语义打分对 item_picker 精挑排序的增益 + 权重标定。

**测的核心**：库是**英文标题**、用户用**中文偏好词** → 关键词天然跨语言失配（「压缩」匹配不到
「Compression」），正是 BGE-M3 语义打分该发力处。量化「纯关键词基线（跨语言失配→偏好盲排序）
vs +语义（跨语言命中）」的增益，并标定权重。

方法（真实数据 + **确定性规则标注**）：
1. 每个场景用 item_search 从真实 Qdrant 拉候选（真实英文标题）。
2. ground truth = **英文属性词是否在标题里**（确定性规则，不依赖 LLM judge——judge 标注在薄库上
   不稳，见 todo-integrate-esci-dataset；规则标注完全可复现）。偏好词只用**中文**，隔离跨语言增益。
3. 扫语义权重（0=纯关键词基线 → 逐档上调），跑 item_picker 排序，用 eval/recall_metrics 的
   Recall@K / MRR / NDCG@K 量化。
4. 对比 + 选最优权重。

诚实边界：规则 gt 只覆盖「属性写进了标题」的情形（写了 compression 才算压缩款），漏掉「是压缩款但
标题没写」的——那类要人工/ESCI 标注。故本消融量化的是**跨语言命中**这一主要增益，非全部。

用法：uv run python scripts/ablation_semantic_pick.py（需真实 Qdrant + EMBED_MODEL；judge 不需要）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import re  # noqa: E402

import app.tools.item_picker as ip  # noqa: E402
from app.eval.recall_metrics import mrr, ndcg_at_k, recall_at_k  # noqa: E402

# 场景：每个给一路偏好（param 决定走 Matcher 正软/正硬 或 Attenuator 负软）。terms **只用中文**
# （库是英文标题→关键词跨语言失配，隔离出语义增益）；gt 用英文属性正则确定性标注。
_Q = "packing cubes 旅行收纳袋 travel organizer"
# gt = 英文属性正则（在标题里=有该属性）；negate=True 时 relevant=**不**含该属性（负向场景）。
SCENARIOS = [
    {
        "name": "正软·压缩",
        "param": "prefer_keywords",
        "terms": ["压缩", "可压缩"],
        "gt": r"compress",
        "negate": False,
    },
    {
        "name": "正软·轻量",
        "param": "prefer_keywords",
        "terms": ["轻量", "超轻便"],
        "gt": r"lightweight|ultralight|ultra-light",
        "negate": False,
    },
    {
        "name": "正软·网眼透气",
        "param": "prefer_keywords",
        "terms": ["网眼", "透气"],
        "gt": r"mesh|breathable",
        "negate": False,
    },
    {
        "name": "正软·鞋袋",
        "param": "prefer_keywords",
        "terms": ["鞋袋", "鞋子收纳"],
        "gt": r"shoe\s*(bag|pouch)",
        "negate": False,
    },
    {
        "name": "正硬·压缩(must)",
        "param": "must_have",
        "terms": ["压缩"],
        "gt": r"compress",
        "negate": False,
    },
    {
        "name": "负软·避迷彩军事",
        "param": "deprioritize_keywords",
        "terms": ["迷彩", "军事风", "战术"],
        "gt": r"camo|tactical|military",
        "negate": True,
    },
]

SEM_SCALES = [0.0, 0.3, 0.5, 0.8, 1.2]  # 0=纯关键词基线；其余为语义权重档
K = 10


def _rule_relevant(candidates: list, gt: str, negate: bool) -> list[str]:
    """确定性标注：英文属性正则命中标题=有该属性；negate 时 relevant=不含该属性。"""
    pat = re.compile(gt, re.IGNORECASE)
    out = []
    for c in candidates:
        hit = bool(pat.search(f"{c.title} {c.brand}"))
        if hit != negate:  # negate=False:命中=relevant；negate=True:不命中=relevant
            out.append(c.item_id)
    return out


async def _rank(candidates: list, param: str, terms: list[str]) -> list[str]:
    """跑 item_picker，返回排序后的 item_id 列表（top_k 放大到全展示）。"""
    out = await ip.item_picker.ainvoke(
        {"candidates": [c.model_dump() for c in candidates], param: terms, "top_k": 100}
    )
    return [c.item_id for c in out.picks]


async def main() -> None:
    try:
        from app.recall.towers import get_tower_client

        if not get_tower_client().remote:
            print("跳过：EMBED_MODEL 未配置（本地 hash 编码器无语义），消融无意义。")
            return
    except Exception as e:  # noqa: BLE001
        print(f"跳过：{type(e).__name__}: {e}")
        return

    from app.tools.item_search import item_search

    # 候选池按 query 缓存（多个场景共享同一 query→同一池，只 item_search 一次，确定性）。
    pool_cache: dict[str, list] = {}
    prepared: list[dict] = []
    for sc in SCENARIOS:
        if _Q not in pool_cache:
            so = await item_search.ainvoke({"query": _Q, "platform": "all", "top_k": 20})
            pool_cache[_Q] = so.candidates
        cands = pool_cache[_Q]
        if len(cands) < 6:
            print(f"[跳过] 候选过少({len(cands)})")
            continue
        relevant = _rule_relevant(cands, sc["gt"], sc["negate"])
        frac = len(relevant) / len(cands)
        # 需 relevant 占比适中(既非全无、也非近乎全部)，否则 ranking 无区分度、指标退化。
        if not (0.1 <= frac <= 0.7):
            print(f"[跳过] 场景「{sc['name']}」标注退化(relevant={len(relevant)}/{len(cands)})")
            continue
        prepared.append({**sc, "cands": cands, "relevant": relevant})
        print(f"[就绪] {sc['name']}：候选 {len(cands)}，满足偏好 {len(relevant)}")

    if not prepared:
        print("无可用场景（标注全退化或候选不足），消融中止。")
        return

    # 扫权重（聚合）
    async def _metrics(scale: float) -> tuple[dict[str, float], list[tuple[float, float]]]:
        ip._W_MATCH_SEM = scale
        ip._W_ATTEN_SEM = scale
        ip._W_MATCH_HARD_SEM = scale
        per = []
        for p in prepared:
            ranked = await _rank(p["cands"], p["param"], p["terms"])
            per.append(
                (
                    recall_at_k(ranked, p["relevant"], K),
                    ndcg_at_k(ranked, p["relevant"], K),
                    mrr(ranked, p["relevant"]),
                )
            )
        agg = {
            "recall": sum(x[0] for x in per) / len(per),
            "ndcg": sum(x[1] for x in per) / len(per),
            "mrr": sum(x[2] for x in per) / len(per),
        }
        return agg, [(x[0], x[1]) for x in per]

    print(f"\n{'配置':<14}{'Recall@' + str(K):>12}{'NDCG@' + str(K):>12}{'MRR':>10}")
    print("-" * 48)
    base_agg, base_per = await _metrics(0.0)
    print(
        f"{'关键词基线':<14}{base_agg['recall']:>12.3f}{base_agg['ndcg']:>12.3f}{base_agg['mrr']:>10.3f}"
    )
    for scale in SEM_SCALES[1:]:
        agg, _ = await _metrics(scale)
        d = agg["ndcg"] - base_agg["ndcg"]
        print(
            f"{'+语义 ' + str(scale):<14}{agg['recall']:>12.3f}{agg['ndcg']:>12.3f}"
            f"{agg['mrr']:>10.3f}   ΔNDCG={d:+.3f}"
        )

    # 每场景明细（基线 vs W=0.5）
    _, per05 = await _metrics(0.5)
    print(f"\n{'每场景 Recall@' + str(K):<18}{'基线':>8}{'W=0.5':>8}")
    for p, b, s in zip(prepared, base_per, per05, strict=True):
        print(f"{p['name']:<18}{b[0]:>8.3f}{s[0]:>8.3f}")

    print(f"\n场景数 {len(prepared)}，K={K}。基线=纯关键词(中文词对英文标题→跨语言失配)。")


if __name__ == "__main__":
    asyncio.run(main())
