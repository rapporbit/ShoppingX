"""生成召回评测金标集（query → 应被召回的 card_id 列表），refdocs 13-1 §5.2。

**关系是确定性构造的，不靠 LLM 拍脑袋：** 同一个品类下的 bestseller/attribute/price_range
三张卡天然互为「相关」——查这个品类就该召回它自己的卡。所以 ``relevant`` = 该品类的全部
card_id（ground truth 由构造保证）。LLM 只负责把品类名**改写成更自然的购物口语 query**
（覆盖名词类/属性类/口语类多种形态），让评测不只测「字面同名召回」。无 LLM 配置时降级为
直接用品类名，金标集照样可用——只是 query 形态单一。

用法：
    uv run python scripts/eval/build_category_golden.py --n 40
    uv run python scripts/eval/build_category_golden.py --n 40 --no-llm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv  # noqa: E402

from app.recall.category_kb import CategoryCard  # noqa: E402
from app.recall.kb_client import DEFAULT_CARDS_PATH  # noqa: E402

load_dotenv()  # LLM 改写 query 需要 .env 里的 LLM_* 配置

GOLDEN_PATH = Path("data/eval/category_recall.jsonl")

PARAPHRASE_PROMPT = (
    "你是电商搜索评测助手。给定一个商品品类名，写一句**自然的购物 query**——"
    "像真实用户会说的话，可以更口语、带点属性或场景，但仍明确指向这个品类。"
    "只输出 query 本身，不要解释、不要引号。\n品类：{category}"
)


def _load_cards() -> list[CategoryCard]:
    path = Path(DEFAULT_CARDS_PATH)
    if not path.exists():
        raise SystemExit(f"找不到卡片 {path}，请先跑 scripts/build_category_kb.py")
    return [
        CategoryCard.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def _paraphrase(category: str) -> str:
    from app.agent.llm import get_judge_llm

    resp = await get_judge_llm().ainvoke(PARAPHRASE_PROMPT.format(category=category))
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return text.strip().splitlines()[0].strip() or category


# NDCG 的 gain 按 relevant 列表顺序递减，所以这里**按重要性**排 relevant，而非字母序——
# 否则 NDCG 奖励的是「card_id 字母序靠前者排前」这种无意义的伪信号。爆款卡最该被先召回，
# 其次属性、再价格。
_TYPE_PRIORITY = {"bestseller": 0, "attribute": 1, "price_range": 2}


def _card_rank(card_id: str) -> int:
    for t, p in _TYPE_PRIORITY.items():
        if card_id.endswith(t):
            return p
    return len(_TYPE_PRIORITY)


async def main(n: int, use_llm: bool) -> None:
    cards = _load_cards()
    by_cat: dict[str, list[str]] = defaultdict(list)
    for c in cards:
        by_cat[c.category].append(c.card_id)
    # 取卡片最全（≥2 张）的品类，按字母序稳定取前 n（可复现）。
    cats = sorted(c for c, ids in by_cat.items() if len(ids) >= 2)[:n]

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cat in cats:
        query = cat
        if use_llm:
            try:
                query = await _paraphrase(cat)
            except Exception as exc:  # noqa: BLE001 —— LLM 不可用就降级用品类名
                print(f"  [降级] {cat} 改写失败（{type(exc).__name__}），用品类名")
        # relevant 按重要性排序（爆款>属性>价格），让 NDCG 的位置增益有意义。
        relevant = sorted(by_cat[cat], key=lambda cid: (_card_rank(cid), cid))
        rows.append({"query": query, "category": cat, "relevant": relevant})

    with GOLDEN_PATH.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"金标集写入 {GOLDEN_PATH}（{len(rows)} 条 query，LLM={'on' if use_llm else 'off'}）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="金标 query 条数（品类数）")
    parser.add_argument("--no-llm", action="store_true", help="不用 LLM 改写，直接用品类名")
    args = parser.parse_args()
    asyncio.run(main(args.n, use_llm=not args.no_llm))
