"""ETL：用 LLM 为每个品类生成别名/口语说法（中英混合），供检索第一段「品类定位」用。

手写 ``CATEGORY_ALIASES`` 只有十几条，扩展性为零；真实 query 却大量是非标准写法
（「旅行收纳」「packing cubes」「出差整理袋」都指向 travel accessories）。本模块离线为
每个品类生成 5-10 条别名，写进该品类**所有卡片**的 ``aliases`` 字段——既进 BM25 索引
（词面命中），也并入 ``search_text()`` 编码（语义命中）。

**磁盘缓存**：按（品类名, 模型名）缓存生成结果，复跑建库不重复花钱；换 LLM 模型自动失效。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import BaseModel, Field

from scripts.etl._llm import LLM_SEM, etl_llm_model, get_etl_llm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 别名缓存（gitignore 的 data/ 下，随建库产物走）。
CACHE_PATH = PROJECT_ROOT / "data" / "rag" / "_alias_cache.json"


class _Aliases(BaseModel):
    aliases: list[str] = Field(
        description="5-10 条该品类的别名/口语说法，中英混合",
        min_length=3,
        max_length=12,
    )


_SYSTEM = """\
You are an e-commerce search expert. Given a product category name, list 5-10 \
alternative names real shoppers would type when looking for this category.

Include a mix of:
- English synonyms and common sub-type names (e.g. "packing cubes" for travel accessories)
- Simplified Chinese equivalents shoppers would actually type (e.g. 旅行收纳)
- Colloquial or scenario phrasings, kept SHORT (2-5 words each)

Rules:
- Each alias must clearly point to THIS category, not a neighboring one.
- No marketing fluff, no full sentences, no duplicates.
- Do not repeat the category name itself verbatim.
"""


def _load_cache() -> dict[str, list[str]]:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _cache_key(category: str, model: str) -> str:
    return f"{model}::{category}"


async def _generate_one(llm, category: str) -> list[str]:
    structured = llm.with_structured_output(_Aliases)
    async with LLM_SEM:
        try:
            result: _Aliases = await structured.ainvoke(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Category: {category}"},
                ]
            )
        except Exception as e:
            print(f"  ⚠ {category}: 别名生成失败 ({type(e).__name__}: {e})")
            return []
    # 去重 + 剔除与品类名撞车的条目（大小写不敏感）。
    seen: set[str] = {category.lower()}
    out: list[str] = []
    for a in result.aliases:
        a = " ".join(a.strip().split())
        if a and a.lower() not in seen:
            seen.add(a.lower())
            out.append(a)
    return out[:10]


async def build_alias_map(categories: list[str], *, progress: bool = True) -> dict[str, list[str]]:
    """为给定品类批量生成别名，返回 ``{品类: [别名…]}``（带磁盘缓存，缓存键含模型名）。"""
    model = etl_llm_model()
    cache = _load_cache()
    todo = [c for c in categories if _cache_key(c, model) not in cache]
    if progress:
        print(f"  别名缓存命中 {len(categories) - len(todo)} / 待生成 {len(todo)}")

    if todo:
        llm = get_etl_llm()
        results = await asyncio.gather(*(_generate_one(llm, c) for c in todo))
        for cat, aliases in zip(todo, results, strict=True):
            if aliases:  # 失败的不写缓存，下次复跑重试
                cache[_cache_key(cat, model)] = aliases
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    return {c: cache.get(_cache_key(c, model), []) for c in categories}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    cats = sys.argv[1:] or ["travel accessories", "headphones & earbuds"]
    print(json.dumps(asyncio.run(build_alias_map(cats)), ensure_ascii=False, indent=2))
