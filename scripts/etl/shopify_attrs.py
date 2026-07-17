"""ETL：从 Shopify 商品分类法抽「属性骨架卡」（attribute_schema），桥接到本项目品类。

前三类卡（bestseller / attribute / price_range）都是**数值统计**——告诉你这个品类多少钱、
口碑如何、谁在卖爆，但回答不了「买这个品类到底在挑什么维度」。Shopify Product Taxonomy
为每个叶子品类挂了一组**属性 + 枚举取值**（Material=[Canvas, Leather…] / Features=[Anti-theft,
Convertible…]），正是这层语义骨架。本模块把它桥接到我们已有的品类上：

1. 候选集：取 Shopify ``level<=3`` 且**去掉通用噪声属性后仍 >=2 个属性**的品类（广品类对广品类，
   避免「luggage」错配到「Lunch Suitcases」这种过细叶子）。
2. 映射：用 BGE-M3 给候选品类名（full_name 路径）与我们的品类名各自编码，取 **top-3 语义
   近邻**做候选。子串匹配在这里完全不可用（``mug`` 会命中「Shaving Mugs」），必须走语义向量。
3. 裁决：配了 LLM 时由 LLM 在 top-3 里**选一个或全拒**（单最近邻会出「car care 配到
   Baby health items」这类语义近却语用错的匹配，纯相似度阈值治不了）；裁决结果按
   （模型, 品类, 候选）缓存，复跑不重复花钱。没配 LLM 退回「top-1 ≥ 阈值」的旧行为。
4. 出卡：``confidence`` 取匹配相似度（LLM 确认过的给 0.6 下限保底）；被拒/够不上阈值的
   品类不强行出卡（冷启动留给在线 WebSearch 兜底）。

**已知局限（Phase 3 再补）：** Shopify 的取值是「允许值」清单、无频率，本模块按其原序裁剪并把
常见材质/特性前置；真实「典型取值占比」需用商品数据（Amazon Reviews 2023 / milistu）聚合。
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from app.recall.towers import TowerClient
from scripts.etl._llm import LLM_SEM, etl_llm_model, get_etl_llm
from scripts.etl.aggregate import _slug

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = PROJECT_ROOT / "data" / "rag" / "shopify-taxonomy.json"
# 候选品类向量缓存（建库一次性编码，复跑直接读；gitignore）。
EMB_CACHE_PATH = PROJECT_ROOT / "data" / "rag" / "_shopify_emb_cache.npz"
# LLM 裁决缓存（键含模型名 + 候选集，任一变化自动失效）。
ADJ_CACHE_PATH = PROJECT_ROOT / "data" / "rag" / "_shopify_adj_cache.json"

# 通用/低区分度属性：对「这个品类该看哪些维度」几乎无信息，反而稀释卡片，剔掉。
DENY_ATTRS = {
    "age group",
    "infant age group",
    "target gender",
    "country",
    "allergen information",
    "care instructions",
    "cleaning instructions",
    "pattern",
    "color",
    "size",
}
# 无频率信号下的折中：把常见材质/特性提到取值表前面，保证 plastic/leather 等高频词不被裁掉
# （Shopify 取值按字母序，纯截断会把 Plastic(P)/Nylon(N) 这类常见值切没——正是「不要塑料的」要的）。
COMMON_FIRST = {
    "plastic",
    "leather",
    "faux leather",
    "metal",
    "aluminum",
    "stainless steel",
    "steel",
    "ceramic",
    "glass",
    "wood",
    "bamboo",
    "cotton",
    "nylon",
    "polyester",
    "silicone",
    "canvas",
    "rubber",
    "carbon fiber",
    "wool",
    "linen",
}

MAX_LEVEL = 3  # 候选品类的最深层级（广品类对广品类）
MIN_ATTRS = 2  # 去噪后至少剩这么多属性才算候选
MAX_ATTRS = 8  # 单卡最多保留几个属性维度
MAX_VALS = 15  # 单个维度最多保留几个取值
DEFAULT_THRESHOLD = 0.55  # 无 LLM 时的旧行为：top-1 相似度门槛，低于此不出卡
TOP_CANDS = 3  # 送 LLM 裁决的候选数
MIN_CAND_SIM = 0.45  # 候选入围线（略低于旧阈值：给 LLM 捞回「相似度偏低但语用对」的机会）
ADJ_CONF_FLOOR = 0.6  # LLM 确认过的匹配，confidence 下限（过 admit 的 0.5 门且体现人裁背书）
ENCODE_BATCH = 32  # SiliconFlow /embeddings 单请求批量上限保守取 32


class _Pick(BaseModel):
    choice: int = Field(
        description="0-based index of the best-matching Shopify category, or -1 if none truly fits"
    )


_ADJ_SYSTEM = """\
You match e-commerce categories. Given OUR category and a few Shopify taxonomy \
candidates (with their attribute dimensions), pick the candidate whose ATTRIBUTES \
would genuinely help a shopper choose products in OUR category.

Rules:
- Judge by meaning and shopping intent, not string similarity.
- If no candidate's attributes fit OUR category, answer -1. A wrong match is \
worse than no match.
"""


def _resolve_attrs(node: dict, attr_by_id: dict) -> dict[str, list[str]]:
    """把一个品类节点的属性解析成 ``{属性名: [取值…]}``（去噪 + 裁剪 + 常见值前置）。"""
    out: dict[str, list[str]] = {}
    for att in node.get("attributes", []):
        ad = attr_by_id.get(att.get("id"))
        if not ad or ad["name"].strip().lower() in DENY_ATTRS:
            continue
        vals = [v["name"] for v in ad.get("values", []) if v.get("name") and v["name"] != "Other"]
        if not vals:
            continue
        # 常见值前置（稳定排序：命中 COMMON_FIRST 的排前，组内保持原序）。
        vals.sort(key=lambda v: 0 if v.lower() in COMMON_FIRST else 1)
        out[ad["name"]] = vals[:MAX_VALS]
        if len(out) >= MAX_ATTRS:
            break
    return out


def load_candidates() -> dict[str, dict[str, list[str]]]:
    """加载 Shopify 候选品类：``{full_name: {属性: [取值]}}``（level<=3 且去噪后 >=2 属性）。"""
    d = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    attr_by_id = {a["id"]: a for a in d["attributes"]}
    cand: dict[str, dict[str, list[str]]] = {}
    for vert in d["verticals"]:
        for node in vert["categories"]:
            if node["level"] > MAX_LEVEL:
                continue
            schema = _resolve_attrs(node, attr_by_id)
            if len(schema) >= MIN_ATTRS:
                cand[node["full_name"]] = schema
    return cand


async def _encode_batched(tower: TowerClient, texts: list[str]) -> np.ndarray:
    mats = []
    for i in range(0, len(texts), ENCODE_BATCH):
        mats.append(await tower.encode_texts(texts[i : i + ENCODE_BATCH]))
    return np.vstack(mats)


async def _candidate_embeddings(tower: TowerClient, names: list[str]) -> np.ndarray:
    """候选品类向量（带磁盘缓存：名单**与编码模型**都不变才复用）。

    缓存必须连模型一起 key——只认名单会在换 ``EMBED_MODEL`` 后复用上一个模型的旧向量，
    与新模型编码的品类向量做跨模型余弦，要么维度不符直接崩、要么算出无意义相似度灌进知识库。
    """
    model = tower._model or "local"  # noqa: SLF001
    if EMB_CACHE_PATH.exists():
        z = np.load(EMB_CACHE_PATH, allow_pickle=True)
        # 旧版缓存可能没有 model key（KeyError）——当缓存未命中处理重算，别让陈旧缓存崩掉建库。
        if "model" in z.files and list(z["names"]) == names and str(z["model"]) == model:
            return z["emb"]
    emb = await _encode_batched(tower, names)
    np.savez(EMB_CACHE_PATH, emb=emb, names=np.array(names, dtype=object), model=model)
    return emb


def _load_adj_cache() -> dict[str, int]:
    if ADJ_CACHE_PATH.exists():
        try:
            return json.loads(ADJ_CACHE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


async def _adjudicate_one(llm, category: str, cand_lines: list[str]) -> int:
    """让 LLM 在候选里选一个（返回 0-based 下标）或全拒（返回 -1）；调用失败也算全拒。"""
    structured = llm.with_structured_output(_Pick)
    prompt = f"OUR category: {category}\n\nCandidates:\n" + "\n".join(
        f"{i}. {line}" for i, line in enumerate(cand_lines)
    )
    async with LLM_SEM:
        try:
            result: _Pick = await structured.ainvoke(
                [
                    {"role": "system", "content": _ADJ_SYSTEM},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception as e:
            print(f"  ⚠ {category}: Shopify 裁决失败，按拒绝处理 ({type(e).__name__})")
            return -1
    return result.choice if 0 <= result.choice < len(cand_lines) else -1


async def _resolve_matches(
    categories: list[str], names: list[str], cand: dict, sims: np.ndarray, threshold: float
) -> dict[str, tuple[str, float]]:
    """为每个品类定出最终匹配 ``{品类: (Shopify full_name, confidence)}``（不匹配则缺席）。

    有 LLM：top-3 入围（≥ MIN_CAND_SIM）→ LLM 选/拒，裁决按（模型, 品类, 候选）缓存。
    无 LLM：退回旧行为 top-1 ≥ threshold。
    """
    matched: dict[str, tuple[str, float]] = {}
    if not os.environ.get("OPENAI_API_KEY"):
        for row, cat in enumerate(categories):
            idx = int(sims[row].argmax())
            score = float(sims[row][idx])
            if score >= threshold:
                matched[cat] = (names[idx], round(min(0.95, score), 2))
        return matched

    cache = _load_adj_cache()
    model = etl_llm_model()
    llm = None
    pending: list[tuple[str, str, list[tuple[int, float]]]] = []  # (cat, key, 候选[(idx,score)])
    tasks = []
    for row, cat in enumerate(categories):
        order = np.argsort(sims[row])[::-1][:TOP_CANDS]
        cands = [(int(i), float(sims[row][i])) for i in order if sims[row][i] >= MIN_CAND_SIM]
        if not cands:
            continue
        key = f"{model}::{cat}::" + "|".join(names[i] for i, _ in cands)
        if key in cache:
            choice = cache[key]
        else:
            if llm is None:
                llm = get_etl_llm()
            lines = [f"{names[i]} — attributes: {', '.join(cand[names[i]])}" for i, _ in cands]
            pending.append((cat, key, cands))
            tasks.append(_adjudicate_one(llm, cat, lines))
            continue
        if choice >= 0:
            idx, score = cands[choice]
            matched[cat] = (names[idx], round(min(0.95, max(score, ADJ_CONF_FLOOR)), 2))

    if tasks:
        choices = await asyncio.gather(*tasks)
        for (cat, key, cands), choice in zip(pending, choices, strict=True):
            cache[key] = choice
            if choice >= 0:
                idx, score = cands[choice]
                matched[cat] = (names[idx], round(min(0.95, max(score, ADJ_CONF_FLOOR)), 2))
        ADJ_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        ADJ_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return matched


async def build_attribute_cards(
    categories: list[str],
    tower: TowerClient,
    now: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict]:
    """为给定品类各产一张 attribute_schema 卡草稿（top-3 语义候选 + LLM 裁决，被拒不出卡）。

    返回 dict 草稿（与 :func:`aggregate.cards_for` 同形），交由 build 脚本走 admit + 编码向量。
    """
    cand = load_candidates()
    names = list(cand.keys())
    cand_emb = await _candidate_embeddings(tower, names)
    cat_emb = await _encode_batched(tower, categories)

    sims = cat_emb @ cand_emb.T  # 两边都已 L2 归一 → 内积即余弦
    matched = await _resolve_matches(categories, names, cand, sims, threshold)

    cards: list[dict] = []
    for cat, (full_name, score) in matched.items():
        schema = cand[full_name]
        dims = " / ".join(schema.keys())
        evidence = [f"映射自 Shopify 品类：{full_name}（置信度 {score:.2f}）"]
        evidence += [f"{name}: {', '.join(vals)}" for name, vals in schema.items()]
        cards.append(
            {
                "card_id": f"{_slug(cat)}_attribute_schema",
                "category": cat,
                "card_type": "attribute_schema",
                "summary": f"属性维度：{dims}",
                "raw_evidence": evidence,
                "last_updated": now,
                "confidence": score,  # _resolve_matches 已 round + 封顶
            }
        )
    return cards
