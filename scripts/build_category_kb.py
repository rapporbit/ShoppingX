"""构建 RAG 品类知识库（可复现，产物 gitignore）。

离线流程：聚合抽卡 → 入库门禁 → 编码向量 → 写 JSONL →（配了 OpenSearch 则）建索引 +
注册 Hybrid 管道 + bulk 灌库。``category_insight`` 运行时只读这份产物。

用法：
    uv run python scripts/build_category_kb.py            # 全量
    uv run python scripts/build_category_kb.py --limit 40 # 只取前 N 个品类（快速试跑）

不配 ``OPENSEARCH_HOST`` 时只产 JSONL，由本地回退后端读取（离线可跑）；配了则同时灌进
OpenSearch，走引擎层 Hybrid。两种情况都用同一个 TowerClient 编码，保证向量空间一致。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv  # noqa: E402

from app.recall.category_kb import CategoryCard  # noqa: E402
from app.recall.kb_client import DEFAULT_CARDS_PATH  # noqa: E402
from app.recall.towers import DEFAULT_LOCAL_DIM, TowerClient  # noqa: E402
from scripts.etl.admit import admit  # noqa: E402
from scripts.etl.aggregate import aggregate, cards_for  # noqa: E402

# 离线建库脚本不经 app.agent.llm 那条链，必须自己加载 .env，否则 EMBED_* 读不到、
# TowerClient 静默退化成本地哈希回退——正是这会把「假向量」写进知识库（P0）。
load_dotenv()

ENCODE_BATCH = 64


async def _encode_vectors(cards: list[CategoryCard], tower: TowerClient) -> None:
    """给每张卡片编码 content_vector（建库时一次性算好，运行时本地后端直接用）。"""
    for i in range(0, len(cards), ENCODE_BATCH):
        batch = cards[i : i + ENCODE_BATCH]
        mat = await tower.encode_texts([c.search_text() for c in batch])
        for card, vec in zip(batch, mat, strict=True):
            card.content_vector = [float(x) for x in vec]


async def main(limit: int | None, require_remote: bool = False) -> None:
    print("聚合商品数据…")
    aggs = aggregate()
    print(f"  归一品类数：{len(aggs)}")

    # 入库门禁：抽卡 → admit，记录拒绝原因分布（诚实地报通过率）。
    admitted: list[CategoryCard] = []
    rejected = 0
    cats = sorted(aggs.values(), key=lambda a: a.count, reverse=True)
    if limit:
        cats = cats[:limit]
    for agg in cats:
        for raw in cards_for(agg):
            ok, reason = admit(raw)
            if ok:
                admitted.append(CategoryCard(**raw))
            else:
                rejected += 1
    print(f"  门禁通过 {len(admitted)} 张 / 拒绝 {rejected} 张")
    if not admitted:
        print("没有可入库的卡片，退出。")
        return

    tower = TowerClient()
    print(f"编码器：{'远程真实 embedding' if tower.remote else f'本地确定性回退 dim={tower.dim}'}")
    if require_remote and not tower.remote:
        raise SystemExit(
            "❌ --require-remote：EMBED_MODEL 未生效，当前是本地哈希回退。"
            "拒绝把假向量写进知识库。\n"
            "   检查 .env 的 EMBED_MODEL / EMBED_BASE_URL / EMBED_API_KEY 是否已配置且被加载。"
        )

    # attribute_schema 属性骨架卡：把 Shopify 分类法的「属性+取值」桥接到本项目品类。
    # 仅在真实 embedding 下构建——语义最近邻匹配靠向量，本地哈希回退下匹配纯属噪声。
    if tower.remote:
        from datetime import UTC, datetime

        from scripts.etl.shopify_attrs import build_attribute_cards

        now = datetime.now(UTC).isoformat(timespec="seconds")
        distinct_cats = sorted({c.category for c in admitted})
        print(f"构建 attribute_schema 卡（{len(distinct_cats)} 个品类匹配 Shopify）…")
        drafts = await build_attribute_cards(distinct_cats, tower, now)
        kept = 0
        for raw in drafts:
            ok, _ = admit(raw)
            if ok:
                admitted.append(CategoryCard(**raw))
                kept += 1
        print(
            f"  attribute_schema 入库 {kept} 张 / 未达匹配阈值或被拒 "
            f"{len(distinct_cats) - kept} 个品类"
        )
    else:
        print("本地回退编码：跳过 attribute_schema（语义匹配需真实 embedding）")

    # LLM 属性分布卡：用 LLM 按品类生成多维度属性分布（Material / Connectivity / ...），
    # 补上 aggregate.py 只算评分分布的缺口（refdocs 要的是「材质：尼龙 60% / 帆布 25%」级别）。
    # 需要 LLM API（LLM_MAIN），没配则跳过。card_type 仍为 "attribute"，与现有评分分布卡共存。
    if os.environ.get("OPENAI_API_KEY"):
        from scripts.etl.llm_attributes import build_llm_attribute_cards

        distinct_cats = sorted({c.category for c in admitted})
        print(f"LLM 生成属性分布卡（{len(distinct_cats)} 个品类）…")
        llm_drafts = await build_llm_attribute_cards(distinct_cats)
        llm_kept = 0
        for raw in llm_drafts:
            ok, _ = admit(raw)
            if ok:
                admitted.append(CategoryCard(**raw))
                llm_kept += 1
        print(f"  LLM 属性卡入库 {llm_kept} 张 / 被拒 {len(llm_drafts) - llm_kept} 张")
    else:
        print("未配置 OPENAI_API_KEY：跳过 LLM 属性分布卡生成。")

    # 别名扩展：为每个品类生成中英别名/口语说法，注入该品类**所有**卡片——参与 BM25
    # （aliases 字段）与向量编码（search_text 并入），是两段式检索第一段「品类定位」
    # 接住非标准写法 query 的关键数据。必须在 _encode_vectors 之前注入。
    if os.environ.get("OPENAI_API_KEY"):
        from scripts.etl.llm_aliases import build_alias_map

        distinct_cats = sorted({c.category for c in admitted})
        print(f"生成品类别名（{len(distinct_cats)} 个品类）…")
        alias_map = await build_alias_map(distinct_cats)
        for card in admitted:
            card.aliases = alias_map.get(card.category, [])
        n_alias = sum(1 for c in distinct_cats if alias_map.get(c))
        print(f"  有别名的品类 {n_alias}/{len(distinct_cats)}")
    else:
        print("未配置 OPENAI_API_KEY：跳过别名生成（品类定位将只靠品类名本身）。")

    await _encode_vectors(admitted, tower)

    # 维度门禁：真实 embedding 不可能落到本地回退维度（256）。命中则说明发生了静默回退，
    # 与其把假向量写盘（复刻 P0），不如直接失败。
    dim = len(admitted[0].content_vector)
    if require_remote and dim == DEFAULT_LOCAL_DIM:
        raise SystemExit(
            f"❌ --require-remote：向量维度={dim} 等于本地回退维度，疑似静默回退，已中止。"
        )
    print(f"向量维度：{dim}")

    # 写 JSONL（本地回退后端的数据源）。
    out_path = Path(os.environ.get("CATEGORY_CARDS_PATH", DEFAULT_CARDS_PATH))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for card in admitted:
            # content_vector 是 exclude 字段，手动并回 JSONL 供本地后端读取。
            payload = {**card.model_dump(), "content_vector": card.content_vector}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"卡片写入 {out_path}（{len(admitted)} 张）")

    # 配了 OpenSearch 才灌库。
    if os.environ.get("OPENSEARCH_HOST"):
        from scripts.etl.os_setup import bulk_cards, make_client, recreate_index, register_pipeline

        dim = len(admitted[0].content_vector)
        print(f"灌入 OpenSearch（dim={dim}）…")
        client = make_client()
        recreate_index(client, dim)
        register_pipeline(client)
        n = bulk_cards(client, admitted)
        print(f"  已索引 {n} 张到 {client.transport.hosts}")
    else:
        print("未配置 OPENSEARCH_HOST：跳过灌库，仅产出 JSONL（本地回退后端可用）。")

    await tower.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只取月销前 N 个品类")
    parser.add_argument(
        "--require-remote",
        action="store_true",
        help="强制走真实 embedding：若退化到本地哈希回退则直接报错（建生产库时用）",
    )
    args = parser.parse_args()
    asyncio.run(main(args.limit, require_remote=args.require_remote))
