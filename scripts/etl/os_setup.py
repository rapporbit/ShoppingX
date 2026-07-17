"""OpenSearch 索引 mapping / Hybrid 搜索管道注册 / bulk 写入（refdocs 13 §3、13-1 §3）。

只在配置了 ``OPENSEARCH_HOST`` 时被 ``build_category_kb.py`` 调用——把卡片灌进真 OpenSearch，
让 ``category_insight`` 走引擎层的 KNN+BM25 加权融合。没配 OpenSearch 时整段跳过，知识库
仅以 JSONL 形式存在，由本地回退后端读取。
"""

from __future__ import annotations

import os
from typing import Any

from app.recall.category_kb import CategoryCard
from app.recall.kb_client import HYBRID_PIPELINE, HYBRID_WEIGHTS, INDEX_NAME, VECTOR_FIELD


def make_client() -> Any:
    from opensearchpy import OpenSearch

    return OpenSearch(
        hosts=[
            {
                "host": os.environ["OPENSEARCH_HOST"],
                "port": int(os.environ.get("OPENSEARCH_PORT", "9200")),
            }
        ],
        http_auth=(
            os.environ.get("OPENSEARCH_USER", "admin"),
            os.environ.get("OPENSEARCH_PASS", "admin"),
        ),
        use_ssl=os.environ.get("OPENSEARCH_USE_SSL", "false").lower() == "true",
        verify_certs=False,
        ssl_show_warn=False,
    )


def _index_mapping(dim: int) -> dict:
    """同一份文档同时存：结构化字段 + 英文全文字段 + KNN 向量。

    **英文分词器（主动更正 refdocs）**：语料是英文 Amazon 商品，用内置 ``english``
    分析器；中文 query 的跨语言匹配靠 KNN 向量兜底。``cosinesimil`` 配 L2 归一化向量。
    """
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "properties": {
                "card_id": {"type": "keyword"},
                # keyword 子字段供两段式检索第二段按品类精确取卡（term 查询）。
                "category": {
                    "type": "text",
                    "analyzer": "english",
                    "fields": {"raw": {"type": "keyword"}},
                },
                "card_type": {"type": "keyword"},
                # 别名中英混合：standard 分词器对 CJK 按单字切，词面路对中文 query 才有信号
                # （english 分析器的词干化只伺候英文）。
                "aliases": {"type": "text", "analyzer": "standard"},
                "summary": {"type": "text", "analyzer": "english"},
                "raw_evidence": {"type": "text", "analyzer": "english"},
                "last_updated": {"type": "date"},
                "confidence": {"type": "float"},
                VECTOR_FIELD: {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    },
                },
            }
        },
    }


def register_pipeline(client: Any) -> None:
    """注册 Hybrid 搜索管道：min_max 归一 + 算数平均加权融合（权重对应 KNN/BM25 子路顺序）。"""
    client.transport.perform_request(
        "PUT",
        f"/_search/pipeline/{HYBRID_PIPELINE}",
        body={
            "description": "KNN + BM25 双路召回的归一与加权融合",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": list(HYBRID_WEIGHTS)},
                        },
                    }
                }
            ],
        },
    )


def recreate_index(client: Any, dim: int) -> None:
    """重建索引（删旧 → 建新），保证 mapping 与当前编码维度一致。"""
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    client.indices.create(index=INDEX_NAME, body=_index_mapping(dim))


def bulk_cards(client: Any, cards: list[CategoryCard]) -> int:
    """把卡片 bulk 写入索引（含 content_vector）。"""
    from opensearchpy import helpers

    actions = [
        {
            "_index": INDEX_NAME,
            "_id": c.card_id,
            # 用 model_dump 拿不到 exclude 的 content_vector，这里手动并回去。
            "_source": {**c.model_dump(), VECTOR_FIELD: c.content_vector},
        }
        for c in cards
    ]
    ok, _ = helpers.bulk(client, actions, refresh=True)
    return ok
