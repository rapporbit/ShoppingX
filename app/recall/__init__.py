"""向量召回 + 汇率/关税/运费 + RAG 知识库（后端内部能力，不暴露给模型）。

公开入口（供 M4 的 item_search / price_compare / shipping_calc 等工具调用）：
- :class:`TowerClient` / :func:`get_tower_client`：query / 商品文本编码
- :class:`QdrantRecall` / :func:`get_recall_client`：Qdrant hybrid 召回（dense + 稀疏 BM25）
- :class:`RecallCandidate`：召回候选的归一化结构
- :func:`to_base` / :func:`estimate_duty` / :func:`estimate_shipping`：汇率/关税/运费
"""

from app.recall.category_kb import CategoryCard, normalize_category
from app.recall.duty import DutyEstimate, estimate_duty
from app.recall.fx import to_base, to_base_or_none
from app.recall.kb_client import KBClient, get_kb_client, should_disable_bm25
from app.recall.qdrant_store import QdrantRecall, get_recall_client
from app.recall.reranker import RerankerClient, get_reranker
from app.recall.schemas import ItemRecord, RecallCandidate
from app.recall.shipping import ShippingEstimate, estimate_shipping
from app.recall.towers import TowerClient, get_tower_client

__all__ = [
    "CategoryCard",
    "DutyEstimate",
    "ItemRecord",
    "KBClient",
    "QdrantRecall",
    "RecallCandidate",
    "RerankerClient",
    "ShippingEstimate",
    "TowerClient",
    "estimate_duty",
    "estimate_shipping",
    "get_kb_client",
    "get_recall_client",
    "get_reranker",
    "get_tower_client",
    "normalize_category",
    "should_disable_bm25",
    "to_base",
    "to_base_or_none",
]
