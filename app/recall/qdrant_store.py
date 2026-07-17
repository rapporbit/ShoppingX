"""Qdrant 召回客户端（BGE-M3 dense + payload filter），替代 Faiss 召回层。

对齐选型文档 `docs/plans/召回引擎选型思路.md`：召回 = **dense 语义 + 结构化/full-text filter**。
精确命中/硬约束是 filter 问题、非打分,故**不挂 sparse 打分腿**（§4 scoring vs filtering 转念）。

双后端：
- ``QDRANT_URL`` 配了 → 连真 server（prod）。
- 否则 ``QDRANT_PATH``（默认 ``./data/qdrant``）→ on-disk 本地模式（持久化，**单进程独占**）。
- ``QDRANT_PATH=":memory:"`` → 内存模式（测试，建+查同进程）。

数据模型：collection 名取自 ``QDRANT_COLLECTION``（默认 ``shoppingx_items``），命名向量
``dense``(COSINE)，payload = 归一商品字段（platform / price_usd / rating 等标量供 filter）。
检索：单路 dense + 多维 payload filter。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from qdrant_client import QdrantClient, models

from app.recall.schemas import ItemRecord, RecallCandidate

COLLECTION = os.environ.get("QDRANT_COLLECTION", "shoppingx_items")
DENSE_VEC = "dense"
DEFAULT_QDRANT_PATH = "./data/qdrant"
# 单次 upsert 的 point 数上限：server 模式下整批走一个 HTTP 请求，Qdrant 默认体上限 32MB。
# 一条 1024 维 dense point（向量 JSON 序列化 ~18KB + payload）约 22KB，512 条 ≈ 11MB，留足余量。
# 不分批时 amazon 单平台 5 万点 ≈ 1.1GB → 400 Bad Request（payload too large）。
UPSERT_BATCH = 512


def make_client() -> QdrantClient:
    """按 env 造 Qdrant 客户端：server / on-disk 本地 / 内存。"""
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))
    path = os.environ.get("QDRANT_PATH", DEFAULT_QDRANT_PATH)
    if path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=path)


class QdrantRecall:
    """Qdrant 后端召回：建 collection / upsert / dense + filter search。"""

    def __init__(self, client: QdrantClient | None = None) -> None:
        self._client = client or make_client()

    @property
    def client(self) -> QdrantClient:
        return self._client

    # ---------------------- 建库 / 写入 ----------------------
    def ensure_collection(self, dim: int, *, recreate: bool = False) -> None:
        """建 collection（dense COSINE，向量与 HNSW 图均落盘）。``recreate`` 时先删后建。

        ``on_disk=True``：138 万点 × 1024 维 float32 ≈ 5.4GB，全驻内存会顶满容器上限。
        代价是打分需读盘，查询延迟高于内存态；未挂量化副本，故无 rescore 加速。
        """
        if recreate and self._client.collection_exists(COLLECTION):
            self._client.delete_collection(COLLECTION)
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                COLLECTION,
                vectors_config={
                    DENSE_VEC: models.VectorParams(
                        size=dim, distance=models.Distance.COSINE, on_disk=True
                    )
                },
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )
            self.ensure_payload_indexes()

    def ensure_payload_indexes(self) -> None:
        """给 filter 字段建 payload 索引：百万级下无索引等于全表扫描。

        字段与 :meth:`search` 的 filter 维度一一对应（platform / price_usd / rating），外加
        ``item_id``——它不是检索维度，而是 :meth:`similar`（「搜同款」）按业务 id 反查 point 用的：
        point id 是建库时的自增整数，业务侧只有 item_id，没这个索引就得全表扫 138 万点。
        重复建同名索引 Qdrant 直接返回 ok，故本方法幂等、可重入。
        """
        for field, schema in (
            ("platform", models.PayloadSchemaType.KEYWORD),
            ("price_usd", models.PayloadSchemaType.FLOAT),
            ("rating", models.PayloadSchemaType.FLOAT),
            ("item_id", models.PayloadSchemaType.KEYWORD),
        ):
            self._client.create_payload_index(
                COLLECTION, field_name=field, field_schema=schema, wait=True
            )

    def upsert(self, records: list[ItemRecord], dense: np.ndarray, *, start_id: int) -> None:
        """写入一批 point：dense 向量 + payload（归一字段，供展示与 filter）。

        分批发送：server 模式下单请求体受 32MB 限制，整平台一次 upsert（5 万点 ~1.1GB）会
        被拒（400 payload too large）。按 ``UPSERT_BATCH`` 切块，逐块 upsert。
        """
        points: list[models.PointStruct] = []
        for offset, (rec, vec) in enumerate(zip(records, dense, strict=True)):
            points.append(
                models.PointStruct(
                    id=start_id + offset,
                    vector={DENSE_VEC: [float(x) for x in vec]},
                    payload=rec.model_dump(),  # embed_text 已 exclude，不入 payload
                )
            )
        for i in range(0, len(points), UPSERT_BATCH):
            self._client.upsert(COLLECTION, points[i : i + UPSERT_BATCH])

    # ---------------------- 检索 ----------------------
    def search(
        self,
        dense_vec: np.ndarray,
        top_k: int = 20,
        platform: str | Sequence[str] = "all",
        *,
        price_usd_max: float | None = None,
        min_rating: float | None = None,
    ) -> list[RecallCandidate]:
        """dense 召回 + 多维 payload filter。

        - ``dense_vec``：``TowerClient.encode_query`` 产出的请求向量（个性化靠偏好词并入检索词，
          不走向量融合）。
        - ``platform``：``"all"`` 不过滤；单个平台名转精确 filter（MatchValue）；**平台名序列**转
          多值 filter（MatchAny）——上层的「启用平台集合」（``agent.platform_scope``）走这条：用户
          没勾的平台不该被捞进来，所以那里的 "all" 只等于「全部**启用**平台」而非全库。
        - ``price_usd_max``：预算上限（USD），过滤 payload 的 ``price_usd``（建库时预折算）。
        - ``min_rating``：最低评分，过滤 payload 的 ``rating``。

        平台名先 ``strip().lower()`` 归一再做 filter（payload 里平台名全小写）：模型可能生成
        ``Amazon``/``AMAZON`` 等等价但不规范的串，不归一会匹配落空、静默召回 0 条。
        """
        names = [platform] if isinstance(platform, str) else list(platform)
        names = [n.strip().lower() for n in names if n and n.strip()]
        must: list[models.Condition] = []
        if names == ["all"] or not names:
            pass  # 全库不过滤（离线脚本 / 单测的旧口径）
        elif len(names) == 1:
            must.append(
                models.FieldCondition(key="platform", match=models.MatchValue(value=names[0]))
            )
        else:
            must.append(models.FieldCondition(key="platform", match=models.MatchAny(any=names)))
        if price_usd_max is not None:
            must.append(
                models.FieldCondition(key="price_usd", range=models.Range(lte=price_usd_max))
            )
        if min_rating is not None:
            must.append(models.FieldCondition(key="rating", range=models.Range(gte=min_rating)))
        flt = models.Filter(must=must) if must else None
        dense = [float(x) for x in np.asarray(dense_vec, dtype=np.float32).ravel()]
        res = self._client.query_points(
            COLLECTION,
            query=dense,
            using=DENSE_VEC,
            query_filter=flt,
            limit=top_k,
            with_payload=True,
        )
        out: list[RecallCandidate] = []
        for p in res.points:
            payload = p.payload or {}
            out.append(RecallCandidate(**payload, score=float(p.score)))
        return out


    def similar(self, item_id: str, top_k: int = 8) -> list[RecallCandidate]:
        """「搜同款」：按已入库商品的向量找全库近邻（不重新 embed，向量在服务端取）。

        两步：``item_id`` → point id（走 payload 索引 scroll），再 ``query=<point id>`` 让 Qdrant
        以那条 point 的 dense 向量做近邻检索。**不带任何 filter**——同款就是「整个库里最像它的」，
        平台/价格/评分都不该在这里限死（诚实提示：库里 amazon 占绝大多数，故近邻多半也是 amazon）。

        商品自身必然是自己的最近邻（score≈1），多取一条再按 item_id 剔除自身。
        item_id 在库里不存在（老会话的收藏、换过库）时返回空列表，由上层决定怎么呈现。
        """
        found, _ = self._client.scroll(
            COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="item_id", match=models.MatchValue(value=item_id))
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        if not found:
            return []
        res = self._client.query_points(
            COLLECTION,
            query=found[0].id,  # 按 point id 取近邻：向量不出服务端，省一趟 1024 维往返
            using=DENSE_VEC,
            limit=top_k + 1,
            with_payload=True,
        )
        out: list[RecallCandidate] = []
        for p in res.points:
            payload = p.payload or {}
            if payload.get("item_id") == item_id:
                continue  # 自己
            out.append(RecallCandidate(**payload, score=float(p.score)))
        return out[:top_k]


@lru_cache(maxsize=1)
def get_recall_client() -> QdrantRecall:
    """进程内共享的召回客户端（主/子 Agent 及各工具复用，避免重复连接）。"""
    return QdrantRecall()
