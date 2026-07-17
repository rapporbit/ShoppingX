"""M3 验收：召回基础设施的确定性测试（不依赖真实 embedding / 不依赖 data/ 索引）。

覆盖 ROADMAP M3 验收点（已升级到 Qdrant dense + filter）：
- QdrantRecall.search 对样例 query 返回带 metadata 的候选（``:memory:`` 自建小库，自洽）
- to_base 多币种归一正确
另含 TowerClient（确定性 + 归一 + 双通道融合）与 duty/shipping 估算的边界。
"""

from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.recall.duty import estimate_duty, lookup_duty_rate
from app.recall.fx import UnknownCurrencyError, to_base
from app.recall.qdrant_store import QdrantRecall
from app.recall.schemas import ItemRecord
from app.recall.shipping import estimate_shipping
from app.recall.towers import TowerClient
from app.utils.clean import _parse_int, _parse_number


# ---------- 清洗：脏文本数字解析（建索引前的前置步骤，逻辑在 app/utils/clean.py） ----------
def test_parse_number_handles_scientific_notation() -> None:
    # amazon 价格大量是科学计数法，漏指数会把 17.95 解析成 1.795（差一个数量级）。
    assert _parse_number("1.795000000000000e+01") == pytest.approx(17.95)
    assert _parse_number('"3.999900000000000e+02"') == pytest.approx(399.99)
    # 常规小数 / 带千分位 / 币符。
    assert _parse_number("57.79") == pytest.approx(57.79)
    assert _parse_number("$1,299.00") == pytest.approx(1299.0)
    # 脏值返回 None，不崩。
    assert _parse_number("null") is None
    assert _parse_number("") is None
    assert _parse_int("1.2e+04") == 12000


# ---------- TowerClient（本地确定性回退） ----------
async def test_tower_local_deterministic_and_normalized() -> None:
    tower = TowerClient(model=None, local_dim=128)
    v1 = await tower.encode_query("帆布旅行收纳袋")
    v2 = await tower.encode_query("帆布旅行收纳袋")
    # 同输入恒同输出（确定性）。
    assert np.allclose(v1, v2)
    assert v1.shape == (128,)
    # L2 归一化：模长为 1（空文本除外）。
    assert pytest.approx(float(np.linalg.norm(v1)), abs=1e-5) == 1.0
    # 不同文本给出不同向量。
    v3 = await tower.encode_query("硅胶分装套装")
    assert not np.allclose(v1, v3)


# ---------- QdrantRecall（:memory: 自建小库，dense + filter） ----------
async def _build_tiny_recall(dim: int = 32) -> QdrantRecall:
    """造一个两平台、确定性向量的 ``:memory:`` Qdrant 小库，供 dense 检索测试自洽。

    走 ``encode_texts`` 这条**生产同款**编码路径（含归一化）+ 真 upsert，让测试打在
    QdrantRecall 的公开接口上。COSINE 内积分恒正、确定性强，便于断言排序/得分。
    """
    tower = TowerClient(model=None, local_dim=dim)
    fixtures = [
        ItemRecord(
            item_id="A1",
            platform="amazon",
            title="canvas travel bag",
            price=20.0,
            embed_text="canvas travel bag",
        ),
        ItemRecord(
            item_id="A2",
            platform="amazon",
            title="silicone bottle set",
            price=9.0,
            embed_text="silicone bottle set",
        ),
        ItemRecord(
            item_id="S1",
            platform="shopee",
            title="canvas storage pouch",
            price=15.0,
            currency="SGD",
            embed_text="canvas storage pouch",
        ),
    ]
    recall = QdrantRecall(QdrantClient(location=":memory:"))
    encoded = await tower.encode_texts([r.embed_text for r in fixtures])
    recall.ensure_collection(dim, recreate=True)
    recall.upsert(fixtures, np.asarray(encoded, dtype="float32"), start_id=0)
    return recall


async def test_recall_search_returns_candidates_with_metadata() -> None:
    recall = await _build_tiny_recall()
    tower = TowerClient(model=None, local_dim=32)

    req = await tower.encode_query("canvas travel bag")
    results = recall.search(req, top_k=3, platform="all")
    assert results, "应至少召回一条"
    top = results[0]
    # 候选带 metadata（标题/价格/平台）+ 得分。
    assert top.title == "canvas travel bag"
    assert top.platform == "amazon"
    assert top.price == 20.0
    assert top.score > 0
    # 召回按分降序。
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))


async def test_recall_single_platform_filter() -> None:
    recall = await _build_tiny_recall()
    tower = TowerClient(model=None, local_dim=32)
    req = await tower.encode_query("canvas")
    only_shopee = recall.search(req, top_k=5, platform="shopee")
    assert only_shopee and all(c.platform == "shopee" for c in only_shopee)
    # 不存在的平台 → 空结果，不报错（payload filter 命中 0 条）。
    assert recall.search(req, top_k=5, platform="nope") == []


def test_recall_missing_collection_raises() -> None:
    # 没建 collection 直接检索 → 报错，不静默返垃圾。
    recall = QdrantRecall(QdrantClient(location=":memory:"))
    with pytest.raises(Exception):  # noqa: B017,PT011 (qdrant 本地模式抛 collection 不存在)
        recall.search(np.zeros(32, dtype="float32"), top_k=3)


# ---------- 汇率归一 ----------
def test_to_base_multi_currency() -> None:
    assert to_base(100, "USD", "USD") == 100.0
    assert pytest.approx(to_base(100, "EUR", "USD")) == 108.0
    # 往返折算应近似还原。
    assert pytest.approx(to_base(to_base(100, "EUR", "USD"), "USD", "EUR")) == 100.0
    # 非 USD 基准。
    assert pytest.approx(to_base(108, "USD", "EUR")) == 100.0


def test_to_base_unknown_currency() -> None:
    with pytest.raises(UnknownCurrencyError):
        to_base(100, "XYZ")


# ---------- 关税 ----------
def test_duty_rate_lookup_and_de_minimis() -> None:
    assert lookup_duty_rate("Men's Running Shoes") == 0.16  # 命中 shoes
    assert lookup_duty_rate("一个没见过的品类") == pytest.approx(0.07)  # 兜底
    # 日本免征额 130（课税价格 1 万日元）：低于则免税。
    free = estimate_duty(50, "shoes", "JP")
    assert free.duty_free and free.amount == 0.0
    assert free.threshold_usd == pytest.approx(130.0)
    # 高于免征额则按税率计。
    taxed = estimate_duty(1000, "shoes", "JP")
    assert not taxed.duty_free
    assert taxed.amount == pytest.approx(160.0)
    # 美国自 2025-08 取消 de minimis（免征额 0）：小额包裹同样计税，不再一律免税。
    us = estimate_duty(50, "shoes", "US")
    assert not us.duty_free
    assert us.amount == pytest.approx(8.0)


# ---------- 运费 ----------
def test_shipping_free_threshold_and_weight() -> None:
    # 满 50 包邮。（amazon 从美国发货，寄日本 → 跨境件）
    free = estimate_shipping("amazon", "JP", weight_kg=2.0, item_price_base=60)
    assert free.free and free.amount == 0.0
    # 未满则按 起步价 + 重量加价（跨境 us 档：6 + 4*1.0 = 10）。
    paid = estimate_shipping("amazon", "JP", weight_kg=1.0, item_price_base=20)
    assert not paid.free
    assert paid.amount == pytest.approx(10.0)
    # 东南亚区域更贵（跨境 sea 档：5 + 7*1.0 = 12）。
    sea = estimate_shipping("shopee", "US", weight_kg=1.0, item_price_base=20)
    assert sea.amount == pytest.approx(12.0)
    # 国内件（amazon 美国发货 → 美国收货）：不出关、不走国际干线，用更便宜的国内档 3 + 2*1.0 = 5。
    dom = estimate_shipping("amazon", "US", weight_kg=1.0, item_price_base=20)
    assert dom.domestic and dom.amount == pytest.approx(5.0)
