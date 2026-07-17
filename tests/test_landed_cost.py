"""收货国 → 关税 → 到手价这条链的确定性测试。

钉住四件事（每一件都对应一个曾经踩过或差点踩到的坑）：
1. 收货国解析是**纯规则**的：同一句话永远出同一个国家，且英文常用词（in / id / ca）不被误当 ISO 码。
2. 免征额按国家生效：关税不再像从前那样（US $800 免征额 + 库里商品普遍低价）恒为 0。
3. **机制兜底**：模型漏传 dest_country 时，shipping_calc 仍拿到系统认定的国家——正确性不靠 prompt。
4. 四层优先级：用户原话 > 会话 slots > 长期记忆 > 默认国，且只有落到默认国才算「假设」。
"""

from __future__ import annotations

import pytest

from app.api.context import get_dest_country, is_dest_country_assumed
from app.recall.duty import DE_MINIMIS_USD, estimate_duty
from app.recall.geo import DEFAULT_DEST_COUNTRY, resolve_dest_country
from app.tools.schemas import ItemCandidate
from app.tools.shipping_calc import shipping_calc
from app.utils.thread_ctx import thread_scope


def _candidate(price_usd: float, item_id: str = "i1") -> ItemCandidate:
    return ItemCandidate(
        item_id=item_id,
        platform="amazon",
        title="Test Item",
        price=price_usd,
        currency="USD",
        price_usd=price_usd,
        category="shoes",
    )


# ---------- 1. 收货国解析（纯规则） ----------
@pytest.mark.parametrize(
    ("text", "expected", "explicit"),
    [
        ("想买跑鞋，寄到日本", "JP", True),
        ("发中国就行", "CN", True),  # 裸动词紧邻国名，无「到/往」也认
        ("ship to Germany please", "DE", True),
        ("我要寄去印尼", "ID", True),  # 「印度尼西亚」含「印度」，不能被判成 IN
        ("买个包，发 JP", "JP", True),  # 大写 ISO 码认
        ("送到香港", "HK", True),
        ("日本直邮的墨镜", "JP", True),  # 后置语境
        ("美国清关方便的型号", "US", True),  # 后置语境
        ("收货地：新加坡", "SG", True),  # 记忆条目常见格式
        ("旅行收纳袋什么价位，我在日本", "JP", True),  # 人称化居住地表达认
        ("人在美国，买个空气炸锅", "US", True),
        # 关键反例：裸国名是产地 / 流派 / 风格修饰，不带收货语境不认（线上真实事故：
        # 「英国文学」曾被判成寄往 GB 并毒化会话 slots）。
        ("给我推几本英国文学作品，要求是讽刺现实社会类型的", DEFAULT_DEST_COUNTRY, False),
        ("日本料理的刀具", DEFAULT_DEST_COUNTRY, False),
        ("德国工艺的锅", DEFAULT_DEST_COUNTRY, False),
        ("在美国很火的水杯", DEFAULT_DEST_COUNTRY, False),  # 「在」不做语境词
        # 混合例：产地国名与收货国名同现，紧邻性消歧——「英国」旁是「文学」，「中国」旁是语境词。
        ("我要英国文学作品，寄到中国", "CN", True),
        ("我要英国文学作品，中国收货", "CN", True),
        ("从英国寄到中国的茶具", "CN", True),  # 发货地在语境词**前**，方向性挡掉 GB
        # 关键反例：小写英文常用词不能被当成 ISO 码。
        ("shoes in 300 budget", DEFAULT_DEST_COUNTRY, False),  # "in" ≠ 印度
        ("美元预算200的耳机", DEFAULT_DEST_COUNTRY, False),  # 「美元」≠ 美国
        ("便宜的行李箱", DEFAULT_DEST_COUNTRY, False),  # 没提 → 默认国
    ],
)
def test_resolve_dest_country(text: str, expected: str, explicit: bool) -> None:
    assert resolve_dest_country(text) == (expected, explicit)


def test_resolve_dest_country_is_deterministic() -> None:
    """同一句话解析 100 次必须完全一致（这正是不交给 LLM 的理由）。"""
    text = "预算 500 的相机，寄到日本"
    assert {resolve_dest_country(text) for _ in range(100)} == {("JP", True)}


# ---------- 2. 免征额按国家生效 ----------
def test_de_minimis_varies_by_country() -> None:
    price = 50.0
    # 日本免征额 130 → $50 免税；中国仅 7 → 同一件商品要缴税。
    assert estimate_duty(price, "shoes", "JP").duty_free
    cn = estimate_duty(price, "shoes", "CN")
    assert not cn.duty_free
    assert cn.amount == pytest.approx(price * 0.16)  # shoes 税率 16%
    # 自由港不征关税，且与免征额无关。
    assert estimate_duty(9999.0, "shoes", "HK").amount == 0.0
    # 免征额门槛要带回给调用方（供回复解释「为什么免税」）。
    assert estimate_duty(price, "shoes", "JP").threshold_usd == DE_MINIMIS_USD["JP"]


# ---------- 2b. 国内单不收关税（关税是「进口」税） ----------
def test_domestic_order_has_no_duty() -> None:
    """发货国 == 收货国 → 不过境、不清关，一分钱关税都没有。

    改造前 estimate_duty 只看收货国、不看发货地，amazon(美国仓) → 美国收货照样按 16% 征税；
    只因 US 免征额曾是 $800、库里商品普遍低价，才没人发现。
    """
    us_domestic = estimate_duty(500.0, "shoes", "US", platform="amazon")
    assert us_domestic.domestic and us_domestic.duty_free
    assert us_domestic.amount == 0.0

    # 同一件商品从中国的 shein 发到美国 → 跨境，照收。
    cross_border = estimate_duty(500.0, "shoes", "US", platform="shein")
    assert not cross_border.domestic
    assert cross_border.amount == pytest.approx(500.0 * 0.16)

    # 反过来寄中国：shein 成了国内单，amazon 成了跨境单。
    assert estimate_duty(500.0, "shoes", "CN", platform="shein").domestic
    assert not estimate_duty(500.0, "shoes", "CN", platform="amazon").domestic

    # 平台未知 / 未传 → 保守按跨境算（宁可多收，不可漏收该有的税）。
    assert not estimate_duty(500.0, "shoes", "US").domestic


@pytest.mark.asyncio
async def test_domestic_vs_crossborder_reorders_landed(tmp_path) -> None:
    """同价商品：寄美国时美国仓的更划算，寄中国时中国仓的更划算——到手价排序随收货国翻转。"""

    def cand(platform: str) -> ItemCandidate:
        c = _candidate(45.0, item_id=platform)
        c.platform = platform
        return c

    with thread_scope("t-domestic", tmp_path):
        us = await shipping_calc.ainvoke(
            {"item_ids": [], "dest_country": "US", "candidates": [cand("amazon"), cand("shein")]}
        )
        cn = await shipping_calc.ainvoke(
            {"item_ids": [], "dest_country": "CN", "candidates": [cand("amazon"), cand("shein")]}
        )
    assert us.items[0].platform == "amazon"  # 寄美国 → 美国仓国内单最便宜
    assert cn.items[0].platform == "shein"  # 寄中国 → 中国仓国内单最便宜
    assert us.items[0].duty_usd == 0.0 and cn.items[0].duty_usd == 0.0


# ---------- 3. 机制兜底：模型漏传 dest_country ----------
@pytest.mark.asyncio
async def test_shipping_calc_falls_back_to_context_dest(tmp_path) -> None:
    """模型没传 dest_country（空串）→ 工具仍按会话上下文认定的国家算，不退回硬编码的 US。"""
    with thread_scope("t-dest", tmp_path):
        out = await shipping_calc.ainvoke(
            {"item_ids": [], "dest_country": "", "candidates": [_candidate(50.0)]}
        )
    assert out.dest_country == DEFAULT_DEST_COUNTRY  # planner 没跑 → env 默认国
    item = out.items[0]
    assert item.landed_usd is not None
    assert item.landed_usd == pytest.approx(
        50.0 + (item.shipping_usd or 0.0) + (item.duty_usd or 0.0)
    )


@pytest.mark.asyncio
async def test_shipping_calc_dest_changes_landed_cost(tmp_path) -> None:
    """同一件商品换收货国 → 到手价必须真的变（改造前它恒等于货价 + 常数）。"""
    with thread_scope("t-dest2", tmp_path):
        jp = await shipping_calc.ainvoke(
            {"item_ids": [], "dest_country": "JP", "candidates": [_candidate(50.0)]}
        )
        cn = await shipping_calc.ainvoke(
            {"item_ids": [], "dest_country": "CN", "candidates": [_candidate(50.0)]}
        )
    assert jp.items[0].duty_usd == 0.0  # 低于日本免征额
    assert (cn.items[0].duty_usd or 0.0) > 0.0  # 高于中国免征额
    assert (cn.items[0].landed_usd or 0.0) > (jp.items[0].landed_usd or 0.0)


# ---------- 4. 四层优先级 ----------
@pytest.mark.asyncio
async def test_dest_country_layers(tmp_path, monkeypatch) -> None:
    """用户原话 > 会话 slots > 默认国；只有落到默认国才算「假设」，只有第 1 层算「本轮明示」。"""
    from app.api import context as ctx
    from app.memory.session_state import SessionPrefState
    from app.tools.planner import resolve_dest_country_layered

    with thread_scope("t-layers", tmp_path):
        # 第 1 层：本轮原话压过一切（哪怕 slots 里存着别的国家），且是唯一 stated_now=True 的层。
        ctx.set_session_pt(SessionPrefState(slots={"dest_country": "SG"}))
        assert await resolve_dest_country_layered("寄到日本") == ("JP", False, True)

        # 第 2 层：本轮没提 → 用会话 slots 里存着的收货国，不算「假设」也不算「本轮明示」。
        assert await resolve_dest_country_layered("再推荐几个") == ("SG", False, False)

        # 裸国名（产地/流派修饰）不触发第 1 层 → 仍落到 slots，不被「英国」劫持。
        assert await resolve_dest_country_layered("我要英国文学作品") == ("SG", False, False)

        # 第 4 层：既没提、slots 也空 → 落默认国，标记为「假设」（回复里必须声明）。
        ctx.set_session_pt(SessionPrefState())
        assert await resolve_dest_country_layered("再推荐几个") == (
            DEFAULT_DEST_COUNTRY,
            True,
            False,
        )


@pytest.mark.asyncio
async def test_assumed_dest_not_persisted_to_slots(tmp_path) -> None:
    """系统假设的默认国不该被当成用户事实沉进会话状态（否则下一轮它会伪装成「用户说过」）。"""
    from app.api.context import set_dest_country

    with thread_scope("t-assumed", tmp_path):
        set_dest_country(DEFAULT_DEST_COUNTRY, assumed=True)
        assert is_dest_country_assumed() is True
        assert get_dest_country() == DEFAULT_DEST_COUNTRY

        set_dest_country("JP", assumed=False)  # 用户明说过
        assert is_dest_country_assumed() is False
