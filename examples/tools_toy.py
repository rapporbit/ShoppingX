"""M1 示例共用的玩具工具与教学版 system prompt。

被 01_min_loop.py / 02_stream.py 共用，避免重复。工具按主线规范写：
async + Pydantic 输出 + 给模型看的 docstring；实现是假数据，重点演示循环，真实版见 M11+。
"""

import asyncio

from langchain_core.tools import tool
from pydantic import BaseModel


class PlannerOutput(BaseModel):
    category: str
    budget_cny: int | None
    excludes: list[str]
    style: str | None


@tool
async def planner(intent: str) -> PlannerOutput:
    """把购物意图拆成结构化字段（品类/预算/排除项/风格）。

    何时调用：用户需求含多个约束（预算、材质、风格）时，先调它再去搜。
    """
    # 玩具实现：固定返回一份拆解结果（真实 NLP 拆解见 M11 planner）。
    return PlannerOutput(category="旅行收纳", budget_cny=300, excludes=["塑料"], style="小众")


class Candidate(BaseModel):
    name: str
    platform: str
    price_cny: float
    material: str


class ItemSearchOutput(BaseModel):
    query: str
    candidates: list[Candidate]


@tool
async def item_search(query: str) -> ItemSearchOutput:
    """单平台商品检索，返回候选商品列表。

    何时调用：需要按关键词找商品时。同一关键词不要反复搜。
    """
    await asyncio.sleep(0)  # 占位：模拟一次异步 IO
    return ItemSearchOutput(
        query=query,
        candidates=[
            Candidate(name="帆布旅行收纳袋", platform="shopee", price_cny=65, material="帆布"),
            Candidate(name="硅胶分装套装", platform="amazon", price_cny=89, material="硅胶"),
            Candidate(name="塑料洗漱包", platform="aliexpress", price_cny=39, material="塑料"),
        ],
    )


class PlatformPrice(BaseModel):
    platform: str
    landed_price_cny: float


class PriceCompareOutput(BaseModel):
    item_name: str
    prices: list[PlatformPrice]
    cheapest_platform: str


@tool
async def price_compare(item_name: str) -> PriceCompareOutput:
    """跨平台比价，返回各平台含运费的到手价与最便宜平台。

    何时调用：已有候选商品、需要对比到手价时。
    """
    prices = [
        PlatformPrice(platform="shopee", landed_price_cny=65),
        PlatformPrice(platform="amazon", landed_price_cny=89),
        PlatformPrice(platform="aliexpress", landed_price_cny=72),
    ]
    cheapest = min(prices, key=lambda p: p.landed_price_cny).platform
    return PriceCompareOutput(item_name=item_name, prices=prices, cheapest_platform=cheapest)


# 教学最小 system prompt：只放 role + workflow + termination，重点教「循环 + 收尾」。
SYSTEM_PROMPT = """<role>
你是 ShoppingX 购物助手（教学最小版），只负责：拆解需求、搜商品、跨平台比价。
</role>
<workflow>
按 Think → Act → Observe → Reflect 推进：含多约束的复杂需求先用 planner 拆解，
再用 item_search 搜商品，需要比价时用 price_compare，然后给推荐。
</workflow>
<termination>
最重要：信息足够（已搜到商品并比好价）就立刻用自然语言给出推荐并结束，不要再调任何工具，
不要反复搜同一关键词。
</termination>"""
