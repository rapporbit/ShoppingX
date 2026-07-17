"""运费估算（简化模型）。

到手价的另一块：跨境运费。真实运费随承运商、体积重、时效档浮动；本项目按简化模型——
**起步价 + 按重量加价**，并按平台所在区域分档（东南亚平台到欧美更贵），再叠加一个
「满额包邮」的常见规则。够 Agent 估个「大概运费」即可，不接真实物流报价。

约定：所有费用以基准币（默认 USD）计。重量缺省给一个保守默认值。
"""

from __future__ import annotations

from pydantic import BaseModel

from app.recall.geo import platform_origin_country

# 平台 → 发货区域（决定跨境运费档位）。对齐 utils.clean.PLATFORMS 的 5 个平台——eBay 在清洗阶段
# 已整体剔除，库里没有它的商品。表外平台走 DEFAULT_SHIPPING（下面 estimate 的 .get 兜底）。
PLATFORM_REGION: dict[str, str] = {
    "amazon": "us",
    "walmart": "us",
    "shein": "cn",
    "lazada": "sea",  # 东南亚
    "shopee": "sea",
}

# 各发货区域的运费参数：起步价 + 每千克加价（USD）。
SHIPPING_TABLE: dict[str, dict[str, float]] = {
    "us": {"base": 6.0, "per_kg": 4.0},
    "cn": {"base": 4.0, "per_kg": 6.0},
    "sea": {"base": 5.0, "per_kg": 7.0},
}
DEFAULT_SHIPPING = {"base": 8.0, "per_kg": 8.0}

# 国内件（发货国 == 收货国）：不出关、不走国际干线，比跨境便宜得多。上面那几档都是**跨境**运费，
# 用它们算「美国仓发美国」会平白多收一笔国际运费（此前 dest_country 参数收了却没用，就是这个坑）。
DOMESTIC_SHIPPING = {"base": 3.0, "per_kg": 2.0}

DEFAULT_WEIGHT_KG = 0.5  # 重量未知时的保守估值
FREE_SHIPPING_THRESHOLD_USD = 50.0  # 货价满此额包邮（简化的常见规则）


class ShippingEstimate(BaseModel):
    """运费估算结果。"""

    amount: float
    currency: str
    region: str
    weight_kg: float
    free: bool
    domestic: bool = False  # True=国内件（发货国==收货国），走更便宜的国内档


def estimate_shipping(
    platform: str = "",
    dest_country: str = "US",
    weight_kg: float | None = None,
    item_price_base: float | None = None,
    base: str = "USD",
) -> ShippingEstimate:
    """估算运费。

    - ``platform``：用于推断发货区域（决定运费档）。
    - ``dest_country``：收货国；与平台发货国相同则按**国内件**算（更便宜的 DOMESTIC_SHIPPING 档）。
    - ``item_price_base``：货价（基准币），满包邮阈值则免运费。
    - ``weight_kg``：缺省用保守默认重量。
    """
    region = PLATFORM_REGION.get(platform.strip().lower(), "other")
    weight = weight_kg if weight_kg is not None else DEFAULT_WEIGHT_KG

    # 国内件 vs 跨境件：发货国 == 收货国就不出关，走国内档。
    origin = platform_origin_country(platform)
    domestic = bool(origin) and origin == dest_country.strip().upper()
    params = DOMESTIC_SHIPPING if domestic else SHIPPING_TABLE.get(region, DEFAULT_SHIPPING)

    if item_price_base is not None and item_price_base >= FREE_SHIPPING_THRESHOLD_USD:
        return ShippingEstimate(
            amount=0.0,
            currency=base,
            region=region,
            weight_kg=weight,
            free=True,
            domestic=domestic,
        )

    amount = params["base"] + params["per_kg"] * weight
    return ShippingEstimate(
        amount=round(amount, 2),
        currency=base,
        region=region,
        weight_kg=weight,
        free=False,
        domestic=domestic,
    )
