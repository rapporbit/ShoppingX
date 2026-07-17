"""关税估算（简化模型，无 HS Code）。

到手价（landed cost）的一块：跨境买东西除了货价和运费，还可能被进口国收关税。
真实关税要查 HS 编码 + 原产国 + 贸易协定，极其繁琐；本项目按 CLAUDE.md「不做精细
关税对账」的定位，用**按品类的近似税率 + 免征额（de minimis）**做估算，够 Agent 给
用户一个「大概到手多少钱」的判断。

模型：``关税 = 货价(基准币) × 该品类税率``，但两种情况免税：
1. **国内订单**（发货国 == 收货国）——关税是**进口**税，amazon 从美国发货寄到美国不过境、
   不清关，一分钱关税都没有。这一条比免征额更优先，也是本模块最容易被忽略的前提。
2. 跨境但货价低于进口国免征额（de minimis）。
"""

from __future__ import annotations

from pydantic import BaseModel

from app.recall.geo import platform_origin_country

# 各品类近似从价税率（ad valorem）。命中靠品类文本里的关键词（小写包含匹配）。
DUTY_RATES: dict[str, float] = {
    "apparel": 0.16,  # 服饰鞋帽税率通常偏高
    "shoes": 0.16,
    "footwear": 0.16,
    "electronics": 0.05,
    "phone": 0.05,
    "computer": 0.03,
    "toys": 0.0,
    "books": 0.0,
    "beauty": 0.08,
    "jewelry": 0.11,
    "home": 0.06,
    "kitchen": 0.06,
    "tools": 0.04,
    "sports": 0.07,
    "bags": 0.12,
    "luggage": 0.12,
    "food": 0.10,
}
DEFAULT_DUTY_RATE = 0.07  # 命不中品类时的兜底税率

# 免征额表的口径时点。**静态表、近似值**：真实免征额随政策变动（例：美国 2025 年取消了对全球
# 的 de minimis 豁免，本表已按此设为 0），且各国常有「关税免征额 ≠ 增值税免征额」的双重门槛。
# 本项目不做精细关税对账（见 CLAUDE.md 不覆盖项），此表只求量级正确、够 Agent 给个「大概到手多少」。
DE_MINIMIS_AS_OF = "2025-08"

# 各进口国的免征额（de minimis，以 USD 计）：货价低于此值免关税。
DE_MINIMIS_USD: dict[str, float] = {
    "US": 0.0,  # 2025-08 起取消 de minimis 豁免（此前为 $800）
    "CN": 7.0,  # 行邮税起征点，折合约 50 元人民币
    "JP": 130.0,  # 课税价格 1 万日元
    "KR": 150.0,
    "TW": 60.0,  # NT$2,000
    "SG": 290.0,  # S$400
    "MY": 110.0,  # RM500
    "TH": 42.0,  # 1,500 THB
    "VN": 40.0,  # 1,000,000 VND
    "PH": 175.0,  # 10,000 PHP
    "ID": 3.0,
    "IN": 0.0,  # 实质无免征额
    "CA": 15.0,  # C$20（美加墨协定下更高，此处取一般口径）
    "MX": 50.0,
    "BR": 0.0,  # 无免征额（低值包裹按固定税率征）
    "GB": 170.0,  # £135
    "EU": 165.0,  # €150（关税免征额；增值税无免征，本项目不算 VAT）
    "DE": 165.0,
    "FR": 165.0,
    "AU": 660.0,  # A$1,000
}
DEFAULT_DE_MINIMIS_USD = 100.0

# 自由港 / 无进口关税地区：不论货价多少，关税恒为 0（不是「免征额很高」，是压根不征）。
DUTY_FREE_COUNTRIES: frozenset[str] = frozenset({"HK", "MO"})


class DutyEstimate(BaseModel):
    """关税估算结果。"""

    rate: float
    amount: float
    currency: str
    duty_free: bool
    dest_country: str
    threshold_usd: float  # 本次采用的免征额门槛（供回复解释「为什么免税」）
    domestic: bool = False  # True=国内订单（发货国==收货国），压根不过境，无进口关税


def lookup_duty_rate(category: str) -> float:
    """按品类文本匹配税率（关键词小写包含），命不中用兜底税率。"""
    text = (category or "").lower()
    for keyword, rate in DUTY_RATES.items():
        if keyword in text:
            return rate
    return DEFAULT_DUTY_RATE


def estimate_duty(
    price_base: float,
    category: str = "",
    dest_country: str = "US",
    base: str = "USD",
    platform: str = "",
) -> DutyEstimate:
    """估算关税。``price_base`` 是已折算到 ``base`` 币的货价。

    三种免税情形，按优先级：
    1. **国内订单**（``platform`` 的发货国 == ``dest_country``）——不过境就没有进口关税。
       ``platform`` 留空 / 未知平台时按跨境算（保守，不会漏收该收的税）。
    2. 自由港（HK / MO）——压根不征进口关税。
    3. 跨境但低于进口国免征额（de minimis）。
    """
    country = dest_country.strip().upper()
    rate = lookup_duty_rate(category)

    # 国内订单：关税是**进口**税，发货国和收货国是同一个国家就不存在进口环节。这一条必须在
    # 免征额之前判——否则 amazon(US 发货) → US 收货会被当跨境，按 16% 白收一笔税。
    origin = platform_origin_country(platform)
    if origin and origin == country:
        return DutyEstimate(
            rate=0.0,
            amount=0.0,
            currency=base,
            duty_free=True,
            dest_country=country,
            threshold_usd=0.0,
            domestic=True,
        )

    # 自由港：不征进口关税，与「免征额」无关（threshold 记 0 表示不适用门槛概念）。
    if country in DUTY_FREE_COUNTRIES:
        return DutyEstimate(
            rate=0.0,
            amount=0.0,
            currency=base,
            duty_free=True,
            dest_country=country,
            threshold_usd=0.0,
        )

    threshold = DE_MINIMIS_USD.get(country, DEFAULT_DE_MINIMIS_USD)
    if price_base < threshold:
        return DutyEstimate(
            rate=rate,
            amount=0.0,
            currency=base,
            duty_free=True,
            dest_country=country,
            threshold_usd=threshold,
        )
    return DutyEstimate(
        rate=rate,
        amount=round(price_base * rate, 2),
        currency=base,
        duty_free=False,
        dest_country=country,
        threshold_usd=threshold,
    )
