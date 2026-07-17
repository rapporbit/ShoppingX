"""汇率归一：把各平台不同币种的价格折算到统一基准币（默认 USD）。

跨平台比价（M4 的 ``price_compare``）的前置：6 个平台报价币种各异（USD/EUR/SGD/MYR…），
不归一就没法比。这里用一张**静态汇率表**做简化模型——本项目不接实时汇率源，也不做
精细汇率对账（见 CLAUDE.md 不覆盖项），表值代表某一时刻的近似中间价，够比价用。

约定：``FX_TO_USD[c]`` = 1 单位币种 ``c`` 值多少 USD。归一到任意基准币时先折到 USD 再折出去。
"""

from __future__ import annotations

# 1 单位该币种 ≈ 多少 USD（近似中间价，静态表）。
FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CNY": 0.14,
    "JPY": 0.0064,
    "SGD": 0.74,
    "MYR": 0.21,
    "PHP": 0.017,
    "THB": 0.028,
    "IDR": 0.000062,
    "VND": 0.000039,
    "BRL": 0.18,
    "INR": 0.012,
    "AUD": 0.66,
    "CAD": 0.73,
    "HKD": 0.128,
    "TWD": 0.031,
    # 拉美三币种：shopee 主力报价币（data/ 里约占 shopee 的 62%），缺则跨平台比价大面积崩。
    # 取某一时刻近似中间价的 USD 量级，够比价用（本项目不接实时汇率源）。
    "MXN": 0.058,
    "CLP": 0.0011,
    "COP": 0.00025,
}


class UnknownCurrencyError(ValueError):
    """汇率表里没有的币种。"""


def _rate_to_usd(currency: str) -> float:
    code = currency.strip().upper()
    if code not in FX_TO_USD:
        raise UnknownCurrencyError(f"未知币种 {currency!r}，请在 FX_TO_USD 中补充。")
    return FX_TO_USD[code]


def to_base(amount: float, currency: str, base: str = "USD") -> float:
    """把 ``amount`` 从 ``currency`` 折算到基准币 ``base``。

    例：``to_base(100, "EUR", "USD")`` → 108.0；``to_base(108, "USD", "EUR")`` → 100.0。
    """
    usd = amount * _rate_to_usd(currency)
    return usd / _rate_to_usd(base)


def to_base_or_none(
    amount: float | None, currency: str, base: str = "USD", ndigits: int = 2
) -> float | None:
    """``to_base`` 的「软」版本：金额缺失或币种未知时返回 ``None``，不抛错。

    跨平台批处理（比价 / 品类聚合）里，一条脏数据不该崩掉整批——折不动就跳过。
    """
    if amount is None:
        return None
    try:
        return round(to_base(amount, currency, base), ndigits)
    except UnknownCurrencyError:
        return None
