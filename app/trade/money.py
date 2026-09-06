"""金额值对象：**整数最小单位**存储，不用 float。

**为什么不复用检索侧的 float 价格。** `app/recall/fx.py` 那套是给比价用的——把六个平台的报价折算
到 USD 排个序，误差 0.1% 无所谓。订单不一样：它是要落库、要对账、要在「确认卡上写的数」与「库里
存的数」之间逐分对上的。`0.1 + 0.2 != 0.3` 在排序里看不出来，在「三件商品加起来差一分钱」上就是
一张对不平的单。

所以订单侧一律 `amount_minor: int`（分 / 厘的整数）。转换只在边界发生：从检索侧的 float 价格构造
（`from_major`，一次四舍五入定死）、往展示层输出（`to_major`）。中间的加法与乘法全是整数运算。

**minor 倍率不是恒定的 100**：日元没有小数位（1 JPY 就是最小单位），韩元、越南盾同理。拿 100 去
乘日元价格，一单就差 100 倍——这类币种在本仓的商品数据里真实存在（lazada / shopee 的部分报价）。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

from app.recall.fx import FX_TO_USD, UnknownCurrencyError

# 零小数位币种：最小单位就是 1。其余按 100（分）。
# 只列本仓 FX 表里出现的那几个，别照抄 ISO 4217 全表——列了不用的只会让人以为支持它们。
_ZERO_DECIMAL = frozenset({"JPY", "VND", "IDR", "CLP", "COP"})


def minor_factor(currency: str) -> int:
    """该币种 1 个主单位 = 多少最小单位。"""
    return 1 if currency.strip().upper() in _ZERO_DECIMAL else 100


class CurrencyMismatchError(ValueError):
    """两笔不同币种的钱做了加法。"""


@dataclass(frozen=True, slots=True)
class Money:
    """一笔钱：最小单位整数 + 币种。不可变，运算一律产生新对象。"""

    amount_minor: int
    currency: str

    def __post_init__(self) -> None:
        code = self.currency.strip().upper()
        if code not in FX_TO_USD:
            raise UnknownCurrencyError(f"未知币种 {self.currency!r}")
        object.__setattr__(self, "currency", code)

    @classmethod
    def from_major(cls, amount: float | str | Decimal, currency: str) -> Money:
        """从主单位金额构造（检索侧的 float 价格进订单域的**唯一**入口）。

        走 Decimal + ROUND_HALF_UP 而不是 `round()`：后者是银行家舍入（`round(2.675, 2)` 给
        2.67），在钱上没人期待这个行为，而且它先把 float 的二进制误差带了进来。
        """
        factor = minor_factor(currency)
        quantized = (Decimal(str(amount)) * factor).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return cls(int(quantized), currency)

    def to_major(self) -> Decimal:
        """回到主单位（展示 / 序列化用）。返回 Decimal，不返回 float——别在出口处又把误差引回来。"""
        return Decimal(self.amount_minor) / minor_factor(self.currency)

    def add(self, other: Money) -> Money:
        if other.currency != self.currency:
            raise CurrencyMismatchError(f"{self.currency} 不能与 {other.currency} 相加")
        return Money(self.amount_minor + other.amount_minor, self.currency)

    def multiply(self, quantity: int) -> Money:
        """按整数数量相乘（订单行的 单价 × 件数）。

        只接受整数：单价乘小数在订单里没有语义——折扣该是「折后单价」这个独立字段，不是把
        0.85 乘进来再留下一个说不清是怎么来的数。
        """
        if quantity < 0:
            raise ValueError(f"数量不能为负：{quantity}")
        return Money(self.amount_minor * quantity, self.currency)

    def __str__(self) -> str:
        return f"{self.to_major()} {self.currency}"
