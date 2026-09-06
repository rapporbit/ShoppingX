"""收货地址值对象。

**不做地址校验**（邮编格式、省市匹配、门牌真实性）：那是一整个行业的活，本仓的交易域是 mock，
编一套半吊子校验只会在演示时把合法地址拒掉。这里只保证两件事——**国家能解析出来**（到手价的
关税运费全看它，见 `recall/geo.py` 的四层解析），以及**收件人与地址行非空**（空地址落库等于
一张永远发不出去的单）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.recall.geo import resolve_dest_country


@dataclass(frozen=True, slots=True)
class Address:
    """收货地址。``country`` 是 ISO 两位码，由 ``recall.geo`` 从自由文本解析。"""

    recipient: str
    line: str
    country: str
    phone: str = ""

    @classmethod
    def parse(cls, recipient: str, line: str, country_hint: str = "", phone: str = "") -> Address:
        """从自由文本构造。``country_hint`` 为空时从地址行里认国家。

        国家解析走与到手价**同一条**通路（`resolve_dest_country`），不另起炉灶：否则同一段
        「寄到日本」在运费那里算 JP、在订单这里存 US，两个数字各自都对、合起来是错的。
        """
        recipient, line = recipient.strip(), line.strip()
        if not recipient or not line:
            raise ValueError("收件人与地址行不能为空")
        country, _ = resolve_dest_country(country_hint or line)
        return cls(recipient=recipient, line=line, country=country, phone=phone.strip())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Address:
        return cls(
            recipient=data["recipient"],
            line=data["line"],
            country=data["country"],
            phone=data.get("phone", ""),
        )

    def masked(self) -> str:
        """给用户看的一行摘要，地址行只留头尾——确认卡与订单列表都用它。"""
        line = self.line if len(self.line) <= 12 else f"{self.line[:6]}…{self.line[-4:]}"
        return f"{self.recipient}｜{line}｜{self.country}"
