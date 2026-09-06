"""收货地址值对象。

**不做地址校验**（邮编格式、省市匹配、门牌真实性）：那是一整个行业的活，本仓的交易域是 mock，
编一套半吊子校验只会在演示时把合法地址拒掉。这里只保证两件事——**国家能解析出来**（到手价的
关税运费全看它，见 `recall/geo.py` 的四层解析），以及**收件人与地址行非空**（空地址落库等于
一张永远发不出去的单）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from app.recall.geo import DEFAULT_DEST_COUNTRY, match_country_name


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

        国家解析仍走 `recall/geo.py` 这**同一张国名表**，不另起炉灶：否则同一段「日本」在运费
        那里算 JP、在订单这里存 US，两个数字各自都对、合起来是错的。但走的是**无门控版**
        (:func:`match_country_name`) 而非自由发言用的 :func:`resolve_dest_country`——后者要求
        国名紧邻「寄到 / ship to」这类收货语境词（那是为了不把「英国文学」判成寄往英国），而
        地址行天生是纯地名（「Tokyo, Japan」「日本东京」），永远凑不出语境词，于是**全部静默
        落默认 CN**：订单存错收货国还不报错。地址行的语义本身就是门控，不该再要求语境词。

        两个入口分开走：``country_hint`` 是专门的国家字段，「JP」这样的裸码要认；``line`` 是
        整段地址文本，**关掉裸码**避开美国州缩写撞车（见 `match_country_name` 的说明）。
        hint 给了却认不出（模型偶尔传「Japan 东京都」这类杂串或干脆写错）时**回落到地址行**再
        认一次，而不是直接吃默认国——地址行里往往正明写着国家。两路都不中才落
        :data:`DEFAULT_DEST_COUNTRY`。
        """
        recipient, line = recipient.strip(), line.strip()
        if not recipient or not line:
            raise ValueError("收件人与地址行不能为空")
        hint = (country_hint or "").strip()
        country = match_country_name(hint) if hint else ""
        if not country:
            country = match_country_name(line, allow_iso_code=False)
        return cls(
            recipient=recipient,
            line=line,
            country=country or DEFAULT_DEST_COUNTRY,
            phone=phone.strip(),
        )

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
