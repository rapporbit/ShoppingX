"""收货国解析：从用户原话里**确定性**认出寄往哪个国家。

到手价（landed cost）的关税一腿依赖收货国——免征额（de minimis）按国家差了两个数量级
（US $800 vs CN $7），收货国判错，关税就整条算废。这里和 ``planner.resolve_budget_currency``
是同构问题：**让模型每轮自由猜必然抖动**（「预算 500」曾被轮流猜成 ₹/¥/$），所以走纯规则、
不调模型——同一句话永远解析出同一个国家。

两路匹配，**国名在前、ISO 码在后**：
1. 国名（中英文，大小写不敏感）——但**必须带收货语境**（「寄到日本」「日本收货」），裸国名
   不认。教训是线上真实事故：「推几本**英国**文学作品」被判成寄往 GB，还写进会话 slots 毒化
   了整个会话——商品语境里裸国名极易是产地 / 流派 / 风格修饰（英国文学、日本料理、德国工艺），
   和「宁可漏认、不可错判」同一原则：漏认落到 slots / 长期记忆 / 默认国三层兜底，代价接近零；
   错判则静默毒化整条到手价。
2. 裸 ISO 码（**大小写敏感，只认大写**）——「发 JP」。之所以必须大小写敏感：小写的
   ``in`` / ``id`` / ``ca`` / ``us`` / ``de`` 都是高频英文词，不敏感匹配会把
   「shoes **in** 300 budget」判成寄往印度。ISO 码本身已足够保守，不再叠加语境门控。
"""

from __future__ import annotations

import os
import re

# 用户没写收货国时钉死的默认值（确定性兜底，可经 env 调）。默认中国——与 DEFAULT_BUDGET_CURRENCY
# 的 CNY 同源（面向中文用户）。注意 CN 免征额仅 $7：默认口径下几乎每单都会算出非零关税，
# 因此 `dest_country_assumed` 为真时**必须**在回复里注明假设（见 prompts.yml <constraints>）。
DEFAULT_DEST_COUNTRY = (os.getenv("DEFAULT_DEST_COUNTRY", "CN") or "CN").strip().upper()

# 收货国在会话级 P_t 里的 slot 名（slots 是「单值客观事实、覆盖式 patch」，收货国天然是其中一员）。
# 定义在这个零依赖的纯规则模块里，好让 planner（写）与 curator（沉淀）都能引用它，而不必让
# memory 层反过来 import app.tools（会拖进整个工具包，且有循环 import 风险）。
DEST_COUNTRY_SLOT = "dest_country"

# 支持的收货国（有 de minimis 口径的，见 duty.DE_MINIMIS_USD）。
SUPPORTED_COUNTRIES = "CN,HK,MO,TW,JP,KR,SG,MY,TH,VN,PH,ID,IN,US,CA,MX,BR,GB,DE,FR,AU".split(",")

# 平台 → 发货国（ISO 码）。**关税只在跨境时收**：amazon 从美国发货、用户也在美国收货，那是国内
# 订单，不该有任何进口关税。此前 estimate_duty 只看收货国、不看发货地，把国内单也当跨境征税。
# 对齐 shipping.PLATFORM_REGION 的平台集合（eBay 在清洗阶段已整体剔除，库里没有它的商品）。
# shopee / lazada 是东南亚多国平台，无单一发货国——取新加坡为代表（粗略但比「视为跨境」更接近
# 事实：SEA 平台的用户绝大多数也在 SEA）。表外平台留空 → 一律按跨境算（保守）。
PLATFORM_ORIGIN_COUNTRY: dict[str, str] = {
    "amazon": "US",
    "walmart": "US",
    "shein": "CN",
    "shopee": "SG",
    "lazada": "SG",
}


def platform_origin_country(platform: str) -> str:
    """平台的发货国 ISO 码；未知平台返回空串（调用方按跨境处理）。"""
    return PLATFORM_ORIGIN_COUNTRY.get((platform or "").strip().lower(), "")


# 一、国名（大小写不敏感）。**按特异性排序，第一个命中即返回**：
#   - 「印度尼西亚」必须先于「印度」，否则印尼被判成印度。
#   - 香港 / 澳门 / 台湾单列在「中国」之前：关税口径完全不同（HK 零关税、TW 有自己的免征额），
#     且「香港」不含「中国」二字、不会互相抢，但排前面更稳。
#   - 「美元 / 日元」这类币种词不含「美国 / 日本」，天然不会误命中（币种是另一套表，别互相抢）。
_COUNTRY_NAME_PATTERNS: list[tuple[str, str]] = [
    ("HK", r"香港|港澳|Hong\s?Kong"),
    ("MO", r"澳门|澳門|Macau|Macao"),
    ("TW", r"台湾|台灣|Taiwan"),
    ("ID", r"印尼|印度尼西亚|Indonesia"),  # 必须先于 IN（「印度尼西亚」含「印度」）
    ("IN", r"印度|India"),
    ("CN", r"中国|中國|国内|國內|大陆|大陸|内地|China|Mainland"),
    ("JP", r"日本|Japan"),
    ("KR", r"韩国|韓國|首尔|Korea"),
    ("SG", r"新加坡|狮城|Singapore"),
    ("MY", r"马来西亚|馬來西亞|大马|Malaysia"),
    ("TH", r"泰国|泰國|Thailand"),
    ("VN", r"越南|Vietnam"),
    ("PH", r"菲律宾|菲律賓|Philippines"),
    ("US", r"美国|美國|United\s?States|America"),
    ("CA", r"加拿大|Canada"),
    ("MX", r"墨西哥|Mexico"),
    ("BR", r"巴西|Brazil"),
    ("GB", r"英国|英國|United\s?Kingdom|Britain|England"),
    ("DE", r"德国|德國|Germany"),
    ("FR", r"法国|法國|France"),
    ("AU", r"澳大利亚|澳洲|Australia"),
]

# 二、裸 ISO 码（**大小写敏感，只认大写**，理由见模块 docstring）。
_ISO_CODE_RE = re.compile(r"\b(" + "|".join(SUPPORTED_COUNTRIES) + r")\b")

# 三、收货语境（国名门控）。国名只有紧邻这些语境词才算收货国，消歧靠**紧邻性**而非打分：
# 「我要英国文学作品，寄到中国」——「英国」旁边是「文学」凑不出语境，「寄到中国」一击命中。
#   - 前置：动词在国名前。寄/发/送/运/邮 允许裸接国名（「发中国就行」），方向助词「到/往/去/回」
#     可选。残余风险「发日本风的裙子」误判 JP——罕见（口语多说「日系/和风」），接受。
#   - 后置：名词在国名后。刻意不收单字「收」（「被美国收购的品牌」会误判 US）。
#   - 裸「在」不做语境词：「在英国很流行」立刻误伤回来。但「我在/人在/住在」是人称化的居住地
#     表达（「我在日本」），语义确定指向收货地，单独收进来。
_CTX_BEFORE = (
    r"(?:寄|发|送|运|邮|快递|直邮|海运|空运)\s*(?:到|往|去|回)?"
    r"|收货地(?:址)?(?:在|是|为|：|:)?"
    r"|(?:我|人)\s*(?:现|目前)?在|住在|居住在|常住"
    r"|(?:ship(?:ping)?|deliver(?:y)?|send(?:ing)?|mail(?:ing)?)\s+to"
)
_CTX_AFTER = r"收货|直邮|清关|入关"


def match_country_name(text: str) -> str:
    """无门控的裸国名 / ISO 码匹配，命中返回 ISO 码、否则空串。

    **只给语义已确定是收货地的文本用**（如长期记忆里 ``category="location"`` 的条目
    「常用收货地：中国」）——条目类目本身就是门控，再要求语境词反而会把它漏掉。
    用户自由发言一律走门控版 :func:`resolve_dest_country`。
    """
    for code, pattern in _COUNTRY_NAME_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return code
    m = _ISO_CODE_RE.search(text)
    return m.group(1) if m else ""


def resolve_dest_country(text: str) -> tuple[str, bool]:
    """从用户原话里解析收货国，返回 ``(ISO 码, 是否明示)``。

    国名必须带收货语境（前置动词或后置名词，见 :data:`_CTX_BEFORE` / :data:`_CTX_AFTER`）
    才算「明示」（``True``）；大写 ISO 码单独出现即算。都没命中则落
    :data:`DEFAULT_DEST_COUNTRY` 且「非明示」（``False``）——供回复标注「已按寄往 X 估算」。

    纯规则、不调模型：与 ``planner.resolve_budget_currency`` 同一套确定性范式。
    仍按 :data:`_COUNTRY_NAME_PATTERNS` 的特异性顺序，第一个命中即返回（印尼先于印度）。
    """
    for code, pattern in _COUNTRY_NAME_PATTERNS:
        if re.search(rf"(?:{_CTX_BEFORE})[的\s]*(?:{pattern})", text, re.IGNORECASE) or re.search(
            rf"(?:{pattern})\s*(?:{_CTX_AFTER})", text, re.IGNORECASE
        ):
            return code, True
    m = _ISO_CODE_RE.search(text)  # 大小写敏感：只认 "JP" 不认 "jp"/"in"/"id"
    if m:
        return m.group(1), True
    return DEFAULT_DEST_COUNTRY, False
