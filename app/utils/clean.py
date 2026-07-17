"""平台商品数据清洗——独立前置步骤（5 平台，eBay 已剔除）。

把 ``data/platforms/*.csv`` 各家异构 schema 归一成统一的 :class:`CleanItem`，再过一道
**严格质量闸**（宁缺毋滥）和**平台内去重**，产出可直接喂建索引/比价/精挑的干净商品表。

设计取舍：
- **eBay 不要**——卖家名近全 ``█`` 掩码、描述列全空、库存/销量列空，清洗性价比过低，整体剔除。
- **货币只规整不归一**——大写化 + 修拼写（``GPB→GBP``），不折算 USD；``price_usd`` 留给
  后续接 ``recall/fx`` 时再加（YAGNI）。
- **可用性信号不可靠**——各平台 availability 列填充/口径不一，故只丢「明确缺货」的行
  （shopee ``stock==0`` / walmart 配送自提皆 false / amazon 文案含 unavailable），
  信号缺失（lazada/shein）一律视为在售，不误杀。

纯函数 + 数据结构，不读文件；CSV 读取与落盘在 ``scripts/clean_platforms.py``。
"""

from __future__ import annotations

import html
import json
import re
import unicodedata

import ftfy
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from pydantic import BaseModel

# langdetect 默认带随机性，固定种子让 desc_lang 可复现。
DetectorFactory.seed = 0

# 清洗只覆盖这 5 个平台（eBay 剔除）。
PLATFORMS: tuple[str, ...] = ("amazon", "lazada", "shein", "shopee", "walmart")

# 描述截断长度（过长稀释语义、撑大表体，精挑/选购理由用片段足够）。
DESC_CLIP = 500
# 短于此长度的描述特征太少，语言检测不可靠，留空不猜。
MIN_LANG_LEN = 15
# 价格下限：低于此值基本是占位/垃圾（如 Lazada 的 1e-02=0.01）。跨币种无统一上限
# （IDR/VND 天然是大数），故只设下限。
MIN_PRICE = 0.1
# 货币拼写订正（源数据里见过 GPB 这种 typo）。
_CURRENCY_FIX = {"GPB": "GBP"}


class CleanItem(BaseModel):
    """清洗后的统一商品记录（13 列）。

    7 个必填字段由质量闸保证非空/有效；其余可空，缺失不丢行。
    """

    # —— 必填（质量闸保证）——
    platform: str
    item_id: str
    title: str
    price: float
    currency: str
    image_url: str
    url: str
    # —— 可空保留 ——
    brand: str = ""
    category: str = ""
    rating: float | None = None
    reviews_count: int | None = None
    sold: int | None = None
    description: str = ""
    desc_lang: str = ""
    initial_price: float | None = None


# 各平台 CSV 列名 → 归一字段（按列表顺序取第一个非空）。eBay 已从表中剔除。
PLATFORM_FIELDS: dict[str, dict[str, list[str]]] = {
    "amazon": {
        "item_id": ["asin", "parent_asin"],
        "title": ["title"],
        "brand": ["brand", "manufacturer"],
        "description": ["description"],
        "price": ["final_price", "initial_price"],
        "initial_price": ["initial_price"],
        "currency": ["currency"],
        "rating": ["rating"],
        "reviews_count": ["reviews_count"],
        "sold": ["bought_past_month"],
        "category": ["categories"],
        "url": ["url"],
        "image_url": ["image_url"],
    },
    "lazada": {
        "item_id": ["sku", "mpn"],
        "title": ["title"],
        "brand": ["brand"],
        "description": ["product_description"],
        "price": ["final_price", "initial_price"],
        "initial_price": ["initial_price"],
        "currency": ["currency"],
        "rating": ["rating"],
        "reviews_count": ["reviews"],
        "sold": ["number_sold"],
        "category": ["breadcrumb"],
        "url": ["url"],
        "image_url": ["image"],
    },
    "shein": {
        "item_id": ["product_id"],
        "title": ["product_name"],
        "brand": ["brand"],
        "description": ["description"],
        "price": ["final_price", "initial_price"],
        "initial_price": ["initial_price"],
        "currency": ["currency"],
        "rating": ["rating"],
        "reviews_count": ["reviews_count"],
        "sold": [],
        "category": ["category", "root_category", "category_tree"],
        "url": ["url"],
        "image_url": ["main_image"],
    },
    "shopee": {
        "item_id": ["id"],
        "title": ["title"],
        "brand": ["brand"],
        "description": ["Product Description"],
        "price": ["final_price", "initial_price"],
        "initial_price": ["initial_price"],
        "currency": ["currency"],
        "rating": ["rating"],
        "reviews_count": ["reviews"],
        "sold": ["sold"],
        "category": ["breadcrumb"],
        "url": ["url"],
        "image_url": ["image"],
    },
    "walmart": {
        "item_id": ["product_id", "sku"],
        "title": ["product_name"],
        "brand": ["brand"],
        "description": ["description"],
        "price": ["final_price", "initial_price"],
        "initial_price": ["initial_price"],
        "currency": ["currency"],
        "rating": ["rating"],
        "reviews_count": ["review_count"],
        "sold": [],
        "category": ["category_name"],
        "url": ["url"],
        "image_url": ["main_image"],
    },
}

# 抓第一个数字，含可选小数与**科学计数法指数**（Walmart 96% / Lazada 37% / Shopee 27%
# 的价格形如 2.29e+01，漏掉指数会把 22.9 解析成 2.29，差一个数量级）。
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_WS_RE = re.compile(r"\s+")


def _first_nonempty(row: dict[str, str], cols: list[str]) -> str:
    for col in cols:
        val = (row.get(col) or "").strip()
        if val and val.lower() not in ("null", "none", "nan"):
            return val
    return ""


def _parse_number(raw: str) -> float | None:
    """从脏文本抽第一个数字（去千分位/引号/币符），支持科学计数法。"""
    if not raw:
        return None
    m = _NUM_RE.search(raw.replace(",", "").replace('"', ""))
    return float(m.group()) if m else None


def _parse_int(raw: str) -> int | None:
    val = _parse_number(raw)
    return int(val) if val is not None else None


def _parse_sold(raw: str) -> int | None:
    """销量信号：只保留正整数。``0`` / 空一律视为「无数据」返回 None——

    各平台对「未售出」与「不提供销量」都写成 0/空（如 Shopee 样本 sold 全是 0），
    对人气排序而言两者都是无信息，统一成 None 才不会让「0 销量」污染排序。
    """
    val = _parse_int(raw)
    return val if val and val > 0 else None


def _parse_category(raw: str) -> str:
    """品类可能是 JSON 列表 / 面包屑 / 纯文本，统一成 ``A > B > C``。"""
    if not raw:
        return ""
    text = raw.strip()
    if text.startswith(("[", "{")):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                parts = [str(x) for x in data if isinstance(x, str | int | float)]
                return " > ".join(parts)
        except (json.JSONDecodeError, ValueError):
            pass
    return text


def _norm_currency(raw: str) -> str:
    cur = raw.strip().strip('"').upper()
    return _CURRENCY_FIX.get(cur, cur)


def _norm_title_key(title: str) -> str:
    """去重用的标题归一键：小写 + 折叠空白。"""
    return _WS_RE.sub(" ", title.strip().lower())


_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
# emoji / 图形符 / 箭头 / 国旗 / 变体选择符 / ZWJ / keycap（商品描述 28% 含 emoji）。
_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\U00002600-\U000027bf\U00002300-\U000023ff"
    "\U00002b00-\U00002bff\U00002190-\U000021ff\U0001f1e6-\U0001f1ff"
    "\U0000fe00-\U0000fe0f\U0000200d\U000020e3]"
)
# 装饰符：项目符 / 制表线 / 块元素 / 几何图形。
_DECO_RE = re.compile("[•·▪▫►◄◆◇■□●○★☆※‣⁃∙\U00002500-\U0000257f\U00002580-\U000025ff]")
# 残余控制字符（清洗末段兜底，换行/制表已在空白折叠前转空格）。
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
# 重复标点折叠：3+ 连排压成 1（!!!→! ---→- …但词内单连字符 PK-2 保留）。
_REPEAT_PUNCT_RE = re.compile(r"([!?.•·*~_=-])\1{2,}")
# 中文标点 → 半角，统一中英混用（NFKC 不覆盖 CJK 标点块）。
_CJK_PUNCT = str.maketrans(
    {
        "，": ",",
        "。": ".",
        "、": ",",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "—": "-",
        "～": "~",
        "…": ".",
        "「": " ",
        "」": " ",
        "『": " ",
        "』": " ",
        "《": " ",
        "》": " ",
    }
)


def clean_text(s: str, *, strip_html: bool = True) -> str:
    """商品文本清洗 + 轻量归一（喂 embedding / 落盘描述）。

    覆盖：乱码(mojibake) / HTML 标签实体 / URL / 全角半角 / 中英标点混用 / 全角空格 /
    emoji / 装饰符 / 控制字符 / 重复标点 / 空白混乱，并截到 ``DESC_CLIP``。

    **多语言安全**：只动符号 / emoji / 控制字符 / 标点宽度，不删任何文字、变音符、®©™、数字货币。
    ``strip_html=False`` 跳过标签 / URL / 实体解码（标题 / 品牌用，几乎无 HTML）。**幂等**。
    """
    if not s:
        return ""
    s = ftfy.fix_text(s)  # 0. 乱码/mojibake 修复
    if strip_html:
        s = html.unescape(s)  # 1. HTML 实体（早做，连带解出转义标签）
    s = unicodedata.normalize("NFKC", s)  # 2. 全角→半角；全角空格/nbsp → 普通空格
    if strip_html:
        s = _TAG_RE.sub(" ", s)  # 3. HTML 标签
        s = _URL_RE.sub(" ", s)  # 4. URL
    s = s.translate(_CJK_PUNCT)  # 5. 中文标点 → 半角
    s = _EMOJI_RE.sub(" ", s)  # 6a. emoji
    s = _DECO_RE.sub(" ", s)  # 6b. 装饰符
    s = _REPEAT_PUNCT_RE.sub(r"\1", s)  # 7. 折叠重复标点
    s = _CTRL_RE.sub(" ", s)  # 8. 残余控制字符
    s = _WS_RE.sub(" ", s).strip()  # 9. 空白折叠
    return s[:DESC_CLIP]  # 10. 截断


def detect_lang(text: str) -> str:
    """检测描述语言（ISO 639-1，如 en/es/id）。太短或检测失败返回 ""。

    传入的描述已经 :func:`clean_text` 清洗（剥了标签/URL/emoji），这里直接判定。
    """
    stripped = text.strip()
    if len(stripped) < MIN_LANG_LEN:
        return ""
    try:
        return detect(stripped)
    except LangDetectException:
        return ""


def is_available(platform: str, row: dict[str, str]) -> bool:
    """只在「明确缺货」时返回 False；信号缺失/不可靠一律视为在售（不误杀）。"""
    if platform == "shopee":
        return (row.get("stock") or "").strip() != "0"
    if platform == "walmart":
        deliver = (row.get("available_for_delivery") or "").strip().lower()
        pickup = (row.get("available_for_pickup") or "").strip().lower()
        return not (deliver == "false" and pickup == "false")
    if platform == "amazon":
        return "unavailable" not in (row.get("availability") or "").lower()
    return True  # lazada / shein：无可靠信号，默认在售


def normalize_row(platform: str, row: dict[str, str]) -> dict:
    """把一行原始 CSV 映射 + 解析成归一字段 dict（未过闸，价格可能为 None）。"""
    f = PLATFORM_FIELDS[platform]
    description = clean_text(_first_nonempty(row, f["description"]))
    return {
        "platform": platform,
        "item_id": _first_nonempty(row, f["item_id"]),
        "title": _first_nonempty(row, f["title"]),
        "brand": _first_nonempty(row, f["brand"]),
        "category": _parse_category(_first_nonempty(row, f["category"])),
        "price": _parse_number(_first_nonempty(row, f["price"])),
        "initial_price": _parse_number(_first_nonempty(row, f["initial_price"])),
        "currency": _norm_currency(_first_nonempty(row, f["currency"])),
        "rating": _parse_number(_first_nonempty(row, f["rating"])),
        "reviews_count": _parse_int(_first_nonempty(row, f["reviews_count"])),
        "sold": _parse_sold(_first_nonempty(row, f["sold"])),
        "description": description,
        "desc_lang": detect_lang(description),
        "url": _first_nonempty(row, f["url"]),
        "image_url": _first_nonempty(row, f["image_url"]),
    }


def admit(d: dict) -> tuple[bool, str]:
    """严格质量闸：7 个必填字段缺一不可，价格须 > 下限。返回 ``(是否放行, 原因)``。"""
    if not d["title"]:
        return False, "no_title"
    if not d["item_id"]:
        return False, "no_id"
    price = d["price"]
    if price is None:
        return False, "no_price"
    if price < MIN_PRICE:
        return False, "bad_price"  # <=0 或 1e-02 占位垃圾
    if not d["currency"]:
        return False, "no_currency"
    if not d["image_url"]:
        return False, "no_image"
    if not d["url"]:
        return False, "no_url"
    return True, "ok"


def _completeness(d: dict) -> int:
    """非空可选字段计数——去重撞键时留更完整的那行。"""
    keys = ("brand", "category", "rating", "reviews_count", "sold", "description", "initial_price")
    return sum(1 for k in keys if d.get(k))


def admit_and_dedupe(dicts: list[dict]) -> tuple[list[CleanItem], dict[str, int]]:
    """对已归一的行做质量闸 + 去重，返回 ``(干净记录, 丢弃统计)``。

    去重键 = (归一标题, 价格分桶)，**不含 item_id**——Lazada 近重复刷量 listing 各自 sku
    不同但标题+价格相同，带上 item_id 反而折叠不掉，正是要消的对象；撞键留更完整的一行。

    抽出供 :func:`clean_platform`（CSV 各平台）与 :func:`clean_rag_amazon`（RAG 扩充源）
    共用，保证两条来源走**同一套**入库标准与去重口径。
    """
    stats: dict[str, int] = {}
    best: dict[tuple, dict] = {}
    for d in dicts:
        ok, reason = admit(d)
        if not ok:
            stats[f"drop_{reason}"] = stats.get(f"drop_{reason}", 0) + 1
            continue
        key = (_norm_title_key(d["title"]), round(d["price"], 2))
        prev = best.get(key)
        if prev is None or _completeness(d) > _completeness(prev):
            if prev is not None:
                stats["drop_dup"] = stats.get("drop_dup", 0) + 1
            best[key] = d
        else:
            stats["drop_dup"] = stats.get("drop_dup", 0) + 1
    return [CleanItem(**d) for d in best.values()], stats


def clean_platform(
    platform: str, rows: list[dict[str, str]]
) -> tuple[list[CleanItem], dict[str, int]]:
    """清洗单个平台：可用性 → 归一 → 质量闸 → 平台内去重。返回 ``(干净记录, 统计)``。"""
    stats: dict[str, int] = {"read": len(rows)}
    dicts: list[dict] = []
    for row in rows:
        if not is_available(platform, row):
            stats["drop_unavailable"] = stats.get("drop_unavailable", 0) + 1
            continue
        dicts.append(normalize_row(platform, row))
    items, dstats = admit_and_dedupe(dicts)
    stats.update(dstats)
    stats["kept"] = len(items)
    return items, stats


# —— RAG amazon 扩充源（data/rag/amazon_products.csv）——
# 字段名与 platforms/amazon-products.csv 不同（imgUrl/productURL/stars/boughtInLastMonth），
# 无 brand/description，币种恒 USD（productURL 100% www.amazon.com），故单独映射。


def normalize_rag_amazon_row(row: dict[str, str], cat_lookup: dict[str, str]) -> dict:
    """把 ``amazon_products.csv`` 一行映射成与 :func:`normalize_row` 同构的归一 dict。

    - currency 恒 ``USD``（数据集 productURL 全 www.amazon.com，价格量级亦吻合美元）；
    - category 由 ``category_id`` 经 ``cat_lookup`` 维表（amazon_categories.csv）转名；
    - brand / description 数据集无 → 留空（embed_text 退化为 ``title | 类目``）；
    - sold 取 ``boughtInLastMonth``（过去一月购买量分档），``_parse_sold`` 把 0/空归 None。
    """
    return {
        "platform": "amazon",
        "item_id": (row.get("asin") or "").strip(),
        "title": (row.get("title") or "").strip(),
        "brand": "",
        "category": cat_lookup.get((row.get("category_id") or "").strip(), ""),
        "price": _parse_number(row.get("price") or ""),
        "initial_price": _parse_number(row.get("listPrice") or ""),
        "currency": "USD",
        "rating": _parse_number(row.get("stars") or ""),
        "reviews_count": _parse_int(row.get("reviews") or ""),
        "sold": _parse_sold(row.get("boughtInLastMonth") or ""),
        "description": "",
        "desc_lang": "",
        "url": (row.get("productURL") or "").strip(),
        "image_url": (row.get("imgUrl") or "").strip(),
    }


def clean_rag_amazon(
    rows: list[dict[str, str]], cat_lookup: dict[str, str]
) -> tuple[list[CleanItem], dict[str, int]]:
    """清洗 RAG amazon 抽样行：归一 → 质量闸 → 去重（与 CSV 平台同一套标准）。"""
    stats: dict[str, int] = {"read": len(rows)}
    dicts = [normalize_rag_amazon_row(r, cat_lookup) for r in rows]
    items, dstats = admit_and_dedupe(dicts)
    stats.update(dstats)
    stats["kept"] = len(items)
    return items, stats
