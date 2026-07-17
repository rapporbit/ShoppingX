# 清洗后平台商品数据集（`data/platforms/clean/`）

> 生成方式：`uv run python scripts/clean_platforms.py`
> 产物：
> - `products.jsonl` —— 合表（一行一条商品，JSON），通用加载入口。
> - `by_platform/{platform}.jsonl` —— 分平台表，对齐召回层「按平台分库」；`scripts/build_item_index.py` 从这批分平台表起步建 **Qdrant hybrid 索引**（dense BGE-M3 + 稀疏 BM25）。
>
> 本目录在 `data/` 下，**已 gitignore，靠脚本复现，不入库**。两份产物同源同次生成，行数一致（合表 = 各平台之和）。

## 1. 这是什么

把 `data/platforms/` 下 **5 个平台**的原始爬取 CSV（Amazon / Lazada / SHEIN / Shopee / Walmart）洗成**统一 schema、过严格质量闸、平台内去重**后的干净商品表。直接喂给召回建索引（`scripts/build_item_index.py`）、比价、精挑等下游。

清洗逻辑在 `app/utils/clean.py`（纯函数，可测）；本数据集是它的产物。

> **eBay 已整体剔除**：原始 eBay 样本卖家名近全 `█` 掩码、描述列全空、库存/销量列空，清洗性价比过低，不纳入。

## 2. 规模与留存

| 平台 | 原始 | 保留 | 留存率 | 主要丢弃 |
|------|----:|----:|----:|------|
| Amazon | 1000 | 995 | 100% | 5 无图 |
| SHEIN | 1000 | 979 | 98% | 21 重复 |
| Walmart | 1000 | 969 | 97% | 9 缺货 · 22 重复 |
| Shopee | 1000 | 878 | 88% | 121 `stock=0` 缺货 · 1 重复 |
| Lazada | 1000 | 606 | 61% | 56 `1e-02` 垃圾价 · 338 近重复刷量 listing |
| **合计** | **5000** | **4427** | | |

## 3. Schema（15 列）

### 必填（质量闸保证非空/有效）

| 列 | 类型 | 说明 |
|----|------|------|
| `platform` | str | 平台：`amazon`/`lazada`/`shein`/`shopee`/`walmart` |
| `item_id` | str | 平台内唯一主键（asin/sku/id/product_id）。**全表 0 重复** |
| `title` | str | 商品标题（召回编码主字段 + 展示） |
| `price` | float | 现价，**原币种**，已规整（剥币符/千分位、解科学计数法）。`> 0.1` |
| `currency` | str | 原币种 ISO 码，已大写 + 订正拼写（`GPB→GBP`） |
| `image_url` | str | 主图 URL |
| `url` | str | 商品页 URL（溯源/跳转） |

### 可空保留（缺失不丢行）

| 列 | 类型 | 填充率 | 说明 |
|----|------|------:|------|
| `brand` | str | 84% | 品牌（Shopee 仅 ~22%，故可空） |
| `category` | str | 100% | 归一面包屑 `A > B > C` |
| `rating` | float? | ~64%* | 评分 0~5（`*` 其余为 0 = 无评分） |
| `reviews_count` | int? | ~54%* | 评论数（`*` 其余为 0） |
| `sold` | int? | 16% | 销量人气，**仅正整数**。见 §5 注意 |
| `description` | str | ~100% | 商品描述，**已内容清洗**（去 HTML/emoji/乱码、全角转半角、统一标点）+ 截断 ≤500 字符。见 §4 第 4 条。7 条空描述 |
| `desc_lang` | str | 99% | 描述语言（ISO 639-1）。见 §5 |
| `initial_price` | float? | 94% | 原价（算折扣/省了多少）。原币种 |

## 4. 清洗规则

1. **字段映射**：各平台异构列名 → 统一字段（`app/utils/clean.py:PLATFORM_FIELDS`）。
2. **数值归一**：价格剥 `$`/引号/千分位，**解科学计数法**（Walmart 96% / Lazada 37% / Shopee 27% 的价格形如 `2.29e+01`，漏指数会差一个数量级）。
3. **品类归一**：JSON 列表 / 面包屑 / 纯文本 → 统一 `A > B > C`。
4. **描述文本清洗**：去 HTML 标签 / 实体 / URL，修乱码（ftfy mojibake），全角转半角（NFKC），中文标点归一为半角，清 **emoji（原始 28% 含）** / 装饰符 / 控制字符，折叠重复标点与混乱空白（39% 含连续空格/换行）。铁律 **只清符号、不删文字**——泰文/中文/变音符（café 的 é）/®©™ 一律原样保留，多语言安全。逻辑在 `app/utils/clean.py:clean_text`，同结果喂 `desc_lang` 检测。
5. **质量闸（宁缺毋滥）**：7 个必填字段缺一即丢；价格须 `> 0.1`（剔 `1e-02` 占位垃圾）。
6. **可用性过滤**：只丢**明确缺货**（Shopee `stock=0`、Walmart 配送自提皆 false、Amazon 文案含 unavailable）；信号缺失的平台默认在售，不误杀。
7. **平台内去重**：按 `(归一标题, 价格)` 折叠，留最完整一行。

## 5. 使用注意（诚实标注）

- **`price` 暂不可跨平台直接比**：保留 14 种原币种（USD 2910、MYR 430、IDR 247、MXN 217…），**未折算 USD**（货币归一延后给 `recall/fx`）。跨平台比价前必须先折算，否则会把 IDR 标价当 USD 比错。`price_usd` 列待 fx 接入后补。
- **`sold` 只对 Lazada + Amazon 有效**：Lazada 483 条有效（最大 112057，信号最强）、Amazon 204 条（来自 `bought_past_month`）。**Shopee/SHEIN/Walmart 样本不提供销量**，`sold` 全为 `None`——这是源数据决定的，不是清洗丢的；`0`/空一律按「无数据」置 `None`，避免污染人气排序。
- **`desc_lang`：约 21% 描述为非英语**（`es` 520、`id` 166、`th` 73、`vi` 55…），集中在 **Shopee（西语/印尼/泰/越，约 88% 非英）与 Lazada**；Amazon/SHEIN/Walmart 近乎全英。召回用的 BGE-M3 是跨语言模型不受影响，但给用户的最终文案可据此决定是否翻译。
- **去重会折叠颜色变体**：Lazada 同标题同价的不同颜色款（如某手机壳 56 个色）被折成 1 张代表卡——对召回去冗余是有意为之，代价是不保留 `color` 变体信息。
- **Amazon 含少量他站泄漏**：31 条 INR（amazon.in）+ 2 条 GBP（amazon.co.uk），随货币归一延后保留。
- **`rating`/`reviews_count` 的 0 即无评价**：两列对所有行都有数值，但约 36% / 46% 为 `0`，表示该商品无评分/评论，并非缺字段。

## 6. 加载示例

```python
import json
rows = [json.loads(line) for line in open("data/platforms/clean/products.jsonl", encoding="utf-8")]
# 只看有销量信号的 Lazada 热卖
hot = [r for r in rows if r["platform"] == "lazada" and r.get("sold")]
hot.sort(key=lambda r: r["sold"], reverse=True)
```

对应 Pydantic 模型：`app.utils.clean.CleanItem`。
