# 数据集报告（`data/`）

> 生成日期：2026-06-24
> 范围：`data/` 目录下全部数据文件

## 1. 总览

`data/` 目录包含两类数据，按子目录组织：

| 子目录 | 定位 | 文件数 | 总体量（约） |
|--------|------|--------|--------------|
| `platforms/` | 多电商平台**商品采集数据**（每平台抓取样本，字段贴近原始爬取结果） | 6 个 CSV | ~36 MB |
| `rag/` | RAG / 检索增强用的**商品语料与分类体系（taxonomy）** | 6 个文件 | ~560 MB |

目录结构：

```
data/
├── platforms/                # 6 大电商平台商品样本（按平台分文件）
│   ├── amazon-products.csv
│   ├── ebay-products.csv
│   ├── lazada-products.csv
│   ├── shein-products.csv
│   ├── shopee-products.csv
│   └── walmart-products.csv
└── rag/                      # 商品大语料 + 分类体系 + 品类知识库
    ├── amazon_products.csv                       # 142 万条商品
    ├── amazon_categories.csv                     # 类目字典
    ├── amazon-all-categories-bestsellers.csv     # 各类目畅销榜
    ├── milistu-amazon-products-2023-slim.parquet # 2023 商品元数据（列式）
    ├── category_cards.jsonl                      # 品类知识库卡片（2044 张，category_insight 数据源）
    ├── google-product-taxonomy.txt               # Google 商品分类树
    └── shopify-taxonomy.json                      # Shopify 商品分类树
```

---

## 2. `platforms/` —— 多平台商品采集数据

每个文件对应一个电商平台，字段直接来自抓取结果，**各平台 schema 不统一**（列名、粒度、嵌套 JSON 字段各异），使用前通常需要做字段对齐。

每个文件**均为正好 1,000 条记录**（用 CSV 解析器核实；`wc -l` 物理行数会因字段内嵌换行而虚高，见下表对比）。

| 文件 | 平台 | 记录数 | 物理行数(`wc -l`) | 大小 | 列数 | 特点 |
|------|------|--------|------------------|------|------|------|
| `amazon-products.csv` | Amazon | 1,000 | 1,002 | 5.8 MB | 55 | 字段最丰富：BSR 排名、buybox、变体、评论、退货等 |
| `ebay-products.csv` | eBay | 1,000 | 1,001 | 15 MB | 38 | 含卖家评分/评价、运费/退货、相关商品（多为长 JSON） |
| `lazada-products.csv` | Lazada | 1,000 | 1,001 | 2.3 MB | 29 | 东南亚市场，含 `number_sold`、`gmv`、超级卖家标识 |
| `shein-products.csv` | SHEIN | 1,000 | 1,001 | 2.4 MB | 28 | 服饰为主，含颜色/尺码/可选尺码、`offers` |
| `shopee-products.csv` | Shopee | 1,000 | **22,412** | 3.1 MB | 37 | 含 `sold`、`flash_sale`、`gmv_cal`、卖家响应率；描述字段内嵌大量换行 |
| `walmart-products.csv` | Walmart | 1,000 | 1,012 | 7.7 MB | 44 | 含 GTIN/UPC、货架（aisle）、单价、成分等 |

### 通用字段（跨平台大致可对齐的概念）

- **标识**：`url` / `product_id` / `asin` / `sku` / `id`
- **基本信息**：`title` / `product_name`、`brand`、`description`、`images`
- **价格**：`initial_price`（原价）、`final_price`（现价）、`currency`、`discount`
- **销量/库存**：`sold` / `number_sold` / `available_count` / `stock` / `in_stock`
- **评价**：`rating` / `rating_stars`、`reviews_count` / `review_count`、`top_reviews`
- **卖家**：`seller_name`、`seller_rating`、`seller_id`
- **分类**：`categories` / `breadcrumbs` / `category_tree` / `root_category`

### 各平台代表性专有字段

- **Amazon**：`root_bs_rank` / `bs_rank`（Best Sellers Rank）、`buybox_seller`、`number_of_sellers`、`buybox_prices`、`variations`、`amazon_choice`、`bought_past_month`、`country_of_origin`
- **eBay**：`condition`（新旧）、`sold_count`、`ships_to` / `excludes_shipping`、`return_policy`、`payment_details`、`related_sponsored_items`
- **Lazada**：`lazmall`、`is_super_seller`、`seller_ship_on_time`、`seller_chat_response`、`gmv`
- **SHEIN**：`color` / `size` / `all_available_sizes`、`offers`、`category_tree`、`country_code`
- **Shopee**：`favorite`（收藏数）、`flash_sale` / `flash_sale_time`、`vouchers`、`seller_followers`、`seller_chats_responded_percentage`、`gmv_cal`
- **Walmart**：`gtin` / `upc`、`aisle`、`unit_price` / `unit`、`free_returns`、`available_for_delivery` / `available_for_pickup`、`ingredients`

> 备注：多数 schema 列（如 `categories`、`variations`、`specifications`、`top_reviews`）实际存储的是**序列化后的 JSON / 列表字符串**，解析时需二次反序列化。

---

## 3. `rag/` —— 商品语料与分类体系

供检索增强（RAG）、类目映射、向量化召回等用途。

| 文件 | 内容 | 规模 | 格式 |
|------|------|------|------|
| `amazon_products.csv` | Amazon 商品主表（精简字段，适合大规模检索） | **1,426,337** 行 / 359 MB | CSV |
| `amazon_categories.csv` | 类目 ID → 名称字典 | 248 行 / 6.7 KB | CSV |
| `amazon-all-categories-bestsellers.csv` | 各类目畅销榜（含评论列表） | 1,316 行 / 10 MB | CSV |
| `milistu-amazon-products-2023-slim.parquet` | 2023 年 Amazon 商品元数据（列式、含描述/特性） | 89 MB | Parquet |
| `google-product-taxonomy.txt` | Google 官方商品分类树（2021-09-21 版） | 5,595 类目 / 472 KB | 文本 |
| `shopify-taxonomy.json` | Shopify 商品分类体系（2026-08-unstable 版） | 90 MB | JSON |
| `category_cards.jsonl` | **品类知识库卡片**（`category_insight` 工具的数据源） | **2,044** 张 / ~45 MB | JSONL |

### 3.1 `amazon_products.csv`（142 万条，核心检索语料）

列：`asin, title, imgUrl, productURL, stars, reviews, price, listPrice, category_id, isBestSeller, boughtInLastMonth`

- 通过 `category_id` 关联 **`amazon_categories.csv`** 的 `id` 字段获得类目名称。
- 字段精简、行数大，是做向量召回 / 关键词检索的主语料。

### 3.2 `amazon_categories.csv`（类目字典）

列：`id, category_name`，共 248 个类目（如 `Beading & Jewelry Making`、`Fabric Decorating`…），作为 `amazon_products.csv` 的维表。

### 3.3 `amazon-all-categories-bestsellers.csv`（畅销榜）

列：`,product_name, category, categoryRank, noRatings, cost, REVIEWLIST, product_url`

- 第一列为无名索引列；`categoryRank` 形如 `#1`；`REVIEWLIST` 为序列化的评论对象列表（含评论正文、地区、日期），体量较大。

### 3.4 `milistu-amazon-products-2023-slim.parquet`

推断列（来自文件元数据）：`main_category, title, average_rating, rating_number, features, description, price, images, categories, details, parent_asin, store, date_first_available`。

- 对应公开数据集 **milistu / Amazon-Products-2023**（McAuley Amazon Reviews 2023 商品元数据的精简版）。
- 含富文本（`description` / `features` / `details`），适合做 RAG 文档切分与嵌入。
- 列式存储，建议用 `duckdb` / `pyarrow` / `pandas` 读取（当前机器未安装这些工具，需先安装）。

### 3.5 分类体系（taxonomy）

- **`google-product-taxonomy.txt`**：每行 `ID - A > B > C` 形式的层级路径，共 5,595 条，版本 2021-09-21，可用于标准化类目映射。
- **`shopify-taxonomy.json`**：结构化 JSON，`version: 2026-08-unstable`，按 `verticals` → `categories` 组织，每个节点有 `id`（`gid://shopify/TaxonomyCategory/...`）、`level`、`name`、`full_name`，适合做层级类目树与跨平台类目对齐。

### 3.6 `category_cards.jsonl`（品类知识库）

`category_insight` 工具的数据源。247 个品类、2,044 张结构化卡片。每行一张卡，schema 见 `app/recall/category_kb.py:CategoryCard`。

**卡片类型与数据来源：**

| card_type | 数量 | 数据来源 | 内容 |
|---|---|---|---|
| `attribute`（LLM 多维度） | 1,062 | **LLM 生成**（`scripts/etl/llm_attributes.py`） | 每品类 3-5 个选购维度的典型分布，如 `"Material: Nylon 40% / Polyester 25% / Canvas 15%"` |
| `attribute`（评分分布） | 247 | 真实数据聚合（`scripts/etl/aggregate.py`） | 评分档位分布，如 `"评分分布：4.5★+ 58% / 4.0–4.5★ 23% / ..."` |
| `price_range` | 247 | 真实数据聚合（`scripts/etl/aggregate.py`） | p5–p95 分位切档：budget / mid / premium 三档 USD 区间 |
| `attribute_schema` | 246 | Shopify 分类法（`scripts/etl/shopify_attrs.py`） | 该品类的选购维度名 + 典型取值列表（无分布比例） |
| `bestseller` | 242 | 真实数据聚合（`scripts/etl/aggregate.py`） | 月销头部热卖款名称 / 价格 / 评分 / 月销量 |

**为什么 attribute 卡有两种来源：** refdocs 设计的 attribute 卡应是多维度属性分布（如「材质：尼龙 60% / 帆布 25%」），但真实商品数据集缺少结构化属性字段（只有标题和价格），仅能算出评分分布。LLM 生成卡补上了这个缺口——数值是近似的（directional accuracy），足以给 `item_picker` 提供「这个品类什么属性主流」的锚点。LLM 生成卡的 `confidence` 固定 0.6（低于真实数据卡的 0.5–0.95），`raw_evidence` 标注 `"LLM-generated"`。

**构建命令：** `uv run python scripts/build_category_kb.py`（全量重建；需 `.env` 配置 LLM API + Embedding API）。

**每张卡的字段：**

| 字段 | 说明 |
|---|---|
| `card_id` | 唯一标识，格式 `{品类slug}_{card_type}` |
| `category` | 归一品类名（经 `normalize_category` 处理） |
| `card_type` | `bestseller` / `attribute` / `price_range` / `attribute_schema` |
| `summary` | 已提炼的一段结论（有格式约定，下游 parser 靠此解析） |
| `raw_evidence` | 支撑结论的 1-3 段原始证据（不回传给模型） |
| `confidence` | 0-1 置信度（真实数据 0.5-0.95；LLM 生成 0.6） |
| `content_vector` | 1024 维 BGE-M3 稠密向量（占文件体积 ~95%，仅供本地回退检索） |

---

## 4. 数据质量与使用注意

1. **Schema 不统一**：`platforms/` 下 6 个平台列名各异，跨平台分析前需建立统一字段映射（价格、销量、评分、类目）。
2. **嵌套字段**：大量列是序列化的 JSON / 列表字符串（`variations`、`specifications`、`top_reviews`、`category_tree`、`REVIEWLIST` 等），需二次解析。
3. **价格字段含噪**：可见 `"null"` 字符串、带引号嵌套（如 `"""57.79"""`）等，清洗时注意类型与异常值。
4. **部分字段疑似脱敏**：样本中出现卖家名被掩码（如 `Orv███tor███`），统计卖家维度时需留意。
5. **物理行数 ≠ 记录数**：6 个平台均为 1,000 条记录；Shopee 因描述字段内嵌大量换行，`wc -l` 高达 2.2 万行，必须用 CSV 解析器（正确处理引号内换行）而非按行切分来读取。
6. **大文件需专用工具**：`amazon_products.csv`（359 MB）、两个 90 MB 级文件建议用 DuckDB / 流式读取，避免一次性载入内存；当前环境暂无 `pandas`/`pyarrow`/`duckdb`。
7. **关联关系**：`rag/amazon_products.csv.category_id` ↔ `rag/amazon_categories.csv.id` 为唯一显式可连接的主外键关系。

---

## 5. 典型用途建议

- **`platforms/`**：跨平台商品/价格/销量对比、卖家分析、竞品与选品分析。
- **`rag/`**：构建商品检索与问答的 RAG 知识库（主语料 = `amazon_products.csv` + parquet 富文本），并以 Google / Shopify taxonomy 做类目标准化与跨源对齐。
