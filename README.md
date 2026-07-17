# ShoppingX

**在线 Demo：<https://shopx.oiuu.de>** —— 可直接注册试用（长期偏好按账号跨会话沉淀）。

跨平台交互式购物 Agent。用户用自然语言描述购物意图（如「想买便宜又抗造的旅行三件套，预算 300，
不要塑料的」），系统并行检索不同电商平台、比价、估算到手价（关税 + 运费），输出带选购理由的
清单，并把偏好沉淀为跨会话记忆。

**技术栈**：FastAPI · Qdrant · OpenSearch · Langfuse · AG-UI · WebSocket · React

## 特性

- **意图驱动循环** — Think → Act → Observe → Reflect，由模型判断信息是否足够后自行收尾。
- **并行子 Agent** — 跨平台检索时派生共享工具集的子 Agent 并发执行，上下文隔离；深度、超时、
  结果截断与循环检测在运行时强制约束，避免递归失控。
- **向量召回 + 精排** — 商品侧 Qdrant dense（BGE-M3，1024 维）+ payload filter；品类知识库
  OpenSearch Hybrid（KNN + BM25）+ cross-encoder 精排。
- **分层记忆** — 跨会话长期偏好 + 会话短期状态；会话结束后由记忆管家离线写入与矛盾消解。
- **缓存友好压缩** — 长对话在保住 Prompt Cache 前缀的前提下压缩较旧历史，控制 token 成本。
- **过程治理** — 工具调用前后挂载单步断言、漂移检测、熔断与阶段权限，失败尽早拦下。
- **实时可视化** — AG-UI 事件经 WebSocket 按会话推送，前端可见每一步。
- **离线评测** — 召回三指标 + 端到端动态 Rubric；改 prompt / 工具后可回归对照。

## 架构

```
用户 ──▶ FastAPI ──▶ AgentLoop ──┬──▶ 工具（检索 / 比价 / 运费 / 精挑 / 收尾）
              │                  └──▶ 子 Agent（跨平台并行）
              │
              ├──▶ Qdrant（商品向量）  OpenSearch（品类知识库）
              ├──▶ 记忆 Store（长期偏好）
              └──▶ WebSocket ──▶ React（AG-UI 事件流）
```

主循环经 Hook 管道插入断言、权限与熔断；会话阶段单向推进（规划 → 检索 → 比价 → 收尾），
避免在检索与比价之间来回抖动。

请求形态：`POST /api/task` 立即返回 `thread_id`，Agent 后台执行；前端 `WS /ws/{thread_id}`
订阅事件。信息足够时调用终结工具给出清单，否则继续循环。

## 工具（摘要）

主 Agent 与子 Agent 共用同一工具集。核心能力：

| 工具 | 作用 |
| --- | --- |
| `planner` | 意图结构化（预算 / 品类 / 硬约束 / 软偏好） |
| `item_search` | 单平台商品检索 |
| `category_insight` | 品类洞察（Hybrid RAG + 精排） |
| `item_picker` | 按偏好精挑（硬门 + 关键词 + 语义分，不调 LLM） |
| `price_compare` / `shipping_calc` | 比价（汇率归一）/ 到手价（关税 + 运费） |
| `web_search` / `ask_user` | 外部事实 / 澄清 |
| `forget_preference` | 用户主动撤回长期偏好 |
| `shopping_summary` / `chat_fallback` | 终结：清单收尾 / 非购物闲聊 |
| `dispatch_tool` / `parallel_dispatch_tool` | 派生子 Agent |

跨平台场景会并行 fork：子 Agent 可独立检索，结果截断后合流；深度与迭代有上限（默认子 Agent
不可再 fork，超时与步数硬限制）。

## 快速开始

**前置**：Python 3.11+、uv、Node 18+、Docker

```bash
uv sync
cp .env.example .env          # 填入 LLM / embedding / reranker endpoint

docker compose -f docker/docker-compose.yml up -d   # Qdrant / OpenSearch / Redis

# 可选：全量数据（清洗 + 语料 + 建索引；建索引约数小时）
uv run python scripts/clean_platforms.py
uv run python scripts/sample_rag_amazon.py --all
uv run python scripts/build_item_index.py --require-remote

uv run uvicorn app.api.server:app --reload
cd frontend && npm install && npm run dev
```

- 未配置 `EMBED_MODEL` 时编码器退化为本地确定性实现，可离线跑通（检索质量下降）。
- `--require-remote`：检测到静默退化为本地哈希编码则中止，避免假向量入库。
- embedding / reranker / OpenSearch / Langfuse 均有本地 fallback，远程故障只降级不中断主链路。

```bash
uv run ruff check . && uv run mypy app && uv run pytest   # 927 tests
```

## 配置

完整键见 [`.env.example`](.env.example)：

| 组 | 关键项 |
| --- | --- |
| LLM | `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `LLM_MAIN` / `LLM_JUDGE` / `LLM_FAST` |
| 召回 | `ANN_BACKEND` / `EMBED_DIM` / `RELEVANCE_FLOOR` |
| 知识库 | `OPENSEARCH_*` / `CATEGORY_CARDS_PATH` |
| 记忆与账户 | `DATABASE_URL`（缺省 SQLite `var/globex.db`） |
| 预算 / 压缩 | `RETRIEVAL_BUDGET` / `TOKEN_BUDGET_USD` / `COMPRESS_*` |

## API（核心）

| 端点 | 说明 |
| --- | --- |
| `POST /api/task` | 提交任务，返回 `thread_id` |
| `WS /ws/{thread_id}` | AG-UI 事件流 |
| `POST /api/task/{thread_id}/cancel` | 取消任务 |
| `GET /api/history/{thread_id}` | 会话历史 |
| `GET /api/preferences/{user_id}` | 长期偏好 |
| `POST /api/upload` · `GET /api/files/...` | 会话文件（路径穿越防护） |
| `GET /api/health` · `GET /metrics` | 健康检查 / Prometheus |

## 数据

商品语料来自 Kaggle 公开数据集，原始约 **150 万** 条。清洗后 **1,380,680** 条向量入 Qdrant，覆盖 amazon / shein / walmart / shopee / lazada。

平台规模严重不均（amazon 占绝大多数），无 filter 的 `platform="all"` 几乎只命中 amazon。
跨平台检索对每个平台带 `platform=` filter 并行检索后再合流。

## 评测基准

评测分三层，互不替代：

1. **商品 ANN** — 延迟与存储策略（Qdrant）
2. **品类 RAG** — Recall / MRR / NDCG（Hybrid + 精排）
3. **端到端** — 动态 Rubric（P0 / P1 / P2）

模块变更先过模块指标，再跑端到端。端到端低分时先检查评分器假阳性，再改 Agent。

### 1. 商品召回延迟（Qdrant 存储档位）

单机 Qdrant，**1,380,680** 点 × 1024 维，12 条 query，`top_k=20`。  
冷 = 每条 query 首次命中；热 = 后续重复。

| 存储档位 | 冷 P50 | 冷 P95 | 热 P50 | 热 P95 | recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **向量 + HNSW 全落盘**（当前部署） | 297.2 ms | 400.2 ms | 11.1 ms | 16.8 ms | 基准 |
| HNSW 入内存，向量落盘 | 136.2 ms | 429.0 ms | 12.3 ms | 23.5 ms | 100% |
| 图入内存 + int8 量化常驻 | 25.7 ms | 73.9 ms | 12.1 ms | 27.4 ms | 95.4% |

> 冷查询 n=12，P95 接近 max，关注量级差即可。

### 2. 品类 RAG：Hybrid 粗排 + Cross-Encoder 精排

`category_insight`：OpenSearch Hybrid（默认 **KNN 0.7 + BM25 0.3**）粗排 Top-30，再
BGE-Reranker 精排到 Top-8。粗排已够自信或候选不足时短路跳过 rerank。

| 指标 | 仅 Hybrid 粗排 | 粗排 + Rerank | 说明 |
| --- | ---: | ---: | --- |
| Recall@8 | ~0.62 | ~0.81 | 主要质量收益 |
| MRR | ~0.55 | ~0.74 | 首条更准 |
| 延迟 P50 | ~25 ms | ~75 ms | rerank 约 +50 ms |
| 延迟 P99 | ~80 ms | ~140 ms | 短路命中率约 30% |

P50 增加约 50 ms 相对主循环一轮 Think 可接受。

本仓库 40 条金标（两阶段：品类 resolve + 精确取卡）：

| 指标 | 实测 | 门禁 |
| --- | ---: | ---: |
| Recall@10 | **0.983** | ≥ 0.75 |
| MRR | **0.981** | ≥ 0.65 |
| NDCG@10 | **0.754** | ≥ 0.70 |


```bash
uv run python scripts/eval/run_category_recall.py
```

### 3. 上下文压缩与 Prompt Cache

一次 `item_search` 可占约 **3 万** token；十余轮后上下文易顶窗口。压缩必须与缓存命中一起算：

| 方案 | 压缩率 | 缓存命中率 | 综合成本 |
| --- | ---: | ---: | --- |
| 不压缩 | 0% | 85% | 基准 |
| 盲目压缩（破坏前缀） | 30% | 15% | 更高（约 **+297%**） |
| 缓存友好压缩（断点策略） | 25% | 80% | 最低（约 **-35%**） |

策略：可缓存前缀保持稳定；仅压缩断点之后的较旧轮次，最近 K 轮保留全文。有效 Prompt Cache
通常可降 **50–80%** TTFT、**40–50%** prompt 成本，但要求前缀精确匹配。

### 4. 端到端动态 Rubric

每条 query 动态生成细则：**P0** 一票否决 · **P1** 扣分 · **P2** 质量 1–5 分，由 judge 模型打分。
自动 judge 与人工打标一致性可达 **90%+**。总分：P0 任一失败 → 0；否则
`(p2_avg/5)×100 − 10×P1违规数`；**≥ 70** 视为高分轨迹。种子集约 50 条，作回归基线而非大样本
统计检验。

迭代对照（固定细则缓存，避免尺子抖动）：

| 轮次 | 均分 | 变更 |
| --- | ---: | --- |
| 基线 | 32.8 | — |
| 校准评分器 | **52.8（+61%）** | 轨迹脱敏、评分纪律、意图透传（Agent 代码未改） |
| Agent 改进 | 55.9 | 意图分类、检索预算 |
| 继续改进 | **60.5** | harness 治理、精挑策略 |

常见评分器假阳性：把内部轨迹中的 id 当「泄露」、无预算 query 凭空造价格红线、闲聊意图强加
检索规范——故低分先核尺子。定点缺陷（多跑取中位数）：

| 缺陷 | 基线中位 | 改后中位 |
| --- | ---: | ---: |
| 信息足够仍过度澄清 | 16.7 | 95 |
| 口头收尾未调终结工具 | 10 | 93.3 |

评测 → 定位 bad case → 改 prompt / 工具 / 机制 → 再评测。

```bash
uv run python scripts/eval/run_rubric.py
uv run python scripts/eval/distill_fewshot.py   # 可选：高分轨迹 → 示例
uv run python scripts/eval/evolve_p0.py         # 可选：P0 类 → 防护规则
```

### 5. 过程治理（单步断言 / 漂移检测）

端到端是事后分；过程侧在工具步与多轮中拦截浪费与跑偏：

| 能力 | 指标 | 关闭 | 开启 | 变化 |
| --- | --- | ---: | ---: | ---: |
| 单步验证 | 无效轮次占比 | ~18% | ~5% | -13 pp |
| 单步验证 | 平均请求 token | 48K | 42K | -12% |
| 漂移检测 | 回答与 query 不相关率 | ~12% | ~4% | -8 pp |
| 漂移检测 | 偏好被遗忘率 | ~5% | ~5% | -0 pp |
| 漂移检测 | 10 轮以上长对话 Rubric | 0.68 | 0.76 | +0.08 |


## 项目结构

```
app/
  agent/          主循环、子 Agent、模型路由、预算
  harness/        Hook、断言、漂移、阶段、熔断
  tools/          业务工具 + dispatch
  recall/         向量召回、精排、汇率/关税/运费、品类库
  memory/         长期偏好、记忆管家
  compress/       上下文压缩
  security/       输入过滤、白名单、输出审核、脱敏
  observability/  Langfuse、metrics、日志
  eval/           Rubric 与召回指标
  evolution/      bad case → 规则
  api/            FastAPI、WebSocket、AG-UI
frontend/         React + Vite
scripts/          ETL、索引、评测（scripts/eval/）
tests/            927 tests
```



## 已知限制

- 不接真实平台下单链路：无 OAuth、支付、物流；商品为离线快照。
- 汇率 / 关税 / 运费为查表估算，演示 landed cost，非财务级对账。
- 评测以离线回归为主；种子集规模有限，不是大规模公开榜。


