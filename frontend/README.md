# ShoppingX 前端（React + Vite）

M10 前后端闭环的前端：对话框 → 实时 AGUI 事件流 → 精选商品卡 → 最终清单（可下载）→ 长期偏好面板。
重点不是 UI 细节，而是**怎么消费 AGUI 事件流**和 **connect-first 不丢事件**。

## 跑起来

```bash
# 1) 先起后端（项目根）
uv run uvicorn app.api.server:app --reload --port 8000

# 2) 再起前端
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Vite 把 `/api` 与 `/ws` 反向代理到后端 `:8000`（见 `vite.config.ts`），所以前端代码全用同源相对路径。

## connect-first（关键约定）

后端任务一启动就上报 `session_created` 等早期事件。若按「POST 起任务 → 再连 WS」的顺序，
这些早期事件会因连接还没建好而丢。本前端反过来：

1. 本地生成 `thread_id`（`crypto.randomUUID`）
2. 先连 `WS /ws/{thread_id}`
3. 收到后端 `ws_ready` 控制帧（连接已登记）后，**才** `POST /api/task`

这样第一个事件必落在已登记的连接上，0 缓冲关掉竞态。逻辑都在 `src/hooks/useShoppingXTask.ts`。

## 结构

| 文件 | 作用 |
| --- | --- |
| `hooks/useShoppingXTask.ts` | 任务状态机 + connect-first WS（核心） |
| `components/EventStream.tsx` | AGUI 事件流可视化（含 fork 高亮） |
| `components/ProductCards.tsx` | 商品卡（消费 task_result 的结构化 items） |
| `components/FinalAnswer.tsx` | 最终清单 markdown 渲染 + 产物下载 |
| `components/PreferencePanel.tsx` | 长期偏好面板（`GET /api/preferences/{user_id}`） |
| `api.ts` / `types.ts` | HTTP 封装 / 前后端事件类型 |

## 构建

```bash
npm run build      # tsc -b（strict）+ vite build → dist/
```

## 商品动作：详情 / 对比 / 下单表单 / 确认卡按钮

对着参考项目（globex-agent）前端补的四块交互，共同原则是**前端不新增任何绕过 Agent 的写路径**：

- `ProductDetail.tsx`：商品详情弹窗（大图 / 全部理由 / 收藏 / 搜同款 / 加入对比 / 去下单）。数据就是收尾下发的那份结构化商品，不另外请求。
- `ProductComparison.tsx`：勾选 2～4 件并排看；「让 Agent 帮我比一比」把 item_id 组成一句话发进对话，比价仍由 `price_compare` / `shipping_calc` 在会话内跑。
- `OrderIntentForm.tsx`：收件人 / 地址 / 数量一次填齐后组成一句话发出，模型照旧调 `create_order(confirmed=False)` 出确认卡。收件信息只存 localStorage。
- `OrderCard.tsx`：确认卡上的「确认下单 / 先不下单」只是替用户把那句话发出去；`expires_at` 由后端出卡时给（`app/tools/_order_guard.py` 的 `PREVIEW_TTL_SECONDS`），过期后按钮灰掉、后端也会拒掉过期确认并重新出卡。
- 偏好面板「本次会话」区多了选购摘要（当前需求 / 品类 / 预算 / 槽位），字段来自 `GET /api/session/{tid}/constraints` 与 `session_constraints` 事件。

三个弹窗共用 `Modal.tsx`；平台名 / 理由拆行 / 展示价抽在 `productText.ts`，卡片、详情、对比对同一件商品的解读一致。
