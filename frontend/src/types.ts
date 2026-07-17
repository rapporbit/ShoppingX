// 前后端唯一约定：AGUI 事件结构。后端 monitor.py 每条事件都是这个信封，
// 前端只看 event 字段分发、看 data 取业务字段（见 app/api/monitor.py）。

import type { PrefDomain } from "./domains";

// 域枚举定义在 domains.ts（那里还有中文标签 / 下拉顺序），这里转出一手，组件按需从任一处取。
export type { PrefDomain };

export type AguiEvent = {
  type: "monitor_event";
  event:
    | "session_created"
    | "assistant_call"
    | "tool_start"
    | "tool_end"
    | "fork"
    | "queue_status"
    // curator 本轮沉淀了新长期偏好（data.preferences: [{content, dedup_key}]）——回复下方画一行
    // 「记住了 … ✕」，✕ 即 DELETE 掉那条。自动写入，但看得见、撤得掉。
    | "memory_updated"
    // 本轮**读取侧**用到了哪些长期记忆（data: {domains, excluded, attenuated}）——思考过程里画一行
    // 「按你的长期偏好：排除 N 项、降权 M 项」。记忆最危险的失败是静默的：一条偏好误杀了一批商品，
    // 用户只会觉得「怎么老是搜不出东西」，且归因不到记忆头上。这一行就是解药。
    | "memory_applied"
    | "task_result"
    | "task_cancelled"
    | "error"
    | "clarification_request"
    // 收尾文案的流式增量（data.text: 累计全文）：shopping_summary 边生成边推，任务还在跑时
    // 逐字渲染清单文案（感知延迟优化）。瞬态事件：不进活动流回看、不参与断线补发。
    | "summary_delta"
    // 商品卡先出货（data.items: ProductItem[]）：item_picker 一定稿就推，不等收尾文案生成。
    // 与 summary_delta 是同一条思路的两半——那个管文案、这个管卡片。收尾的 task_result 会用
    // 定稿那批原样覆盖（两者同构）。不进活动流（它是结果本身，不是一行「思考」）。
    | "items_preview"
    // 会话级 P_t 约束快照（data: SessionSnapshot）：planner 每轮落 P_t 后推，偏好面板「本次
    // 会话」区据此实时刷新。瞬态：断线重连后面板走 GET /api/session/{tid}/constraints 主动拉。
    | "session_constraints";
  message: string;
  data: Record<string, unknown>;
  thread_id: string | null;
  timestamp: string;
  // Redis Stream id（D 块事件回放）：断线重连时前端把它当 last_event_id 上送补发缺口、并据此去重。
  // 后端 Redis 降级时无此字段（退回纯直播、无回放）。
  id?: string;
};

// 一件商品：shopping_summary 随 task_result 下发的收尾精选（带选购理由）。
// landed_usd 只在本轮真跑过 shipping_calc 时才有——planner 按意图判 tasks，用户没问到手价就不算，
// 那时只有 price_usd（货价）。卡片据此照实标注，不把货价冒充成到手价。
export type ProductItem = {
  item_id: string;
  platform: string;
  title: string;
  landed_usd?: number | null; // 到手价（含税运）。本轮没跑 shipping_calc 则为空。
  price_usd?: number | null; // 货价（未含税运）。没有 landed_usd 时显示它。
  reason?: string;
  image_url?: string; // 商品图 URL（来自离线数据集），缺失 / 加载失败时卡片回退渐变占位。
  url?: string; // 平台商品页 URL，点击卡片新标签页打开；缺失则卡片不可点。
  score?: number; // 仅「搜同款」的近邻结果带：与源商品的向量相似度（0~1）。
  // 套装槽位名（「一套齐」轮才非空）：任一卡带 slot 即按槽分组渲染（组头 = 槽名 + 该槽花费），
  // 代替平台胶囊筛选。走结构化字段而非理由文案的【槽名】前缀（前缀会被收尾 LLM 重写时丢掉）。
  slot?: string;
};

// 本轮全树（主 + 各 fork 子 Agent）token 用量。随 task_result 事件下发、随 turns.json 落盘回看。
// total = input + output（计费口径）；cost_usd 是本轮估算成本（F 块 FinOps 记账）。
// cache_read = 命中前缀缓存折扣档的 input；cache_hit_rate = cache_read/input（0~1），
// 是「压缩 + cache breakpoint 有没有真生效」的健康指标。旧数据可能缺这两字段，故可选。
export type TurnTokens = {
  input: number;
  output: number;
  total: number;
  cost_usd: number;
  cache_read?: number;
  cache_hit_rate?: number;
};

// 长期偏好（GET /api/preferences/{user_id} 返回）。
// dedup_key 是删除时的 handle；source 区分「Agent 学到的」和「你手填的」。
//
// blocking = 硬淘汰权：命中即把商品从结果里删掉，且用户看不到被删了什么。所以它**只由用户显式
// 授予**——后端在唯一落库口上强制 source="agent" 的一律 false，Agent 学到的 dislike 只减分。
// last_confirmed_at 取代了原来的 recency_weight：后端删掉了半衰期衰减，这个时间戳**不参与任何
// 打分**，只供 UI 提示「这条很久没用过了」。注意后端 SQLite 存的是 naive datetime，ISO 串没有
// 时区后缀 —— 按 UTC 解析，别当本地时间。
export type Preference = {
  dedup_key: string;
  content: string;
  category: string;
  polarity: "like" | "dislike";
  blocking: boolean;
  domain: PrefDomain;
  slug: string;
  keywords: string[];
  source: "agent" | "user";
  created_at: string;
  last_confirmed_at: string;
};

// 一条偏好的可编辑结构（POST /parse 的返回、POST / PUT 的请求体）。
// 与 Preference 的差别：没有 dedup_key（它由 polarity/category/domain/slug 派生）、没有元数据
// （source / created_at / last_confirmed_at 由后端决定，不由用户填）。
export type PrefDraft = {
  content: string;
  category: string;
  domain: PrefDomain;
  slug: string;
  polarity: "like" | "dislike";
  blocking: boolean;
  keywords: string[];
};

// memory_updated 事件里的一条：回复下方那行「记住了 …」，✕ 用 dedup_key 删。
export type LearnedPref = {
  content: string;
  dedup_key: string;
};

// 会话级 P_t 约束（session_constraints 事件 / GET /api/session/{tid}/constraints）。
// id 是后端发的跨轮身份（c1/c2/…），删除按它打 DELETE；source_quote 让用户看懂是自己哪句话。
export type SessionConstraint = {
  id: string;
  content: string;
  source_quote: string;
  polarity: "like" | "dislike";
  blocking: boolean;
};

export type SessionSnapshot = {
  epoch: number;
  budget_usd: number | null;
  category: string;
  constraints: SessionConstraint[];
};

// GET /api/history/{tid} 返回的一条逐轮对话（后端 turns.json 累加的精简 user→assistant 对）。
// assistant 轮额外带回看专用字段：items（精选商品卡）、activity（思考过程 AGUI 事件流），
// 让历史回看也能还原商品卡与「思考过程」折叠区，而非只剩结论文本。无则字段缺省（闲聊 / 旧数据）。
export type HistoryTurn = {
  role: "user" | "assistant";
  content: string;
  // 本轮参考图的文件名（只挂 user 轮）。存名不存图：图本体在服务端 uploaded/<thread_id>/ 下，
  // 前端拿名去 GET /api/uploads 取——这样回看旧会话时图还在，而不是随刷新蒸发的 blob。
  images?: string[];
  items?: ProductItem[];
  activity?: AguiEvent[];
  elapsed_ms?: number; // 本轮总耗时（毫秒），前端在该轮右下角显示「用时」。
  tokens?: TurnTokens; // 本轮 token 用量，前端在该轮右下角与「用时」并排显示「token 消耗」。
};

// 侧栏历史列表的一条会话。后端按 threadId 存逐轮对话，但「有哪些会话」这层索引后端没有，
// 故纯前端维护：threadId 关联后端历史，title 取首轮 query，updatedAt 用于按最近活跃排序。
export type SessionMeta = {
  threadId: string;
  title: string;
  updatedAt: number;
};

// 后台管理页面的一个可调参数。**整份表单由后端 /api/admin/config 的响应驱动**——前端不硬编码
// 任何参数名、范围或默认值，后端 registry 加一项，页面自动多一项。value/default 的实际类型由
// kind 决定（int/float → number，str → string，bool → boolean）。
export type AdminParam = {
  key: string;
  group: string;
  label: string;
  kind: "int" | "float" | "str" | "bool";
  value: number | string | boolean;
  default: number | string | boolean;
  // override=后台改过的 / env=.env 配的 / default=代码默认值。用于在 UI 上标出「这条被改过」。
  source: "override" | "env" | "default";
  help: string;
  // 非空即在 UI 上显示醒目告警：标定证伪 / 未标定 / 调错会做反推荐的参数。
  warning: string;
  minimum: number | null;
  maximum: number | null;
  allow_empty: boolean;
  // 密钥类（API key）：value 恒为空串（后端永不回显），只给 masked 供核对。留空提交 = 不改。
  secret: boolean;
  masked: string;
};

export type AdminConfig = {
  groups: Record<string, { label: string; desc: string }>;
  params: AdminParam[];
};
