// 前后端唯一约定：AGUI 事件结构。后端 monitor.py 每条事件都是这个信封，
// 前端只看 event 字段分发、看 data 取业务字段（见 app/api/monitor.py）。


// 域枚举定义在 domains.ts（那里还有中文标签 / 下拉顺序），这里转出一手，组件按需从任一处取。

export type AguiEvent = {
  type: "monitor_event";
  event:
    | "session_created"
    | "assistant_call"
    | "tool_start"
    | "tool_end"
    // 后端 A4-1 删派发后不再下发，仅为老会话回放保留；ActivityFeed 当普通 info 行画。
    | "fork"
    | "queue_status"
    // curator 本轮沉淀了新长期偏好（data.preferences: [{content, dedup_key}]）——回复下方画一行
    // 「记住了 … ✕」，✕ 即 DELETE 掉那条。自动写入，但看得见、撤得掉。
    | "memory_updated"
    // 本轮**读取侧**用到了哪些长期记忆（data: {domains, excluded, attenuated}）——思考过程里画一行
    // 「按你的长期偏好：排除 N 项、降权 M 项」。记忆最危险的失败是静默的：一条偏好误杀了一批商品，
    // 用户只会觉得「怎么老是搜不出东西」，且归因不到记忆头上。这一行就是解药。
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
    | "session_constraints"
    // 交易确认卡（对齐参考项目 confirmation.required / resolved，data: {confirmation}）：
    // 载荷是一条完整的服务端确认记录。真源在库里（GET /api/threads/{id}/confirmations），事件只是
    // 「有变化」的通知；前端按 confirmation_id 合并、决议单向推进（lib/confirmations.ts）。
    | "confirmation_required"
    | "confirmation_resolved"
    // 选购指南卡（data: {guide: GuideData}）：present_guide 收尾即推，前端画成一张分节卡。
    // 与 items_preview 同一待遇——结果本身，不进活动流；进回放存档，刷新不丢。
    | "guide_ready";
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
  // 槽位名（「一套齐」/「多类并列」轮才非空）：任一卡带 slot 即按槽分组渲染（组头 = 槽名 +
  // 该槽花费），代替平台胶囊筛选。走结构化字段而非理由文案的【槽名】前缀（前缀会被收尾 LLM
  // 重写时丢掉）。
  slot?: string;
  // 卡片附加行（后端 SummaryItem / _preview_item 同构下发）：品牌、评分（只有分、没有评价数——
  // 数据集评价数恒为 0，不显示）、到手价寄往哪（只在 landed_usd 有值时非空）。
  brand?: string;
  rating?: number | null;
  dest_country?: string;
  // 槽位形态："parallel" = 几类互不相干的东西分头推荐（跑鞋 + 耳机），此时**不显示合计**——
  // 把它们的价格加起来对用户没有任何意义。空 / 缺省 = 「一套齐」，照常显示这一套合计多少钱。
  slot_mode?: string;
};

// 一张订单（后端 Order.snapshot()）。金额在后端按最小单位整数算，这里拿到的已是主单位小数。
export type OrderSnapshot = {
  order_id: string;
  status: "DRAFT" | "CONFIRMED" | "CANCELLED";
  currency: string;
  total: number;
  address: string;
  created_at: string;
  cancel_reason?: string;
  lines: {
    platform: string;
    item_id: string;
    title: string;
    unit_price: number;
    quantity: number;
    landed_usd?: number | null;
  }[];
};

// 交易确认记录（后端 Confirmation.envelope()，字段名对齐参考项目 TradeConfirmation）。
// 一张确认卡 = 库里一条记录：pending → approved | rejected，expired 是按时钟算的派生态。
// 决议**只走 HTTP**（用户点按钮），模型没有对应工具；金额是最小单位整数（分），前端换算显示。
export type ShippingAddress = {
  recipient_name: string;
  country: string;
  state: string;
  city: string;
  address_line: string;
  postal_code: string;
  phone: string;
};

export type ConfirmationLine = {
  platform: string;
  item_id: string;
  title: string;
  unit_price_minor: number;
  currency: string;
  quantity: number;
  landed_usd?: number | null;
};

export type TradeConfirmation = {
  confirmation_id: string;
  operation_id: string;
  buyer_id: string;
  session_id: string;
  action: "create" | "cancel";
  status: "pending" | "approved" | "rejected";
  payload: {
    items: ConfirmationLine[];
    shipping_address: ShippingAddress;
    total_amount_minor: number;
    currency: string;
    amount_scope: "merchandise_only";
    order_kind: string;
    order_id?: string;
    reason?: string;
  };
  snapshot_hash: string;
  expires_at: string;
  expired: boolean;
  result: { order_id: string; status: "CONFIRMED" | "CANCELLED"; total_amount_minor: number; currency: string } | null;
  created_at: string;
  resolved_at: string | null;
};

// 下单意向表单的提交体（POST /api/threads/{id}/confirmations/orders）。
export type PrepareOrderInput = {
  items: { item_id: string; quantity: number }[];
  shipping_address: ShippingAddress;
};

// 本轮 token 用量（主环所有模型调用合计）。随 task_result 事件下发、随 turns.json 落盘回看。
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

// 选购指南（guide_ready 事件，后端 present_guide 的结构化那份）：一节一条标准 + 假设 + 来源。
// 正文另有一条路——工具排好的 markdown 已并回 final_text，所以历史回看即便不画这张卡也看得到
// 全文（与 ask_user 的 chips 同一取舍：卡只在当轮与 inflight 回放里活）。
export type GuideSection = { title: string; points: string[] };
export type GuideSource = { title: string; url: string };
export type GuideData = {
  topic: string;
  sections: GuideSection[];
  assumptions: string[];
  sources: GuideSource[];
  closing: string;
};

// 本轮的「实验与自进化」归属（批 4）：提示词版本 / A/B 桶号 / 注入了哪几条策略 / 读了哪些 skill。
// 随 task_result 下发、随 turns 落盘回看。此前这些只在 Langfuse trace 里看得到，产品面全盲。
// ab_bucket = -1 表示匿名（不参与实验）；in_experiment=false 即对照组 / 匿名 / 实验未开。
// MCP 工具调用不在这里：它们发生在 SearchAgent（worker）那侧，主 loop 看不到。
export type TurnExperiment = {
  prompt_version: string;
  ab_bucket: number;
  in_experiment: boolean;
  strategies: string[];
  skills: string[];
};

// 一条长期记忆（GET /api/preferences/{user_id} 返回）。
// 字段就是模型看到的那几个：页面上给用户看的，和注入给模型的是同一份东西。
// category 决定注入优先级（constraint 每轮必注入，其余按新鲜度补位），不决定杀伤力。
// updated_at 只参与补位排序与保留期，不参与任何打分。注意后端 SQLite 存的是 naive datetime，
// ISO 串没有时区后缀 —— 按 UTC 解析，别当本地时间。
export type MemoryCategory = "preference" | "constraint" | "context";

export type Preference = {
  key: string;
  value: string;
  category: MemoryCategory;
  updated_at: string;
  source_session: string;
};

// 手填 / 修改一条记忆的请求体（POST / PUT 共用），与 save_memory 工具同形态。
export type FactWrite = {
  key: string;
  value: string;
  category: MemoryCategory;
};

// memory_updated 事件里的一条：回复下方那行「记住了 …」，✕ 用事实的 key 删。
export type LearnedPref = {
  content: string;
  key: string;
};

// 会话级 P_t 约束（session_constraints 事件 / GET /api/session/{tid}/constraints）。
// id 形如 `<bucket>:<term>`（bucket = exclude / avoid / prefer），删除按它打 DELETE；
// source_quote 现恒为空串、epoch 恒为 0（P_t 已退回词表结构，按词撤回，不再存原话）。
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
  // 「本次选购摘要」：planner 累积的一句话意图与已确定的槽位（如 收货国 / 尺码），偏好面板
  // 「本次会话」区展示，让用户看得见 Agent 当前以为的需求是什么。旧快照可能缺这几个字段。
  current_intent?: string;
  slots?: Record<string, string>;
  turn?: number;
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
  experiment?: TurnExperiment; // 提示词版本 / A/B 桶 / 策略 / skill，回看时画成一行 chip。
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

// GET /api/skills/catalog 的一条：内置（skills/ 下的 SKILL.md）或个人（my/ 前缀）skill 的目录项。
// 只有 name / description，正文按需由 Agent 的 Skill 工具读，或在用户 / 显式选中时由服务端注入。
export type SkillCatalogItem = {
  name: string;
  description: string;
  source: "builtin" | "user";
};

// 个人 Skill 全量（GET /api/skills，编辑面板用）。catalog_name 即 `my/<name>`，是发任务时要传的 skill 字段。
export type UserSkill = {
  name: string;
  catalog_name: string;
  description: string;
  body: string;
  version: number;
  updated_at: string | null;
};
