// 后端 HTTP 接口的薄封装。全用同源相对路径——开发期靠 Vite 代理转到 :8000（见 vite.config.ts）。
//
// 一律走 authFetch 而不是裸 fetch（M16）：它负责带上 token，并把 401（token 过期 / 被吊销）统一
// 退回登录页。漏一个裸 fetch，那个接口在开了鉴权后会静默 401——前端只会当成「没数据」，不会报错。
import { authFetch } from "./auth";
import { loadPlatforms } from "./settings";
import type {
  AdminConfig,
  AguiEvent,
  HistoryTurn,
  Preference,
  PrefDraft,
  ProductItem,
  SessionSnapshot,
} from "./types";

// GET /api/task/{tid}/inflight 返回：该 thread 是否仍有任务在后台跑（刷新/切回时据此续看）。
// running=true 时带「正在跑那一轮」的提问原文与已发生的事件，供前端重建该轮再重连 WS 续直播。
export type Inflight = {
  running: boolean;
  query: string | null;
  // 正在跑那一轮的参考图文件名——刷新后要能立刻把图画回气泡，而不是等任务收尾落库才出现。
  images: string[];
  events: AguiEvent[];
};

// POST /api/upload：把参考图传到本次会话目录，拿回服务端落盘用的文件名。
// 图不随任务请求走——任务只带**文件名**，图片本体留在服务端盘上由 image_understand 工具去读。
// 必须先上传、拿到 filename，再发任务（否则 Agent 开局预读图时文件还没落地）。
export async function uploadImage(threadId: string, file: File): Promise<string> {
  const form = new FormData();
  form.append("thread_id", threadId);
  form.append("file", file);
  // 不设 Content-Type：交给浏览器自动带上 multipart 的 boundary，手写会漏 boundary 导致后端解析失败。
  const resp = await authFetch("/api/upload", { method: "POST", body: form });
  if (resp.status === 415) throw new Error("只支持图片（jpg / png / webp / gif / bmp）");
  if (resp.status === 413) throw new Error("图片太大了，换一张小一点的");
  if (!resp.ok) throw new Error(`上传失败：HTTP ${resp.status}`);
  return (await resp.json()).filename as string;
}

export async function startTaskRequest(
  query: string,
  threadId: string,
  userId?: string,
  imagePaths?: string[],
): Promise<void> {
  // platforms 在发任务这一刻从设置里读（唯一真源，见 settings.ts）：默认只搜 amazon，用户在设置
  // 里勾了多个平台才跨平台并行 fork 比价。不必把它一路穿过 hook 的参数链。
  const resp = await authFetch("/api/task", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      thread_id: threadId,
      user_id: userId,
      platforms: loadPlatforms(),
      image_paths: imagePaths?.length ? imagePaths : undefined,
    }),
  });
  // fetch 只在网络层失败才 reject；4xx/5xx 仍 resolve。不显式查 ok，任务起不来时 UI 会一直卡
  // 在 running（等一个永不到来的终结事件）。这里查 ok 并抛，让 hook 的 .catch 翻到 error 态。
  //
  // 402 单独成一类错（QuotaExhaustedError）：它不是「出错了、再试试」，而是「今天的额度花完了，
  // 再试也一样」。UI 要据此换一套文案（显示重置时刻）并把输入框锁掉，而不是丢一句 HTTP 402。
  if (resp.status === 402) {
    const detail = (await resp.json().catch(() => null))?.detail;
    throw new QuotaExhaustedError(detail ?? null);
  }
  if (!resp.ok) throw new Error(`启动任务失败：HTTP ${resp.status}`);
}

// 一个周期（UTC 自然日）内的 credit 余额。enabled=false 表示后端没开配额（demo / 本地），前端
// 整块隐藏余额条——不给用户看一个恒为满格、毫无意义的进度条。
export type Quota = {
  enabled: boolean;
  period: string;
  used_credits: number;
  limit_credits: number;
  remaining_credits: number;
  task_count: number;
  reset_at: string;
  exhausted: boolean;
};

export class QuotaExhaustedError extends Error {
  quota: Quota | null;
  constructor(quota: Quota | null) {
    super("今日 credit 已用完");
    this.name = "QuotaExhaustedError";
    this.quota = quota;
  }
}

// 本地时区的「几点几分」——余额条与耗尽提示都要告诉用户「什么时候回满」。后端给的 reset_at 是
// UTC 零点，直接原样显示会让人以为额度在半夜某个奇怪的钟点才恢复（对中国用户是早上 8 点）。
export function formatResetAt(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString(undefined, { hour: "2-digit", minute: "2-digit", weekday: "short" });
}

// 启动任务失败时给用户看的一句话。额度耗尽是**可预期的正常状态**，不该像别的错误那样甩一个
// HTTP 码；其余失败原样透出（网络断了、后端挂了，用户重试是有意义的）。
export function describeStartError(e: unknown): string {
  if (e instanceof QuotaExhaustedError) {
    const at = e.quota ? formatResetAt(e.quota.reset_at) : "";
    return at
      ? `今日 credit 已用完，额度将在 ${at} 重置后恢复。`
      : "今日 credit 已用完，请等待额度重置。";
  }
  return String(e);
}

export async function fetchQuota(): Promise<Quota | null> {
  try {
    const resp = await authFetch("/api/quota");
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null; // 拉不到余额只是余额条不显示，不该拖垮页面（与 fetchSessions 同一处理）
  }
}

export async function cancelTaskRequest(threadId: string): Promise<void> {
  await authFetch(`/api/task/${threadId}/cancel`, { method: "POST" });
}

export async function fetchPreferences(userId: string): Promise<Preference[]> {
  const resp = await authFetch(`/api/preferences/${encodeURIComponent(userId)}`);
  if (!resp.ok) return [];
  const body = await resp.json();
  return body.preferences ?? [];
}

// 一句自然语言 → 结构化偏好草稿（**只解析，不落库**）。前端把草稿渲染成可编辑卡，
// 用户确认 / 改完再调 addPreferences 落库——不让 LLM 猜的 polarity/domain/keywords 悄悄进库。
// 「绝不推荐」（blocking）尤其如此：LLM 拿不准一律给 false，要不要授予硬淘汰权由用户在草稿卡上勾。
// 解析不出时后端返 400 —— 把 detail 抛出去让页面提示「换个说法」，不静默吞掉。
export async function parsePreference(userId: string, text: string): Promise<PrefDraft[]> {
  const resp = await authFetch(`/api/preferences/${encodeURIComponent(userId)}/parse`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.detail ?? "解析失败");
  return body.drafts ?? [];
}

// 落库若干条结构化偏好（source=user：此后不被 curator 覆盖、也不衰减）。
export async function addPreferences(userId: string, entries: PrefDraft[]): Promise<Preference[]> {
  const resp = await authFetch(`/api/preferences/${encodeURIComponent(userId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ entries }),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.detail ?? "添加失败");
  return body.added ?? [];
}

// 改一条偏好的结构化字段。改 polarity/category/domain/slug 会换一把 dedup_key，故 URL 传**旧** key，
// 后端「删旧 + 写新」——字段没变时新旧同 key，等价于覆盖。
export async function updatePreference(
  userId: string,
  oldKey: string,
  draft: PrefDraft,
): Promise<Preference[]> {
  const resp = await authFetch(
    `/api/preferences/${encodeURIComponent(userId)}/entry/${encodeURIComponent(oldKey)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(draft),
    },
  );
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.detail ?? "保存失败");
  return body.updated ?? [];
}

// 删一条偏好（页面上每行的 ×，以及回复下方「记住了 …」那行的撤销）。
// dedup_key 形如 dislike:material:global:plastic，冒号在 URL path 段里合法，但仍编码一次防意外。
export async function deletePreference(userId: string, dedupKey: string): Promise<void> {
  await authFetch(
    `/api/preferences/${encodeURIComponent(userId)}/${encodeURIComponent(dedupKey)}`,
    { method: "DELETE" },
  );
}

// 读本次会话累积的 P_t 约束集（偏好面板「本次会话」区；打开面板 / 断线重连时主动拉）。
// 拉不到只兜底空快照——面板少一个区，不拖垮页面。
export async function fetchSessionConstraints(threadId: string): Promise<SessionSnapshot> {
  const empty: SessionSnapshot = { epoch: 0, budget_usd: null, category: "", constraints: [] };
  try {
    const resp = await authFetch(`/api/session/${encodeURIComponent(threadId)}/constraints`);
    if (!resp.ok) return empty;
    return (await resp.json()) as SessionSnapshot;
  } catch {
    return empty;
  }
}

// 删本次会话的一条 P_t 约束（面板「本次会话」区每行的 ×）——约束抽取出错时的人纠错入口。
// 后端删完会经 WS 推新快照（session_constraints 事件），无需手动重拉。
export async function deleteSessionConstraint(threadId: string, cid: string): Promise<void> {
  await authFetch(
    `/api/session/${encodeURIComponent(threadId)}/constraints/${encodeURIComponent(cid)}`,
    { method: "DELETE" },
  );
}

// 「我的资料」：收货地 / 预算上限这类硬事实，用户显式设定（零 LLM）。
// 只传的字段才更新；传空串 / 0 = 清除该项。返回更新后的全量偏好。
export async function updateProfile(
  userId: string,
  patch: { dest_country?: string; budget_max_usd?: number },
): Promise<Preference[]> {
  const resp = await authFetch(`/api/preferences/${encodeURIComponent(userId)}/profile`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.detail ?? "保存失败");
  return body.preferences ?? [];
}

// 读某 thread 的逐轮对话（GET /api/history/{tid}），用于刷新 / 重进页面后「回看」并接着聊。
// 后端对未跑过的 thread 返回空 turns（非 404），故这里出错只兜底空数组，不打断渲染。
export async function fetchHistory(threadId: string): Promise<HistoryTurn[]> {
  const resp = await authFetch(`/api/history/${encodeURIComponent(threadId)}`);
  if (!resp.ok) return [];
  const body = await resp.json();
  return body.turns ?? [];
}

// 查某 thread 是否仍有任务在后台跑（GET /api/task/{tid}/inflight），用于刷新/切回对话时自动续看。
// 任务与 WS 解耦：断订阅不打断任务，重新进入对话时靠它判断要不要重建「正在跑那一轮」并续直播。
// 出错（含网络层）一律兜底「未在跑」，不打断历史回看的渲染。
export async function fetchInflight(threadId: string): Promise<Inflight> {
  try {
    const resp = await authFetch(`/api/task/${encodeURIComponent(threadId)}/inflight`);
    if (!resp.ok) return { running: false, query: null, images: [], events: [] };
    const body = await resp.json();
    return {
      running: Boolean(body.running),
      query: body.query ?? null,
      images: body.images ?? [],
      events: body.events ?? [],
    };
  } catch {
    return { running: false, query: null, images: [], events: [] };
  }
}

// ——— 商品卡「♡ 收藏」———
// 不注入 prompt、不进长期偏好库；但经 app.memory.affinity 聚合成弱信号，给精挑里同类属性小幅加分
// （只上浮不淘汰，见 store.FavoriteItem / memory.affinity）。存整张卡的快照而非只存 id——收藏跨会话，
// 而候选池随会话清，换个会话按 id 早捞不回商品了。

export async function fetchFavorites(userId: string): Promise<ProductItem[]> {
  try {
    const resp = await authFetch(`/api/favorites/${encodeURIComponent(userId)}`);
    if (!resp.ok) return [];
    return (await resp.json()).favorites ?? [];
  } catch {
    return [];
  }
}

export async function addFavorite(userId: string, item: ProductItem): Promise<void> {
  await authFetch(`/api/favorites/${encodeURIComponent(userId)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      item_id: item.item_id,
      title: item.title,
      platform: item.platform,
      price_usd: item.price_usd ?? null,
      landed_usd: item.landed_usd ?? null,
      image_url: item.image_url ?? "",
      url: item.url ?? "",
    }),
  });
}

export async function removeFavorite(userId: string, itemId: string): Promise<void> {
  await authFetch(
    `/api/favorites/${encodeURIComponent(userId)}/${encodeURIComponent(itemId)}`,
    { method: "DELETE" },
  );
}

// ——— 商品卡「搜同款」———
// 拿这件商品在召回库里的向量找全库近邻（GET /api/similar/{item_id}）。**不经过 Agent**：一次纯
// 向量检索、0 次 LLM、亚秒级返回。库里 amazon 占绝大多数，故近邻多半也是 amazon——如实呈现，
// 别把它当跨平台比价。回来的卡只有货价（price_usd）没有到手价：那要跑 shipping_calc。
export async function fetchSimilar(itemId: string, topK = 8): Promise<ProductItem[]> {
  try {
    const resp = await authFetch(`/api/similar/${encodeURIComponent(itemId)}?top_k=${topK}`);
    if (!resp.ok) return [];
    return (await resp.json()).items ?? [];
  } catch {
    return [];
  }
}

// ——— 我的会话清单（M16）———
// 侧栏历史的真源。此前它只活在浏览器的 localStorage 里：换台设备、清个缓存，后端数据明明还在，
// 用户却再也找不回自己的对话。现在按 token 里的身份从归属表查，登录到哪台机器都是同一份。
export type SessionMeta = {
  thread_id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export async function fetchSessions(): Promise<SessionMeta[]> {
  try {
    const resp = await authFetch("/api/sessions");
    if (!resp.ok) return [];
    return (await resp.json()).sessions ?? [];
  } catch {
    return []; // 拉不到清单只是侧栏空着，不该拖垮整个页面
  }
}

// 从我的清单里删掉一段会话（侧栏的删除按钮）。删的是归属记录 = 入口，对话正文仍在磁盘上。
export async function deleteSession(threadId: string): Promise<void> {
  await authFetch(`/api/sessions/${encodeURIComponent(threadId)}`, { method: "DELETE" });
}

// 下载会话产物（summary.md / result.json）。
//
// **为什么不是一个 <a href> 直链**：产物接口现在要校验属主（M16），而浏览器对 <a href> 发起的
// 请求带不上 Authorization 头——直链会稳定 401，用户点了没反应还不知道为什么。所以改成：用带
// token 的请求把文件取成一段内存数据，再造一个临时链接点它，下载完即回收。
export async function downloadFile(threadId: string, filename: string): Promise<void> {
  const resp = await authFetch(`/api/files/${encodeURIComponent(threadId)}/${filename}`);
  if (!resp.ok) throw new Error(`下载失败：HTTP ${resp.status}`);
  const url = URL.createObjectURL(await resp.blob());
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url); // 不回收就是一路泄漏内存（blob 会被一直held住）
}

// 取回本会话上传的参考图，用于在对话气泡里回显。
//
// 同 downloadFile 的理由：<img src="/api/uploads/..."> 带不上 Authorization 头，而这个口要校验
// 属主（别人的会话图不给看），直链只会稳定 401。所以取成 blob 再转 object URL 交给 <img>。
// **调用方必须在卸载时 revokeObjectURL**，否则每次回看旧会话都会漏一份图在内存里。
export async function fetchUploadedImage(threadId: string, filename: string): Promise<string> {
  const resp = await authFetch(`/api/uploads/${encodeURIComponent(threadId)}/${filename}`);
  if (!resp.ok) throw new Error(`取图失败：HTTP ${resp.status}`);
  return URL.createObjectURL(await resp.blob());
}

// --- 后台管理（仅管理员）---------------------------------------------------

// 当前用户是不是管理员。403 = 不是（或鉴权没开、或没配 ADMIN_USERNAMES），静默返回 false——
// 这个判断只用于决定「要不要显示后台入口」，不是错误场景，不该弹提示打扰普通用户。
export async function checkAdmin(): Promise<boolean> {
  try {
    const resp = await authFetch("/api/admin/whoami");
    return resp.ok;
  } catch {
    return false;
  }
}

export async function fetchAdminConfig(): Promise<AdminConfig> {
  const resp = await authFetch("/api/admin/config");
  if (!resp.ok) throw new Error(`读取配置失败：HTTP ${resp.status}`);
  return (await resp.json()) as AdminConfig;
}

// 改参数。后端整批校验，任一非法则整批不生效（400 带中文原因，如「商品卡展示上限：不能大于 30」），
// 故这里把 detail 原样抛给调用方显示——那句话本就是写给人看的。
export async function updateAdminConfig(
  values: Record<string, number | string | boolean>,
): Promise<AdminConfig> {
  const resp = await authFetch("/api/admin/config", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ values }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => null);
    throw new Error(detail?.detail ?? `保存失败：HTTP ${resp.status}`);
  }
  return (await resp.json()) as AdminConfig;
}

// 恢复默认。keys 省略 = 全部恢复。值退回 .env 基线或代码默认值。
export async function resetAdminConfig(keys?: string[]): Promise<AdminConfig> {
  const resp = await authFetch("/api/admin/config/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keys: keys ?? null }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => null);
    throw new Error(detail?.detail ?? `恢复默认失败：HTTP ${resp.status}`);
  }
  return (await resp.json()) as AdminConfig;
}
