// 本地设置（localStorage）：「启用平台」与「收货国」两项。
//
// 为什么默认只勾 amazon：召回库里 99.75% 的商品是 amazon，其余平台近乎空库。默认跨 5 平台的结果
// 是「同轮发 5 条 item_search、4 条空手而归」——白烧 ~60% 的 token 和一整轮墙钟。所以默认单平台
// （一条检索走完），用户明确勾上多个平台才真正同轮 batch 检索、跨平台比价。
//
// 这里是唯一真源：设置抽屉写它、api.startTaskRequest 发任务时读它随请求带给后端。后端另有一份
// 默认值与收口（app/agent/platform_scope.py）——前端不传 / 传脏值也不会把没启用的平台搜出来。

export type PlatformOption = { id: string; label: string; note: string };

// 与后端 app/utils/clean.py 的 PLATFORMS 对齐（ebay 不在召回库里，故不列）。
export const PLATFORM_OPTIONS: PlatformOption[] = [
  { id: "amazon", label: "Amazon", note: "主力库，商品最全" },
  { id: "walmart", label: "Walmart", note: "样本较少" },
  { id: "shein", label: "SHEIN", note: "样本较少" },
  { id: "shopee", label: "Shopee", note: "样本较少" },
  { id: "lazada", label: "Lazada", note: "样本较少" },
];

const STORAGE_KEY = "shoppingx.platforms";
const DEFAULT_PLATFORMS = ["amazon"];
const VALID = new Set(PLATFORM_OPTIONS.map((p) => p.id));

// 读启用平台。localStorage 不可用（隐私模式）/ 值脏 / 一个都不剩 → 回落默认单平台。
export function loadPlatforms(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [...DEFAULT_PLATFORMS];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [...DEFAULT_PLATFORMS];
    const clean = parsed.filter((p): p is string => typeof p === "string" && VALID.has(p));
    return clean.length > 0 ? clean : [...DEFAULT_PLATFORMS];
  } catch {
    return [...DEFAULT_PLATFORMS];
  }
}

// 写启用平台。全部取消勾选时存回默认（amazon）——「一个平台都不搜」不是有意义的状态。
export function savePlatforms(platforms: string[]): string[] {
  const clean = platforms.filter((p) => VALID.has(p));
  const next = clean.length > 0 ? clean : [...DEFAULT_PLATFORMS];
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // 存不进去（隐私模式）就只在本次会话内生效，不打断使用。
  }
  return next;
}

// ── 收货国（顶栏「寄往」框）─────────────────────────────────────────────────────
// 到手价的关税运费全看收货国（CN 免征额 $7、US $0、AU $660），所以它常驻顶栏、看得见点得动，
// 而不是让模型在文案里声明「这是推定的」。框里永远是**上一次实际用的收货国**：用户点选、或
// planner 解析出本轮收货国（「寄到日本」/ 长期记忆 / 默认国）都会写回这里，下一轮随任务带给后端。
// 后端解析顺序：本轮原话 > 这里的值 > 前几轮原话 > 长期记忆 > 默认国（app/tools/planner.py）。

export type DestOption = { code: string; label: string };

// 与后端 app/recall/geo.py 认得的国家对齐——关税 / 运费表只按这些 ISO 码算。
export const DEST_OPTIONS: DestOption[] = [
  { code: "CN", label: "中国" },
  { code: "HK", label: "中国香港" },
  { code: "MO", label: "中国澳门" },
  { code: "TW", label: "中国台湾" },
  { code: "JP", label: "日本" },
  { code: "KR", label: "韩国" },
  { code: "SG", label: "新加坡" },
  { code: "MY", label: "马来西亚" },
  { code: "TH", label: "泰国" },
  { code: "VN", label: "越南" },
  { code: "PH", label: "菲律宾" },
  { code: "ID", label: "印度尼西亚" },
  { code: "IN", label: "印度" },
  { code: "US", label: "美国" },
  { code: "CA", label: "加拿大" },
  { code: "MX", label: "墨西哥" },
  { code: "BR", label: "巴西" },
  { code: "GB", label: "英国" },
  { code: "DE", label: "德国" },
  { code: "FR", label: "法国" },
  { code: "AU", label: "澳大利亚" },
];

const DEST_KEY = "shoppingx.dest_country";
// 下单表单存收件地址的键（OrderIntentForm.tsx）：没选过收货国时拿它的国家当初值——用户填过的
// 收货地址就是他的默认地址。
const ORDER_ADDRESS_KEY = "shoppingx.order.address.v2";
const DEST_EVENT = "shoppingx:dest-country";

// 自由文本（下单表单里填的「CN」「日本」）→ ISO 码；认不出返回空串。
function matchDest(text: string): string {
  const t = text.trim();
  if (!t) return "";
  const byCode = DEST_OPTIONS.find((o) => o.code === t.toUpperCase());
  if (byCode) return byCode.code;
  return DEST_OPTIONS.find((o) => o.label === t)?.code ?? "";
}

// 读收货国：选过的 > 下单地址的国家 > 空串（空串 = 不随任务带，后端按原话 / 记忆 / 默认国解析）。
export function loadDestCountry(): string {
  try {
    const picked = matchDest(localStorage.getItem(DEST_KEY) ?? "");
    if (picked) return picked;
    const raw = localStorage.getItem(ORDER_ADDRESS_KEY);
    return raw ? matchDest(String(JSON.parse(raw)?.country ?? "")) : "";
  } catch {
    return "";
  }
}

// 写收货国并通知顶栏。认不出的码不写（后端回传的总是 ISO 码，这里只是防脏值）。
export function saveDestCountry(code: string): void {
  const clean = matchDest(code);
  if (!clean) return;
  try {
    localStorage.setItem(DEST_KEY, clean);
  } catch {
    // 存不进去（隐私模式）就只在本页生效。
  }
  window.dispatchEvent(new CustomEvent<string>(DEST_EVENT, { detail: clean }));
}

export function destLabel(code: string): string {
  return DEST_OPTIONS.find((o) => o.code === code)?.label ?? code;
}

// 订阅收货国变化（顶栏用）。返回取消订阅函数，直接给 useEffect 当清理。
export function onDestCountryChange(cb: (code: string) => void): () => void {
  const h = (e: Event) => cb((e as CustomEvent<string>).detail);
  window.addEventListener(DEST_EVENT, h);
  return () => window.removeEventListener(DEST_EVENT, h);
}
