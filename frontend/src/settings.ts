// 本地设置（localStorage）：目前只有「启用平台」一项。
//
// 为什么默认只勾 amazon：召回库里 99.75% 的商品是 amazon，其余平台近乎空库。默认跨 5 平台 fork
// 的结果是「5 个子 Agent 出去、4 个空手而归」——白烧 ~60% 的 token 和一整轮墙钟。所以默认单平台
// （主流程直接检索、不 fork），用户明确勾上多个平台才真正触发跨平台并行比价。
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
