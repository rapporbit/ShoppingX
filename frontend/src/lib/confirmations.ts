import type { TradeConfirmation } from "../types";

// 确认记录的读入与合并（移植自参考项目 lib/confirmations.ts）。
//
// readConfirmations：只有结构完整的服务端快照才能成为可点击的确认卡——缺字段、金额对不平、地址
// 四必填缺一的记录直接丢掉，宁可少画一张也不画一张点下去会 409 的卡。
// mergeConfirmations：决议单向推进。事件流与 GET 列表两条路都会送来快照，顺序不保证；一张已经
// approved 的卡若再收到迟到的 pending 快照，不能让按钮重新出现。

const record = (v: unknown): v is Record<string, unknown> =>
  !!v && typeof v === "object" && !Array.isArray(v);
const text = (v: unknown): v is string => typeof v === "string" && !!v.trim();
const minor = (v: unknown): v is number =>
  typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
const currency = (v: unknown): v is string => typeof v === "string" && /^[A-Z]{3}$/.test(v);

export function readConfirmations(value: unknown): TradeConfirmation[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is TradeConfirmation => {
      if (
        !record(item) ||
        !text(item.confirmation_id) ||
        !text(item.operation_id) ||
        !text(item.buyer_id) ||
        !text(item.session_id) ||
        !text(item.snapshot_hash) ||
        !text(item.expires_at) ||
        !Number.isFinite(Date.parse(item.expires_at)) ||
        !["create", "cancel"].includes(String(item.action)) ||
        !["pending", "approved", "rejected"].includes(String(item.status))
      )
        return false;
      const p = item.payload;
      if (
        !record(p) ||
        p.amount_scope !== "merchandise_only" ||
        !minor(p.total_amount_minor) ||
        !currency(p.currency) ||
        !Array.isArray(p.items) ||
        !p.items.length
      )
        return false;
      let total = 0;
      for (const line of p.items) {
        if (
          !record(line) ||
          !text(line.item_id) ||
          !text(line.title) ||
          !minor(line.unit_price_minor) ||
          line.currency !== p.currency ||
          !minor(line.quantity) ||
          line.quantity <= 0
        )
          return false;
        total += line.unit_price_minor * line.quantity;
      }
      if (!Number.isSafeInteger(total) || total !== p.total_amount_minor) return false;
      const address = p.shipping_address;
      if (
        !record(address) ||
        !["recipient_name", "country", "city", "address_line"].every((k) => text(address[k])) ||
        !["state", "postal_code", "phone"].every((k) => typeof address[k] === "string")
      )
        return false;
      if (item.action === "cancel" && (!text(p.order_id) || !text(p.reason))) return false;
      if (item.status === "approved") {
        const r = item.result;
        if (
          !record(r) ||
          !text(r.order_id) ||
          !["CONFIRMED", "CANCELLED"].includes(String(r.status)) ||
          !minor(r.total_amount_minor) ||
          !currency(r.currency)
        )
          return false;
      } else if (item.result !== null && item.result !== undefined) return false;
      return true;
    })
    .map((item) => ({
      ...item,
      result: item.result ?? null,
      expired: item.expired === true || Date.parse(item.expires_at) <= Date.now(),
    }));
}

export function mergeConfirmations(
  previous: TradeConfirmation[],
  next: TradeConfirmation[],
): TradeConfirmation[] {
  const entries = new Map(previous.map((c) => [c.confirmation_id, c]));
  for (const item of next) {
    const old = entries.get(item.confirmation_id);
    entries.set(
      item.confirmation_id,
      old && old.status !== "pending" && item.status === "pending" ? old : item,
    );
  }
  const all = [...entries.values()].sort((a, b) => a.created_at.localeCompare(b.created_at));
  // 一张单被取消确认卡取消了 → 当初那张下单卡的 result 也标成 CANCELLED（同一张单的两种叙述要一致）。
  const cancelled = new Set(
    all.filter((c) => c.result?.status === "CANCELLED").map((c) => c.result!.order_id),
  );
  return all
    .map((c) =>
      c.result && cancelled.has(c.result.order_id)
        ? { ...c, result: { ...c.result, status: "CANCELLED" as const } }
        : c,
    )
    .slice(-20);
}

// 最小单位 → 主单位。零小数位币种（JPY / KRW …）倍率 1，与后端 app/trade/money.py 同一张表。
const ZERO_DECIMAL = new Set(["JPY", "VND", "IDR", "CLP", "COP"]);
export function minorToMajor(minor: number, cur: string): number {
  return ZERO_DECIMAL.has(cur.toUpperCase()) ? minor : minor / 100;
}
export function money(minor: number, cur: string): string {
  const v = minorToMajor(minor, cur);
  return `${ZERO_DECIMAL.has(cur.toUpperCase()) ? v.toFixed(0) : v.toFixed(2)} ${cur}`;
}
