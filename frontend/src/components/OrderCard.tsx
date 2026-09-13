import { useEffect, useState } from "react";
import type { OrderCardPayload } from "../types";

// 订单卡（批 1 交易域）。三种形态共用一个组件、靠 kind 区分：
//   preview   —— **尚未下单**的确认卡。这是用户下一句「确认」的依据，所以「未下单」四个字要显眼，
//                别让它长得像一张下单成功的回执。
//   placed    —— 下单成功，带订单号。
//   cancelled —— 已取消。
//
// 确认卡上的「确认下单 / 先不下单」两个按钮**只是替用户把那句话发进对话**（onConfirm / onDecline
// 由 App 接到 startTask），模型照旧再调一次 create_order(confirmed=True)。前端不自己拼下单
// 请求：那条路会绕过 Agent 的全部闸（候选校验、幂等、归属、确认门）。
// 有效期（expires_at）由后端出卡时给；过期后按钮灰掉，后端也会拒掉过期确认并重新出卡。

const STATUS_LABEL: Record<string, string> = {
  DRAFT: "草稿",
  CONFIRMED: "已下单",
  CANCELLED: "已取消",
};

function money(v: number | null | undefined, currency: string): string {
  return v == null ? "—" : `${v.toFixed(2)} ${currency}`;
}

function expiryText(iso: string | undefined, now: number): { expired: boolean; label: string } {
  if (!iso) return { expired: false, label: "" };
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return { expired: false, label: "" };
  if (t <= now) return { expired: true, label: "这张确认卡已过期，请重新生成" };
  const min = Math.max(1, Math.round((t - now) / 60000));
  return { expired: false, label: `${min} 分钟内有效` };
}

export function OrderCard({
  payload,
  busy = false,
  onConfirm,
  onDecline,
}: {
  payload: OrderCardPayload;
  busy?: boolean; // 任务跑着 / 不是最后一轮 → 按钮不可用
  onConfirm?: () => void;
  onDecline?: () => void;
}) {
  const { kind, order, preview, total_display, address } = payload;
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (kind !== "preview" || !payload.expires_at) return;
    const t = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(t);
  }, [kind, payload.expires_at]);

  if (kind === "preview") {
    const exp = expiryText(payload.expires_at, now);
    const canAct = Boolean(onConfirm && onDecline) && !busy && !exp.expired;
    return (
      <div className="order-card order-card-preview">
        <div className="order-card-head">
          <span className="order-badge order-badge-pending">待确认 · 尚未下单</span>
          <span className="order-total">{total_display ?? "—"}</span>
        </div>
        <ul className="order-lines">
          {(preview ?? []).map((line) => (
            <li key={line.item_id}>
              <span className="order-line-title">{line.title}</span>
              <span className="order-line-meta">
                {line.platform} · {money(line.unit_price, line.currency)} × {line.quantity}
              </span>
            </li>
          ))}
        </ul>
        {address ? <div className="order-address">寄往：{address}</div> : null}
        <div className="order-hint">
          {exp.expired ? exp.label : "确认无误请点「确认下单」或回复「确认」，我再为你下单。"}
          {!exp.expired && exp.label && <span className="order-expiry"> · {exp.label}</span>}
        </div>
        {onConfirm && onDecline && (
          <div className="order-actions">
            <button className="btn-ghost" disabled={!canAct} onClick={onDecline}>
              先不下单
            </button>
            <button className="btn-primary" disabled={!canAct} onClick={onConfirm}>
              确认下单
            </button>
          </div>
        )}
      </div>
    );
  }

  if (!order) return null;

  return (
    <div className={`order-card order-card-${order.status.toLowerCase()}`}>
      <div className="order-card-head">
        <span className="order-no">{order.order_id}</span>
        <span
          className={
            order.status === "CANCELLED"
              ? "order-badge order-badge-cancelled"
              : "order-badge order-badge-placed"
          }
        >
          {STATUS_LABEL[order.status] ?? order.status}
        </span>
        <span className="order-total">{money(order.total, order.currency)}</span>
      </div>
      <ul className="order-lines">
        {order.lines.map((line) => (
          <li key={`${order.order_id}-${line.item_id}`}>
            <span className="order-line-title">{line.title}</span>
            <span className="order-line-meta">
              {line.platform} · {money(line.unit_price, order.currency)} × {line.quantity}
            </span>
          </li>
        ))}
      </ul>
      {order.address ? <div className="order-address">寄往：{order.address}</div> : null}
      {order.cancel_reason ? (
        <div className="order-hint">取消原因：{order.cancel_reason}</div>
      ) : null}
    </div>
  );
}
