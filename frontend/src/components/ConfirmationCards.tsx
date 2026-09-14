import { useEffect, useState } from "react";
import type { TradeConfirmation } from "../types";
import { money } from "../lib/confirmations";

// 交易确认卡（移植自参考项目 ConfirmationCards.tsx）。一张卡四态：pending（等你点）/ approved /
// rejected / expired（pending 且过了 expires_at，按时钟算）。按钮只在 pending 且未过期时可点；
// 点下去走 HTTP resolve，**不经模型**——对话里说「确认」不算数，这是设计，不是缺口。
// approved 的下单卡上可以直接申请取消（填原因 → 再出一张取消确认卡）。

type Props = {
  confirmations: TradeConfirmation[];
  busy: boolean;
  error: string | null;
  onResolve: (c: TradeConfirmation, approved: boolean) => void;
  onCancelOrder: (orderId: string, reason: string) => void;
  onRefresh: () => void;
};

export function isConfirmationExpired(c: TradeConfirmation, now: number): boolean {
  if (c.status !== "pending") return false;
  const t = Date.parse(c.expires_at);
  return c.expired || !Number.isFinite(t) || t <= now;
}

function expiryLabel(expiresAt: string, now: number): string {
  const t = Date.parse(expiresAt);
  if (!Number.isFinite(t)) return "有效期需重新核验";
  const min = Math.max(1, Math.ceil((t - now) / 60000));
  return `${min} 分钟内有效`;
}

function Card({
  c,
  now,
  busy,
  cancelled,
  cancelPending,
  onResolve,
  onCancelOrder,
}: {
  c: TradeConfirmation;
  now: number;
  busy: boolean;
  cancelled: boolean;
  cancelPending: boolean;
  onResolve: Props["onResolve"];
  onCancelOrder: Props["onCancelOrder"];
}) {
  const [showCancel, setShowCancel] = useState(false);
  const [reason, setReason] = useState("");
  const { payload, status, action, result } = c;
  const expired = isConfirmationExpired(c, now);
  const pending = status === "pending";
  const cancellation = action === "cancel";
  const actionable = pending && !expired && !cancelled;
  const addr = payload.shipping_address;
  const label = pending
    ? cancelled && cancellation
      ? "订单已取消"
      : expired
        ? "确认已过期"
        : "等待你确认"
    : status === "rejected"
      ? "已放弃"
      : cancelled
        ? "订单已取消"
        : "已下单";
  const orderId = result?.order_id || payload.order_id;
  const count = payload.items.reduce((n, it) => n + it.quantity, 0);

  return (
    <article className={`confirmation-card ${pending && !expired ? "is-pending" : "is-resolved"}`}>
      <header className="confirmation-card-heading">
        <div>
          <span className="confirmation-eyebrow">{cancellation ? "取消确认" : "下单确认"}</span>
          <h3>{!pending ? label : cancellation ? "确认取消这张订单" : "核对后再确认"}</h3>
        </div>
        <span className={`confirmation-status ${expired ? "is-expired" : ""} is-${status}`}>
          {label}
        </span>
      </header>

      <details className="confirmation-details" open={pending}>
        <summary>
          {money(payload.total_amount_minor, payload.currency)} · {count} 件商品
          <span>查看明细与收货信息</span>
        </summary>
        <ul className="confirmation-items">
          {payload.items.map((it) => (
            <li key={`${c.confirmation_id}-${it.item_id}`}>
              <div>
                <strong>{it.title}</strong>
                <small>
                  {it.platform} · 数量 {it.quantity}
                </small>
              </div>
              <div className="confirmation-item-price">
                <span>{money(it.unit_price_minor * it.quantity, it.currency)}</span>
                <small>单价 {money(it.unit_price_minor, it.currency)}</small>
              </div>
            </li>
          ))}
        </ul>
        <div className="confirmation-address">
          <span>收货信息</span>
          <div>
            <strong>
              {addr.recipient_name}
              {addr.phone ? ` · ${addr.phone}` : ""}
            </strong>
            <p>
              {[addr.country, addr.state, addr.city, addr.address_line, addr.postal_code]
                .filter(Boolean)
                .join(" · ")}
            </p>
          </div>
        </div>
        {cancellation && <p className="confirmation-cancel-reason">取消原因：{payload.reason}</p>}
        <div className="confirmation-total">
          <div>
            <span>商品金额合计</span>
            <small>不含运费与关税</small>
          </div>
          <strong>{money(payload.total_amount_minor, payload.currency)}</strong>
        </div>
        <p className="confirmation-scope">模拟交易：只记录订单，没有支付、物流与库存。</p>
        {orderId && (
          <p className="confirmation-order-id">
            订单号 <span>{orderId}</span>
          </p>
        )}
      </details>
      {/* 决议区与取消区见下半段 */}
      <CardActions
        c={c}
        expired={expired}
        pending={pending}
        cancellation={cancellation}
        actionable={actionable}
        cancelled={cancelled}
        cancelPending={cancelPending}
        busy={busy}
        now={now}
        showCancel={showCancel}
        setShowCancel={setShowCancel}
        reason={reason}
        setReason={setReason}
        onResolve={onResolve}
        onCancelOrder={onCancelOrder}
      />
    </article>
  );
}

function CardActions({
  c,
  expired,
  pending,
  cancellation,
  actionable,
  cancelled,
  cancelPending,
  busy,
  now,
  showCancel,
  setShowCancel,
  reason,
  setReason,
  onResolve,
  onCancelOrder,
}: {
  c: TradeConfirmation;
  expired: boolean;
  pending: boolean;
  cancellation: boolean;
  actionable: boolean;
  cancelled: boolean;
  cancelPending: boolean;
  busy: boolean;
  now: number;
  showCancel: boolean;
  setShowCancel: (v: boolean) => void;
  reason: string;
  setReason: (v: string) => void;
  onResolve: Props["onResolve"];
  onCancelOrder: Props["onCancelOrder"];
}) {
  if (pending) {
    return (
      <div className="confirmation-decision">
        <p className={`confirmation-expiry ${expired ? "is-expired" : ""}`}>
          {expired ? "这张确认卡已过期，请重新生成。" : expiryLabel(c.expires_at, now)}
        </p>
        <div className="confirmation-actions">
          <button
            type="button"
            className="btn-ghost"
            disabled={busy || !actionable}
            onClick={() => onResolve(c, false)}
          >
            {cancellation ? "保留订单" : "先不下单"}
          </button>
          <button
            type="button"
            className="btn-primary"
            disabled={busy || !actionable}
            onClick={() => onResolve(c, true)}
          >
            {busy ? "正在处理…" : cancellation ? "确认取消" : "确认下单"}
          </button>
        </div>
      </div>
    );
  }
  if (c.status !== "approved" || cancellation || !c.result || cancelled) return null;
  const orderId = c.result.order_id;
  return (
    <div className="confirmation-cancel-area">
      {cancelPending ? (
        <p className="confirmation-hint">已有待确认的取消申请，请在上面的取消确认卡里处理。</p>
      ) : showCancel ? (
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (!reason.trim() || busy) return;
            onCancelOrder(orderId, reason.trim());
            setShowCancel(false);
            setReason("");
          }}
        >
          <label>
            取消原因
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              required
              maxLength={200}
              rows={2}
              placeholder="请填写本次取消原因"
              disabled={busy}
            />
          </label>
          <div className="confirmation-actions">
            <button type="button" className="btn-ghost" disabled={busy} onClick={() => setShowCancel(false)}>
              返回
            </button>
            <button type="submit" className="btn-primary" disabled={busy || !reason.trim()}>
              {busy ? "正在准备…" : "生成取消确认卡"}
            </button>
          </div>
        </form>
      ) : (
        <button type="button" className="confirmation-text-button" disabled={busy} onClick={() => setShowCancel(true)}>
          申请取消这张订单
        </button>
      )}
    </div>
  );
}

export function ConfirmationCards({ confirmations, busy, error, onResolve, onCancelOrder, onRefresh }: Props) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!confirmations.some((c) => c.status === "pending")) return;
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 15_000);
    return () => window.clearInterval(t);
  }, [confirmations]);
  const cancelledOrders = new Set(
    confirmations.flatMap((c) => (c.result?.status === "CANCELLED" ? [c.result.order_id] : [])),
  );
  const pendingCancellations = new Set(
    confirmations.flatMap((c) =>
      c.action === "cancel" && c.status === "pending" && !isConfirmationExpired(c, now) && c.payload.order_id
        ? [c.payload.order_id]
        : [],
    ),
  );
  if (!confirmations.length && !error) return null;
  return (
    <section className="confirmations-section" aria-label="交易确认" aria-busy={busy}>
      <div className="confirmations-heading">
        <div>
          <span className="confirmation-eyebrow">由你作决定</span>
          <h2>交易确认</h2>
        </div>
        <button type="button" className="btn-ghost" disabled={busy} onClick={onRefresh}>
          {busy ? "正在同步…" : "刷新记录"}
        </button>
      </div>
      {error && (
        <div className="confirmation-error" role="alert">
          <strong>请核对操作状态</strong>
          <p>{error}</p>
        </div>
      )}
      <div className="confirmation-list">
        {confirmations.map((c) => (
          <Card
            key={c.confirmation_id}
            c={c}
            now={now}
            busy={busy}
            cancelled={cancelledOrders.has(c.result?.order_id || c.payload.order_id || "")}
            cancelPending={pendingCancellations.has(c.result?.order_id || "")}
            onResolve={onResolve}
            onCancelOrder={onCancelOrder}
          />
        ))}
      </div>
    </section>
  );
}
