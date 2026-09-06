import type { OrderCardPayload } from "../types";

// 订单卡（批 1 交易域）。三种形态共用一个组件、靠 kind 区分：
//   preview   —— **尚未下单**的确认卡。这是用户下一句「确认」的依据，所以「未下单」四个字要显眼，
//                别让它长得像一张下单成功的回执。
//   placed    —— 下单成功，带订单号。
//   cancelled —— 已取消。
//
// 卡片**不带「确认下单」按钮**：确认走对话（用户回一句「确认」，模型再调一次 create_order）。
// 加个按钮意味着前端要自己拼一次下单请求，那条路绕过了 Agent 的全部闸（候选校验、幂等、
// 归属），而它省下的只是用户敲两个字。

const STATUS_LABEL: Record<string, string> = {
  DRAFT: "草稿",
  CONFIRMED: "已下单",
  CANCELLED: "已取消",
};

function money(v: number | null | undefined, currency: string): string {
  return v == null ? "—" : `${v.toFixed(2)} ${currency}`;
}

export function OrderCard({ payload }: { payload: OrderCardPayload }) {
  const { kind, order, preview, total_display, address } = payload;

  if (kind === "preview") {
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
        <div className="order-hint">确认无误请回复「确认」，我再为你下单。</div>
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
