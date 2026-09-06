import { useCallback, useEffect, useState } from "react";
import { cancelOrder, fetchOrders } from "../api";
import type { OrderSnapshot } from "../types";
import { CloseIcon, RefreshIcon } from "./icons";

// 「我的订单」抽屉（右侧滑出，与收藏 / 偏好同一套外壳）。
//
// 取消按钮**只在 CONFIRMED 的单上出现**：状态机只允许这一种迁移，给已取消的单画一个按了会报
// 409 的按钮，是在制造一次注定失败的点击。后端仍会再校验一遍——按钮的隐藏是体验，不是权限。
export function OrdersDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [orders, setOrders] = useState<OrderSnapshot[]>([]);
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setOrders(await fetchOrders());
    setLoading(false);
  }, []);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  const drop = async (orderId: string) => {
    const res = await cancelOrder(orderId);
    setNote(res.message);
    await load();
  };

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer drawer-wide ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">我的订单</div>
          <div className="drawer-tools">
            <button className="icon-btn" onClick={() => void load()} disabled={loading} title="刷新">
              <RefreshIcon width={16} height={16} className={loading ? "spin" : ""} />
            </button>
            <button className="icon-btn" onClick={onClose} title="关闭">
              <CloseIcon width={18} height={18} />
            </button>
          </div>
        </div>

        <div className="fav-note">
          演示用的模拟交易：有订单状态与取消，<b>没有支付、物流与库存</b>。
        </div>
        {note ? <div className="drawer-user">{note}</div> : null}

        {orders.length === 0 ? (
          <div className="drawer-empty">还没有订单。在对话里说「买第 2 个」就能下单。</div>
        ) : (
          <ul className="fav-list">
            {orders.map((o) => (
              <li key={o.order_id} className="order-row">
                <div className="order-card-head">
                  <span className="order-no">{o.order_id}</span>
                  <span
                    className={
                      o.status === "CANCELLED"
                        ? "order-badge order-badge-cancelled"
                        : "order-badge order-badge-placed"
                    }
                  >
                    {o.status === "CANCELLED" ? "已取消" : "已下单"}
                  </span>
                  <span className="order-total">
                    {o.total.toFixed(2)} {o.currency}
                  </span>
                </div>
                <ul className="order-lines">
                  {o.lines.map((line) => (
                    <li key={`${o.order_id}-${line.item_id}`}>
                      <span className="order-line-title">{line.title}</span>
                      <span className="order-line-meta">×{line.quantity}</span>
                    </li>
                  ))}
                </ul>
                {o.status === "CONFIRMED" && (
                  <button className="ghost-btn" onClick={() => void drop(o.order_id)}>
                    取消这单
                  </button>
                )}
              </li>
            ))}
          </ul>
        )}
      </aside>
    </>
  );
}
