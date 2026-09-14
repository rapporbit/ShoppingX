import { useEffect, useState, type FormEvent } from "react";
import type { ProductItem } from "../types";
import { Modal } from "./Modal";
import { itemRef } from "./productText";

// 下单意向表单：收件人 / 地址 / 数量一次填齐，免得 Agent 用 ask_user 一句一句追问。
// 提交后**不直接打下单接口**——把信息组成一句话发进对话，仍由模型调 create_order(confirmed=False)
// 出确认卡、经候选校验 / 幂等 / 归属那一整套闸。表单省掉的是追问的来回，不是任何一道闸。
// 收件信息记在 localStorage，下次自动带出（只存本机，不上传到偏好库）。
const KEY = "shoppingx.order.address";

type Draft = { recipient: string; address: string; country: string; phone: string };

function loadDraft(): Draft {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { recipient: "", address: "", country: "", phone: "", ...JSON.parse(raw) };
  } catch {
    /* 坏数据当没有 */
  }
  return { recipient: "", address: "", country: "", phone: "" };
}

export function OrderIntentForm({
  item,
  onClose,
  onSubmit,
}: {
  item: ProductItem | null;
  onClose: () => void;
  onSubmit: (text: string) => void;
}) {
  const [draft, setDraft] = useState<Draft>(loadDraft);
  const [qty, setQty] = useState(1);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (item) {
      setQty(1);
      setErr(null);
    }
  }, [item]);

  if (!item) return null;
  const set = (k: keyof Draft) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setDraft((d) => ({ ...d, [k]: e.target.value }));

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!draft.recipient.trim() || !draft.address.trim()) {
      setErr("收件人和收货地址都要填。");
      return;
    }
    localStorage.setItem(KEY, JSON.stringify(draft));
    const extra = [
      draft.country.trim() && `收货国家：${draft.country.trim()}`,
      draft.phone.trim() && `电话：${draft.phone.trim()}`,
    ]
      .filter(Boolean)
      .join("；");
    onSubmit(
      `我要买 ${itemRef(item)} × ${qty}。收件人：${draft.recipient.trim()}；收货地址：${draft.address.trim()}` +
        (extra ? `；${extra}` : "") +
        "。请先给我确认卡，我确认后再下单。",
    );
    onClose();
  };

  return (
    <Modal open title="填写收件信息" onClose={onClose}>
      <form className="intent-form" onSubmit={submit}>
        <div className="intent-item">
          <span className="intent-item-title">{item.title}</span>
          <label className="intent-qty">
            数量
            <input
              type="number"
              min={1}
              max={99}
              value={qty}
              onChange={(e) => setQty(Math.max(1, Math.min(99, Number(e.target.value) || 1)))}
            />
          </label>
        </div>
        <label>
          收件人 *
          <input value={draft.recipient} onChange={set("recipient")} placeholder="姓名" autoFocus />
        </label>
        <label>
          收货地址 *
          <input value={draft.address} onChange={set("address")} placeholder="国家 / 城市 / 街道门牌" />
        </label>
        <div className="intent-row">
          <label>
            国家（可选）
            <input value={draft.country} onChange={set("country")} placeholder="如 US / 日本" />
          </label>
          <label>
            电话（可选）
            <input value={draft.phone} onChange={set("phone")} placeholder="便于配送联系" />
          </label>
        </div>
        {err && <div className="intent-err">{err}</div>}
        <div className="intent-hint">
          提交后 Agent 会先出一张<strong>确认卡</strong>（不会直接下单），你看过再点确认。
        </div>
        <div className="compare-actions">
          <button type="button" className="btn-ghost" onClick={onClose}>
            取消
          </button>
          <button type="submit" className="btn-primary">
            生成确认卡
          </button>
        </div>
      </form>
    </Modal>
  );
}
