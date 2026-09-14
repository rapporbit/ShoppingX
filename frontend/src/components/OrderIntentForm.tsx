import { useEffect, useState, type FormEvent } from "react";
import type { PrepareOrderInput, ProductItem, ShippingAddress } from "../types";
import { Modal } from "./Modal";
import { platformName } from "./productText";

// 下单意向表单（对齐参考项目 OrderIntentForm）：收件信息 + 数量一次填齐，提交**直连**
// POST /api/threads/{id}/confirmations/orders 由服务端生成确认卡，不经模型、零 LLM 往返。
// 候选校验 / 幂等 / 归属 / 有效期这些闸全在服务端那条路上，表单省掉的只是追问的来回。
// 收件信息记在 localStorage，下次自动带出（只存本机，不上传到偏好库）。
const KEY = "shoppingx.order.address.v2";

const EMPTY: ShippingAddress = {
  recipient_name: "",
  country: "",
  state: "",
  city: "",
  address_line: "",
  postal_code: "",
  phone: "",
};

function loadDraft(defaultCountry: string): ShippingAddress {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) return { ...EMPTY, ...JSON.parse(raw) };
  } catch {
    /* 坏数据当没有 */
  }
  return { ...EMPTY, country: defaultCountry };
}

export function OrderIntentForm({
  item,
  busy,
  error,
  onClose,
  onPrepare,
}: {
  item: ProductItem | null;
  busy: boolean;
  error: string | null;
  onClose: () => void;
  onPrepare: (input: PrepareOrderInput) => Promise<boolean>;
}) {
  const [addr, setAddr] = useState<ShippingAddress>(() => loadDraft(item?.dest_country || "CN"));
  const [qty, setQty] = useState("1");
  const [localErr, setLocalErr] = useState<string | null>(null);

  useEffect(() => {
    if (item) {
      setQty("1");
      setLocalErr(null);
      setAddr(loadDraft(item.dest_country || "CN"));
    }
  }, [item]);

  if (!item) return null;
  const field = (k: keyof ShippingAddress) =>
    (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) =>
      setAddr((a) => ({ ...a, [k]: e.target.value }));
  const q = Number(qty);
  const qtyOk = qty.trim() !== "" && Number.isSafeInteger(q) && q > 0 && q <= 99;
  const price =
    typeof item.landed_usd === "number"
      ? `$${item.landed_usd.toFixed(2)} 到手价`
      : typeof item.price_usd === "number"
        ? `$${item.price_usd.toFixed(2)} 货价`
        : "";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (busy) return;
    const n = Object.fromEntries(
      Object.entries(addr).map(([k, v]) => [k, v.trim()]),
    ) as unknown as ShippingAddress;
    if (![n.recipient_name, n.country, n.city, n.address_line].every(Boolean)) {
      setLocalErr("请补全收件人、国家或地区、城市与详细地址。");
      return;
    }
    if (!qtyOk) {
      setLocalErr("数量要是 1～99 的整数。");
      return;
    }
    setLocalErr(null);
    localStorage.setItem(KEY, JSON.stringify(n));
    const ok = await onPrepare({
      items: [{ item_id: item.item_id, quantity: q }],
      shipping_address: n,
    });
    if (ok) onClose();
  };

  return (
    <Modal open title="填写收件信息" onClose={() => !busy && onClose()}>
      <form className="intent-form" onSubmit={(e) => void submit(e)}>
        <div className="intent-item">
          <div>
            <span className="intent-item-title">{item.title}</span>
            <small className="intent-item-meta">
              {platformName(item.platform)}
              {price ? ` · ${price}` : ""} · 确认卡金额按平台原币种货价算，不含税运
            </small>
          </div>
          <label className="intent-qty">
            数量
            <input type="number" min={1} max={99} step={1} value={qty} inputMode="numeric"
              onChange={(e) => setQty(e.target.value)} disabled={busy} />
          </label>
        </div>
        <fieldset disabled={busy}>
          <label>
            收件人 *
            <input value={addr.recipient_name} onChange={field("recipient_name")} autoComplete="shipping name"
              maxLength={100} placeholder="收件人姓名" autoFocus />
          </label>
          <div className="intent-row">
            <label>
              国家或地区 *
              <input value={addr.country} onChange={field("country")} autoComplete="shipping country"
                maxLength={40} placeholder="如 CN / US / 日本" />
            </label>
            <label>
              省 / 州
              <input value={addr.state} onChange={field("state")} autoComplete="shipping address-level1" maxLength={100} />
            </label>
          </div>
          <label>
            城市 *
            <input value={addr.city} onChange={field("city")} autoComplete="shipping address-level2" maxLength={100} />
          </label>
          <label>
            详细地址 *
            <textarea value={addr.address_line} onChange={field("address_line")} autoComplete="shipping street-address"
              maxLength={500} rows={2} placeholder="街道、楼栋与门牌号" />
          </label>
          <div className="intent-row">
            <label>
              邮政编码
              <input value={addr.postal_code} onChange={field("postal_code")} autoComplete="shipping postal-code" maxLength={30} />
            </label>
            <label>
              联系电话
              <input type="tel" value={addr.phone} onChange={field("phone")} autoComplete="shipping tel" maxLength={40} />
            </label>
          </div>
        </fieldset>
        {(localErr || error) && <div className="intent-err">{localErr || error}</div>}
        <div className="intent-hint">
          提交后服务端生成一张<strong>确认卡</strong>，不会下单；你在卡上点「确认下单」才算数。
        </div>
        <div className="compare-actions">
          <button type="button" className="btn-ghost" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button type="submit" className="btn-primary" disabled={busy || !qtyOk}>
            {busy ? "正在核对并生成…" : "生成确认卡"}
          </button>
        </div>
      </form>
    </Modal>
  );
}
