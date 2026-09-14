import type { ProductItem } from "../types";
import { Modal } from "./Modal";
import { itemRef, platformName, priceKind, shownPrice, splitReasons } from "./productText";

// 商品对比：把用户勾选的几件并排放在一张表里（图 / 平台 / 价格口径 / 理由），一眼看差别。
// 表里只有前端已有的结构化字段；「哪件更值」这种判断交回 Agent——底部按钮把这几件的 item_id
// 组成一句话发出去，让 price_compare / shipping_calc 在会话内跑，结果仍以对话与商品卡呈现。
export function ProductComparison({
  open,
  items,
  busy,
  onClose,
  onRemove,
  onClear,
  onAsk,
}: {
  open: boolean;
  items: ProductItem[];
  busy: boolean;
  onClose: () => void;
  onRemove: (item: ProductItem) => void;
  onClear: () => void;
  onAsk: (text: string) => void;
}) {
  const ask = () => {
    const refs = items.map((it, i) => `${i + 1}. ${itemRef(it)}`).join("\n");
    onAsk(`请对比下面这几件商品的价格、到手价（含税运）和各自优缺点，最后告诉我更推荐哪一件、为什么：\n${refs}`);
    onClose();
  };
  const mixed = new Set(items.map(priceKind).filter(Boolean)).size > 1;

  return (
    <Modal open={open} title={`商品对比（${items.length} 件）`} wide onClose={onClose}>
      {items.length === 0 ? (
        <div className="drawer-empty">还没有加入对比的商品。在商品卡上点「对比」勾选 2～4 件。</div>
      ) : (
        <>
          <div className="compare-scroll">
            <table className="compare-table">
              <thead>
                <tr>
                  <th />
                  {items.map((it) => (
                    <th key={it.item_id}>
                      <div className="compare-thumb">
                        {it.image_url ? (
                          <img src={it.image_url} alt="" referrerPolicy="no-referrer" />
                        ) : (
                          <span aria-hidden>🛍️</span>
                        )}
                      </div>
                      <div className="compare-title" title={it.title}>
                        {it.title}
                      </div>
                      <button className="compare-remove" onClick={() => onRemove(it)}>
                        移出
                      </button>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th>平台</th>
                  {items.map((it) => (
                    <td key={it.item_id}>{platformName(it.platform)}</td>
                  ))}
                </tr>
                <tr>
                  <th>价格</th>
                  {items.map((it) => {
                    const p = shownPrice(it);
                    const k = priceKind(it);
                    return (
                      <td key={it.item_id}>
                        {p == null ? "—" : `$${p.toFixed(2)}`}
                        {k && <div className="price-label">{k === "landed" ? "到手价" : "货价"}</div>}
                      </td>
                    );
                  })}
                </tr>
                {items.some((it) => it.slot) && (
                  <tr>
                    <th>槽位</th>
                    {items.map((it) => (
                      <td key={it.item_id}>{it.slot ?? "—"}</td>
                    ))}
                  </tr>
                )}
                <tr>
                  <th>选购理由</th>
                  {items.map((it) => (
                    <td key={it.item_id}>
                      <ul className="compare-reasons">
                        {splitReasons(it.reason).map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    </td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          {mixed && (
            <div className="compare-note">
              价格口径不一致（有的是到手价、有的只是货价），直接比数字会误导——点下面让 Agent 统一算到手价再比。
            </div>
          )}
          <div className="compare-actions">
            <button className="btn-ghost" onClick={onClear}>
              清空
            </button>
            <button className="btn-primary" disabled={busy || items.length < 2} onClick={ask}>
              让 Agent 帮我比一比
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}
