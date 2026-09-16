import { useEffect, useState } from "react";
import type { Comparison } from "../api";
import { compareItems } from "../api";
import type { ProductItem } from "../types";
import { Modal } from "./Modal";
import { platformName, priceKind, shownPrice, splitReasons } from "./productText";

// 商品对比：把用户勾选的几件并排放在一张表里（图 / 平台 / 价格口径 / 理由），一眼看差别。
// 上半张表全是前端本地已有的结构化字段，离线也在。
//
// 「哪件更值、各自适合谁」这一档判断由 C4 的 present_comparison 填进下半张表：点按钮直接打
// POST /api/threads/{tid}/compare（**不走 AgentLoop**，一次 fast 模型调用），结果按 item_id
// 逐列填、推荐那列高亮。此前这里是把几件拼成一句话发给 Agent，回来的是一段散文——表填不满，
// 用户还得自己在文字里找哪句说的是哪件。
export function ProductComparison({
  open,
  threadId,
  items,
  busy,
  onClose,
  onRemove,
  onClear,
}: {
  open: boolean;
  threadId: string | null;
  items: ProductItem[];
  busy: boolean;
  onClose: () => void;
  onRemove: (item: ProductItem) => void;
  onClear: () => void;
}) {
  const [verdict, setVerdict] = useState<Comparison | null>(null);
  const [asking, setAsking] = useState(false);

  // 勾选变了，上一次的结论就不再对应这几列了——留着会张冠李戴，直接清掉。
  useEffect(() => {
    setVerdict(null);
  }, [items.map((it) => it.item_id).join(",")]);

  const ask = async () => {
    if (!threadId) return;
    setAsking(true);
    const out = await compareItems(threadId, items.map((it) => it.item_id));
    setVerdict(out);
    setAsking(false);
  };

  const entry = (itemId: string) => verdict?.items.find((e) => e.item_id === itemId);
  const isPick = (itemId: string) => !!verdict?.recommended_item_id && verdict.recommended_item_id === itemId;
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
                    <th key={it.item_id} className={isPick(it.item_id) ? "compare-pick" : undefined}>
                      {isPick(it.item_id) && <div className="compare-badge">最推荐</div>}
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
                {verdict && (
                  <>
                    <tr>
                      <th>优势</th>
                      {items.map((it) => (
                        <td key={it.item_id} className={isPick(it.item_id) ? "compare-pick" : undefined}>
                          <ul className="compare-reasons">
                            {(entry(it.item_id)?.pros ?? []).map((p, i) => (
                              <li key={i}>{p}</li>
                            ))}
                          </ul>
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <th>短板</th>
                      {items.map((it) => {
                        const cons = entry(it.item_id)?.cons ?? [];
                        return (
                          <td key={it.item_id} className={isPick(it.item_id) ? "compare-pick" : undefined}>
                            {cons.length === 0 ? (
                              <span className="compare-muted">没查到明显短板</span>
                            ) : (
                              <ul className="compare-reasons">
                                {cons.map((c, i) => (
                                  <li key={i}>{c}</li>
                                ))}
                              </ul>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                    <tr>
                      <th>适合谁</th>
                      {items.map((it) => (
                        <td key={it.item_id} className={isPick(it.item_id) ? "compare-pick" : undefined}>
                          {entry(it.item_id)?.best_for || "—"}
                        </td>
                      ))}
                    </tr>
                  </>
                )}
              </tbody>
            </table>
          </div>
          {verdict?.recommendation_reason && (
            <div className="compare-verdict">
              <strong>更推荐：</strong>
              {items.find((it) => it.item_id === verdict.recommended_item_id)?.title ?? "—"}
              <div>{verdict.recommendation_reason}</div>
            </div>
          )}
          {verdict && !verdict.recommended_item_id && verdict.items.length > 0 && (
            <div className="compare-note">这几件各有取舍，没有一件明显更好——按上面的「适合谁」对号入座。</div>
          )}
          {(verdict?.note || mixed) && (
            <div className="compare-note">
              {verdict?.note ||
                "价格口径不一致（有的是到手价、有的只是货价），直接比数字会误导——要严格比总花费，先让 Agent 都算一遍到手价。"}
            </div>
          )}
          <div className="compare-actions">
            <button className="btn-ghost" onClick={onClear}>
              清空
            </button>
            <button className="btn-primary" disabled={busy || asking || items.length < 2} onClick={ask}>
              {asking ? "对比中…" : verdict ? "重新比一比" : "让 Agent 帮我比一比"}
            </button>
          </div>
        </>
      )}
    </Modal>
  );
}
