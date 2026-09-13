import type { ProductItem } from "../types";
import { ExternalLinkIcon, SearchIcon } from "./icons";
import { Modal } from "./Modal";
import { platformName, splitReasons } from "./productText";

// 商品详情弹窗：卡片上放不下的都在这——大图、完整标题、价格口径、全部选购理由，以及针对这一件
// 的四个动作（收藏 / 搜同款 / 加入对比 / 去下单）。数据仍是收尾下发的那份结构化商品，这里不
// 另外请求、不从文案里猜任何字段：卡片没有的信息，详情页也不会凭空长出来。
export function ProductDetail({
  item,
  favorited,
  compared,
  busy,
  onClose,
  onFavorite,
  onSimilar,
  onCompare,
  onOrder,
}: {
  item: ProductItem | null;
  favorited: boolean;
  compared: boolean;
  busy: boolean; // 任务跑着时禁掉「去下单」：那一步要发一句话，跑着的时候发不出去
  onClose: () => void;
  onFavorite: (item: ProductItem, undo: boolean) => void;
  onSimilar: (item: ProductItem) => void;
  onCompare: (item: ProductItem) => void;
  onOrder: (item: ProductItem) => void;
}) {
  if (!item) return null;
  const reasons = splitReasons(item.reason);
  const href = item.url?.trim();
  const price =
    typeof item.landed_usd === "number"
      ? { num: item.landed_usd, label: "到手价（含税运）" }
      : typeof item.price_usd === "number"
        ? { num: item.price_usd, label: "货价（未含税运）" }
        : null;

  return (
    <Modal open title="商品详情" onClose={onClose}>
      <div className="detail">
        <div className="detail-media">
          {item.image_url ? (
            <img src={item.image_url} alt={item.title} referrerPolicy="no-referrer" />
          ) : (
            <span className="thumb-glyph" aria-hidden>
              🛍️
            </span>
          )}
        </div>
        <div className="detail-main">
          <h3 className="detail-title">{item.title}</h3>
          <div className="detail-meta">
            <span className="supplier-name">{platformName(item.platform)}</span>
            {item.slot && <span className="detail-slot">槽位：{item.slot}</span>}
            <span className="detail-id">ID {item.item_id}</span>
          </div>
          {price ? (
            <div className="detail-price">
              <span className="price-num">${price.num.toFixed(2)}</span>
              <span className="price-label">{price.label}</span>
            </div>
          ) : (
            <div className="detail-price price-label">暂无价格</div>
          )}
          {reasons.length > 0 && (
            <div className="detail-reasons">
              <div className="match-head">选购理由</div>
              <ul>
                {reasons.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
            </div>
          )}
          <div className="detail-actions">
            <button className="btn-ghost" onClick={() => onFavorite(item, favorited)}>
              {favorited ? "♥ 已收藏" : "♡ 收藏"}
            </button>
            <button className="btn-ghost" onClick={() => onSimilar(item)}>
              <SearchIcon width={13} height={13} /> 搜同款
            </button>
            <button
              className={`btn-ghost ${compared ? "on" : ""}`}
              aria-pressed={compared}
              onClick={() => onCompare(item)}
            >
              {compared ? "✓ 已加入对比" : "＋ 加入对比"}
            </button>
            {href && (
              <a className="btn-ghost" href={href} target="_blank" rel="noreferrer noopener">
                <ExternalLinkIcon width={13} height={13} /> 在 {platformName(item.platform)} 查看
              </a>
            )}
            <button
              className="btn-primary"
              disabled={busy}
              title={busy ? "等这一轮跑完再下单" : "填写收件信息，让 Agent 先出确认卡"}
              onClick={() => onOrder(item)}
            >
              去下单
            </button>
          </div>
        </div>
      </div>
    </Modal>
  );
}
