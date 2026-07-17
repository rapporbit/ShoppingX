import { useCallback, useEffect, useState } from "react";
import { fetchFavorites, removeFavorite } from "../api";
import type { ProductItem } from "../types";
import { CloseIcon, ExternalLinkIcon, RefreshIcon } from "./icons";

// 收藏抽屉（右侧滑出，与偏好管理同一套外壳）。
//
// **它和隔壁的「偏好管理」定位不同，但不再是「完全不影响 Agent」**：偏好被注入 prompt、显式改变
// Agent 的行为；收藏不进 prompt、不进长期偏好库，但会经 app.memory.affinity 聚合成一条**弱信号**——
// 被你收藏 ≥2 次的属性（材质/功能/风格）会在精挑里给同类候选**小幅加分**（零 LLM、只加分不淘汰）。
// 所以它首先仍是「我看中过的东西」的回看清单，其次才顺带把行为轻轻喂给排序。这是刻意的克制：收藏
// 一件推不出可靠偏好（可能只是想再比比价），所以只做弱加分，绝不据此淘汰任何候选。
//
// 后端存的是**整张卡的快照**（title/价格/图/链接），不是只存 item_id——收藏跨会话长期留着，
// 而候选池随会话清理，换个会话按 id 早就捞不回商品了。
type FavoritesDrawerProps = {
  userId: string;
  open: boolean;
  refreshKey: number;
  onClose: () => void;
  onChanged: (items: ProductItem[]) => void;
};

export function FavoritesDrawer({
  userId,
  open,
  refreshKey,
  onClose,
  onChanged,
}: FavoritesDrawerProps) {
  const [items, setItems] = useState<ProductItem[]>([]);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    const favs = await fetchFavorites(userId);
    setItems(favs);
    onChanged(favs); // 抽屉是真源：拉到的列表同步回 App，让卡上的 ♡ 实心态跟着对
    setLoading(false);
  }, [userId, onChanged]);

  useEffect(() => {
    if (open) void load();
  }, [open, refreshKey, load]);

  const drop = async (itemId: string) => {
    await removeFavorite(userId, itemId);
    await load();
  };

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer drawer-wide ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">
            <span className="fav-glyph">♥</span>
            我的收藏
          </div>
          <div className="drawer-tools">
            <button className="icon-btn" onClick={() => void load()} disabled={loading} title="刷新">
              <RefreshIcon width={16} height={16} className={loading ? "spin" : ""} />
            </button>
            <button className="icon-btn" onClick={onClose} title="关闭">
              <CloseIcon width={18} height={18} />
            </button>
          </div>
        </div>

        <div className="drawer-user">用户：{userId}</div>
        <div className="fav-note">
          收藏首先是给你自己回看的清单；同时收藏多了，同类属性会<b>轻微</b>影响精挑排序（只上浮、不淘汰）。想明确换一批，直接说更准。
        </div>

        {items.length === 0 ? (
          <div className="drawer-empty">
            还没有收藏。在商品卡右上角点 ♡ 就能存进来，换个会话也还在。
          </div>
        ) : (
          <ul className="fav-list">
            {items.map((it) => (
              <li key={it.item_id} className="fav-row">
                {it.image_url ? (
                  <img
                    className="fav-thumb"
                    src={it.image_url}
                    alt={it.title}
                    loading="lazy"
                    referrerPolicy="no-referrer"
                  />
                ) : (
                  <div className="fav-thumb placeholder">🛍️</div>
                )}
                <div className="fav-main">
                  <div className="fav-title" title={it.title}>
                    {it.title}
                  </div>
                  <div className="fav-meta">
                    {typeof it.landed_usd === "number"
                      ? `$${it.landed_usd.toFixed(2)} 到手价`
                      : typeof it.price_usd === "number"
                        ? `$${it.price_usd.toFixed(2)} 货价`
                        : "价格未知"}
                    <span className="fav-platform">{it.platform}</span>
                  </div>
                </div>
                <div className="fav-acts">
                  {it.url && (
                    <a
                      className="icon-btn"
                      href={it.url}
                      target="_blank"
                      rel="noreferrer noopener"
                      title="打开商品页"
                    >
                      <ExternalLinkIcon width={15} height={15} />
                    </a>
                  )}
                  <button
                    className="icon-btn"
                    onClick={() => void drop(it.item_id)}
                    title="取消收藏"
                  >
                    <CloseIcon width={16} height={16} />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </aside>
    </>
  );
}
