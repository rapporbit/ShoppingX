import { useEffect, useState } from "react";
import { fetchSimilar } from "../api";
import type { ProductItem } from "../types";
import { CloseIcon, ExternalLinkIcon } from "./icons";

// 「搜同款」抽屉（右侧滑出，与收藏 / 偏好共用一套外壳）。
//
// 和隔壁两个抽屉的**性质**很不一样：这里的内容不是存量数据，而是点开那一刻现算的一次向量近邻
// 检索——拿源商品在召回库里的那条向量去找全库最像的几件。**不过 Agent、不烧 LLM**，亚秒级。
//
// 诚实边界：库里 amazon 占了绝大多数，所以近邻结果多半也是 amazon 的商品——它是「同款/相似款
// 还有哪些卖法」，不是跨平台比价（那条路请直接跟 Agent 说）。价格只有货价，没有到手价：到手价
// 要跑 shipping_calc 算关税运费，不是这条纯检索通路该做的事。
type SimilarDrawerProps = {
  source: ProductItem | null; // 点了哪张卡的「搜同款」；null = 抽屉关着
  onClose: () => void;
};

export function SimilarDrawer({ source, onClose }: SimilarDrawerProps) {
  const [items, setItems] = useState<ProductItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!source) return;
    let alive = true; // 快速连点两张卡的「搜同款」时，别让先回来的旧结果覆盖后点的那张
    setLoading(true);
    setItems([]);
    void fetchSimilar(source.item_id).then((res) => {
      if (!alive) return;
      setItems(res);
      setLoading(false);
    });
    return () => {
      alive = false;
    };
  }, [source]);

  const open = Boolean(source);

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      {/* 比收藏/偏好抽屉宽得多：这里是「看图辨同款」的场景，460px 一列塞不下能看清的图。 */}
      <aside className={`drawer drawer-similar ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">
            <span className="fav-glyph">⌕</span>
            相似商品
          </div>
          <div className="drawer-tools">
            <button className="icon-btn" onClick={onClose} title="关闭">
              <CloseIcon width={18} height={18} />
            </button>
          </div>
        </div>

        {source && (
          <div className="similar-source">
            {source.image_url && (
              <img
                className="similar-source-thumb"
                src={source.image_url}
                alt={source.title}
                referrerPolicy="no-referrer"
              />
            )}
            <div className="similar-source-main">
              <div className="similar-source-label">以此为准</div>
              <div className="similar-source-title" title={source.title}>
                {source.title}
              </div>
              <div className="similar-source-note">
                按商品向量在召回库里找的近邻，<b>没有</b>经过 Agent 挑选，价格是货价（未含税运）。
              </div>
            </div>
          </div>
        )}

        {loading ? (
          <div className="drawer-empty">正在找相似商品…</div>
        ) : items.length === 0 ? (
          <div className="drawer-empty">
            没找到相似商品——这件商品可能不在当前召回库里（比如换过库的老收藏）。
          </div>
        ) : (
          // 两列网格、上图下文：同款靠「看图」辨认，图必须够大——列表行里那种 44px 缩略图
          // 只能看出个色块。整卡可点（有 url 时）直接开商品页，不再单独放一个跳转小图标。
          <ul className="similar-grid">
            {items.map((it) => {
              const Wrapper = it.url ? "a" : "div";
              const linkProps = it.url
                ? { href: it.url, target: "_blank" as const, rel: "noreferrer noopener" }
                : {};
              return (
                <li key={it.item_id}>
                  <Wrapper className={`similar-card ${it.url ? "clickable" : ""}`} {...linkProps}>
                    <div className="similar-thumb">
                      {it.image_url ? (
                        <img
                          src={it.image_url}
                          alt={it.title}
                          loading="lazy"
                          referrerPolicy="no-referrer"
                        />
                      ) : (
                        <span aria-hidden>🛍️</span>
                      )}
                      {typeof it.score === "number" && (
                        <span className="similar-score">相似度 {it.score.toFixed(2)}</span>
                      )}
                    </div>
                    <div className="similar-body">
                      <div className="similar-title" title={it.title}>
                        {it.title}
                      </div>
                      <div className="similar-meta">
                        <span className="similar-price">
                          {typeof it.price_usd === "number"
                            ? `$${it.price_usd.toFixed(2)}`
                            : "价格未知"}
                        </span>
                        {typeof it.price_usd === "number" && (
                          <span className="similar-price-label">货价</span>
                        )}
                        <span className="similar-platform">{it.platform}</span>
                        {it.url && <ExternalLinkIcon width={13} height={13} className="similar-go" />}
                      </div>
                    </div>
                  </Wrapper>
                </li>
              );
            })}
          </ul>
        )}
      </aside>
    </>
  );
}
