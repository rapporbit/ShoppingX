import { useMemo, useState } from "react";
import type { ProductItem } from "../types";
import { CheckIcon, ExternalLinkIcon, GlobeIcon, SearchIcon } from "./icons";

// 商品结果区（复刻 Accio）：平台筛选胶囊 + 商品卡网格。卡片把 shopping_summary 随 task_result
// 下发的结构化精选商品（平台 / 到手价 / 选购理由 / 商品图）呈现成「看得见」的卡。
//
// 图区优先显示数据集里的真实商品图（image_url）；URL 缺失或加载失败时，才回退到「按 item_id
// 派生的稳定柔和渐变 + 平台名」占位——不伪造图片，也不让裂图破坏版式。

// 与后端 app/utils/clean.py 的 PLATFORMS 对齐（eBay 在清洗阶段整体剔除，库里没有它的商品）。
const PLATFORM_LABEL: Record<string, string> = {
  amazon: "Amazon",
  lazada: "Lazada",
  shein: "SHEIN",
  shopee: "Shopee",
  walmart: "Walmart",
};

function platformName(p: string): string {
  return PLATFORM_LABEL[p.toLowerCase()] ?? p;
}

// 由 item_id 派生稳定色相，给图区一个不抖动的柔和渐变（同一商品每次渲染一致）。
function hueFrom(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) % 360;
  return h;
}

function Thumb({ item }: { item: ProductItem }) {
  // 真实图加载失败（热链被拒 / 死链）就翻到占位，避免裂图。每件卡独立记一份失败态。
  const [failed, setFailed] = useState(false);
  const hue = hueFrom(item.item_id || item.title);
  const bg = `linear-gradient(135deg, hsl(${hue} 55% 92%), hsl(${(hue + 40) % 360} 50% 86%))`;
  const showImage = Boolean(item.image_url) && !failed;

  return (
    <div className="card-thumb" style={{ background: bg }}>
      {showImage ? (
        <img
          className="thumb-img"
          src={item.image_url}
          alt={item.title}
          loading="lazy"
          referrerPolicy="no-referrer"
          onError={() => setFailed(true)}
        />
      ) : (
        <span className="thumb-glyph" aria-hidden>
          🛍️
        </span>
      )}
      <span className="thumb-platform">{platformName(item.platform)}</span>
    </div>
  );
}

function Card({
  item,
  favorited,
  onFavorite,
  onSimilar,
}: {
  item: ProductItem;
  favorited: boolean;
  onFavorite: (item: ProductItem, undo: boolean) => void;
  onSimilar: (item: ProductItem) => void;
}) {
  const reasons = (item.reason ?? "")
    .split(/[；;\n]+/)
    .map((s) => s.trim())
    .filter(Boolean);

  // 有商品页 URL 才让整卡可点：渲染成新标签页打开的链接（外站，带 noreferrer）。无 URL 退化为
  // 普通 article（不可点）——离线数据偶有缺链，宁可不可点也不给死链。
  const href = item.url?.trim();
  const Wrapper = href ? "a" : "article";
  const linkProps = href
    ? { href, target: "_blank" as const, rel: "noreferrer noopener" }
    : {};

  // 整卡是 <a>：卡内的按钮必须自己吃掉点击，否则点它会顺带跳到外站商品页。
  const toggleFavorite = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onFavorite(item, favorited);
  };

  const openSimilar = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onSimilar(item);
  };

  return (
    <Wrapper className={`product-card ${href ? "clickable" : ""}`} {...linkProps}>
      <button
        className={`card-fav ${favorited ? "on" : ""}`}
        title={favorited ? "取消收藏" : "收藏"}
        aria-pressed={favorited}
        onClick={toggleFavorite}
      >
        {favorited ? "♥" : "♡"}
      </button>
      <Thumb item={item} />
      {href && (
        <span className="card-visit" aria-hidden>
          <ExternalLinkIcon width={13} height={13} />
          在 {platformName(item.platform)} 查看
        </span>
      )}
      <div className="card-body">
        <h4 className="card-title" title={item.title}>
          {item.title}
        </h4>

        {/* 本轮跑过 shipping_calc 才有到手价（含税运）；没跑就只有货价——照实标注，不冒充到手价。 */}
        {typeof item.landed_usd === "number" ? (
          <div className="card-price">
            <span className="price-num">${item.landed_usd.toFixed(2)}</span>
            <span className="price-label">到手价（含税运）</span>
          </div>
        ) : (
          typeof item.price_usd === "number" && (
            <div className="card-price">
              <span className="price-num">${item.price_usd.toFixed(2)}</span>
              <span className="price-label">货价（未含税运）</span>
            </div>
          )
        )}

        {/* 只标平台，不加「官方 ✅ 认证」那类背书：商品来自离线数据集，没有任何一方为它背书。 */}
        <div className="card-supplier">
          <GlobeIcon width={14} height={14} />
          <span className="supplier-name">{platformName(item.platform)}</span>
        </div>

        {reasons.length > 0 && (
          <div className="card-match">
            <div className="match-head">
              <CheckIcon width={14} height={14} />
              选购理由
            </div>
            <ul>
              {reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          </div>
        )}

        {/* 搜同款：一次纯向量近邻检索（不过 Agent、不烧 LLM），结果在右侧抽屉里给。 */}
        <button className="card-similar" onClick={openSimilar} title="按商品向量找相似商品">
          <SearchIcon width={13} height={13} />
          搜同款
        </button>
      </div>
    </Wrapper>
  );
}

// 一件商品的展示价：优先到手价（含税运），没算过就用货价——只取数字，口径由卡片自己标注。
function shownPrice(it: ProductItem): number | null {
  if (typeof it.landed_usd === "number") return it.landed_usd;
  if (typeof it.price_usd === "number") return it.price_usd;
  return null;
}

// 「一套齐」套装轮的分组视图：按槽位（床品 / 台灯 / …）分节渲染，组头带槽名与该槽花费，
// 底部一行合计。平台胶囊在这里没有意义（一套里每槽一件、通常同平台），整个替换掉。
function BundleGroups({
  items,
  favorited,
  onFavorite,
  onSimilar,
}: {
  items: ProductItem[];
  favorited: Set<string>;
  onFavorite: (item: ProductItem, undo: boolean) => void;
  onSimilar: (item: ProductItem) => void;
}) {
  // 保序分组：槽的顺序 = 后端组合优选给出的顺序（essential 在前），不重排。
  const groups: { slot: string; items: ProductItem[] }[] = [];
  for (const it of items) {
    const slot = it.slot?.trim() || "其他";
    const g = groups.find((g) => g.slot === slot);
    if (g) g.items.push(it);
    else groups.push({ slot, items: [it] });
  }
  const priced = items.map(shownPrice).filter((p): p is number => p != null);
  const total = priced.reduce((s, p) => s + p, 0);

  // 套装常态是组合优选每槽恰好一件——每组还配整宽组头就是 N 个「大标题配孤卡」，
  // 右边 3/4 全空白。此时退化成一个统一网格，槽名改做卡片顶上的小标签；
  // 只有某槽真有多件备选时才值得分节。
  if (groups.every((g) => g.items.length === 1)) {
    return (
      <section className="results">
        <div className="product-grid">
          {groups.map((g) => (
            <div className="bundle-cell" key={g.slot}>
              <div className="bundle-cell-slot">{g.slot}</div>
              <Card
                item={g.items[0]}
                favorited={favorited.has(g.items[0].item_id)}
                onFavorite={onFavorite}
                onSimilar={onSimilar}
              />
            </div>
          ))}
        </div>
        {priced.length > 0 && (
          <div className="bundle-total">
            这一套合计约 <strong>${total.toFixed(2)}</strong>
            {priced.length < items.length && "（个别商品缺价格，未计入）"}
          </div>
        )}
      </section>
    );
  }

  return (
    <section className="results">
      {groups.map((g) => {
        const sub = g.items.map(shownPrice).filter((p): p is number => p != null);
        return (
          <div className="bundle-group" key={g.slot}>
            <div className="bundle-group-head">
              <span className="bundle-slot-name">{g.slot}</span>
              {sub.length > 0 && (
                <span className="bundle-slot-price">
                  ${sub.reduce((s, p) => s + p, 0).toFixed(2)}
                </span>
              )}
            </div>
            <div className="product-grid">
              {g.items.map((it) => (
                <Card
                  key={`${it.platform}-${it.item_id}`}
                  item={it}
                  favorited={favorited.has(it.item_id)}
                  onFavorite={onFavorite}
                  onSimilar={onSimilar}
                />
              ))}
            </div>
          </div>
        );
      })}
      {/* 合计由卡片价格求和，与用户眼前的数字必然一致；预算与剩余在上方收尾文案里。
          口径混标时（部分到手价 / 部分货价）用「约」弱化，卡片各自的标注才是权威。 */}
      {priced.length > 0 && (
        <div className="bundle-total">
          这一套合计约 <strong>${total.toFixed(2)}</strong>
          {priced.length < items.length && "（个别商品缺价格，未计入）"}
        </div>
      )}
    </section>
  );
}

// 收尾（task_result）下发的精选商品卡。卡右上角一个 ♡：把商品存进收藏夹，跨会话可回看。
// **它是弱信号，不是显式指令** —— 收藏只经行为亲和给同类属性一点加分；想让 Agent 换一批、别再推
// 某类东西，还是直接在对话框里说，那才带得上原因、也才泛化得了（「不要 Nike」比逐个点掉三双 Nike
// 有用得多，也才可能触发淘汰而非仅仅上浮）。
export function ProductCards({
  items,
  favorited,
  onFavorite,
  onSimilar,
}: {
  items: ProductItem[];
  favorited: Set<string>;
  onFavorite: (item: ProductItem, undo: boolean) => void;
  onSimilar: (item: ProductItem) => void;
}) {
  const [active, setActive] = useState<string>("all");

  const platforms = useMemo(() => {
    const seen = new Set<string>();
    for (const it of items) seen.add(it.platform.toLowerCase());
    return Array.from(seen);
  }, [items]);

  if (items.length === 0) return null;

  // 套装轮（任一卡带槽位名）→ 按槽分组视图，平台胶囊让位给槽位组头。
  if (items.some((it) => it.slot?.trim())) {
    return (
      <BundleGroups
        items={items}
        favorited={favorited}
        onFavorite={onFavorite}
        onSimilar={onSimilar}
      />
    );
  }

  const shown =
    active === "all" ? items : items.filter((it) => it.platform.toLowerCase() === active);

  return (
    <section className="results">
      <div className="results-tabs">
        <button
          className={`tab ${active === "all" ? "active" : ""}`}
          onClick={() => setActive("all")}
        >
          <GlobeIcon width={15} height={15} />
          Global sites
          <span className="tab-count">{items.length}</span>
        </button>
        {platforms.map((p) => {
          const n = items.filter((it) => it.platform.toLowerCase() === p).length;
          return (
            <button
              key={p}
              className={`tab ${active === p ? "active" : ""}`}
              onClick={() => setActive(p)}
            >
              {platformName(p)}
              <span className="tab-count">{n}</span>
            </button>
          );
        })}
      </div>

      <div className="product-grid">
        {shown.map((it) => (
          <Card
            key={`${it.platform}-${it.item_id}`}
            item={it}
            favorited={favorited.has(it.item_id)}
            onFavorite={onFavorite}
            onSimilar={onSimilar}
          />
        ))}
      </div>
    </section>
  );
}
