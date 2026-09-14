import { memo, useMemo, useState } from "react";
import type { ProductItem } from "../types";
import { CheckIcon, ExternalLinkIcon, GlobeIcon, SearchIcon } from "./icons";
import { platformName, shownPrice, splitReasons } from "./productText";

// 商品结果区（复刻 Accio）：平台筛选胶囊 + 商品卡网格。卡片把 shopping_summary 随 task_result
// 下发的结构化精选商品（平台 / 到手价 / 选购理由 / 商品图）呈现成「看得见」的卡。
//
// 图区优先显示数据集里的真实商品图（image_url）；URL 缺失或加载失败时，才回退到「按 item_id
// 派生的稳定柔和渐变 + 平台名」占位——不伪造图片，也不让裂图破坏版式。

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

// 入场错峰：前 6 张按序号各晚 55ms 出现，后面的一起出（CSS .product-card 的 card-in 动画读这个
// 变量；prefers-reduced-motion 下动画整体关掉）。
function staggerStyle(index: number): React.CSSProperties {
  return { animationDelay: `${Math.min(index, 5) * 55}ms` };
}

// 单卡按 memo 包起来：收尾文案是流式逐字推的（summary_delta），每一段都让父组件重渲——
// 卡片的 props（item 引用 / 收藏态 / 回调）不变时不必跟着重绘整张卡。回调由 App 用 useCallback
// 固定引用，否则 memo 形同虚设。
const Card = memo(function Card({
  item,
  index = 0,
  favorited,
  compared,
  onFavorite,
  onSimilar,
  onDetail,
  onCompare,
}: {
  item: ProductItem;
  index?: number;
  favorited: boolean;
  compared: boolean;
  onFavorite: (item: ProductItem, undo: boolean) => void;
  onSimilar: (item: ProductItem) => void;
  onDetail: (item: ProductItem) => void;
  onCompare: (item: ProductItem) => void;
}) {
  const reasons = splitReasons(item.reason);

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

  const stop = (fn: () => void) => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    fn();
  };

  const openSimilar = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onSimilar(item);
  };

  return (
    <Wrapper
      className={`product-card ${href ? "clickable" : ""}`}
      style={staggerStyle(index)}
      {...linkProps}
    >
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
        {/* 顶行：品牌 + 评分。评分只给分不给评价数（数据集评价数恒为 0，显示「(0)」等于说零评价）；
            两者都没有就整行不渲染，标题顶上去。 */}
        {(item.brand || typeof item.rating === "number") && (
          <div className="card-topline">
            <span className="card-brand">{item.brand || ""}</span>
            {typeof item.rating === "number" && (
              <span className="card-rating" title="平台评分（离线数据集）">
                <i aria-hidden>★</i> {item.rating.toFixed(1)}
              </span>
            )}
          </div>
        )}
        <h4 className="card-title" title={item.title}>
          {item.title}
        </h4>

        {/* 本轮跑过 shipping_calc 才有到手价（含税运）；没跑就只有货价——照实标注，不冒充到手价。
            到手价只在某个收货国下成立，标上「寄往 XX」；只有货价时另给一行灰字说明到手价还没估。 */}
        {typeof item.landed_usd === "number" ? (
          <div className="card-price">
            <span className="price-num">${item.landed_usd.toFixed(2)}</span>
            <span className="price-label">
              到手价
              {item.dest_country ? ` · 寄往 ${item.dest_country}` : ""}
              （含税运）
            </span>
          </div>
        ) : (
          typeof item.price_usd === "number" && (
            <>
              <div className="card-price">
                <span className="price-num">${item.price_usd.toFixed(2)}</span>
                <span className="price-label">货价（未含税运）</span>
              </div>
              <div className="card-landed-pending">到手价待收货地与税运估算</div>
            </>
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

        {/* 卡内动作：详情 / 对比 / 搜同款。整卡是 <a>，三个按钮都得自己吃掉点击。
            搜同款是一次纯向量近邻检索（不过 Agent、不烧 LLM），结果在右侧抽屉里给。 */}
        <div className="card-actions">
          <button className="card-similar" onClick={stop(() => onDetail(item))} title="看大图与全部理由">
            详情
          </button>
          <button
            className={`card-similar ${compared ? "on" : ""}`}
            aria-pressed={compared}
            onClick={stop(() => onCompare(item))}
            title={compared ? "移出对比" : "加入对比（最多 4 件）"}
          >
            {compared ? "✓ 对比中" : "对比"}
          </button>
          <button className="card-similar" onClick={openSimilar} title="按商品向量找相似商品">
            <SearchIcon width={13} height={13} />
            搜同款
          </button>
        </div>
      </div>
    </Wrapper>
  );
});

// 槽位轮的分组视图：按槽位（床品 / 台灯 / …）分节渲染，组头带槽名与该槽花费。平台胶囊在
// 这里没有意义（分组本身就是筛选维度），整个替换掉。
//
// 两种形态共用这个视图，差别只在底部那行合计：「一套齐」要给总价（用户关心这一套多少钱），
// 「多类并列」（跑鞋 + 耳机）**不给**——把互不相干的几类加总，那个数字没有任何含义，还会
// 让用户以为得一起买。
function BundleGroups({
  items,
  favorited,
  compared,
  onFavorite,
  onSimilar,
  onDetail,
  onCompare,
}: {
  items: ProductItem[];
  favorited: Set<string>;
  compared: Set<string>;
  onFavorite: (item: ProductItem, undo: boolean) => void;
  onSimilar: (item: ProductItem) => void;
  onDetail: (item: ProductItem) => void;
  onCompare: (item: ProductItem) => void;
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
  const parallel = items.some((it) => it.slot_mode === "parallel");
  const Total = () =>
    parallel || priced.length === 0 ? null : (
      <div className="bundle-total">
        这一套合计约 <strong>${total.toFixed(2)}</strong>
        {priced.length < items.length && "（个别商品缺价格，未计入）"}
      </div>
    );

  // 套装常态是组合优选每槽恰好一件——每组还配整宽组头就是 N 个「大标题配孤卡」，
  // 右边 3/4 全空白。此时退化成一个统一网格，槽名改做卡片顶上的小标签；
  // 只有某槽真有多件备选时才值得分节。
  if (groups.every((g) => g.items.length === 1)) {
    return (
      <section className="results">
        <div className="product-grid">
          {groups.map((g, i) => (
            <div className="bundle-cell" key={g.slot}>
              <div className="bundle-cell-slot">{g.slot}</div>
              <Card
                item={g.items[0]}
                index={i}
                favorited={favorited.has(g.items[0].item_id)}
                compared={compared.has(g.items[0].item_id)}
                onFavorite={onFavorite}
                onSimilar={onSimilar}
                onDetail={onDetail}
                onCompare={onCompare}
              />
            </div>
          ))}
        </div>
        <Total />
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
              {g.items.map((it, i) => (
                <Card
                  key={`${it.platform}-${it.item_id}`}
                  item={it}
                  index={i}
                  favorited={favorited.has(it.item_id)}
                  compared={compared.has(it.item_id)}
                  onFavorite={onFavorite}
                  onSimilar={onSimilar}
                  onDetail={onDetail}
                  onCompare={onCompare}
                />
              ))}
            </div>
          </div>
        );
      })}
      {/* 合计由卡片价格求和，与用户眼前的数字必然一致；预算与剩余在上方收尾文案里。
          口径混标时（部分到手价 / 部分货价）用「约」弱化，卡片各自的标注才是权威。
          并列形态不显示（见 BundleGroups 头部注释）。 */}
      <Total />
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
  compared,
  onFavorite,
  onSimilar,
  onDetail,
  onCompare,
}: {
  items: ProductItem[];
  favorited: Set<string>;
  compared: Set<string>;
  onFavorite: (item: ProductItem, undo: boolean) => void;
  onSimilar: (item: ProductItem) => void;
  onDetail: (item: ProductItem) => void;
  onCompare: (item: ProductItem) => void;
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
        compared={compared}
        onFavorite={onFavorite}
        onSimilar={onSimilar}
                onDetail={onDetail}
                onCompare={onCompare}
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
        {shown.map((it, i) => (
          <Card
            key={`${it.platform}-${it.item_id}`}
            item={it}
            index={i}
            favorited={favorited.has(it.item_id)}
            compared={compared.has(it.item_id)}
            onFavorite={onFavorite}
            onSimilar={onSimilar}
            onDetail={onDetail}
            onCompare={onCompare}
          />
        ))}
      </div>
    </section>
  );
}
