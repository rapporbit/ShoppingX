import { domainLabel, isGlobalDomain } from "../domains";
import type { Preference } from "../types";

// 列表里的一条偏好——**把结构化字段全摊开**，而不是只显示一句 content。
// 用户要能一眼看出「这条到底会怎么影响推荐」：是直接淘汰还是只压低排序、拿哪些词去匹配、
// 在哪个品类生效。这些字段就是 Agent 真正的行为依据，藏起来就谈不上透明。

const CATEGORY_CN: Record<string, string> = {
  material: "材质",
  style: "风格",
  brand: "品牌",
  budget: "预算",
  color: "颜色",
  size: "尺寸",
  location: "收货地",
  other: "其它",
};

// 超过这个天数没被确认过，才提示「很久没用过了」。
// 纯提示、不参与任何打分——后端**故意**删掉了半衰期衰减：与其让系统按一个没人能解释的指数函数
// 把偏好偷偷打七折，不如把「久未复现」摆到用户眼前，由他自己决定删不删。
const STALE_DAYS = 90;

// 后端 SQLite 存的是 naive datetime，ISO 串没有时区后缀（如 2026-01-05T10:00:00）——
// 补个 Z 按 UTC 解析，否则会被当成本地时间，在 UTC+ 时区里算出偏小的天数。
function daysSince(iso: string): number {
  const withTz = /(Z|[+-]\d{2}:?\d{2})$/.test(iso) ? iso : `${iso}Z`;
  const ms = Date.now() - new Date(withTz).getTime();
  return Number.isFinite(ms) ? Math.floor(ms / 86_400_000) : 0;
}

function staleHint(pref: Preference): string | null {
  const days = daysSince(pref.last_confirmed_at);
  if (days < STALE_DAYS) return null;
  const months = Math.floor(days / 30);
  return months >= 12 ? "一年多没用过了" : `${months} 个月没用过了`;
}

// 一句人话解释这条偏好的实际效果（结构化字段 → 行为）。
// dislike 的两档分界不是「LLM 猜的力度」而是**谁说的**：用户亲手勾了「绝不推荐」才有权淘汰商品，
// Agent 学到的一律只减分。like 走的是检索词通路（keywords 拼进 query），不是加分项。
function effectOf(p: Preference): string {
  if (p.polarity === "dislike") return p.blocking ? "命中即淘汰" : "命中则减分";
  return "关键词进检索";
}

type PreferenceItemProps = {
  pref: Preference;
  onEdit: () => void;
  onDelete: () => void;
};

export function PreferenceItem({ pref, onEdit, onDelete }: PreferenceItemProps) {
  const stale = staleHint(pref);
  const global = isGlobalDomain(pref.domain);

  return (
    <li className={`pref-item pref-${pref.polarity}`}>
      <span className="pref-mark">{pref.polarity === "dislike" ? "🚫" : "❤️"}</span>
      <div className="pref-text">
        <span className="pref-content">{pref.content}</span>

        <div className="pref-meta">
          <span className="pref-cat">{CATEGORY_CN[pref.category] ?? pref.category}</span>
          <span className={`pref-effect ${pref.blocking ? "pref-blocking" : "pref-soft"}`}>
            {effectOf(pref)}
          </span>
          {/* 生效范围：global 是唯一会跨品类杀商品的档位，必须和「仅某品类」在视觉上区分开 */}
          <span
            className={`pref-scope ${global ? "pref-scope-global" : ""}`}
            title={
              global
                ? "跨品类底线：买任何东西时都生效"
                : pref.domain === "other"
                  ? "判不出品类：这条几乎不生效，建议改成具体品类"
                  : `只在买${domainLabel(pref.domain)}类商品时生效`
            }
          >
            {global ? "全局生效" : `仅 ${domainLabel(pref.domain)}`}
          </span>
          <span className="pref-src">{pref.source === "user" ? "✍️ 你填的" : "🤖 学到的"}</span>
          {/* 久未复现只提示、不衰减打分。删不删由用户定，系统不替他做主 */}
          {stale && (
            <span className="pref-stale" title="它仍在全额生效。不需要了就删掉">
              ⏳ {stale}
            </span>
          )}
        </div>

        {/* keywords 才是 item_picker 真正拿去匹配的原子词——单独一行摊开，别混在灰色小字里 */}
        {pref.keywords.length > 0 && (
          <div className="pref-kw">
            {pref.keywords.map((k) => (
              <span key={k} className="pref-kw-chip">
                {k}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="pref-ops">
        <button className="pref-op" title="编辑这条偏好" onClick={onEdit}>
          ✎
        </button>
        <button className="pref-op pref-del" title="删除这条偏好" onClick={onDelete}>
          ✕
        </button>
      </div>
    </li>
  );
}
