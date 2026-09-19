import type { MemoryCategory, Preference } from "../types";

// 列表里的一条长期记忆——显示的就是注入给模型的那三个字段（`[category] key: value`）。
// 页面上看得见的，和模型读得到的是同一份东西；上一版摊开的 polarity / blocking / domain /
// keywords 随 M4 一起删了，那些字段模型根本没见过。

const CATEGORY_CN: Record<MemoryCategory, string> = {
  constraint: "硬规则",
  preference: "取向",
  context: "背景",
};

const CATEGORY_HINT: Record<MemoryCategory, string> = {
  constraint: "每轮都会读到：推荐必须遵守",
  preference: "按新鲜度补位注入，影响挑选倾向",
  context: "按新鲜度补位注入的身份 / 家庭 / 账户背景",
};

// 超过这个天数没更新过，才提示「很久没用过了」。
// 纯提示、不参与任何打分——与其让系统按一个没人能解释的函数把它偷偷打七折，不如把「久未复现」
// 摆到用户眼前，由他自己决定删不删。（真到期不返回是另一回事，由 MEMORY_RETENTION_DAYS 管。）
const STALE_DAYS = 90;

// 后端 SQLite 存的是 naive datetime，ISO 串没有时区后缀（如 2026-01-05T10:00:00）——
// 补个 Z 按 UTC 解析，否则会被当成本地时间，在 UTC+ 时区里算出偏小的天数。
function daysSince(iso: string): number {
  const withTz = /(Z|[+-]\d{2}:?\d{2})$/.test(iso) ? iso : `${iso}Z`;
  const ms = Date.now() - new Date(withTz).getTime();
  return Number.isFinite(ms) ? Math.floor(ms / 86_400_000) : 0;
}

function staleHint(pref: Preference): string | null {
  const days = daysSince(pref.updated_at);
  if (days < STALE_DAYS) return null;
  const months = Math.floor(days / 30);
  return months >= 12 ? "一年多没更新了" : `${months} 个月没更新了`;
}

type PreferenceItemProps = {
  pref: Preference;
  onEdit: () => void;
  onDelete: () => void;
};

export function PreferenceItem({ pref, onEdit, onDelete }: PreferenceItemProps) {
  const stale = staleHint(pref);
  const isRule = pref.category === "constraint";

  return (
    <li className={`pref-item pref-${pref.category}`}>
      <span className="pref-mark">{isRule ? "🚫" : "❤️"}</span>
      <div className="pref-text">
        <span className="pref-content">{pref.value}</span>

        <div className="pref-meta">
          <span className="pref-key" title="同 key 只存一条，改主意时用原 key 覆盖">
            {pref.key}
          </span>
          <span
            className={`pref-cat ${isRule ? "pref-blocking" : "pref-soft"}`}
            title={CATEGORY_HINT[pref.category]}
          >
            {CATEGORY_CN[pref.category] ?? pref.category}
          </span>
          {/* 久未更新只提示、不打折。删不删由用户定，系统不替他做主 */}
          {stale && (
            <span className="pref-stale" title="它仍在全额生效。不需要了就删掉">
              ⏳ {stale}
            </span>
          )}
        </div>
      </div>

      <div className="pref-ops">
        <button className="pref-op" title="编辑这条记忆" onClick={onEdit}>
          ✎
        </button>
        <button className="pref-op pref-del" title="删除这条记忆" onClick={onDelete}>
          ✕
        </button>
      </div>
    </li>
  );
}
