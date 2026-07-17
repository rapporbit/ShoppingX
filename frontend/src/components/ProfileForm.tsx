import { useEffect, useState } from "react";
import { updateProfile } from "../api";
import type { Preference } from "../types";

// 「我的资料」——收货地 / 预算上限这类**硬事实**，用户显式设定，不靠 LLM 从聊天里猜。
// 它们和口味偏好（材质/风格）的失败模式不同：猜错收货国会把关税直接算错、猜错预算会把候选全卡掉，
// 后果是「结果明显不对」而不是「推荐差一点」，所以值得单独一块让用户自己填。
// 后端把它们存成固定 slug 的普通偏好条目（like:location:global:ship_to / …:budget_max），
// 于是收货国自动被 planner 的收货国解析第 3 层消费，无需改检索链路。

// 与后端 geo.SUPPORTED_COUNTRIES 对齐的常用子集（列全 21 个对下拉没意义，够用即可）。
const COUNTRIES: [string, string][] = [
  ["", "未设置"],
  ["CN", "中国大陆"],
  ["HK", "中国香港"],
  ["TW", "中国台湾"],
  ["JP", "日本"],
  ["KR", "韩国"],
  ["SG", "新加坡"],
  ["MY", "马来西亚"],
  ["TH", "泰国"],
  ["US", "美国"],
  ["CA", "加拿大"],
  ["GB", "英国"],
  ["DE", "德国"],
  ["FR", "法国"],
  ["AU", "澳大利亚"],
];

// 从全量偏好里反解出资料项：后端用的是固定 dedup_key，前端据此认。
const SHIP_TO_KEY = "like:location:global:ship_to";
const BUDGET_KEY = "like:budget:global:budget_max";

function parseShipTo(prefs: Preference[]): string {
  const e = prefs.find((p) => p.dedup_key === SHIP_TO_KEY);
  return e ? (e.content.match(/[A-Z]{2}\b/)?.[0] ?? "") : "";
}

function parseBudget(prefs: Preference[]): string {
  const e = prefs.find((p) => p.dedup_key === BUDGET_KEY);
  return e ? (e.content.match(/[\d.]+/)?.[0] ?? "") : "";
}

type ProfileFormProps = {
  userId: string;
  prefs: Preference[];
  onSaved: (prefs: Preference[]) => void;
};

export function ProfileForm({ userId, prefs, onSaved }: ProfileFormProps) {
  const [country, setCountry] = useState("");
  const [budget, setBudget] = useState("");
  const [saving, setSaving] = useState(false);

  // 列表重拉后同步回表单（用户可能在对话里说过「寄到日本」，curator 也会写 location 条目）。
  useEffect(() => {
    setCountry(parseShipTo(prefs));
    setBudget(parseBudget(prefs));
  }, [prefs]);

  const dirty = country !== parseShipTo(prefs) || budget !== parseBudget(prefs);

  const save = async () => {
    setSaving(true);
    try {
      const n = Number(budget);
      onSaved(
        await updateProfile(userId, {
          dest_country: country,
          // 空输入 = 清除（后端把 <=0 当清除）；非法输入按 0 处理，同样是清除，不会写脏值。
          budget_max_usd: budget.trim() === "" || !Number.isFinite(n) ? 0 : n,
        }),
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="pref-section">
      <div className="pref-section-title">我的资料</div>
      <div className="pref-section-hint">这两项影响关税和预算过滤，建议自己设定，别让 Agent 猜。</div>
      <div className="profile-row">
        <label htmlFor="pf-country">常用收货地</label>
        <select id="pf-country" value={country} onChange={(e) => setCountry(e.target.value)}>
          {COUNTRIES.map(([code, name]) => (
            <option key={code} value={code}>
              {name}
            </option>
          ))}
        </select>
      </div>
      <div className="profile-row">
        <label htmlFor="pf-budget">预算上限</label>
        <input
          id="pf-budget"
          type="number"
          min="0"
          placeholder="不限"
          value={budget}
          onChange={(e) => setBudget(e.target.value)}
        />
        <span className="profile-unit">USD</span>
      </div>
      <button className="pref-save" onClick={() => void save()} disabled={!dirty || saving}>
        {saving ? "保存中…" : dirty ? "保存" : "已保存"}
      </button>
    </section>
  );
}
