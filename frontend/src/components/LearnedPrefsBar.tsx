import { useState } from "react";
import { deletePreference } from "../api";
import type { LearnedPref } from "../types";

// 回复下方那一行「🧠 记住了：不接受塑料材质 ✕」。
//
// 这是整个记忆透明度设计的落点：curator 在后台自动学、自动写（不弹确认框——收尾这一刻用户的
// 注意力在「买不买这件」上，此时弹「是否记住？」只会被无脑点掉，筛不出任何东西），但写完必须
// **当场告诉用户写了什么**，并且撤销成本只有一次点击。当撤销这么便宜时，事前确认就不值得存在。
type LearnedPrefsBarProps = {
  userId: string;
  prefs: LearnedPref[];
  onForget: () => void; // 通知偏好面板重拉（删掉的那条不该还挂在列表里）
};

export function LearnedPrefsBar({ userId, prefs, onForget }: LearnedPrefsBarProps) {
  // 已撤销的 dedup_key：本地即时隐藏，不等重拉（这一行的价值就在「点了立刻消失」）。
  const [forgotten, setForgotten] = useState<Set<string>>(new Set());

  const forget = async (key: string) => {
    setForgotten((cur) => new Set(cur).add(key));
    await deletePreference(userId, key);
    onForget();
  };

  const shown = prefs.filter((p) => !forgotten.has(p.dedup_key));
  if (shown.length === 0) return null;

  return (
    <div className="learned-bar">
      <span className="learned-icon">🧠</span>
      <span className="learned-label">记住了</span>
      {shown.map((p) => (
        <span key={p.dedup_key} className="learned-chip">
          {p.content}
          <button
            className="learned-forget"
            title="别记这条"
            onClick={() => void forget(p.dedup_key)}
          >
            ✕
          </button>
        </span>
      ))}
    </div>
  );
}
