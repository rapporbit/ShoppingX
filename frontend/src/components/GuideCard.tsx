import type { GuideData } from "../types";
import { downloadFile } from "../api";
import { DownloadIcon } from "./icons";

// 选购指南卡（S3 / present_guide）：一节一条标准、要点成列表、来源单独一栏可点。
//
// 为什么不复用 FinalAnswer 的 markdown 正文：这份内容的结构本来就是「几条标准 × 几个要点 + 一份
// 出处」，后端已经把它拆成字段了。画成卡之后节与节分得开、来源不混在正文末尾，扫读比一整篇
// markdown 快得多。正文那份没浪费——它并回了 final_text，落盘产物与历史回看读的都是它。
//
// 来源链接一律 target=_blank + rel=noreferrer：站外链接，不能让它有机会拿到 window.opener。
export function GuideCard({ guide, threadId }: { guide: GuideData; threadId: string | null }) {
  const { topic, sections, assumptions, sources, closing } = guide;
  return (
    <div className="guide-card">
      {topic && <div className="guide-topic">{topic}</div>}

      {assumptions.length > 0 && (
        <div className="guide-assumptions">
          按这些假设讲的（不对就说一声）：{assumptions.join("；")}
        </div>
      )}

      <ol className="guide-sections">
        {sections.map((s, i) => (
          <li key={`${i}-${s.title}`} className="guide-section">
            {s.title && <div className="guide-section-title">{s.title}</div>}
            <ul className="guide-points">
              {s.points.map((p, j) => (
                <li key={j}>{p}</li>
              ))}
            </ul>
          </li>
        ))}
      </ol>

      {closing && <div className="guide-closing">{closing}</div>}

      {sources.length > 0 && (
        <div className="guide-sources">
          <span className="guide-sources-label">参考来源</span>
          {sources.map((s, i) => (
            <a key={i} href={s.url} target="_blank" rel="noreferrer noopener">
              {s.title || hostOf(s.url)}
            </a>
          ))}
        </div>
      )}

      {threadId && (
        <div className="final-downloads">
          {/* 与 FinalAnswer 同一条口径：产物接口要校验属主，<a href> 带不上 token。 */}
          {["summary.md", "result.json"].map((name) => (
            <button key={name} type="button" onClick={() => void downloadFile(threadId, name)}>
              <DownloadIcon width={14} height={14} />
              {name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// 没给标题的来源退回域名（与后端 _label 同口径）：裸 url 摊在卡上会撑破排版。
function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}
