import { marked } from "marked";
import { downloadFile } from "../api";
import { DownloadIcon } from "./icons";

// 渲染前先把原始 HTML 标签转义掉，再交给 marked——清单文案是 LLM 生成、可能夹带 web_search
// 回来的内容，marked 默认不消毒，直接 dangerouslySetInnerHTML 就给了 <img onerror> 这类注入
// 可乘之机。转义 < > & 后，markdown 语法（#、*、[]() 等不含尖括号）照常渲染，但任何裸 HTML 失效。
// 这是零依赖的最小消毒；要更强（保留部分安全标签）可上 DOMPurify，属生产化硬化、非本主线。
function safeMarkdown(md: string): string {
  const escaped = md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return marked.parse(escaped, { async: false }) as string;
}

// 最终购物清单：Accio 把助手回答直接排在对话流里（无独立卡头），所以这里渲染成正文 markdown，
// 末尾给一行轻量产物下载（summary.md 人读 / result.json 机器读，对应后端 _write_session_artifacts）。
export function FinalAnswer({ markdown, threadId }: { markdown: string; threadId: string | null }) {
  const html = safeMarkdown(markdown);
  return (
    <div className="final">
      <div className="final-body markdown" dangerouslySetInnerHTML={{ __html: html }} />
      {threadId && (
        <div className="final-downloads">
          {/* 用按钮而不是 <a href>：产物接口要校验属主，而 <a> 发的请求带不上 token（见 api.downloadFile）。 */}
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
