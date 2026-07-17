import { CloseIcon, ComposeIcon, HeartIcon } from "./icons";
import type { SessionMeta } from "../types";

// 左侧会话栏：顶部「新建对话」，中部历史对话列表（点条目切会话 / 悬停可删），底部「长期偏好」。
// 历史列表的会话索引由 useShoppingXTask 维护在 localStorage（后端只按 threadId 存逐轮对话，无会话清单层）。
type SidebarProps = {
  sessions: SessionMeta[];
  activeThreadId: string | null;
  onNewChat: () => void;
  onSelectConversation: (threadId: string) => void;
  onDeleteConversation: (threadId: string) => void;
  onOpenPreferences: () => void;
  prefsOpen: boolean;
  // 窄屏下侧栏是抽屉，open 决定它是否滑入；宽屏侧栏常驻，这个 class 不起作用。
  open: boolean;
};

export function Sidebar({
  sessions,
  activeThreadId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onOpenPreferences,
  prefsOpen,
  open,
}: SidebarProps) {
  return (
    <nav className={`sidebar ${open ? "open" : ""}`}>
      <div className="sidebar-head">
        <div className="sidebar-logo" aria-hidden>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
            <path d="M12 3 3 20h4l5-10 5 10h4Z" fill="currentColor" />
          </svg>
        </div>
        <span className="sidebar-brand">ShoppingX</span>
      </div>

      <button className="new-chat-btn" onClick={onNewChat}>
        <ComposeIcon width={18} height={18} />
        新建对话
      </button>

      <div className="history">
        <div className="history-label">历史对话</div>
        {sessions.length === 0 ? (
          <p className="history-empty">还没有对话记录</p>
        ) : (
          <ul className="history-list">
            {sessions.map((s) => (
              <li
                key={s.threadId}
                className={`history-item ${s.threadId === activeThreadId ? "active" : ""}`}
                onClick={() => onSelectConversation(s.threadId)}
                title={s.title}
              >
                <span className="history-item-title">{s.title}</span>
                {/* 删除用 span 而非 button：避免按钮套按钮的非法嵌套；stopPropagation 防误触发切换 */}
                <span
                  className="history-del"
                  role="button"
                  aria-label="删除该对话"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteConversation(s.threadId);
                  }}
                >
                  <CloseIcon width={14} height={14} />
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <button
        className={`sidebar-foot-btn ${prefsOpen ? "active" : ""}`}
        onClick={onOpenPreferences}
      >
        <HeartIcon width={18} height={18} />
        长期偏好
      </button>
    </nav>
  );
}
