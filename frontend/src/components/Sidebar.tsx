import { CloseIcon, ComposeIcon, HeartIcon, ListIcon, SparkleIcon } from "./icons";
import type { SessionMeta } from "../types";

// 左侧栏：品牌 → 「新建对话」 → 主导航（收藏 / 订单 / Skill / 长期偏好）→ 历史对话列表。
// 主导航从顶栏搬下来：这四个都是「我的东西」，放侧栏才是导航，堆顶栏像调试面板。
// 历史列表的会话索引由 useShoppingXTask 维护在 localStorage（后端只按 threadId 存逐轮对话，无会话清单层）。
export type SidebarPanel = "favorites" | "orders" | "skills" | "preferences";

type SidebarProps = {
  sessions: SessionMeta[];
  activeThreadId: string | null;
  onNewChat: () => void;
  onSelectConversation: (threadId: string) => void;
  onDeleteConversation: (threadId: string) => void;
  // 打开某个面板（右侧抽屉）；activePanel 决定哪一项高亮，null = 都没开
  onOpenPanel: (panel: SidebarPanel) => void;
  activePanel: SidebarPanel | null;
  favoriteCount: number;
  // 窄屏下侧栏是抽屉，open 决定它是否滑入；宽屏侧栏常驻，这个 class 不起作用。
  open: boolean;
};

const NAV: { id: SidebarPanel; label: string; icon: JSX.Element; title: string }[] = [
  {
    id: "favorites",
    label: "收藏",
    icon: <HeartIcon width={17} height={17} />,
    title: "我收藏的商品（收藏多了会轻微影响精挑排序）",
  },
  {
    id: "orders",
    label: "订单",
    icon: <ListIcon width={17} height={17} />,
    title: "我的订单（模拟交易，无支付与物流）",
  },
  {
    id: "skills",
    label: "Skill",
    icon: <span className="nav-glyph">/</span>,
    title: "我的 Skill：自写选购方案，输入框敲 / 可选用",
  },
  {
    id: "preferences",
    label: "长期偏好",
    icon: <SparkleIcon width={17} height={17} />,
    title: "长期偏好：会注入提示词，显式改变推荐",
  },
];

export function Sidebar({
  sessions,
  activeThreadId,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onOpenPanel,
  activePanel,
  favoriteCount,
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

      <div className="sidebar-nav">
        {NAV.map((item) => (
          <button
            key={item.id}
            className={`sidebar-nav-item ${activePanel === item.id ? "active" : ""}`}
            onClick={() => onOpenPanel(item.id)}
            title={item.title}
          >
            {item.icon}
            <span>{item.label}</span>
            {item.id === "favorites" && favoriteCount > 0 && (
              <span className="nav-count">{favoriteCount}</span>
            )}
          </button>
        ))}
      </div>

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
    </nav>
  );
}
