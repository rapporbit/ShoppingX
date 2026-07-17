import { formatResetAt, type Quota } from "../api";
import type { TaskStatus } from "../hooks/useShoppingXTask";
import { BoltIcon, GlobeIcon, HeartIcon, MenuIcon } from "./icons";

// 顶栏：左侧是当前会话标题（即本轮 query）+ 运行状态点；右侧只留本项目真实存在的入口——
// 「长期偏好」按钮 + 代表当前用户的头像。Accio 那种纯装饰的 Upgrade / Share 仍然没有。
//
// **credit 余额条是有功能的**（M18，与那些装饰件的区别就在这）：它显示的是后端 usage_ledger 里
// 真实累计的当日成本，归零时 POST /api/task 会真的 402 拒任务。后端没开配额（demo / 本地）时
// quota.enabled=false，整块不渲染——不给用户看一个恒为满格、点了也没意义的进度条。
const STATUS_TEXT: Record<TaskStatus, string> = {
  idle: "待命",
  connecting: "连接中",
  running: "进行中",
  waiting: "等待回复",
  done: "已完成",
  cancelled: "已取消",
  error: "出错",
};

type TopBarProps = {
  title: string;
  status: TaskStatus;
  // 展示用的是**用户名**，不是 user_id：后者是一串随机 hex（防止靠猜相邻 id 撞到别人的身份），
  // 拿它取首字母只会得到两个乱码字符。
  username: string;
  platformCount: number;
  favoriteCount: number;
  quota: Quota | null;
  onOpenPreferences: () => void;
  onOpenSettings: () => void;
  // 后台管理入口。非管理员传 null —— 按钮整个不渲染，而不是渲染成禁用态：普通用户没必要知道
  // 有这么个东西存在（真正的门在后端，这里只是不摆一个必然 403 的按钮）。
  onOpenAdmin: (() => void) | null;
  onOpenFavorites: () => void;
  onLogout: () => void;
  // 窄屏专用：会话栏在手机上收成了抽屉，得有个入口把它唤回来。宽屏侧栏常驻，此按钮 CSS 隐藏。
  onOpenNav: () => void;
};

function initials(name: string): string {
  const parts = name.split(/[-_\s]+/).filter(Boolean);
  const letters = parts.length >= 2 ? parts[0][0] + parts[1][0] : name.slice(0, 2);
  return letters.toUpperCase();
}

// 今日 credit 余额条。三档配色（充足 / 见底 / 耗尽）——「快没了」必须在用户发下一条 query 之前
// 就看得见，等 402 弹出来才知道，那一条 query 的上下文已经白打了。
function QuotaMeter({ quota }: { quota: Quota }) {
  const pct = quota.limit_credits
    ? Math.min(100, Math.round((quota.used_credits / quota.limit_credits) * 100))
    : 0;
  const level = quota.exhausted ? "empty" : pct >= 80 ? "low" : "ok";
  const title = quota.exhausted
    ? `今日 credit 已用完（${formatResetAt(quota.reset_at)} 重置）`
    : `今日已用 ${quota.used_credits} / ${quota.limit_credits} credits，${formatResetAt(quota.reset_at)} 重置`;
  return (
    <div className={`quota-meter quota-${level}`} title={title}>
      <div className="quota-bar">
        <div className="quota-fill" style={{ width: `${pct}%` }} />
      </div>
      <span className="quota-text">
        {quota.remaining_credits.toLocaleString()} <span className="quota-unit">credits</span>
      </span>
    </div>
  );
}

export function TopBar({
  title,
  status,
  username,
  platformCount,
  favoriteCount,
  quota,
  onOpenPreferences,
  onOpenSettings,
  onOpenAdmin,
  onOpenFavorites,
  onLogout,
  onOpenNav,
}: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-title">
        <button className="nav-toggle" onClick={onOpenNav} aria-label="打开会话栏">
          <MenuIcon width={20} height={20} />
        </button>
        <span className="title-text" title={title}>
          {title}
        </span>
        <span className={`status-dot status-${status}`} title={STATUS_TEXT[status]} />
      </div>

      <div className="topbar-actions">
        {quota?.enabled && <QuotaMeter quota={quota} />}
        {/* 平台入口常驻顶栏并显示已启用个数：跨平台 fork 是本项目最贵的一步，用户该随时看得见
            自己开着几个平台，而不是点进设置才知道。 */}
        <button
          className="ghost-btn"
          onClick={onOpenSettings}
          title="检索平台设置（默认只搜 Amazon）"
        >
          <GlobeIcon width={18} height={18} />
          <span>{platformCount > 1 ? `${platformCount} 个平台` : "单平台"}</span>
        </button>
        {/* 收藏与「长期偏好」分开摆：后者被注入 prompt、显式改变推荐；前者是回看清单，只经行为亲和
            给同类属性一点弱加分。两者力度差一个量级，各是各的入口，别让用户以为收藏＝显式教了 Agent。 */}
        <button
          className="ghost-btn"
          onClick={onOpenFavorites}
          title="我收藏的商品（收藏多了会轻微影响精挑排序）"
        >
          <span className="fav-glyph">♥</span>
          <span>收藏{favoriteCount > 0 ? ` ${favoriteCount}` : ""}</span>
        </button>
        {onOpenAdmin && (
          <button className="ghost-btn" onClick={onOpenAdmin} title="后台管理：模型与检索参数">
            <BoltIcon width={18} height={18} />
            <span>后台</span>
          </button>
        )}
        <button className="ghost-btn" onClick={onOpenPreferences}>
          <HeartIcon width={18} height={18} />
          <span>长期偏好</span>
        </button>
        <button
          className="avatar"
          title={`${username} 的长期偏好`}
          onClick={onOpenPreferences}
        >
          {initials(username)}
        </button>
        <button className="ghost-btn" onClick={onLogout} title={`退出 ${username}`}>
          退出
        </button>
      </div>
    </header>
  );
}
