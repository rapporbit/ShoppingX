import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Globe, LogOut, Menu, SlidersHorizontal } from "lucide-react";
import { formatResetAt, type Quota } from "../api";
import type { TaskStatus } from "../hooks/useShoppingXTask";
import { Tooltip } from "./ui/Tooltip";

// 顶栏三段（对照 daydream.ing）：左侧一颗悬浮胶囊导航（☰ 历史 / 对话 / 收藏 / 订单），正中衬线字标 +
// 运行状态点，右侧 credit 条、检索平台数、头像。侧栏不再常驻——它收成抽屉，由胶囊最左的 ☰ 唤出；
// Skill / 长期偏好一天点不了几次，留在抽屉里。退出和后台收进头像菜单。
//
// **credit 余额条是有功能的**（M18）：它显示的是后端 usage_ledger 里真实累计的当日成本，归零时
// POST /api/task 会真的 402 拒任务。后端没开配额（demo / 本地）时 quota.enabled=false，整块不渲染。
const STATUS_TEXT: Record<TaskStatus, string> = {
  idle: "待命",
  connecting: "连接中",
  running: "进行中",
  waiting: "等待回复",
  done: "已完成",
  cancelled: "已取消",
  interrupted: "已中断",
  error: "出错",
};

// 胶囊导航里的三个去处。"chat" = 回到对话（关掉所有整页面板）。
export type TopNavTarget = "chat" | "favorites" | "orders";

type TopBarProps = {
  // 当前停在哪：没开面板就是对话；开的是 Skill / 偏好这类不在胶囊里的面板时传 null（三项都不高亮）。
  active: TopNavTarget | null;
  onNavigate: (target: TopNavTarget) => void;
  favoriteCount: number;
  status: TaskStatus;
  // 展示用的是**用户名**，不是 user_id：后者是一串随机 hex，取首字母只会得到两个乱码字符。
  username: string;
  // 试用身份：顶栏多一条「试用中 · 注册可保留记录」入口，头像菜单名字显示「试用访客」而非 guest_xxx。
  isGuest: boolean;
  onUpgrade: () => void;
  platformCount: number;
  quota: Quota | null;
  onOpenSettings: () => void;
  // 后台管理入口。非管理员传 null —— 菜单项整个不渲染，而不是禁用态：真正的门在后端。
  onOpenAdmin: (() => void) | null;
  onLogout: () => void;
  // 会话栏在所有宽度下都是抽屉，这是唤出它的入口。
  onOpenNav: () => void;
};

function initials(name: string): string {
  const parts = name.split(/[-_\s]+/).filter(Boolean);
  const letters = parts.length >= 2 ? parts[0][0] + parts[1][0] : name.slice(0, 2);
  return letters.toUpperCase();
}

// 今日 credit 余额条。三档配色（充足 / 见底 / 耗尽）——「快没了」必须在用户发下一条 query 之前就看得见。
function QuotaMeter({ quota }: { quota: Quota }) {
  const pct = quota.limit_credits
    ? Math.min(100, Math.round((quota.used_credits / quota.limit_credits) * 100))
    : 0;
  const level = quota.exhausted ? "empty" : pct >= 80 ? "low" : "ok";
  const title = quota.exhausted
    ? `今日 credit 已用完（${formatResetAt(quota.reset_at)} 重置）`
    : `今日已用 ${quota.used_credits} / ${quota.limit_credits} credits，${formatResetAt(quota.reset_at)} 重置`;
  return (
    <Tooltip content={title}>
      <div className={`quota-meter quota-${level}`} tabIndex={-1}>
        <div className="quota-bar">
          <div className="quota-fill" style={{ width: `${pct}%` }} />
        </div>
        <span className="quota-text">
          {quota.remaining_credits.toLocaleString()} <span className="quota-unit">credits</span>
        </span>
      </div>
    </Tooltip>
  );
}

// 头像下拉：用户名 / 后台管理（仅管理员）/ 退出。Radix DropdownMenu 管点外关闭、Esc、方向键与焦点回落。
function AvatarMenu({
  username,
  isGuest,
  onOpenAdmin,
  onLogout,
}: Pick<TopBarProps, "username" | "isGuest" | "onOpenAdmin" | "onLogout">) {
  const shown = isGuest ? "试用访客" : username;
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <button className="avatar" aria-label={`账户菜单：${shown}`}>
          {isGuest ? "试" : initials(username)}
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu" align="end" sideOffset={8} collisionPadding={8}>
          <DropdownMenu.Label className="menu-label">{shown}</DropdownMenu.Label>
          <DropdownMenu.Separator className="menu-sep" />
          {onOpenAdmin && (
            <DropdownMenu.Item className="menu-item" onSelect={onOpenAdmin}>
              <SlidersHorizontal size={15} strokeWidth={1.75} />
              后台管理
            </DropdownMenu.Item>
          )}
          <DropdownMenu.Item className="menu-item danger" onSelect={onLogout}>
            <LogOut size={15} strokeWidth={1.75} />
            {isGuest ? "结束试用" : "退出登录"}
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

const NAV_ITEMS: { key: TopNavTarget; label: string }[] = [
  { key: "chat", label: "对话" },
  { key: "favorites", label: "收藏" },
  { key: "orders", label: "订单" },
];

export function TopBar({
  active,
  onNavigate,
  favoriteCount,
  status,
  username,
  isGuest,
  onUpgrade,
  platformCount,
  quota,
  onOpenSettings,
  onOpenAdmin,
  onLogout,
  onOpenNav,
}: TopBarProps) {
  return (
    <header className="topbar">
      <nav className="topnav" aria-label="主导航">
        <Tooltip content="历史对话 / Skill / 长期偏好">
          <button className="topnav-menu" onClick={onOpenNav} aria-label="打开会话栏">
            <Menu size={17} strokeWidth={1.75} />
          </button>
        </Tooltip>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.key}
            className={`topnav-item ${active === item.key ? "active" : ""}`}
            aria-current={active === item.key ? "page" : undefined}
            onClick={() => onNavigate(item.key)}
          >
            {item.label}
            {item.key === "favorites" && favoriteCount > 0 && (
              <span className="topnav-count">{favoriteCount}</span>
            )}
          </button>
        ))}
      </nav>

      {/* 字标绝对居中：左右两段宽度不等，靠 flex 居中会偏。点它 = 回到对话。 */}
      <button className="wordmark" onClick={() => onNavigate("chat")}>
        ShoppingX
        <Tooltip content={STATUS_TEXT[status]}>
          <span className={`status-dot status-${status}`} tabIndex={-1} />
        </Tooltip>
      </button>

      <div className="topbar-actions">
        {isGuest && (
          <Tooltip content="试用额度是正式账号的 1/5；注册后试用期间的会话、偏好与收藏全部保留">
            <button className="upgrade-chip" onClick={onUpgrade}>
              试用中 · 注册保留记录
            </button>
          </Tooltip>
        )}
        {quota?.enabled && <QuotaMeter quota={quota} />}
        {/* 平台入口常驻顶栏并显示已启用个数：跨平台并行检索是本项目最贵的一步，用户该随时看得见
            自己开着几个平台，而不是点进设置才知道。 */}
        <Tooltip content="检索平台设置（默认只搜 Amazon）">
          <button className="ghost-btn" onClick={onOpenSettings}>
            <Globe size={17} strokeWidth={1.75} />
            <span>{platformCount > 1 ? `${platformCount} 个平台` : "单平台"}</span>
          </button>
        </Tooltip>
        <AvatarMenu
          username={username}
          isGuest={isGuest}
          onOpenAdmin={onOpenAdmin}
          onLogout={onLogout}
        />
      </div>
    </header>
  );
}
