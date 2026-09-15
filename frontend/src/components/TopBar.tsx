import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Globe, LogOut, Menu, SlidersHorizontal } from "lucide-react";
import { formatResetAt, type Quota } from "../api";
import type { TaskStatus } from "../hooks/useShoppingXTask";
import { Tooltip } from "./ui/Tooltip";

// 顶栏只放「本轮 / 本次会话」相关的东西：左侧当前会话标题 + 运行状态点；右侧 credit 条、检索平台数、
// 头像。收藏 / 订单 / Skill / 长期偏好是「我的东西」，属于导航，已下沉到侧栏；退出和后台收进头像菜单——
// 它们一天点不了一次，不值得常驻一个按钮位。
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
  error: "出错",
};

type TopBarProps = {
  title: string;
  status: TaskStatus;
  // 展示用的是**用户名**，不是 user_id：后者是一串随机 hex，取首字母只会得到两个乱码字符。
  username: string;
  platformCount: number;
  quota: Quota | null;
  onOpenSettings: () => void;
  // 后台管理入口。非管理员传 null —— 菜单项整个不渲染，而不是禁用态：真正的门在后端。
  onOpenAdmin: (() => void) | null;
  onLogout: () => void;
  // 窄屏专用：会话栏在手机上收成了抽屉，得有个入口把它唤回来。宽屏侧栏常驻，此按钮 CSS 隐藏。
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
  onOpenAdmin,
  onLogout,
}: Pick<TopBarProps, "username" | "onOpenAdmin" | "onLogout">) {
  return (
    <DropdownMenu.Root modal={false}>
      <DropdownMenu.Trigger asChild>
        <button className="avatar" aria-label={`账户菜单：${username}`}>
          {initials(username)}
        </button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu" align="end" sideOffset={8} collisionPadding={8}>
          <DropdownMenu.Label className="menu-label">{username}</DropdownMenu.Label>
          <DropdownMenu.Separator className="menu-sep" />
          {onOpenAdmin && (
            <DropdownMenu.Item className="menu-item" onSelect={onOpenAdmin}>
              <SlidersHorizontal size={15} strokeWidth={1.75} />
              后台管理
            </DropdownMenu.Item>
          )}
          <DropdownMenu.Item className="menu-item danger" onSelect={onLogout}>
            <LogOut size={15} strokeWidth={1.75} />
            退出登录
          </DropdownMenu.Item>
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}

export function TopBar({
  title,
  status,
  username,
  platformCount,
  quota,
  onOpenSettings,
  onOpenAdmin,
  onLogout,
  onOpenNav,
}: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-title">
        <button className="nav-toggle" onClick={onOpenNav} aria-label="打开会话栏">
          <Menu size={20} strokeWidth={1.75} />
        </button>
        <span className="title-text" title={title}>
          {title}
        </span>
        <Tooltip content={STATUS_TEXT[status]}>
          <span className={`status-dot status-${status}`} tabIndex={-1} />
        </Tooltip>
      </div>

      <div className="topbar-actions">
        {quota?.enabled && <QuotaMeter quota={quota} />}
        {/* 平台入口常驻顶栏并显示已启用个数：跨平台并行检索是本项目最贵的一步，用户该随时看得见
            自己开着几个平台，而不是点进设置才知道。 */}
        <Tooltip content="检索平台设置（默认只搜 Amazon）">
          <button className="ghost-btn" onClick={onOpenSettings}>
            <Globe size={17} strokeWidth={1.75} />
            <span>{platformCount > 1 ? `${platformCount} 个平台` : "单平台"}</span>
          </button>
        </Tooltip>
        <AvatarMenu username={username} onOpenAdmin={onOpenAdmin} onLogout={onLogout} />
      </div>
    </header>
  );
}
