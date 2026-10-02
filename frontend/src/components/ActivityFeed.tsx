import { useEffect, useRef, useState } from "react";
import type { AguiEvent } from "../types";
import { AnimatePresence, motion } from "motion/react";
import type { LucideIcon } from "lucide-react";
import {
  BookOpen,
  Check,
  CheckCircle2,
  ChevronDown,
  Clock,
  Globe,
  ListChecks,
  Loader2,
  MessageCircleQuestion,
  Scale,
  Search,
  Sparkles,
  Truck,
  XCircle,
} from "lucide-react";

// 「思考过程」活动流 —— 形式与动效向 Claude 网页版的思考块看齐：
//   · 标题只有一行文字 + 箭头；运行中文字扫光（.shimmer-text），换步骤时旧字上滑淡出、新字上滑淡入
//   · 展开区是一条时间线：左侧竖线串起每步的小图标，右侧是步骤名与一句结果；思考文本直接成段显示
//   · 展开/收起用 motion 的 height auto + 透明度过渡
//   · 每个工具一行：旋转中 → 该工具的图标 / 报错，而不是 start/end 两行流水账
//   · 运行中展开，任务收尾 600ms 后自动收起，把版面让给最终清单
const TOOL_LABEL: Record<string, string> = {
  planner: "需求拆解",
  chat_fallback: "对话回复",
  web_search: "联网检索",
  category_insight: "品类洞察",
  item_search: "商品检索",
  item_picker: "智能精选",
  price_compare: "跨平台比价",
  shipping_calc: "到手价测算",
  shopping_summary: "生成购物清单",
  ask_user: "向用户提问",
};

const toolLabel = (tool?: string) => (tool ? TOOL_LABEL[tool] ?? tool : "工具调用");

// 展开后要看的是「这一步想出了什么」（tool_end 的人读摘要 result），而不是 card_count 这类元信息；
// 没有 result 时才退回入参串。
function detailText(evt: AguiEvent): string {
  const d = evt.data ?? {};
  if (evt.event === "assistant_call") return String(d.preview ?? "");
  if (typeof d.result === "string" && d.result) return d.result;
  return Object.entries(d)
    .filter(([k, v]) => k !== "tool" && k !== "result" && v != null && v !== "")
    .map(([k, v]) => `${k}=${v}`)
    .join(" · ");
}

type StepState = "running" | "done" | "error" | "info";
type Step = { evt: AguiEvent; state: StepState };

// tool_start / tool_end 合并成同一行（Accio 就是一个工具一行、图标随状态变），
// 未闭合的 start 保持旋转。同轮 batch 会有同名工具同时在跑（如跨平台 / 多槽位的 item_search），
// 故按工具名维护一个队列。
function buildSteps(events: AguiEvent[]): Step[] {
  const rows: Step[] = [];
  const open = new Map<string, number[]>();
  for (const evt of events) {
    const tool = String(evt.data?.tool ?? "");
    if (evt.event === "tool_start") {
      const q = open.get(tool) ?? [];
      q.push(rows.push({ evt, state: "running" }) - 1);
      open.set(tool, q);
    } else if (evt.event === "tool_end") {
      const state: StepState = evt.data?.error ? "error" : "done";
      const q = open.get(tool);
      const idx = q?.shift();
      if (idx != null) rows[idx] = { evt, state };
      else rows.push({ evt, state });
    } else {
      // 老会话回放里可能还有 `fork` 事件（删派发前留下的），落到这里当普通 info 行画，
      // 不再有专属图标与「子任务并行处理中」文案。
      rows.push({ evt, state: "info" });
    }
  }
  return rows;
}

const TOOL_ICON: Record<string, LucideIcon> = {
  planner: ListChecks,
  web_search: Globe,
  category_insight: BookOpen,
  item_search: Search,
  item_picker: Sparkles,
  price_compare: Scale,
  shipping_calc: Truck,
  ask_user: MessageCircleQuestion,
};

// 收尾类事件也会进 events：在时间线末尾画成一行状态，不展开详情（task_result 的 data 是整份结果）。
const TERMINAL_LABEL: Record<string, string> = {
  task_result: "完成",
  task_cancelled: "已取消",
  task_interrupted: "已中断",
};

function stepIcon({ evt, state }: Step): LucideIcon {
  if (evt.event === "task_result") return CheckCircle2;
  if (state === "running") return Loader2;
  if (state === "error" || evt.event === "error") return XCircle;
  if (evt.event in TERMINAL_LABEL) return XCircle;
  if (state === "info") return Clock;
  return TOOL_ICON[String(evt.data?.tool ?? "")] ?? Check;
}

// 运行中标题：优先播报最后一个还在跑的工具，否则退回「正在思考」。同轮 batch 时多个同名工具
// 同时在跑，只播报最后一个——标题是给人看进度的，不是列清单。
function headline(steps: Step[]): string {
  const running = [...steps].reverse().find((s) => s.state === "running");
  if (running) return `${toolLabel(String(running.evt.data?.tool ?? ""))}中…`;
  return "正在思考…";
}

function StepRow({ step }: { step: Step }) {
  const { evt, state } = step;
  const [open, setOpen] = useState(false);

  if (evt.event === "clarification_request") {
    return (
      <div className="step-row">
        <MessageCircleQuestion size={15} strokeWidth={1.75} className="step-icon ask" />
        <div className="step-body">
          <p className="step-text ask">向用户提问：{String(evt.data?.question ?? "")}</p>
        </div>
      </div>
    );
  }

  if (evt.event === "queue_status") {
    const ahead = Math.max(0, Number(evt.data?.position ?? 1) - 1);
    const eta = Number(evt.data?.estimated_wait_seconds ?? 0);
    return (
      <div className="step-row">
        <Loader2 size={15} strokeWidth={1.75} className="step-icon running" />
        <div className="step-body">
          <p className="step-text">
            排队中：前面还有 {ahead} 个任务{eta > 0 ? `，预计等待约 ${eta} 秒` : ""}
          </p>
        </div>
      </div>
    );
  }

  const Icon = stepIcon(step);
  const iconState = evt.event === "error" ? "error" : state;
  const terminal = TERMINAL_LABEL[evt.event];
  const detail = terminal ? "" : detailText(evt);

  // 思考文本直接成段显示（Claude 网页版同款），不折成「思考 + 一行预览」。
  if (evt.event === "assistant_call" && detail) {
    return (
      <div className="step-row">
        <Icon size={15} strokeWidth={1.75} className={`step-icon ${iconState}`} />
        <div className="step-body">
          <p className="step-text">{detail}</p>
        </div>
      </div>
    );
  }

  const label =
    terminal ??
    (evt.event === "session_created"
      ? "会话已创建，开始规划"
      : evt.event === "assistant_call"
        ? "思考"
        : evt.event === "error"
          ? "出错"
          : toolLabel(String(evt.data?.tool ?? "")));
  const expandable = Boolean(detail);

  return (
    <div className={`step-row ${expandable ? "expandable" : ""}`}>
      <Icon size={15} strokeWidth={1.75} className={`step-icon ${iconState}`} />
      <div className="step-body">
        <button
          type="button"
          className="step-head"
          onClick={() => expandable && setOpen((v) => !v)}
          disabled={!expandable}
          aria-expanded={expandable ? open : undefined}
        >
          <span className="step-label">{label}</span>
          {detail && !open && <span className="step-preview">{detail}</span>}
          {expandable && (
            <ChevronDown size={14} strokeWidth={2} className={`thought-chev ${open ? "open" : ""}`} />
          )}
        </button>
        <AnimatePresence initial={false}>
          {detail && open && (
            <motion.div
              className="step-detail-wrap"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2, ease: "easeOut" }}
            >
              <div className="step-detail">{detail}</div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

type ActivityFeedProps = { events: AguiEvent[]; running: boolean };

export function ActivityFeed({ events, running }: ActivityFeedProps) {
  const steps = buildSteps(events);
  // 展开态三层：用户显式点过（override 优先）→ 否则跑的时候展开、收尾后自动收起。
  const [override, setOverride] = useState<boolean | null>(null);
  // 初值跟 running 走：回看历史轮时直接是收起态，不会先展开再在 600ms 后缩回去。
  const [auto, setAuto] = useState(running);
  const stepsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (running) {
      setAuto(true);
      return;
    }
    if (steps.length === 0) return;
    // 收尾后延迟收起：让最后一行的打勾先被看见，再把版面让给最终清单（Accio 是 600ms）。
    const t = setTimeout(() => setAuto(false), 600);
    return () => clearTimeout(t);
  }, [running, steps.length]);

  const open = override ?? auto;

  // 有新步骤就滚到底。跑的时候连滚 280ms（rAF）盖住展开动画期间的高度变化，避免最后一行被截在视口外。
  useEffect(() => {
    const el = stepsRef.current;
    if (!el || !open) return;
    if (!running) {
      el.scrollTop = el.scrollHeight;
      return;
    }
    let raf = 0;
    const until = performance.now() + 280;
    const tick = () => {
      el.scrollTop = el.scrollHeight;
      if (performance.now() < until) raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [steps.length, open, running]);

  if (steps.length === 0 && !running) return null;

  const toolCount = steps.filter((s) => s.state === "done" || s.state === "error").length;
  const title = running ? headline(steps) : toolCount > 0 ? `思考过程 · ${toolCount} 步` : "思考过程";
  // 老会话的 activity 里不一定有收尾事件：没有就补一行「完成」，让时间线有个终点。
  const hasTerminal = steps.some((s) => s.evt.event in TERMINAL_LABEL || s.evt.event === "error");

  return (
    <div className="thought">
      <button
        type="button"
        className="thought-head"
        onClick={() => setOverride(!open)}
        aria-expanded={open}
      >
        <span className="thought-title">
          {/* key=文案：换步骤时旧字上滑淡出、新字从下方淡入。 */}
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={title}
              className={running ? "shimmer-text" : undefined}
              initial={{ y: 6, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              exit={{ y: -6, opacity: 0 }}
              transition={{ duration: 0.16, ease: "easeOut" }}
            >
              {title}
            </motion.span>
          </AnimatePresence>
        </span>
        <ChevronDown size={15} strokeWidth={2} className={`thought-chev ${open ? "open" : ""}`} />
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            className="thought-panel"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="thought-steps" ref={stepsRef}>
              {/* key 只用下标：tool_start 就地变成 tool_end（转圈→图标），行不重挂、不重播入场动画。 */}
              {steps.map((s, i) => (
                <StepRow key={i} step={s} />
              ))}
              {!running && steps.length > 0 && !hasTerminal && (
                <div className="step-row">
                  <CheckCircle2 size={15} strokeWidth={1.75} className="step-icon" />
                  <div className="step-body">
                    <p className="step-text">完成</p>
                  </div>
                </div>
              )}
              {running && steps.length === 0 && (
                <div className="skeleton">
                  <span style={{ width: "180px" }} />
                  <span style={{ width: "140px" }} />
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
