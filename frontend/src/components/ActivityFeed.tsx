import { useEffect, useRef, useState } from "react";
import { domainLabel } from "../domains";
import type { AguiEvent } from "../types";
import {
  CheckCircleIcon,
  ChevronRight,
  ErrorCircleIcon,
  ForkIcon,
  SparkleIcon,
  SpinnerIcon,
} from "./icons";

// 「思考过程」活动流 —— 结构与动效对齐 Accio 的 thought-process 组件：
//   · 运行中标题是一行会扫光的渐变文字（.shimmer-text），实时显示当前在做什么
//   · 展开/收起用 grid 0fr→1fr 过渡，无需测高
//   · 每个工具一行：旋转中 → 打勾 / 报错，而不是 start/end 两行流水账
//   · 任务收尾 600ms 后自动收起，把版面让给最终清单
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
  dispatch_tool: "派发子任务",
  ask_user: "向用户提问",
};

const toolLabel = (tool?: string) => (tool ? TOOL_LABEL[tool] ?? tool : "工具调用");

// 展开后要看的是「这一步想出了什么」（tool_end 的人读摘要 result），而不是 card_count 这类元信息；
// 没有 result 时才退回入参串。
function detailText(evt: AguiEvent): string {
  const d = evt.data ?? {};
  if (evt.event === "fork") return String(d.demands ?? "");
  if (evt.event === "assistant_call") return String(d.preview ?? "");
  if (typeof d.result === "string" && d.result) return d.result;
  return Object.entries(d)
    .filter(([k, v]) => k !== "tool" && k !== "result" && v != null && v !== "")
    .map(([k, v]) => `${k}=${v}`)
    .join(" · ");
}

type StepState = "running" | "done" | "error" | "fork" | "info";
type Step = { evt: AguiEvent; state: StepState };

// tool_start / tool_end 合并成同一行（Accio 就是一个工具一行、图标随状态变），
// 未闭合的 start 保持旋转。并行 fork 会有同名工具同时在跑，故按工具名维护一个队列。
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
    } else if (evt.event === "fork") {
      rows.push({ evt, state: "fork" });
    } else {
      rows.push({ evt, state: "info" });
    }
  }
  return rows;
}

const STEP_ICON = {
  running: SpinnerIcon,
  done: CheckCircleIcon,
  error: ErrorCircleIcon,
  fork: ForkIcon,
  info: SparkleIcon,
} as const;

// 运行中标题：优先播报最后一个还在跑的工具，否则退回「正在思考」。
function headline(steps: Step[]): string {
  const running = [...steps].reverse().find((s) => s.state === "running");
  if (running) return `${toolLabel(String(running.evt.data?.tool ?? ""))}中…`;
  const forking = steps.some((s) => s.state === "fork");
  return forking ? "子任务并行处理中…" : "正在思考…";
}

function StepRow({ step }: { step: Step }) {
  const { evt, state } = step;
  const [open, setOpen] = useState(false);

  if (evt.event === "clarification_request") {
    return (
      <div className="step-row">
        <div className="clarification-row">
          <span className="clarification-icon">?</span>
          <span>向用户提问：{String(evt.data?.question ?? "")}</span>
        </div>
      </div>
    );
  }

  // 本轮长期记忆的**读取侧**生效情况。这套记忆系统真正的病不是复杂，是复杂且**不可观测**：
  // 一条偏好误杀了一批商品，用户看不到任何提示，只会觉得「这破 Agent 老是搜不出东西」，且归因
  // 不到记忆头上。所以这行不折叠、词全摊开——被排除的词尤其要写清楚，那些商品是真的没了。
  if (evt.event === "memory_applied") {
    const excluded = (evt.data?.excluded as string[] | undefined) ?? [];
    const attenuated = (evt.data?.attenuated as string[] | undefined) ?? [];
    const domains = (evt.data?.domains as string[] | undefined) ?? [];
    return (
      <div className="step-row">
        <div className="memory-row">
          <span className="memory-icon">🧠</span>
          <div className="memory-body">
            <span className="memory-label">
              {evt.message || "按你的长期偏好筛选"}
              {domains.length > 0 && (
                <span className="memory-domain">
                  （本轮品类：{domains.map(domainLabel).join(" / ")}）
                </span>
              )}
            </span>
            <div className="memory-terms">
              {excluded.map((t) => (
                <span key={`x-${t}`} className="memory-chip memory-excluded" title="含此词的商品已被移出结果">
                  ✕ {t}
                </span>
              ))}
              {attenuated.map((t) => (
                <span key={`a-${t}`} className="memory-chip memory-attenuated" title="含此词的商品被压低排序，仍在候选里">
                  ↓ {t}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (evt.event === "queue_status") {
    const ahead = Math.max(0, Number(evt.data?.position ?? 1) - 1);
    const eta = Number(evt.data?.estimated_wait_seconds ?? 0);
    return (
      <div className="step-row">
        <div className="step-head" style={{ cursor: "default" }}>
          <SpinnerIcon width={14} height={14} className="step-icon running" />
          <span className="step-label">
            排队中：前面还有 {ahead} 个任务{eta > 0 ? `，预计等待约 ${eta} 秒` : ""}
          </span>
        </div>
      </div>
    );
  }

  const Icon = STEP_ICON[state];
  const label =
    evt.event === "session_created"
      ? "会话已创建，开始规划"
      : state === "fork"
        ? "派发子任务并行处理"
        : evt.event === "assistant_call"
          ? "思考"
          : toolLabel(String(evt.data?.tool ?? ""));
  const detail = detailText(evt);
  const expandable = Boolean(detail);

  return (
    <div className={`step-row ${expandable ? "expandable" : ""}`}>
      <button
        type="button"
        className="step-head"
        onClick={() => expandable && setOpen((v) => !v)}
        disabled={!expandable}
        aria-expanded={expandable ? open : undefined}
      >
        <Icon width={14} height={14} className={`step-icon ${state}`} />
        <span className="step-label">{label}</span>
        {detail && !open && <span className="step-preview">{detail}</span>}
        {expandable && (
          <ChevronRight width={13} height={13} className={`thought-chev ${open ? "open" : ""}`} />
        )}
      </button>
      {detail && open && <div className="step-detail">{detail}</div>}
    </div>
  );
}

type ActivityFeedProps = { events: AguiEvent[]; running: boolean };

export function ActivityFeed({ events, running }: ActivityFeedProps) {
  const steps = buildSteps(events);
  // 展开态三层：用户显式点过（override 优先）→ 否则跑的时候展开、收尾后自动收起。
  const [override, setOverride] = useState<boolean | null>(null);
  const [auto, setAuto] = useState(true);
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

  return (
    <div className="thought">
      <button
        type="button"
        className={`thought-head ${running ? "is-running" : ""}`}
        onClick={() => setOverride(!open)}
        aria-expanded={open}
      >
        <SparkleIcon width={14} height={14} className="thought-spark" />
        <span className={running ? "shimmer-text" : undefined}>
          {running ? headline(steps) : "查看思考过程"}
        </span>
        <ChevronRight width={14} height={14} className={`thought-chev ${open ? "open" : ""}`} />
      </button>

      {running && (
        <div className="thought-progress">
          <i />
        </div>
      )}

      <div className={`thought-panel ${open ? "open" : ""}`}>
        <div className="thought-inner">
          <div className="thought-divider" />
          <div className="thought-steps" ref={stepsRef}>
            {/* key 只用下标：tool_start 就地变成 tool_end（转圈→打勾），行不重挂、不重播入场动画。 */}
            {steps.map((s, i) => (
              <StepRow key={i} step={s} />
            ))}
            {running && steps.length === 0 && (
              <div className="skeleton">
                <span style={{ width: "180px" }} />
                <span style={{ width: "140px" }} />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
