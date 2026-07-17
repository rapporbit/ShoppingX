import { useCallback, useEffect, useRef, useState } from "react";
import {
  cancelTaskRequest,
  deleteSession,
  describeStartError,
  fetchHistory,
  fetchInflight,
  fetchSessions,
  startTaskRequest,
  uploadImage,
  type SessionMeta as ApiSessionMeta,
} from "../api";
import { THREAD_KEY, wsToken } from "../auth";
import type {
  AguiEvent,
  HistoryTurn,
  LearnedPref,
  ProductItem,
  SessionMeta,
  SessionSnapshot,
  TurnTokens,
} from "../types";

// 任务运行状态：idle 未开始 / connecting 正在建 WS / running 任务进行中 / done 收尾 / cancelled / error。
export type TaskStatus = "idle" | "connecting" | "running" | "waiting" | "done" | "cancelled" | "error";

// 对话里的一轮：用户一句 query + 助手这一轮的全部产出。多轮续聊下，turns 累加成完整对话流，
// 实时事件只更新**最后一轮**（同一时刻只有一个任务在跑）。回看历史时按 turns.json 重建：除结论
// 文案外，还原 items（商品卡）与 events（思考过程，后端按轮持久化的 activity），status=done。
export type Turn = {
  id: string;
  query: string;
  // 本轮用户发的参考图文件名（服务端 uploaded/<thread_id>/ 下）。渲染时按名去 /api/uploads 取，
  // 所以刷新和历史回看都还在——图是用户「说的话」的一部分，不该只活在这次页面生命周期里。
  images: string[];
  events: AguiEvent[];
  items: ProductItem[];
  finalAnswer: string | null;
  // 收尾文案的流式预览（summary_delta 事件的累计全文）：任务还在跑时逐字渲染，让用户在
  // task_result 之前 ~10s 就开始读清单。定稿到达（task_result）即清空，finalAnswer 接管。
  streamingText: string | null;
  status: TaskStatus;
  errorMsg: string | null;
  // 本轮总耗时（毫秒，后端权威口径）。完成后才有值，前端在该轮右下角显示「用时」。
  elapsedMs: number | null;
  // 本轮 token 用量（后端全树记账）。完成后才有值，与「用时」并排显示「token 消耗」。
  tokens: TurnTokens | null;
  // Agent 正在等待用户澄清时的问题文本（status="waiting" 时有值）。
  clarificationQuestion: string | null;
  // 澄清带的可点选项（ask_user 传了 options 时有值）：前端在展示区内嵌一张可点选卡片、不复用聊天框。
  // 为空/缺省 → 回退老式自由文本输入。multiSelect=多选清单（套装组成勾选）；preselected=默认勾选项。
  clarificationOptions?: string[] | null;
  clarificationMultiSelect?: boolean;
  clarificationPreselected?: string[] | null;
  // 本轮 curator 沉淀的新长期偏好（memory_updated 事件，在 task_result 之后到）。回复下方画一行
  // 「记住了 … ✕」——写入是自动的（不弹确认框打断购物），但必须看得见、且一键撤得掉。
  learnedPrefs: LearnedPref[];
};

const TERMINAL: ReadonlySet<TaskStatus> = new Set(["done", "cancelled", "error"]);

// 会话 threadId 存 localStorage：刷新 / 重开页面后用它拉历史「回看」并接着聊（续聊的前端落点）。
// 它只是「上次看的是哪段」这个**光标**，不是数据本身——数据在后端，按账号归属。key 定义在 auth.ts，
// 因为登出时必须连它一起清（否则下一个登录的人会指着上一个人的会话，见那边的注释）。

// 侧栏的会话清单从 GET /api/sessions 来（M16）。改造前它纯活在 localStorage 里：换台设备、清个
// 缓存，后端数据明明还在，用户却再也找不回自己的对话。现在清单按 token 里的身份从归属表查，
// 登录到哪台机器都是同一份。
function toSessionMeta(s: ApiSessionMeta): SessionMeta {
  return {
    threadId: s.thread_id,
    title: s.title || "新对话",
    updatedAt: Date.parse(s.updated_at) || 0,
  };
}

function newId(): string {
  if (crypto.randomUUID) return crypto.randomUUID().replace(/-/g, "");
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

// 同源 WS 地址：开发期 location.host 是 :5173，Vite 把 /ws 代理到后端 :8000。
// 断线重连时带 last_event_id（D 块），后端据此从 Redis Stream 补发断开窗口的缺口事件。
// token 走 query（M16）：浏览器的 WebSocket API 不让设自定义请求头，Authorization 挂不上去，
// 只能挂 URL 上——后端在握手前读它验属主（server._ws_authorized）。代价是 URL 更容易被记进各类
// 访问日志，所以这枚 token 是有过期时间的短期凭证（见 auth.ts 的说明）。
function wsUrl(threadId: string, lastEventId?: string | null): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const q = new URLSearchParams();
  if (lastEventId) q.set("last_event_id", lastEventId);
  const token = wsToken();
  if (token) q.set("token", token);
  const qs = q.toString();
  return `${proto}://${location.host}/ws/${threadId}${qs ? `?${qs}` : ""}`;
}

// 比较两个 Redis Stream id（形如 "毫秒-序号"）：返回正数表示 a 更新。乱序到达时据此只记「最大」id。
function cmpStreamId(a: string, b: string): number {
  const [am, asq] = a.split("-").map(Number);
  const [bm, bsq] = b.split("-").map(Number);
  return am !== bm ? am - bm : asq - bsq;
}

// 断线自动重连上限：超过则判定后端真不可用，翻 error，不无限重连。
const MAX_RECONNECT = 5;

// 收尾（task_result）后为 memory_updated 多等的宽限时长。curator 是后处理、跑在主回复下发之后，
// 一次 LLM 判定通常 2~6s；收尾就关连接会把「记住了 …」那一行整条丢掉。等不到就静默关（偏好
// 仍已落库，用户在偏好面板照样看得到，只是这一轮少了那行提示）——降级不报错。
const MEMORY_GRACE_MS = 12000;

// 把后端逐轮历史（扁平的 user/assistant 序列）配对重建成 Turn[]：一对 user→assistant 一轮。
// assistant 轮带回来的 items（商品卡）/ activity（思考过程事件流）一并填回该轮，让回看不只剩
// 结论文本——商品卡与「思考过程」折叠区都照样还原（与实时跑出来的那轮观感一致）。
// 末尾落单的 user（理论上不出现，后端总成对写）也保留成一轮、答案留空，避免丢消息。
function rebuildTurns(history: HistoryTurn[]): Turn[] {
  const turns: Turn[] = [];
  for (const h of history) {
    if (h.role === "user") {
      turns.push({
        id: newId(),
        query: h.content,
        images: h.images ?? [],
        events: [],
        items: [],
        finalAnswer: null,
        streamingText: null,
        status: "done",
        errorMsg: null,
        elapsedMs: null,
        tokens: null,
        clarificationQuestion: null,
        learnedPrefs: [],
      });
    } else if (h.role === "assistant" && turns.length > 0 && turns[turns.length - 1].finalAnswer === null) {
      const last = turns[turns.length - 1];
      last.finalAnswer = h.content;
      last.items = h.items ?? [];
      last.events = h.activity ?? [];
      last.elapsedMs = h.elapsed_ms ?? null;
      last.tokens = h.tokens ?? null;
    }
  }
  return turns;
}

/**
 * 驱动**整段对话**的前端状态机。两件事相对单轮版有变：
 *
 * 1. **会话级 threadId**：一段对话固定一个 threadId（首次发言时生成、存 localStorage），后续
 *    每轮复用它 POST /api/task —— 后端据此把上一轮结论回喂进开局，达成「接着聊」。「新建对话」
 *    才换新 threadId。
 * 2. **多轮 turns**：每轮一条 Turn，实时事件只更新最后一轮；挂载时按 threadId 拉 /api/history
 *    回看历史轮。
 *
 * 不变的仍是 **connect-first**：本地先有 threadId → 连 WS → 收 ws_ready 再 POST 起任务，
 * 保证早期事件（session_created）不丢（根治竞态，见后端 server.py 模块说明）。
 */
export function useShoppingXTask() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  // 会话级 P_t 约束快照（session_constraints 事件实时推；换会话清空，偏好面板打开时主动拉兜底）。
  const [sessionConstraints, setSessionConstraints] = useState<SessionSnapshot | null>(null);
  const [status, setStatus] = useState<TaskStatus>("idle");
  // 侧栏历史会话列表（按 updatedAt 倒序）。首屏为空，挂载后从后端拉——它是登录用户的会话，
  // 不再是这台浏览器的会话。
  const [sessions, setSessions] = useState<SessionMeta[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  // 用 ref 镜像，让 WS 回调读到**当前**值而非闭包捕获的旧值：status 决定 onclose 要不要兜底；
  // threadId 让续聊各轮稳定复用同一个（state 更新异步，回调里读 ref 才可靠）。
  const statusRef = useRef<TaskStatus>("idle");
  const threadIdRef = useRef<string | null>(null);
  // 事件回放（D 块）：lastEventId = 本轮已收到的最大 stream id（重连补发的断点）；seenIds 去重
  // （补发与直播可能重叠）；reconnect = 当前连续重连次数（连上即清零，超 MAX_RECONNECT 放弃）。
  const lastEventIdRef = useRef<string | null>(null);
  const seenIdsRef = useRef<Set<string>>(new Set());
  const reconnectRef = useRef(0);
  // 本轮参考图在服务端的文件名：startTask 里先上传拿到，ws_ready 时随 POST /api/task 带上。
  // 走 ref 而不是给 openSocket 加参数——它只在「首连发任务」那一刻被读一次，用完即清。
  const pendingImagesRef = useRef<string[]>([]);
  // 「已对该 thread 发起过续看」的守卫：记当前正在续看 / 直播的 thread_id。resumeIfRunning 同步据此
  // 去重——StrictMode（dev）会把挂载 effect 跑两次、追加式重建在跑轮不幂等，会重出两条一样的对话；
  // 同步守卫（在 fetchInflight 之前就置位）避开 promise 解析顺序的竞态。离开对话（teardownActive）清空，
  // 切走再切回能重新续看。刷新是新页面、ref 复位为 null，照常续看。
  const resumedThreadRef = useRef<string | null>(null);
  // 重连退避定时器 id：unmount 时要清掉，否则挂起的 setTimeout 会在组件卸载后再建一个 WS（泄漏）。
  const reconnectTimerRef = useRef<number | null>(null);
  // openSocket 实现放 ref：它要递归调用自己（重连），useCallback 自引用麻烦；每次渲染重赋一份
  // 读最新值的实现（函数体只读 ref / 稳定的 useCallback，不依赖闭包 state）。
  const openSocketRef = useRef<
    (tid: string, query: string, userId: string | undefined, reconnect: boolean) => void
  >(() => {});

  const setStatusSafe = useCallback((s: TaskStatus) => {
    statusRef.current = s;
    setStatus(s);
  }, []);

  // 把一段会话登记 / 刷新到历史列表：新会话以首轮 query 作 title 插到最前；已存在的只刷新
  // updatedAt（title 一旦定下就不被后续轮覆盖——首轮 query 即整段主题），刷完按最近活跃重排。
  //
  // 这里只做**乐观的本地更新**（让侧栏立刻出现这段对话，不必等一次往返）。真正落库的是后端：
  // POST /api/task 会 claim_thread 认领归属，下次拉 /api/sessions 自然一致。
  const touchSession = useCallback((tid: string, title: string) => {
    setSessions((prev) => {
      const now = Date.now();
      const existing = prev.some((s) => s.threadId === tid);
      const next = existing
        ? prev.map((s) => (s.threadId === tid ? { ...s, updatedAt: now } : s))
        : [{ threadId: tid, title: title.slice(0, 80) || "新对话", updatedAt: now }, ...prev];
      next.sort((a, b) => b.updatedAt - a.updatedAt);
      return next;
    });
  }, []);

  // 挂载时拉一次我的会话清单（登录用户的，不是这台浏览器的）。
  useEffect(() => {
    void fetchSessions().then((list) => setSessions(list.map(toSessionMeta)));
  }, []);

  // 只改「最后一轮」（实时事件的归属轮）。同一时刻仅一个任务在跑，故末轮即活动轮。
  const patchLastTurn = useCallback((delta: (t: Turn) => Partial<Turn>) => {
    setTurns((prev) => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      return [...prev.slice(0, -1), { ...last, ...delta(last) }];
    });
  }, []);

  // 进入某对话时：若该 thread 仍有任务在后台跑，重建「正在跑的那一轮」并续直播（刷新/切回共用）。
  // 任务与 WS 解耦、永不因切换/刷新被隐式打断（teardownActive 不再 cancel），这里只是重新订阅 +
  // 补缺口：拉 /inflight 拿到在跑轮的提问与已发生事件 → 追加一条 running 轮 → reconnect 模式开 WS
  // （不重发 POST，带 last_event_id 续直播）。未在跑则什么都不做（历史轮已由 fetchHistory 渲染）。
  const resumeIfRunning = useCallback(
    (tid: string) => {
      // 同步去重：同一 thread 只发起一次续看。StrictMode 双跑挂载 effect / 重复进入都会调到这里，
      // 不挡就会追加出两条一样的在跑轮。置位放在 fetchInflight 之前（同步），不受 promise 解析顺序影响。
      if (resumedThreadRef.current === tid) return;
      resumedThreadRef.current = tid;
      fetchInflight(tid)
        .then((inflight) => {
          // 拉取期间用户若又切走了，丢弃这次结果，别把旧 thread 的在跑轮糊到当前视图。
          if (threadIdRef.current !== tid || !inflight.running) return;
          const replayed = inflight.events ?? [];
          // 回放里的 items_preview 与实时路径同样待遇：不进 events（不是思考行），而是还原成商品卡——
          // 刷新 / 切回时已推过的卡片跟着回来，不必干等收尾重发一遍。取最后一条（picker 可能跑多次）。
          const events = replayed.filter((e) => e.event !== "items_preview");
          const lastPreview = [...replayed].reverse().find((e) => e.event === "items_preview");
          const previewItems = (lastPreview?.data.items as ProductItem[]) ?? [];
          // 澄清等待中刷新/切回：回放里最后一条 clarification_request 之后若还没出现 ask_user 的
          // tool_end（用户回复 / 超时兜底都会发它），说明 Agent 仍阻塞在这个提问上——恢复 waiting
          // 状态与横幅。不恢复的话输入框回不了话，用户只能干等 ask_user 超时。
          let pendingQuestion: string | null = null;
          let pendingOptions: string[] | null = null;
          let pendingMulti = false;
          let pendingPreselected: string[] | null = null;
          for (let i = replayed.length - 1; i >= 0; i--) {
            const e = replayed[i];
            if (e.event === "tool_end" && e.data.tool === "ask_user") break;
            if (e.event === "clarification_request") {
              pendingQuestion = (e.data.question as string) ?? "";
              pendingOptions = (e.data.options as string[] | undefined) ?? null;
              pendingMulti = (e.data.multi_select as boolean | undefined) ?? false;
              pendingPreselected = (e.data.preselected as string[] | undefined) ?? null;
              break;
            }
          }
          // 重置回放游标：seenIds 装已收到事件去重；lastEventId 取最大，作为重连补发的断点。
          // 游标按**全部**回放事件算（含 preview），否则重连会把它当缺口再补一遍。
          const seen = new Set<string>();
          let maxId: string | null = null;
          for (const e of replayed) {
            if (!e.id) continue;
            seen.add(e.id);
            if (!maxId || cmpStreamId(e.id, maxId) > 0) maxId = e.id;
          }
          seenIdsRef.current = seen;
          lastEventIdRef.current = maxId;
          reconnectRef.current = 0;
          // 追加一条「正在跑」的活动轮（历史轮已渲染在前），填入已发生的事件还原思考过程。
          setTurns((prev) => [
            ...prev,
            {
              id: newId(),
              query: inflight.query ?? "",
              images: inflight.images ?? [],
              events: [...events],
              items: previewItems,
              finalAnswer: null,
              streamingText: null,
              status: pendingQuestion !== null ? ("waiting" as TaskStatus) : "running",
              errorMsg: null,
              elapsedMs: null,
              tokens: null,
              clarificationQuestion: pendingQuestion,
              clarificationOptions: pendingOptions,
              clarificationMultiSelect: pendingMulti,
              clarificationPreselected: pendingPreselected,
              learnedPrefs: [],
            },
          ]);
          setStatusSafe(pendingQuestion !== null ? "waiting" : "running");
          openSocketRef.current(tid, inflight.query ?? "", undefined, true);
        })
        .catch(() => undefined);
    },
    [setStatusSafe],
  );

  // 挂载时回看：localStorage 存过 threadId 就拉历史重建对话流，让刷新/重进能接着上次聊。
  // 拉完历史再 resumeIfRunning：若该 thread 仍有任务在后台跑（刷新前发起、未收尾），自动续看。
  useEffect(() => {
    const saved = localStorage.getItem(THREAD_KEY);
    if (!saved) return;
    threadIdRef.current = saved;
    setThreadId(saved);
    fetchHistory(saved)
      .then((history) => {
        if (threadIdRef.current !== saved) return; // 期间已切走，丢弃
        const restored = rebuildTurns(history);
        if (restored.length) {
          setTurns(restored);
          setStatusSafe("done");
          // 兼容旧版（彼时只存单个 threadId、无 sessions 索引）：补登一条，免得历史列表空着。
          touchSession(saved, restored[0].query);
        }
        resumeIfRunning(saved);
      })
      .catch(() => undefined);
  }, [setStatusSafe, touchSession, resumeIfRunning]);

  // 组件卸载时关掉残留连接 + 清掉挂起的重连定时器，免得 onmessage 继续对已卸载组件 setState
  // （泄漏 + React 告警），也免得退避中的 setTimeout 在卸载后又建一个新 WS。
  useEffect(
    () => () => {
      if (reconnectTimerRef.current !== null) window.clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    },
    [],
  );

  // 建 WS 并绑回调。reconnect=false 是首连（收 ws_ready 后 POST 起任务）；reconnect=true 是断线
  // 重连（任务还在后台跑，不重发 POST，靠 last_event_id 补发缺口 + 直播续看）。每次渲染重赋实现，
  // 函数体只读 ref / 稳定 callback，闭包无旧 state 之虞。
  openSocketRef.current = (tid, query, userId, reconnect) => {
    const prevWs = wsRef.current;
    if (prevWs) {
      prevWs.onmessage = prevWs.onerror = prevWs.onclose = null;
      prevWs.close();
    }
    const ws = new WebSocket(wsUrl(tid, reconnect ? lastEventIdRef.current : null));
    wsRef.current = ws;
    // 收尾后等 memory_updated 的宽限定时器（见 task_result 分支）。收到即清并关；超时自兜底。
    let closeTimer: number | null = null;

    ws.onmessage = (ev) => {
      // 后端帧有两类：connect-first 的 {type:"ws_ready"} 控制帧、和 {type:"monitor_event",...}
      // 事件帧；还可能有非 JSON 的 pong。先按宽松形状解析，解析失败就忽略这帧，别让回调炸掉。
      let msg: { type?: string };
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }

      // connect-first 握手：连接登记完成才发任务，确保不丢早期事件。
      if (msg.type === "ws_ready") {
        reconnectRef.current = 0; // 连上即清零重连计数
        setStatusSafe("running");
        patchLastTurn(() => ({ status: "running" }));
        if (!reconnect) {
          // 仅首连发任务；重连时任务已在后台跑，重发会起第二个任务（同 thread 会被覆盖，但语义错）。
          const images = pendingImagesRef.current;
          pendingImagesRef.current = []; // 用完即清：下一轮没传图就不该还带着上一轮的
          startTaskRequest(query, tid, userId, images).catch((e) => {
            // 额度耗尽（402）在这里与其它启动失败走同一条路：翻 error 态、把这一轮标红。区别只在
            // 文案（describeStartError 认得它，给出重置时刻）。App 侧的余额条会因 status 变 error
            // 而重拉 /api/quota，随即把输入框锁掉——不必再从这里回传什么。
            setStatusSafe("error");
            patchLastTurn(() => ({ status: "error", errorMsg: describeStartError(e) }));
            ws.close();
          });
        }
        return;
      }

      if (msg.type !== "monitor_event") return;
      const evt = msg as AguiEvent;
      // 去重 + 记断点（D 块）：补发与直播可能重叠，按 stream id 去重；lastEventId 记最大（容忍乱序到达）。
      if (evt.id) {
        if (seenIdsRef.current.has(evt.id)) return;
        seenIdsRef.current.add(evt.id);
        if (!lastEventIdRef.current || cmpStreamId(evt.id, lastEventIdRef.current) > 0) {
          lastEventIdRef.current = evt.id;
        }
      }
      // 召回预览卡：任务还在跑就先把候选画出来（感知提速的落点）。跨平台 fork 时每个子 Agent 各推
      // 一批，按 item_id 合并去重。**不进 events**——ActivityFeed 只认 tool_start/tool_end 那套画
      // 思考行，混进去会多出一条「工具调用·运行中」的脏行。
      // 「记住了 …」那一行：curator 是后处理，这条事件在 task_result **之后**才到，故不进 events
      // （ActivityFeed 的思考流已随收尾定格），直接挂到该轮的 learnedPrefs 上追加渲染。
      // 收尾文案流式预览：累计全文（幂等，乱序/丢条不错位），只更新最后一轮的 streamingText。
      // **不进 events**——ActivityFeed 只画思考行，几十条渐进文本混进去全是脏行。
      if (evt.event === "summary_delta") {
        const text = (evt.data.text as string) ?? "";
        if (text) patchLastTurn(() => ({ streamingText: text }));
        return;
      }

      // 商品卡先出货：item_picker 定稿即推，卡片先上桌、文案（summary_delta）随后逐字补。
      // **不进 events**——它是结果本身，混进 ActivityFeed 会多出一行没有工具名的脏行。
      // 收尾时 task_result 的定稿会原样覆盖（同构、同序），用户不会看到卡片跳动。
      if (evt.event === "items_preview") {
        const preview = (evt.data.items as ProductItem[]) ?? [];
        if (preview.length) patchLastTurn(() => ({ items: preview }));
        return;
      }

      // 会话约束快照（瞬态事件）：只喂偏好面板的「本次会话」区，**不进 events**（不是思考行）。
      if (evt.event === "session_constraints") {
        setSessionConstraints(evt.data as unknown as SessionSnapshot);
        return;
      }

      if (evt.event === "memory_updated") {
        const learned = (evt.data.preferences as LearnedPref[]) ?? [];
        patchLastTurn((t) => ({ learnedPrefs: [...t.learnedPrefs, ...learned] }));
        // 本轮最后一条事件——收尾时挂起的宽限定时器可以提前兑现了。
        if (closeTimer !== null) {
          clearTimeout(closeTimer);
          closeTimer = null;
          ws.close();
        }
        return;
      }

      patchLastTurn((t) => ({ events: [...t.events, evt] }));

      switch (evt.event) {
        case "task_result":
          patchLastTurn((t) => ({
            streamingText: null, // 定稿到达，流式预览退场（finalAnswer 接管渲染）
            finalAnswer: (evt.data.final_answer as string) ?? "",
            // 定稿覆盖预览卡；但**定稿为空时保留已推的预览**——模型偶尔不走 shopping_summary 就
            // 收尾（如直接文字作答），那时 items 为空，清掉会让用户眼看着卡片凭空消失。
            items: ((evt.data.items as ProductItem[]) ?? []).length
              ? (evt.data.items as ProductItem[])
              : t.items,
            elapsedMs: (evt.data.elapsed_ms as number) ?? null,
            tokens: (evt.data.tokens as TurnTokens) ?? null,
            status: "done",
          }));
          setStatusSafe("done");
          // **不立刻关连接**：curator 是后处理，memory_updated（「记住了 …」那一行）在 task_result
          // 之后几秒才发——收尾即 close 会把它整条丢掉（本轮真出现过：偏好已落库，页面却没那行）。
          // 留一段宽限：收到 memory_updated 立刻关；curator 没学到东西 / 失败则超时自动关。
          closeTimer = window.setTimeout(() => ws.close(), MEMORY_GRACE_MS);
          break;
        case "task_cancelled":
          patchLastTurn(() => ({ status: "cancelled" }));
          setStatusSafe("cancelled");
          ws.close();
          break;
        case "error":
          patchLastTurn(() => ({
            status: "error",
            errorMsg: (evt.data.message as string) ?? evt.message,
          }));
          setStatusSafe("error");
          ws.close();
          break;
        case "clarification_request":
          patchLastTurn(() => ({
            status: "waiting" as TaskStatus,
            clarificationQuestion: (evt.data.question as string) ?? "",
            clarificationOptions: (evt.data.options as string[] | undefined) ?? null,
            clarificationMultiSelect: (evt.data.multi_select as boolean | undefined) ?? false,
            clarificationPreselected: (evt.data.preselected as string[] | undefined) ?? null,
          }));
          setStatusSafe("waiting");
          break;
      }
    };

    ws.onerror = () => {
      // 不在这里报错：交给 onclose 决定「重连还是放弃」——断线自动重连补发是 D 块的核心。
    };

    ws.onclose = () => {
      if (closeTimer !== null) {
        clearTimeout(closeTimer); // 连接已断（含用户切走 / 刷新），别再让宽限定时器空转
        closeTimer = null;
      }
      if (TERMINAL.has(statusRef.current)) return; // 正常收尾后服务端关连接，忽略
      // 非终态断开（刷新 / 断网 / 代理重置）：自动重连，带 last_event_id 补发断开窗口的缺口（D 块）。
      // 任务在后台与 WS 解耦地继续跑，重连只是重新订阅 + 补洞，用户无缝续看。
      if (reconnectRef.current < MAX_RECONNECT) {
        reconnectRef.current += 1;
        const delay = Math.min(800 * 2 ** (reconnectRef.current - 1), 8000); // 指数退避，封顶 8s
        reconnectTimerRef.current = window.setTimeout(() => {
          reconnectTimerRef.current = null;
          if (!TERMINAL.has(statusRef.current) && threadIdRef.current === tid) {
            openSocketRef.current(tid, query, userId, true);
          }
        }, delay);
      } else {
        setStatusSafe("error");
        patchLastTurn(() => ({ status: "error", errorMsg: "连接已断开，多次重连失败。" }));
      }
    };
  };

  const startTask = useCallback(
    async (query: string, userId?: string, files?: File[]) => {
      // 「只发一张图、一个字不打」是完整意图（就照这张图找），不能被空 query 守卫拦掉——
      // 否则用户点发送后界面毫无反应，而这恰恰是图搜最自然的用法。
      if (!query.trim() && !files?.length) return;
      if (statusRef.current === "connecting" || statusRef.current === "running" || statusRef.current === "waiting") {
        return;
      }
      // 只发图不打字时补一句默认意图：空 query 会让 planner 拆出一片空白，会话标题也是空的。
      // 这句话说的正是用户此刻的意思，让它成为这一轮如实的提问原文。
      query = query.trim() || "照这张参考图，帮我找类似的商品";
      // 上一轮若在 ws_ready 之前就失败（WS 没连上），它上传的图还挂在 ref 上。开局先清，
      // 免得这一轮纯文字提问却莫名带着上一轮的参考图。
      pendingImagesRef.current = [];
      // 先把旧连接的回调摘掉再关，避免上一轮 socket 残留帧改到这一轮的 state。
      const prev = wsRef.current;
      if (prev) {
        prev.onmessage = prev.onerror = prev.onclose = null;
        prev.close();
      }

      // 会话 threadId：首轮生成并落 localStorage；后续轮复用（续聊的关键——同 thread 才接上文）。
      let tid = threadIdRef.current;
      if (!tid) {
        tid = newId();
        threadIdRef.current = tid;
        setThreadId(tid);
        localStorage.setItem(THREAD_KEY, tid);
      }
      // 登记到历史列表：新会话以首轮 query 入列，续聊轮则把它顶到最近活跃。
      touchSession(tid, query);

      // 追加一条活动轮（其余轮已冻结为历史）。不复位别的轮，多轮对话流逐条累加。
      setTurns((prevTurns) => [
        ...prevTurns,
        { id: newId(), query, images: [], events: [], items: [], finalAnswer: null, streamingText: null, status: "connecting", errorMsg: null, elapsedMs: null, tokens: null, clarificationQuestion: null, learnedPrefs: [] },
      ]);
      setStatusSafe("connecting");

      // 参考图先上传、后连 WS：Agent 一开局就会去读图（Harness 预跑 image_understand，先于
      // planner），文件必须在任务起跑前已经落在服务端盘上。上传失败就把这一轮标红收场——
      // 用户明明是拿图来搜的，静默丢掉图去跑纯文字检索只会给出一份文不对题的清单。
      if (files?.length) {
        try {
          pendingImagesRef.current = await Promise.all(files.map((f) => uploadImage(tid, f)));
          // 图名进本轮 turn：气泡里立刻画出缩略图。用服务端回的文件名而非本地 blob——这一轮
          // 刷新后、乃至下次从历史点回来，走的都是同一条渲染路径（/api/uploads 取图）。
          patchLastTurn(() => ({ images: pendingImagesRef.current }));
        } catch (e) {
          setStatusSafe("error");
          patchLastTurn(() => ({
            status: "error",
            errorMsg: e instanceof Error ? e.message : "参考图上传失败",
          }));
          return;
        }
      }

      // 新一轮：复位回放游标（新事件流从头，首连不带 last_event_id、不补发）。
      lastEventIdRef.current = null;
      seenIdsRef.current = new Set();
      reconnectRef.current = 0;
      // 本轮起就在 tid 上直播：置位续看守卫，挡掉同 thread 的重复续看（这一轮已经在看了）。
      resumedThreadRef.current = tid;
      openSocketRef.current(tid, query, userId, false);
    },
    [setStatusSafe, patchLastTurn, touchSession],
  );

  const cancelTask = useCallback(() => {
    const tid = threadIdRef.current;
    if (tid) cancelTaskRequest(tid).catch(() => undefined);
  }, []);

  const sendClarification = useCallback(
    (text: string) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !text.trim()) return;
      ws.send(JSON.stringify({ type: "clarification_response", text: text.trim() }));
      patchLastTurn(() => ({
        status: "running" as TaskStatus,
        clarificationQuestion: null,
        clarificationOptions: null,
      }));
      setStatusSafe("running");
    },
    [patchLastTurn, setStatusSafe],
  );

  // 摘掉并关闭当前 WS 连接——切换 / 新建 / 删除会话前调用，免得旧 socket 的残留帧改到新视图的
  // state（串台）。**只断订阅、不取消任务**：任务与 WS 解耦，切走 / 刷新都不该打断后台在跑的请求
  // （用户语义「看别的对话不影响我的请求」）。要真正终止得点显式「取消」按钮（cancelTask）。先把
  // 回调摘掉再 close，onclose 便不会触发自动重连。
  const teardownActive = useCallback(() => {
    const prev = wsRef.current;
    if (prev) {
      prev.onmessage = prev.onerror = prev.onclose = null;
      prev.close();
    }
    wsRef.current = null;
    // 离开当前对话：清掉续看守卫，下次进入（切回 / 选别的会话）能重新发起续看。
    resumedThreadRef.current = null;
  }, []);

  // 点侧栏某条历史：切到它的 threadId，拉 /api/history 回看那段对话，并可接着聊（同 thread 续上文）。
  const selectConversation = useCallback(
    (tid: string) => {
      if (threadIdRef.current === tid) return;
      teardownActive();
      threadIdRef.current = tid;
      setThreadId(tid);
      localStorage.setItem(THREAD_KEY, tid);
      setTurns([]);
      setSessionConstraints(null); // 旧会话的约束别糊到新会话；面板打开时按新 thread 主动拉
      setStatusSafe("idle");
      fetchHistory(tid)
        .then((history) => {
          // 拉取期间用户若又点了别的会话，丢弃这次结果，别把旧会话的轮糊到当前。
          if (threadIdRef.current !== tid) return;
          const restored = rebuildTurns(history);
          setTurns(restored);
          setStatusSafe(restored.length ? "done" : "idle");
          // 切回的这段对话若仍有任务在后台跑（切走时没被取消），重建在跑轮 + 续直播。
          resumeIfRunning(tid);
        })
        .catch(() => undefined);
    },
    [setStatusSafe, teardownActive, resumeIfRunning],
  );

  // 从历史列表删除一段会话：删后端的归属记录（否则刷新一下它就回来了——清单的真源在后端），
  // 但 output/<tid>/ 里的对话正文仍留着，语义与改造前一致「删的是入口，不是数据」。
  // 删的若是当前会话，顺带清空主区回到空态（等价新建对话）。
  const deleteConversation = useCallback(
    (tid: string) => {
      // 先本地摘掉（点了就消失，不等往返），再打后端。删除是幂等的，失败了大不了刷新后重现。
      setSessions((prev) => prev.filter((s) => s.threadId !== tid));
      void deleteSession(tid);
      if (threadIdRef.current === tid) {
        teardownActive();
        threadIdRef.current = null;
        localStorage.removeItem(THREAD_KEY);
        setThreadId(null);
        setTurns([]);
        setSessionConstraints(null);
        setStatusSafe("idle");
      }
    },
    [setStatusSafe, teardownActive],
  );

  // 新建对话：换一段全新会话——断连接、清空轮、丢掉旧 threadId（含 localStorage），回到空态。
  // 不动 sessions 列表：旧会话起任务时已登记进历史，新建只是「不再指向它」，下次发言才生成新 threadId。
  const newConversation = useCallback(() => {
    teardownActive();
    threadIdRef.current = null;
    localStorage.removeItem(THREAD_KEY);
    setThreadId(null);
    setTurns([]);
    setSessionConstraints(null);
    setStatusSafe("idle");
  }, [setStatusSafe, teardownActive]);

  const running = status === "connecting" || status === "running";
  const waiting = status === "waiting";
  return {
    threadId,
    turns,
    status,
    running,
    waiting,
    sessions,
    sessionConstraints,
    setSessionConstraints,
    startTask,
    cancelTask,
    sendClarification,
    newConversation,
    selectConversation,
    deleteConversation,
  };
}
