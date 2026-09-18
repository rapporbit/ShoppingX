// k6 压测：走真前端那条路（connect-first：连 WS → 等 ws_ready → POST /api/task → 读事件到终态）。
// 为什么不用 scripts/loadtest.py：Python 客户端每建一条连接要 4~9ms 纯 CPU，500 个虚拟用户
// 各开 WS + HTTP 两条连接，客户端自己就排出 5s，量到的全是假延迟（2026-09-18 实测）。k6 是 Go
// 写的，500 VU 不到一核，客户端先自证不是瓶颈。
//
// 跑法（另一个终端先起打桩服务，见 loadtest_stub_server.py）：
//   k6 run scripts/loadtest.k6.js                              # 默认 100/200/500 三档突发
//   k6 run -e BASE=http://127.0.0.1:8199 -e VUS=500 scripts/loadtest.k6.js
// macOS 先 `sudo sysctl -w kern.ipc.somaxconn=2048`，否则 500×2 条连接把 128 的监听队列打穿。
import http from "k6/http";
import ws from "k6/ws";
import { Counter, Trend } from "k6/metrics";

const BASE = __ENV.BASE || "http://127.0.0.1:8199";
const WS_BASE = BASE.replace("http://", "ws://").replace("https://", "wss://");
const TIMEOUT_MS = Number(__ENV.TIMEOUT_MS || 60000);

const rejected = new Counter("rejected_429");
const succeeded = new Counter("succeeded");
const failed = new Counter("failed_other");
const t429 = new Trend("latency_429", true); // 连 WS 起 → 拿到 429
const tFirst = new Trend("latency_first_event", true); // 连 WS 起 → 第一条 monitor_event
const tTotal = new Trend("latency_total", true); // 连 WS 起 → 终态

// 一档「同时按下」突发：VUS 个虚拟用户各跑一次。分档用 -e VUS=100/200/500 跑三遍，
// 每遍一张独立汇总，比一份里按 scenario 标签拆好读。
const VUS = Number(__ENV.VUS || 100);
export const options = {
  scenarios: {
    burst: { executor: "per-vu-iterations", vus: VUS, iterations: 1, maxDuration: "90s" },
  },
  thresholds: { failed_other: ["count==0"] },
  summaryTrendStats: ["p(50)", "p(95)", "max"],
};

function tid() {
  return "k6-" + Math.random().toString(16).slice(2, 14);
}

export default function () {
  const thread = tid();
  const t0 = Date.now();
  let first = null;
  ws.connect(`${WS_BASE}/ws/${thread}`, {}, (socket) => {
    socket.setTimeout(() => {
      failed.add(1);
      socket.close();
    }, TIMEOUT_MS);
    socket.on("message", (raw) => {
      const msg = JSON.parse(raw);
      if (msg.type === "ws_ready") {
        const res = http.post(
          `${BASE}/api/task`,
          JSON.stringify({ query: "买一个通勤双肩包，预算 300", thread_id: thread }),
          { headers: { "Content-Type": "application/json" }, timeout: `${TIMEOUT_MS}ms` },
        );
        if (res.status === 429) {
          rejected.add(1);
          t429.add(Date.now() - t0);
          socket.close();
        } else if (res.status >= 400) {
          failed.add(1);
          socket.close();
        }
        return;
      }
      if (msg.type !== "monitor_event") return;
      if (first === null) {
        first = Date.now() - t0;
        tFirst.add(first);
      }
      if (msg.event === "task_result") {
        succeeded.add(1);
        tTotal.add(Date.now() - t0);
        socket.close();
      } else if (msg.event === "task_cancelled" || msg.event === "error") {
        failed.add(1);
        socket.close();
      }
    });
    socket.on("error", () => failed.add(1));
  });
}
