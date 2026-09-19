"""多副本验收：四个场景，每个都给出「判据 + 实测」两行，最后一张总表。

对应 `docker/docker-compose.multi.yml`（2 API + 2 worker + MySQL 8 + Redis，打了桩不调模型）。
跑法::

    python scripts/multi_scenarios.py all          # 四个都跑
    python scripts/multi_scenarios.py dup          # 只跑某一个

**为什么非要两个副本**：1-1~1-4 那几刀（预扣、threads 条件更新、Redis dedup、关停 interrupted）
在单进程下全都「看着是对的」——进程内一个字典也能挡住同 thread 双击。把状态挪进 DB / Redis 的收益
只有在两个进程同时抢同一个 thread 时才显形，也只有那时才证伪得了。

**所有请求都从容器内发**：compose 不发布端口（在 VPS 上跑时这是硬要求），所以走
``docker exec <容器> python -c ...``，镜像里有 python，不必额外装 curl。
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from typing import Any

COMPOSE = ["docker", "compose", "-f", "docker/docker-compose.multi.yml"]
API1, API2 = "globex-multi-api1", "globex-multi-api2"
MYSQL = "globex-multi-mysql"
USER_ID = "acceptance0000000000000000000000"
TOKEN = ""  # main() 里换到手后填上；空串 = 不带 Authorization 头


def _run(cmd: list[str], timeout: int = 120) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"命令失败：{' '.join(cmd[:4])}…\n{out.stderr.strip()[:400]}")
    return out.stdout.strip()


def _http(container: str, target: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """从 ``container`` 里向 ``target`` 发一条请求，返回 ``{"code", "body"}``。

    入参走 base64 而不是往代码字符串里拼 —— 拼进去的 query 一旦带引号就是一场引号地狱，而验收脚本
    最不该输在这种地方。异常也收进返回值（``code=0``）：连不上本身就是某些场景要的结论。
    """
    blob = base64.b64encode(json.dumps(payload or {}).encode()).decode()
    auth = f"'Authorization':'Bearer {TOKEN}'," if TOKEN else ""
    code = (
        "import base64,json,urllib.request,urllib.error\n"
        f"url='http://{target}:8000{path}'\n"
        f"data=base64.b64decode('{blob}') if {payload is not None} else None\n"
        "req=urllib.request.Request(url,data=data,"
        f"headers={{{auth}'Content-Type':'application/json'}})\n"
        "try:\n"
        "    r=urllib.request.urlopen(req,timeout=30)\n"
        "    print(json.dumps({'code':r.status,'body':json.loads(r.read() or b'{}')}))\n"
        "except urllib.error.HTTPError as e:\n"
        "    print(json.dumps({'code':e.code,'body':e.read().decode()[:300]}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'code':0,'body':str(e)[:300]}))"
    )
    return json.loads(_run(["docker", "exec", container, "python", "-c", code]))


def post(container: str, target: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _http(container, target, path, payload)


def get(container: str, target: str, path: str) -> dict[str, Any]:
    return _http(container, target, path, None)


def sql(query: str) -> list[list[str]]:
    """查验收库。判据尽量落到 SQL 上——HTTP 响应说的是「这次请求怎么回的」，表里才是真相。"""
    out = _run(
        ["docker", "exec", MYSQL, "mysql", "-uroot", "-pglobex", "-N", "-B", "globex", "-e", query]
    )
    return [line.split("\t") for line in out.splitlines() if line]


def ensure_user_and_token() -> str:
    """建一个验收用户并换一张 token。

    **holds 只在开了鉴权时才生效**（quota_enabled 要求有可信身份），所以这套验收必须带身份跑——
    关着鉴权跑，run_holds 会是空表，1-1 那几刀一条都验不到（第一次跑正是这么翻的车）。

    用户直接插库：发证口只签 token 不建人，而 ``claim_thread`` 拿 sub 去 users 表查，查无此人就
    401。验收台不需要真密码（摘要填个占位）。
    """
    sql(
        f"INSERT IGNORE INTO users (id, username, password_hash, created_at) "
        f"VALUES ('{USER_ID}', 'acceptance', 'x', NOW())"
    )
    # 清掉上一轮跑剩的在飞 hold：并发上限按「这个人有几条 queued/running」算，残留会让下一次跑
    # 直接 429（第一次重跑就是这么卡住的）。只动验收用户自己的行。
    sql(
        f"UPDATE run_holds SET state='settled' "
        f"WHERE user_id='{USER_ID}' AND state IN ('queued','running')"
    )
    resp = post(API1, "api1", "/api/auth/token", {"user_id": USER_ID})
    body = resp.get("body")
    token = body.get("access_token", "") if isinstance(body, dict) else ""
    if not token:
        raise RuntimeError(f"换 token 失败：{resp}")
    return token


def wait_for(fn: Any, limit_s: float = 60, step: float = 1.0) -> Any:
    """轮询等条件成立，返回最后一次取值（超时不抛，交给调用方断言，失败信息更有用）。"""
    deadline = time.monotonic() + limit_s
    value = None
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(step)
    return value


# ── 场景 1：同 thread 双击打到两台 API ──────────────────────────────────────────
def scenario_dup() -> tuple[bool, str]:
    """判据：同一句话打到两台副本，**只有一个 run 起来**，另一台回 already_running。

    进程内字典在这里必然漏判——两台各有各的字典，各自都觉得「这个 thread 没人在跑」。真相挪进
    threads 的条件 UPDATE 之后，判定与占位是同一条语句，数据库保证只有一条的影响行数是 1。
    """
    tid = f"dup-{int(time.time())}"
    body = {"query": "买个帐篷 sleep=20", "thread_id": tid}
    first = post(API1, "api1", "/api/task", body)
    second = post(API2, "api2", "/api/task", body)
    states = sorted([first["body"].get("status", "?"), second["body"].get("status", "?")])
    rows = sql(f"SELECT active_run_id, run_status FROM threads WHERE id='{tid}'")
    ok = "already_running" in states and len({s for s in states} - {"already_running"}) == 1
    ok = ok and len(rows) == 1
    return ok, f"两次响应={states}；threads 行={rows}"


# ── 场景 2：kill -9 一个 worker ────────────────────────────────────────────────
def scenario_kill_worker() -> tuple[bool, str]:
    """判据：跑到一半的任务**不丢**——消息留在 PEL 里，另一个 worker 在 claim idle 后领回重跑。

    这条是 at-least-once 的兑现方式，也是 1-3 之后**唯一**还会走 PEL 重跑的路径：SIGKILL 下没有
    任何收尾代码能跑，与「关停掐断」那条路正好互补。
    """
    tid = f"kill-{int(time.time())}"
    task = post(API1, "api1", "/api/task/async", {"query": "买个睡袋 sleep=12", "thread_id": tid})
    task_id = task["body"].get("task_id", "")
    if not task_id:
        return False, f"任务没起来：{task}"
    time.sleep(4)  # 等它真的被某个 worker 领走并跑起来
    _run(["docker", "kill", "-s", "KILL", "globex-multi-worker1"])
    got = wait_for(
        lambda: (get(API1, "api1", f"/api/task/{task_id}")["body"] or {}).get("state") == "done",
        limit_s=90,
    )
    state = get(API1, "api1", f"/api/task/{task_id}")["body"].get("state")
    _run([*COMPOSE, "up", "-d", "worker1"])
    return bool(got), f"kill -9 worker1 后任务最终 state={state}"


# ── 场景 3：杀掉 Redis ────────────────────────────────────────────────────────
def scenario_kill_redis() -> tuple[bool, str]:
    """判据：Redis 没了就**明确拒绝**（5xx），不能静默吞掉任务。

    队列、dedup 窗口、控制面全在 Redis 上。它挂了还假装收下请求，用户等到的是永远转圈——宁可当场
    告诉他「现在不行」。
    """
    _run([*COMPOSE, "stop", "redis"])
    try:
        resp = post(API1, "api1", "/api/task", {"query": f"买个炉头 {time.time()}"})
        ok = resp["code"] >= 500 or resp["code"] == 0
        detail = f"Redis 停机时响应 code={resp['code']} body={str(resp['body'])[:120]}"
    finally:
        _run([*COMPOSE, "start", "redis"])
        time.sleep(5)
    return ok, detail


# ── 场景 4：滚动重启掐断在飞任务 ───────────────────────────────────────────────
def scenario_rolling() -> tuple[bool, str]:
    """判据（1-3）：被掐的任务落 ``interrupted``、预扣**只结算一次**、thread 占位已还。

    改造前这里是「不 ack、留 PEL、十分钟后静默整轮重跑」，账会再记一次而用户看不见那一跑。
    """
    tid = f"roll-{int(time.time())}"
    body = {"query": "买个登山杖 sleep=120", "thread_id": tid}
    task = post(API1, "api1", "/api/task/async", body)
    task_id = task["body"].get("task_id", "")
    if not task_id:
        return False, f"任务没起来：{task}"
    time.sleep(5)
    _run([*COMPOSE, "stop", "worker1", "worker2"], timeout=180)
    state = (get(API1, "api1", f"/api/task/{task_id}")["body"] or {}).get("state")
    holds = sql(
        f"SELECT state, credits_held, credits_charged FROM run_holds WHERE run_id='{task_id}'"
    )
    threads = sql(f"SELECT active_run_id, run_status FROM threads WHERE id='{tid}'")
    _run([*COMPOSE, "up", "-d", "worker1", "worker2"])
    ok = state == "interrupted" and len(holds) == 1 and holds[0][0] == "settled"
    return ok, f"state={state}；run_holds={holds}；threads={threads}"


SCENARIOS = {
    "dup": ("同 thread 双击打到两台 API", scenario_dup),
    "kill-worker": ("kill -9 一个 worker，任务被接管", scenario_kill_worker),
    "kill-redis": ("Redis 停机时明确拒绝", scenario_kill_redis),
    "rolling": ("关停掐断落 interrupted 且只计一次费", scenario_rolling),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("which", choices=[*SCENARIOS, "all"])
    args = p.parse_args()
    names = list(SCENARIOS) if args.which == "all" else [args.which]
    global TOKEN
    TOKEN = ensure_user_and_token()  # 先有身份，holds 那几条判据才有数据可看
    print(f"验收身份就绪：user_id={USER_ID[:12]}…")
    results: list[tuple[str, bool, str]] = []
    for name in names:
        title, fn = SCENARIOS[name]
        print(f"\n▶ {name}：{title}", flush=True)
        try:
            ok, detail = fn()
        except Exception as exc:  # 一个场景炸掉不该让其余三个不跑
            ok, detail = False, f"异常：{exc}"
        print(f"  {'PASS' if ok else 'FAIL'} — {detail}", flush=True)
        results.append((name, ok, detail))
    print("\n=== 汇总 ===")
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:12} {detail}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
