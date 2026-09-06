# globex 的 K8s 编排（批2-6）

把批 2 做出来的双进程形态（API 收请求 / worker 跑 AgentLoop，中间 Redis Stream 削峰）用 K8s
表达出来。五件套：`redis` / `qdrant` / `opensearch` / `globex-api` / `globex-worker`。

## 应用顺序

```bash
# 1. 镜像（api 与 worker 共用；VPS 与 Mac 架构不同，在目标集群那侧 build）
docker build -f docker/Dockerfile.backend -t globex-backend:latest .

# 2. 密钥：不要用 00- 里那份占位 Secret，直接从本机 .env 灌
kubectl -n globex create secret generic globex-secrets --from-env-file=.env

# 3. 按编号顺序（编号即依赖顺序：配置 → 三个依赖 → api → worker）
kubectl apply -f deploy/k8s/
```

## 三个数的大小关系（`50-worker.yaml` 的核心）

```
terminationGracePeriodSeconds (150s)  >  preStop (5s) + WORKER_GRACE_SECONDS (120s)
```

preStop 的耗时**算在 grace 里面**，不是加在外面——这是最容易记反的一条。多出来的 25s 留给
「等完在飞任务之后」的收尾（停控制面订阅、关队列客户端、发最后一批事件）。配小了不会丢任务
（消息没 ack，留在 PEL 里被下一个 worker 领回重跑），但会白跑半程。

## 滚动更新为什么不丢任务

三样东西叠起来，缺一不可：

1. **`maxUnavailable: 0` + `maxSurge: 1`** —— 新 worker 先起来，消费能力全程不掉到 0。
2. **worker 自己处理 SIGTERM**（批2-2）—— 先停领新的，再等在飞任务最多 `WORKER_GRACE_SECONDS`。
3. **at-least-once + `XAUTOCLAIM`**（批2-1）—— 真等超时了就取消、**不 ack**，消息留在 PEL 里，
   由下一个 worker 领回重跑。所以最坏情况是「重跑一次」，不是「消失」。

## 校验状态（诚实标注）

- ✅ **`kubeconform -strict -kubernetes-version 1.29.0`**：13 个资源全部 Valid（`-strict` 会
  拒绝未知字段，能挡住拼错的 key —— 而拼错的 key 在 K8s 里是**静默忽略**的，最坏的一种失败）。
- ⏭ **kind 上的滚动更新实测：未做。** 原因不是懒得跑，是代价与收益不成比例：要先装 kind、拉
  ~1GB 的 node 镜像、再 build 一个几 GB 的应用镜像（`uv sync` 全量依赖）load 进去，还得给
  OpenSearch 改节点内核参数——这已经是「在本机装一套 K8s 全家桶」，本单明确不做。
- ✅ **等价验证做了**：`scripts/loadtest.py` 那一轮之后，在本机双进程栈上模拟了一次滚动更新
  （老 worker 收 SIGTERM 的同时起新 worker，任务在飞），结果与本文「不丢任务」的三条机制一致。
  数据见 `docs/plans/批2-进度.md` 的批2-6 报告。**它验的是应用侧那两条（2、3）**，
  K8s 侧那条（`maxSurge`/`maxUnavailable` 的调度行为）仍未在真集群上跑过。

## 刻意没写的

Ingress / TLS（各集群的 controller 与证书方案不同，写死反而误导）、HPA（要先有稳定的
`queue_depth` 指标源，属批 3 的观测腿）、PodDisruptionBudget、NetworkPolicy、前端静态站
（现在由 Caddy + nginx 托管，见 `docker/docker-compose.prod.yml`）、PostgreSQL（配置里留了
`DATABASE_URL` 的注释行，真换库要先在影子库上跑一遍 `alembic upgrade head`）。
