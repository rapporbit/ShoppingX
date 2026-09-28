# shoppingx 的 K8s 编排

把双进程形态（API 收请求 / worker 跑 AgentLoop，中间 Redis Stream 削峰）用 K8s
表达出来。四件套：`redis` / `qdrant` / `shoppingx-api` / `shoppingx-worker`。

## 应用顺序

```bash
# 1. 镜像（api 与 worker 共用；VPS 与 Mac 架构不同，在目标集群那侧 build）
docker build -f docker/Dockerfile.backend -t shoppingx-backend:latest .

# 2. 密钥：不要用 00- 里那份占位 Secret，直接从本机 .env 灌
kubectl -n shoppingx create secret generic shoppingx-secrets --from-env-file=.env

# 3. 按编号顺序（编号即依赖顺序：配置 → 三个依赖 → api → worker）
kubectl apply -f deploy/k8s/
```

## 三个数的大小关系（`50-worker.yaml` 的核心）

```
terminationGracePeriodSeconds (365s)  >  preStop (5s) + WORKER_GRACE_SECONDS (330s)
```

preStop 的耗时**算在 grace 里面**，不是加在外面——这是最容易记反的一条。多出来的 30s 留给
「等完在飞任务之后」的收尾（每条被掐的任务按 interrupted 收尾，再停控制面订阅、关队列客户端）。
WORKER_GRACE_SECONDS 本身必须 ≥ MAIN_AGENT_TIMEOUT_SEC（300），否则每次发布都在掐马上就会自己
收尾的任务。配小了不会丢任务（SIGKILL 下消息没 ack，留在 PEL 里被下一个 worker 领回重跑），但会
白跑半程、token 花两次。

## 滚动更新为什么不丢任务

三样东西叠起来，缺一不可：

1. **`maxUnavailable: 0` + `maxSurge: 1`** —— 新 worker 先起来，消费能力全程不掉到 0。
2. **worker 自己处理 SIGTERM**—— 先停领新的，再等在飞任务最多 `WORKER_GRACE_SECONDS`。
3. **at-least-once + `XAUTOCLAIM`**—— 真等超时了就取消、**不 ack**，消息留在 PEL 里，
   由下一个 worker 领回重跑。所以最坏情况是「重跑一次」，不是「消失」。

## 校验

- **`kubeconform -strict -kubernetes-version 1.29.0`**：13 个资源全部 Valid（`-strict` 会
  拒绝未知字段，能挡住拼错的 key —— 而拼错的 key 在 K8s 里是**静默忽略**的，最坏的一种失败）。
- **滚动更新的应用侧验证**：在本机双进程栈上模拟滚动更新（老 worker 收 SIGTERM 的同时起新
  worker，任务在飞），结果与本文「不丢任务」的三条机制一致。

## 不包含

Ingress / TLS（各集群的 controller 与证书方案不同，写死反而误导）、HPA（要先有稳定的
`queue_depth` 指标源，属观测侧）、PodDisruptionBudget、NetworkPolicy、前端静态站
（现在由 Caddy + nginx 托管，见 `docker/docker-compose.prod.yml`）、PostgreSQL（配置里留了
`DATABASE_URL` 的注释行，真换库要先在影子库上跑一遍 `alembic upgrade head`）。
