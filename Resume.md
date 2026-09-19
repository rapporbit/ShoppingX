# Resume_2 - 复制 - 复制 - 复制

## 基本信息

### 张家龙
Agent / 后端开发工程师
- 生日: 2002/02
- 邮箱: 1985339196@qq.com
- 电话: 18373135721
- 地址: 四川成都

## 教育经历

### 电子科技大学（985） | 电子信息

_硕士研究生 | 2024/09 - 2027/06_

### 东北大学（985） | 智能医学工程

_工学学士 | 2020/09 - 2024/06_

## 项目经历

### ShoppingX · 跨境购物 Agent（Agent 核心）

_开发者 | 2026/02 - 2026/08_

**项目描述：** 基于 LLM 的跨境购物 Agent，集成单 Agent ReAct 主循环、同轮并发多平台检索比价、Harness 控制层、长期记忆的注入与写入、Rubric 式 LLM-as-Judge 测评、embedding 和 reranker 模型微调以及 planner 模型后训练，覆盖 5 个平台 150 万商品，为用户输出跨境商品选购推荐清单。

**技术栈：** AgentScope 2.0 · Qdrant · BGE-M3 / Reranker · Qwen · Langfuse · MCP · ms-swift · vLLM

-   **调度与安全：**单 Agent 按 ReAct 循环自主决策，跨平台、多商品检索靠同轮多工具调用并发；代码层设置终结拦截、迭代上限、循环检测约束 Agent 轨迹；写操作经只读标记、权限引擎、前端确认卡三级拦截。

-   **延迟优化：**日志定位延迟瓶颈在模型解码层；通过减少模型调用次数和每轮输入 token 量，将高频组合步骤交由框架自动执行，候选商品使用 ASIN 传递。**平均模型调用 9→4 次，输入 token 79k→22.5k，耗时 28.3s→12.5s。**

-   **Harness Hook：**在 Agent 生命周期抽出 6 个 hook 切点，按终止、安全、预算、重复、漂移五类注册。修复单轮换品类时模型反复检索、反复被安全 hook 拦截的死循环。

-   **上下文工程：**按稳定性分层，静态内容前置复用前缀缓存，偏好、记忆等动态内容后置注入；业务流程外挂为 SKILL.md按意图动态加载。60 条标注验收 SKILL 零误触发。

-   **长期记忆：** 事实按 constraint / preference / context 分类，constraint 每轮全量注入，其余按更新时间注入 + 模型按需召回；写入分为模型显式保存与回合后小模型抽取双通道，PII 过滤器统一拦截。

-   **Rubric 动态评测：**用 judge 模型对线上 query 动态生成三档 Rubrics 打分，分数回注 Langfuse。**过程中发现 judge 模型存在三类系统性误判，校准后均分 32.8→52.8**；bad case 持续通过 prompt、Hook 规则、few-shot 沉淀迭代消化。

-   **检索模型微调：**针对正例在召回池但排名靠后的问题，embedding 以 ESCI 人工标注做对比学习，reranker 模型用分级标签做 listwise 训练。**R@20 0.259 → 0.591，端到端测评均分 56.4 → 68.0。**

-   **Planner 模型后训练：**针对小模型结构化输出不稳、意图拆解偏离检索需求的问题，使用 2154 条合成逐轮样本 SFT 冷启 + best-of-8 拒绝采样。**格式合法率 63% → 100%，检索命中 0.588 → 0.636。**

[https://shopx.oiuu.de](https://shopx.oiuu.de)

### ShoppingX · Agent 服务化后端

_开发者 | 2026/02 - 2026/08_

**项目描述：**为长耗时 Agent 任务设计的服务化后端，集成异步任务队列与多副本无状态部署、任务取消与断线续看、外部依赖限流熔断与降级、用户级计费与配额，保证多用户并发下任务不丢、结果可追溯。

**技术栈：** FastAPI · asyncio · Redis · MySQL · WebSocket · Docker

-   **任务调度：**使用 Redis Stream 消费者组做任务队列，at-least-once 投递 + 任务 id 幂等去重；超时未 ACK 由其他消费者 XAUTOCLAIM 接管；SIGTERM 时优雅退出。实测任一 worker kill -9，在跑任务被接管重跑，长任务不丢失。

-   **缓存设计：** 进程内 LRU + Redis 两级缓存，检索缓存 key 取 planner 结构化输出归一化拼接而非原始 query，解决同义表述不命中。索引版本号做 key 前缀实现商品库重建即失效，空结果短 TTL + singleflight 防穿透与击穿。

-   **实时推送：**WebSocket 连接收到 ws\_ready 再 POST 启动任务，首条事件不丢；事件持久化到 Redis Stream，WebSocket 断线重连带 last\_event\_id 用 XRANGE 补发；多实例间 Pub/Sub 转发事件与取消指令，解耦任务与实例。

-   **幂等与并发：**针对并发多开、崩溃免单两条 Credits 透支路径，任务开始前预扣，结束按用量结算；MySQL 行锁 + CAS 控制并发上限与重复提交；写工具以 run\_id + tool\_call\_id 作幂等键。同一任务重跑或并发重复提交，只扣一次 Credits、订单只写入一次。

-   **限流与熔断：**模型请求统一经网关，Redis Lua 令牌桶按供应商分 RPM / TPM 桶；断路器统计 5xx 错误与首 token 超时，跳闸后切备用模型；执行池满返回 429 + Retry-After，非核心依赖失败直接跳过。**压测 500 并发成功率 100%、首事件 P95 0.83s，超额请求 P95 0.86s 内收到 429。**

## 专业技能

-   **Agent 与 LLM 工程：** 了解 Agent Loop、Harness 工程、上下文管理、前缀缓存、MCP 协议；有 Rubrics 评测与 LLM-as-Judge 实践经验

-   **检索与模型训练：**了解召回、精排、Hybrid 检索全链路；了解 SFT、PPO、GRPO、DPO

-   **数据库与中间件：**了解 Redis 数据类型、分布式锁、Stream；了解 MySQL 索引、事务、MVCC、锁

-   **Java 基础：**了解 Java 集合、反射、泛型等；了解 JUC 并发编程；了解 JVM 的工作原理