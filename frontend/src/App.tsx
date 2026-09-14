import { useCallback, useEffect, useMemo, useState } from "react";
import {
  addFavorite,
  fetchFavorites,
  checkAdmin,
  fetchQuota,
  fetchSkillCatalog,
  formatResetAt,
  removeFavorite,
  type Quota,
} from "./api";
import type { ProductItem, SkillCatalogItem, TurnExperiment } from "./types";
import { ActivityFeed } from "./components/ActivityFeed";
import { ClarificationChoices } from "./components/ClarificationChoices";
import { FavoritesDrawer } from "./components/FavoritesDrawer";
import { SkillsDrawer } from "./components/SkillsDrawer";
import { FinalAnswer } from "./components/FinalAnswer";
import { LearnedPrefsBar } from "./components/LearnedPrefsBar";
import { PreferenceDrawer } from "./components/PreferenceDrawer";
import { ConfirmationCards } from "./components/ConfirmationCards";
import { OrdersDrawer } from "./components/OrdersDrawer";
import { ProductCards } from "./components/ProductCards";
import { ProductDetail } from "./components/ProductDetail";
import { ProductComparison } from "./components/ProductComparison";
import { OrderIntentForm } from "./components/OrderIntentForm";
import { QueryImages } from "./components/QueryImages";
import { SimilarDrawer } from "./components/SimilarDrawer";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { AdminDrawer } from "./components/AdminDrawer";
import { Sidebar, type SidebarPanel } from "./components/Sidebar";
import { InputBar } from "./components/InputBar";
import { TopBar } from "./components/TopBar";
import { SparkleIcon } from "./components/icons";
import { Landing } from "./components/Landing";
import { Legal, type LegalPage } from "./components/Legal";
import { Login } from "./components/Login";
import { useShoppingXTask } from "./hooks/useShoppingXTask";
import { clearSession, loadSession, type Session } from "./auth";
import { loadPlatforms, savePlatforms } from "./settings";

// 法务页的路由。只认三个固定路径，不引 react-router——nginx 已经把所有路径 fallback 到
// index.html（docker/nginx-spa.conf），读一次 pathname 就够了，为三个静态页拖进一个路由库不值当。
// 页脚里用的是普通 <a>：整页重载，对静态页完全够用，也省掉一套链接拦截。
const LEGAL_ROUTES: Record<string, LegalPage> = {
  "/about": "about",
  "/privacy": "privacy",
  "/terms": "terms",
};

function readLegalRoute(): LegalPage | null {
  const path = window.location.pathname.replace(/\/+$/, "") || "/";
  return LEGAL_ROUTES[path] ?? null;
}

/**
 * 登录闸（M16）。没有身份就不渲染工作区——不是为了好看，是因为工作区一挂载就会去拉会话清单、
 * 偏好、收藏，全都得带 token。未登录时渲染它，只会打出一串注定 401 的请求。
 *
 * 身份也从此**只有一个来源**：后端签发的 token。此前这里硬编码着一个 "demo-user" 常量
 * ——所有访客共用一个人，谁的偏好都写进同一个 Store，谁都能翻谁的会话。
 *
 * 未登录的默认落点是**落地页**，不是登录框：在开口要密码之前，先告诉人这是什么东西。
 * 点「开始」才翻到 Login（不改 URL——登录不是一个值得被收藏或分享的地址）。
 */
export default function App() {
  const [session, setSession] = useState<Session | null>(loadSession);
  const [legal, setLegal] = useState<LegalPage | null>(readLegalRoute);
  const [wantLogin, setWantLogin] = useState(false);

  // 浏览器前进/后退：法务页是真实 URL，用户按回退键就得回得去。
  useEffect(() => {
    const onPop = () => setLegal(readLegalRoute());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const goHome = () => {
    window.history.pushState({}, "", "/");
    setLegal(null);
  };

  // 法务页对登录状态无所谓：登录用户点页脚的「隐私政策」也该看得到。
  if (legal) return <Legal page={legal} onHome={goHome} />;

  if (!session) {
    return wantLogin ? (
      <Login onDone={setSession} onBack={() => setWantLogin(false)} />
    ) : (
      <Landing onStart={() => setWantLogin(true)} />
    );
  }

  return (
    <Workspace
      session={session}
      onLogout={() => {
        clearSession();
        setSession(null);
        setWantLogin(false); // 退出后回落地页，而不是甩一个空登录框在脸上
      }}
    />
  );
}

// 把毫秒耗时格式化成人读小字：<1 分用「X.X 秒」，≥1 分用「M 分 S 秒」。
function formatElapsed(ms: number): string {
  const totalSec = ms / 1000;
  if (totalSec < 60) return `用时 ${totalSec.toFixed(1)} 秒`;
  const min = Math.floor(totalSec / 60);
  const sec = Math.round(totalSec % 60);
  return `用时 ${min} 分 ${sec} 秒`;
}

// 把 token 数压成人读小字：<1k 原样，<1M 用「X.Xk」，否则用「X.XXM」。
function formatTokens(n: number): string {
  if (n < 1000) return String(n);
  if (n < 1_000_000) return `${(n / 1000).toFixed(1)}k`;
  return `${(n / 1_000_000).toFixed(2)}M`;
}

// 本轮「实验与自进化」chip 行：提示词版本（实验组高亮）/ 注入的策略 / 读过的 skill。
// 对照组且没策略没 skill 时整行不画——绝大多数轮次就是这样，别让每条回复都挂一排灰字。
function ExperimentChips({ exp }: { exp: TurnExperiment }) {
  const hasStrategies = exp.strategies.length > 0;
  const hasSkills = exp.skills.length > 0;
  if (!exp.in_experiment && !hasStrategies && !hasSkills) return null;
  return (
    <div className="turn-experiment" aria-label="本轮实验与自进化归属">
      <span
        className={`exp-chip${exp.in_experiment ? " exp-chip-live" : ""}`}
        title={
          exp.ab_bucket < 0
            ? "匿名，不参与提示词 A/B"
            : `提示词 A/B 桶 ${exp.ab_bucket}${exp.in_experiment ? "（实验组）" : "（对照组）"}`
        }
      >
        prompt {exp.prompt_version}
      </span>
      {exp.strategies.map((k) => (
        <span key={`s-${k}`} className="exp-chip exp-chip-strategy" title="本轮注入的成功策略（system prompt 末尾）">
          策略 {k.split(":").pop()}
        </span>
      ))}
      {exp.skills.map((k) => (
        <span key={`k-${k}`} className="exp-chip exp-chip-skill" title="本轮按 description 读取的 skill 正文">
          skill {k}
        </span>
      ))}
    </div>
  );
}

// 空态的示例意图：点一下即作为 query 发起，降低首屏「不知道说什么」的门槛。
const SAMPLES = [
  "通勤背包，能装 16 寸笔记本，防泼水，预算 400 以内",
  "降噪耳机，通勤用，预算 800，佩戴舒适优先",
  "大容量家居收纳箱，租房搬家不心疼，结实耐用",
  "健身房要用的一套：背包 + 蓝牙耳机 + 运动鞋，预算 1000 内",
];

function Workspace({ session, onLogout }: { session: Session; onLogout: () => void }) {
  const userId = session.userId; // 唯一身份来源：后端 token 里的 sub
  const {
    threadId,
    turns,
    status,
    running,
    waiting,
    sessions,
    sessionConstraints,
    setSessionConstraints,
    confirmations,
    confirmationBusy,
    confirmationError,
    refreshConfirmations,
    prepareOrder,
    prepareCancel,
    resolveConfirmation,
    startTask,
    cancelTask,
    sendClarification,
    newConversation,
    selectConversation,
    deleteConversation,
  } = useShoppingXTask();

  const [prefsOpen, setPrefsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [adminOpen, setAdminOpen] = useState(false);
  // 是不是管理员由后端说了算（GET /api/admin/whoami，403 即否）——前端不自己判断用户名，
  // 那等于把白名单抄一份到浏览器里，既会漂移又毫无安全意义。登录后探一次，用于决定是否摆入口。
  const [isAdmin, setIsAdmin] = useState(false);
  // 窄屏下会话栏是抽屉（宽屏它常驻，这个 state 无人过问）。切换/新建会话后要自己收起来——
  // 手机上抽屉盖着大半个屏幕，点完不关，用户就看不到自己刚打开的那段对话。
  const [navOpen, setNavOpen] = useState(false);
  const [favsOpen, setFavsOpen] = useState(false);
  const [ordersOpen, setOrdersOpen] = useState(false);
  // Skill 目录（内置 + 我的）喂输入框 / 菜单；抽屉增删改后 bump refresh 重拉。
  const [skillsOpen, setSkillsOpen] = useState(false);
  const [skillCatalog, setSkillCatalog] = useState<SkillCatalogItem[]>([]);
  const [skillRefresh, setSkillRefresh] = useState(0);
  useEffect(() => {
    void fetchSkillCatalog(userId).then(setSkillCatalog);
  }, [userId, skillRefresh]);
  // 「搜同款」抽屉：存的是**源商品**（点了哪张卡），非 null 即打开——相似结果由抽屉自己现拉。
  const [similarOf, setSimilarOf] = useState<ProductItem | null>(null);

  // 商品详情 / 对比 / 下单意向：都是针对「眼前这几件」的动作，状态只活在本页，不落盘。
  const [detailOf, setDetailOf] = useState<ProductItem | null>(null);
  const [orderIntentOf, setOrderIntentOf] = useState<ProductItem | null>(null);
  const [compareList, setCompareList] = useState<ProductItem[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);
  const compareIds = useMemo(() => new Set(compareList.map((c) => c.item_id)), [compareList]);
  const COMPARE_MAX = 4;
  // 换会话（新建 / 切换）就清掉这几样：它们绑的是「眼前这几件」，item_id 只在本会话的候选
  // 登记表里有效——带到别的会话去对比 / 下单，后端根本找不到那些 id。
  useEffect(() => {
    setDetailOf(null);
    setOrderIntentOf(null);
    setCompareList([]);
    setCompareOpen(false);
    setSimilarOf(null);
  }, [threadId]);
  // 传给商品卡的回调用 useCallback 固定引用：Card 是 memo 的，收尾流式逐字推时父组件每段都重渲，
  // 回调引用一变 memo 就白包。
  const toggleCompare = useCallback((item: ProductItem) => {
    setCompareList((cur) =>
      cur.some((c) => c.item_id === item.item_id)
        ? cur.filter((c) => c.item_id !== item.item_id)
        : cur.length >= COMPARE_MAX
          ? cur
          : [...cur, item],
    );
  }, []);
  // 详情 / 对比 / 表单要「说一句话」时都走这条：跟输入框发消息完全同一条路。
  const say = (text: string) => startTask(text, userId);

  // 收藏（♡）：用户级、跨会话、以回看为主 —— 不进 prompt、不进长期偏好库，但会经 app.memory.affinity
  // 聚合成弱信号，在 item_picker 精挑里给同类属性小幅加分（只上浮不淘汰）。开局拉一次就够（抽屉打开时自己会再刷）。
  const [favorites, setFavorites] = useState<ProductItem[]>([]);
  const favoriteIds = useMemo(() => new Set(favorites.map((f) => f.item_id)), [favorites]);

  useEffect(() => {
    void fetchFavorites(userId).then(setFavorites);
  }, []);

  // 乐观更新：先改 UI 再发请求。点 ♡ 要立刻见到实心，等一个 round-trip 会显得迟钝。
  const handleFavorite = useCallback(
    async (item: ProductItem, undo: boolean) => {
      setFavorites((cur) =>
        undo ? cur.filter((f) => f.item_id !== item.item_id) : [item, ...cur],
      );
      if (undo) await removeFavorite(userId, item.item_id);
      else await addFavorite(userId, item);
    },
    [userId],
  );

  // 启用平台：真源在 localStorage（api.startTaskRequest 发任务时读它），这里持一份镜像供顶栏显示
  // 启用个数、抽屉做勾选。全取消会被 savePlatforms 兜回默认（amazon），故回填它的返回值。
  const [platforms, setPlatforms] = useState<string[]>(loadPlatforms);
  const togglePlatform = (id: string) => {
    setPlatforms((cur) =>
      savePlatforms(cur.includes(id) ? cur.filter((p) => p !== id) : [...cur, id]),
    );
  };

  // 今日 credit 余额（M18）。开局拉一次；此后**每次任务落地就重拉**——不只是 done：取消和出错
  // 同样已经烧掉了 token（后端在 finally 里照记不误），余额条要跟着掉，否则用户会以为「取消了就
  // 不扣」。额度耗尽（402）也走 error 这条路，重拉后 exhausted=true，输入框随即锁上。
  const [quota, setQuota] = useState<Quota | null>(null);
  useEffect(() => {
    void fetchQuota().then(setQuota);
  }, []);

  // 管理员身份由后端说了算（403 即否），换账号必须重探——否则上一个管理员的入口会留在页面上。
  // 依赖 userId 而非 session 对象：只有「换人」才需要重判，其余字段变了不该触发多余请求。
  useEffect(() => {
    void checkAdmin().then(setIsAdmin);
  }, [userId]);
  useEffect(() => {
    if (status === "done" || status === "cancelled" || status === "error") {
      void fetchQuota().then(setQuota);
    }
  }, [status]);
  const quotaExhausted = quota?.enabled === true && quota.exhausted;

  // 偏好可能在任务收尾被写回，故每次任务真正完成（done）就 +1 触发抽屉重拉。
  // 用单调计数而非 turns.length 当代理——多轮下 length 会变但偏好不一定每轮都更，计数更稳。
  const [prefRefresh, setPrefRefresh] = useState(0);
  useEffect(() => {
    if (status === "done") setPrefRefresh((n) => n + 1);
  }, [status]);

  const hasConversation = turns.length > 0;
  // 会话标题取第一轮 query（整段对话的「主题」），多轮续聊也不跳来跳去。
  const title = turns[0]?.query || "ShoppingX 跨境购物 Agent";

  const activePanel: SidebarPanel | null = favsOpen
    ? "favorites"
    : ordersOpen
      ? "orders"
      : skillsOpen
        ? "skills"
        : prefsOpen
          ? "preferences"
          : null;

  return (
    <div className="app">
      <Sidebar
        sessions={sessions}
        activeThreadId={threadId}
        open={navOpen}
        onNewChat={() => {
          newConversation();
          setNavOpen(false);
        }}
        onSelectConversation={(id) => {
          selectConversation(id);
          setNavOpen(false);
        }}
        onDeleteConversation={deleteConversation}
        onOpenPanel={(panel) => {
          // 四个面板互斥：开一个就把别的关掉，免得右侧叠两层抽屉。
          setFavsOpen(panel === "favorites");
          setOrdersOpen(panel === "orders");
          setSkillsOpen(panel === "skills");
          setPrefsOpen(panel === "preferences");
          setNavOpen(false);
        }}
        activePanel={activePanel}
        favoriteCount={favorites.length}
      />
      {/* 会话栏抽屉的遮罩：只在窄屏 + 抽屉展开时可点（CSS 里宽屏直接 display:none）。 */}
      <div
        className={`nav-scrim ${navOpen ? "show" : ""}`}
        onClick={() => setNavOpen(false)}
      />

      <div className={`workspace ${activePanel ? "paged" : ""}`}>
        <TopBar
          title={title}
          status={status}
          username={session.username}
          platformCount={platforms.length}
          quota={quota}
          onOpenSettings={() => setSettingsOpen(true)}
          onOpenAdmin={isAdmin ? () => setAdminOpen(true) : null}
          onLogout={onLogout}
          onOpenNav={() => setNavOpen(true)}
        />

        {/* 「我的东西」四个面板：收藏 / 订单 / Skill / 长期偏好。它们不再是右侧 360px 抽屉，而是
            占满对话区的整页视图——侧栏点项切页，关闭即回对话。对话 DOM 只是被 CSS 藏起来，
            不卸载，切回来滚动位置和输入草稿都还在。 */}
        <div className={`panel-host ${activePanel ? "page" : ""}`}>
          <PreferenceDrawer
            userId={userId}
            open={prefsOpen}
            refreshKey={prefRefresh}
            onClose={() => setPrefsOpen(false)}
            threadId={threadId}
            session={sessionConstraints}
            onSessionChange={setSessionConstraints}
          />

          <FavoritesDrawer
            userId={userId}
            open={favsOpen}
            refreshKey={favorites.length}
            onClose={() => setFavsOpen(false)}
            onChanged={setFavorites}
          />

          <OrdersDrawer
            open={ordersOpen}
            canCancel={threadId !== null}
            busy={confirmationBusy}
            onClose={() => setOrdersOpen(false)}
            onCancel={prepareCancel}
          />

          <SkillsDrawer
            userId={userId}
            open={skillsOpen}
            onClose={() => setSkillsOpen(false)}
            onChanged={() => setSkillRefresh((n) => n + 1)}
          />
        </div>

        <main className="conversation">
          <div className="conversation-inner">
            {!hasConversation ? (
              <section className="welcome">
                <div className="welcome-mark">
                  <SparkleIcon width={26} height={26} />
                </div>
                <h1>今天想淘点什么？</h1>
                <p>
                  用一句话说清预算、品类和偏好，ShoppingX 会跨平台并行检索、比价、算到手价，给你一份带选购理由的清单。
                </p>
                <div className="welcome-samples">
                  {/* 额度耗尽时一并禁掉示例：输入框已经锁了，还留着能点的入口，点下去只会打一次
                      注定 402 的请求，再把这一轮标红——用户白挨一个错误。 */}
                  {SAMPLES.map((s) => (
                    <button
                      key={s}
                      className="sample-chip"
                      disabled={quotaExhausted}
                      onClick={() => startTask(s, userId)}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </section>
            ) : (
              <section className="thread">
                {turns.map((turn, idx) => {
                  const turnRunning = turn.status === "connecting" || turn.status === "running";
                  // 产物文件（summary.md / result.json）按 thread 覆盖式写，只有最新一轮的在盘上是当前的，
                  // 故下载入口只挂最后一轮，避免历史轮点出来的是新一轮的产物（详见后端 history 设计取舍）。
                  const isLast = idx === turns.length - 1;
                  return (
                    <div key={turn.id} className="turn">
                      <div className="msg-row user">
                        <div className="user-msg">
                          {/* 参考图在气泡上方：它是这句话的宾语（「找这个同款」里的「这个」），
                              读的顺序应当是先看到图、再看到那句话。 */}
                          {turn.images.length > 0 && (
                            <QueryImages threadId={threadId} images={turn.images} />
                          )}
                          <div className="bubble user-bubble">{turn.query}</div>
                        </div>
                      </div>

                      <div className="msg-row assistant">
                        <div className="assistant-block">
                          {(turn.events.length > 0 || turnRunning) && (
                            <ActivityFeed events={turn.events} running={turnRunning} />
                          )}

                          {/* ask_user 的问题：渲染成一条普通 assistant 消息（不是横幅、不是另一种卡）；
                              带 options 时选项挂在气泡下方。只对最新一轮且正等待回复时可交互。 */}
                          {isLast && turn.status === "waiting" && turn.clarificationQuestion !== null && (
                            <ClarificationChoices
                              question={turn.clarificationQuestion}
                              options={turn.clarificationOptions ?? []}
                              multiSelect={turn.clarificationMultiSelect ?? false}
                              preselected={turn.clarificationPreselected}
                              onSubmit={sendClarification}
                            />
                          )}

                          {turn.errorMsg && (
                            <div className="error-banner">
                              <span>⚠</span>
                              {turn.errorMsg}
                            </div>
                          )}

                          {turn.status === "cancelled" && !turn.errorMsg && (
                            <div className="info-banner">任务已取消。</div>
                          )}

                          {turn.finalAnswer && (
                            <FinalAnswer
                              markdown={turn.finalAnswer}
                              threadId={isLast ? threadId : null}
                            />
                          )}

                          {/* 收尾文案流式预览：summary 还在生成时逐字先看（感知延迟优化）。
                              定稿（task_result）一到 streamingText 即清空，由上面的 finalAnswer 接管。 */}
                          {!turn.finalAnswer && turn.streamingText && (
                            <FinalAnswer markdown={turn.streamingText} threadId={null} />
                          )}

                          {/* 「记住了 …」：curator 后台学到的新长期偏好，自动写入但看得见、撤得掉。
                              不弹确认框——收尾这一刻用户在想「买不买这件」，此时弹窗只会被无脑点掉。 */}
                          {turn.learnedPrefs.length > 0 && (
                            <LearnedPrefsBar
                              userId={userId}
                              prefs={turn.learnedPrefs}
                              onForget={() => setPrefRefresh((n) => n + 1)}
                            />
                          )}

                          {/* 商品卡「先出货、后出文案」：item_picker 一定稿就经 items_preview 推上来
                              （卡片的每个字段那一刻都已确定），用户不必再等收尾那轮解码 + 文案生成。
                              收尾的 task_result 用定稿那批原样覆盖；文案由上面的 streamingText 逐字补。 */}
                          {turn.items.length > 0 && (
                            <ProductCards
                              items={turn.items}
                              favorited={favoriteIds}
                              onFavorite={handleFavorite}
                              onSimilar={setSimilarOf}
                              compared={compareIds}
                              onDetail={setDetailOf}
                              onCompare={toggleCompare}
                            />
                          )}

                          {/* 本轮结束后在右下角用小字标注用时 + token 消耗（后端权威口径，实时与回看一致）。
                              token 总量主显，hover 看输入/输出/成本拆分（全树记账，含 fork 子 Agent）。 */}
                          {/* 实验与自进化归属：提示词版本 / 注入策略 / 读过的 skill（后端随 task_result 下发，回看同源）。 */}
                          {turn.experiment && <ExperimentChips exp={turn.experiment} />}

                          {(turn.elapsedMs != null || turn.tokens != null) && (
                            <div className="turn-elapsed">
                              {turn.elapsedMs != null && formatElapsed(turn.elapsedMs)}
                              {turn.elapsedMs != null && turn.tokens != null && " · "}
                              {turn.tokens != null && (
                                <span
                                  title={`输入 ${turn.tokens.input.toLocaleString()} · 输出 ${turn.tokens.output.toLocaleString()}${
                                    turn.tokens.cost_usd ? ` · $${turn.tokens.cost_usd.toFixed(4)}` : ""
                                  }${
                                    turn.tokens.cache_hit_rate != null
                                      ? ` · 缓存命中 ${(turn.tokens.cache_hit_rate * 100).toFixed(0)}%`
                                      : ""
                                  }`}
                                >
                                  {formatTokens(turn.tokens.total)} tokens
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </section>
            )}
            {/* 交易确认卡（会话级）：一张卡从出现到决议可能跨好几轮，故不挂在某一轮下面，而是
                跟在整段对话之后。决议只走 HTTP（点按钮），对话里说「确认」不算数。 */}
            {turns.length > 0 && (
              <ConfirmationCards
                confirmations={confirmations}
                busy={confirmationBusy}
                error={confirmationError}
                onResolve={(c, approved) => void resolveConfirmation(c, approved)}
                onCancelOrder={(id, reason) => void prepareCancel(id, reason)}
                onRefresh={() => void refreshConfirmations()}
              />
            )}
          </div>
        </main>

        {compareList.length > 0 && (
          <div className="compare-bar">
            <span>
              已选 <strong>{compareList.length}</strong> / {COMPARE_MAX} 件对比
            </span>
            <button className="btn-ghost" onClick={() => setCompareList([])}>
              清空
            </button>
            <button className="btn-primary" onClick={() => setCompareOpen(true)}>
              对比
            </button>
          </div>
        )}

        <InputBar
          running={running}
          waiting={waiting}
          blockedReason={
            quotaExhausted && quota
              ? `今日 credit 已用完，${formatResetAt(quota.reset_at)} 重置后可继续。`
              : null
          }
          // 带 options 的澄清由消息下方的选项接管——此时收起底部输入框，免得两处都能作答。
          clarificationHasChoices={
            (turns[turns.length - 1]?.clarificationOptions?.length ?? 0) > 0
          }
          skills={skillCatalog}
          onSend={(text, files, skill) => startTask(text, userId, files, skill)}
          onCancel={cancelTask}
          onClarify={sendClarification}
        />
      </div>

      <SimilarDrawer source={similarOf} onClose={() => setSimilarOf(null)} />

      <ProductDetail
        item={detailOf}
        favorited={detailOf ? favoriteIds.has(detailOf.item_id) : false}
        compared={detailOf ? compareIds.has(detailOf.item_id) : false}
        busy={running || waiting}
        onClose={() => setDetailOf(null)}
        onFavorite={(it, undo) => void handleFavorite(it, undo)}
        onSimilar={(it) => {
          setDetailOf(null);
          setSimilarOf(it);
        }}
        onCompare={toggleCompare}
        onOrder={(it) => {
          setDetailOf(null);
          setOrderIntentOf(it);
        }}
      />

      <ProductComparison
        open={compareOpen}
        items={compareList}
        busy={running || waiting}
        onClose={() => setCompareOpen(false)}
        onRemove={toggleCompare}
        onClear={() => setCompareList([])}
        onAsk={say}
      />

      <OrderIntentForm
        item={orderIntentOf}
        busy={confirmationBusy}
        error={confirmationError}
        onClose={() => setOrderIntentOf(null)}
        onPrepare={prepareOrder}
      />

      <AdminDrawer open={adminOpen} onClose={() => setAdminOpen(false)} />

      <SettingsDrawer
        open={settingsOpen}
        platforms={platforms}
        onToggle={togglePlatform}
        onClose={() => setSettingsOpen(false)}
      />
    </div>
  );
}
