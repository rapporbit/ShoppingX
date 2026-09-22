// 登录态（M16）：token 的存取、注册 / 登录、带凭证的 fetch。
//
// 在此之前，前端的"身份"是一行 `const DEMO_USER = "demo-user"` —— 所有访客共用同一个人：谁的偏好
// 都写进同一个 Store，谁都能看谁的会话。现在身份来自后端签发的 token，user_id 一律由后端从 token
// 里解出来（前端传什么都不作数）。
//
// token 存 localStorage 的取舍要说清楚：它比 httpOnly Cookie 更容易被页面上的恶意脚本读走（XSS
// 一旦得手就能偷 token）。选它是因为本项目前后端分离、WebSocket 还要把 token 挂 query 上，Cookie
// 方案在这两处都别扭。代价是 token 有过期时间（默认 24h），泄漏窗口有限；真要更稳，得上 httpOnly
// Cookie + CSRF token + WS 一次性 ticket，那是另一摊工程。

const TOKEN_KEY = "shoppingx.token";
const USER_KEY = "shoppingx.user";
// 「上次看的是哪段会话」这个光标（会话数据本身在后端，这里只是个书签）。它归 auth 管，是因为
// **登出必须连它一起清**：否则换个账号登进来，前端还指着上一个人的 thread——拉他的历史被 403、
// 发消息也被 403（那不是你的会话），新用户会卡在一个「一句话都发不出去」的界面里，且毫无头绪。
export const THREAD_KEY = "shoppingx.threadId";

// isGuest：免登录试用身份（后端 POST /api/auth/guest 签发）。它和正式账号唯一的前端差别是顶栏
// 多一条「试用中 · 注册可保留记录」，以及注册时带上当前 token（后端据此原地升级、user_id 不变）。
export type Session = { userId: string; username: string; token: string; isGuest: boolean };

export function loadSession(): Session | null {
  try {
    const token = localStorage.getItem(TOKEN_KEY);
    const raw = localStorage.getItem(USER_KEY);
    if (!token || !raw) return null;
    const { userId, username, isGuest } = JSON.parse(raw);
    if (!userId) return null;
    return { userId, username, token, isGuest: Boolean(isGuest) };
  } catch {
    // 隐私模式禁读 / 值被改坏 → 当作未登录，跳登录页重来即可。
    return null;
  }
}

function saveSession(s: Session): void {
  try {
    localStorage.setItem(TOKEN_KEY, s.token);
    localStorage.setItem(
      USER_KEY,
      JSON.stringify({ userId: s.userId, username: s.username, isGuest: s.isGuest }),
    );
  } catch {
    // 存不下也让本次会话继续（内存里还有），只是刷新后要重新登录。
  }
}

export function clearSession(): void {
  try {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(THREAD_KEY); // 别把上一个人的会话光标留给下一个登录的人
  } catch {
    /* 忽略：清不掉也不影响跳登录页 */
  }
}

export function authHeader(): Record<string, string> {
  const token = loadSession()?.token;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

// WS 不能设自定义请求头（浏览器的 WebSocket API 就是不让），token 只能挂 query 上——后端 ws
// 握手前读它验属主（见 server._ws_authorized）。
export function wsToken(): string | null {
  return loadSession()?.token ?? null;
}

async function submit(
  path: string,
  username: string,
  password: string,
  extraHeaders: Record<string, string> = {},
): Promise<Session> {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...extraHeaders },
    body: JSON.stringify({ username, password }),
  });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.detail ?? "请求失败");
  const session: Session = {
    userId: body.user_id,
    username: body.username,
    token: body.access_token,
    isGuest: Boolean(body.is_guest),
  };
  saveSession(session);
  return session;
}

// 免登录试用：一次点击换一枚访客 token。失败（429 名额 / 限流、404 未开鉴权）原话抛给落地页显示。
export async function startGuest(): Promise<Session> {
  const resp = await fetch("/api/auth/guest", { method: "POST" });
  const body = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(body.detail ?? "试用申请失败");
  const session: Session = {
    userId: body.user_id,
    username: body.username,
    token: body.access_token,
    isGuest: true,
  };
  saveSession(session);
  return session;
}

// 试用中注册 = 升级：带上访客 token，后端把访客那行原地改成正式账号（user_id 不变，会话 / 偏好 /
// 收藏全留）。不是访客时不带 token——正式账号再注册一个新号是另一个人，后端也会 409 拦住带 token 的。
export const register = (username: string, password: string): Promise<Session> =>
  submit(
    "/api/auth/register",
    username,
    password,
    loadSession()?.isGuest ? authHeader() : {},
  );

export const login = (username: string, password: string): Promise<Session> =>
  submit("/api/auth/login", username, password);

// 全站唯一的 fetch 出口：自动带上 token，并把「token 过期 / 被吊销」统一处理掉。
//
// 401 且**本来带着 token** → 说明这枚 token 已经不作数了（过期 / 账号没了）：清掉登录态、刷新页面
// 回到登录页。否则用户会卡在一个「看起来登录着、但每个请求都悄悄失败」的界面里，对着空列表反复点，
// 不知道自己该重新登录。
//
// 反过来，**本来就没 token** 时的 401 绝不能走这条路：那时 clearSession + reload 只会刷新回同一个
// 未登录页面、再发一次请求、再 401 —— 无限刷新循环。没 token 的 401 是「你还没登录」，正常返回给
// 调用方即可（页面本来就该显示登录页）。
export async function authFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const hadToken = Boolean(loadSession()?.token);
  const resp = await fetch(input, {
    ...init,
    headers: { ...(init.headers ?? {}), ...authHeader() },
  });
  if (resp.status === 401 && hadToken) {
    clearSession();
    location.reload();
  }
  return resp;
}
