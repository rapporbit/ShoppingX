// 登录 / 注册页（M16）。未登录时它是整个应用的全部——没有身份就没有会话、没有偏好。
import { useState } from "react";

import { login, register, type Session } from "../auth";

const MIN_PASSWORD = 8; // 与后端 MIN_PASSWORD_LEN 对齐：前端先拦一道，省一次注定失败的往返

// onBack：从落地页翻进来的，就得能翻回去——不给退路的登录框是死胡同。
export function Login({
  onDone,
  onBack,
}: {
  onDone: (s: Session) => void;
  onBack?: () => void;
}) {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (username.trim().length < 3) return setError("用户名至少 3 个字符");
    if (password.length < MIN_PASSWORD) return setError(`密码至少 ${MIN_PASSWORD} 个字符`);

    setBusy(true);
    try {
      const fn = isRegister ? register : login;
      onDone(await fn(username.trim(), password));
    } catch (err) {
      // 后端对「密码错」与「查无此人」回同一句话（防用户名探测），这里如实透传，不自作聪明地细分。
      setError(err instanceof Error ? err.message : "登录失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-page">
      <form className="login-card" onSubmit={submit}>
        <h1 className="login-title">ShoppingX</h1>
        <p className="login-sub">
          {isRegister ? "创建账号，你的偏好与会话会跟着账号走" : "登录以继续你的购物会话"}
        </p>

        <input
          className="login-input"
          placeholder="用户名"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          autoFocus
        />
        <input
          className="login-input"
          type="password"
          placeholder={`密码（至少 ${MIN_PASSWORD} 位）`}
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          // 告诉密码管理器这是新密码还是老密码，否则浏览器会把新注册的密码往「当前密码」里填。
          autoComplete={isRegister ? "new-password" : "current-password"}
        />

        {error && <div className="login-error">{error}</div>}

        <button className="login-submit" type="submit" disabled={busy}>
          {busy ? "请稍候…" : isRegister ? "注册并进入" : "登录"}
        </button>

        <button
          type="button"
          className="login-switch"
          onClick={() => {
            setIsRegister(!isRegister);
            setError("");
          }}
        >
          {isRegister ? "已有账号？去登录" : "还没有账号？去注册"}
        </button>

        {onBack && (
          <button type="button" className="login-back" onClick={onBack}>
            ← 返回首页
          </button>
        )}

        <p className="login-legal">
          注册即表示同意 <a href="/terms">服务条款</a> 与 <a href="/privacy">隐私政策</a>
        </p>
      </form>
    </div>
  );
}
