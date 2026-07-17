import { useCallback, useEffect, useState } from "react";
import { fetchAdminConfig, resetAdminConfig, updateAdminConfig } from "../api";
import type { AdminConfig, AdminParam } from "../types";
import { AdminParamField } from "./AdminParamField";
import { BoltIcon, CloseIcon, RefreshIcon } from "./icons";

// 后台管理面板：热更新模型档位与检索 / 展示参数。仅管理员可见（入口由 App 按 checkAdmin 决定）。
//
// **整份表单由后端响应驱动**（groups + params），前端不硬编码任何参数名/范围/默认值——后端
// registry.py 加一项，这里自动多一项，不用改前端。
//
// **只提交改动过的字段**（不是整份快照）：后端存的是「相对基线的差集」，把没动过的参数也一并 PUT
// 会把它们全部固化成 override，之后即使改了代码默认值或 .env 也再不生效——库里那份旧值会一直盖着。
type Props = {
  open: boolean;
  onClose: () => void;
};

type Draft = Record<string, number | string | boolean>;

export function AdminDrawer({ open, onClose }: Props) {
  const [config, setConfig] = useState<AdminConfig | null>(null);
  const [draft, setDraft] = useState<Draft>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");

  const adopt = useCallback((cfg: AdminConfig) => {
    setConfig(cfg);
    setDraft(Object.fromEntries(cfg.params.map((p) => [p.key, p.value])));
  }, []);

  const load = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      adopt(await fetchAdminConfig());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [adopt]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const changed = config?.params.filter((p) => draft[p.key] !== p.value) ?? [];

  const save = async () => {
    if (!changed.length) return;
    setBusy(true);
    setError("");
    try {
      const values = Object.fromEntries(changed.map((p) => [p.key, draft[p.key]]));
      adopt(await updateAdminConfig(values));
      setToast(`已保存 ${changed.length} 项，对新任务生效`);
      setTimeout(() => setToast(""), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const resetAll = async () => {
    // 全量恢复默认会一次性抹掉所有调参结果，且无法撤销——值得拦一下。
    if (!window.confirm("把所有参数恢复成 .env / 代码默认值？当前的后台改动会全部丢失。")) return;
    setBusy(true);
    setError("");
    try {
      adopt(await resetAdminConfig());
      setToast("已全部恢复默认");
      setTimeout(() => setToast(""), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // 单项恢复默认。这是密钥唯一的「清空」途径——留空提交是「不改」而非「设空」（见后端 secret 规则）。
  const resetOne = async (key: string) => {
    setBusy(true);
    setError("");
    try {
      adopt(await resetAdminConfig([key]));
      setToast(`${key} 已恢复默认`);
      setTimeout(() => setToast(""), 3000);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const byGroup = (g: string): AdminParam[] => config?.params.filter((p) => p.group === g) ?? [];

  return (
    <>
      <div className={`drawer-scrim ${open ? "show" : ""}`} onClick={onClose} />
      <aside className={`drawer admin-drawer ${open ? "open" : ""}`} aria-hidden={!open}>
        <div className="drawer-head">
          <div className="drawer-title">
            <BoltIcon width={18} height={18} />
            后台管理
          </div>
          <div className="drawer-tools">
            <button className="icon-btn" onClick={() => void load()} disabled={busy} title="刷新">
              <RefreshIcon width={16} height={16} className={busy ? "spin" : ""} />
            </button>
            <button className="icon-btn" onClick={onClose} title="关闭">
              <CloseIcon width={18} height={18} />
            </button>
          </div>
        </div>

        <div className="drawer-list admin-body">
          <p className="admin-intro">
            改动<strong>只对新任务生效</strong>，正在跑的任务不受影响。参数存库，重启后仍在。
          </p>
          {error && <p className="admin-error">{error}</p>}
          {toast && <p className="admin-toast">{toast}</p>}
          {!config && busy && <p className="admin-empty">加载中…</p>}

          {config &&
            Object.entries(config.groups).map(([key, meta]) => (
              <section key={key} className="admin-group">
                <h3>{meta.label}</h3>
                <p className="admin-group-desc">{meta.desc}</p>
                {byGroup(key).map((p) => (
                  <AdminParamField
                    key={p.key}
                    param={p}
                    draft={draft[p.key] ?? p.value}
                    onChange={(v) => setDraft((d) => ({ ...d, [p.key]: v }))}
                    onReset={() => void resetOne(p.key)}
                  />
                ))}
              </section>
            ))}
        </div>

        <div className="admin-foot">
          <button className="ghost-btn" onClick={() => void resetAll()} disabled={busy || !config}>
            全部恢复默认
          </button>
          <button className="admin-save-btn" onClick={() => void save()} disabled={busy || !changed.length}>
            {changed.length ? `保存 ${changed.length} 项改动` : "无改动"}
          </button>
        </div>
      </aside>
    </>
  );
}
