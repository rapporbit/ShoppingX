import type { AdminParam } from "../types";

// 后台管理面板里的**一个**参数控件。按 kind 渲染不同输入：bool→开关、str→文本框、数值→滑块+数字框。
//
// 数值为什么滑块和数字框都给：滑块用来「感受量级」（权重 0.2 和 2.0 差一个数量级，拖一下就知道
// 自己在改什么），数字框用来精确落值（0.35 这种拖不准）。两者双向绑定同一个 value。
//
// **warning 不折叠、不藏进 tooltip**：这些文案是「此值被真实数据标定证伪」「调大会把弱证据变成
// 主导排序因子」级别的信息，藏起来等于没写。宁可页面长一点。
type Props = {
  param: AdminParam;
  // 用户当前编辑中的值（未保存）。与 param.value（服务端生效值）分开，才能标出「有未保存改动」。
  draft: number | string | boolean;
  onChange: (value: number | string | boolean) => void;
  // 单项恢复默认。密钥要「清空」只能走这条路（留空提交是「不改」，不是「设空」）。
  onReset: () => void;
};

export function AdminParamField({ param, draft, onChange, onReset }: Props) {
  const dirty = draft !== param.value;
  // int 用 1 步进；float 的步进按范围定：0~1 的阈值类要 0.01 才调得动，0~10 的权重类 0.1 够用。
  const step = param.kind === "int" ? 1 : (param.maximum ?? 1) <= 1 ? 0.01 : 0.1;

  return (
    <div className={`admin-field ${dirty ? "dirty" : ""}`}>
      <div className="admin-field-head">
        <label htmlFor={param.key}>
          {param.label}
          {/* 只在被后台改过时标；env 配的算「部署方的基线」，不算异常状态，不标。 */}
          {param.source === "override" && <span className="admin-badge">已改</span>}
          {dirty && <span className="admin-badge dirty-badge">未保存</span>}
        </label>
        <span className="admin-field-right">
          <code className="admin-key">{param.key}</code>
          {/* 只在真被后台改过时给：env/default 态点它没有任何意义（本来就是基线值）。 */}
          {param.source === "override" && (
            <button className="admin-reset-one" onClick={onReset} title="这一项恢复默认">
              恢复
            </button>
          )}
        </span>
      </div>

      {param.secret ? (
        // 密码框 + 不回显。placeholder 承担全部信息量：配没配、是哪一把、留空什么也不会发生。
        // autoComplete=new-password 防浏览器把用户自己的登录密码填进 API key 框。
        <input
          id={param.key}
          className="admin-text"
          type="password"
          autoComplete="new-password"
          value={String(draft)}
          placeholder={param.masked ? `当前 ${param.masked}，留空则不改` : "未配置"}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : param.kind === "bool" ? (
        <label className="admin-switch">
          <input
            id={param.key}
            type="checkbox"
            checked={Boolean(draft)}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span>{draft ? "开" : "关"}</span>
        </label>
      ) : param.kind === "str" ? (
        <input
          id={param.key}
          className="admin-text"
          type="text"
          value={String(draft)}
          placeholder={param.allow_empty ? "留空 = 用默认" : "必填"}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <div className="admin-number">
          <input
            type="range"
            min={param.minimum ?? 0}
            max={param.maximum ?? 100}
            step={step}
            value={Number(draft)}
            onChange={(e) => onChange(Number(e.target.value))}
          />
          <input
            id={param.key}
            className="admin-num-box"
            type="number"
            min={param.minimum ?? undefined}
            max={param.maximum ?? undefined}
            step={step}
            value={String(draft)}
            // 允许中途出现空串/半截数字（用户正在打字），不在这里强转数字否则删不掉最后一位。
            onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))}
          />
        </div>
      )}

      <p className="admin-help">{param.help}</p>
      {param.warning && <p className="admin-warning">⚠️ {param.warning}</p>}
      <p className="admin-meta">
        {/* 密钥没有「默认值」可言（代码里它是必配项），显示「默认（空）」只会让人以为空是个合法状态。 */}
        {!param.secret && (
          <>
            默认 <code>{String(param.default) || "（空）"}</code>
            {" · "}
          </>
        )}
        {param.minimum !== null && param.maximum !== null && (
          <>
            范围 {param.minimum} ~ {param.maximum}
            {" · "}
          </>
        )}
        来源 {param.source === "override" ? "后台" : param.source === "env" ? ".env" : "代码默认"}
      </p>
    </div>
  );
}
