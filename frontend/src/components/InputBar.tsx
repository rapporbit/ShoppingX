import { useEffect, useMemo, useRef, useState } from "react";
import type { SkillCatalogItem } from "../types";
import { ArrowUp, ImageIcon, StopIcon } from "./icons";

// 只有**独立词首**的 / 才算命令（行首或空格后）：https:// 、路径中间、「3/4」这类比例写法保持普通文字。
// 只在光标停在命令末尾时弹菜单——命令必须是文本末尾这一段。
const SLASH_RE = /(?:^|\s)\/([\p{L}\p{N}_./-]*)$/u;

// 一次最多带几张参考图：与后端预跑上限（MAX_PREFILL_IMAGES）对齐——多传的图后端也不会看，
// 与其让用户白传，不如在这里就挡住并说清楚。
const MAX_IMAGES = 3;

type InputBarProps = {
  running: boolean;
  waiting?: boolean;
  clarificationQuestion?: string | null;
  // 该轮澄清带了可点选卡片：由展示区的卡片接管作答，这里收起 composer，只留一行提示。
  clarificationHasChoices?: boolean;
  // 非 null 即「不能再发**新任务**了」（目前唯一来源：今日 credit 用尽）。文案直接展示给用户。
  blockedReason?: string | null;
  // 输入框 / 菜单的候选（内置 + 我的 skill 目录）。空表 = 不弹菜单。
  skills?: SkillCatalogItem[];
  onSend: (text: string, files?: File[], skill?: string) => void;
  onCancel: () => void;
  onClarify?: (text: string) => void;
};

export function InputBar({
  running,
  waiting,
  clarificationQuestion,
  clarificationHasChoices,
  blockedReason,
  skills = [],
  onSend,
  onCancel,
  onClarify,
}: InputBarProps) {
  const [text, setText] = useState("");
  // / 选中的 skill：以 chip 挂在输入框上方随本轮发出；发送 / 点 × 即清。
  const [skill, setSkill] = useState<SkillCatalogItem | null>(null);
  const [menuIndex, setMenuIndex] = useState(0);
  const [images, setImages] = useState<File[]>([]);
  const [imgError, setImgError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // **额度用尽不挡澄清回复**（waiting 时放行）：那一轮任务早在入口就放过行了、成本上限也已压好，
  // 它现在正停在这里等用户答一句话。此时把输入框锁死，任务只能一路等到超时——用户既回不了话，
  // 也白等一场。要挡的只是「再起一个新任务」。
  const blocked = !!blockedReason && !waiting;
  // 有图时允许空文字发送：「就照这张图找」是完整的意图，不该逼用户再补一句废话。
  const canSend = (text.trim().length > 0 || images.length > 0) && !running && !blocked;

  // / 菜单：文本末尾是 /xxx 且不在澄清轮时弹出，按 name/description 子串过滤。
  const slash = !waiting && !running ? text.match(SLASH_RE) : null;
  const slashQuery = slash ? slash[1].toLowerCase() : null;
  const menu = useMemo(() => {
    if (slashQuery === null) return [];
    return skills.filter(
      (s) => s.name.toLowerCase().includes(slashQuery) || s.description.toLowerCase().includes(slashQuery),
    );
  }, [skills, slashQuery]);
  const menuOpen = menu.length > 0;
  const activeIndex = Math.min(menuIndex, Math.max(0, menu.length - 1));

  // 选中：把 /xxx 这段从文本里摘掉，chip 顶上去。选内置 skill 也走同一条路（服务端按 name 找）。
  const pickSkill = (item: SkillCatalogItem) => {
    setSkill(item);
    setText((t) => t.replace(SLASH_RE, (m) => (m.startsWith("/") ? "" : m[0])).trimEnd());
    setMenuIndex(0);
  };

  // 预览用的 object URL：**只在 images 变化时**重建。裸在渲染体里 map 会让每敲一个字符都
  // create 一批新 URL、revoke 一批旧 URL，并逼 <img> 重新解码一遍。卸载时 revoke，不然内存不回收。
  const previews = useMemo(() => images.map((f) => URL.createObjectURL(f)), [images]);
  useEffect(() => {
    return () => previews.forEach(URL.revokeObjectURL);
  }, [previews]);

  const addImages = (picked: File[]) => {
    if (!picked.length) return;
    const room = MAX_IMAGES - images.length;
    if (room <= 0) {
      setImgError(`最多带 ${MAX_IMAGES} 张图`);
      return;
    }
    setImgError(null);
    setImages((prev) => [...prev, ...picked.slice(0, room)]);
  };

  // 只收图片：剪贴板 / 文件选择器里混进来的非图片，传到后端也是 415，整轮直接失败。
  const onlyImages = (list: FileList | null) =>
    Array.from(list ?? []).filter((f) => f.type.startsWith("image/"));

  const submit = () => {
    if (!canSend) return;
    if (waiting && onClarify) {
      onClarify(text);
    } else {
      // 澄清轮不带图（那是在回答 Agent 的提问，不是发起新检索）。
      onSend(text, images.length ? images : undefined, skill?.name);
    }
    setText("");
    setImages([]);
    setImgError(null);
    setSkill(null);
  };

  const placeholder = blocked
    ? "今日额度已用完"
    : waiting
      ? "回复 Agent 的问题..."
      : "说说你想买什么，越具体越好（预算、用途、不要什么）";
  const boxClass = `composer-box ${running ? "is-running" : ""} ${waiting ? "is-waiting" : ""} ${
    blocked ? "is-blocked" : ""
  } ${dragging ? "is-dragging" : ""}`;

  // 拖拽落图：澄清轮 / 运行中 / 额度用尽都不收（与附图按钮的可用条件一致）。
  const canAttach = !waiting && !running && !blocked;

  // 澄清带可点选卡片时，作答入口在展示区那张卡片上——这里只留一行指引，收起输入框，防止双入口作答。
  if (waiting && clarificationHasChoices) {
    return (
      <div className="composer">
        <p className="composer-hint">请在上方卡片中点选作答 ↑</p>
      </div>
    );
  }

  return (
    <div className="composer">
      {blocked && (
        <div className="quota-banner">
          <span className="quota-banner-icon">!</span>
          <span>{blockedReason}</span>
        </div>
      )}
      {waiting && clarificationQuestion && (
        <div className="clarification-banner">
          <span className="clarification-banner-icon">?</span>
          <span>{clarificationQuestion}</span>
        </div>
      )}
      <div
        className={boxClass}
        onDragOver={(e) => {
          if (!canAttach) return;
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={(e) => {
          // 只在真正离开整个输入框时收起提示——掠过内部子元素也会冒泡 dragleave。
          if (e.currentTarget.contains(e.relatedTarget as Node | null)) return;
          setDragging(false);
        }}
        onDrop={(e) => {
          if (!canAttach) return;
          e.preventDefault();
          setDragging(false);
          addImages(onlyImages(e.dataTransfer.files));
        }}
      >
        {dragging && (
          <div className="composer-drop-hint">
            <ImageIcon width={18} height={18} />
            <span>松手，用这张图找同款</span>
          </div>
        )}
        {images.length > 0 && (
          <div className="composer-images">
            {images.map((f, i) => (
              <div className="composer-image" key={`${f.name}-${i}`}>
                <img src={previews[i]} alt={f.name} />
                <button
                  className="composer-image-remove"
                  onClick={() => setImages((prev) => prev.filter((_, j) => j !== i))}
                  title="移除"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        )}
        {imgError && <div className="composer-image-error">{imgError}</div>}
        {skill && (
          <div className="composer-skill-chip" title={skill.description}>
            <span className="skill-glyph">/</span>
            <span className="composer-skill-name">{skill.name}</span>
            <button className="composer-skill-remove" onClick={() => setSkill(null)} title="取消选用">
              ×
            </button>
          </div>
        )}
        {menuOpen && (
          <ul className="slash-menu" role="listbox">
            {menu.map((item, i) => (
              <li
                key={item.name}
                role="option"
                aria-selected={i === activeIndex}
                className={`slash-item ${i === activeIndex ? "active" : ""}`}
                onMouseDown={(e) => {
                  e.preventDefault(); // 别让 textarea 失焦
                  pickSkill(item);
                }}
                onMouseEnter={() => setMenuIndex(i)}
              >
                <span className="slash-name">
                  {item.name}
                  {item.source === "user" && <em>我的</em>}
                </span>
                <span className="slash-desc">{item.description}</span>
              </li>
            ))}
          </ul>
        )}
        <textarea
          className="composer-input"
          placeholder={placeholder}
          value={text}
          disabled={blocked}
          rows={1}
          // 截图后直接 Ctrl/Cmd+V 贴进来——找参考图最顺手的动作，不该逼用户先存盘再点按钮。
          onPaste={(e) => {
            const files = onlyImages(e.clipboardData.files);
            if (files.length) {
              e.preventDefault();
              addImages(files);
            }
          }}
          onChange={(e) => {
            setText(e.target.value);
            setMenuIndex(0);
            e.target.style.height = "auto";
            e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
          }}
          onKeyDown={(e) => {
            if (menuOpen) {
              // 菜单开着时上下键选、Enter/Tab 选中、Esc 关（关 = 把 / 留在文本里当普通字符）。
              if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                e.preventDefault();
                setMenuIndex((i) => (i + (e.key === "ArrowDown" ? 1 : menu.length - 1)) % menu.length);
                return;
              }
              if (e.key === "Enter" || e.key === "Tab") {
                e.preventDefault();
                pickSkill(menu[activeIndex]);
                return;
              }
            }
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />

        <div className="composer-toolbar">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp,image/gif,image/bmp"
            multiple
            hidden
            onChange={(e) => {
              addImages(onlyImages(e.target.files));
              e.target.value = ""; // 清空，否则同一张图第二次选不触发 change
            }}
          />
          {/* 澄清轮不给传图：那一轮是在回答 Agent 的提问，不是发起新检索 */}
          {!waiting && (
            <button
              className={`attach-btn ${images.length ? "has-images" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              disabled={running || blocked}
              title="上传参考图找同类相似品（也可直接粘贴截图或拖进来）"
            >
              <ImageIcon width={16} height={16} />
              <span>
                {images.length ? `参考图 ${images.length}/${MAX_IMAGES}` : "图搜同款"}
              </span>
            </button>
          )}
          {running ? (
            <button className="send-btn stop" onClick={onCancel} title="停止">
              <StopIcon width={16} height={16} />
            </button>
          ) : (
            <button
              className={`send-btn ${canSend ? "ready" : ""} ${waiting ? "is-clarify" : ""}`}
              onClick={submit}
              disabled={!canSend}
              title={waiting ? "回复" : "发送"}
            >
              <ArrowUp width={18} height={18} />
            </button>
          )}
        </div>
      </div>
      <p className="composer-hint">ShoppingX 跨境购物 Agent · 结果由模型与离线数据生成，仅供参考</p>
    </div>
  );
}
