import { useEffect, useState } from "react";
import { fetchUploadedImage } from "../api";

// 用户气泡里的参考图（M20 图搜）。图不是附件，是这一轮提问的一部分——用户说「找这个同款」时，
// 「这个」指的就是它。对话里不把它画出来，回看时那句话就没了主语。
//
// 取图必须走 fetch → blob → object URL：接口要校验属主，而 <img src> 带不上 Authorization 头
// （直链稳定 401）。代价是要自己管生命周期，见下面的回收。
type QueryImagesProps = {
  threadId: string | null;
  images: string[];
};

export function QueryImages({ threadId, images }: QueryImagesProps) {
  const [urls, setUrls] = useState<string[]>([]);
  const [zoomed, setZoomed] = useState<string | null>(null);

  useEffect(() => {
    if (!threadId || images.length === 0) return;
    let alive = true;
    const created: string[] = [];

    void (async () => {
      const got = await Promise.all(
        images.map((name) =>
          // 单张失败不连坐其余张：一张图 404（比如老会话的图被清过）不该让整轮的图全不显示。
          fetchUploadedImage(threadId, name).catch(() => null),
        ),
      );
      const ok = got.filter((u): u is string => u !== null);
      created.push(...ok);
      // 组件已卸载（用户切走了会话）：此时 setState 是徒劳的，但 URL 已经造出来了，得就地回收。
      if (!alive) {
        ok.forEach(URL.revokeObjectURL);
        return;
      }
      setUrls(ok);
    })();

    return () => {
      alive = false;
      created.forEach(URL.revokeObjectURL); // 不回收 = 每次回看旧会话都漏一份图在内存里
    };
  }, [threadId, images]);

  if (urls.length === 0) return null;

  return (
    <>
      <div className="query-images">
        {urls.map((u) => (
          <button
            key={u}
            className="query-image"
            onClick={() => setZoomed(u)}
            title="点击查看大图"
          >
            <img src={u} alt="参考图" />
          </button>
        ))}
      </div>

      {/* 大图浮层。点任意处关闭——看图是个「瞥一眼」的动作，不该逼用户找关闭按钮。 */}
      {zoomed && (
        <div className="image-lightbox" onClick={() => setZoomed(null)} role="presentation">
          <img src={zoomed} alt="参考图（大图）" />
        </div>
      )}
    </>
  );
}
