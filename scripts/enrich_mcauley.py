"""用 McAuley Amazon Reviews 2023 给 amazon 主库补 brand / features / 评论数（流式、只用标准库）。

主库（``amazon_rag.jsonl``）是子 ASIN、无 brand；McAuley meta 以 parent_asin 为键。两段式：
  ① 流式扫 ``review_categories/<cat>.jsonl``，只取 ``asin → parent_asin``，命中主库 id 的留下；
  ② 流式扫 ``meta_categories/meta_<cat>.jsonl``，parent_asin 命中
     （映射到的父 ∪ 主库 id 本身）才 json 解析。
大文件不落盘（curl | 逐行处理即弃）；按品类断点：每品类完成写 done 标记，流中断从字节偏移续传。
资源：curl 限速；内存 / 磁盘守卫越线即退出（外层再用 systemd CPUQuota / Nice 压）。

用法（ids 文件 = 每行一个主库 item_id）：
    python3 scripts/enrich_mcauley.py --ids ids.txt --out out/ --rate 10M [--only Appliances]
HF token 读 ``~/.cache/huggingface/token``（每次请求现读，中途放进去即生效；没有就匿名）。
"""

from __future__ import annotations

import argparse
import json
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
# 跳过 Books/Kindle/Movies/CDs/Digital_Music/Magazine/Software/Gift_Cards：主库几乎没有这些品类。
# 顺序 = 短标题重灾区优先，早上没跑完也先拿到最值钱的部分。
CATEGORIES = [
    "Clothing_Shoes_and_Jewelry",
    "Amazon_Fashion",
    "Sports_and_Outdoors",
    "Electronics",
    "Cell_Phones_and_Accessories",
    "Home_and_Kitchen",
    "Beauty_and_Personal_Care",
    "All_Beauty",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Toys_and_Games",
    "Baby_Products",
    "Tools_and_Home_Improvement",
    "Automotive",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Office_Products",
    "Arts_Crafts_and_Sewing",
    "Grocery_and_Gourmet_Food",
    "Video_Games",
    "Industrial_and_Scientific",
    "Musical_Instruments",
    "Appliances",
    "Handmade_Products",
    "Subscription_Boxes",
    "Unknown",
]
TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"
# 评论正文里的引号是转义的（\"），不会误中这两个模式；
# asin 前一个字符是引号，也不会误中 parent_asin。
_ASIN_RE = re.compile(rb'"asin": "([^"]+)"')
_PARENT_RE = re.compile(rb'"parent_asin": "([^"]+)"')
_DETAIL_KEYS = re.compile(
    r"material|fabric|brand|color|size|weight|dimension|style|closure|sole|type", re.I
)
MAX_RSS_MB = 1024
MIN_FREE_GB = 5


def _guard(out: Path) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
        1024 * 1024 if sys.platform == "darwin" else 1024
    )
    free = shutil.disk_usage(out).free / 1e9
    if rss > MAX_RSS_MB or free < MIN_FREE_GB:
        sys.exit(f"[guard] 越线退出 rss={rss:.0f}MB free={free:.1f}GB")


def _auth_args(out: Path) -> list[str]:
    """有 token 就写成 600 权限的 header 文件交给 curl（-H @file），不进进程命令行。"""
    if not TOKEN_FILE.is_file() or not (tok := TOKEN_FILE.read_text().strip()):
        return []
    hdr = out / ".hf_header"
    hdr.touch(mode=0o600)
    hdr.write_text(f"Authorization: Bearer {tok}\n")
    return ["-H", f"@{hdr}"]


def stream_lines(url: str, rate: str, out: Path, retries: int = 30):
    """逐行产出完整行；curl 中断则从已消费的字节偏移续传（半行丢弃重读）。"""
    offset, tries = 0, 0
    while True:
        cmd = ["curl", "-sSfL", "--limit-rate", rate, *_auth_args(out), "-r", f"{offset}-", url]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        assert proc.stdout is not None
        tail = b""
        for line in proc.stdout:
            if not line.endswith(b"\n"):
                tail = line
                break
            offset += len(line)
            yield line
        rc = proc.wait()
        if rc == 0:
            if tail:
                yield tail
            return
        tries += 1
        if tries > retries:
            raise RuntimeError(f"curl 连续失败 {tries} 次 rc={rc} offset={offset} {url}")
        print(f"[retry] rc={rc} offset={offset / 1e9:.2f}GB 第{tries}次，60s 后续传", flush=True)
        time.sleep(60)


def _last(pat: re.Pattern[bytes], line: bytes) -> str | None:
    hits = pat.findall(line)
    return hits[-1].decode() if hits else None


def map_children(cat: str, ids: set[str], rate: str, out: Path) -> dict[str, str]:
    """① reviews：主库 id 作为子 asin 出现时记下它的 parent_asin。"""
    c2p: dict[str, str] = {}
    nbytes, t0, mark = 0, time.time(), 1e9
    for line in stream_lines(f"{BASE}/review_categories/{cat}.jsonl", rate, out):
        nbytes += len(line)
        asin = _last(_ASIN_RE, line)
        if asin in ids and asin not in c2p:
            if parent := _last(_PARENT_RE, line):
                c2p[asin] = parent
        if nbytes >= mark:
            mark += 1e9
            _guard(out)
            print(
                f"  [{cat}] reviews {nbytes / 1e9:.0f}GB hits={len(c2p)} "
                f"{nbytes / 1e6 / (time.time() - t0):.1f}MB/s",
                flush=True,
            )
    return c2p


def _record(child: str, parent: str, cat: str, d: dict) -> dict:
    det = d.get("details") or {}
    if isinstance(det, str):
        try:
            det = json.loads(det)
        except ValueError:
            det = {}
    return {
        "item_id": child,
        "parent_asin": parent,
        "source_cat": cat,
        "store": (d.get("store") or "").strip(),
        "title": d.get("title") or "",
        "features": [f[:200] for f in (d.get("features") or [])[:8]],
        "description": " ".join(d.get("description") or [])[:600],
        "details": {k: str(v)[:120] for k, v in det.items() if _DETAIL_KEYS.search(k)},
        "average_rating": d.get("average_rating"),
        "rating_number": d.get("rating_number"),
        "price": d.get("price"),
        "categories": d.get("categories") or [],
    }


def join_meta(cat: str, ids: set[str], c2p: dict[str, str], rate: str, out: Path) -> int:
    """② meta：parent_asin 命中才 json 解析；主库 id 本身就是父的也收。写 enrich_<cat>.jsonl。"""
    p2c: dict[str, list[str]] = {}
    for child, parent in c2p.items():
        p2c.setdefault(parent, []).append(child)
    tmp, n, nbytes, mark = out / f"enrich_{cat}.jsonl.tmp", 0, 0, 1e9
    with tmp.open("w", encoding="utf-8") as f:
        for line in stream_lines(f"{BASE}/meta_categories/meta_{cat}.jsonl", rate, out):
            nbytes += len(line)
            if nbytes >= mark:
                mark += 1e9
                _guard(out)
                print(f"  [{cat}] meta {nbytes / 1e9:.0f}GB written={n}", flush=True)
            parent = _last(_PARENT_RE, line)
            children = list(p2c.get(parent, []))
            if parent in ids and parent not in children:
                children.append(parent)
            if not children:
                continue
            d = json.loads(line)
            for child in children:
                f.write(json.dumps(_record(child, parent, cat, d), ensure_ascii=False) + "\n")
                n += 1
    tmp.rename(out / f"enrich_{cat}.jsonl")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="McAuley 2023 补全 amazon 主库")
    ap.add_argument("--ids", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--rate", default="10M")
    ap.add_argument("--only", nargs="*", help="只跑这些品类（冒烟用）")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = {ln.strip() for ln in a.ids.open() if ln.strip()}
    print(
        f"ids={len(ids)} rate={a.rate} token={'yes' if TOKEN_FILE.is_file() else 'no'}", flush=True
    )
    for cat in a.only or CATEGORIES:
        done = a.out / f"done_{cat}"
        if done.exists():
            continue
        _guard(a.out)
        t0 = time.time()
        c2p_path = a.out / f"c2p_{cat}.json"
        if c2p_path.exists():  # ① 已完成、② 中断过：只重跑 ②
            c2p = json.loads(c2p_path.read_text())
        else:
            c2p = map_children(cat, ids, a.rate, a.out)
            c2p_path.write_text(json.dumps(c2p))
        n = join_meta(cat, ids, c2p, a.rate, a.out)
        done.write_text(f"{len(c2p)} {n} {time.time() - t0:.0f}s\n")
        print(
            f"[done] {cat} mapped={len(c2p)} enriched={n} {(time.time() - t0) / 60:.1f}min",
            flush=True,
        )


if __name__ == "__main__":
    main()
