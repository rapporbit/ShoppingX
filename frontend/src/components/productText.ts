// 商品卡 / 详情 / 对比 共用的小工具：平台显示名、理由拆行、展示价。
// 从 ProductCards.tsx 抽出来，让三个视图对同一件商品的解读一致（同一个价、同一组理由）。

import type { ProductItem } from "../types";

// 与后端 app/utils/clean.py 的 PLATFORMS 对齐（eBay 在清洗阶段整体剔除，库里没有它的商品）。
const PLATFORM_LABEL: Record<string, string> = {
  amazon: "Amazon",
  lazada: "Lazada",
  shein: "SHEIN",
  shopee: "Shopee",
  walmart: "Walmart",
};

export function platformName(p: string): string {
  return PLATFORM_LABEL[p.toLowerCase()] ?? p;
}

// 选购理由按「；/ ; / 换行」拆成条目。
export function splitReasons(reason: string | undefined): string[] {
  return (reason ?? "")
    .split(/[；;\n]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

// 一件商品的展示价：优先到手价（含税运），没算过就用货价——只取数字，口径由各视图自己标注。
export function shownPrice(it: ProductItem): number | null {
  if (typeof it.landed_usd === "number") return it.landed_usd;
  if (typeof it.price_usd === "number") return it.price_usd;
  return null;
}

export function priceKind(it: ProductItem): "landed" | "goods" | null {
  if (typeof it.landed_usd === "number") return "landed";
  if (typeof it.price_usd === "number") return "goods";
  return null;
}

// 详情 / 对比里引用一件商品给 Agent 看的写法：标题 + item_id。item_id 是后端候选登记表的键，
// 带上它模型才能不靠标题模糊匹配（同名不同平台的商品很常见）。
export function itemRef(it: ProductItem): string {
  return `《${it.title}》（item_id: ${it.item_id}，${platformName(it.platform)}）`;
}
