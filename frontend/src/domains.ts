// 偏好域 —— 与后端 app/memory/domains.py 的 PrefDomain / DOMAIN_LABELS 一一对应。
//
// 域决定一条偏好**在哪些轮次生效**：读取端（item_picker / item_search / prompt 注入）拿本轮
// planner 判出的品类域与它做一次字符串相等比较。所以这里必须是封闭枚举、且与后端逐字一致——
// 前端多写一个自由文本域，后端会 422；少写一个，用户就选不到那个域。
//
// 两个逃生舱语义相反，别搞反：
//   other  —— 判不出是哪个域。保守档：这条偏好几乎不生效（只有本轮没判出域时才兜底生效）。
//   global —— 跨品类底线（过敏 / 伦理 / 收货地）。**唯一**会跨品类杀商品的档位。

export type PrefDomain =
  | "apparel"
  | "footwear"
  | "bags"
  | "jewelry_watches"
  | "electronics"
  | "computers"
  | "phones"
  | "home_kitchen"
  | "furniture"
  | "garden"
  | "beauty"
  | "health"
  | "food"
  | "sports"
  | "toys_baby"
  | "books_media"
  | "auto"
  | "pet"
  | "office"
  | "tools"
  | "other"
  | "global";

// 下拉里的展示顺序：先两个逃生舱（最常选的默认 other + 需要慎选的 global），再按品类分组。
export const DOMAIN_ORDER: PrefDomain[] = [
  "other",
  "global",
  "apparel",
  "footwear",
  "bags",
  "jewelry_watches",
  "electronics",
  "computers",
  "phones",
  "home_kitchen",
  "furniture",
  "garden",
  "beauty",
  "health",
  "food",
  "sports",
  "toys_baby",
  "books_media",
  "auto",
  "pet",
  "office",
  "tools",
];

// 中文标签（取自后端 DOMAIN_LABELS，逃生舱的长说明这里改短——下拉里要能一眼扫完）。
export const DOMAIN_LABELS: Record<PrefDomain, string> = {
  apparel: "服饰",
  footwear: "鞋履",
  bags: "箱包",
  jewelry_watches: "首饰腕表",
  electronics: "消费电子",
  computers: "电脑及配件",
  phones: "手机及配件",
  home_kitchen: "家居厨房",
  furniture: "家具",
  garden: "园艺户外",
  beauty: "美妆个护",
  health: "保健医疗",
  food: "食品饮料",
  sports: "运动户外",
  toys_baby: "玩具母婴",
  books_media: "图书影音",
  auto: "汽车用品",
  pet: "宠物用品",
  office: "办公文具",
  tools: "工具五金",
  other: "判不出品类",
  global: "跨品类底线",
};

// 下拉选项里跟在标签后面的一行说明，只给两个逃生舱——它们的行为和普通品类域不是一回事。
export const DOMAIN_HINTS: Partial<Record<PrefDomain, string>> = {
  other: "几乎不生效，Agent 判不出域时的保守兜底",
  global: "所有品类都生效，安全 / 过敏 / 伦理才用",
};

// 是否为「跨品类生效」的域——列表里要把它单独标出来（它是唯一会跨品类影响结果的档位）。
export const isGlobalDomain = (d: string): boolean => d === "global";

export const domainLabel = (d: string): string =>
  DOMAIN_LABELS[d as PrefDomain] ?? d;
