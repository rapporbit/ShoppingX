// 落地页（未登录首屏）。
//
// 在此之前，未登录访客看到的是一个孤零零的登录框——不说这是什么、能干什么、凭什么值得你注册。
// 这是「demo 感」最浓的一处：真实产品在要你的密码之前，总要先告诉你它是干什么的。
//
// 这里只讲**本项目真的做得到**的三件事，不吹跨平台实时比价（数据是快照，见 SiteFooter 的声明）。
// 落地页吹的每一句，用户 30 秒后就会在结果页里对账——吹过了头，反而坐实了 demo 感。
import { SiteFooter } from "./SiteFooter";

const FEATURES = [
  {
    title: "一句话说需求，不用凑关键词",
    body: "「便宜又抗造的旅行三件套，预算 300，不要塑料，喜欢小众一点」——预算、材质、风格、硬约束，它自己拆得清，不用你翻译成搜索框里的几个词。",
  },
  {
    title: "算的是到手价，不是标价",
    body: "标价 $18.79 不等于你付 $18.79。关税、运费按收货国一起折进去，跨平台比的是同一个口径的价——这才是你真正掏的钱。",
  },
  {
    title: "它会记住你的偏好",
    body: "你说过「不要塑料」「偏爱小众品牌」，下次不用再说一遍。偏好跟着账号走，可以随时在偏好面板里改或删掉。",
  },
];

const STEPS = [
  { n: "1", t: "说出需求", d: "自然语言，越具体越好" },
  { n: "2", t: "它去检索、比价、精挑", d: "过程可展开看，不是黑箱" },
  { n: "3", t: "拿到带理由的清单", d: "每件为什么选它，写给你看" },
];

export function Landing({ onStart }: { onStart: () => void }) {
  return (
    <div className="landing">
      <header className="landing-nav">
        <div className="landing-brand">
          <span className="landing-logo" aria-hidden>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 3 3 20h4l5-10 5 10h4Z" fill="currentColor" />
            </svg>
          </span>
          <span className="landing-name">ShoppingX</span>
        </div>
        <button className="landing-signin" onClick={onStart}>
          登录 / 注册
        </button>
      </header>

      <section className="landing-hero">
        <h1 className="landing-title">
          把「想买点什么」交给它，
          <br />
          你只管挑。
        </h1>
        <p className="landing-sub">
          说一句话，ShoppingX 替你跨平台找货、算到手价（含税含运）、按你的偏好精挑，
          最后给一份写清了「为什么是它」的清单。
        </p>
        <button className="landing-cta" onClick={onStart}>
          免费开始
        </button>
        <p className="landing-cta-note">注册即用，无需绑卡</p>
      </section>

      <section className="landing-features">
        {FEATURES.map((f) => (
          <div className="landing-card" key={f.title}>
            <h3>{f.title}</h3>
            <p>{f.body}</p>
          </div>
        ))}
      </section>

      <section className="landing-steps">
        <h2 className="landing-h2">怎么用</h2>
        <ol className="steps-row">
          {STEPS.map((s) => (
            <li key={s.n}>
              <span className="step-n">{s.n}</span>
              <div>
                <strong>{s.t}</strong>
                <span className="step-d">{s.d}</span>
              </div>
            </li>
          ))}
        </ol>
      </section>

      <SiteFooter />
    </div>
  );
}
