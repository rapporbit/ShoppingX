// 关于 / 隐私政策 / 服务条款。
//
// 内容按**本项目真实的行为**写：存了什么、发给了谁、留多久。隐私政策是少数几种「写得漂亮不如
// 写得准确」的文本——照抄模板写上「我们不会将数据分享给第三方」，而代码里每一条 query 都发给了
// 第三方 LLM，那不是合规，是撒谎。
//
// 路由不引 react-router：nginx 已把所有路径 fallback 到 index.html（见 docker/nginx-spa.conf），
// 前端读一次 pathname 就够了。为三个静态页拖进一个路由库不值当。
import { SiteFooter } from "./SiteFooter";

export type LegalPage = "about" | "privacy" | "terms";

const UPDATED = "2026 年 7 月";

export function Legal({ page, onHome }: { page: LegalPage; onHome: () => void }) {
  return (
    <div className="legal">
      <header className="landing-nav">
        <button className="landing-brand as-button" onClick={onHome}>
          <span className="landing-logo" aria-hidden>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <path d="M12 3 3 20h4l5-10 5 10h4Z" fill="currentColor" />
            </svg>
          </span>
          <span className="landing-name">ShoppingX</span>
        </button>
        <button className="landing-signin" onClick={onHome}>
          返回首页
        </button>
      </header>

      <article className="legal-body">
        {page === "about" && <About />}
        {page === "privacy" && <Privacy />}
        {page === "terms" && <Terms />}
      </article>

      <SiteFooter />
    </div>
  );
}

function About() {
  return (
    <>
      <h1>关于 ShoppingX</h1>
      <p className="legal-lede">
        ShoppingX 是一个对话式购物助手：你用一句话说出购物意图，它替你检索商品、把关税运费折进价格里、
        按你的长期偏好精挑，最后给一份带选购理由的清单。
      </p>

      <h2>它能做什么</h2>
      <p>
        理解带约束的自然语言需求（预算、材质、风格、「不要什么」）；跨平台检索并按到手价（含税含运）
        统一口径比价；把你说过的偏好沉淀下来，下次不必重复。每一轮的检索与推理过程都可以展开查看，
        不做黑箱。
      </p>

      <h2>它现在还做不到什么</h2>
      <p>
        商品数据来自公开电商数据集的<strong>快照</strong>，不是实时行情——价格、库存和链接都可能已经变动，
        请以平台的实际页面为准。ShoppingX 不代售、不下单、不收款，也不与任何电商平台有隶属或合作关系；
        购买行为发生在你自己和平台之间。
      </p>

      <h2>联系</h2>
      <p>
        产品建议、bug、合作，都可以发到 <a href="mailto:hello@shopx.oiuu.de">hello@shopx.oiuu.de</a>。
      </p>
    </>
  );
}

function Privacy() {
  return (
    <>
      <h1>隐私政策</h1>
      <p className="legal-updated">最后更新：{UPDATED}</p>

      <h2>我们收集什么</h2>
      <p>
        <strong>账号信息</strong>：用户名和密码。密码以哈希形式存储，我们无法读出你的原始密码。注册不需要邮箱、
        手机号或任何真实身份信息。
        <br />
        <strong>使用数据</strong>：你发起的购物需求（query）、对话内容、你上传的图片、会话产生的结果，
        以及从对话中提炼的长期偏好（如「不要塑料材质」）。
        <br />
        <strong>用量记录</strong>：每次任务消耗的额度，用于配额限制。
      </p>

      <h2>数据发往哪里</h2>
      <p>
        为了理解你的需求并生成回答，你的对话内容（含上传的图片）会发送给<strong>第三方大模型服务商</strong>处理；
        使用联网搜索能力时，检索词会发送给第三方搜索服务。请不要在对话中输入身份证号、银行卡号、密码
        等敏感个人信息——它们没有必要，也会随对话一并发出。
      </p>

      <h2>我们不做什么</h2>
      <p>我们不出售你的数据，不用它投放广告，不将它提供给上述必要服务商之外的任何第三方。</p>

      <h2>你的控制权</h2>
      <p>
        长期偏好可以随时在偏好面板里查看、修改或删除；历史会话可以逐条删除。需要注销账号并清除全部数据，
        发邮件到 <a href="mailto:hello@shopx.oiuu.de">hello@shopx.oiuu.de</a>。
      </p>

      <h2>Cookie</h2>
      <p>我们不使用广告或分析类 Cookie。登录状态保存在你浏览器的本地存储里，退出登录即清除。</p>
    </>
  );
}

function Terms() {
  return (
    <>
      <h1>服务条款</h1>
      <p className="legal-updated">最后更新：{UPDATED}</p>

      <h2>服务内容</h2>
      <p>
        ShoppingX 提供商品检索与选购建议。<strong>我们不销售商品、不处理订单、不收取货款</strong>——
        所有购买行为发生在你与电商平台之间，与我们无关。
      </p>

      <h2>信息准确性</h2>
      <p>
        商品数据来自公开数据集快照，价格、库存、规格与链接均可能过时或有误。关税与运费为
        <strong>估算值</strong>，实际以平台结算与海关为准。请在下单前自行核实。我们尽力但不保证信息准确、
        完整或持续可用。
      </p>

      <h2>使用规范</h2>
      <p>
        请勿滥用本服务：包括自动化批量请求、绕过额度限制、逆向或干扰服务运行，以及输入违法内容。
        我们保留在滥用发生时限制或终止账号的权利。
      </p>

      <h2>额度</h2>
      <p>
        每个账号有每日额度限制，用尽后需等待次日重置。额度不是商品，不可购买、转让或兑现。
      </p>

      <h2>免责</h2>
      <p>
        本服务按「现状」提供。因依据本服务的建议而产生的购买决策及其后果，由你自行承担。
        在法律允许的最大范围内，我们不对由此产生的任何损失负责。
      </p>

      <h2>变更</h2>
      <p>条款可能更新，重大变更会在本页标注更新日期。继续使用即表示接受更新后的条款。</p>
    </>
  );
}
