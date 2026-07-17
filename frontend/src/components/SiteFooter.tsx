// 落地页 / 法务页共用的页脚。
//
// 「关于 / 隐私 / 条款 + 一个能联系上人的邮箱」是正式网站与 demo 最省钱的一道分界线：它们本身
// 没有功能，但没有它们，任何一个认真的访客都会默认「这东西背后没人」。
//
// 数据来源声明也放这里，且必须说实话：商品来自公开数据集快照，不是实时行情。把这句摆在明处，
// 用户点进商品发现价格对不上时，是「我早知道」而不是「这站在骗我」——同一件事，两种信任结局。
export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <span className="footer-logo" aria-hidden>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M12 3 3 20h4l5-10 5 10h4Z" fill="currentColor" />
            </svg>
          </span>
          <span>ShoppingX</span>
        </div>

        <nav className="footer-links">
          <a href="/about">关于</a>
          <a href="/privacy">隐私政策</a>
          <a href="/terms">服务条款</a>
          <a href="mailto:hello@shopx.oiuu.de">联系我们</a>
        </nav>
      </div>

      <p className="footer-note">
        商品数据来自公开电商数据集快照，价格、库存与链接可能已变动，仅供比较参考，请以平台实际页面为准。
        ShoppingX 与 Amazon、eBay、Shopee 等平台无隶属关系。
      </p>

      <p className="footer-copy">© {new Date().getFullYear()} ShoppingX</p>
    </footer>
  );
}
