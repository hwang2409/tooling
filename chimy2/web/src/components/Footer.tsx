export function Footer() {
  const iso = new Date().toISOString().slice(0, 16).replace("T", " ");
  return (
    <footer className="site-footer">
      <div className="site-footer__brand">
        chimy<span className="site-footer__two">2</span>
        <span className="site-footer__tag">a from-scratch software rasterizer</span>
      </div>
      <div className="site-footer__meta">
        <span>Rendered on CPU</span>
        <span className="site-footer__dot" />
        <span className="tabular">{iso}</span>
      </div>
    </footer>
  );
}
