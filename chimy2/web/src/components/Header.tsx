import { useEffect, useState } from "react";

export function Header() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 40);
    handler();
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  return (
    <header className={`site-header${scrolled ? " site-header--scrolled" : ""}`}>
      <a className="brand" href="#top" aria-label="chimy2 — home">
        <span className="brand__mark">
          <em>c</em>himy<span className="brand__two">2</span>
        </span>
        <span className="brand__eyebrow">v0.1 · software rasterizer</span>
      </a>
      <nav className="site-nav" aria-label="primary">
        <a href="#live">Live</a>
        <a href="#scenes">Scenes</a>
        <a href="#about">About</a>
        <a
          href="https://github.com/hwang2409/tooling"
          target="_blank"
          rel="noreferrer noopener"
          className="site-nav__ext"
        >
          Source ↗
        </a>
      </nav>
    </header>
  );
}
