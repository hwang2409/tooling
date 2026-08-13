import { SCENES } from "../data/scenes";

const STATS = [
  { value: "~10k", label: "Lines of Rust" },
  { value: "2", label: "Runtime deps" },
  { value: "8", label: "Shader modes" },
  { value: "8", label: "Scenes" },
];

export function Hero() {
  return (
    <section className="hero" id="top">
      <div className="hero__grain" aria-hidden />
      <div className="hero__glow hero__glow--a" aria-hidden />
      <div className="hero__glow hero__glow--b" aria-hidden />

      <div className="hero__inner">
        <div className="hero__eyebrow">
          <span className="hero__eyebrow-mark" />
          <span>A software rasterizer, in your browser</span>
        </div>

        <h1 className="hero__title">
          <span className="hero__line">Every&nbsp;pixel</span>
          <span className="hero__line hero__line--italic">is drawn by hand.</span>
        </h1>

        <p className="hero__lede">
          Chimy2 is a hand-written CPU rasterizer in Rust — no GPU, no shading
          language, no graphics API. What runs below is the same renderer that
          produces the eight native scenes further down this page,{" "}
          <em>compiled to WebAssembly and driven by roughly 200 lines of glue</em>.
        </p>

        <div className="hero__meta">
          <div className="hero__stats">
            {STATS.map((stat) => (
              <div className="hero__stat" key={stat.label}>
                <span className="hero__stat-value">{stat.value}</span>
                <span className="hero__stat-label">{stat.label}</span>
              </div>
            ))}
          </div>
          <div className="hero__cta">
            <a href="#live" className="hero__cta-primary">
              <span>Enter the machine</span>
              <span className="hero__cta-arrow" aria-hidden>
                ↓
              </span>
            </a>
            <a href="#scenes" className="hero__cta-secondary">
              See the gallery
            </a>
          </div>
        </div>
      </div>

      <div className="hero__strip" aria-hidden>
        <div className="hero__strip-track">
          {[...SCENES, ...SCENES].map((scene, i) => (
            <span className="hero__strip-cell" key={`${scene.id}-${i}`}>
              <img src={scene.image} alt="" loading="lazy" />
              <span className="hero__strip-tag">{scene.name}</span>
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}
