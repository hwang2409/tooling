const STATS = [
  { value: "37", label: "Feature milestones" },
  { value: "~10k", label: "Lines of Rust" },
  { value: "2", label: "Runtime deps" },
  { value: "8", label: "Scenes" },
];

const INVENTORY = [
  { group: "Geometry", items: ["Perspective-correct scanline raster", "Depth buffer", "Tiled bin pass", "QEM LOD", "Morph targets", "CPU skinning"] },
  { group: "Shading", items: ["Cook-Torrance GGX", "IBL bake", "Blinn-Phong", "Toon ramp", "PSX affine", "Wireframe"] },
  { group: "Lighting", items: ["CSM × 4", "PCSS", "Point cube shadows", "SSAO", "Basic shadow maps"] },
  { group: "Post", items: ["Bloom", "Depth of field", "ACES tonemap", "FXAA", "Vignette"] },
  { group: "IO", items: ["glTF 2.0", "OBJ / MTL", "QOI", "Bitmap font", "Strict JSON scene format"] },
  { group: "Runtime", items: ["Native viewer", "Deterministic CPU particles", "wasm32 boundary", "Instanced draw + culling"] },
];

export function About() {
  return (
    <section id="about" className="about">
      <div className="section-head">
        <span className="section-head__idx">03</span>
        <h2 className="section-head__title">About · What is in the box</h2>
        <span className="section-head__meta">from scratch · Rust</span>
      </div>

      <div className="about__stats">
        {STATS.map((stat) => (
          <div className="about__stat" key={stat.label}>
            <span className="about__stat-value tabular">{stat.value}</span>
            <span className="about__stat-label">{stat.label}</span>
          </div>
        ))}
      </div>

      <div className="about__body">
        <p>
          <em>Everything is hand-written:</em> the vector math, the
          perspective-correct scanline rasterizer, the depth buffer and its
          bin-parallel tiles, the Cook-Torrance GGX BRDF, image-based lighting
          bake, cascaded shadow maps, PCSS soft shadows, point-light cube shadows,
          SSAO, depth of field, bloom, FXAA, vignette, ACES tone-mapping, the QOI
          decoder, the OBJ/MTL and glTF 2.0 loaders, the strict JSON scene
          format, the bitmap font, the CPU particle system, and the WebAssembly
          boundary you are watching this text through.
        </p>
        <p>
          The two runtime dependencies are the platform windowing shim
          (<code>winit</code>+<code>softbuffer</code>, used only by the native
          viewer) and the operating-system allocator. The browser build is
          <code> wasm32-unknown-unknown</code> with no imports beyond memory —
          it ships as a single flat framebuffer that JavaScript copies into an
          <code> ImageData</code>.
        </p>
      </div>

      <div className="about__inventory">
        {INVENTORY.map((group) => (
          <div className="about__group" key={group.group}>
            <h3 className="about__group-title">{group.group}</h3>
            <ul className="about__group-list">
              {group.items.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}
