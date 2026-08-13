export type Demo = {
  id: string;
  file: string;
  name: string;
  tag: string;
  hue: string;
};

// The eight curated scenes. Clicking one fetches its JSON, hands the bytes to
// the wasm via `scene_alloc` + `load_scene_json`, and the live rasterizer draws
// it in the plinth canvas. `hue` seeds the small colour dot in the rail — keyed
// to the scene's dominant palette.
export const DEMOS: Demo[] = [
  { id: "01-hero",           file: "./scenes/01-hero.scene.json",           name: "Hero",           tag: "CSM · point shadow · SSAO · bloom · ACES", hue: "#4a7bc8" },
  { id: "02-pbr-materials",  file: "./scenes/02-pbr-materials.scene.json",  name: "PBR Sweep",      tag: "Cook-Torrance GGX · gold + sapphire",       hue: "#e9c893" },
  { id: "03-soft-shadows",   file: "./scenes/03-soft-shadows.scene.json",   name: "Soft Shadows",   tag: "PCSS · contact hardening",                  hue: "#8e8778" },
  { id: "04-cascades",       file: "./scenes/04-cascades.scene.json",       name: "Cascades",       tag: "CSM ×4 · 66 instanced icosahedra",          hue: "#c9c3b7" },
  { id: "05-particles",      file: "./scenes/05-particles.scene.json",      name: "Particles",      tag: "Deterministic CPU emitters · bloom",        hue: "#f2b45c" },
  { id: "06-depth-of-field", file: "./scenes/06-depth-of-field.scene.json", name: "Depth of Field", tag: "Circle-of-confusion DoF · focus 5.6 m",     hue: "#7fbfa0" },
  { id: "07-glow",           file: "./scenes/07-glow.scene.json",           name: "Glow",           tag: "HDR point lights · bloom · ACES",           hue: "#d64a8a" },
  { id: "08-minimal",        file: "./scenes/08-minimal.scene.json",        name: "Minimal",        tag: "Basic shadow · FXAA · one object",          hue: "#ede7dd" },
];
