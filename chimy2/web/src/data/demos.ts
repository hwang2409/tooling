export type Demo = {
  id: string;
  name: string;
  image: string;
  tag: string;
  hue: string;
};

// Each entry is a hand-authored scene rendered natively (see chimy2/tests/web_scenes.rs).
// The browser bundles the pre-rendered PNG as the visible artwork; the live wasm chip
// beside the plinth proves the same rasterizer runs on the CPU in a browser thread.
export const DEMOS: Demo[] = [
  { id: "01-hero",          name: "Hero",           image: "./gallery/01-hero.png",           tag: "CSM · point shadow · SSAO · bloom · ACES", hue: "#4a7bc8" },
  { id: "02-pbr-materials", name: "PBR Sweep",      image: "./gallery/02-pbr-materials.png",  tag: "Cook-Torrance GGX · gold + sapphire",       hue: "#e9c893" },
  { id: "03-soft-shadows",  name: "Soft Shadows",   image: "./gallery/03-soft-shadows.png",   tag: "PCSS · contact hardening",                  hue: "#8e8778" },
  { id: "04-cascades",      name: "Cascades",       image: "./gallery/04-cascades.png",       tag: "CSM ×4 · 66 instanced icosahedra",          hue: "#c9c3b7" },
  { id: "05-particles",     name: "Particles",      image: "./gallery/05-particles.png",      tag: "Deterministic CPU emitters · bloom",        hue: "#f2b45c" },
  { id: "06-depth-of-field",name: "Depth of Field", image: "./gallery/06-depth-of-field.png", tag: "Circle-of-confusion DoF · focus 5.6 m",     hue: "#7fbfa0" },
  { id: "07-glow",          name: "Glow",           image: "./gallery/07-glow.png",           tag: "HDR point lights · bloom · ACES",           hue: "#d64a8a" },
  { id: "08-minimal",       name: "Minimal",        image: "./gallery/08-minimal.png",        tag: "Basic shadow · FXAA · one object",          hue: "#ede7dd" },
];
