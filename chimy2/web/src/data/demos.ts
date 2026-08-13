export type Demo = {
  id: string;
  name: string;
  tag: string;
  hue: string;
};

// The 8 rail entries. `hue` is the small colour dot in the demo list. Each entry
// maps 1:1 to the wasm `render_frame(mode)` argument (0..7); see chimy2/src for the
// shader table.
export const DEMOS: Demo[] = [
  { id: "01-hero",          name: "Hero",           tag: "CSM · point shadow · SSAO · bloom · ACES", hue: "#4a7bc8" },
  { id: "02-pbr-materials", name: "PBR Sweep",      tag: "Cook-Torrance GGX · gold + sapphire",       hue: "#e9c893" },
  { id: "03-soft-shadows",  name: "Soft Shadows",   tag: "PCSS · contact hardening",                  hue: "#8e8778" },
  { id: "04-cascades",      name: "Cascades",       tag: "CSM ×4 · 66 instanced icosahedra",          hue: "#c9c3b7" },
  { id: "05-particles",     name: "Particles",      tag: "Deterministic CPU emitters · bloom",        hue: "#f2b45c" },
  { id: "06-depth-of-field",name: "Depth of Field", tag: "Circle-of-confusion DoF · focus 5.6 m",     hue: "#7fbfa0" },
  { id: "07-glow",          name: "Glow",           tag: "HDR point lights · bloom · ACES",           hue: "#d64a8a" },
  { id: "08-minimal",       name: "Minimal",        tag: "Basic shadow · FXAA · one object",          hue: "#ede7dd" },
];
