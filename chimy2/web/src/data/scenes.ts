export type Scene = {
  id: string;
  file: string;
  image: string;
  name: string;
  desc: string;
  features: string[];
};

export const SCENES: Scene[] = [
  {
    id: "01-hero",
    file: "./scenes/01-hero.scene.json",
    image: "./gallery/01-hero.png",
    name: "Hero",
    desc: "The everything shot. CSM directional plus two point lights, one with cube shadow, hard-lit metals over an SSAO-shaded floor, bloom and ACES to finish.",
    features: ["CSM", "Cube shadow", "SSAO", "Bloom", "ACES"],
  },
  {
    id: "02-pbr-materials",
    file: "./scenes/02-pbr-materials.scene.json",
    image: "./gallery/02-pbr-materials.png",
    name: "PBR Sweep",
    desc: "Cook-Torrance GGX sweep. Gold at r=0.10, 0.35, 0.70. Sapphire dielectric at r=0.20, 0.55. Same rig, only roughness changes.",
    features: ["GGX", "IBL", "Metallic", "Roughness"],
  },
  {
    id: "03-soft-shadows",
    file: "./scenes/03-soft-shadows.scene.json",
    image: "./gallery/03-soft-shadows.png",
    name: "Soft Shadows",
    desc: "PCSS with contact hardening. Three columns at different heights show the penumbra widening as receiver distance grows.",
    features: ["PCSS", "Contact harden", "Point fill"],
  },
  {
    id: "04-cascades",
    file: "./scenes/04-cascades.scene.json",
    image: "./gallery/04-cascades.png",
    name: "Cascades",
    desc: "66 instanced icosahedra under a 4-cascade CSM. Instance grid uses a single mesh; the cascade splits keep resolution stable to the horizon.",
    features: ["CSM × 4", "Instancing", "SSAO"],
  },
  {
    id: "05-particles",
    file: "./scenes/05-particles.scene.json",
    image: "./gallery/05-particles.png",
    name: "Particles",
    desc: "Three deterministic CPU emitters over a warm point light. Every particle is a billboard quad through the same rasterizer that drew the meshes behind it.",
    features: ["CPU particles", "Bloom", "Point light"],
  },
  {
    id: "06-depth-of-field",
    file: "./scenes/06-depth-of-field.scene.json",
    image: "./gallery/06-depth-of-field.png",
    name: "Depth of Field",
    desc: "Circle-of-confusion DoF pass focused at 5.6 metres. Foreground orange and background violet dissolve; the near-camera sphere stays crisp.",
    features: ["DoF", "SSAO", "Bloom"],
  },
  {
    id: "07-glow",
    file: "./scenes/07-glow.scene.json",
    image: "./gallery/07-glow.png",
    name: "Glow",
    desc: "Three HDR-bright coloured point lights hit polished dielectrics; the highlights push past 1.0 in linear space and bloom folds them back through ACES.",
    features: ["HDR", "Bloom", "ACES", "Three lights"],
  },
  {
    id: "08-minimal",
    file: "./scenes/08-minimal.scene.json",
    image: "./gallery/08-minimal.png",
    name: "Minimal",
    desc: "One dark object on a cream floor under a basic shadow map and FXAA. A single sample of the same pipeline at rest.",
    features: ["Basic shadow", "FXAA"],
  },
];
