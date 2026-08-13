export type ShaderMode = {
  key: string;
  name: string;
  tag: string;
  desc: string;
};

export const SHADER_MODES: ShaderMode[] = [
  {
    key: "1",
    name: "Studio",
    tag: "BLINN-PHONG",
    desc: "Blinn-Phong with normal mapping on the showcase mesh under warm key and cool rim.",
  },
  {
    key: "2",
    name: "Toon",
    tag: "TOON",
    desc: "Cel-shaded ramp lighting with hard bands and a wine outline.",
  },
  {
    key: "3",
    name: "PSX",
    tag: "PSX",
    desc: "Snapped vertex positions plus low-precision affine warp for the 1999 look.",
  },
  {
    key: "4",
    name: "Dither",
    tag: "DITHER",
    desc: "Ordered-dither posterization traded across a warm terracotta palette.",
  },
  {
    key: "5",
    name: "Fog",
    tag: "FOG",
    desc: "Depth-blended atmospheric fog against a cool base.",
  },
  {
    key: "6",
    name: "Normals",
    tag: "NORMALS",
    desc: "World-space surface normals visualised directly as RGB.",
  },
  {
    key: "7",
    name: "Wire",
    tag: "WIREFRAME",
    desc: "Barycentric wireframe rendering with anti-aliased edges over a dark inkwell.",
  },
  {
    key: "8",
    name: "Skeleton",
    tag: "GLTF · SKINNED",
    desc: "glTF 2.0 with CPU skeletal skinning, animated over its arm rig.",
  },
];
