// chimy2 — browser driver for the wasm software rasterizer.
//
// Two runtime deps: the platform (browser) and the WebAssembly runtime.
// Everything below is plain ES modules and DOM.

const SHADER_MODES = [
  {
    key: "1",
    name: "Studio",
    desc: "Blinn-Phong with normal mapping on the showcase mesh under warm key and cool rim.",
    tag: "BLINN-PHONG",
  },
  {
    key: "2",
    name: "Toon",
    desc: "Cel-shaded ramp lighting with hard bands and a wine outline.",
    tag: "TOON",
  },
  {
    key: "3",
    name: "PSX",
    desc: "Snapped vertex positions plus low-precision affine warp for the 1999 look.",
    tag: "PSX",
  },
  {
    key: "4",
    name: "Dither",
    desc: "Ordered-dither posterization traded across a warm terracotta palette.",
    tag: "DITHER",
  },
  {
    key: "5",
    name: "Fog",
    desc: "Depth-blended atmospheric fog against a cool base.",
    tag: "FOG",
  },
  {
    key: "6",
    name: "Normals",
    desc: "World-space surface normals visualised directly as RGB.",
    tag: "NORMALS",
  },
  {
    key: "7",
    name: "Wire",
    desc: "Barycentric wireframe rendering with anti-aliased edges over a dark inkwell.",
    tag: "WIREFRAME",
  },
  {
    key: "8",
    name: "Skeleton",
    desc: "glTF 2.0 with CPU skeletal skinning, animated over its arm rig.",
    tag: "GLTF · SKINNED",
  },
];

const SCENES = [
  {
    id: "01-hero",
    file: "scenes/01-hero.scene.json",
    image: "gallery/01-hero.png",
    name: "Hero",
    desc: "The everything shot. CSM directional plus two point lights, one with cube shadow, hard-lit metals over an SSAO-shaded floor, bloom and ACES to finish.",
    features: ["CSM", "Cube shadow", "SSAO", "Bloom", "ACES"],
  },
  {
    id: "02-pbr-materials",
    file: "scenes/02-pbr-materials.scene.json",
    image: "gallery/02-pbr-materials.png",
    name: "PBR Sweep",
    desc: "Cook-Torrance GGX sweep. Gold at r=0.10, 0.35, 0.70. Sapphire dielectric at r=0.20, 0.55. Same rig, only roughness changes.",
    features: ["GGX", "IBL", "Metallic", "Roughness"],
  },
  {
    id: "03-soft-shadows",
    file: "scenes/03-soft-shadows.scene.json",
    image: "gallery/03-soft-shadows.png",
    name: "Soft Shadows",
    desc: "PCSS with contact hardening. Three columns at different heights show the penumbra widening as receiver distance grows.",
    features: ["PCSS", "Contact harden", "Point fill"],
  },
  {
    id: "04-cascades",
    file: "scenes/04-cascades.scene.json",
    image: "gallery/04-cascades.png",
    name: "Cascades",
    desc: "66 instanced icosahedra under a 4-cascade CSM. Instance grid uses a single mesh; the cascade splits keep resolution stable to the horizon.",
    features: ["CSM × 4", "Instancing", "SSAO"],
  },
  {
    id: "05-particles",
    file: "scenes/05-particles.scene.json",
    image: "gallery/05-particles.png",
    name: "Particles",
    desc: "Three deterministic CPU emitters over a warm point light. Every particle is a billboard quad through the same rasterizer that drew the meshes behind it.",
    features: ["CPU particles", "Bloom", "Point light"],
  },
  {
    id: "06-depth-of-field",
    file: "scenes/06-depth-of-field.scene.json",
    image: "gallery/06-depth-of-field.png",
    name: "Depth of Field",
    desc: "Circle-of-confusion DoF pass focused at 5.6 metres. Foreground orange and background violet dissolve; the near-camera sphere stays crisp.",
    features: ["DoF", "SSAO", "Bloom"],
  },
  {
    id: "07-glow",
    file: "scenes/07-glow.scene.json",
    image: "gallery/07-glow.png",
    name: "Glow",
    desc: "Three HDR-bright coloured point lights hit polished dielectrics; the highlights push past 1.0 in linear space and bloom folds them back through ACES.",
    features: ["HDR", "Bloom", "ACES", "Three lights"],
  },
  {
    id: "08-minimal",
    file: "scenes/08-minimal.scene.json",
    image: "gallery/08-minimal.png",
    name: "Minimal",
    desc: "One dark object on a cream floor under a basic shadow map and FXAA. A single sample of the same pipeline at rest.",
    features: ["Basic shadow", "FXAA"],
  },
];

const canvas = document.getElementById("viewport");
const stage = document.getElementById("stage");
const context = canvas.getContext("2d", { alpha: false });
const fpsOut = document.getElementById("fps-out");
const fpsTag = document.getElementById("fps-tag");
const frameOut = document.getElementById("frame-out");
const frameTag = document.getElementById("frame-tag");
const orbitOut = document.getElementById("orbit-out");
const modeTag = document.getElementById("mode-tag");
const errorPanel = document.getElementById("error");
const hint = document.getElementById("hint");
const picker = document.getElementById("picker");
const gallery = document.getElementById("gallery");
const detailImg = document.getElementById("detail-img");
const detailCap = document.getElementById("detail-cap");
const detailTitle = document.getElementById("detail-title");
const detailDesc = document.getElementById("detail-desc");
const detailFeatures = document.getElementById("detail-features");
const detailCode = document.getElementById("detail-code");
const copyJsonBtn = document.getElementById("copy-json");
const openJsonBtn = document.getElementById("open-json");
const footerTime = document.getElementById("footer-time");

let wasm;
let imageData;
let mode = 0;
let paused = false;
let yaw = 0;
let pitch = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let frameCount = 0;
let sessionFrames = 0;
let fpsTimestamp = 0;
let hintTimer = 0;
let selectedSceneIndex = 0;
let currentSceneJson = "";

function showError(message) {
  errorPanel.textContent = message;
  errorPanel.hidden = false;
}

function clearError() {
  errorPanel.hidden = true;
  errorPanel.textContent = "";
}

async function loadWasm() {
  const url = "./chimy2.wasm";
  if (WebAssembly.instantiateStreaming) {
    try {
      const response = await fetch(url);
      return (await WebAssembly.instantiateStreaming(response, {})).instance;
    } catch (error) {
      console.info("streaming wasm load failed; using array-buffer fallback", error);
    }
  }
  const response = await fetch(url);
  const bytes = await response.arrayBuffer();
  return (await WebAssembly.instantiate(bytes, {})).instance;
}

function resize(width, height) {
  canvas.width = width;
  canvas.height = height;
  imageData = context.createImageData(width, height);
}

function setMode(next) {
  mode = ((next % SHADER_MODES.length) + SHADER_MODES.length) % SHADER_MODES.length;
  const shader = SHADER_MODES[mode];
  modeTag.textContent = shader.tag;
  for (const button of picker.children) {
    button.dataset.active = String(Number(button.dataset.index) === mode);
  }
}

function stepMode(delta) {
  setMode(mode + delta);
  hint.classList.add("hide");
}

function resetOrbit() {
  yaw = 0;
  pitch = 0;
}

function drawFrame(time) {
  if (paused) return;
  const result = wasm.exports.render_frame(time, yaw, pitch, mode);
  if (result !== 0) {
    showError(`render failed (${result})`);
    return;
  }
  clearError();
  const pointer = wasm.exports.framebuffer_ptr();
  const length = wasm.exports.framebuffer_len();
  const pixels = new Uint8Array(wasm.exports.memory.buffer, pointer, length);
  imageData.data.set(pixels);
  context.putImageData(imageData, 0, 0);

  sessionFrames += 1;
  frameCount += 1;
  if (time - fpsTimestamp >= 500) {
    const fps = Math.round((frameCount * 1000) / (time - fpsTimestamp));
    const label = fps.toString().padStart(2, "0");
    fpsOut.textContent = label;
    fpsTag.textContent = label;
    frameCount = 0;
    fpsTimestamp = time;
  }
  if (sessionFrames % 3 === 0) {
    frameOut.textContent = sessionFrames.toString();
    frameTag.textContent = `FRAME ${sessionFrames}`;
    orbitOut.textContent = `${yaw.toFixed(2)} · ${pitch.toFixed(2)}`;
  }
}

function loop(time) {
  drawFrame(time);
  requestAnimationFrame(loop);
}

// ── pointer / keyboard input ─────────────────────────────────────
canvas.addEventListener("pointerdown", (event) => {
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
  hint.classList.add("hide");
});
canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.008;
  pitch += (event.clientY - lastY) * 0.006;
  pitch = Math.max(-1.35, Math.min(1.35, pitch));
  lastX = event.clientX;
  lastY = event.clientY;
});
function stopDragging(event) {
  dragging = false;
  canvas.classList.remove("dragging");
  if (event?.pointerId !== undefined) canvas.releasePointerCapture(event.pointerId);
}
canvas.addEventListener("pointerup", stopDragging);
canvas.addEventListener("pointercancel", stopDragging);
canvas.addEventListener("dblclick", () => resetOrbit());

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
  const digit = Number(event.key);
  if (Number.isInteger(digit) && digit >= 1 && digit <= SHADER_MODES.length) {
    setMode(digit - 1);
    return;
  }
  if (event.key === "[") { stepMode(-1); return; }
  if (event.key === "]") { stepMode(1); return; }
  if (event.key === "ArrowLeft") { selectScene(selectedSceneIndex - 1); return; }
  if (event.key === "ArrowRight") { selectScene(selectedSceneIndex + 1); return; }
  if (event.key.toLowerCase() === "r") { resetOrbit(); return; }
  if (event.key === " ") {
    event.preventDefault();
    paused = !paused;
    document.getElementById("live-dot").classList.toggle("off", paused);
  }
});

// ── shader picker ────────────────────────────────────────────────
function buildPicker() {
  const fragment = document.createDocumentFragment();
  SHADER_MODES.forEach((entry, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pill";
    button.role = "tab";
    button.dataset.index = String(index);
    button.dataset.active = String(index === mode);
    button.innerHTML = `
      <span class="num-tag">MODE ${String(index + 1).padStart(2, "0")}</span>
      <span class="title">${entry.name}</span>
      <span class="desc">${entry.desc}</span>
    `;
    button.addEventListener("click", () => {
      setMode(index);
      hint.classList.add("hide");
    });
    fragment.appendChild(button);
  });
  picker.appendChild(fragment);
}

// ── scene gallery ────────────────────────────────────────────────
function buildGallery() {
  const fragment = document.createDocumentFragment();
  SCENES.forEach((scene, index) => {
    const tile = document.createElement("button");
    tile.type = "button";
    tile.className = "tile";
    tile.role = "option";
    tile.dataset.index = String(index);
    tile.dataset.active = String(index === selectedSceneIndex);
    const idxLabel = String(index + 1).padStart(2, "0");
    const tagText = scene.features.slice(0, 3).map((f) => `<span>${f}</span>`).join("");
    tile.innerHTML = `
      <div class="thumb"><img src="${scene.image}" alt="${scene.name} scene" loading="lazy"></div>
      <div class="body">
        <div class="head">
          <span class="name">${scene.name}</span>
          <span class="idx-tag">${idxLabel}</span>
        </div>
        <span class="desc">${scene.desc}</span>
        <div class="tags">${tagText}</div>
      </div>
    `;
    tile.addEventListener("click", () => selectScene(index));
    fragment.appendChild(tile);
  });
  gallery.appendChild(fragment);
}

function highlightJson(source) {
  const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let out = escape(source);
  out = out.replace(/(&quot;[^&]*?&quot;)(\s*:)/g, (_, key, tail) => `<span class="k">${key}</span>${tail}`);
  out = out.replace(/:\s*(&quot;[^&]*?&quot;)/g, (m, str) => `: <span class="s">${str}</span>`);
  out = out.replace(/(?<![\w"])(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/gi, '<span class="n">$1</span>');
  return out;
}

async function selectScene(nextIndex) {
  const wrapped = ((nextIndex % SCENES.length) + SCENES.length) % SCENES.length;
  selectedSceneIndex = wrapped;
  const scene = SCENES[wrapped];
  for (const tile of gallery.children) {
    tile.dataset.active = String(Number(tile.dataset.index) === wrapped);
  }
  detailImg.src = scene.image;
  detailImg.alt = `${scene.name} — rendered 960×640 by scene_viewer`;
  detailCap.textContent = `${scene.name.toUpperCase()} · 960 × 640 · native`;
  detailTitle.textContent = scene.name;
  detailDesc.textContent = scene.desc;
  detailFeatures.innerHTML = scene.features.map((f) => `<li>${f}</li>`).join("");
  openJsonBtn.href = scene.file;
  detailCode.innerHTML = "<em style=\"color: var(--muted);\">Loading scene JSON…</em>";
  try {
    const response = await fetch(scene.file);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    currentSceneJson = text;
    detailCode.innerHTML = highlightJson(text);
  } catch (error) {
    console.warn(`failed to fetch ${scene.file}`, error);
    detailCode.textContent = `// failed to load ${scene.file}\n// ${error.message}`;
  }
}

copyJsonBtn.addEventListener("click", async () => {
  if (!currentSceneJson) return;
  try {
    await navigator.clipboard.writeText(currentSceneJson);
    copyJsonBtn.textContent = "Copied";
    setTimeout(() => { copyJsonBtn.textContent = "Copy JSON"; }, 1400);
  } catch {
    copyJsonBtn.textContent = "Copy failed";
    setTimeout(() => { copyJsonBtn.textContent = "Copy JSON"; }, 1400);
  }
});

// ── boot ────────────────────────────────────────────────────────
buildPicker();
buildGallery();
selectScene(0);
setMode(0);

const now = new Date();
footerTime.textContent = now.toISOString().slice(0, 16).replace("T", " ");

try {
  const instance = await loadWasm();
  wasm = instance;
  const initResult = wasm.exports.init(960, 640);
  if (initResult !== 0) throw new Error(`init failed (${initResult})`);
  resize(wasm.exports.framebuffer_width(), wasm.exports.framebuffer_height());
  fpsTimestamp = performance.now();
  requestAnimationFrame(loop);
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => hint.classList.add("hide"), 4600);
} catch (error) {
  showError(error.message || String(error));
  console.error(error);
}
