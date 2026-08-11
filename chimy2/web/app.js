const canvas = document.querySelector("#canvas");
const context = canvas.getContext("2d", { alpha: false });
const modeSelect = document.querySelector("#mode");
const fpsReadout = document.querySelector("#fps");
const errorReadout = document.querySelector("#error");

let wasm;
let imageData;
let yaw = 0;
let pitch = 0;
let dragging = false;
let lastX = 0;
let lastY = 0;
let lastFrame = performance.now();
let frameCount = 0;
let fpsTimestamp = lastFrame;

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

function showError(message) {
  errorReadout.textContent = message;
}

function resize(width, height) {
  canvas.width = width;
  canvas.height = height;
  imageData = context.createImageData(width, height);
}

function drawFrame(time) {
  const result = wasm.exports.render_frame(
    time,
    yaw,
    pitch,
    Number(modeSelect.value),
  );
  if (result !== 0) {
    showError(`render failed (${result})`);
    return;
  }
  const pointer = wasm.exports.framebuffer_ptr();
  const length = wasm.exports.framebuffer_len();
  const pixels = new Uint8Array(wasm.exports.memory.buffer, pointer, length);
  imageData.data.set(pixels);
  context.putImageData(imageData, 0, 0);

  frameCount += 1;
  if (time - fpsTimestamp >= 500) {
    fpsReadout.textContent = `${Math.round(frameCount * 1000 / (time - fpsTimestamp))} fps`;
    frameCount = 0;
    fpsTimestamp = time;
  }
  lastFrame = time;
}

function loop(time) {
  drawFrame(time);
  requestAnimationFrame(loop);
}

canvas.addEventListener("pointerdown", (event) => {
  dragging = true;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.classList.add("dragging");
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  if (!dragging) return;
  yaw += (event.clientX - lastX) * 0.008;
  pitch += (event.clientY - lastY) * 0.006;
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

try {
  const instance = await loadWasm();
  wasm = instance;
  const width = wasm.exports.framebuffer_width ? 640 : canvas.width;
  const height = wasm.exports.framebuffer_height ? 480 : canvas.height;
  const result = wasm.exports.init(width, height);
  if (result !== 0) throw new Error(`init failed (${result})`);
  resize(wasm.exports.framebuffer_width(), wasm.exports.framebuffer_height());
  requestAnimationFrame(loop);
} catch (error) {
  showError(error.message);
  console.error(error);
}
