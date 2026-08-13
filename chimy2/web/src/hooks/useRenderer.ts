import { useEffect, useRef, useState } from "react";
import { loadWasm, type Chimy2Instance } from "../wasm/loadWasm";

export type RendererState =
  | { status: "loading" }
  | { status: "loading-scene"; scene: string }
  | { status: "ready"; width: number; height: number }
  | { status: "error"; message: string };

export type Telemetry = {
  fps: number;
  frame: number;
  yaw: number;
  pitch: number;
};

export type OrbitRef = {
  yaw: number;
  pitch: number;
};

const RUNTIME_WASM_URL = "./chimy2.wasm";

// Total-pixel safety cap for the CPU rasterizer on huge windows. Above this,
// we scale both dimensions down proportionally and let CSS smooth-upscale.
const PIXEL_BUDGET = 1_300_000;
// Wasm-side MAX_DIMENSION guard (see chimy2/src/wasm_api.rs).
const MAX_DIMENSION = 2048;
// Fallback size if the plinth element has no laid-out size at boot.
const FALLBACK_WIDTH = 960;
const FALLBACK_HEIGHT = 640;
// Debounce window for ResizeObserver callbacks.
const RESIZE_DEBOUNCE_MS = 300;
// Minimum dimension delta (px) that triggers a re-init.
const RESIZE_EPSILON = 32;

function measureAndClamp(rect: { width: number; height: number }): [number, number] {
  // CPU rasterizer: pin devicePixelRatio to 1 (never render retina).
  let w = Math.max(1, Math.floor(rect.width));
  let h = Math.max(1, Math.floor(rect.height));
  if (w < 2 || h < 2) {
    w = FALLBACK_WIDTH;
    h = FALLBACK_HEIGHT;
  }
  const area = w * h;
  if (area > PIXEL_BUDGET) {
    const s = Math.sqrt(PIXEL_BUDGET / area);
    w = Math.max(1, Math.floor(w * s));
    h = Math.max(1, Math.floor(h * s));
  }
  if (w > MAX_DIMENSION) w = MAX_DIMENSION;
  if (h > MAX_DIMENSION) h = MAX_DIMENSION;
  return [w, h];
}

export function useRenderer(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  options: {
    sceneFile: string;
    paused: boolean;
    orbit: React.MutableRefObject<OrbitRef>;
    onError?: (message: string) => void;
  },
) {
  const [state, setState] = useState<RendererState>({ status: "loading" });
  const [telemetry, setTelemetry] = useState<Telemetry>({
    fps: 0,
    frame: 0,
    yaw: 0,
    pitch: 0,
  });
  const [bootTick, setBootTick] = useState(0);
  const instanceRef = useRef<Chimy2Instance | null>(null);
  const sceneReadyRef = useRef(false);
  const lastLoadedSceneRef = useRef<string | null>(null);

  const pausedRef = useRef(options.paused);
  useEffect(() => {
    pausedRef.current = options.paused;
  }, [options.paused]);

  useEffect(() => {
    let cancelled = false;
    let rafId = 0;
    let imageData: ImageData | null = null;
    let ctx: CanvasRenderingContext2D | null = null;
    let instance: Chimy2Instance | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;
    let currentWidth = 0;
    let currentHeight = 0;
    let frameCount = 0;
    let sessionFrames = 0;
    let fpsTimestamp = 0;
    let lastTelemetry = 0;

    const canvas = canvasRef.current;
    const plinth = canvas?.parentElement ?? null;
    if (!canvas) {
      setState({ status: "error", message: "canvas element missing" });
      return;
    }

    const applySize = (width: number, height: number) => {
      if (!instance) return;
      const rc = instance.exports.init(width, height);
      if (rc !== 0) throw new Error(`init failed (${rc})`);
      const actualW = instance.exports.framebuffer_width();
      const actualH = instance.exports.framebuffer_height();
      canvas.width = actualW;
      canvas.height = actualH;
      if (!ctx) {
        ctx = canvas.getContext("2d", { alpha: false });
        if (!ctx) throw new Error("2d context unavailable");
      }
      imageData = ctx.createImageData(actualW, actualH);
      currentWidth = actualW;
      currentHeight = actualH;
    };

    const measureTarget = (): [number, number] => {
      const rect = (plinth ?? canvas).getBoundingClientRect();
      return measureAndClamp(rect);
    };

    (async () => {
      try {
        instance = await loadWasm(RUNTIME_WASM_URL);
        if (cancelled) return;
        const [w, h] = measureTarget();
        applySize(w, h);
        instanceRef.current = instance;
        setBootTick((n) => n + 1);
        fpsTimestamp = performance.now();

        const loop = (time: number) => {
          if (cancelled) return;
          if (
            !pausedRef.current &&
            instanceRef.current &&
            imageData &&
            ctx &&
            sceneReadyRef.current
          ) {
            const result = instanceRef.current.exports.render_frame(
              time,
              options.orbit.current.yaw,
              options.orbit.current.pitch,
              0,
            );
            if (result !== 0) {
              options.onError?.(`render failed (${result})`);
            } else {
              const ptr = instanceRef.current.exports.framebuffer_ptr();
              const len = instanceRef.current.exports.framebuffer_len();
              const view = new Uint8Array(
                instanceRef.current.exports.memory.buffer,
                ptr,
                len,
              );
              imageData.data.set(view);
              ctx.putImageData(imageData, 0, 0);
              sessionFrames += 1;
              frameCount += 1;

              if (time - fpsTimestamp >= 500) {
                const fps = Math.round((frameCount * 1000) / (time - fpsTimestamp));
                frameCount = 0;
                fpsTimestamp = time;
                setTelemetry({
                  fps,
                  frame: sessionFrames,
                  yaw: options.orbit.current.yaw,
                  pitch: options.orbit.current.pitch,
                });
                lastTelemetry = time;
              } else if (time - lastTelemetry >= 100) {
                setTelemetry((prev) => ({
                  ...prev,
                  frame: sessionFrames,
                  yaw: options.orbit.current.yaw,
                  pitch: options.orbit.current.pitch,
                }));
                lastTelemetry = time;
              }
            }
          }
          rafId = requestAnimationFrame(loop);
        };
        rafId = requestAnimationFrame(loop);

        // React to plinth resizes: debounce, then re-init the wasm at the new
        // size and let the scene-load effect fire again to restore state.
        if (plinth && typeof ResizeObserver !== "undefined") {
          resizeObserver = new ResizeObserver(() => {
            if (debounceTimer !== null) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
              debounceTimer = null;
              if (cancelled || !instance) return;
              const [nextW, nextH] = measureTarget();
              if (
                Math.abs(nextW - currentWidth) < RESIZE_EPSILON &&
                Math.abs(nextH - currentHeight) < RESIZE_EPSILON
              ) {
                return;
              }
              try {
                sceneReadyRef.current = false;
                applySize(nextW, nextH);
                // Bumping bootTick re-runs the scene-load effect, which
                // re-issues load_scene_json against the freshly re-inited wasm.
                setBootTick((n) => n + 1);
              } catch (error) {
                const message = error instanceof Error ? error.message : String(error);
                setState({ status: "error", message });
                options.onError?.(message);
              }
            }, RESIZE_DEBOUNCE_MS);
          });
          resizeObserver.observe(plinth);
        }
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : String(error);
        setState({ status: "error", message });
        options.onError?.(message);
      }
    })();

    return () => {
      cancelled = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (debounceTimer !== null) clearTimeout(debounceTimer);
      resizeObserver?.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load the requested scene JSON into the wasm instance whenever the scene
  // changes (or the wasm just finished booting / was re-inited on resize).
  // Sequence: fetch(json) -> scene_alloc(len) -> copy bytes -> load_scene_json.
  // Scene mesh assets are embedded via `include_str!` fallback in
  // scene/assets.rs, so no browser-side asset fetching beyond the JSON.
  useEffect(() => {
    if (bootTick === 0) return;
    const instance = instanceRef.current;
    if (!instance) return;
    let cancelled = false;
    sceneReadyRef.current = false;
    setState({ status: "loading-scene", scene: options.sceneFile });
    (async () => {
      try {
        const response = await fetch(options.sceneFile);
        if (!response.ok) {
          throw new Error(`fetch ${options.sceneFile}: HTTP ${response.status}`);
        }
        const bytes = new Uint8Array(await response.arrayBuffer());
        if (cancelled) return;
        const ptr = instance.exports.scene_alloc(bytes.length);
        if (ptr === 0) throw new Error("scene_alloc returned null");
        const target = new Uint8Array(instance.exports.memory.buffer, ptr, bytes.length);
        target.set(bytes);
        const rc = instance.exports.load_scene_json(ptr, bytes.length);
        if (rc !== 0) throw new Error(`load_scene_json rc=${rc}`);
        if (cancelled) return;
        // Only reset orbit on a real scene change; resize-triggered reloads
        // preserve the user's current camera.
        if (lastLoadedSceneRef.current !== options.sceneFile) {
          options.orbit.current.yaw = 0;
          options.orbit.current.pitch = 0;
        }
        lastLoadedSceneRef.current = options.sceneFile;
        sceneReadyRef.current = true;
        const width = instance.exports.framebuffer_width();
        const height = instance.exports.framebuffer_height();
        setState({ status: "ready", width, height });
      } catch (error) {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : String(error);
        setState({ status: "error", message });
        options.onError?.(message);
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options.sceneFile, bootTick]);

  return { state, telemetry };
}
