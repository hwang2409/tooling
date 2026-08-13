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

  const pausedRef = useRef(options.paused);
  useEffect(() => {
    pausedRef.current = options.paused;
  }, [options.paused]);

  useEffect(() => {
    let cancelled = false;
    let rafId = 0;
    let imageData: ImageData | null = null;
    let ctx: CanvasRenderingContext2D | null = null;
    let frameCount = 0;
    let sessionFrames = 0;
    let fpsTimestamp = 0;
    let lastTelemetry = 0;

    (async () => {
      try {
        const instance = await loadWasm(RUNTIME_WASM_URL);
        if (cancelled) return;
        const initResult = instance.exports.init(960, 640);
        if (initResult !== 0) throw new Error(`init failed (${initResult})`);
        const width = instance.exports.framebuffer_width();
        const height = instance.exports.framebuffer_height();
        const canvas = canvasRef.current;
        if (!canvas) throw new Error("canvas element missing");
        canvas.width = width;
        canvas.height = height;
        ctx = canvas.getContext("2d", { alpha: false });
        if (!ctx) throw new Error("2d context unavailable");
        imageData = ctx.createImageData(width, height);
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
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Load the requested scene JSON into the wasm instance whenever the scene
  // changes (or the wasm just finished booting). Sequence:
  //   fetch(json) -> scene_alloc(len) -> copy bytes -> load_scene_json(ptr,len)
  // The wasm's scene mesh assets are embedded via the Rust `include_str!`
  // fallback in scene/assets.rs, so `render_frame` returns rc=0 for every
  // shipped scene without any browser-side asset fetching beyond the JSON.
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
        options.orbit.current.yaw = 0;
        options.orbit.current.pitch = 0;
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
