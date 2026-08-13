import { useEffect, useRef, useState } from "react";
import { loadWasm, type Chimy2Instance } from "../wasm/loadWasm";

export type RendererState =
  | { status: "loading" }
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

// Vite serves public/ contents from the site root in both dev and production,
// so a document-relative URL works in both environments.
const RUNTIME_WASM_URL = "./chimy2.wasm";

export function useRenderer(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  options: {
    mode: number;
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
  const instanceRef = useRef<Chimy2Instance | null>(null);

  // Keep the mutable options in refs so the RAF loop reads current values
  // without needing to be torn down and re-established on every render.
  const modeRef = useRef(options.mode);
  const pausedRef = useRef(options.paused);
  useEffect(() => {
    modeRef.current = options.mode;
  }, [options.mode]);
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
        setState({ status: "ready", width, height });
        fpsTimestamp = performance.now();

        const loop = (time: number) => {
          if (cancelled) return;
          if (!pausedRef.current && instanceRef.current && imageData && ctx) {
            const result = instanceRef.current.exports.render_frame(
              time,
              options.orbit.current.yaw,
              options.orbit.current.pitch,
              modeRef.current,
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
                // Cheap frame/orbit refresh at ~10 Hz so the readout tracks the
                // pointer without hitting React on every RAF tick.
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

  return { state, telemetry };
}
