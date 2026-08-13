import { useEffect, useRef, useState } from "react";
import { useRenderer, type OrbitRef } from "../hooks/useRenderer";
import { DEMOS } from "../data/demos";

const CLAMP_PITCH = 1.35;

type Props = {
  index: number;
};

export function Stage({ index }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const orbitRef = useRef<OrbitRef>({ yaw: 0, pitch: 0 });
  const dragRef = useRef({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const demo = DEMOS[index];

  const { state, telemetry } = useRenderer(canvasRef, {
    sceneFile: demo.file,
    paused: false,
    orbit: orbitRef,
    onError: setErrorMessage,
  });

  useEffect(() => {
    const reset = () => {
      orbitRef.current.yaw = 0;
      orbitRef.current.pitch = 0;
    };
    window.addEventListener("chimy2:reset-orbit", reset);
    return () => window.removeEventListener("chimy2:reset-orbit", reset);
  }, []);

  useEffect(() => {
    setErrorMessage(null);
  }, [demo.file]);

  const ready = state.status === "ready";

  return (
    <figure className={`stage${dragging ? " stage--dragging" : ""}`}>
      <figcaption className="caption">
        <span className={`caption__dot${ready ? "" : " caption__dot--off"}`} aria-hidden />
        <span className="caption__idx">{String(index + 1).padStart(2, "0")}</span>
        <span className="caption__name">{demo.name}</span>
        <span className="caption__sep" aria-hidden>·</span>
        <span className="caption__meta">{demo.tag}</span>
        <span className="caption__spacer" aria-hidden />
        <span className="caption__stats tabular">
          {telemetry.fps.toString().padStart(2, "0")} fps · 960 × 640
        </span>
      </figcaption>

      <div className="plinth">
        <canvas
          ref={canvasRef}
          className="plinth__canvas"
          width={960}
          height={640}
          aria-label={`live CPU rasterizer — ${demo.name} (drag to orbit)`}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            setDragging(true);
            dragRef.current = { x: event.clientX, y: event.clientY };
          }}
          onPointerMove={(event) => {
            if (!dragging) return;
            const dx = event.clientX - dragRef.current.x;
            const dy = event.clientY - dragRef.current.y;
            dragRef.current = { x: event.clientX, y: event.clientY };
            orbitRef.current.yaw += dx * 0.008;
            orbitRef.current.pitch = clamp(
              orbitRef.current.pitch + dy * 0.006,
              -CLAMP_PITCH,
              CLAMP_PITCH,
            );
          }}
          onPointerUp={(event) => {
            try {
              event.currentTarget.releasePointerCapture(event.pointerId);
            } catch {
              // pointer may already be released; ignore
            }
            setDragging(false);
          }}
          onPointerCancel={() => setDragging(false)}
          onDoubleClick={() => {
            orbitRef.current.yaw = 0;
            orbitRef.current.pitch = 0;
          }}
        />
        {state.status === "loading" && (
          <div className="plinth__overlay" role="status">
            <span className="spinner" />
            <span className="plinth__overlay-text">booting wasm</span>
          </div>
        )}
        {state.status === "error" && (
          <div className="plinth__overlay plinth__overlay--error" role="alert">
            <span>{errorMessage ?? state.message}</span>
            <span className="plinth__overlay-hint">
              Serve <code>dist/</code> over HTTP so the browser can fetch the wasm.
            </span>
          </div>
        )}
      </div>
    </figure>
  );
}

function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}
