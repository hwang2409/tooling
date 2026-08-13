import { useEffect, useRef, useState } from "react";
import { useRenderer, type OrbitRef } from "../hooks/useRenderer";
import { SHADER_MODES } from "../data/modes";
import { ModePicker } from "./ModePicker";
import { TelemetryPanel } from "./TelemetryPanel";

const CLAMP_PITCH = 1.35;

type Props = {
  mode: number;
  onModeChange: (next: number) => void;
  paused: boolean;
  onPausedChange: (next: boolean) => void;
  onTelemetryUpdate?: (telemetry: {
    fps: number;
    frame: number;
    mode: number;
  }) => void;
};

export function LiveViewport(props: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const orbitRef = useRef<OrbitRef>({ yaw: 0, pitch: 0 });
  const [dragging, setDragging] = useState(false);
  const [hintShown, setHintShown] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const { state, telemetry } = useRenderer(canvasRef, {
    mode: props.mode,
    paused: props.paused,
    orbit: orbitRef,
    onError: setErrorMessage,
  });

  useEffect(() => {
    if (!hintShown) return;
    const id = window.setTimeout(() => setHintShown(false), 4600);
    return () => window.clearTimeout(id);
  }, [hintShown]);

  useEffect(() => {
    const reset = () => {
      orbitRef.current.yaw = 0;
      orbitRef.current.pitch = 0;
    };
    window.addEventListener("chimy2:reset-orbit", reset);
    return () => window.removeEventListener("chimy2:reset-orbit", reset);
  }, []);

  useEffect(() => {
    props.onTelemetryUpdate?.({
      fps: telemetry.fps,
      frame: telemetry.frame,
      mode: props.mode,
    });
  }, [telemetry.fps, telemetry.frame, props.mode, props.onTelemetryUpdate]);

  const activeMode = SHADER_MODES[props.mode];

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(true);
    setHintShown(false);
    dragStateRef.current = { x: event.clientX, y: event.clientY };
  };

  const dragStateRef = useRef({ x: 0, y: 0 });

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragging) return;
    const dx = event.clientX - dragStateRef.current.x;
    const dy = event.clientY - dragStateRef.current.y;
    dragStateRef.current = { x: event.clientX, y: event.clientY };
    orbitRef.current.yaw += dx * 0.008;
    orbitRef.current.pitch = clamp(
      orbitRef.current.pitch + dy * 0.006,
      -CLAMP_PITCH,
      CLAMP_PITCH,
    );
  };

  const releasePointer = (event: React.PointerEvent<HTMLCanvasElement>) => {
    try {
      event.currentTarget.releasePointerCapture(event.pointerId);
    } catch {
      // release can throw if the pointer is already gone; ignore.
    }
    setDragging(false);
  };

  const resetOrbit = () => {
    orbitRef.current.yaw = 0;
    orbitRef.current.pitch = 0;
  };

  return (
    <section id="live" className="live">
      <SectionHead index="01" title="Live · rasterized on your CPU" meta="960 × 640 · rgba8" />
      <div className="live__grid">
        <div className={`stage${dragging ? " stage--dragging" : ""}${state.status === "error" ? " stage--error" : ""}`}>
          <div className="stage__scanlines" aria-hidden />
          <div className="stage__vignette" aria-hidden />

          <div className="stage__corner stage__corner--tl">
            <span className={`stage__dot${props.paused ? " stage__dot--off" : ""}`} />
            <span>chimy2 / wasm</span>
          </div>
          <div className="stage__corner stage__corner--tr">
            <span className="tabular">
              {telemetry.fps.toString().padStart(2, "0")}
            </span>
            <span className="stage__corner-unit">FPS</span>
          </div>
          <div className="stage__corner stage__corner--bl">
            <span className="stage__corner-mode">{activeMode.tag}</span>
          </div>
          <div className="stage__corner stage__corner--br">
            <span className="tabular">FRAME {telemetry.frame}</span>
          </div>

          <canvas
            ref={canvasRef}
            className="stage__canvas"
            width={960}
            height={640}
            aria-label="live CPU rasterizer output"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={releasePointer}
            onPointerCancel={releasePointer}
            onDoubleClick={resetOrbit}
          />

          {state.status === "loading" && (
            <div className="stage__overlay stage__overlay--loading" role="status">
              <span className="stage__loader" />
              <span>Booting wasm · compiling shaders</span>
            </div>
          )}
          {state.status === "error" && (
            <div className="stage__overlay stage__overlay--error" role="alert">
              <span className="stage__loader stage__loader--error" />
              <span>{errorMessage ?? state.message}</span>
              <span className="stage__error-hint">
                Serve the built <code>dist/</code> over HTTP so the browser can fetch the wasm.
              </span>
            </div>
          )}

          <div className={`stage__hint${hintShown && state.status === "ready" ? "" : " stage__hint--hidden"}`}>
            Drag to orbit · Keys <kbd>1</kbd>…<kbd>8</kbd>
          </div>
        </div>

        <TelemetryPanel
          telemetry={telemetry}
          paused={props.paused}
          modeTag={activeMode.tag}
          onTogglePause={() => props.onPausedChange(!props.paused)}
          onReset={resetOrbit}
          status={state.status}
        />
      </div>

      <ModePicker mode={props.mode} onChange={props.onModeChange} />
    </section>
  );
}

function SectionHead(props: { index: string; title: string; meta: string }) {
  return (
    <div className="section-head">
      <span className="section-head__idx">{props.index}</span>
      <h2 className="section-head__title">{props.title}</h2>
      <span className="section-head__meta">{props.meta}</span>
    </div>
  );
}

function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}
