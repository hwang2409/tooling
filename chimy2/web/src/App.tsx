import { useCallback, useEffect, useRef, useState } from "react";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { LiveViewport } from "./components/LiveViewport";
import { SceneGallery } from "./components/SceneGallery";
import { SceneDetail } from "./components/SceneDetail";
import { About } from "./components/About";
import { Footer } from "./components/Footer";
import { TelemetryPill } from "./components/TelemetryPill";
import { SHADER_MODES } from "./data/modes";
import { SCENES } from "./data/scenes";
import { useSceneRoute } from "./hooks/useSceneRoute";

export default function App() {
  const [mode, setMode] = useState(0);
  const [paused, setPaused] = useState(false);
  const [sceneIndex, setSceneIndex] = useSceneRoute(0);
  const [telemetry, setTelemetry] = useState({ fps: 0, frame: 0, mode: 0 });
  const detailAnchorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        (target instanceof HTMLElement && target.isContentEditable)
      ) {
        return;
      }
      const digit = Number(event.key);
      if (Number.isInteger(digit) && digit >= 1 && digit <= SHADER_MODES.length) {
        setMode(digit - 1);
        return;
      }
      if (event.key === "[") {
        setMode((m) => (m - 1 + SHADER_MODES.length) % SHADER_MODES.length);
        return;
      }
      if (event.key === "]") {
        setMode((m) => (m + 1) % SHADER_MODES.length);
        return;
      }
      if (event.key === "ArrowLeft") {
        setSceneIndex(sceneIndex - 1);
        return;
      }
      if (event.key === "ArrowRight") {
        setSceneIndex(sceneIndex + 1);
        return;
      }
      if (event.key.toLowerCase() === "r") {
        // Reset dispatched via a custom event so the LiveViewport orbit ref is the
        // only source of truth for orbit state — no lifted state, no rerender.
        window.dispatchEvent(new CustomEvent("chimy2:reset-orbit"));
        return;
      }
      if (event.key === " ") {
        event.preventDefault();
        setPaused((p) => !p);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sceneIndex, setSceneIndex]);

  const onTelemetry = useCallback(
    (next: { fps: number; frame: number; mode: number }) => {
      setTelemetry((prev) =>
        prev.fps === next.fps && prev.frame === next.frame && prev.mode === next.mode
          ? prev
          : next,
      );
    },
    [],
  );

  const goToScene = useCallback(
    (index: number) => {
      const wrapped =
        ((index % SCENES.length) + SCENES.length) % SCENES.length;
      setSceneIndex(wrapped);
      requestAnimationFrame(() => {
        detailAnchorRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      });
    },
    [setSceneIndex],
  );

  return (
    <>
      <Header />
      <main className="page">
        <Hero />

        <LiveViewport
          mode={mode}
          onModeChange={setMode}
          paused={paused}
          onPausedChange={setPaused}
          onTelemetryUpdate={onTelemetry}
        />

        <section id="scenes" className="scenes">
          <div className="section-head">
            <span className="section-head__idx">02</span>
            <h2 className="section-head__title">
              Scene format · Curated gallery
            </h2>
            <span className="section-head__meta">
              rendered natively · <span className="tabular">960 × 640</span>
            </span>
          </div>
          <p className="scenes__lede">
            These eight scenes are authored in chimy2's strict JSON scene format
            (<code>docs/scene.md</code>). Every camera, light, mesh, shadow map,
            particle emitter, HUD glyph and post effect is described by data —
            then rendered by the same pipeline the browser runs above.
          </p>

          <SceneGallery selectedIndex={sceneIndex} onSelect={goToScene} />

          <div ref={detailAnchorRef} aria-hidden />

          <SceneDetail
            index={sceneIndex}
            onPrev={() => setSceneIndex(sceneIndex - 1)}
            onNext={() => setSceneIndex(sceneIndex + 1)}
          />
        </section>

        <About />
      </main>
      <Footer />

      <TelemetryPill
        fps={telemetry.fps}
        frame={telemetry.frame}
        mode={mode}
      />
    </>
  );
}
