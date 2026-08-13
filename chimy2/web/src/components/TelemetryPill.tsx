import { useEffect, useState } from "react";
import { SHADER_MODES } from "../data/modes";

type Props = {
  fps: number;
  frame: number;
  mode: number;
};

export function TelemetryPill(props: Props) {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const handler = () => setVisible(window.scrollY > window.innerHeight * 0.8);
    handler();
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const mode = SHADER_MODES[props.mode];

  return (
    <a
      href="#live"
      className={`pill-hud${visible ? " pill-hud--visible" : ""}`}
      aria-label="Return to live viewport"
    >
      <span className="pill-hud__dot" />
      <span className="pill-hud__slot">
        <span className="pill-hud__key">FPS</span>
        <span className="pill-hud__val tabular">
          {props.fps.toString().padStart(2, "0")}
        </span>
      </span>
      <span className="pill-hud__sep" />
      <span className="pill-hud__slot">
        <span className="pill-hud__key">FRAME</span>
        <span className="pill-hud__val tabular">{props.frame}</span>
      </span>
      <span className="pill-hud__sep" />
      <span className="pill-hud__slot">
        <span className="pill-hud__key">MODE</span>
        <span className="pill-hud__val">{mode.tag}</span>
      </span>
    </a>
  );
}
