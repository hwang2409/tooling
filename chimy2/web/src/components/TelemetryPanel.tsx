import type { Telemetry } from "../hooks/useRenderer";

type Props = {
  telemetry: Telemetry;
  paused: boolean;
  modeTag: string;
  onTogglePause: () => void;
  onReset: () => void;
  status: "loading" | "ready" | "error";
};

export function TelemetryPanel(props: Props) {
  const { telemetry } = props;
  return (
    <aside className="telemetry" aria-label="renderer telemetry">
      <div className="telemetry__card">
        <h3 className="telemetry__title">Frame</h3>
        <dl className="kv">
          <dt>Resolution</dt>
          <dd className="tabular">960 × 640</dd>
          <dt>FPS</dt>
          <dd className="tabular kv__accent">
            {telemetry.fps.toString().padStart(2, "0")}
          </dd>
          <dt>Frame</dt>
          <dd className="tabular">{telemetry.frame}</dd>
          <dt>Yaw · Pitch</dt>
          <dd className="tabular">
            {telemetry.yaw.toFixed(2)} · {telemetry.pitch.toFixed(2)}
          </dd>
          <dt>Mode</dt>
          <dd className="kv__alt">{props.modeTag}</dd>
          <dt>Status</dt>
          <dd className={statusClass(props.status, props.paused)}>
            {statusLabel(props.status, props.paused)}
          </dd>
        </dl>
      </div>

      <div className="telemetry__card">
        <h3 className="telemetry__title">Shortcuts</h3>
        <div className="keys">
          <div>
            <span>Cycle shader</span>
            <span>
              <kbd>1</kbd>–<kbd>8</kbd>
            </span>
          </div>
          <div>
            <span>Prev / next mode</span>
            <span>
              <kbd>[</kbd> <kbd>]</kbd>
            </span>
          </div>
          <div>
            <span>Prev / next scene</span>
            <span>
              <kbd>←</kbd> <kbd>→</kbd>
            </span>
          </div>
          <div>
            <span>Reset orbit</span>
            <span>
              <kbd>R</kbd>
            </span>
          </div>
          <div>
            <span>Pause loop</span>
            <span>
              <kbd>Space</kbd>
            </span>
          </div>
        </div>
      </div>

      <div className="telemetry__actions">
        <button type="button" className="btn" onClick={props.onTogglePause}>
          {props.paused ? "Resume" : "Pause"}
        </button>
        <button type="button" className="btn" onClick={props.onReset}>
          Reset orbit
        </button>
      </div>
    </aside>
  );
}

function statusLabel(status: Props["status"], paused: boolean) {
  if (status === "loading") return "Booting";
  if (status === "error") return "Halted";
  return paused ? "Paused" : "Live";
}

function statusClass(status: Props["status"], paused: boolean) {
  if (status === "error") return "kv__hot";
  if (paused) return "kv__muted";
  if (status === "loading") return "kv__muted";
  return "kv__accent";
}
