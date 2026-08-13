import { useEffect, useRef, useState } from "react";
import { SCENES } from "../data/scenes";
import { useSceneJson } from "../hooks/useSceneJson";
import { highlightJson } from "../highlight";

type Props = {
  index: number;
  onPrev: () => void;
  onNext: () => void;
};

export function SceneDetail(props: Props) {
  const scene = SCENES[props.index];
  const json = useSceneJson(props.index);
  const [copyState, setCopyState] = useState<"idle" | "copied" | "failed">("idle");
  const copyTimer = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (copyTimer.current !== null) window.clearTimeout(copyTimer.current);
    };
  }, []);

  useEffect(() => {
    setCopyState("idle");
  }, [scene.id]);

  const handleCopy = async () => {
    if (json.status !== "ready") return;
    try {
      await navigator.clipboard.writeText(json.text);
      setCopyState("copied");
    } catch {
      setCopyState("failed");
    }
    if (copyTimer.current !== null) window.clearTimeout(copyTimer.current);
    copyTimer.current = window.setTimeout(() => setCopyState("idle"), 1600);
  };

  const idxLabel = String(props.index + 1).padStart(2, "0");
  const lineCount =
    json.status === "ready" ? json.text.split("\n").length : null;

  return (
    <article className="detail" aria-labelledby={`scene-${scene.id}-title`}>
      <div className="detail__lens">
        <img src={scene.image} alt={`${scene.name} — rendered 960×640 by scene_viewer`} />
        <div className="detail__lens-caption">
          <span>{scene.name.toUpperCase()}</span>
          <span className="detail__lens-caption-sep">·</span>
          <span className="tabular">960 × 640</span>
          <span className="detail__lens-caption-sep">·</span>
          <span>native</span>
        </div>
        <button
          type="button"
          className="detail__lens-nav detail__lens-nav--prev"
          aria-label="Previous scene"
          onClick={props.onPrev}
        >
          ←
        </button>
        <button
          type="button"
          className="detail__lens-nav detail__lens-nav--next"
          aria-label="Next scene"
          onClick={props.onNext}
        >
          →
        </button>
      </div>

      <div className="detail__about">
        <div className="detail__eyebrow">
          <span className="tabular">SCENE {idxLabel}</span>
          <span>·</span>
          <span>{scene.id}</span>
        </div>
        <h3 className="detail__title" id={`scene-${scene.id}-title`}>
          {scene.name}
        </h3>
        <p className="detail__desc">{scene.desc}</p>
        <ul className="detail__features">
          {scene.features.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
        <div className="detail__cta">
          <button type="button" className="btn" onClick={handleCopy}>
            {copyState === "copied"
              ? "Copied"
              : copyState === "failed"
                ? "Copy failed"
                : "Copy JSON"}
          </button>
          <a
            className="btn"
            href={scene.file}
            target="_blank"
            rel="noreferrer noopener"
          >
            Open file ↗
          </a>
        </div>
      </div>

      <div className="detail__code">
        <div className="detail__code-head">
          <span className="detail__code-title">
            <span className="tabular">{scene.file.replace("./", "web/public/")}</span>
          </span>
          <span className="detail__code-meta tabular">
            {lineCount !== null ? `${lineCount} lines` : "loading…"}
          </span>
        </div>
        {json.status === "ready" ? (
          <pre
            className="detail__code-pre"
            aria-label="scene json"
            dangerouslySetInnerHTML={{ __html: highlightJson(json.text) }}
          />
        ) : json.status === "loading" ? (
          <pre className="detail__code-pre detail__code-pre--muted">
            <em>Loading scene JSON…</em>
          </pre>
        ) : (
          <pre className="detail__code-pre detail__code-pre--muted">
            <em>Failed to load {scene.file}: {json.message}</em>
          </pre>
        )}
      </div>
    </article>
  );
}
