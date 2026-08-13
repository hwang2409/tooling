import { useEffect, useState } from "react";
import { SCENES } from "../data/scenes";

const cache = new Map<string, string>();

export function useSceneJson(index: number) {
  const scene = SCENES[index];
  const [state, setState] = useState<
    | { status: "loading" }
    | { status: "ready"; text: string }
    | { status: "error"; message: string }
  >(() => {
    const hit = cache.get(scene.file);
    return hit ? { status: "ready", text: hit } : { status: "loading" };
  });

  useEffect(() => {
    let cancelled = false;
    const hit = cache.get(scene.file);
    if (hit) {
      setState({ status: "ready", text: hit });
      return;
    }
    setState({ status: "loading" });
    fetch(scene.file)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.text();
      })
      .then((text) => {
        if (cancelled) return;
        cache.set(scene.file, text);
        setState({ status: "ready", text });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : String(error);
        setState({ status: "error", message });
      });
    return () => {
      cancelled = true;
    };
  }, [scene.file]);

  return state;
}
