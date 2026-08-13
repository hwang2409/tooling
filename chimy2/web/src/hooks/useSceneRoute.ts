import { useEffect, useState } from "react";
import { SCENES } from "../data/scenes";

function parseHash(hash: string): number | null {
  const match = hash.match(/^#\/scene\/(\d{1,2})$/);
  if (!match) return null;
  const raw = Number.parseInt(match[1], 10);
  if (!Number.isFinite(raw) || raw < 1 || raw > SCENES.length) return null;
  return raw - 1;
}

export function useSceneRoute(initial = 0) {
  const [index, setIndex] = useState<number>(() => {
    if (typeof window === "undefined") return initial;
    return parseHash(window.location.hash) ?? initial;
  });

  useEffect(() => {
    const handler = () => {
      const next = parseHash(window.location.hash);
      if (next !== null && next !== index) setIndex(next);
    };
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, [index]);

  const update = (next: number) => {
    const wrapped = ((next % SCENES.length) + SCENES.length) % SCENES.length;
    setIndex(wrapped);
    const label = `#/scene/${String(wrapped + 1).padStart(2, "0")}`;
    if (window.location.hash !== label) {
      history.replaceState(null, "", label);
    }
  };

  return [index, update] as const;
}
