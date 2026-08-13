import { useEffect, useState } from "react";
import { DEMOS } from "../data/demos";

// Hash routes stay `#/scene/NN` for backwards compatibility with the older
// gallery URLs; NN is 1-indexed and clamps into the demo list.
function parseHash(hash: string): number | null {
  const match = hash.match(/^#\/scene\/(\d{1,2})$/);
  if (!match) return null;
  const raw = Number.parseInt(match[1], 10);
  if (!Number.isFinite(raw) || raw < 1 || raw > DEMOS.length) return null;
  return raw - 1;
}

function wrap(next: number): number {
  return ((next % DEMOS.length) + DEMOS.length) % DEMOS.length;
}

export function useDemoRoute(initial = 0) {
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
    const wrapped = wrap(next);
    setIndex(wrapped);
    const label = `#/scene/${String(wrapped + 1).padStart(2, "0")}`;
    if (window.location.hash !== label) {
      history.replaceState(null, "", label);
    }
  };

  return [index, update] as const;
}
