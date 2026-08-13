import { useEffect } from "react";
import { Stage } from "./components/Stage";
import { DemoList } from "./components/DemoList";
import { DEMOS } from "./data/demos";
import { useDemoRoute } from "./hooks/useDemoRoute";

export default function App() {
  const [index, setIndex] = useDemoRoute(0);

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
      if (Number.isInteger(digit) && digit >= 1 && digit <= DEMOS.length) {
        setIndex(digit - 1);
        return;
      }
      if (event.key === "ArrowDown" || event.key === "ArrowRight") {
        event.preventDefault();
        setIndex(index + 1);
        return;
      }
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
        event.preventDefault();
        setIndex(index - 1);
        return;
      }
      if (event.key.toLowerCase() === "r") {
        window.dispatchEvent(new CustomEvent("chimy2:reset-orbit"));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, setIndex]);

  return (
    <main className="room">
      <aside className="rail">
        <a className="mark" href="#/scene/01" aria-label="chimy2">
          <span className="mark__word">chimy</span>
          <span className="mark__two">2</span>
        </a>
        <p className="rail__caption">
          Eight live demos of a CPU rasterizer written from scratch. Every
          frame drawn in <span className="rail__cap-em">wasm</span>, on this
          thread.
        </p>

        <DemoList index={index} onSelect={setIndex} />

        <footer className="rail__foot">
          <div className="hint">
            <kbd>↑</kbd><kbd>↓</kbd><span>pick</span>
          </div>
          <div className="hint">
            <kbd>1</kbd>–<kbd>8</kbd><span>jump</span>
          </div>
          <div className="hint">
            <kbd>drag</kbd><span>orbit</span>
          </div>
          <div className="hint">
            <kbd>R</kbd><span>reset</span>
          </div>
        </footer>
      </aside>

      <section className="panel" aria-label="live render">
        <Stage index={index} />
      </section>
    </main>
  );
}
