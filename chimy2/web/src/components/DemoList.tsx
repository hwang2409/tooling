import { DEMOS } from "../data/demos";

type Props = {
  index: number;
  onSelect: (next: number) => void;
};

export function DemoList({ index, onSelect }: Props) {
  return (
    <ol className="list" role="listbox" aria-label="demos">
      {DEMOS.map((demo, i) => {
        const active = i === index;
        return (
          <li key={demo.name}>
            <button
              type="button"
              className={`row${active ? " row--active" : ""}`}
              role="option"
              aria-selected={active}
              onClick={() => onSelect(i)}
            >
              <span className="row__num">{String(i + 1).padStart(2, "0")}</span>
              <span className="row__name">{demo.name}</span>
              <span
                className="row__dot"
                aria-hidden
                style={{ background: demo.hue }}
              />
              <span className="row__tag">{demo.tag}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
