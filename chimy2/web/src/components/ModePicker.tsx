import { SHADER_MODES } from "../data/modes";

type Props = {
  mode: number;
  onChange: (index: number) => void;
};

export function ModePicker(props: Props) {
  return (
    <div className="picker" role="tablist" aria-label="shader modes">
      {SHADER_MODES.map((entry, index) => {
        const active = index === props.mode;
        return (
          <button
            key={entry.tag}
            type="button"
            role="tab"
            aria-selected={active}
            className={`pill${active ? " pill--active" : ""}`}
            onClick={() => props.onChange(index)}
          >
            <span className="pill__num">MODE {String(index + 1).padStart(2, "0")}</span>
            <span className="pill__title">{entry.name}</span>
            <span className="pill__desc">{entry.desc}</span>
          </button>
        );
      })}
    </div>
  );
}
