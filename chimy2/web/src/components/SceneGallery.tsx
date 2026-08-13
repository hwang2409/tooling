import { SCENES } from "../data/scenes";

type Props = {
  selectedIndex: number;
  onSelect: (index: number) => void;
};

export function SceneGallery(props: Props) {
  return (
    <div className="gallery" role="listbox" aria-label="scene gallery">
      {SCENES.map((scene, index) => {
        const active = index === props.selectedIndex;
        const idxLabel = String(index + 1).padStart(2, "0");
        return (
          <button
            key={scene.id}
            type="button"
            role="option"
            aria-selected={active}
            className={`tile${active ? " tile--active" : ""}`}
            onClick={() => props.onSelect(index)}
          >
            <span className="tile__thumb">
              <img src={scene.image} alt={`${scene.name} scene render`} loading="lazy" />
              <span className="tile__strip" aria-hidden>
                <span />
                <span />
                <span />
              </span>
            </span>
            <span className="tile__body">
              <span className="tile__head">
                <span className="tile__name">{scene.name}</span>
                <span className="tile__idx tabular">{idxLabel}</span>
              </span>
              <span className="tile__desc">{scene.desc}</span>
              <span className="tile__tags">
                {scene.features.slice(0, 4).map((f) => (
                  <span key={f}>{f}</span>
                ))}
              </span>
            </span>
          </button>
        );
      })}
    </div>
  );
}
