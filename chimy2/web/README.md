# chimy2 browser demo

A React + Vite showcase for the chimy2 CPU rasterizer. The renderer is still
the same Rust crate compiled to `wasm32-unknown-unknown` — the browser side is
a thin driver that copies the flat framebuffer into a canvas each frame.

## project shape

```
web/
  index.html            # Vite entry
  package.json          # pinned deps: react, react-dom, vite, plugin-react, ts
  vite.config.ts
  tsconfig.json
  src/
    main.tsx            # ReactDOM bootstrap
    App.tsx             # layout + keyboard router + shared state
    styles.css          # design tokens + all component styles
    data/               # SCENES + SHADER_MODES
    hooks/              # useRenderer, useSceneRoute, useSceneJson
    wasm/loadWasm.ts    # thin wasm loader
    components/         # Header, Hero, LiveViewport, ModePicker, ...
  public/               # served at / — includes chimy2.wasm, scenes/, gallery/
  dist/                 # committed built bundle (byte-checked in CI)
  build.sh              # rebuild the wasm blob into public/
```

## dev

Install once, then run the Vite dev server. The wasm blob under `public/` is
served as-is at `/chimy2.wasm`; hot reload works for anything under `src/`.

```sh
cd chimy2/web
npm ci        # install pinned deps from package-lock.json
npm run dev   # http://localhost:5173
```

## build

```sh
cd chimy2/web
npm run build   # tsc --noEmit + vite build → dist/
```

The build is deterministic across a fresh checkout (all dep versions are
pinned exactly and reproduced from `package-lock.json`). CI byte-compares the
generated `dist/` against the version committed in this repo — if it drifts,
the `web-build` job fails with instructions to rebuild and re-commit.

## static deploy

`dist/` is a fully static bundle. Any file server that serves `.wasm` with an
`application/wasm` content type works. The simplest local sanity check:

```sh
python3 -m http.server 8000 --directory chimy2/web/dist
```

Then open <http://localhost:8000/>.

The bundle uses relative asset paths (`base: "./"` in `vite.config.ts`), so it
also works when served from a subdirectory (GitHub Pages, S3 with a prefix,
etc.) with no further configuration.

## rebuilding the wasm

If you touch the renderer (`chimy2/src/**`), refresh the wasm blob:

```sh
./chimy2/web/build.sh   # cargo build + copy → chimy2/web/public/chimy2.wasm
```

The `wasm-check` CI job byte-compares this file against a fresh Cargo build to
catch drift.

## controls

- Drag the canvas to orbit; double-click to reset.
- `1`–`8` cycle shader modes; `[` / `]` step through them.
- `←` / `→` walk the scene gallery.
- `R` resets the orbit; `Space` pauses the render loop.
- `#/scene/03` deep-links directly to a specific gallery scene.
