# newt rendered showcase

the examples use chimy2's production mesh pipeline by default. each render
uses tessellated geometry, GGX materials, a directional shadow map, SSAO,
bloom, ACES tonemapping, and deterministic dithering. `--wireframe` keeps the
older debug path for short-term comparison.

all commands run from `newt/`.

## demos

```text
cargo run --release --example biped_walk -- --steps 5000 --frames-dir /tmp/newt-walk
cargo run --release --example tendon_lift -- --frames 900 --frames-dir /tmp/newt-tendon
cargo run --release --example solver_stack -- --frames 800 --frames-dir /tmp/newt-stack
cargo run --release --example pile -- --frames 800 --frames-dir /tmp/newt-pile
cargo run --release --example cartpole -- --frames 900 --frames-dir /tmp/newt-cartpole
cargo run --release --example arm -- --frames 1800 --frames-dir /tmp/newt-arm
```

the default output size is kept by each example for compatibility. use
`--size 960x640` for the art-direction review size. non-biped demos write the
final numbered capture as `frame-00.ppm`; biped writes eight fixed-phase
captures. chimy2 exposes PPM output, so no image dependency is added.

convert a capture to PNG on macOS:

```text
sips -s format png /tmp/newt-walk/frame-08.ppm --out /tmp/newt-walk/frame-08.png
```

assemble a numbered sequence into an MP4:

```text
ffmpeg -framerate 60 -i /tmp/newt-walk/frame-%02d.ppm -c:v libx264 -pix_fmt yuv420p /tmp/newt-walk.mp4
```

## live viewer

```text
cargo run --release --example viewer -- --scenario walk
cargo run --release --example viewer -- --scenario pile --frames 600
cargo run --release --example viewer -- --model models/arm.json
```

the viewer steps the world at its fixed `dt` and renders independently.

| key | action |
| --- | --- |
| space | pause or resume |
| `.` | single fixed step |
| `[` / `]` | halve or double speed, clamped to 0.25x–4x |
| `1`–`3` | camera presets |
| `a` / `d` | orbit left or right |
| `w` / `s` | orbit up or down |
| escape | close the window |

the current chimy2 input API exposes physical key state, not pointer motion.
the viewer therefore uses keyboard orbit controls until a public pointer-input
surface is available.

## composition table

composition values live in the `showcase_support::composition` table. This
keeps target, camera, accent, and framing changes cheap during review.

| demo | target | camera | accent |
| --- | --- | --- | --- |
| biped walk | follow root | close three-quarter follow | blue |
| tendon lift | sphere and box | three-quarter side view | cyan |
| solver stack | tower center | high dramatic three-quarter | orange |
| pile | pile center | wide three-quarter | red |
| cartpole | cart pivot | side view | amber |
| arm | shoulder and tip | side view | violet |
