# newt demo showcase

the showcase demos render solid shaded meshes through chimy2. each demo uses
simple materials, readable colors, and one fixed camera. the biped uses a
follow camera.

video is the default output. each run advances the deterministic simulation
by 10 fixed simulation steps per video frame and writes 60 fps. with the
showcase timestep of 0.005 seconds, this is 3x simulation speed. a 5000-step
biped walk covers 25 seconds of simulation in an 8.33 second video.
rendering does not change the simulation state.

all commands run from `newt/`.

## video demos

```text
cargo run --release --example biped_walk -- --steps 5000 --out demo-biped.mp4
cargo run --release --example tendon_lift -- --frames 900 --out demo-tendon.mp4
cargo run --release --example solver_stack -- --frames 800 --out demo-stack.mp4
cargo run --release --example pile -- --frames 800 --out demo-pile.mp4
cargo run --release --example hfield_demo -- --frames 1800 --out demo-hfield.mp4
cargo run --release --example cartpole -- --frames 900 --out demo-cartpole.mp4
cargo run --release --example cartpole -- --velocity --frames 900 --out demo-cartpole-velocity.mp4
cargo run --release --example arm -- --frames 1800 --out demo-arm.mp4
```

`--out` defaults to a demo-specific mp4 in the current directory. the
implementation writes numbered ppm frames to a temporary directory, invokes
the system `ffmpeg`, and removes the temporary frames after assembly.

install `ffmpeg` if a run reports that it is missing. macos homebrew users
can install it with `brew install ffmpeg`.

use `--still PATH` for one final PPM frame. `--wireframe` keeps the old debug
renderer. `--frames-dir` remains as a legacy sparse-PPM compatibility mode.

the output size is demo-specific. use `--size 960x640` when a larger video is
needed.

## live viewer

the viewer is the interactive half of the showcase. it steps the world at its
fixed `dt` and renders every display update.

```text
cargo run --release --example viewer -- --scenario walk
cargo run --release --example viewer -- --scenario pile --frames 600
cargo run --release --example viewer -- --model models/arm.json
```

| key | action |
| --- | --- |
| space | pause or resume |
| `.` | single fixed step while paused |
| `[` / `]` | halve or double speed, clamped to 0.25x–4x |
| `1`–`3` | camera presets |
| `a` / `d` | orbit left or right |
| `w` / `s` | orbit up or down |
| escape | close the window |

the chimy2 input API exposes physical key state, not pointer motion. the
viewer uses keyboard orbit controls until a public pointer-input surface is
available.

## determinism check

the showcase adapter includes a test that runs the same simulation with and
without rendering. it checks the final state bytes. run it with:

```text
cargo test --manifest-path newt/Cargo.toml --example showcase_support rendering_does_not_change_deterministic_simulation_state
```
