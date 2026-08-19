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
cargo run --release --example biped_walk -- --steps 5000 --out demos/demo-biped.mp4
cargo run --release --example tendon_lift -- --frames 3000 --out demos/demo-tendon.mp4
cargo run --release --example solver_stack -- --frames 3000 --out demos/demo-stack.mp4
cargo run --release --example pile -- --frames 3000 --out demos/demo-pile.mp4
cargo run --release --example hfield_demo -- --frames 3000 --out demos/demo-hfield.mp4
cargo run --release --example cartpole -- --frames 3000 --out demos/demo-cartpole.mp4
cargo run --release --example cartpole -- --velocity --frames 3000 --out demos/demo-cartpole-velocity.mp4
cargo run --release --example arm -- --frames 3000 --out demos/demo-arm.mp4
cargo run --release --example muscle_pendulum -- --frames 3000 --out demos/demo-muscle-pendulum.mp4
cargo run --release --example showcase_features -- --scene joints --frames 3000 --out demos/demo-joints.mp4
cargo run --release --example showcase_features -- --scene equalities --frames 3000 --out demos/demo-equalities.mp4
cargo run --release --example showcase_features -- --scene geoms --frames 3000 --out demos/demo-geoms.mp4
cargo run --release --example showcase_features -- --scene sensors --frames 3000 --out demos/demo-sensors.mp4
cargo run --release --example showcase_features -- --scene solvers --frames 3000 --out demos/demo-solvers.mp4
cargo run --release --example showcase_features -- --scene integrators --frames 3000 --out demos/demo-integrators.mp4
cargo run --release --example tendon_cylinder_pulley -- --frames 3000 --out demos/demo-tendon-cylinder-pulley.mp4
cargo run --release --example chain -- --frames 3000 --out demos/demo-chain.mp4
cargo run --release --example linkage -- --frames 3000 --out demos/demo-linkage.mp4
cargo run --release --example pendulum -- --frames 3000 --out demos/demo-pendulum.mp4
cargo run --release --example tumble -- --frames 3000 --out demos/demo-tumble.mp4
cargo run --release --example roll -- --frames 3000 --out demos/demo-roll.mp4
```

the joints scene shows hinge, ball, and slide joints. the equalities scene
shows connect, weld, and distance rows. `linkage` shows joint coupling. its
connect anchors stay at static reference points because tree-link connect
anchors are not supported by the engine yet. the contacts scene shows capsule,
cylinder, and ellipsoid geoms. the sensor scene
renders live joint, velocity, and accelerometer values in its hud. the solver
and integrator scenes run the same stack side by side with pgs/newton and
euler/implicitfast.

convex mesh ccd is not available on `origin/main` yet. the dedicated video is
deferred until the convex ccd routes land; the existing `pile` demo covers
supported convex mesh contacts.

the muscle pendulum starts at zero activation, holds control at zero, drives
from `0` to `1`, then returns to zero. the rendered demo is 5.016667 seconds
at 60 fps. its hud activation extrema are `0.000000..1.000000`.

`--out` defaults to a demo-specific mp4 in the current directory. the
implementation writes numbered ppm frames to a temporary directory, invokes
the system `ffmpeg`, and removes the temporary frames after assembly.

install `ffmpeg` if a run reports that it is missing. macos homebrew users
can install it with `brew install ffmpeg`.

legacy examples still accept `--wireframe` for one-frame PPM debug output.

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
