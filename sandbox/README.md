# newt sandbox viewer

this binary joins newt physics to the chimy2 native software renderer.
geometry is converted to fixed local meshes when a scene loads. each frame
only the geom transform uniforms change.

## run

from the repository root:

```bash
cargo run --release --manifest-path sandbox/Cargo.toml
```

the built-in scenes are `biped-simple`, `pendulum`, `arm`, `stack`, and
`pile`. the biped is the humanoid-scale performance check.

## controls

- `space`: play or pause
- `r`: reset the current scene
- `-` / `[` and `=` / `]`: change speed from 0.25x to 4x
- `n` / `p`: next or previous scene
- `1` through `5`: select a scene
- hold left mouse and drag: orbit
- mouse wheel: zoom
- `escape`: exit

the loop advances newt with its fixed timestep and renders at the window
rate. chimy2 owns the window and framebuffer. sandbox owns scene selection,
mesh translation, simulation timing, and input policy.

## capture and measurement

record a scripted 20-second interaction without opening a window:

```bash
cargo run --release --manifest-path sandbox/Cargo.toml -- \
  --record /tmp/newt-36-record 20
```

the command writes temporary numbered PPM frames, encodes
`/tmp/newt-36-record/newt-sandbox-demo.mp4` with system `ffmpeg`, then removes
the temporary frames. the script cycles scenes, orbits the camera, pauses,
resumes, and changes speed.

measure the live biped viewer for 30 seconds:

```bash
cargo run --release --manifest-path sandbox/Cargo.toml -- --measure 30
```

this reports min, average, and p99 frame, simulation, render, and present
times. measurement mode does not record frames.
