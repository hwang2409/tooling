# data-driven scenes

`chimy2::scene::Scene` loads JSON through the same hand-written parser used
by glTF. The loader is strict. Unknown fields fail with a path-qualified
error, such as `objects[2].transform.scale: expected 3 numbers`.

Objects stay in file order. The renderer submits them in that order. This is
part of the format contract and keeps transparent and opaque results stable.

All numeric values pass through the renderer's public sanitizing setters when
the scene builds postfx, lights, materials, and instances. The loader also
clamps schema values that have no renderer setter, such as camera clip planes
and integer sizes.

## schema

- `camera` is required. It contains `position`, `target`, `fov`, `near`, and
  `far`. Angles use radians.
- `environment` contains an RGBA `background`. `skybox` takes six QOI face
  paths: `px`, `nx`, `py`, `ny`, `pz`, and `nz`. `ibl` contains bake settings.
- `lights` is an ordered list of `directional` and `point` lights. Directional
  shadows accept `basic`, `csm`, and `pcss`. Point shadows accept `cube`.
- `objects` is an ordered list. Each object has a mesh path, optional material
  override, TRS transform, optional instancing, or optional LOD settings.
  Instancing and LOD cannot be combined on one object. Use separate objects.
  OBJ paths are loaded by the existing OBJ loader. The material `type` is
  `blinn_phong` or `ggx` and maps to existing renderer material capabilities.
- `postfx` is an ordered list of `ssao`, `dof`, `bloom`, `fxaa`, `vignette`,
  and `aces` passes. Pass parameters use the existing pass setters.
- `particles` contains deterministic emitter settings. `hud` contains text,
  screen coordinates, integer scale, and normalized RGBA color.

The complete shipped example is
[`examples/showcase.scene.json`](../examples/showcase.scene.json). Run it with:

```text
cargo run --bin scene_viewer -- --scene examples/showcase.scene.json --screenshot /tmp/showcase.ppm
```

## annotated example

```json
{
  "camera": {
    "position": [0, 1.4, 6.2],
    "target": [0, 0.3, 0],
    "fov": 0.95,
    "near": 0.1,
    "far": 100
  },
  "environment": {
    "background": [0.008, 0.012, 0.028, 1],
    "skybox": {"px": "../assets/skybox_px.qoi", "nx": "../assets/skybox_nx.qoi", "py": "../assets/skybox_py.qoi", "ny": "../assets/skybox_ny.qoi", "pz": "../assets/skybox_pz.qoi", "nz": "../assets/skybox_nz.qoi"},
    "ibl": {"intensity": 0.8, "irradiance_size": 16, "prefilter_size": 32, "prefilter_levels": 5}
  },
  "lights": [
    {"type": "directional", "direction": [-0.35, 0.75, 1], "color": [1, 0.92, 0.82], "shadow": {"type": "csm", "map_size": 1024, "cascades": 3, "lambda": 0.55}},
    {"type": "point", "position": [2, 2.5, 3], "color": [1, 0.25, 0.08], "constant": 1, "linear": 0.04, "quadratic": 0.01, "shadow": {"type": "cube", "map_size": 512}}
  ],
  "objects": [
    {"mesh": "../assets/icosahedron.obj", "material": {"type": "ggx", "metallic": 0.8, "roughness": 0.24}, "transform": {"position": [-1.25, 0.1, 0], "rotation": [0, 0.4, 0], "scale": [1.1, 1.1, 1.1]}, "instancing": {"count": 3, "grid": {"dimensions": [3, 1, 1], "spacing": [1.0, 0, 0]}}},
    {"mesh": "../assets/cube.obj", "material": {"type": "blinn_phong", "diffuse": [0.95, 0.24, 0.06], "shininess": 32}, "transform": {"position": [1.25, -0.35, 0], "rotation": [0.2, -0.5, 0.15], "scale": [0.8, 0.8, 0.8]}},
    {"mesh": "../assets/cube.obj", "material": {"type": "blinn_phong", "diffuse": [0.18, 0.7, 0.3]}, "transform": {"position": [0, -0.25, -0.4], "scale": [1, 1, 1]}, "lod": {"ratios": [0.5, 0.05], "thresholds": [1000, 0.1]}}
  ],
  "postfx": [{"type": "ssao", "radius": 0.55}, {"type": "bloom"}, {"type": "aces", "exposure": 1.15}],
  "particles": [{"position": [0, -0.8, 0], "emission_rate": 2, "lifetime_steps": 90, "initial_velocity": [0, 1.8, 0], "gravity": [0, -2.4, 0], "capacity": 128}],
  "hud": [{"text": "CHIMY2 SCENE FORMAT", "x": 20, "y": 20, "scale": 2, "color": [0.85, 0.92, 1, 1]}]
}
```
