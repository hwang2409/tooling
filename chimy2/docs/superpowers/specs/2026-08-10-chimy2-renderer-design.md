# chimy2 — Software Rasterizer Design

Date: 2026-08-10
Status: approved by Henry (design conversation, tooling-dev orchestrator session)
Predecessor: [chimy](https://github.com/hwang2409/chimy) — C/SDL2 software renderer

## Purpose

Rebuild chimy's 3D renderer from scratch in Rust, and this time climb the whole
hill chimy stopped short of: z-buffer, barycentric rasterization,
perspective-correct interpolation, shading, textures, a real camera.
Architecture is a "mini-GPU": a programmable pipeline seam where every material
is a shader, and the raster core never changes.

## Non-goals

- No GPU APIs (OpenGL/Metal/Vulkan/wgpu). The pipeline is CPU code we write.
- No renderer, math, image, or mesh crates. See dependency rule below.
- No ray tracing. This is a rasterizer.
- No cross-platform polish beyond "runs on macOS"; Linux support is incidental
  via winit/softbuffer, not tested.

## Dependency rule (the chimy rule)

Two runtime crates only:

- `winit` — window creation + input events
- `softbuffer` — present a `&[u32]` pixel buffer to the window

Everything else is hand-written: vector/matrix/quaternion math, OBJ parsing,
image decoding, rasterization, threading. Dev-dependencies that do not ship in
the renderer (e.g. `criterion` for benches) are allowed.

## Architecture

```
app / demos
  └─ scene: meshes, materials, camera, lights
       └─ pipeline<VS, FS>                    ← the seam
            ├─ vertex stage    VS(&Vertex, &Uniforms) -> (ClipPos, Varyings)
            ├─ clip + cull     near-plane frustum clip, backface cull
            ├─ raster core     barycentric scan, perspective-correct
            │                  varying interpolation, z-test
            └─ fragment stage  FS(&Varyings, &Uniforms) -> Color
  └─ targets: Framebuffer { color: Vec<u32>, linear HDR sidecar, depth: Vec<f32> }
  └─ present: softbuffer blit
```

Module layout (single crate, `src/`):

| Module | Responsibility | Depends on |
|---|---|---|
| `math` | `Vec2/3/4`, `Mat4`, `Quat`; `std::ops` overloads | nothing |
| `pipeline` | vertex/fragment traits, `Varyings` trait, draw call orchestration | `math` |
| `raster` | triangle setup, barycentric scan, depth test, varying interp | `math`, `pipeline` traits |
| `clip` | near-plane clipping, backface cull | `math` |
| `mesh` | OBJ parser (v/vn/vt/f), mesh struct | `math` |
| `image` | PPM + QOI decoders, `Texture` + sampling (nearest, bilinear) | nothing |
| `camera` | quaternion camera, orbit + WASD controllers, view/proj matrices | `math` |
| `shaders` | flat, blinn-phong, textured — all written against the seam | `pipeline`, `image` |
| `fb` | `Framebuffer`, clear, resize | nothing |
| `present` | winit event loop, softbuffer blit, input plumbing | `fb` |
| `demos/` (bins) | scenes wiring everything together | all |

Key contracts:

- `Varyings` has one job: interpolate by barycentric weights
  (`fn lerp3(a, b, c, w: Vec3) -> Self`). Implemented for structs of
  f32/vec fields.
- The raster core is generic over `Varyings` + fragment shader and knows
  nothing about lighting, textures, or materials.
- Perspective correctness: varyings are interpolated in 1/w space; the raster
  core owns this, shaders never see it.
- Determinism: no clock, no randomness inside `pipeline`/`raster`/`shaders`.
  Same scene in ⇒ same pixels out. This is what makes golden tests possible.

## Milestones

Each milestone is a bounded fleet ticket ending in CI-green, reviewed, merged.

1. **M1 scaffold** — repo layout, CI, winit+softbuffer window with resize,
   `Framebuffer`, full `math` module with unit tests.
2. **M2 first triangle** — pipeline seam, raster core: barycentric fill,
   z-buffer + depth test, flat-color shader. Golden-image test harness lands
   here.
3. **M3 meshes + camera** — OBJ parser, model transforms, quaternion camera
   (orbit + WASD), near-plane clipping, backface cull. Depth-tested mesh
   render replaces painter's sort for good.
4. **M4 shading** — vertex normals, perspective-correct varyings,
   blinn-phong with directional + point lights, all as shaders on the seam.
5. **M5 textures** — PPM + QOI decoders, UV sampling, nearest + bilinear
   filtering, textured shader. Dev-side script converts PNG assets to QOI.
6. **M6 performance** — tile-binned multithreaded rasterizer behind the same
   seam (std::thread, no rayon), criterion benches, target: 30+ fps on a
   100k-triangle mesh at 1280x720 with blinn-phong shading.
7. **M7 demos + polish** — 3-4 demo scenes, screenshots/GIFs, README in
   chimy's voice (why from scratch, what was learned).

### M6 implementation notes

### HDR pipeline note

HDR mode uses the post-processing linear intermediate as the primary color
target. The framebuffer keeps a linear sidecar during raster writes, blending,
and SSAA. LDR mode keeps the original `u32` write path byte-for-byte.

The HDR order is render -> SSAA -> bloom with threshold 1.0 -> optional
vignette -> Narkowicz's fitted ACES approximation -> one sRGB encode -> FXAA.
FXAA stays after encoding because its edge luma matches display values.

The ACES fit is `x * (2.51x + 0.03) / (x * (2.43x + 0.59) + 0.14)`. Exposure
is a sanitized linear multiplier before the fit. Values above one are not
clamped before ACES. The final encoder is the only HDR RGB clamp.

M6 uses fixed 64x64 pixel tiles. This size keeps bin lists small while giving
workers enough pixels to amortize local framebuffer setup and merge costs.
The default worker count is `std::thread::available_parallelism()`. Set
`CHIMY_THREADS` to a positive integer, or call `Pipeline::set_thread_count`.

The pipeline runs the vertex, clip, cull, and viewport stages once per input
triangle. It bins each resulting triangle by its screen-space bounding box.
Each worker owns one tile-local color and depth buffer. It processes that
tile's triangle indices in submission order, then the caller merges the tile.
Tiles do not share storage, so color and depth writes need no locks. The
serial path uses the same raster per-pixel body and is selected for one worker.

## Testing

- `math`, `clip`, `mesh`, `image`: unit tests with hand-computed expectations.
- Rasterizer + shaders: golden-image tests. Render fixed scenes headless into a
  `Framebuffer`, compare against committed PPM goldens byte-for-byte.
  Goldens regenerate via a test flag; diffs reviewed by eye.
- M6 must not change goldens: the parallel rasterizer is correct only if it is
  pixel-identical to the serial one.
- Benches: criterion, per-milestone baselines so M6 has a before/after.

## Error handling

- Asset loading (`mesh`, `image`): `Result` with line/byte context.
- Core pipeline: no fallible paths; degenerate triangles are culled, not errors.
- Window/present errors: fail fast, this is a demo app not a service.

## Risks

- The `Varyings`/shader seam will likely need one revision once blinn-phong and
  textures exist (M4-M5). Expected and contained: the seam is one module.
- QOI/PPM-only textures mean asset conversion friction. One-time dev script.
- softbuffer + macOS has resize/scale-factor quirks; M1 owns them.
- Multithreading in M6 risks non-determinism; the pixel-identical golden rule
  is the guard.
