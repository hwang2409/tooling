# chimy2 shader playground pack

The shader pack uses only `VertexStage`, `FragmentStage`, and `Varyings`.
It does not read depth, interpolation weights, gradients, or other raster data.

`expand_mesh_with_barycentrics` expands each indexed triangle. It gives its
three shader vertices `(1,0,0)`, `(0,1,0)`, and `(0,0,1)`. Toon and wireframe
materials use the interpolated minimum component as an edge test. This gives a
varying-space width. It is stable within a triangle, but it is not a true
screen-space pixel width.

## materials

- `toon`: diffuse bands use thresholds `0.25`, `0.50`, and `0.75`. The four
  band values are `0.15`, `0.40`, `0.70`, and `1.0`. Edge pixels are darkened.
- `psx`: post-divide vertex positions snap to a coarse NDC grid. The fragment
  stage applies a 4x4 Bayer matrix with centered thresholds `(n + 0.5) / 16`
  at three bits per channel. Affine texture
  mapping is not included. The current pipeline has one perspective-correct
  interpolation path, and a second kernel would violate the seam contract.
- `dither`: carries clip-space x/y and w as varyings, then divides after
  interpolation to recover true screen coordinates. It applies the same 4x4
  Bayer matrix to a flat linear RGB color. The default is three bits per
  channel.
- `fog`: computes Euclidean view-space distance in the vertex stage. It uses
  linear fog from `fog_start` to `fog_end`. Fog and base colors stay linear;
  LDR uses `argb8888_linear`, while HDR-capable shaders return linear values
  to the post chain for ACES and the one final sRGB encode.
- `normals`: transforms the normal with the inverse-transpose model matrix and
  remaps `[-1, 1]` to `[0, 1]` in linear RGB.
- `wireframe`: overlays an edge color when the minimum barycentric component is
  below the threshold. The varying-space width limitation matches toon mode.

Run the demo with `cargo run --release --bin shader_playground -- --shader toon`.
Press `n` to cycle modes. Add `--frames N`, `--screenshot path.ppm`, or
`--size WxH` for deterministic headless output.
