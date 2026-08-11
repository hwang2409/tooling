# skybox and environment mapping

`CubeTexture::new` takes faces in this order: `+X, -X, +Y, -Y, +Z, -Z`.
Every face must be square, have the same size, and use the same color space.
The constructor sets each face to `ClampToEdge` before bilinear sampling.

The face convention uses image coordinates. `u` increases right, and `v`
increases down. After dominant-axis selection, the direction maps as follows:

| face | u | v |
| --- | --- | --- |
| +X | `-z / abs(x)` | `-y / abs(x)` |
| -X | ` z / abs(x)` | `-y / abs(x)` |
| +Y | ` x / abs(y)` | ` z / abs(y)` |
| -Y | ` x / abs(y)` | `-z / abs(y)` |
| +Z | ` x / abs(z)` | `-y / abs(z)` |
| -Z | `-x / abs(z)` | `-y / abs(z)` |

The values are remapped from `[-1, 1]` to `[0, 1]`. Axis directions therefore
sample the center of their selected face. The edge clamp avoids cross-face
sampling. It cannot hide a mismatch between authored neighboring edges.

`RenderFrame::draw_skybox` stores one cube map and camera for the frame. Flush
sorts and renders opaque commands first. It then runs the sky pass, followed by
transparent commands. The pass computes a far-plane ray with the inverse
view-projection matrix. It subtracts the camera position before normalization,
so translation does not move the sky. It tests the cleared far depth and does
not write depth. Opaque geometry therefore occludes the sky.

`EnvironmentBlinnPhongShader` uses the existing Blinn-Phong evaluator. It gets
the surface-to-camera vector in world space, reflects its incident negation
around the world normal, and samples the cube map. Cube samples are decoded to
linear RGB before the reflectivity mix. The framebuffer encodes sRGB once.
