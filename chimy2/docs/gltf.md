# glTF support

`GltfAsset` supports JSON `.gltf` files. It supports external `.bin` files and
base64 binary data URIs. Paths are relative and cannot contain parent traversal.
GLB files, sparse accessors, morph targets, and bufferView images are rejected.

Supported primitives use `TRIANGLES` and these attributes:

- `POSITION` is required.
- `NORMAL`, `TEXCOORD_0`, `JOINTS_0`, and `WEIGHTS_0` are optional.
- Indices use unsigned scalar accessors.
- Accessors support `SCALAR`, `VEC2`, `VEC3`, `VEC4`, and `MAT4`.
- Component types 5120 through 5126, normalization, and byte strides are checked.

Nodes use either a matrix or TRS. TRS is composed as `T * R * S`. Matrix nodes
are not decomposed. Animation channels that target matrix nodes are rejected.
Node worlds are rebuilt from the hierarchy for each pure animation time input.

Skins use up to four influences per vertex. Weights below `1e-5` total use the
bind vertex. Other weights are clamped to zero and normalized. Joint matrices
are `joint_world * inverse_bind`. Positions and normals are blended on the CPU.
Normals use each joint matrix inverse-transpose and are renormalized. Skinning
creates a new `Mesh` per frame, so it pays for a full vertex and derived-normal
rebuild before frame submission. This keeps the existing mesh mutation and
cache discipline intact.

Translation, rotation, and scale channels support `LINEAR` and `STEP`.
Rotation uses shortest-path quaternion slerp. `CUBICSPLINE` is rejected.
Animation samples clamp to the first keyframe before the animation starts and
to the last keyframe after it ends.

`alphaMode` supports `OPAQUE` and `BLEND`. `OPAQUE` ignores alpha. `BLEND` uses
the transparent draw queue and carries `baseColorFactor` alpha into the
uniforms. `MASK` is rejected because the current fragment path has no discard
seam.

`baseColorFactor` and `baseColorTexture` provide albedo. Albedo textures use
sRGB decoding. `normalTexture` uses linear decoding. Metallic and roughness
feed the Cook-Torrance GGX shader directly. Base color factors and scalar
factors stay linear. Roughness uses the Disney `alpha = roughness^2`
convention, with a 0.045 floor; direct lighting uses Schlick-GGX geometry with
`k = alpha / 2`. The renderer clamps bright specular values at its single
sRGB encode until the later HDR milestone.

The committed `assets/arm.gltf` is a two-bone, hand-authored test asset. The
viewer uses the same scene and mesh submission path as headless tests:

```text
cargo run --release --bin gltf_viewer -- --frames 30
cargo run --release --bin gltf_viewer -- --frames 30 --screenshot /tmp/arm.ppm
```
