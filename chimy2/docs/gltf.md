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

`baseColorFactor` and `baseColorTexture` provide albedo. Albedo textures use
sRGB decoding. `normalTexture` uses linear decoding. Metallic and roughness
are read but parked until the PBR renderer: metallic maps 4% dielectric
specular toward white, and roughness maps to a Blinn-Phong exponent from 1 to
129.

The committed `assets/arm.gltf` is a two-bone, hand-authored test asset. The
viewer uses the same scene and mesh submission path as headless tests:

```text
cargo run --release --bin gltf_viewer -- --frames 30
cargo run --release --bin gltf_viewer -- --frames 30 --screenshot /tmp/arm.ppm
```
