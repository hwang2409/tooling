# mesh level of detail

`LodMesh` stores the source mesh at level zero and cached simplified meshes
after it. The default chain has three targets: 50%, 25%, and 12.5% of the
source triangle count.

Simplification uses deterministic Garland-Heckbert quadric error metric edge
collapses. Candidate order is `(error.total_cmp, stable edge index)`. The
edge midpoint is the replacement position. UVs use the edge midpoint when
both endpoints have UVs. Normals and tangents are rebuilt from the resulting
faces by `Mesh::new`.

The implementation rejects a collapse when any affected triangle becomes
degenerate or reverses its old orientation. For closed, consistently oriented
meshes, it also rejects a candidate that points inward. Sphere-like meshes
project replacement points back to their detected source radius. This keeps
the common subdivided-sphere case within its source surface.

Selection uses the source AABB. It transforms the AABB center, estimates a
conservative radius, and projects it with the Euclidean norms of projection
matrix rows zero and one. No trigonometric function is used by the selection
math. The selected `LodSelection` is reusable by color and depth passes.

Plain `Mesh` APIs remain unchanged. Use `RenderFrame::draw_lod_mesh` for an
opt-in color submission and `Pipeline::draw_lod_mesh_depth` for a shadow
submission with the same selection.

Reference: Michael Garland and Paul S. Heckbert, “Surface Simplification Using
Quadric Error Metrics,” SIGGRAPH 1997.
