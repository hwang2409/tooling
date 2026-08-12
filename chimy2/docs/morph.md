# morph targets

chimy2 applies glTF morph targets on the cpu before mesh submission. each
vertex accumulates one position delta and any normal deltas for each active
target. normals are normalized after accumulation. the rebuilt mesh bounds
drive submission culling.

the implementation follows glTF 2.0 sections 3.6.3 and 3.7.3:

- <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#animations>
- <https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#_morph_targets>

sparse accessors are rejected by the existing loader. morph targets are not
combined with `LodMesh`.

the demo cost is O(vertex_count × target_count) per frame. reproduce it with:

```text
cargo run --release --bin morph_demo -- --frames 120 --size 960x640 --screenshot target/morph.ppm
```
