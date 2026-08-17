# newt model format (tier 5)

`newt`'s native scene description is a JSON document. It describes a full
world: free bodies, kinematic trees, geoms, sites, PD actuators,
contact-pair filtering, gravity, and timestep. The loader lives at
[`newt::model`](../src/model.rs) and is backed by a hand-written JSON
value parser at [`newt::json`](../src/json.rs) — zero dependencies, no
serde, no libm.

design of record:
[superpowers/specs/2026-08-13-newt-physics-design.md](superpowers/specs/2026-08-13-newt-physics-design.md).

**MJCF compatibility is v1**. v0 is our native format; the schema
choices (`solref`, `damping`/`armature`, PD `dampratio`, `self_collide`)
are named after MuJoCo's XML vocabulary so v1's MJCF path can share
verbs without renaming.

## the strict-by-default rule

The loader REJECTS unknown fields at every scope. A typo like
`"dampign"` is a physics bug in disguise (silent no-ops in a loader are
one of the recurring lesson classes from chimy2). Every validation
error carries a JSON path so the reader can find the offending value
without re-parsing the source:

```
trees[0].links[2].joint.axis: hinge axis must be non-zero
```

Duplicate keys within an object, negative masses, dangling parent
references, actuators pointing at a non-hinge, sites on missing bodies,
zero hinge axes — every branch has a dedicated unit test in
[`newt/src/model.rs`](../src/model.rs).

## top-level fields

```json
{
  "version": "1",
  "gravity": [0.0, 0.0, -9.81],
  "timestep": 0.005,
  "integrator": "RK4",
  "bodies":   [ ... ],
  "trees":    [ ... ],
  "meshes":   [ ... ],
  "hfields":  [ ... ],
  "geoms":    [ ... ],
  "sites":    [ ... ],
  "actuators":[ ... ],
  "contact_pairs": { ... }
}
```

- **version** — must be `"1"` when present. Missing is fine; any other
  value is rejected.
- **gravity** — `[x, y, z]`. Default `[0, 0, -9.81]`.
- **timestep** — fixed integration step size in seconds. Must be > 0.
  Default `0.005`.
- **integrator** — `RK4`, `Euler`, or `implicitfast`. Default `RK4`.
- Every collection is optional. An empty scene loads cleanly.

## bodies (tier-1/2 free bodies)

```json
{
  "name": "ball",
  "mass": 1.0,
  "inertia": {"kind": "solid", "shape": {"kind": "sphere", "radius": 0.5}},
  "pose": {"position": [0, 0, 1.5], "orientation": [0, 0, 0, 1]},
  "velocity": {"linear": [0, 0, 0], "angular_body": [0, 0, 0]}
}
```

- **name** — unique within `bodies`.
- **mass** — kg, must be > 0.
- **inertia** — see [`inertia`](#inertia) below.
- **pose** — optional. `position` defaults to `[0,0,0]`. Orientation
  can be supplied in EITHER (but not both) of two forms:
    - `"orientation": [x, y, z, w]` — a quaternion literal, renormalized
      on load. Convenient when the caller has already computed the
      quaternion.
    - `"orientation_axis_angle": {"axis": [ax, ay, az], "angle": rad}`
      — the loader calls `Quat::from_axis_angle(axis, angle)` and gets
      byte-identical output to a programmatic call. Preferred for
      round-trip anchors against a programmatic scene (see
      `tests/model_load.rs::pile_json_matches_programmatic_construction_exactly`).
  Default is identity.
- **velocity** — optional. `linear` is in world coordinates, mirroring
  [`Body::linear_velocity`](../src/body.rs); `angular_body` is body-
  frame ω.

## trees (tier-3+ kinematic trees)

```json
{
  "name": "arm",
  "self_collide": false,
  "links": [
    { "name": "root",  "joint": {"kind":"fixed"}, "mass": 1.0, "inertia": {...} },
    { "name": "upper", "parent": "root",
      "joint": {"kind":"hinge", "axis":[1,0,0]},
      "joint_offset_in_parent": {"position":[0,0,0]},
      "joint_offset_in_child":  {"position":[0,0,0.25]},
      "mass": 1.0, "inertia": {...}
    }
  ]
}
```

- **name** — unique within `trees`.
- **self_collide** — bool, default `false`. When `false`, contact pairs
  between two geoms attached to the same tree are dropped at load
  time. Turn on for humanoids or wherever self-contact is meaningful.
- **links** — ordered array. Index 0 is the root and must have
  `parent = null` (or the field may be omitted). Every subsequent link
  references its parent by NAME. A link name can only reference a
  parent defined EARLIER in the array (topological order).

### links

- **name** — unique within the tree.
- **parent** — required for index > 0; must reference an earlier link.
  For the root it must be `null` or omitted.
- **joint** — see [joints](#joints) below.
- **joint_offset_in_parent** — pose of the joint anchor in the parent's
  body frame. For the root, this is the world-frame anchor pose (its
  orientation is meaningful for `Fixed` roots and becomes the initial
  quaternion for `Free` roots). Non-root joints must have identity
  orientation (v0 constraint — v1 will lift this).
- **joint_offset_in_child** — pose of the joint anchor in this link's
  body frame. Must have identity orientation (v0 constraint).
- **mass**, **inertia** — same as bodies.

### joints

```json
{"kind": "free"}
{"kind": "fixed"}
{"kind": "hinge",
 "axis":     [1, 0, 0],
 "range":    [-1.5, 1.5],
 "damping":  0.5,
 "armature": 0.02,
 "limit":    {"stiffness": 1000, "damping": 60}
}
{"kind": "slide",
 "axis":     [0, 0, 1],
 "range":    [-0.5, 0.5],
 "damping":  0.1,
 "armature": 0.05,
 "limit":    {"stiffness": 1500, "damping": 50}
}
{"kind": "ball",
 "damping":  0.05,
 "armature": 0.01
}
```

- **kind** = `free` (6-DOF root), `fixed`, `hinge`, `slide`, or `ball`.
- Free roots accept scalar **damping** for all three angular and three
  linear velocity DOFs. The default is `0`.
- Hinges + slides: **axis** (required, normalized on load; zero rejected),
  **range** (optional `[lo, hi]`, lo < hi enforced), **damping**,
  **armature**, **limit.stiffness** / **limit.damping**. Units follow
  the DOF — hinge is rad / rad/s / N·m, slide is m / m/s / N.
- **limit.solref** / **limit.solimp** (optional): per-limit override
  for the PGS constraint solver — same shape as the geom-level
  `solref` / `solimp` blocks (`{"timeconst", "dampratio"}` and
  `{"dmin", "dmax", "width", "midpoint", "power"}` respectively).
  Omitted → `SolRef::DEFAULT` / `SolImp::DEFAULT`. Consulted when
  `solver.mode` is `"pgs"` or `"newton"`.
- Ball: **damping** (isotropic angular, N·m per rad/s), **armature**
  (per-axis rotor inertia, kg·m²). **NO `range` field** — a physically
  correct 3-DOF orientation limit needs the v1 solver landing in a
  follow-up ticket; the loader rejects a `range` on a ball joint with a
  pointer at that deferral.
- `free` is only valid on the root; `hinge` / `slide` / `ball` are
  invalid on the root (they need a parent to anchor against).

## inertia

Three kinds. Every one produces the same body-frame `Mat3` about the
COM.

```json
{"kind": "diag",   "values": [Ixx, Iyy, Izz]}
{"kind": "tensor", "values": [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]}
{"kind": "solid",  "shape":  {"kind": "box",       "half_extents": [hx, hy, hz]}}
{"kind": "solid",  "shape":  {"kind": "sphere",    "radius": r}}
{"kind": "solid",  "shape":  {"kind": "capsule",   "radius": r, "half_height": h}}
{"kind": "solid",  "shape":  {"kind": "cylinder",  "radius": r, "half_height": h}}
{"kind": "solid",  "shape":  {"kind": "ellipsoid", "semi_axes": [ax, ay, az]}}
```

- Diagonal moments must be > 0.
- Tensor: symmetric; product moments (`Ixy`, `Ixz`, `Iyz`) may be any
  sign. Rejected if singular.
- Solid delegates to
  [`geom::solid_box_inertia`](../src/geom.rs) et al., using the same
  formulas the engine tests pin. Mesh bodies do NOT have a solid variant
  (no volume integration) — use `diag` / `tensor` explicitly.

## meshes (v1 tier 2)

Convex-mesh assets referenced by `Mesh { mesh: NAME }` geoms. A geom
lookup by name resolves to a mesh id used at runtime.

```json
{
  "name": "tetra",
  "vertices": [[0,0,0], [1,0,0], [0,1,0], [0,0,1]],
  "faces":    [[0,2,1], [0,1,3], [0,3,2], [1,2,3]]
}
```

- **vertices** — array of `[x, y, z]` triples, ≥ 4, all coordinates
  finite.
- **faces** — array of `[i, j, k]` vertex-index triples, ≥ 4. Each face
  is a CCW-outward triangle. Indices must be non-negative integers in
  range.
- The mesh's convex hull is TRUSTED — see the "convex mesh trust model"
  section in [`docs/contacts.md`](contacts.md). The loader runs only
  structural checks; convexity itself is not verified.

## hfields

Heightfield assets use normalized row-major elevation data:

```json
{
  "name": "terrain",
  "nrow": 3,
  "ncol": 3,
  "size": [2, 2, 0.8, 0.2],
  "data": [0, 0.2, 0.3, 0.1, 0.4, 0.5, 0.2, 0.5, 0.7]
}
```

`nrow` and `ncol` are at least 2. `size` is
`[half_width_x, half_width_y, top_height, base_depth]`, with all values
positive. `data` may also be named `elevation`; it must contain exactly
`nrow*ncol` values in `[0,1]`. PNG files are not part of the zero-dependency
subset.

Reference a field from a geom with
`{"kind":"hfield","hfield":"terrain"}`. Hfield collision supports
sphere, capsule, and box pairs. Mesh and other primitive pairs are rejected.

## geoms

```json
{
  "name": "ground",
  "shape": {"kind": "plane"},
  "attach": {"kind": "static"},
  "local_offset": [0, 0, 0],
  "local_orientation": [0, 0, 0, 1],
  "friction": 0.6,
  "solref": {"timeconst": 0.02, "dampratio": 1.0},
  "solimp": {"dmin": 0.9, "dmax": 0.95, "width": 0.001, "midpoint": 0.5, "power": 2},
  "condim": 3,
  "margin": 0.0,
  "gap":    0.0
}
```

- **shape** — one of:
    - `{"kind":"plane"}`
    - `{"kind":"sphere","radius":r}`
    - `{"kind":"box","half_extents":[hx,hy,hz]}`
    - `{"kind":"capsule","radius":r,"half_height":h}`
    - `{"kind":"cylinder","radius":r,"half_height":h}` (v1)
    - `{"kind":"ellipsoid","semi_axes":[ax,ay,az]}` (v1)
    - `{"kind":"mesh","mesh":"NAME"}` (v1 — references the top-level
      `meshes` asset table)
    - `{"kind":"hfield","hfield":"NAME"}` (v3 — references the top-level
      `hfields` asset table)
- **attach** — one of:
    - `{"kind":"static"}` — only plane geoms may be static.
    - `{"kind":"body","body":"NAME"}` — attaches to a free body.
    - `{"kind":"link","tree":"NAME","link":"NAME"}` — attaches to a
      tree link.
- **local_offset**, **local_orientation** — geom origin relative to the
  parent's body frame (or world for static). Optional.
- **friction** — Coulomb coefficient, ≥ 0. Default `0.5`.
- **solref** — MuJoCo-style `(timeconst, dampratio)`. Optional; default
  is `SolRef::DEFAULT` (`timeconst = 0.02`, critical damping). See
  [`docs/contacts.md`](contacts.md).
- **solimp** — MuJoCo-style 5-parameter impedance sigmoid. Optional;
  default is `SolImp::DEFAULT`. Consulted when `solver.mode` is
  `"pgs"` or `"newton"`. See [`docs/solver.md`](solver.md).
- **condim** — Contact dimensionality. `1` (frictionless), `3`
  (normal + 2 tangents, sliding friction), `4` (adds torsion about
  the normal), or `6` (adds two rolling rows about the tangents).
  Default `3`. Consulted when `solver.mode` is `"pgs"` or `"newton"`. See
  [`docs/solver.md`](solver.md) for the row structure.
- **torsional_friction** — Coulomb coefficient about the contact
  normal, ≥ 0. Default `0`. Only read when the pair's condim ≥ 4;
  paired via `min` with the other geom's value.
- **rolling_friction** — Coulomb coefficient about the contact
  tangents, ≥ 0. Default `0`. Only read when the pair's condim = 6;
  paired via `min` with the other geom's value.
- **margin** — MuJoCo-style contact activation zone (m), ≥ 0. Default `0`.
  See [`docs/contacts.md`](contacts.md) for semantics.
- **gap** — MuJoCo-style force-free zone (m), ≥ 0. Default `0`.

## solver (root object, optional)

Optional top-level constraint-solver configuration. Omitted → defaults
(`SolverMode::Penalty`, 20 iterations, pyramidal cone). Scenes using a
changed contact manifold can have a different trajectory under this default.

```json
{
  "solver": {
    "mode": "pgs",
    "iterations": 30,
    "cone": "pyramidal"
  }
}
```

- **mode** — `"penalty"` (default) or `"pgs"`. See
  [`docs/solver.md`](solver.md) for the model derivation.
- **iterations** — positive integer, number of PGS sweeps per step
  (no early exit — determinism). Default `20`.
- **cone** — `"pyramidal"` (default) or `"elliptic"`.

## equality (root array, optional)

Bilateral constraints solved alongside contacts in the PGS sweep. Four
kinds; every kind carries an optional per-constraint `solref` /
`solimp` (same schemas as on a geom). See [`docs/solver.md`](solver.md)
for the row-count / row-geometry table.

### connect — 3-DOF point coincidence

```json
{
  "kind": "connect",
  "body_a": "ball_1",      // body name, or "world" for a static-world anchor
  "body_b": "ball_2",
  "anchor_a": [0, 0, 0.1], // body-local (world-frame if body_a == "world")
  "anchor_b": [0, 0, -0.1],
  "solref": {"timeconst": 0.01, "dampratio": 1.0},
  "solimp": {"dmin": 0.99, "dmax": 0.999, "width": 0.001, "midpoint": 0.5, "power": 2}
}
```

At least one of `body_a` / `body_b` must reference a real body.

### weld — 6-DOF pose lock

```json
{
  "kind": "weld",
  "body_a": "a",
  "body_b": "b",
  "anchor_a": [0, 0, 0],
  "anchor_b": [0, 0, 0],
  "relative_orientation": [0, 0, 0, 1]  // optional; default IDENTITY
}
```

`relative_orientation` locks the child-in-parent quaternion
(`q_B_world = q_A_world · relative_orientation`). Default `IDENTITY`
locks the two body frames parallel.

### joint — polynomial coupling of two 1-DOF joints on the same tree

```json
{
  "kind": "joint",
  "tree": "linkage",
  "joint_a": "follower",   // link name (hinge or slide) in the tree
  "joint_b": "crank",
  "polycoef": [0, 2, 0]    // q_a = c0 + c1·q_b + c2·q_b²  (up to quadratic)
}
```

Cross-tree coupling is out of scope this tier — both joints must live
on the same tree, and both must be hinge or slide (ball is rejected).

### distance — fixed anchor-to-anchor distance

```json
{
  "kind": "distance",
  "body_a": "a",
  "body_b": "b",
  "anchor_a": [0, 0, 0],
  "anchor_b": [0, 0, 0],
  "distance": 1.0
}
```

`distance` must be ≥ 0. The constraint elides its row for one step
when the current separation drops below 1 µm (degenerate direction);
it re-engages as soon as the pair separates past the guard.

## sites

Named points on a body or tree link with a body-local pose. The world
pose is queried at any time via
[`Scene::site_pose("name")`](../src/model.rs).

```json
{
  "name": "left_heel",
  "attach": {"kind": "link", "tree": "walker", "link": "left_foot"},
  "local_offset": [0.0, -0.08, 0.0],
  "local_orientation": [0, 0, 0, 1]
}
```

Sites cannot be `static`. `local_orientation` is optional (identity by
default).

## actuators (PD position servos)

```json
{
  "name": "shoulder_servo",
  "type": "position",
  "tree": "arm",
  "link": "shoulder",
  "kp":                 200.0,
  "dampratio":          1.0,
  "reflected_inertia":  0.25,
  "clamp":              60.0,
  "target":             0.0
}
```

- **type** — one of `"position"`, `"velocity"`, `"motor"`,
  `"general"`. Everything else is rejected. See
  [`docs/actuators.md`](actuators.md) for the model formulas.
- **tree** / **link** — must reference a hinge OR slide link. Ball and
  free/fixed targets are rejected at load (actuators are single-DOF).
- **target** — initial `ctrl` value at load time (all types).
- **clamp** — symmetric force clamp magnitude; `<= 0` disables (all
  types except `general`, which uses a `forcerange` array instead).

Type-specific fields:

- **position** — `kp` required. One of `kd` OR (`dampratio` +
  `reflected_inertia`). `dampratio` derives `kd = 2·ζ·√(kp · I_ref)`
  — matches MuJoCo's `<position dampratio="…"/>` idiom.
- **velocity** — `kv` required. Torque = `kv · (ctrl − qdot)`.
- **motor** — optional `gear` (default `1.0`). Torque = `gear · ctrl`.
- **general** — `gaintype ∈ {"fixed", "affine"}`, `gainprm` (3-array),
  `biastype ∈ {"none", "affine"}`, `biasprm` (3-array), `gear`,
  `dyntype ∈ {"none", "filter"}`, `dynprm` (scalar; filter tau for
  `dyntype="filter"`), optional `ctrlrange` and `forcerange`
  (2-arrays `[lo, hi]` with `lo < hi`).

## contact_pairs

Optional. When present, replaces the world's auto-generated pair list.

```json
{"explicit": [ {"a": "left_foot_geom", "b": "ground"} ]}
```

or

```json
{"disable":  [ {"a": "hand_geom", "b": "torso_geom"} ]}
```

- **explicit** — use ONLY the listed pairs (auto-generation is
  suppressed).
- **disable** — start from the auto list (including the `self_collide`
  filter per tree) and subtract the listed pairs.
- The two are mutually exclusive: passing both is a load-time error
  (`contact_pairs: "explicit" and "disable" are mutually exclusive`).

## validation coverage

Every branch below is asserted by a dedicated test in
[`newt/src/model.rs::tests`](../src/model.rs) or
[`tests/model_load.rs`](../tests/model_load.rs).

| error                                                        | example test                                         |
|--------------------------------------------------------------|------------------------------------------------------|
| unknown top-level field                                       | `top_level_unknown_field_rejected`                   |
| duplicate top-level key                                       | `duplicate_top_level_field_rejected`                 |
| unknown link field (typo like `"dampign"`)                    | `unknown_link_field_rejected`                        |
| negative or zero mass                                         | `nonpositive_mass_rejected`                          |
| non-positive diagonal inertia                                 | `negative_inertia_axis_rejected`                     |
| wrong-length tensor                                           | `wrong_length_tensor_rejected`                       |
| dangling parent link                                          | `dangling_parent_rejected`                           |
| `hinge` joint at the root                                     | `hinge_root_rejected`                                |
| zero hinge axis                                               | `zero_hinge_axis_rejected`                           |
| hinge range `lo ≥ hi`                                         | `hinge_range_lo_ge_hi_rejected`                      |
| actuator targeting a non-hinge link                           | `actuator_on_non_hinge_rejected`                     |
| actuator missing both `kd` and `dampratio`                    | `actuator_missing_kd_and_dampratio_rejected`         |
| duplicate actuator name                                       | `duplicate_actuator_name_rejected`                   |
| site referencing a missing body                               | `site_on_missing_body_rejected`                      |
| plane geom with a non-static attachment                       | `plane_with_body_attach_rejected`                    |
| duplicate body name                                           | `duplicate_body_name_rejected`                       |
| non-positive timestep                                         | `negative_timestep_rejected`                         |
| unsupported version string                                    | `version_mismatch_rejected`                          |
| contact pair referencing a missing geom                       | `contact_pair_unknown_geom_rejected`                 |
| contact_pairs with both `explicit` and `disable`              | `contact_pairs_explicit_and_disable_together_rejected` |
| zero slide axis                                                | `zero_slide_axis_rejected`                             |
| slide joint at the root                                        | `slide_at_root_rejected`                               |
| ball joint at the root                                         | `ball_at_root_rejected`                                |
| ball joint with a `range` (deferred to the v1 solver)          | `ball_with_range_rejected_with_solver_deferral_hint`   |
| PD actuator on a ball joint (non-1-DOF target)                 | `actuator_on_ball_rejected`                            |

Plus the parser's own layer:
[`newt/src/json.rs::tests`](../src/json.rs) pins malformed inputs
(unterminated strings, trailing commas, `01` / `1.` / `1e` numbers,
runaway nesting, `1e9999` overflow).

## packaged models

- [`models/pendulum.json`](../models/pendulum.json) — double pendulum
  (same numbers as `examples/pendulum.rs`).
- [`models/arm.json`](../models/arm.json) — 3-link commanded arm
  (identical numbers to `examples/arm.rs`, byte-verified by
  `arm_json_matches_programmatic_construction_exactly` in
  [`tests/model_load.rs`](../tests/model_load.rs)).
- [`models/stack.json`](../models/stack.json) — three
  0.35 m boxes with mixed masses (1.2 / 0.9 / 1.5), a small +0.02 m
  x-shift on the middle box, and a `(0, 0.3, 0)` body-frame
  initial angular velocity on the top box. Same symmetry-break pattern
  as [`tests/contacts_golden.rs`](../tests/contacts_golden.rs). The
  earlier draft of stack.json broke symmetry via small yaw rotations
  on the middle and top boxes — that tickled a **latent tier-2
  limitation**: [`contact::box_box`](../src/contact.rs) is a
  vertex-only SAT that misses edge-edge intersections between
  rotated boxes (any nonzero yaw drops all box-box contact
  candidates even when the boxes clearly overlap; the collapse is
  continuous with yaw, not gated at any particular angle). The tier-2 box-box golden
  path only exercises axis-aligned boxes, so the bug never surfaced
  before; the tier-5 round-trip test caught it here. Fixing the
  box-box narrow phase is a tier-2 follow-up (needs a proper SAT or
  MPR implementation); until then, rotated free-body-vs-free-body
  contact is not supported. The stack golden records the exact plane-box
  manifold path; upper boxes may redistribute onto the plane after a
  manifold update. Golden is
  [`tests/goldens/model_stack.bin`](../tests/goldens/model_stack.bin);
  regen with `cargo test regenerate_stack_golden -- --ignored
  --nocapture` on macOS-aarch64.

## running the demos

```sh
# Any packaged model, wireframe render.
cargo run --release --example load -- --model models/arm.json      --frames 900 --out /tmp/newt-arm.ppm
cargo run --release --example load -- --model models/pendulum.json --frames 900 --out /tmp/newt-pendulum.ppm
cargo run --release --example load -- --model models/stack.json    --frames 800 --out /tmp/newt-stack.ppm
sips -s format png /tmp/newt-stack.ppm --out /tmp/newt-stack.png

# The tier-4 arm demo now loads from arm.json too (three-waypoint reach + trail).
cargo run --release --example arm  -- --frames 1800 --out /tmp/newt-arm-waypoints.ppm
```

## verification

The standard fmt / clippy / test / libm-free grep quad — same as
tiers 1–4:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
grep -rnE '\.(sin|cos|tan|exp|ln|powf)\(' newt/src/ \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'
```

The grep must print nothing. `.github/workflows/newt.yml` runs the same
four checks on every push and PR.
