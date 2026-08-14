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
  "bodies":   [ ... ],
  "trees":    [ ... ],
  "geoms":    [ ... ],
  "sites":    [ ... ],
  "actuators":[ ... ],
  "contact_pairs": { ... }
}
```

- **version** — must be `"1"` when present. Missing is fine; any other
  value is rejected.
- **gravity** — `[x, y, z]`. Default `[0, 0, -9.81]`.
- **timestep** — RK4 step size in seconds. Must be > 0. Default `0.005`.
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
- **pose** — optional. `position` defaults to `[0,0,0]`, `orientation`
  to identity. Quaternions are `(x, y, z, w)` and are renormalized on
  load.
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
```

- **kind** = `free` (6-DOF root), `fixed`, or `hinge`.
- Hinges only: **axis** (required, normalized on load; zero rejected),
  **range** (optional `[lo, hi]`, lo < hi enforced), **damping**,
  **armature**, **limit.stiffness** / **limit.damping**.
- `free` is only valid on the root; `hinge` is invalid on the root.

## inertia

Three kinds. Every one produces the same body-frame `Mat3` about the
COM.

```json
{"kind": "diag",   "values": [Ixx, Iyy, Izz]}
{"kind": "tensor", "values": [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]}
{"kind": "solid",  "shape":  {"kind": "box",     "half_extents": [hx, hy, hz]}}
{"kind": "solid",  "shape":  {"kind": "sphere",  "radius": r}}
{"kind": "solid",  "shape":  {"kind": "capsule", "radius": r, "half_height": h}}
```

- Diagonal moments must be > 0.
- Tensor: symmetric; product moments (`Ixy`, `Ixz`, `Iyz`) may be any
  sign. Rejected if singular.
- Solid delegates to
  [`geom::solid_box_inertia`](../src/geom.rs) et al., using the same
  formulas the engine tests pin. Rod inertia isn't a builtin — write it
  by hand with `diag`.

## geoms

```json
{
  "name": "ground",
  "shape": {"kind": "plane"},
  "attach": {"kind": "static"},
  "local_offset": [0, 0, 0],
  "local_orientation": [0, 0, 0, 1],
  "friction": 0.6,
  "solref": {"timeconst": 0.02, "dampratio": 1.0}
}
```

- **shape** — one of `plane`, `sphere` (`radius`), `box`
  (`half_extents`), `capsule` (`radius`, `half_height`).
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

- **type** — v0 supports only `"position"`; anything else is rejected.
- **tree** / **link** — must reference a hinge link. Non-hinge targets
  are rejected at load.
- **kp**, **clamp**, **target** — as in
  [`newt::actuator::PdServo`](../src/actuator.rs).
- **kd** — direct velocity gain. Mutually exclusive with
  `dampratio` / `reflected_inertia`.
- **dampratio** — critical damping ratio. Requires
  `reflected_inertia`, from which `kd = 2·ζ·√(kp · I_ref)`. Matches
  MuJoCo's `<position dampratio="…"/>` idiom.

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
- The two are mutually exclusive; a single object cannot mix them
  through the current v0 grammar (specify one or the other).

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
  0.35 m boxes with mixed masses and small yaw offsets so the
  golden trajectory is symmetry-broken *inside the model file* (per the
  tier-2 lesson: symmetric boxes hide lever-arm bugs). Golden is
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
