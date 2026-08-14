# MJCF loader (v1 tier 7)

`crate::mjcf::load_mjcf_str` / `load_mjcf_path` parses a documented subset
of MuJoCo's XML model format and returns the same [`Scene`](crate::model::Scene)
the JSON loader produces. Real MuJoCo models that stay within the subset
load unchanged; features outside the subset produce a clean error naming
the offender — the loader NEVER silently ignores a tag or attribute (see
the "no silent ignore" doctrine below).

Byte-identical trajectory anchors live in `tests/mjcf_anchor.rs`: three
MJCF fixtures (`models/pendulum.xml`, `models/arm.xml`, `models/stack.xml`)
mirror their JSON counterparts and step to bit-identical states for 600
steps. If you touch either loader, keep the anchors green.

## Doctrine

- **No silent ignore.** Every unknown attribute or child element on a
  supported node errors out with the attribute or element name. Every
  known-but-unsupported feature (mesh geom, tendon, keyframe, asset,
  visual…) errors out with an `unsupported in v1 subset` message
  naming the feature.
- **`<inertial>` first, geoms second.** Body inertia comes from
  `<inertial>` when present; otherwise the loader sums solid inertias
  from mass-carrying `<geom>` children (auto-inertia). Auto-inertia
  only handles geoms at `pos="0 0 0"` with identity orientation —
  supply an explicit `<inertial>` for shifted / rotated geoms.
- **Zero-dependency parser.** `crate::xml` is a hand-written recursive-
  descent XML parser, same doctrine as `crate::json`: no libm, no
  dependencies, position-tracked errors, malformed-input tested. See
  `xml.rs` for the module docstring.

## Supported elements

### Top level (children of `<mujoco>`)

| Element      | Notes                                                     |
| ------------ | --------------------------------------------------------- |
| `<compiler>` | Attributes below.                                         |
| `<option>`   | Attributes below.                                         |
| `<default>`  | Nested class inheritance; see [Defaults + classes](#defaults--classes). |
| `<worldbody>`| Static geoms plus top-level `<body>` roots.               |
| `<actuator>` | `<position>` and `<motor>` children.                      |
| `<sensor>`   | Sensor kinds listed below.                                |
| `<equality>` | `<connect>`, `<weld>`, `<joint>` children.                |
| `<contact>`  | `<pair>` XOR `<exclude>` children (exclusive).            |

Rejected top-level elements (clean `unsupported in v1 subset` error):
`<asset>`, `<tendon>`, `<keyframe>`, `<custom>`, `<visual>`, `<size>`,
`<statistic>`, `<extension>`, `<include>`.

### `<compiler>`

| Attribute     | Supported values                    | Notes |
| ------------- | ----------------------------------- | ----- |
| `angle`       | `radian` (default) or `degree`      | In `degree` mode `<joint range>`, `<body euler>`, and `<body axisangle>` angles are multiplied by π/180 at parse time. |
| `coordinate`  | `local` only                        | `global` errors out; the fixtures we mirror all use local. |
| `eulerseq`    | `xyz` only                          | Other sequences error out. |

Any other `<compiler>` attribute (autolimits, meshdir, texturedir,
boundmass, boundinertia, settotalmass, inertiafromgeom, usethread…)
errors out with the attribute name.

### `<option>`

| Attribute    | Supported values                              | Notes |
| ------------ | --------------------------------------------- | ----- |
| `timestep`   | positive float                                | Sets `World::dt`. |
| `gravity`    | `x y z`                                       | Sets `World::gravity`. |
| `integrator` | `RK4` / `rk4`                                 | Others error (only RK4 is implemented). |
| `cone`       | `pyramidal` or `elliptic`                     | Maps to `World::solver.cone`. |
| `iterations` | positive integer                              | Maps to `World::solver.iterations`. |
| `solver`     | `PGS` / `pgs`                                 | Switches `World::solver.mode` to `Pgs`. Newt's default stays `Penalty` when this attribute is absent. |

Everything else (`wind`, `magnetic`, `density`, `viscosity`, `impratio`,
`o_margin`, `o_solref`, `o_solimp`, `tolerance`, `noslip_*`, `mpr_*`,
`collision`, `jacobian`, `apirate`…) errors out with the attribute name.

### `<worldbody>`

Children: `<geom>` (static), `<body>` (roots). `<site>` at the world
root, `<light>`, `<camera>`, `<frame>` — rejected.

### `<body>`

| Attribute      | Notes |
| -------------- | ----- |
| `name`         | Required. Body names are unique across the whole scene. |
| `pos`          | Body origin in parent frame; default `0 0 0`. |
| `quat`         | MuJoCo `w x y z` order; converted to newt `(x, y, z, w)`. Only the ROOT body of a tree may have non-identity orientation (newt's `joint_offset_in_parent.orientation` is IDENTITY-only for non-root links). |
| `euler`        | `x y z` intrinsic (compiler `eulerseq="xyz"`), degree-scaled per `<compiler>`. |
| `axisangle`    | `ax ay az angle`. |
| `childclass`   | Default class name for descendant elements. |

Child elements: `<inertial>` (optional, see [Body inertia](#body-inertia)),
`<freejoint>` OR `<joint>` (at most one — v1 subset does not support
compound joints), `<geom>` and `<site>` (any count, attached to this
body/link), nested `<body>` (child links).

Newt extension: `<velocity linear="x y z" angular_body="x y z"/>` — a
free-body-only child element that seeds the initial linear /
body-frame angular velocity. Mirrors the JSON model's `"velocity"`
field so `stack.xml` can hit the byte-identical trajectory anchor.
Not part of stock MJCF.

### Body inertia

For each body:

1. If `<inertial>` is present, use it. Required attributes:
   `mass` (> 0) plus one of `diaginertia="Ixx Iyy Izz"` (3 numbers)
   or `fullinertia="Ixx Iyy Izz Ixy Ixz Iyz"` (6 numbers). Any other
   attribute (`pos`, `quat`) must equal the identity value — newt
   requires COM at the body origin and inertia expressed in body
   axes; a non-zero `<inertial pos>` errors out.
2. If `<inertial>` is absent, sum solid inertias from `<geom>`
   children that carry a `mass` attribute. Contributing geoms must
   sit at `pos="0 0 0"` with identity orientation (a rotated /
   offset contributor errors out with the message pointing at
   `<inertial>`).

### `<joint>` / `<freejoint>`

`<freejoint>` is only valid as the sole joint on a top-level body
(and becomes newt's `JointKind::Free`). `<joint>` supports:

| Attribute  | Supported values                       | Notes |
| ---------- | -------------------------------------- | ----- |
| `type`     | `hinge` (default), `slide`, `ball`     | `free` at non-root errors out. |
| `pos`      | joint pivot in body-local frame        | Default `0 0 0`. |
| `axis`     | `x y z`, non-zero                      | Required for hinge / slide; not allowed on ball. |
| `range`    | `lo hi`, degree-scaled if `angle="degree"` | Silent when `limited="false"`. |
| `damping`  | ≥ 0                                    | |
| `armature` | ≥ 0                                    | |
| `limited`  | `true` (default) / `false`             | `false` disables the range even if specified. |
| `class`    | class name                             | Overrides the context class for this joint. |
| `ref`      | `0` only                               | Non-zero `ref` is not supported (would shift q0). |

### `<geom>`

| Attribute    | Notes |
| ------------ | ----- |
| `name`       | Required. Unique across the scene. |
| `type`       | `plane` (static only), `sphere`, `box`, `capsule`, `cylinder`, `ellipsoid`. `mesh`, `hfield`, `sdf` error out. |
| `pos`        | Local offset (or `fromto` for capsule / cylinder — see below). |
| `quat` / `euler` / `axisangle` | Local orientation (same forms as `<body>`). |
| `size`       | Shape-dependent: `radius` (sphere), `hx hy hz` (box), `radius half-length` (capsule / cylinder), `ax ay az` (ellipsoid). Plane sizes are accepted but ignored (newt planes are infinite). |
| `fromto`     | `x1 y1 z1 x2 y2 z2` for capsule / cylinder — sets the center to the midpoint, half-length to `|b-a|/2`, and orientation to align local `+Z` with `b-a`. Overrides the second `size` number. |
| `friction`   | `μ [μ_torsional [μ_rolling]]`. |
| `solref`     | `timeconst dampratio`. |
| `solimp`     | `dmin dmax width` or `dmin dmax width midpoint power`. |
| `condim`     | `1`, `3`, `4`, or `6`. |
| `margin` / `gap` | ≥ 0. |
| `mass`       | Optional; feeds body auto-inertia (see [Body inertia](#body-inertia)). |
| `class`      | Overrides the context class. |

Rendering / filtering attributes (`material`, `rgba`, `group`,
`density`, `contype`, `conaffinity`) error out with the attribute
name. Use `<contact>` for explicit pair / exclude filtering.

### `<site>`

| Attribute    | Notes |
| ------------ | ----- |
| `name`       | Required, unique. |
| `pos`        | Local offset in the parent body's frame. |
| `quat` / `euler` / `axisangle` | Local orientation. |
| `size`       | Accepted (rendering hint) but not stored — sites are pose-only in newt. |
| `class`      | Overrides the context class. |

### `<actuator>`

| Child        | Attributes                                                                | Notes |
| ------------ | ------------------------------------------------------------------------- | ----- |
| `<position>` | `name`, `joint`, `kp`, `kv` OR `dampratio`, `forcerange`, `ctrlrange`, `class`, `ctrllimited`, `forcelimited` | `forcerange="lo hi"` maps to a symmetric `clamp = min(|lo|, |hi|)`. `ctrlrange` is accepted but not enforced (newt takes a scalar target). `dampratio` uses a unit reflected-inertia estimate (`kd = 2·dr·sqrt(kp·1)`); pass `kv` for exact control over the derivative gain. |
| `<motor>`    | `name`, `joint`, `gear`, `forcerange`, `ctrlrange`, `class`, …            | Modeled as a `PdServo` with `kp=0, kd=0`; the caller writes the desired torque into `target` per step. Only the first `gear` scalar is honored. |

Other actuator kinds (`<general>`, `<velocity>`, `<cylinder>`,
`<damper>`, `<muscle>`, `<intvelocity>`) error out.

### `<sensor>`

Child element name is the sensor kind. Supported kinds — the same set
the JSON loader knows:

| Kind             | Attributes  | Attaches to |
| ---------------- | ----------- | ----------- |
| `jointpos`       | `name`, `joint` | 1-DOF joint (hinge / slide). |
| `jointvel`       | `name`, `joint` | 1-DOF joint. |
| `ballquat`       | `name`, `joint` | ball joint. |
| `ballangvel`     | `name`, `joint` | ball joint. |
| `framepos`       | `name`, `site`  | site. |
| `framequat`      | `name`, `site`  | site. |
| `gyro`           | `name`, `site`  | site. |
| `accelerometer`  | `name`, `site`  | site. |
| `touch`          | `name`, `geom`  | geom (newt indexes by geom, not by site). |
| `force`          | `name`, `body`  | tree-link body. |
| `torque`         | `name`, `body`  | tree-link body. |

### `<equality>`

| Child       | Attributes                                                        | Notes |
| ----------- | ----------------------------------------------------------------- | ----- |
| `<connect>` | `name`, `body1`, `body2`, `anchor`, `solref`, `solimp`            | `body1`/`body2` = `"world"` or omitted references the world. Only free-body endpoints are supported (tree-link endpoints error). |
| `<weld>`    | above plus `relpose` (7-tuple `px py pz qw qx qy qz`)             | Relative orientation defaults to identity. |
| `<joint>`   | `name`, `joint1`, `joint2`, `polycoef`, `solref`, `solimp`         | Both joints must live in the same tree. `polycoef` accepts 1–3 numbers (`c0`, `c1`, `c2`); higher-degree entries error out. |

`<distance>`, `<tendon>`, `<flex>` error out.

### `<contact>`

`<pair>` XOR `<exclude>` — combining both errors out. `<pair>` sets an
explicit pair list; `<exclude>` subtracts body-pair-owned geoms from
the auto pair list. Attributes: `pair` takes `name`, `geom1`, `geom2`,
`condim`, `friction`, `margin`, `solref`, `solimp` (only `geom1` and
`geom2` currently drive the pair list — the other pair-level overrides
are accepted but not applied yet, an ergonomic hole to close in a
follow-up ticket if it bites).

## Defaults + classes

MJCF `<default>` lets you factor out attribute defaults per element
kind. The loader follows MuJoCo's resolution order:

1. Every element scoped to the **main** class unless explicitly
   overridden. The top-level `<default>` without a `class` attribute
   defines main; a nested `<default class="X">` defines class `X`.
2. An element's `class="X"` attribute switches its lookup to class
   `X` (for THAT element only).
3. A `<body childclass="X">` scopes class `X` as the default for all
   descendant elements (recursively) unless a descendant overrides
   with its own `class="…"`.
4. Nested `<default class="X">` inherits from its enclosing default —
   attributes not set on `X` fall back to the parent class.

Attribute lookup order for an element:

1. The element's own attribute.
2. The active class's per-element default list.

Anything else (or missing) means the attribute has no value; a
required attribute at that point errors out.

## Errors

Every failure is a `mjcf::MjcfError { path, message }` where `path` is
an XML-style breadcrumb (e.g. `mujoco > worldbody > body[torso] >
joint[left_hip]`). Malformed XML surfaces as an `<xml>` path with the
underlying `xml::Error` offset in the message.

## Rendering the biped

```text
cargo run --release --example load -- \
  --model models/biped-simple.xml --frames 400 --out /tmp/biped.ppm
```

Renders the biped as a torso wireframe plus leg-link rods after
stepping it 400 times (2 s at `dt=0.005`). See
`tests/mjcf_load.rs::biped_simple_loads_and_stands_for_2000_steps`
for the automated smoke that asserts the root stays finite and
bounded across 2000 steps. `models/biped-simple.xml` carries a
provenance comment pointing at `~/me/fun/biped/biped/models/biped.xml`
(the reference biped) and lists every simplification applied to fit
the v1 subset.
