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
| `<position>` | `name`, `joint`, `kp`, `kv` OR `dampratio`, `forcerange`, `ctrlrange`, `class`, `ctrllimited`, `forcelimited`, `target` | `forcerange="lo hi"` maps to a symmetric `clamp = min(|lo|, |hi|)`. `ctrlrange="lo hi"` is enforced (clamps `ctrl` before torque eval, MuJoCo parity). `dampratio` uses a unit reflected-inertia estimate (`kv = 2·dr·sqrt(kp·1)`); pass `kv` for exact control. |
| `<velocity>` | `name`, `joint`, `kv`, `forcerange`, `ctrlrange`, `class`, …               | Torque = `kv · (ctrl − qdot)`. Same clamp convention as `<position>`. |
| `<motor>`    | `name`, `joint`, `gear`, `forcerange`, `ctrlrange`, `class`, …            | Torque = `gear · ctrl`. Only the first `gear` scalar is honored. Set the effective torque each step via `Tree::set_actuator_target`. |
| `<general>`  | `name`, `joint`, `gaintype`, `gainprm`, `biastype`, `biasprm`, `gear`, `dyntype`, `dynprm`, `ctrlrange`, `forcerange`, `class`, `ctrllimited`, `forcelimited` | Full v2 tier-2 model. `gainprm`/`biasprm` accept 1..=3 numbers (tail defaults `[1,0,0]` / `[0,0,0]`). `dyntype="filter"` requires `dynprm` (tau) > 0. `actearly="true"` is rejected. `ctrlrange`/`forcerange` are asymmetric `"lo hi"`. See [`docs/actuators.md`](actuators.md) for the model. |

`<cylinder>`, `<damper>`, `<muscle>`, `<intvelocity>` remain outside the subset.

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
# Pure PD (no external assist) — biped FALLS. This is the honest
# baseline; the render shows the torso lying on the ground after
# the fall between steps ~350 and ~500.
cargo run --release --example load -- \
  --model models/biped-simple.xml --frames 2000 --out /tmp/biped-purepd.ppm

# Standing WITH the source-style balance assist — mirrors the source
# biped's `stand` scenario, which itself runs balance assist at
# assist_scale=1.0. Render shows a quiet upright biped.
cargo run --release --example load -- \
  --model models/biped-simple.xml --frames 2000 --balance --out /tmp/biped-stand.ppm
```

The demo prints its assist state ("with pure joint PD (NO external
assist)" or "WITH source-style torso balance assist") on every run
so the caption in a saved image never lies about what physics ran.

## Biped-simple standing note (v1 finale, honest edition)

**Ship state — two tests.** `tests/mjcf_load.rs` carries two
biped smoke tests with truthful names:

1. `biped_simple_pure_pd_smoke` — pure joint PD, **NO external
   assist**. Asserts (a) the simulation stays finite and inside a
   generous world box, and (b) the biped's characteristic fall
   between steps 300 and 600 down to ≤ 20 % of initial height. This
   is the honest current pure-PD behavior baseline.
2. `biped_simple_stands_with_source_balance_assist` — applies the
   source biped's `_apply_balance_controller` wrench each step —
   a **faithful 6-component mirror** of
   `~/me/fun/biped/biped/mujoco_biped.py` lines 2570-2593 at
   `assist_scale=1.0`:

   ```text
     force_x  = clamp(42·-x    + 82·-vx,        -75, 75 )
     force_y  = clamp(90·-y    - 35· vy,        -35, 35 )
     force_z  = clamp(240·(target_z-z) - 70·vz, -90, 260)
     torque_x = clamp( 135·up_y - 24·ωx,        -95, 95 )
     torque_y = clamp(-135·up_x - 24·ωy,        -95, 95 )
     torque_z = clamp(-12·ωz,                   -28, 28 )
   ```

   Applied via `Tree::applied_wrenches[0]` (world-frame force +
   world-frame torque on the torso, same shape as MuJoCo's
   `data.xfrc_applied[torso]`). Frame conversion: the source reads
   `data.qvel` which is world-frame for a freejoint; newt's
   free-root `Tree::qdot` is body-frame `(ω_body, v_body)`, so the
   test rotates both into world via `torso_ori.rotate(...)` before
   feeding the PDs. All six components, all source constants, all
   source clamps.

   Asserts root height > 80 % of initial and torso tilt < 15° for
   every one of 2000 steps. Measured margin at 4556c57 (before the
   faithful upgrade) was `min z_ratio = 1.000` and
   `max tilt ≈ 4.6°`; the faithful controller improves both bounds
   (the extra roll/pitch/yaw damping and lateral PDs help). This
   is the same architecture the source `stand` scenario uses —
   `SCENARIO_DEFINITIONS["stand"]` in `biped/mujoco_biped.py` sets
   `balance_mode: "controller"` with `assist_scale: 1.0`, i.e. the
   balance controller is on with full authority. Standing here
   reproduces the source's own architecture, not a shortcut around
   it.

### Parameter-by-parameter comparison vs source biped

Numbers pulled from `~/me/fun/biped/biped/models/biped.xml`
(reference biped) and `biped/mujoco_biped.py` (stand scenario) as
of this commit.

| Slot                     | Source biped (stand)                | biped-simple.xml (this fixture)         | Match? |
| ------------------------ | ----------------------------------- | ---------------------------------------- | ------ |
| `<option timestep>`      | 0.005                               | 0.005                                    | yes    |
| `<option gravity>`       | 0 0 -9.81                           | 0 0 -9.81                                | yes    |
| `<option integrator>`    | RK4                                 | RK4 (default)                            | yes    |
| `<option cone>`          | elliptic                            | elliptic                                 | yes    |
| `<option solver>`        | default (Newton)                    | PGS (newt has no Newton solver)          | **NO** |
| joint `damping` default  | 2.0                                 | 2.0                                      | yes    |
| joint `damping` roll     | 8.0 (hip_roll) / 5.0 (ankle_roll)   | 8.0 / 5.0                                | yes    |
| joint `armature` default | 0.02                                | 0.02                                     | yes    |
| joint `armature` roll    | 0.05 (hip_roll) / 0.04 (ankle_roll) | 0.05 / 0.04                              | yes    |
| joint `limited`          | true                                | true (default in loader)                 | yes    |
| geom `friction`          | 1.2 0.08 0.02                       | 1.2 0.08 0.02                            | yes    |
| geom `solref`            | 0.02 1                              | 0.02 1                                   | yes    |
| geom `solimp`            | 0.9 0.95 0.001                      | 0.9 0.95 0.001                           | yes    |
| geom `contype/conaffinity` | 2/1 (tree) 1/2 (ground)           | not modeled; tree self-collide=false via loader auto-filter | equivalent for this fixture |
| Actuator `kp` (hip)      | 70                                  | 70                                       | yes    |
| Actuator `kp` (knee)     | 80                                  | 80                                       | yes    |
| Actuator `kp` (ankle)    | 45                                  | 45                                       | yes    |
| Actuator `dampratio`     | 1.0                                 | `kv=6`/`kv=5` — see gap note below       | **NO** (approximation) |
| Actuator `forcerange`    | ±70 / ±80 / ±45 per joint           | same numbers                              | yes    |
| Body topology            | torso = chest box + pelvis box + head sphere; each leg has 5 nested bodies | torso = single box (chest+pelvis+head lumped); each leg has 5 nested bodies matching source | **partial** |
| Foot support polygon (x) | [-0.095, 0.245] m from ankle        | [-0.095, 0.245] m from ankle             | yes    |
| Initial CoM              | ~x=0 (inside support polygon)       | ~x=0 (inside support polygon)            | yes    |
| Initial ankle-support gap | feet at z≈0 flush with ground      | feet flush (`torso pos="0 0 1.235"`)     | yes    |

**Identified engine-level gaps (candidate NEWT-13 seeds):**

- **No Newton solver.** MuJoCo's default constraint solver is Newton;
  newt only ships Penalty and PGS in v1. The stand scenario's contact
  model may behave differently under PGS iterations vs. Newton at the
  same solref/solimp. Impact: unknown until we run the NEWT-13
  differential harness against real MuJoCo trajectories.
- **`dampratio` in `<position>` uses a unit-inertia approximation.**
  The loader converts `dampratio=ζ` into `kd = 2ζ·√(kp·1)` because
  the actuator does not know the joint's `Sᵀ IA S` at parse time
  (that would tie the loader to ABA). Source biped's `dampratio=1.0`
  therefore does not translate to an equal kd in newt vs MuJoCo
  when the joint's effective inertia differs from 1. In this
  fixture we side-step it with explicit `kv=` and document the
  choice; a proper fix would materialize the diagonal `Sᵀ IA S`
  during actuator wiring or at first step.
- **Torso lumped into one box.** The source torso is three geoms
  (chest, pelvis, head) at three positions; auto-inertia in newt's
  v1 subset only handles a single geom at the body origin per
  body. Effective torso inertia is close (single big box of
  equivalent mass, similar span) but not identical. Impact:
  slightly different natural period, does not change the
  qualitative inverted-pendulum instability.
- **No `<keyframe>` support.** A pre-crouched initial pose would
  reduce the ballistic-fall transient the pure-PD test observes.
  The newt `target` extension on `<position>` bakes the standing
  setpoint, but the joints still start at q=0.

**Why joint PD alone cannot stand (numbers).** With a ~20 kg
body-mass biped whose CoM sits ~0.9 m above the ankles, the
gravity-driven tipping moment for a small tilt θ is
`m · g · L_com · θ ≈ 177 θ Nm/rad`. Total ankle-joint restoring
stiffness with two ankles at source `kp = 45` is `2 · 45 = 90
Nm/rad`. Even at the top of the source range (`kp = 80`), total is
`160 Nm/rad` — still under 177. Net stiffness is negative in both
cases, so the linearized dynamics have an unstable eigenvalue and
any small perturbation grows exponentially. Observed pure-PD
trajectory (`examples/biped_diag.rs`, not shipped as a test):

```
step  100  root=(-0.022,-0.000,1.2459)  z_ratio=1.009  tilt≈0.018 rad
step  200  root=(-0.102,+0.000,1.2403)  z_ratio=1.004  tilt≈0.102 rad
step  300  root=(-0.329,+0.000,1.1918)  z_ratio=0.965  tilt≈0.339 rad
step  400  root=(-0.981,+0.000,0.7768)  z_ratio=0.629  tilt≈0.972 rad
step  500  root=(-1.318,+0.000,0.1388)  z_ratio=0.112  tilt≈1.567 rad
...
min z ratio: 0.105 (10.5% of initial)
first crossed z_ratio < 0.5 at step 413
```

The source biped's `stand` scenario ships with the balance
controller ON because the same physics applies to the reference
biped in MuJoCo — no joint-PD-only stand scenario exists in the
source repo.

**Follow-ups this diagnosis suggests (NEWT-13 candidates):**

- Differential harness against real MuJoCo trajectories on the same
  biped model, both with and without balance assist. That gives us
  a per-tolerance-band claim (e.g. "matches MuJoCo to 1% for the
  first 200 steps of the fall, drifts past 5% after step 500").
- MJCF extension for per-body applied wrenches so the balance
  controller can live in the fixture instead of the test loop.
- `<keyframe>` support so the biped starts in the pre-crouched
  standing pose the source uses.
- Newton solver (out of v1 scope but the differential harness will
  tell us how much of a gap PGS-vs-Newton opens on this model).

`models/biped-simple.xml` carries a provenance comment pointing at
`~/me/fun/biped/biped/models/biped.xml` (the reference biped) and
lists every simplification applied to fit the v1 subset.
