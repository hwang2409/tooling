# Sensors (v2 tier 4)

Sensors let a scene observe its own state without perturbing it. Each
sensor produces a fixed-width contribution to a flat `sensordata: Vec<f32>`
vector on [`crate::world::World`]; the layout is pinned by declaration
order and is stable across steps and platforms.

Sensors are opt-in. A scene with no sensors declared skips the entire
sensor pipeline in `World::step`, so every pre-v1-tier-6 golden path is
bit-for-bit unchanged.

## Semantics per kind

| Kind | Dim | Reference | Frame | Formula (post-step state) |
|------|-----|-----------|-------|---------------------------|
| `jointpos` | 1 | hinge or slide link | joint | `q` at the link's DOF slot |
| `jointvel` | 1 | hinge or slide link | joint | `qdot` at the link's DOF slot |
| `ballquat` | 4 | ball joint link | child-in-parent | `(x, y, z, w)` from `q` |
| `ballangvel` | 3 | ball joint link | child body | `(ωx, ωy, ωz)` from `qdot` |
| `framepos` | 3 | site | world | `p_parent + R_parent · offset` |
| `framequat` | 4 | site | world | `q_parent · local_orientation` |
| `gyro` | 3 | site | site | `R_site^T · ω_parent_body` |
| `accelerometer` | 3 | site | site | see below |
| `touch` | 1 | geom | scalar | Σ normal-force magnitudes on the geom |
| `force` | 3 | non-root link | child body at joint | force parent joint transmits to child |
| `torque` | 3 | non-root link | child body at joint | torque parent joint transmits to child |
| `velocimeter` | 3 | site | site | site linear velocity rotated into the site frame |
| `magnetometer` | 3 | site | site | global magnetic field rotated into the site frame |
| `rangefinder` | 1 | site | scalar | nearest hit along site +Z, or `-1` |
| `subtreecom` | 3 | tree link | world | mass-weighted COM of the link subtree |
| `framelinvel` | 3 | site | world | site linear velocity |
| `frameangvel` | 3 | site | world | parent angular velocity |

For the site-frame kinds, `local_orientation` is the child-in-parent
quaternion: a vector expressed in the site's local frame maps to the
parent body frame via `v_body = local_orientation · v_site`. This is the
same convention [`crate::model::Site`] uses, so a scene loaded from JSON
can hand a `Site` straight into a sensor.

`World::magnetic_field` sets the global field. The default is
`[0, -0.5, 0]`, matching MuJoCo's default option field.

`rangefinder` casts a normalized ray from the site origin along local +Z.
It tests plane, sphere, box, capsule, cylinder, ellipsoid, and convex-mesh
surfaces. It returns the nearest non-negative distance, or `-1` when no
surface is hit. Geoms are visited in declaration order, so equal distances
use the lowest geom index.

### Accelerometer

The accelerometer reports the specific force at the site — what a real
IMU measures. In classical form:

```text
a_site_world  = a_com_world + α_world × r + ω_world × (ω_world × r)
a_proper      = a_site_world − g_world
reading_site  = R_site→world^T · a_proper
```

where `r` is the world-frame vector from the parent-body COM to the site
anchor, `α_world` and `ω_world` are the parent link's world-frame angular
acceleration and velocity, `a_com_world` is the world-frame linear
acceleration of the COM, and `g_world` is the world gravity vector.

Load-bearing consequences:

- A body sitting still under gravity has `a_com_world = 0` (contact
  cancels weight) so `a_proper = −g_world = (0, 0, +9.81)` — the classic
  `+g` upward reading in a world-aligned site frame.
- A body in free fall has `a_com_world = g_world` so `a_proper = 0`.
- A body rotating at constant ω with the site offset by `r` reads the
  centripetal term `ω × (ω × r)` in the site frame (gravity-free case).

Post-step accelerations are computed by re-running the same wrench
assembly and forward-dynamics code paths `World::step` uses at each RK4
sub-stage: for `SolverMode::Penalty` that is one `compute_wrenches` per
free body + one ABA per tree; for `SolverMode::Pgs` it is one
`compute_solver_wrenches` + one ABA per tree. This mirrors MuJoCo's
`mj_sensor` running after `mj_forward` on the post-step state.

### Force / Torque

`Force` and `Torque` report the interaction wrench the parent link must
transmit through the joint to produce the observed acceleration. Both are
in the child's body frame; force is at COM (translation-invariant), torque
is at the joint anchor (translated from the RNE per-link COM wrench:
`τ_at_joint = τ_at_COM − r_com_to_joint × F`). The RNE evaluation uses
the post-step `(q, qdot, qddot)`.

For a statically held horizontal arm (`q̇ = q̈ = 0` under gravity), the
force reading at the forearm elbow equals the weight of the forearm
projected into the child body frame, and the torque reading equals the
gravity-moment about the hinge axis. The
`force_torque_at_elbow_of_horizontal_2link_arm_matches_statics` anchor in
`tests/sensors_battery.rs` derives both by hand and checks the sensor
recovers them to `< 0.05 N` / `< 0.05 N·m`.

### Touch

The touch sensor sums normal-force magnitudes over every contact
involving the designated geom. Under `Penalty` the per-contact force is
`f_n = max(0, k · pen_eff − c · v_n)`; under `Pgs` it is the PGS
constraint impulse divided by `dt` (the actual force the constraint
solver applied that step). Same physical reading either way at
equilibrium — a body resting on the ground has `sum |f_n| ≈ m · g`.

## Evaluation timing (no perturbation)

`World::step` calls `evaluate_sensors` at the end of the step when the
sensor bank is non-empty. The evaluation reads bodies/trees/geoms and
recomputes derived quantities (contacts, wrenches, `qddot`, spatial
accelerations) at the post-step state, then writes into
`world.sensors.data`. It never modifies any simulation state, so a scene
with sensors declared vs one without takes byte-identical trajectories
under the same inputs. When `sensors` is empty the sensor pipeline is not
invoked at all.

The `stepping_with_sensors_matches_stepping_without_bit_for_bit` anchor
proves the no-perturbation contract across 500 steps of a tumbling
asymmetric-inertia body.

## Determinism

Sensor offsets are fixed by declaration order at
`World::add_sensor` / loader time. `sensordata` reads left-to-right in
declaration order (see `SensorBank::slice` for the per-sensor view). The
`sensor_golden_trajectory_is_byte_identical` anchor byte-compares a
mixed-scene `sensordata` snapshot at steps 1, 100, and 500 against a
macOS-aarch64 reference file. macOS regeneration is guarded to the
reference host, matching the existing golden regeneration policy.

## Model reference (JSON)

```json
{
  "sensors": [
    {"name": "shoulder_q",    "kind": "jointpos",     "tree": "arm", "link": "shoulder"},
    {"name": "wrist_omega",   "kind": "ballangvel",   "tree": "arm", "link": "wrist"},
    {"name": "tip_pos",       "kind": "framepos",     "site": "tip"},
    {"name": "tip_orient",    "kind": "framequat",    "site": "tip"},
    {"name": "tip_gyro",      "kind": "gyro",         "site": "tip_imu"},
    {"name": "tip_accel",     "kind": "accelerometer","site": "tip_imu"},
    {"name": "foot_touch",    "kind": "touch",        "geom": "foot_pad"},
    {"name": "elbow_force",   "kind": "force",        "tree": "arm", "link": "elbow"},
    {"name": "elbow_torque",  "kind": "torque",       "tree": "arm", "link": "elbow"},
    {"name": "tip_vel",       "kind": "velocimeter",  "site": "tip_imu"},
    {"name": "tip_mag",       "kind": "magnetometer", "site": "tip_imu"},
    {"name": "tip_range",     "kind": "rangefinder",  "site": "tip_imu"},
    {"name": "tip_com",       "kind": "subtreecom",   "tree": "arm", "link": "wrist"}
  ]
}
```

Field rules (loader in `model.rs`):

- Site kinds require a `site` name that references an earlier entry in
  the top-level `"sites"` array.
- `jointpos` / `jointvel` require a hinge or slide link. `ballquat` /
  `ballangvel` require a ball link. `force` / `torque` require a non-root
  link (needs a parent joint).
- `touch` requires a valid geom name.
- `subtreecom` requires a tree and link. All remaining v2 site sensors
  require a valid site name.
- Every unknown field is rejected with a JSON-path error, same as the
  rest of the loader.

## Demo readout

`newt/examples/arm.rs` runs the PPM waypoint demo and prints a
per-sample sensor table using the sensors declared in
`newt/models/arm.json`. A typical run:

```
frame |  shoulder_q  elbow_q   wrist_q   |  shoulder_qd  elbow_qd  wrist_qd  |  tip_pos            |  gyro                |  accel               |  elbow_force         |  elbow_torque
------+----------------------------------+---------------------------------+---------------------+----------------------+----------------------+----------------------+----------------------
   50 |     1.692    -0.869    -0.489 |     4.203     0.727     2.902 | ( 0.00,  1.03,  0.85) | (  7.83,   0.00,   0.00) | (  0.00, -275.96,  18.72) | (  0.00,  -5.55,   8.19) | (-27.594,  0.000,  0.000)
 ...
```

Rendering paths are unchanged; sensor sampling happens inside the demo's
tick loop via `world.evaluate_sensors(&[])`.
