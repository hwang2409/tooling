# Tendons (v2 tier 3)

Reference doc for `newt::tendon`. Covers the fixed and spatial tendon
models, sphere/cylinder-wrap geometry, pulley branches, actuator transmission,
and PGS limit rows.

MuJoCo is the reference — every semantic choice below either matches
MuJoCo's tendon path or is called out as a documented, tested deviation.

## The two tendon kinds

### Fixed tendon

Length is a linear combination of scalar joint coordinates:

```text
L    = Σ_i coef_i · q_i
Ldot = Σ_i coef_i · qdot_i
```

`i` ranges over a set of hinge or slide joints in the SAME tree. The
Jacobian row `dL/dqdot` is CONSTANT in `q` — the `coef_i` values, placed
at the `v_offset[link_i]` slots. This makes the fixed tendon the ideal
coupling primitive: two joints with `coef = [1, -1]` and a stiff
passive spring behave, in the stiff limit, like a MuJoCo `<equality
joint>` with `polycoef = [0, 1, 0]`. The `fixed_tendon_stiff_coupling_
anchor_matches_equality` test in `tests/tendon_battery.rs` anchors this
against the independent equality-constraint machinery.

### Spatial tendon

A chain of sites (fixed to bodies or the world) connected by straight
segments, optionally deflected by a wrap obstacle:

```text
L = Σ_k length_k
```

Each site is `(link, position_local)`. Each segment between adjacent
sites optionally carries a sphere or infinite-cylinder wrap. A tendon
without a pulley has one branch. Pulley branches contribute
`branch_length / divisor`.

## Sphere wrap: the two-tangent-arc construction

For a segment `AB` deflected by a sphere at `C` with radius `R`:

- Compute `d_A = |A − C|`, `d_B = |B − C|`. Bail on wrap (fall back to
  a straight segment) if either endpoint is INSIDE the sphere — the
  configuration is ill-posed.
- Find the closest point on segment `AB` to `C`. If its distance
  exceeds `R`, or the closest point lies outside the segment, wrap does
  NOT engage — the straight `|B − A|` is the length.
- Otherwise, in the plane spanned by `CA` and `CB` (unique when the
  three points are non-colinear):

  ```text
  t_A = √(d_A² − R²)                  (length of the tangent segment from A)
  t_B = √(d_B² − R²)
  φ_A = arcsin(t_A / d_A) = arccos(R / d_A)   (angle at C from CA to CT_A)
  φ_B = arcsin(t_B / d_B)
  γ   = angle ACB                     (interior angle at C, in radians)
  θ   = max(0, γ − φ_A − φ_B)         (wrap arc angle at C)
  L   = t_A + R · θ + t_B             (segment length under wrap)
  ```

  The tangent points:

  ```text
  x̂_A = (A − C) / d_A                 (unit from C toward A, in-plane)
  ŷ_A = (CB − (CB · x̂_A) x̂_A) / ‖...‖  (in-plane perpendicular toward B)
  T_A = C + (R² / d_A) · x̂_A + (R · t_A / d_A) · ŷ_A
  ```

  and symmetrically for `T_B`.

- Wrap engagement is a discrete switch on `d_perp - R`. `L(q)` is
  continuous at the boundary (as `θ → 0` the arc collapses and
  `t_A + t_B → |AB|`). The Jacobian is continuous too — the envelope
  theorem gives `dL/dA = −(unit tangent from A toward T_A)` on the
  wrap side and `dL/dA = −(unit toward B)` on the straight side, and
  the two limits coincide at the transition — but the FORMULA path
  changes. `wrap_engage_transition_length_continuous` in
  `tests/tendon_battery.rs` anchors this.

- Determinism: the switch is a total function of `(A, B, C, R)`. No
  time hysteresis. Trajectories that oscillate exactly on the boundary
  are pinned to the straight branch by the strict inequality
  `d_perp² < R²`.

### Sidesite

Sidesite is required by MuJoCo only for cylinder and pulley wraps and
for sphere wraps in the degenerate case where `A`, `B`, `C` are colinear.
The natural sphere wrap has a unique arc in the ABC plane (the shorter
one, on the side "away" from segment `AB`) so a sidesite is optional in
the general case.

When `|CA × CB| / (|CA| · |CB|)` falls below
`SIDE_HINT_COLINEARITY_EPS = 1e-4`, we treat `A`, `B`, `C` as colinear.
With a `side_hint_world` present, we build an artificial wrap plane
containing `AB` and the perpendicular component of the hint. With no
hint, the endpoint-inside-sphere case is ill-posed, so newt falls back
to the straight segment and ignores the sphere. MuJoCo raises a
compile-time error for this configuration.

## Cylinder wrap

MuJoCo treats a cylinder wrap as infinite along its local z axis. Newt
projects both endpoints into the plane normal to that axis and applies the
2D circle construction. The shortest tangent pair wins. A sidesite selects
the pair whose circular arc points toward the projected sidesite.

MJCF sidesites on another fixed tree are accepted and stored as world-static
sites. Cross-tree sidesites with a moving joint remain rejected because their
world dependency cannot fit one tree's tendon Jacobian.

The 2D tangent points are lifted back to 3D. Their axial coordinates divide
the endpoint axial change in proportion to the 2D path lengths. The central
arc length uses `sqrt(arc² + axial_change²)`. This matches MuJoCo's
`mju_wrap` construction and stays continuous at engagement.

The envelope theorem supplies endpoint gradients from the 3D tangent lines.
It also supplies the wrapping geom body contribution. Translation uses the
sum of the two tangent forces. Rotation uses each force at its wrap point.

## Pulley branches

Each `<pulley divisor="d"/>` starts a new MJCF branch. The branch before
the first pulley has divisor 1. Every branch length and Jacobian row is
divided by its own divisor before summation. A physical 2:1 pulley uses
one common branch with divisor 1 and two child branches with divisor 2.

## Actuator transmission on tendons

An actuator can target a tendon via `Actuator::on_tendon(tendon_idx)`.
The evaluation model is unchanged: `torque = f(gear, ctrl, act, len,
vel)` where `(len, vel)` are the tendon length and velocity in the
actuator's transmission space. MuJoCo's gear rescales these values
inside the General actuator. The resulting scalar force is distributed
across the tree's DOFs via `Jᵀ · F`, where `J` is the tendon Jacobian
row.

Joint-mode and tendon-mode actuators coexist in the same
`tree.actuators` list. The pass-2 tau assembly in ABA scans only joint-
mode entries; the tendon-mode entries are consumed by
`accumulate_tendon_actuator_qfrc` (called once per ABA call, right
after `accumulate_tendon_passive`), which folds them into a per-DOF
buffer that ABA then adds to `tau_scalar`.

The `motor_on_fixed_tendon_drives_both_joints_proportional_to_coef`
test in `tests/tendon_battery.rs` pins the split: for a coupling
fixed tendon with `coef = [+1, −1]` and a `Motor { gear = 1, ctrl = 10 }`,
the two joints see `τ = +10` and `τ = −10` respectively — the naked
`Jᵀ · F` distribution.

### Free-root transmission

A tendon endpoint on a free-root link IS supported. Its 6 Jacobian
columns land in the root's `qdot` slots (`ω_body_x, ω_body_y, ω_body_z,
v_body_x, v_body_y, v_body_z`), and pass 3 adds them as a body-frame
spatial force to the root's 6×6 RHS. See `tree.rs`'s `pass 3` Free
branch. No test currently exercises a free-root tendon endpoint under
the demo scope (the `tendon_lift` demo uses a slide joint), but the
plumbing is in place.

## Passive spring and damper

Same-shape as MuJoCo:

```text
F_passive = −stiffness · (L − springlength) − damping · Ldot
```

`springlength = None` disables the spring entirely (`stiffness` is
irrelevant when unset). `damping = 0` disables the damper.

The passive force is distributed across the tree's DOFs via `Jᵀ ·
F_passive` inside every ABA call (recomputed at each RK4 sub-stage;
the tendon Jacobian samples the sub-stage state so the integrator sees
the current linearization).

The `spatial_tendon_spring_settles_at_hand_derived_equilibrium` test
anchors this: a slide-hanging mass with a spatial tendon and a
critically-damped spring settles at
`L_eq = springlength + m · g / stiffness`.

## Length limits (PGS solver rows)

Tendon length `L` can be clamped in a range `(lo, hi)`. Each active
violation contributes a PGS solver row on the tree, mirroring how
hinge/slide range limits work:

- Row Jacobian: the tendon's `dL/dqdot` (or its negation on the high
  side so `+f impulse` reduces the violation — the escape convention).
- Row projection: non-negative (unilateral).
- Row solref / solimp: `Tendon::limit_solref` / `limit_solimp` per
  tendon, defaulting to `SolRef::DEFAULT` / `SolImp::DEFAULT`.
- Row appearance: after joint limits and joint-coupling equalities in
  the solver's row order (deterministic; declaration order among
  tendons).

The `tendon_limit_no_creep_over_10k_steps` test anchors persistence
(a hanging mass sitting on the length limit doesn't creep over 10k
steps of gravity dynamics). The
`tendon_limit_solref_override_changes_penetration` test discriminates
a mutant that defaults the solref (comparing a stiff `timeconst =
0.005` tendon against a soft `timeconst = 0.05` one — the two must
differ by more than the discriminator threshold or the per-tendon
solref is being masked, the NEWT-9 lesson).

Solver-row generalization: the tree solver's `TreeRow.sparse_coeffs`
was widened from a fixed `[(u32, f32); 2]` to a `Vec<(u32, f32)>` so a
tendon row can carry N sparse entries. Joint-limit and joint-coupling
rows keep exactly 1 and 2 entries respectively (bit-identical to their
pre-tier-3 assembly). The tree-solver Cholesky solve and PGS sweep
iterate the vector uniformly.

## Sensors

`SensorKind::TendonPos { tree, tendon }` reads `L`; `SensorKind::
TendonVel` reads `Ldot`. Both are 1 scalar. Same layout / offset rules
as every other sensor kind (see `docs/sensors.md`).

## Remaining limits

- Ball joints in a spatial tendon's ancestor chain: `panic!` from
  `site_position_and_jacobian`. Documented (would need a 3-column
  contribution per ball joint).

## Differential parity (MuJoCo oracle)

Five scenarios are captured in `tests/references/`:

- `tendon_coupled.xml` + `.bin`: two coupled pendulums under gravity
  with a fixed tendon. Observed max qpos 5.6e-8, qvel 3.7e-7 over 2 s
  (400 steps). Tolerance 2e-7 / 1e-6 — an order of magnitude on the
  parity floor.
- `tendon_wrap.xml` + `.bin`: hanging mass with a spatial tendon over
  a fixed sphere obstacle. Observed max qpos 1.3e-7, qvel 1.4e-6 over
  2 s. Tolerance 3e-7 / 3e-6 — the wrap-arc's `atan2` / `asin` add
  modest f32-quant inflation over the fixed-tendon path.

Both bounds sit at f32-quantization scale — a mapping bug (wrong sign,
dropped chain-rule) would blow either by orders. See
`docs/differential.md` for the scorecard row format.

The NEWT-27 matched-Euler/PGS captures add three spatial-wrap rows:

| Scenario | qpos observed / bound | qvel observed / bound |
|---|---:|---:|
| `tendon_cylinder_lift` | `8.83e-8 / 2.0e-7` | `4.58e-7 / 1.0e-6` |
| `tendon_pulley_2to1` | `1.95e-7 / 4.0e-7` | `1.18e-6 / 3.0e-6` |
| `tendon_mixed_wrap` | `1.06e-7 / 3.0e-7` | `5.04e-7 / 1.0e-6` |

The symmetry-broken cylinder actuator golden is
`tests/goldens/tendon_wrap_cylinder.bin`. Regeneration is guarded to
macOS aarch64, like the existing tendon golden.

## Demo

```text
cargo run --release --example tendon_lift -- --frames 900 \
  --out /tmp/tendon_lift.ppm
```

Renders a motor-driven spatial tendon lifting a hanging box up and
around a fixed sphere obstacle. The tendon path (straight-arc-straight)
is drawn in cyan; the sphere in blue-grey wireframe; the box in yellow
wireframe; the anchor as a white crosshair. Verified visually: cable
routes from anchor over the sphere down to the box, box lifts as the
motor pulls, arc appears when engaged.

The cylinder and pulley demo uses the standard MP4 pipeline. Its default
run renders 600 video frames (10 seconds) at 3.00x simulation speed:

```text
cargo run --release --example tendon_cylinder_pulley
```
