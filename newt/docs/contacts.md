# newt contacts (tier 2)

collision geoms, penalty contact forces, and pyramidal friction on top of
[tier 1's core](core.md). no joints, no actuators (later tiers). the demos in
this tier are two: **stack** (three spheres piling onto a plane) and **roll**
(two spheres colliding head-on across a plane).

design of record: [superpowers/specs/2026-08-13-newt-physics-design.md](superpowers/specs/2026-08-13-newt-physics-design.md).

## geoms

a `Geom` is a collision shape either bolted to a body (`body: Some(index)`) or
placed statically in the world (`body: None`, planes only in tier 2). every
geom carries a local pose (offset + orientation) so many geoms can decorate
one body, plus a friction coefficient and a [`SolRef`](#solref) contact
stiffness parameter.

| shape | notes |
|-------|-------|
| `Plane` | infinite half-space, static only. outward normal is local +Z; the plane passes through the geom's origin. |
| `Sphere { radius }` | origin at center. |
| `Box { half_extents }` | origin at center, axes along local xyz. |
| `Capsule { radius, half_height }` | axis along local Z. `half_height` is the cylindrical half-length; tip-to-tip is `2*(half_height + radius)`. |

inertia helpers for uniform-density variants live in `newt::geom` (solid
sphere, solid box, solid capsule) and are what `Body::solid_sphere`,
`Body::solid_box`, and `Body::solid_capsule` call. callers with a
custom mass distribution build the inertia tensor themselves and pass it to
`Body::new`.

## narrow-phase coverage (tier 2)

| pair | primitive | contacts per pair |
|------|-----------|-------------------|
| sphere-plane | `contact::sphere_plane` | ≤ 1 |
| box-plane | `contact::box_plane` | ≤ 4 (deepest corners) |
| capsule-plane | `contact::capsule_plane` | ≤ 2 (axis endpoints) |
| sphere-sphere | `contact::sphere_sphere` | ≤ 1 |
| sphere-capsule | `contact::sphere_capsule` | ≤ 1 |
| capsule-capsule | `contact::capsule_capsule` | ≤ 1 |

box-box, box-sphere, and box-capsule are **deferred to a later tier**. the
stacking demo therefore uses spheres, which give equivalent stress on the
contact math (normal spring, tangent pyramid, third-law reaction sum) without
needing SAT-style oriented-box logic. that logic returns in v0-tier5-ish along
with mesh geoms.

pair filtering: when `World::pair_list` is `None`, pairs are enumerated as
`(i, j)` with `i < j` over the geom vector, skipping same-body pairs and
static-vs-static pairs. the order is deterministic and index-stable. a caller
who wants explicit control writes `world.pair_list = Some(vec![...])`.

## contact model

### normal force

the normal spring is parameterized MuJoCo-style: each geom carries a
`SolRef { timeconst, dampratio }` and a pair's effective spring/damper
constants are

- `k = m_eff / timeconst²`
- `c = 2 * dampratio * m_eff / timeconst`

where `m_eff` is the reduced mass of the pair (`m_a` for a body-vs-static
contact, `m_a m_b / (m_a + m_b)` for two dynamic bodies). the two geoms'
solrefs are combined per parameter with a MIN rule — the stiffer time
constant wins AND the less-damped ratio wins. rationale: users typically
over-set the property they care about on one geom and leave the other on
defaults; a "smaller wins" rule lets the non-default value show through.

with contact-point relative normal velocity `v_n = (v_a − v_b) · n`, the
normal force magnitude is

```text
f_n = max(0, k * pen − c * v_n)
```

the clamp keeps the contact from PULLING when the bodies are separating
faster than the spring wants to relax. equilibrium under a constant load
`F_ext` sits at `δ_eq = F_ext / k`: for a 1 kg sphere under 1 g and the
default `SolRef` (timeconst 0.02 s, ζ = 1), this is `9.81 / 2500 ≈ 4 mm`.
the resting-contact anchor test locks that number down.

### friction (pyramidal Coulomb, viscous-with-clamp)

the pair friction coefficient is `μ = min(μ_a, μ_b)`. matches Bullet/ODE
defaults and is monotone in either coefficient.

for each contact we build a deterministic orthonormal tangent basis
`(t1, t2)` perpendicular to `n`. picks the world reference axis whose
alignment with `n` is weakest (X if `|n.z| > 0.9`, else Z), takes the cross
product, and normalizes — no atan / no branches based on floating-point
comparisons that could flip between platforms.

per-tangent friction force is the pyramidal Coulomb form

```text
f_ti = clamp(-c_tangent * v_ti, -μ|f_n|, +μ|f_n|)     for i in {1, 2}
```

with `c_tangent = c_normal` (the spring's own damping coefficient). the
static/kinetic distinction is *effective*: at rest the tangential velocity
sits near zero, so the viscous force is tiny; under sliding it saturates at
the Coulomb cap. designers who need a bright-line stiction can override
`c_tangent` via a stiffer solref on the "sticky" geom.

### force application

each contact contributes an equal-and-opposite wrench to its two owning
bodies (Newton's third law is baked in — the momentum anchor verifies it).
the force at the contact point becomes a linear force at the COM plus a
torque `r_arm × F`. static geoms absorb the reaction silently.

### integration

contact forces are recomputed at each RK4 sub-stage from the interpolated
body state — standard RK4-on-forced-ODE treatment. very stiff underdamped
contacts show some parasitic RK4 damping (the intermediate stages sample
deeper penetrations than the true continuous solution reaches); the bouncing
anchor picks a `dampratio` and `dt` combination that keeps the effective
restitution comfortably above zero. with no geoms at all, the RK4 loop is
bit-identical to tier 1 — the tier-1 tumbling golden still passes.

## anchor tests

| file | pin |
|------|-----|
| `tests/contacts_rest.rs` | sphere on plane converges to `δ_eq = m g / k`; second-half stddev of z stays below `5e-5` (no bounce growth). |
| `tests/contacts_friction.rs::box_below_friction_angle_stays_put` | box at θ = 10° on a μ = 0.5 tilted-gravity incline drifts less than 0.15 m in 4 s. |
| `tests/contacts_friction.rs::box_above_friction_angle_slides_downslope` | box at θ = 45° on a μ = 0.5 incline slides more than 1 m in 4 s (direction verified). |
| `tests/contacts_friction.rs::friction_coefficient_zero_removes_static_hold` | μ = 0 sanity: box slides freely at any nonzero angle. |
| `tests/contacts_rolling.rs::sphere_on_plane_transitions_toward_rolling` | slip `|v_x − r ω_y|` shrinks from ~5 m/s to below 0.1 m/s after 500 steps; sphere spins about +Y and keeps moving forward. |
| `tests/contacts_energy.rs::bouncing_sphere_peaks_are_monotonically_decreasing` | at least 3 detectable peaks, each strictly less than the previous; first peak strictly below the drop height (energy really was lost). |
| `tests/contacts_momentum.rs::head_on_collision_conserves_linear_momentum` | pairwise sphere collision: max `|Σp − Σp₀| < 1e-3` over 400 steps; both spheres exchanged velocity. |
| `tests/contacts_golden.rs` | serialize `(q, qdot)` for the 3-sphere stacking scene at steps 0/100/1000; byte-compared against `tests/goldens/stacking_3_spheres.bin`. |

the golden was generated on macOS aarch64 (same convention as tier 1).
regenerate ONLY on the reference host:

```sh
cargo test --test contacts_golden regenerate_contacts_golden -- --ignored --nocapture
```

the ignored test panics on any other host so an accidental `--ignored` run
cannot silently swap the reference.

## mutation coverage summary

| mutation | test that catches it |
|----------|----------------------|
| flipped contact normal | `contacts_rest.rs` — sphere would be pushed through the plane instead of settling |
| friction applied along the normal | `contacts_friction.rs::box_above_friction_angle_slides_downslope` (direction check fails) AND `contacts_rolling.rs` (ω_y stays near 0) |
| missing damping term | `contacts_energy.rs` — first peak stops decaying, or grows |
| swapped Newton's third law (only one body gets the reaction) | `contacts_momentum.rs` — pairwise momentum drifts by tens of percent |
| combine_solref picks stiffer damping instead of MIN | `contacts_energy.rs` — sphere overdamps and stops bouncing (this is exactly the bug caught during development) |

## running the demos

three spheres dropping and settling into a stack:

```sh
cargo run --release --example stack -- --frames 800 --out /tmp/stack.ppm --size 640x360
```

two spheres rolling and colliding head-on:

```sh
cargo run --release --example roll -- --frames 400 --out /tmp/roll.ppm --size 640x360
```

convert to png on macOS:

```sh
sips -s format png /tmp/stack.ppm --out /tmp/stack.png
sips -s format png /tmp/roll.ppm --out /tmp/roll.png
```

both demos render three great circles per sphere so the spin (roll) or
lack-of-spin (stack, after settling) is visible.

## verification

fmt, clippy, tests, libm-free grep — same triple as tier 1:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
grep -rnE '\.(sin|cos|tan|exp|ln|powf)\(' newt/src/ \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'
```

the grep must print nothing. `.github/workflows/newt.yml` runs the same
four checks on every push and PR.
