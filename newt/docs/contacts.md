# newt contacts (tier 2 + v1 tier 2)

collision geoms, penalty contact forces, and pyramidal friction on top of
[tier 1's core](core.md). no joints, no actuators (later tiers). the demos in
this tier are: **stack** (three boxes piling onto a plane), **roll** (two
spheres colliding head-on across a plane), and **pile** (v1 tier 2 —
cylinder + ellipsoid + mesh tetra + yawed box stack showcasing the new
primitives).

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
| `Cylinder { radius, half_height }` (v1) | solid cylinder, axis along local Z (MuJoCo convention). |
| `Ellipsoid { semi_axes }` (v1) | 3 semi-axes along body-frame X/Y/Z. |
| `Mesh { mesh_id }` (v1) | reference into [`World::meshes`]. See "convex mesh trust model" below. |
| `Hfield { hfield_id }` (v3) | reference into `World::hfields`; normalized elevation grid with a finite triangular-prism base box. |

inertia helpers for uniform-density variants live in `newt::geom`: solid
sphere, box, capsule, cylinder, ellipsoid. `Body::solid_*` constructors call
them. Meshes ship with NO inertia helper — the trust model does not extend
to volume integration; mesh bodies must specify their inertia explicitly.

### convex mesh trust model

`GeomShape::Mesh { mesh_id }` refers to an entry in
`World::meshes: Vec<ConvexMesh>`, which is a `(vertices, faces)` pair. The
engine ASSUMES the polyhedron is convex — matching MuJoCo's asset contract.
`ConvexMesh::validate` (also run by the model loader) checks only the cheap
structural properties: ≥ 4 vertices, ≥ 4 triangular faces, face indices in
range, vertex coordinates finite. Convexity itself is NOT verified.

A non-convex mesh will silently produce incorrect contacts against the
implemented pairs (`plane`, `sphere-mesh`), and always miss internal-cavity
contacts. The mesh author owns this constraint.

### margin / gap (MuJoCo semantics)

Per-geom `margin` and `gap` fields (default 0.0):

- `pair_margin = max(a.margin, b.margin)` widens the CONTACT ACTIVATION
  zone: a contact is emitted whenever the raw signed distance is below
  `pair_margin` (plane colliders also include exact equality). The reported `penetration` on the contact is the shifted
  quantity `pair_margin - dist`, so it is positive even during near-miss
  detection.
- `pair_gap = max(a.gap, b.gap)` is a FORCE-FREE zone: the world zeros the
  contact's normal and friction force while `penetration <= pair_gap`. Use
  it for sensing-only contacts, or to model a small clearance between two
  geoms without applying force until the deeper overlap is reached.

Zero-vs-zero collapses to "detect and force on real overlap". Plane contact
manifolds still follow their collider-specific MuJoCo rules, so scenes that
use them can change when the manifold implementation changes.

## narrow-phase coverage

### support matrix

Rows = A shape, columns = B shape. Symmetric — the dispatcher tries a swap
before giving up. "Impl" = shipping in `contact::narrow_phase`, "Deferred" =
returns an empty buffer AND is flagged by
`World::validate_supported_pairs()`.

|              | Plane   | Sphere  | Box     | Capsule | Cylinder | Ellipsoid | Mesh    | Hfield |
|--------------|---------|---------|---------|---------|----------|-----------|---------|--------|
| **Plane**    | —       | Impl    | Impl    | Impl    | Impl     | Impl      | Impl    | —      |
| **Sphere**   | Impl    | Impl    | Deferred| Impl    | Impl     | Impl      | Impl    | Impl   |
| **Box**      | Impl    | Deferred| Impl    | Deferred| Deferred | Deferred  | Deferred| Impl   |
| **Capsule**  | Impl    | Impl    | Deferred| Impl    | Deferred | Deferred  | Deferred| Impl   |
| **Cylinder** | Impl    | Impl    | Deferred| Deferred| Deferred | Deferred  | Deferred| —      |
| **Ellipsoid**| Impl    | Impl    | Deferred| Deferred| Deferred | Deferred  | Deferred| —      |
| **Mesh**     | Impl    | Impl    | Deferred| Deferred| Deferred | Deferred  | Deferred| —      |
| **Hfield**   | —       | Impl    | Impl    | Impl    | —        | —         | —       | —      |

Contacts-per-pair for the implemented primitives:

| pair | primitive | contacts per pair |
|------|-----------|-------------------|
| sphere-plane | `contact::sphere_plane` | ≤ 1 |
| box-plane | `contact::box_plane` | ≤ 4 (MuJoCo corner scan) |
| capsule-plane | `contact::capsule_plane` | ≤ 2 (MuJoCo endpoint order) |
| cylinder-plane | `contact::cylinder_plane` | ≤ 4 (deepest of 10 sampled cap/rim points) |
| ellipsoid-plane | `contact::ellipsoid_plane` | ≤ 1 (analytical support point) |
| mesh-plane | `contact::mesh_plane` | ≤ 4 (deepest vertices) |
| sphere-sphere | `contact::sphere_sphere` | ≤ 1 |
| sphere-capsule | `contact::sphere_capsule` | ≤ 1 |
| sphere-cylinder | `contact::sphere_cylinder` | ≤ 1 (closest point) |
| sphere-ellipsoid | `contact::sphere_ellipsoid` | ≤ 1 (12-iter Newton) |
| sphere-mesh | `contact::sphere_mesh` | ≤ 1 (closest-point-on-triangle over faces) |
| sphere-hfield | `contact::sphere_hfield` | ≤ 4 deepest prism candidates |
| capsule-hfield | `contact::capsule_hfield` | ≤ 4 deepest endpoint candidates |
| box-hfield | `contact::box_hfield` | ≤ 4 deepest vertex candidates |
| capsule-capsule | `contact::capsule_capsule` | ≤ 1 |
| box-box | `contact::box_box` (full OBB SAT) | ≤ 4 |

### plane-primitive manifold parity

newt follows MuJoCo 3.11.0's primitive plane colliders. The dispatcher
normalizes the emitted contact back to the caller's geom order after the
primitive runs.

For `mjc_PlaneBox`, let `n` be the plane normal, `d` the plane-to-box-center
distance, and `l_i` the plane projection of corner `i` relative to the box
center. Corners use MuJoCo's bit order: bit 0 selects x, bit 1 selects y,
and bit 2 selects z. For each corner in order, newt skips it when
`d + l_i > margin` or `l_i > 0`. It emits the corner when both tests pass,
then stops after four contacts. It does not sort by depth. The raw signed
distance is `r_i = d + l_i`, the reported newt penetration is `margin - r_i`,
and the position is `corner - 0.5 * r_i * n`, the midpoint between the corner
and the plane. This rule explains why a tilted box can produce one, two, or
four contacts without any fitted threshold.

MuJoCo's plane-sphere helper uses the same midpoint position. Plane-capsule
uses the positive axis endpoint first, then the negative endpoint, with the
same raw-distance and midpoint rules. The focused anchors and their capture
provenance are in `tests/contacts_box_plane.rs` and
`tests/references/box_plane_mujoco_anchors.json`.

Deferred pairs are ENFORCED at engine level. Two mechanisms make silent
no-ops impossible:

- `World::step` calls `World::validate_supported_pairs()` on the first
  step after any pair-list or geom-count change and PANICS if any active
  pair falls in the deferred bucket. The panic message names the offending
  geom indices and shape kinds; the check is cached (O(1)) on subsequent
  steps and re-runs when the fingerprint changes (call
  `World::invalidate_pair_check()` after in-place mutation of a geom's
  `shape`).
- The JSON loader rejects an explicit `contact_pairs` entry with an
  unsupported shape combination at load time with a JSON-path error
  pointing at `contact_pairs`.

The scene author's job is to restrict `pair_list` to supported
combinations (as `examples/pile.rs` does) or plug in the GJK/support-based
fallback that lands with the v1 constraint solver. This is the direct
response to the tier-2 `stack.json` incident where an unsupported
box-sphere pair silently no-op'd, letting bodies fall through the ground.

### box-box: edge-edge SAT completion (NEWT-5 incident closure)

Tier 2 shipped box-box as vertex-vs-face only, with the caveat that it
misses edge-vs-edge intersections. v1 tier 2 closes that gap:

- The vertex-vs-face manifold runs FIRST, unchanged. Axis-aligned +
  moderate-yaw stacks still produce their manifold contacts. The shared
  plane manifold can still change a full stacking trajectory.
- If vertex-vs-face returns zero contacts, a 15-axis SAT test
  (6 face normals + 9 edge-edge cross products) determines whether the
  boxes actually overlap. If so, we emit one contact at the closest points
  of the pair of edges producing the minimum-overlap edge-edge axis.

This closes the yawed-stack case: two boxes rotated ≥ 45° relative to each
other, where every corner of the upper hangs over an edge of the lower,
now stack (previously the upper collapsed straight through).

### mesh-plane accuracy note

`mesh_plane` iterates VERTICES. This is exact for a convex polyhedron —
the deepest point on the mesh in any direction is always a vertex — so a
tetrahedron resting on a face emits contacts at its three "down" vertices.

### heightfields

`HeightField` stores an `nrow × ncol` row-major grid and
`size = (half_width_x, half_width_y, top_height, base_depth)`. Data values
are normalized to `[0, 1]`. The surface height is `data * top_height`; the
finite base extends to `-base_depth`.

Each cell uses the fixed `00 → 11` diagonal. Its two top triangles, copied at
the base depth, define the triangular-prism decomposition. Sphere and capsule
collision use the closest point on each top triangle. Box collision tests each
box vertex against those triangles. Candidates use row, column, diagonal, and
vertex order. The existing per-pair cap retains the deepest four candidates.

Hfield collision supports sphere, capsule, and box only. Mesh, cylinder, and
ellipsoid pairs are deferred and rejected by active-pair validation. Hfield
contacts enter the same in-step solver pass and penalty-stage callback as
other contacts.

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

heightfields use the MuJoCo finite-prism model. Each grid cell is split along
the fixed diagonal into two closed triangular prisms. Sphere and capsule
colliders test top, base, and side faces. Capsule tests use the full center
segment, including ridge contacts. Box colliders test the prism faces and keep
the four deepest unique contacts in deterministic order. This preserves side
contacts outside the footprint and base contacts below the terrain.

each contact contributes an equal-and-opposite wrench to its two owning
bodies (Newton's third law is baked in — the momentum anchor verifies it).
the force at the contact point becomes a linear force at the COM plus a
torque `r_arm × F`. static geoms absorb the reaction silently.

### integration

penalty contact forces are recomputed at each RK4 sub-stage from the
interpolated body or tree state. this is the live forced-ODE path and mirrors
MuJoCo's RK4 collision reevaluation. PGS and Newton instead assemble contacts
once at Euler step start. their RK4 constraint forces use zero-order hold;
that is a separate integration residual. with no geoms at all, the RK4 loop
is bit-identical to tier 1 — the tier-1 tumbling golden still passes.

## anchor tests

| file | pin |
|------|-----|
| `tests/contacts_rest.rs` | sphere on plane converges to `δ_eq = m g / k`; second-half stddev of z stays below `5e-5` (no bounce growth). |
| `tests/contacts_box_plane.rs` | fixture-driven box-plane manifolds match MuJoCo 3.11.0 contact count, scan order, midpoint positions, and depths; deep-over-four and exact-margin anchors are covered; a six-candidate newt pose proves the four-contact buffer cap; repeated runs are byte-identical. |
| `tests/contacts_friction.rs::box_below_friction_angle_stays_put` | box at θ = 22° on a μ = 0.5 tilted-gravity incline (below `atan(0.5) ≈ 26.6°`) drifts less than 0.15 m in 4 s. |
| `tests/contacts_friction.rs::box_above_friction_angle_slides_downslope` | box at θ = 32° on a μ = 0.5 incline (above threshold) slides more than 1 m in 4 s (direction verified). |
| `tests/contacts_friction.rs::friction_coefficient_zero_removes_static_hold` | μ = 0 sanity: box slides freely at any nonzero angle. |
| `tests/contacts_rolling.rs::sphere_on_plane_transitions_toward_rolling` | slip `|v_x − r ω_y|` shrinks from ~5 m/s to below 0.1 m/s after 500 steps; sphere spins about +Y and keeps moving forward. |
| `tests/contacts_energy.rs::bouncing_sphere_peaks_are_monotonically_decreasing` | at least 3 detectable peaks, each strictly less than the previous; first peak strictly below the drop height (energy really was lost). |
| `tests/contacts_momentum.rs::head_on_collision_conserves_linear_momentum` | pairwise sphere collision: max `|Σp − Σp₀| < 1e-3` over 400 steps; both spheres exchanged velocity. |
| `tests/contacts_golden.rs::contacts_golden_trajectory_is_byte_identical` | serialize `(q, qdot)` for the 3-BOX symmetry-broken stacking scene (middle box offset +0.02 m in X, top box spinning at 0.3 rad/s about Y) at steps 0/100/1000; byte-compared against `tests/goldens/stacking_3_boxes.bin`. box-box + box-plane contacts + non-trivial `r × F` are all exercised — a zero-lever-arm mutant flips this golden at snapshot 2. |
| `tests/contacts_golden.rs::stacked_boxes_keep_contact_torques_bounded_after_manifold_change` | after 2000 steps (10 s), the source manifold's changed penalty trajectory keeps every box above the plane, keeps the bottom box bounded and upright, and decays the top box's initial ω_y below 1 rad/s. The regenerated golden records this real plane-manifold behavior change. |
| `tests/contacts_geoms_v1.rs::cylinder_rests_on_plane_at_predicted_penetration` | cap-flat cylinder rests at `half_h − g / (4·k)` (4 rim contacts share the load); ω settles to < 0.05 rad/s. |
| `tests/contacts_geoms_v1.rs::ellipsoid_rests_on_plane_at_predicted_penetration` | ellipsoid bottom point sits at `-g / k`; catches an analytic-support-point sign flip. |
| `tests/contacts_geoms_v1.rs::tetrahedron_mesh_rests_on_a_face` | 4-vertex tetra on a face; three "down" vertices share the load, deepest vertex within tolerance of the closed-form single-contact depth. |
| `tests/contacts_geoms_v1.rs::rolling_cylinder_stays_on_its_axis_without_lateral_drift` | cylinder rolling under gravity keeps its axis fixed — lateral (perpendicular-to-rolling) drift < 2 cm over 2 s. |
| `tests/contacts_geoms_v1.rs::yawed_boxes_stack_and_do_not_collapse_through_each_other` | THE NEWT-5 incident closure: two boxes yawed 45° relative to each other, upper dropped just above lower, stack holds. Before edge-edge SAT, the upper collapsed straight through. |
| `tests/contacts_geoms_v1.rs::sphere_touching_cylinder_side_gives_correct_normal_direction` | sphere adjacent to cylinder side yields normal along +X (from cylinder into sphere) and penetration matching hand calculation. |
| `tests/contacts_geoms_v1.rs::sphere_touching_ellipsoid_gives_penetration_matching_axial_case` | sphere on the +X support-axis of an anisotropic ellipsoid; catches the Newton-solver convergence and normal orientation. |
| `tests/contacts_geoms_v1.rs::sphere_touching_mesh_face_gives_correct_penetration` | sphere below a mesh face; catches closest-point-on-triangle bugs. |
| `tests/contacts_geoms_v1.rs::margin_fires_contact_before_geoms_touch` | plane margin 0.05, sphere just above touch: contact fires with shifted penetration `= margin − dist`. |
| `tests/contacts_geoms_v1.rs::gap_zeros_the_normal_force_while_penetration_is_below_it` | pen ≤ gap gives free-fall acceleration on the sphere despite an active contact record. |
| `tests/contacts_geoms_v1.rs::mixed_geom_scene_is_deterministic_across_two_runs` | build the pile scene twice, step 200 times each, byte-compare final state. Guards against non-deterministic iteration order in any of the new primitives (esp. the Newton solver termination). |
| `tests/contacts_geoms_v1.rs::is_pair_supported_covers_new_and_reject_lists` | direct check that `is_pair_supported` returns the expected implemented/deferred verdicts on representative pairs. |
| `tests/contacts_geoms_v1.rs::world_validate_supported_pairs_flags_deferred_cylinder_cylinder` | world validator surfaces a cylinder-cylinder pair as unsupported so a caller can't accidentally rely on it. |

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
| wrong contact-point lever arm in wrench application (e.g. `r_a`/`r_b` zeroed) | `stacked_boxes_keep_contact_torques_bounded_after_manifold_change` — the bottom-box bound or top-box spin-down check fails; the regenerated golden also flips at snapshot 2. |
| box-box nearest-face picks the wrong wall (naive "closest face" instead of pose-delta-aligned) | 3-box demo would collapse to zero-height (upper boxes get pushed DOWN into the lower one); the box golden captures the correct settled height |
| box-box edge-edge SAT fallback disabled or wrong-signed | `yawed_boxes_stack_and_do_not_collapse_through_each_other` — upper collapses through lower (NEWT-5 regression) |
| ellipsoid analytical support point sign-flipped | `ellipsoid_rests_on_plane_at_predicted_penetration` — ellipsoid pushed up not settled |
| cylinder-plane rim sampling too coarse (skip one direction) | `cylinder_rests_on_plane_at_predicted_penetration` — expected 4-contact resting depth becomes 3-contact (33% deeper) |
| sphere-mesh iterates vertices only instead of triangle closest points | `sphere_touching_mesh_face_gives_correct_penetration` — sphere below face center reports 0 penetration |
| sphere-ellipsoid Newton solver iteration count varies (non-fixed termination) | `mixed_geom_scene_is_deterministic_across_two_runs` — final state byte-diff between two runs of the same scene |
| margin shift dropped (raw penetration used instead) | `margin_fires_contact_before_geoms_touch` — no contact fires despite margin > 0 |
| gap ignored in the force computation | `gap_zeros_the_normal_force_while_penetration_is_below_it` — sphere doesn't free-fall inside the gap zone |
| unsupported pair silently returns contacts | `world_validate_supported_pairs_flags_deferred_cylinder_cylinder` — expects the pair in the unsupported list; a bogus `is_pair_supported => true` mutant leaves the list empty |

## running the demos

three boxes dropping through the shared plane and box-box contact paths:

```sh
cargo run --release --example stack -- --frames 1800 --out /tmp/stack.ppm --size 640x360
```

two spheres rolling and colliding head-on:

```sh
cargo run --release --example roll -- --frames 400 --out /tmp/roll.ppm --size 640x360
```

v1-tier-2 pile — cylinder + ellipsoid + mesh tetra + yawed box stack (each
object settles onto its own patch of ground; cross-object pairs among
new-geom pairs are deferred, so the demo restricts its pair list to the
supported combinations):

```sh
cargo run --release --example pile -- --frames 800 --out /tmp/pile.ppm --size 800x480
```

convert to png on macOS:

```sh
sips -s format png /tmp/stack.ppm --out /tmp/stack.png
sips -s format png /tmp/roll.ppm --out /tmp/roll.png
```

the stack demo draws box wireframes (12 edges each); the roll demo draws
three great circles per sphere so the spin is visible.

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
