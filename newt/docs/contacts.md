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
implemented convex pairs, and always miss internal-cavity
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
before giving up. "Impl" = shipping in `contact::narrow_phase`, "CCD" =
deterministic GJK plus EPA with one contact, and "Deferred" = returns an
empty buffer AND is flagged by `World::validate_supported_pairs()`.

|              | Plane   | Sphere  | Box     | Capsule | Cylinder | Ellipsoid | Mesh    | Hfield |
|--------------|---------|---------|---------|---------|----------|-----------|---------|--------|
| **Plane**    | —       | Impl    | Impl    | Impl    | Impl     | Impl      | Impl    | —      |
| **Sphere**   | Impl    | Impl    | Deferred| Impl    | Impl     | Impl      | Impl    | Impl   |
| **Box**      | Impl    | Deferred| Impl    | Deferred| Deferred | Deferred  | Deferred| Impl   |
| **Capsule**  | Impl    | Impl    | Deferred| Impl    | Deferred | Deferred  | Deferred| Impl   |
| **Cylinder** | Impl    | Impl    | Deferred| Deferred| Deferred | Deferred  | Deferred| —      |
| **Ellipsoid**| Impl    | Impl    | Deferred| Deferred| Deferred | Deferred  | Deferred| —      |
| **Mesh**     | Impl    | Impl    | Deferred| Deferred| Deferred | Deferred  | CCD     | —      |
| **Hfield**   | —       | Impl    | Impl    | Impl    | —        | —         | —       | —      |

box-mesh is deferred. Its rotated probe has a measured normal error of `1.0`
with the current EPA witness construction, so the runtime rejects it loudly.
The MuJoCo probe remains as an open finding in the fixture.

Contacts-per-pair for the implemented primitives:

| pair | primitive | contacts per pair |
|------|-----------|-------------------|
| sphere-plane | `contact::sphere_plane` | ≤ 1 |
| box-plane | `contact::box_plane` | ≤ 4 (MuJoCo corner scan) |
| capsule-plane | `contact::capsule_plane` | ≤ 2 (MuJoCo endpoint order) |
| cylinder-plane | `contact::cylinder_plane` | ≤ 4 (deepest of 10 sampled cap/rim points) |
| ellipsoid-plane | `contact::ellipsoid_plane` | ≤ 1 (plane-convex support midpoint) |
| mesh-plane | `contact::mesh_plane` | ≤ 2 (graph-neighbor extension deferred) |
| sphere-sphere | `contact::sphere_sphere` | ≤ 1 |
| sphere-capsule | `contact::sphere_capsule` | ≤ 1 |
| sphere-cylinder | `contact::sphere_cylinder` | ≤ 1 (closest point) |
| sphere-ellipsoid | `contact::sphere_ellipsoid` | ≤ 1 (analytic `mjc_Convex` semantics) |
| sphere-mesh | `contact::sphere_mesh` | ≤ 1 (analytic `mjc_Convex` semantics) |
| sphere-hfield | `contact::sphere_hfield` | ≤ 4 deepest prism candidates |
| capsule-hfield | `contact::capsule_hfield` | ≤ 4 deepest endpoint candidates |
| box-hfield | `contact::box_hfield` | ≤ 4 deepest convex-prism GJK/EPA candidates |
| capsule-capsule | `contact::capsule_capsule` | ≤ 1 |
| box-box | `contact::box_box` (full OBB SAT) | ≤ 4 |
| mesh-mesh `CCD` pair | `contact::ccd_convex_contact` (GJK + EPA) | 1 |

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

`mesh_plane` iterates vertices and keeps two rows. This is exact for the
current evidence set. MuJoCo can add a third row through its mesh graph walk;
that extension remains explicitly deferred.

### heightfields

`HeightField` stores an `nrow × ncol` row-major grid and
`size = (half_width_x, half_width_y, top_height, base_depth)`. Data values
are normalized to `[0, 1]`. The surface height is `data * top_height`; the
finite base extends to `-base_depth`.

Each cell uses the fixed `00 → 11` diagonal. Its two top triangles and the
base define open-sided triangular prisms. The shared diagonal is a crease.
Only outer field sides and the base are walls. Sphere and capsule collision use
the closest point on the full prism surface. Box collision uses native-style
convex-prism GJK plus EPA. Candidates use row, column,
diagonal, and feature order. The existing per-pair cap retains the deepest
four candidates.

#### native CCD routing and box hfields

MuJoCo 3.11.0 routes these default narrow phases. `CCD` means native GJK plus
EPA. `HFieldCCD` applies the convex path to heightfield prism candidates.

| pair family | default MuJoCo route |
|---|---|
| plane with sphere, capsule, cylinder, box | analytic primitive |
| plane with ellipsoid, mesh | `mjc_PlaneConvex` |
| hfield with sphere, capsule, ellipsoid, cylinder, box, mesh | HFieldCCD |
| sphere with sphere, capsule, cylinder, box | analytic primitive |
| sphere with ellipsoid, mesh | `mjc_Convex` |
| capsule with capsule, box | analytic primitive |
| capsule with ellipsoid, cylinder, mesh | CCD |
| ellipsoid with ellipsoid, cylinder, box, mesh | CCD |
| cylinder with cylinder, box, mesh | CCD |
| box with box | analytic primitive |
| box with mesh, mesh with mesh | CCD |

This table follows MuJoCo's `mjCOLLISIONFUNC` table in
`src/engine/engine_collision_driver.c`. The runtime provenance is executable:
`tests/references/hfield_conformance_default_box.xml` has no nativeccd
override. The disabled comparison remains in
`hfield_conformance_nativeccd_disabled.xml`.

NEWT-31 routed box-hfield prism candidates through deterministic in-crate GJK
plus EPA. NEWT-32 keeps sphere-ellipsoid and sphere-mesh on direct analytic
helpers matching MuJoCo's `mjc_Convex` route. Mesh-mesh uses the deterministic
GJK plus EPA path. Ellipsoid-plane and mesh-plane use the support midpoint and
two-point manifold rules from `mjc_PlaneConvex`. Box-mesh remains deferred
because its EPA witness normal is not aligned.

The six probes use isolated MuJoCo 3.11.0 models with one selected dynamic
pair at a time and `mj_forward`. Their executable provenance is
`tests/references/contact_route_probes.json`, regenerated by
`tools/capture_contact_route_probes.py` from the listed `source_xml` files.
Reviewed bounds are separate in `contact_route_probe_bounds.json`.
MuJoCo positions are midpoint positions. Its frame normal is shown after
conversion to newt's B-into-A convention. Newt values are from the
fixture-backed `analytic_convex_route_probes_are_fixture_backed` test.

| probe | pair | MuJoCo default row | newt row | status |
|---|---|---|---|---|
| P1 | sphere-ellipsoid | position `(0.45, 0, 0)`, normal `(+1, 0, 0)`, depth `0.1` | analytic helper, matching within `4.93e-6` normal error | aligned |
| P2 | sphere-mesh | position `(0.2, 0.2, 0.05)`, normal `(0, 0, -1)`, depth `0.1` | analytic helper, matching within `2.35e-6` normal error | aligned |
| P3 | plane-ellipsoid | position `(0, 0, -0.05)`, normal `(0, 0, -1)`, depth `0.1` | support midpoint, matching within `1.12e-8 m` | aligned |
| P4 | plane-mesh | 2 support contacts, midpoint near `z=-0.05` | 2 support contacts, matching within `4.1e-8 m` | aligned |
| P5 | box-mesh | 1 contact; maximum position / normal / depth error `2.5e-7 / 1.0 / 6.0e-8` | reviewed bounds `1e-6 / 1.01 / 1e-6` | deferred: EPA normal mismatch |
| P6 | mesh-mesh | 1 contact; maximum position / normal / depth error `1.8e-7 / 3.51e-5 / 9.7e-8` | reviewed bounds `1e-6 / 4e-5 / 1e-6` | CCD-aligned |

These are measured route-alignment results from the committed fixture. The
fixture keeps four poses per pair, plus a fifth near-touch mesh-mesh pose, and
fails if its source XML or mesh data is missing.

The expanded fixture covers no-contact, shallow, deep, and rotated off-axis
poses for every pair. Mesh-mesh also has a near-touch pose. The route test
reads the MuJoCo rows and asserts the reviewed bounds:

| probe | observed max position / normal / depth error | asserted bounds | count |
|---|---|---|---|
| P1 sphere-ellipsoid | `6.81e-7 / 4.93e-6 / 1.49e-8` | `1e-6 / 1e-5 / 1e-6` | equal |
| P2 sphere-mesh | `2.36e-7 / 2.35e-6 / 2.98e-8` | `1e-6 / 1e-5 / 1e-6` | equal |
| P3 plane-ellipsoid | `1.12e-8 / 0 / 2.24e-8` | `1e-6 / 1e-6 / 1e-6` | equal |
| P4 plane-mesh | `4.1e-8 / 0 / 5.96e-8` | `1e-6 / 1e-6 / 1e-6` | equal |
| P5 box-mesh | `2.5e-7 / 1.0 / 6.0e-8` | `1e-6 / 1.01 / 1e-6` | deferred |
| P6 mesh-mesh | `1.8e-7 / 3.51e-5 / 9.7e-8` | `1e-6 / 4e-5 / 1e-6` | equal |

No-contact poses require zero contacts in both engines. MuJoCo's optional
`multiccd` manifold expansion remains out of scope; the default is one CCD
contact per convex pair.

### tail-pair evidence boundary

NEWT-32 ships the four aligned route pairs plus mesh-mesh.
Box-mesh is deferred because the rotated probe reports normal error `1.0`.
The following ten native CCD routes remain Deferred and fail active-pair
validation: capsule-ellipsoid, capsule-cylinder, capsule-mesh,
ellipsoid-ellipsoid, ellipsoid-cylinder, ellipsoid-box, ellipsoid-mesh,
cylinder-cylinder, cylinder-box, and cylinder-mesh. Their support code has
direct axis tests, but they have no MuJoCo fixture or dynamic parity claim.
The follow-up ticket must add four-pose captures, early/full dynamic anchors,
and mutation coverage before enabling any of them.

The box-mesh and mesh-mesh probes remain in `contact_route_probes.json` as
open and shipped findings. Route bounds are reviewed in
`contact_route_probe_bounds.json`. The shipped mesh-mesh dynamic anchors use
the executable XML sources and capture every step from 0 through 100. The
tumble anchor bounds are `0.01874111 / 0.00334043 / 0` for the early window
and `0.1281665 / 0.00367010 / 0` for the full window. The rotated-drop anchor
bounds are `0.01874084 / 0.00311405 / 0` for the early window and
`0.1281642 / 0.00350464 / 0` for the full window. Each position and
orientation bound is the measured per-step replay maximum plus the reviewed
`1e-4` tolerance. Local `tools/verify_convex_fixtures.py` reruns both pinned
MuJoCo captures, checks byte identity, replays Newt over every captured step,
computes both window maxima, and rejects bounds that do not equal those
generated values. This committed-fixture verifier is part of the local
validation gate. The optional `--ci` mode remains available for a fresh
cross-platform MuJoCo comparison: it skips byte identity, compares every
committed route contact sample and dynamic sample with fresh MuJoCo values,
and uses those fresh states for the Newt replay check. The reviewed
cross-platform capture tolerance is `1e-6` for each compared position,
orientation, normal, and penetration component.
The final normalized macOS/Linux capture observed maxima of
`5.3506e-8 / 1.1241e-15 / 1.3306e-8 / 2.6612e-8` for position, orientation,
normal, and penetration. Contact counts and sample identifiers remain exact.
This tolerance is separate from the committed Newt
parity bounds. Mesh-local route pose orientations are excluded from the `--ci`
value comparison because MuJoCo's mesh compiler can choose different
principal-axis frames across platforms. Mesh contact positions and normals are
transformed into the owning body frame by subtracting the body position and
applying the inverse body rotation. The `--ci` mode compares all three
components of each transformed vector. When contact order differs, it uses
geometry and the nearest complete position, normal, and penetration tuple for
correspondence;
it never reduces a contact to norms or dot products. The Rust fixture keeps
world-frame values for route replay.
The verifier also runs a non-max sample mutation self-test in `--ci` mode. The Rust
fixture test also recomputes both maxima and requires exact float32 equality
between each stored position/orientation bound and its stored maximum plus
`1e-4`.

The plane-mesh route intentionally stays at two rows. MuJoCo can add a third
row through its mesh graph neighbor walk. Newt's graph walk is not implemented
yet, so the deviation is loud in this document and the support code comment.

The default box rows match MuJoCo's contact count. Contact construction is
still bounded, not exact. The steep-box row measures position `0.27`, normal
`0.8`, and depth `0.32` bounds. Fresh default capture tightened the
base-crossing row to position `0.05`, normal `0.42`, and depth `0.025`.
The bounds are stored in
`tests/references/hfield_conformance.json` and asserted by
`tests/hfield_conformance.rs`.

Hfield collision supports sphere, capsule, and box only. Mesh, cylinder, and
ellipsoid hfield pairs are deferred and rejected by active-pair validation.
Hfield
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
the fixed diagonal into two open-sided triangular prisms. The diagonal is a
top crease, not a vertical wall. Sphere and capsule colliders test top, base,
and outer side faces. Box colliders run the native-style convex-prism GJK/EPA
query and keep the four deepest unique contacts in deterministic order. The
closed prism includes the base and outer side walls, so it preserves side
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
| `tests/contacts_geoms_v1.rs::analytic_convex_route_probes_are_fixture_backed` | six MuJoCo route probes with parsed XML provenance and reviewed bounds; corrupted fixture data or XML fails the test. |
| `tests/contacts_geoms_v1.rs::dynamic_enabled_convex_anchors_are_fixture_backed` | two independent mesh-mesh anchors replay every fixture step from 0 through 100, compare per-step errors with generated bounds, and require exact maximum-plus-tolerance bound encoding. |
| `tests/contacts_geoms_v1.rs::enabled_convex_ccd_routes_emit_one_contact` | one-contact smoke coverage for the enabled mesh-mesh CCD route. |
| `tests/contacts_geoms_v1.rs::margin_fires_contact_before_geoms_touch` | plane margin 0.05, sphere just above touch: contact fires with shifted penetration `= margin − dist`. |
| `tests/contacts_geoms_v1.rs::gap_zeros_the_normal_force_while_penetration_is_below_it` | pen ≤ gap gives free-fall acceleration on the sphere despite an active contact record. |
| `tests/contacts_geoms_v1.rs::mixed_geom_scene_is_deterministic_across_two_runs` | build the pile scene twice, step 200 times each, byte-compare final state. Guards against non-deterministic iteration order in any of the new primitives (esp. the Newton solver termination). |
| `tests/contacts_geoms_v1.rs::is_pair_supported_covers_new_and_reject_lists` | direct check that `is_pair_supported` returns the expected analytic, CCD, and deferred verdicts on representative pairs. |
| `tests/contacts_geoms_v1.rs::world_validate_supported_pairs_rejects_deferred_ccd_pairs` | world validation rejects an unevidenced tail pair instead of allowing silent no-contact behavior. |

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
| mesh-mesh CCD support sign or local-axis mutation | `contact::tests::ccd_support_functions_preserve_shape_axes` and `enabled_convex_ccd_routes_emit_one_contact` |
| GJK outer termination loosening (`dot <= 1e-2`) | the fixture-backed mesh-mesh near-touch row fails on contact count; mutation command: edit `contact.rs` at the outer GJK guard, then run `cargo test --test contacts_geoms_v1 analytic_convex_route_probes_are_fixture_backed` |
| margin shift dropped (raw penetration used instead) | `margin_fires_contact_before_geoms_touch` — no contact fires despite margin > 0 |
| gap ignored in the force computation | `gap_zeros_the_normal_force_while_penetration_is_below_it` — sphere doesn't free-fall inside the gap zone |
| enabled mesh-mesh pair removed from the support matrix | `world_validate_supported_pairs_rejects_deferred_ccd_pairs` and the fixture-backed route rows |

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
object settles onto its own patch of ground; hfield and box-sphere/capsule
deferred pairs remain outside this demo):

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

run the local gate from the repository root:

```sh
cd newt
cargo test
cargo test --test alloc_guard --features alloc-guard
cargo clippy --all-targets -- -D warnings
cargo fmt --check
python3 tools/verify_convex_fixtures.py
grep -rnE '\.(sin|cos|tan|exp|ln|powf)\(' src/ \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'
```

the verifier must exit successfully. the grep must print nothing.
