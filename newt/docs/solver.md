# newt/docs/solver.md — MuJoCo Soft-Constraint Contact Model + PGS

Status: v3 tier 3, shipped through NEWT-22 (tree contacts, equality
constraints, condim 4/6).

## Purpose

Replace the v0 penalty spring-damper contact model with MuJoCo's
soft-constraint formulation, solved with Projected Gauss-Seidel over
the convex dual problem. Ship both condim 1 (frictionless) and condim 3
(sliding, normal + 2 tangents), both pyramidal and elliptic friction
cones, and constraint-based joint limits for hinge/slide range.

Penalty remains the default; every pre-v1-tier-4 golden stays
byte-identical.

## Model reference

Every constraint (contact normal, contact tangent, joint limit) reduces
to a scalar row on the tree's or body's generalized velocity. Tree-involved
contact rows share one world-level system across all participating pools.

- Jacobian row `J_i` (1 × nv). For a contact normal with two bodies A
  and B and contact point `p`, world-frame arms `r_a = p - com_a`,
  `r_b = p - com_b`:
  ```
  J_i · qdot = (v_a + ω_a × r_a) · n − (v_b + ω_b × r_b) · n
             = v_rel_at_contact · n
  ```
  For tangent rows the direction is `t1` or `t2` — deterministic basis
  from `world::tangent_basis(n)`. For a joint limit on DOF `k` the row
  is `sign · e_k` where `sign = +1` for a low-side violation
  (q < lo, escape = +q) and `-1` for a high-side violation.
- Violation `r_i > 0`. Contact: `r = penetration - gap` (only active
  when > 0). Joint limit: `r = max(0, lo - q, q - hi)`.
- `(solref, solimp)` — per-constraint parameterization. Contact rows
  combine per pair via `combine_solref` / `combine_solimp`
  (per-component minimum — the "stronger setting wins" rule; see the
  `Geom::solref` docs for rationale). Joint-limit rows read the
  optional `JointLimit::solref` / `JointLimit::solimp` per limit;
  omitted overrides fall back to `SolRef::DEFAULT` /
  `SolImp::DEFAULT`.

### SolImp — the 5-parameter impedance sigmoid

`d(r) ∈ [dmin, dmax] ⊂ (0, 1)` — the fraction of the constraint force
that goes through the projection (hard clamp / cone), with `(1 − d)`
flowing through the diagonal regularization `R`. Stiff contact (d → 1)
enforces the constraint hard; soft (d → dmin) looks like a spring.

Sigmoid `y(x)` on `x = |r|/width`, piecewise power interpolation:

```
y(x) = 0                                              for x ≤ 0
y(x) = midpoint · (x / midpoint)^power                for 0 < x < midpoint
y(x) = 1 − (1 − midpoint) · ((1 − x)/(1 − midpoint))^power  for midpoint ≤ x < 1
y(x) = 1                                              for x ≥ 1
```

Both branches evaluate to `midpoint` at `x = midpoint` with first
derivative `power` — so the sigmoid is C¹ everywhere in (0, 1).

Then `d(r) = dmin + (dmax − dmin) · y(x)`.

`power` is a documented integer ≥ 1 (iterated multiplication — no
libm). Default `power = 2` matches MuJoCo. Default parameters:

```
dmin = 0.9   dmax = 0.95   width = 0.001 m   midpoint = 0.5   power = 2
```

Combine rule (`combine_solimp`): per-component minimum for
`(dmin, dmax, width)`; `midpoint` and `power` fall back to the FIRST
geom's values (arbitrary but stable and documented). Rationale: the
"softer" setting wins on the impedance-fraction axes (a user dialing
in a low-impedance ball against a default plane gets softness), and
smaller `width` engages sooner (the sharper geom controls when
saturation hits).

### Reference acceleration

For each row, use MuJoCo's signed position `s = efc_pos - efc_margin`.
Penetration has `s < 0`. Let `v = J · qvel`.

For positive `solref`, clamp `timeconst` to at least `2 · dt`, then use:

```
b = 2 / (dmax · timeconst)
k = 1 / (dmax² · timeconst² · dampratio²)
a_ref(s, v) = -b · v - k · d(s) · s
```

For negative `solref = (-stiffness, -damping)`, use the direct values:

```
k = stiffness / dmax²
b = damping / dmax
```

The stored values are negative, so the implementation negates them. The
same helper serves contacts, equalities, limits, and both solver modes.
MuJoCo clamps `dmin`, `dmax`, and midpoint to `[1e-4, 0.9999]`, clamps
width to non-negative, and clamps power to at least one. Newt applies the
same clamps.

The reference uses the row velocity at the solve instant. It does not add
the current `J · qvel` again to the impulse residual.

### Regularized dual

Stack the constraint rows into `J`. The primal problem is:

```
find qacc such that
    J · qacc - a_ref = 0                         (equality target)
    subject to per-constraint projections:
        normal / limit rows:  f ≥ 0
        tangent rows:         cone(f_t, f_n, μ) satisfied
```

The convex dual: solve for impulses `f`, one per row, such that

```
A · f + b = 0            (with acceptance-set projections)
A = J M⁻¹ Jᵀ + R          — regularized system matrix
b = Δ(J · qvel)_free - a_ref · dt
                                  — velocity-form acceleration residual
```

where:
- `M` is the joint-space mass matrix (for tree-DOF constraints) or the
  block-diagonal rigid-body inertia (for free-body constraints).
- `Δ(J · qvel)_free` is the non-constraint velocity change for one step.
  Free-body rows include the gravity kick. RK4 reuses these rows with the
  solver's zero-order-hold force semantics.
- `R` is derived from MuJoCo's `diagApprox`, impedance, and pyramid rule.
  For a pyramidal condim-3 contact, each facet gets
  `Rpy = 2 · μ² · Rnormal`.
- `b` and `k` are the exact `mj_makeImpedance` coefficients. `a_ref` is
  the exact `mj_referenceConstraint` expression.

For a free body with mass `m` and world-frame inertia
`I_w = R I_body Rᵀ`, contact at arm `r_arm`:

```
A_ii_body = (r_arm × dir)ᵀ · I_w⁻¹ · (r_arm × dir) + 1/m
```

per body, summed over the two bodies of the pair. Static geoms
contribute zero (infinite mass).

### PGS solve

Fixed iteration count, no early exit — determinism outranks
convergence sensitivity in this engine (the same input must produce
the same trajectory across platforms and runs). Default 20 iterations
in the shipped `SolverConfig::DEFAULT`; more for scenes with heavy
mass ratios (the stack demo uses 30).

Sweep order per iteration:
1. Enumerate contacts in narrow-phase order (already deterministic).
2. Within a contact: normal row first, then t1, then t2. This lets the
   tangent projection see the just-updated normal impulse for its cone
   cap.

Per-row update:
```
residual_i = Σ_j (J_i · d_j) · f_j + R_ii · f_i + bias_i     # d_j = M⁻¹ Jⱼᵀ
delta = -residual_i / diag_i
f_i_new = project_kind(f_i + delta)
```

Free bodies: we maintain a per-body velocity accumulator
`body_delta = M⁻¹ Jᵀ · f_current`; a delta on row i updates it by
`M⁻¹ · Jᵢᵀ · delta_f`. The `J_i · body_delta` inner product then
becomes cheap in the residual.

Tree joint limits build `A` densely (`n_limits × n_limits`, a submatrix of
`M⁻¹`) by precomputing `M⁻¹ · e_j` for each limit via Cholesky solve. Tree
contact rows use the same response construction, but assemble one system
across every tree and free body in the contact set. This handles tree-vs-
static, tree-vs-body, and tree-vs-tree contacts without treating the other
participant as a wall.

### Friction cone projections

Both are hand-derived:

- **Pyramidal** (`project_pyramidal`): per-axis symmetric clamp
  `|f_t| ≤ μ · f_n`. Two independent clamps for `f_t1` and `f_t2`.
  Projects onto a square inscribed in the true circular Coulomb cone
  (conservative on axis-aligned slip, slightly under-friction on 45°
  diagonal slip). One clamp per tangent per iteration.
- **Elliptic** (`project_elliptic`): true circular disc of radius
  `μ · f_n`. Analytic projection:
  ```
  if ‖(f_t1, f_t2)‖ ≤ μ · f_n: unchanged
  else:                        radially rescale to boundary
  ```
  If `f_n ≤ 0` the cone collapses to a point at the origin and the
  projection returns `(0, 0)`. Under elliptic we do a JOINT
  Gauss-Seidel step on both tangents then a single projection to keep
  the update coherent with the cone geometry.

Both cone kinds live behind `world.solver.cone`; the shipped tests
exercise both under the friction incline anchor.

## Integration path — once-per-step under RK4

MuJoCo uses one Euler step per constraint solve. Newt integrates with
RK4 (design spec — kept for accuracy on the free-body path). We keep
RK4 but solve the constraint ONCE per step and hold the resulting
per-body wrenches / per-DOF forces constant across all four RK4
sub-stages (zero-order hold, ZOH).

**Consequences (documented honestly):**

1. Constant force + RK4 exactly integrates polynomials up to degree 3,
   so a constant-wrench body under gravity gets `v = v₀ + a·dt` and
   `p = p₀ + v₀·dt + a·dt²/2` — same as Euler for a constant-force
   step. No extra error introduced by RK4 on the constrained side.
2. The solver targets end-of-step velocity, not per-substage
   constraint satisfaction. If a free body drifts DURING a step and
   accumulates penetration between substages, the next step's solve
   picks it up. Over many steps the impedance sigmoid caps drift by
   construction.
3. Coupled multi-body stacks with the STIFFEST solref
   (`timeconst = 0.02s`, default) plus `dt = 5 ms` sit at the edge of
   the RK4-ZOH stability envelope. Empirically, 3-box stacks with
   default solref diverge over ~500 steps: the solver, given only the
   start-of-step state, cannot see the intra-step wobble that RK4
   introduces, and over-corrects on the following step.

   **Mitigation**: use a slightly softer solref for coupled stacks —
   `SolRef::new(0.05, 1.5)` (period ≈ 0.3s, sample ≈ 60/period) is
   well inside the stability envelope. The shipped stack test and
   `examples/solver_stack.rs` use this.

   **Alternative** (future work — v3): solve per RK4 sub-stage. That
   would remove this constraint but multiplies solver cost by 4 and
   breaks bit-identity with MuJoCo's Euler pathway when we run the
   differential harness (NEWT-13). Revisiting the integrator is a
   design-spec v3 item.

## Scope through NEWT-10 (v1 tier 5)

**Included:**
- Contact condim 1, 3, 4 (adds torsion about the normal), and 6 (adds
  two rolling rows about the tangents). Both cone kinds
  (Pyramidal / Elliptic) for every dimension group.
- Free-body-vs-static and free-body-vs-free-body contacts.
- Hinge and slide joint range limits as constraints (per-tree solver).
- Equality constraints — Connect (3 rows), Weld (6 rows: 3 linear +
  3 angular), JointCoupling (1 tree-space row, up to quadratic
  polynomial), Distance (1 row). Bilateral projection; per-constraint
  `(solref, solimp)`.
- Per-geom `condim`, `solimp`, `torsional_friction`,
  `rolling_friction` on `Geom`; world-level `SolverConfig` with `mode`,
  `iterations`, `cone`.
- JSON model: root `"solver"` and `"equality"` blocks; per-geom
  `"solimp"`, `"condim"`, `"torsional_friction"`,
  `"rolling_friction"` fields; strict validation.

**Deferred (documented):**
- Ball-joint limits (cone / swing-twist).
- Equality constraints attached to tree links (currently free-body /
  world only for the linear-and-angular equalities; joint coupling
  handles tree DOFs).
- The remaining differential scorecard rows outside tree contacts are
  documented in `docs/differential.md`.

Penalty mode remains a separate legacy path. It does not enter the shared
contact system, which preserves penalty-mode trajectories and goldens.

## Equality constraint rows (v1 tier 5)

Equality rows are added to the same PGS sweep as contact and limit
rows. They are BILATERAL: the row's projection is the identity
(unclamped scalar update), so the impulse can push either way to drive
the residual to zero.

For each free-body equality kind:

| Kind | Rows | Row geometry | Residual per row (before sign normalization) |
| --- | --- | --- | --- |
| `Connect` | 3 | Linear | `(p_A − p_B) · e_k` for k in x, y, z |
| `Weld` | 6 | 3 Linear, 3 Angular | 3 anchor components; 3 orientation-error components `2 · imag(q_A · q_target · q_B_conj)` |
| `Distance` | 1 | Linear | `|p_A − p_B| − d0` along the current separation unit vector `u = (p_A − p_B) / |·|` |

Angular rows contribute a `dir · I⁻¹ · dir` diagonal (no linear-mass
term) and apply pure torque `dir · f` on body A, `−dir · f` on body B.
Linear rows are identical to contact rows in shape — same
arm-cross-force torque plumbing.

**Sign normalization (escape convention).** For each row we compute the
signed raw residual `r_raw` and a natural direction `dir_raw`. We
store `r = |r_raw|` and flip `dir` so a positive impulse drives `r`
down: `dir = −sign(r_raw) · dir_raw`. Then `J · qdot = −r_dot` in the
row's convention, matching the contact-normal escape convention.
Equality rows use the same signed-position reference and acceleration-form
bias as contacts. This keeps one source-derived path for all row types and
both solver modes.

**Distance-at-zero-separation guard.** When `|p_A − p_B| <
DISTANCE_DEGENERATE_EPS` (currently 1 µm) the row is elided that step:
no direction is well-defined. The constraint re-engages as soon as the
separation grows past the guard. Documented; the accompanying test
`equality_distance` exercises the orbital case (separation always
well above the guard).

### Tree-space joint-coupling row

A `JointCoupling { link_a, link_b, polycoef: [c0, c1, c2] }` on tree
`t` becomes ONE row in that tree's PGS solve (extending the joint-limit
solver). The residual is
`r_signed = q_a − (c0 + c1·q_b + c2·q_b²)`; the row's sparse
Jacobian is `[+1 at slot_a, −(c1 + 2·c2·q_b) at slot_b]`. Escape
convention flips both coefficients when `r_signed > 0`. The row shares
the tree solver's dense-`A` machinery — precompute `M⁻¹ · e_i` via
Cholesky for each row, then run PGS with bilateral projection on the
coupling rows and non-negative projection on the limit rows.

Cross-tree coupling is out of scope this tier (would need a
world-level `A` matrix). The loader rejects coupling references that
name links on different trees.

## Contact condim 4 and 6

`condim >= 4` appends one Angular row per contact about the contact
normal. Its cone cap is `mu_torsion · f_n` (per-geom
[`torsional_friction`](../src/geom.rs), combined by `min`). This
damps drill-style spin about the normal — sliding rows can't reach
that motion because a sphere-on-plane contact has zero tangential
velocity at the axis of rotation.

`condim == 6` also appends two Angular rows about the two tangent
axes. Their caps are each `mu_roll · f_n` (per-geom
[`rolling_friction`](../src/geom.rs), same `min` combine rule). These
damp rolling — under rolling-without-slipping, the contact point's
tangential velocity is zero, so the sliding rows again do no work; the
rolling rows are the only lever.

**Cone dispatch for the extended blocks.**

- **Pyramidal.** Every row (sliding tangents, torsion, rolling axes)
  is independently clamped `|f_k| ≤ mu_k · f_n`. `mu_k` is `mu_slide`
  for the two tangents, `mu_torsion` for the torsion row, `mu_roll`
  for each rolling row.
- **Elliptic.** Independent 2D-disc / scalar / 2D-disc projections
  per dimension group: sliding tangents onto the disc of radius
  `mu_slide · f_n`; torsion clamped scalar as above; rolling tangents
  onto the disc of radius `mu_roll · f_n`. MuJoCo's cross-group
  coupling (single N-D cone) is a documented deviation — no shipped
  test disambiguates the two formulations, and per-group projections
  keep the sweep local to each row pair.

Row sub-order per contact (deterministic, documented):

    normal → t1 → t2 → torsion → roll1 → roll2

Rows after `normal` only appear when `condim` reaches the
corresponding threshold. Every cone cap uses the just-updated normal
impulse from the current PGS iteration.

## API / model reference

### World-level (Rust)

```rust
use newt::solver::{SolverConfig, SolverMode, ConeKind};

world.solver = SolverConfig {
    mode: SolverMode::Pgs,          // default: Penalty
    iterations: 30,                 // default: 20
    cone: ConeKind::Pyramidal,      // default: Pyramidal
};
```

### Per-geom (Rust)

```rust
let mut g = Geom::r#box(body_idx, half, ...);
g.condim = 3;                       // 1, 3, 4, or 6 (default 3)
g.solimp = SolImp::new(0.9, 0.99, 0.001, 0.5, 2);
g.solref = SolRef::new(0.05, 1.5);  // (timeconst, dampratio)
```

### JSON

```json
{
  "solver": {
    "mode": "pgs",
    "iterations": 30,
    "cone": "pyramidal"
  },
  "geoms": [
    {
      "name": "block",
      "shape": {"kind": "box", "half_extents": [0.3, 0.3, 0.3]},
      "attach": {"kind": "body", "body": "cube"},
      "friction": 0.8,
      "condim": 3,
      "solref": {"timeconst": 0.05, "dampratio": 1.5},
      "solimp": {"dmin": 0.9, "dmax": 0.95, "width": 0.001, "midpoint": 0.5, "power": 2}
    }
  ]
}
```

## Demo

```
cargo run --release --example solver_stack -- --frames 800 --out /tmp/solver_stack.ppm
```

Renders a 6-box tower with a heavy (5 kg) box dropped from z = 5 m
onto the tower. The bottom three boxes hold together under the impact;
the top three are knocked off and settle on the ground alongside the
dropped box. Wireframe PPM via chimy2, camera fixed. Solver:
`SolverMode::Pgs`, `iterations = 30`, `ConeKind::Pyramidal`, solref
`(0.05s, 1.5)` on every geom.

## Demo — v1 tier 5 (equality + coupling)

```
cargo run --release --example linkage -- --frames 800 --out /tmp/linkage.ppm
```

Renders a four-bar-style linkage: two grounded hinges (crank and
follower) linked by a free-body coupler bar constrained via two
`connect` equalities, plus a `joint` coupling that slaves
`follower = −1 · crank`. A servo drives the crank through a slow
sinusoidal sweep; the follower mirrors it, and the coupler bar tracks
both anchors. Uses the free-body PGS solver AND the tree PGS solver
together in one scene.
