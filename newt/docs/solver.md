# newt/docs/solver.md — MuJoCo Soft-Constraint Contact Model + PGS

Status: v1 tier 4, shipped in NEWT-9.

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
to a scalar row on the tree's / body pool's generalized velocity:

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
- `(solref, solimp)` — per-constraint parameterization, combined
  per-pair for contacts via `combine_solref` / `combine_solimp`
  (per-component minimum — the "stronger setting wins" rule; see the
  `Geom::solref` docs for rationale).

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

For each constraint at violation `r > 0` and constraint velocity
`x_dot` (constraint escape velocity; positive = separating), define
`r_dot = -x_dot` (positive = worsening) and take:

```
a_ref(r, r_dot) = -(2 · dampratio / timeconst) · r_dot − (1 / timeconst²) · r
```

This is a critically-damped second-order response with natural
frequency `1/timeconst` and damping `dampratio` (MuJoCo's positive
`solref = (timeconst, dampratio)` convention). For `dampratio = 1` and
`r_dot > 0` the response drives `r → 0` critically.

Convention wart to be aware of: `a_ref` is the target acceleration for
r (constraint violation), NOT for the constraint escape velocity.
Since `r_dot = -x_dot`, sign flips propagate:
`r_ddot = a_ref → x_ddot = -a_ref`.

We evaluate `r_dot` from the constraint velocity AT THE START of the
step, BEFORE the free-step (gravity/qfrc_applied) kick is applied.
Using the post-free-step velocity would inflate `r_dot` by the free
step's contribution and produce over-corrective impulses (empirically:
3-box stacks diverge). This matches MuJoCo's "reference is defined at
the solve instant" behavior.

### Regularized dual

Stack the constraint rows into `J`. The primal problem is:

```
find qdot_new such that
    J · qdot_new + a_ref · dt = 0                (equality target)
    subject to per-constraint projections:
        normal / limit rows:  f ≥ 0
        tangent rows:         cone(f_t, f_n, μ) satisfied
```

The convex dual: solve for impulses `f`, one per row, such that

```
A · f + b = 0            (with acceptance-set projections)
A = J M⁻¹ Jᵀ + R          — regularized system matrix
R = diag((1 − d) / d) · diag(A)   — MuJoCo regularization
b = J · qdot_free + d · a_ref · dt   — bias (residual + reference)
```

where:
- `M` is the joint-space mass matrix (for tree-DOF constraints) or the
  block-diagonal rigid-body inertia (for free-body constraints).
- `qdot_free = qdot + M⁻¹ · (τ_external − h) · dt` is the velocity that
  would arise from all non-constraint forces over one dt. For free
  bodies this ticket approximates by gravity only; higher-order terms
  are folded into the RK4 stages that follow (see "Once-per-step under
  RK4" below).
- `R` sits on the diagonal because it comes from the (1 − d) · f_hard
  "soft split" (a per-constraint linear damping in impulse space).

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

Tree joint limits: we build `A` densely (`n_limits × n_limits`, a
submatrix of `M⁻¹`) by precomputing `M⁻¹ · e_j` for each limit via
Cholesky solve. Trees are independent; each tree's limits solve
independently.

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

## Scope this ticket

**Included:**
- Contact condim 1 (frictionless) and condim 3 (sliding); both cone
  kinds.
- Free-body-vs-static and free-body-vs-free-body contacts.
- Hinge and slide joint range limits as constraints (per-tree solver).
- Per-geom `condim`, `solimp` on `Geom`; world-level `SolverConfig`
  with `mode`, `iterations`, `cone`.
- JSON model: root `"solver"` block and per-geom `"solimp"`,
  `"condim"` fields, strict validation.

**Deferred (documented):**
- Contact condim 4 / 6 (torsional / rolling) — land with the
  equality-constraints ticket.
- Ball-joint limits (cone / swing-twist) — need the ball-joint DOF
  layout that ships with equality constraints too.
- Cross-tree / body-vs-link contacts under solver mode — remain on
  the penalty pathway this ticket. Not required for any shipped test.
- MuJoCo differential parity — the biped venv comparison is NEWT-13.

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
g.condim = 3;                       // 1 or 3 (default 3); 4/6 rejected
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
