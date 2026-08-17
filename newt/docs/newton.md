# newt Newton constraint solver

Status: Newton and PGS are the soft-constraint modes. Penalty remains the
legacy default path.

The Newton solver uses the same soft-constraint rows as PGS. It changes the
numerical method only. It assembles dense systems for free-body constraints,
tree limits and equalities, and all tree-involved contacts. Sparse
factorization is future work.

## Primal problem

Let `qacc` be the generalized acceleration for one solve. Let
`qacc_smooth` be the acceleration from external forces, gravity, bias, and
actuator forces before constraints. The smooth dynamics cost is

```text
  C_smooth(qacc) = 1/2 (qacc - qacc_smooth)^T M (qacc - qacc_smooth).
```

For row `i`, let `J_i` be its generalized Jacobian and let `a_ref_i` be the
MuJoCo reference acceleration. Define the constraint-space residual

```text
  s_i(qacc) = J_i qacc + a_ref_i.
```

A positive `s_i` means that the row escapes the violation. A negative value
means that the row still moves into the violation. A one-sided row contributes

```text
  C_i(s_i) = 1/2 w_i min(s_i, 0)^2.
```

The full convex primal cost is

```text
  C(qacc) = C_smooth(qacc) + Σ_i C_i(s_i).
```

The weight `w_i` is the inverse soft compliance after the `solimp` split. The
implementation uses the equivalent regularized dual after eliminating
`qacc`. This avoids forming a second generalized-coordinate copy of the
mass matrix. The dual variable is the constraint impulse `f` and its cost is

```text
  D(f) = 1/2 f^T (J M^-1 J^T + R) f + b^T f.
```

Here `b` is the existing PGS bias and

```text
  R_ii = (1 - d_i) / d_i · (J M^-1 J^T)_ii.
```

The primal and dual have the same optimum. The Newton module minimizes this
dense dual quadratic, then recovers `qacc` and the applied wrench from `f`.
This is why PGS and Newton use identical row assembly and agree on stable
scenes.

## Scalar zones

For `C_i(s) = 1/2 w min(s, 0)^2`, the zones are:

| zone | condition | gradient | Hessian |
| --- | --- | --- | --- |
| inactive | `s > 0` | `0` | `0` |
| boundary | `s = 0` | `0` | `0` by convention |
| active | `s < 0` | `w s` | `w` |

The gradient with respect to generalized acceleration is `J_i^T` times the
scalar gradient. The Hessian contribution is `w J_i^T J_i` in the active
zone. The boundary uses the inactive-side Hessian. This choice is fixed and
f32 deterministic.

Normal contact and joint-limit rows use the non-negative projection
`f_i >= 0`. Bilateral equality rows use the identity projection.

## Pyramidal friction zones

For a contact with normal residual `n`, tangent residuals `t1` and `t2`, and
friction coefficient `mu`, the pyramidal cone has two affine faces:

```text
  e1 = |t1| - mu n
  e2 = |t2| - mu n.
```

The tangential cone cost is

```text
  C_cone = 1/2 w max(e1, 0)^2 + 1/2 w max(e2, 0)^2.
```

Each face has one of these exact active gradients:

```text
  v1 = (-mu, sign(t1), 0)
  v2 = (-mu, 0, sign(t2)).
```

For an active face `e`, its derivatives are

```text
  gradient = w e v
  Hessian  = w v v^T.
```

The contact normal one-sided term is assembled separately. The zones are:

| zone | condition |
| --- | --- |
| interior | `e1 < 0` and `e2 < 0` |
| boundary | either face equals zero and neither face is positive |
| outside | `e1 > 0` or `e2 > 0` |

Torsional friction is a scalar cone bound. Rolling friction uses a second
pyramidal pair. All bounds use the current normal impulse, so a zero normal
impulse collapses every tangential bound to zero.

## Elliptic status

`elliptic_derivatives` contains the exact curved-cone derivatives for

```text
  1/2 w max(sqrt(t1² + t2²) - mu n, 0)².
```

The live Newton loader rejects `solver = Newton` with `cone = elliptic`.
The curved projection makes the current piecewise-quadratic line search
invalid. JSON and MJCF reject this pair with an error that names
`cone=pyramidal`. PGS elliptic cones remain supported.

## Newton step and line search

The dense Hessian is the regularized Delassus matrix. Cholesky supplies the
Newton direction:

```text
  H p = -(H f + b).
```

The initial impulse is zero. It is feasible for every supported projection.
The pyramidal projection path is piecewise affine in step length `alpha`.
The line search collects every zone boundary in `0 <= alpha <= 1`. On each
interval, the projected impulse is affine. The minimum of the quadratic on
that interval is the closed-form root

```text
  alpha* = alpha_mid - (gradient · p) / (p^T H p).
```

The best interval candidate is selected. A step is accepted only when it
reduces the f32 cost. This gives a monotone cost sequence without random
backtracking. If the improvement is at most
`1e-7 * (1 + abs(cost))`, the solve converges. The iteration cap is
`world.solver.iterations`; the default is 20. A zero iteration request is
invalid in loaded configurations.

The solve records the initial and accepted costs in `NewtonResult::costs`.
Tests assert monotonicity and exact hand-derived derivatives for inactive,
boundary, active, interior, and outside zones.

On the live penetrated stack anchor, Newton accepted 1 step before reaching
the scaled `1e-7` cost threshold. PGS uses its fixed 30-sweep cap because its
legacy path has no early-exit test. The comparison is therefore PGS: 30
sweeps versus Newton: 1 accepted step.

## Integration and force recovery

Newton solves once at the start of an integration step. The resulting
wrenches use the same zero-order hold as PGS across RK4 stages. Euler and
implicit-fast apply the wrench in their single semi-implicit solve.

For every row, the recovered impulse is converted to force by `f / dt`.
Linear rows apply equal and opposite force at the contact arm. Angular rows
apply equal and opposite torque. This is equivalent to recovering the
constraint force from

```text
  f_constraint = -R^-1 (J qacc + a_ref)
```

and keeps touch sensors on the actual applied normal force.

For the primal row sign used above, inactive rows have zero KKT multiplier.
Cone rows recover the projected impulse from the solved dual system.

## Tree contacts and the biped anchor

Tree-involved contacts use the selected soft-constraint solver in PGS and
Newton modes. The world assembles one dense row system for all tree links and
free bodies that touch those links. A link Jacobian row maps each contact
direction into the tree's `qdot` layout. The same row carries the impulse to
the opposite body or tree link, so mixed contacts conserve internal momentum.

Penalty mode still calls the legacy tree wrench callback. Its plane contacts
use the shared source-parity manifold, so affected penalty trajectories and
goldens can change.

The solved normal impulse stays aligned with the original contact index. Touch
sensors therefore report solver forces for tree contacts, including contacts
that sit inside a force-free gap. The biped anchor now exercises solver-based
ground and self contacts instead of a penalty fallback.

The runtime checks the solver configuration at every `World::step` call.
Programmatic `solver = Newton` plus `cone = elliptic` fails with the same
message as JSON and MJCF loading:

```text
solver=newton with cone=elliptic is not supported yet; use cone=pyramidal
```

## Selection

JSON:

```json
{
  "solver": {
    "mode": "newton",
    "iterations": 20,
    "cone": "pyramidal"
  }
}
```

MJCF:

```xml
<option solver="Newton" iterations="20" cone="pyramidal"/>
```

`solver = PGS` remains the existing soft-constraint mode. Omitting the
solver keeps the legacy default; source-manifold changes can affect contact
scenes under that mode.

## Verification anchors

The Newton test set includes:

- PGS/Newton agreement on a resting contact;
- deterministic duplicate Newton runs;
- monotone cost and convergence tests on dense hand-built systems;
- hand-derived scalar and cone zone derivatives;
- live Newton cost traces with monotone accepted costs and published
  PGS-versus-Newton iteration counts;
- contact-index writeback when a force-free contact precedes an active one;
- condim 4 torsional contact agreement;
- Newton stack and incline byte goldens;
- symmetry-broken PGS and Newton tree-contact byte goldens;
- JSON and MJCF solver selection and loud elliptic rejection.

Penalty keeps its legacy callback, but plane-manifold changes can still move
penalty trajectories and their goldens. The solver-mode tree goldens and
matched tree-contact differential fixtures are new because PGS and Newton now
own tree contact forces.

The measured cross-solver bounds use 120 steps and add modest headroom:

| scene | max position delta | max velocity delta | test bounds |
| --- | ---: | ---: | --- |
| stack | `1.130524441e-3` | `3.262443095e-2` | `1.5e-3`, `4.0e-2` |
| incline | `9.781371802e-3` | `2.716029808e-2` | `1.2e-2`, `3.5e-2` |
| equality linkage | `5.820766091e-9` | `7.836956684e-8` | `1.0e-6`, `1.0e-5` |
| joint-limit swing | `0` | `0` | `1.0e-5`, `1.0e-4` |

The condim 4 anchor adds torsional friction to a spinning sphere on a plane.
It checks finite state and PGS/Newton position, velocity, and spin agreement.
