# newt dynamics (v1 tier 3)

CRB mass matrix `M(q)` and RNE inverse dynamics `τ(q, qdot, qddot)` for
kinematic trees, plus a hand-rolled dense Cholesky factor + solve.
Building blocks the v1 soft-constraint solver (next ticket) consumes:
`M`, the bias vector `h(q, qdot)`, and cheap `M x = b` solves. Builds on
[joints](joints.md) (`Tree`, ABA, spatial algebra).

design of record:
[superpowers/specs/2026-08-13-newt-physics-design.md](superpowers/specs/2026-08-13-newt-physics-design.md).

## the equation of motion

Every function in this module carries one equation on its back:

```text
M(q) · qddot + h(q, qdot) = τ_applied
```

- `M(q)` is the joint-space mass matrix. Symmetric, positive-definite for
  any physically valid tree (positive masses, positive-definite link
  inertias, nonnegative armature).
- `h(q, qdot)` is the bias vector: Coriolis + centrifugal + gravity
  torques. This is what MuJoCo calls `qfrc_bias`.
- `τ_applied` is the sum of every generalized force delivered into each
  joint DOF: actuators, damping, joint limits, `qfrc_applied`,
  `applied_wrenches`, contact impulses, everything.

`inverse_dynamics(q, qdot, qddot)` returns `M · qddot + h − S(f_ext)` —
the residual `τ_applied` the caller would need to supply given the
external wrenches passed in.

### what bias / inverse_dynamics EXCLUDE

`bias_forces` and `inverse_dynamics` compute only the LHS. Everything on
the RHS `τ_applied` — `damping`, `range` limits, `actuators`,
`qfrc_applied`, `tree.applied_wrenches` — is NOT consulted. This matches
MuJoCo's `qfrc_bias` semantics; the v1 constraint solver will assemble
the RHS from those buffers and add the constraint impulses on top.

Exception: `external_wrenches` (parameter, world-frame per link at COM)
IS consulted, because gravity and one-off world-frame wrenches are the
canonical "environment" side of Newton-Euler. Passing zeros is fine and
matches the pure-inverse-dynamics case.

### armature

`armature` on a hinge or slide adds to that DOF's diagonal in `M`:
`M[i, i] += armature`. On a ball joint, `armature · I₃` adds to the 3×3
block. Free-root has no armature slot. `inverse_dynamics` accordingly
adds `armature · qddot_slot` to each joint DOF's `τ`, keeping the
identity `τ = M·qddot + h` consistent when armature is present.

## algorithms

### RNE (Featherstone RBDA §5, Table 5.1)

Pass 1 (root → leaves): compute per-link body-frame velocity `v[i]`,
acceleration `a[i]`, and rigid-body wrench `f[i]`:

```text
v[0] = qdot[0..6] as SpatialMotion  (free root; else zero)
a[0] = qddot[0..6] as SpatialMotion  (free root; else zero)
for i in 1..n:
    Xup[i] = xup_for_joint(link[i], q_slot[i])
    v[i]   = Xup[i] · v[parent] + S[i] · qdot_slot[i]
    c[i]   = v[i] × (S[i] · qdot_slot[i])     // Coriolis / centrifugal bias
    a[i]   = Xup[i] · a[parent] + S[i] · qddot_slot[i] + c[i]
    f[i]   = I[i] · a[i] + v[i] ×* (I[i] · v[i]) − f_ext[i]
```

Pass 2 (leaves → root): extract τ, propagate f up:

```text
for i in (n-1)..1:
    τ_slot[i] = S[i]ᵀ · f[i] + armature[i] · qddot_slot[i]
    f[parent] += Xup[i]ᵀ · f[i]
if root is Free:
    τ[0..6] = f[0] packed as (torque, linear) in root body frame
```

Gravity is applied as per-link body-force `m_i · g_world` at each COM
(rotated to body frame). Same convention as [`crate::tree::aba`], not
Featherstone's "base acceleration = −g" trick — it keeps the free-root
and fixed-root paths algebraically identical.

### CRB (Featherstone RBDA §6, Table 6.2)

Pass 1: compute Xup per link.

Pass 2 (leaves → root): accumulate composite spatial inertias:

```text
Ic[i] = I_link[i] initially (as Mat6)
for i in (n-1)..=1:
    Ic[parent[i]] += Xup[i]ᵀ Ic[i] Xup[i]      // pull_back through Xup
```

Pass 3: fill each column of `M`:

```text
for each joint i (fixed skipped):
    for each subspace column k of S[i]:
        F = Ic[i] · S[i, k]
        // Within-joint block (i-diagonal):
        for each column k' of S[i]:
            M[i_slot+k', i_slot+k] = S[i, k']ᵀ · F
        M[i_slot+k, i_slot+k] += armature[i]
        // Off-diagonal ancestor blocks:
        j_child = i
        while j_child has parent j:
            F = Xup[j_child]ᵀ · F                // pull back one level
            j_child = j
            for each column k'' of S[j]:
                val = S[j, k'']ᵀ · F
                M[j_slot+k'', i_slot+k] = val
                M[i_slot+k,  j_slot+k''] = val   // symmetry
```

For a free root, `S[0]` is the 6-column identity in root body frame.
Walking up terminates at the root and copies the 6 components of the
pulled-back `F` straight into `M[0..6, i_slot+k]`. The 6×6 diagonal
block `M[0..6, 0..6]` is `Ic[0]` materialized as [`Mat6`].

### Cholesky

Standard Cholesky-Banachiewicz form on a symmetric positive-definite
`n × n`. Row-major storage; `L` is lower triangular with zeros in the
strict upper triangle. `L L^T = A`.

```text
for i in 0..n:
    for j in 0..=i:
        sum = A[i, j] − Σ_{k<j} L[i, k] L[j, k]
        if i == j:
            if sum <= 0: not positive-definite → None
            L[i, i] = sqrt(sum)
        else:
            L[i, j] = sum / L[j, j]
```

Solve `L L^T x = b`: forward substitute `L y = b`, then back-substitute
`L^T x = y`. Both are `O(n²)` — cheap on the tree DOFs we care about
(dozens, not thousands). Only transcendental: `f32::sqrt`, which is
IEEE-exact and deterministic on every platform newt targets.

## API

```rust
// Standalone functions in newt::dynamics.
pub fn mass_matrix(tree: &Tree) -> Vec<f32>;
pub fn bias_forces(tree: &Tree, gravity: Vec3) -> Vec<f32>;
pub fn inverse_dynamics(
    tree: &Tree,
    qddot: &[f32],
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
) -> Vec<f32>;
pub fn cholesky(a: &[f32], n: usize) -> Option<Vec<f32>>;
pub fn cholesky_solve(l: &[f32], n: usize, b: &[f32]) -> Vec<f32>;

// Convenience methods on Tree.
impl Tree {
    pub fn mass_matrix(&self) -> Vec<f32>;
    pub fn bias_forces(&self, gravity: Vec3) -> Vec<f32>;
    pub fn inverse_dynamics(
        &self, qddot: &[f32], gravity: Vec3, ext: &ExternalWrenches,
    ) -> Vec<f32>;
}

// World-level (picks up world.gravity).
impl World {
    pub fn mass_matrix(&self, tree_idx: usize) -> Vec<f32>;
    pub fn bias_forces(&self, tree_idx: usize) -> Vec<f32>;
    pub fn inverse_dynamics(
        &self, tree_idx: usize, qddot: &[f32], ext: &ExternalWrenches,
    ) -> Vec<f32>;
}
```

`M` is stored row-major (`M[row * nv + col]`). `qddot` and returned `τ`
follow the same slot layout as `tree.qdot` (see [joints](joints.md)):
6 body-frame spatial slots for a free root, 1 scalar per hinge/slide,
3 body-frame ω slots per ball joint. For a free root, `τ[0..6]` is a
body-frame spatial force at the root COM `(τ_ang, τ_lin)`.

## the ABA↔RNE round-trip identity (primary anchor)

`ABA` and `RNE` are two independent algorithms that share only the
spatial-algebra primitives (`Xform`, `SpatialInertia`, `Mat6`) and the
joint-subspace definitions. Neither can silently confirm the other:
`ABA` builds up an articulated inertia `IA[i]` and eliminates each DOF
in a "rank-1 update" pass; `RNE` walks a rigid-body Newton-Euler
recursion with no articulated-inertia state. If either has a mutation
in its per-link pass — a flipped Xup, a dropped Coriolis term, a
mislabeled subspace column — the identity

```text
qddot = ABA(q, qdot, τ_input)
τ_out = RNE(q, qdot, qddot)
assert τ_out ≈ τ_input   (5e-4 tolerance)
```

will drift. This is why the anchor test runs it over multiple seeded
`(q, qdot, τ)` triples on a mixed-joint tree — five seeds so a lucky
zero-crossing at one configuration cannot pass. Free-root variant
verifies the root-slot path additionally: since ABA does not consume
`qfrc_applied[0..6]` (see [`crate::tree::Tree::qfrc_applied`] docs), the
reconstructed `τ_out[0..6]` must be ~0 (no root wrench was applied).

## anchor tests

| file / name | pin |
|-------------|-----|
| `tests/dynamics_crb_rne.rs::round_trip_aba_rne_reconstructs_tau_fixed_root` | THE primary anchor. Mixed fixed-root tree, hinge/slide/ball, 5 seeded states. `RNE(ABA(τ)) ≈ τ` to 5e-4. Fails on any per-link recursion bug in either algorithm. |
| `tests/dynamics_crb_rne.rs::round_trip_aba_rne_reconstructs_tau_free_root` | Free-root variant. Reconstructed root `τ[0..6]` must be ~0; internal slots reconstruct input. Catches free-root-only bugs (Xup shortcut, root DOF mis-count). |
| `tests/dynamics_crb_rne.rs::mass_matrix_is_symmetric_and_positive_definite` | Fixed-root + free-root scenes, 3 seeds each. `max |M − Mᵀ| < 1e-6 · max|M|`. Cholesky succeeds. Random-probe quadratic form `qdotᵀ M qdot > 0`. |
| `tests/dynamics_crb_rne.rs::mass_matrix_solve_matches_aba` | 3 seeded fixed-root states, `qddot_aba == cholesky_solve(M, τ − bias)` to 5e-4. Cross-validates CRB against ABA via the Cholesky path — the same solve loop the v1 solver will call every step. |
| `tests/dynamics_crb_rne.rs::kinetic_energy_matches_half_qdot_m_qdot` | Both scene kinds, 3 seeds each, armature stripped. Rigid-body KE via world-frame link twists equals `0.5 · qdotᵀ M qdot` (relative tol 5e-4). Independent of CRB's internals — uses only forward kinematics + spatial inertias. |
| `tests/dynamics_crb_rne.rs::single_pendulum_mass_matrix_and_gravity_bias_match_hand_form` | 1-DOF closed form: `M[0][0] = m·L² + armature + I_com`; bias at q=0 (hanging down) = 0; bias at q=π/2 (horizontal) = +m·g·L. |
| `tests/dynamics_crb_rne.rs::rne_static_two_link_arm_matches_hand_gravity_comp` | Two-link planar arm at nontrivial angles (q1=0.3, q2=−0.4). `bias(q, 0)` matches the negated hand-computed world-frame gravity torque at each pivot. |
| `tests/dynamics_golden.rs::dynamics_mass_and_bias_are_byte_identical` | Fixed mixed-joint state (hinge + slide + ball, symmetry broken on every DOF). Serialized `M` (25 f32) + `bias` (5 f32) = 120 bytes, byte-compared against `tests/goldens/dynamics_mass_bias.bin`. macOS-aarch64 regen guard mirrors the tier-1/2/joints goldens. |

### mutation coverage summary

| mutation | test that catches it |
|----------|----------------------|
| missing Coriolis `c[i] = v × Sqdot` in RNE pass 1 | `round_trip_aba_rne_reconstructs_tau_*` — non-zero `qdot` in the seed makes `c[i]` non-zero; drift shows up in τ_out immediately. Also `kinetic_energy_matches_half_qdot_m_qdot` if the bug corrupts the velocity used for KE elsewhere. |
| armature dropped from M's diagonal | `single_pendulum_mass_matrix_and_gravity_bias_match_hand_form` — M[0][0] closed form includes armature; a drop flips the numeric check. Also `mass_matrix_solve_matches_aba` because ABA still uses armature in D, so the two disagree. |
| transposed / wrong-way Xup in the CRB Ic pull-back | `mass_matrix_solve_matches_aba` — the resulting M would not equal ABA's implicit M and Cholesky-solved qddot would drift. Also `mass_matrix_is_symmetric_and_positive_definite` if the transpose asymmetry survives. |
| ball-joint subspace column ordering swap | `round_trip_aba_rne_reconstructs_tau_*` (fixed-root scene includes a ball at the leaf). The reconstructed τ for the ball's 3 DOFs would swap components. |
| free-root packing swap (angular ↔ linear in τ or a[0]) | `round_trip_aba_rne_reconstructs_tau_free_root` — root τ would not zero out; a Cholesky-solve of M would fail to reproduce ABA either. |
| forgetting armature term in RNE (`τ += armature·qddot_slot`) | `round_trip_aba_rne_reconstructs_tau_*` — with armature ≠ 0, RNE would return `τ − armature·qddot` where ABA absorbed the armature term, so the internal-joint reconstruction would miss by exactly `armature · qddot`. |
| off-by-one in Cholesky index bounds | `cholesky_recovers_identity` (unit test) — the reference 3×3 output pins the diagonal + off-diagonal indices tightly. |

## verification

Same quad as tiers 1/2 and joints:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
grep -rnE '\.(sin|cos|tan|exp|ln|powf)\(' newt/src/ \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'
```

The grep must print nothing. `.github/workflows/newt.yml` re-runs these
on every push and PR.

## regenerating the golden (macOS-only)

```sh
cargo test --test dynamics_golden regenerate_dynamics_golden -- --ignored --nocapture
```

The ignored test guards on `target_os = "macos"` + `target_arch =
"aarch64"` and panics on any other host so an accidental `--ignored`
run cannot silently swap the reference bytes.

## non-goals of this tier

- **No sparse M or LDLᵀ / LTDL solvers.** The v1 solver will hit CRB on
  trees with a few dozen DOFs where dense Cholesky is comfortable.
  Sparse factorization is a v3 item (see the design spec's frontier
  tier).
- **No analytic derivatives.** Also v3.
- **No solver.** RNE + CRB feed the future PGS constraint solver; this
  ticket lands only the building blocks.
