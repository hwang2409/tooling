# newt joints (tier 3 / v1 tier 1)

kinematic trees, hinge / slide / ball joints (with limits, damping,
armature), free-root 6-DOF joint, and Featherstone's Articulated Body
Algorithm (ABA) for O(n) forward dynamics. builds on tier 1's
[core](core.md) and tier 2's [contacts](contacts.md). the demos are
**pendulum** (double-pendulum tip trace), **chain** (5-link chain onto
the ground plane), and **cartpole** (cart on a slide + pole on a hinge,
PD servo on the slide).

design of record:
[superpowers/specs/2026-08-13-newt-physics-design.md](superpowers/specs/2026-08-13-newt-physics-design.md).

## data model

a `Tree` is an ordered list of `Link`s. `links[0]` is the root; its
`parent` is `None` and its joint is either `Free` (6-DOF) or `Fixed`
(rigidly anchored at `joint_offset_in_parent`, which for the root is a
world pose). every other link has `parent: Some(i)` with `i < own_index`
(topological order — one forward pass in the vector visits parents before
children).

each link's body frame origin sits at its COM, matching the tier-1 `Body`
convention. `inertia_body` is expressed in that frame about the COM. joint
offsets place the joint anchor relative to the parent's and child's body-
frame origins.

## generalized state (MuJoCo-style dense q/qdot)

`q` (positions) and `qdot` (velocities) are dense vectors laid out by
joint order. slot counts per joint kind:

| joint | `nq` | `nv` | position layout | velocity layout |
|-------|------|------|-----------------|-----------------|
| `Free` (root only) | 7 | 6 | `(px, py, pz, qx, qy, qz, qw)` | `(ωx, ωy, ωz, vx, vy, vz)` body-frame at COM |
| `Fixed` | 0 | 0 | — | — |
| `Hinge` | 1 | 1 | angle (rad) | rate (rad/s) |
| `Slide` | 1 | 1 | displacement (m) | rate (m/s) |
| `Ball` | 4 | 3 | child→parent quaternion `(qx, qy, qz, qw)` (renormalized at step end) | body-frame ω `(ωx, ωy, ωz)` |

`qfrc_applied` is a dense `nv`-vector of user-applied generalized forces
(tier-4 actuators plug in here). zero-initialized on `push_link`.

## joint kind: `Hinge`

```text
JointKind::Hinge {
    axis: Vec3,               // unit axis in the joint frame
    range: Option<(f32, f32)>, // optional (low, high) limits
    damping: f32,             // τ_damp = -damping * qdot
    armature: f32,            // rotor inertia on the axis (kg·m²)
    limit: HingeLimit { stiffness, damping },
}
```

- **axis** — a unit vector in the joint frame; the joint frame coincides
  with the child body frame at `q = 0` (v0 convention: joint-offset
  orientations = IDENTITY). the axis is fixed in both parent and child
  frames.
- **damping** — applied as a joint-local torque `-damping · qdot`. no
  frame issues; it's a scalar on the joint DOF.
- **armature** — enters ABA's diagonal as `Sᵀ IA S + armature`. this
  models reflected rotor inertia (typical for actuated joints) and the
  hand-computed acceleration ratio anchor pins its arithmetic
  ([`joints_damping_armature_limits.rs`](../tests/joints_damping_armature_limits.rs)).
- **range** — enforced as a smooth one-sided spring-damper penalty
  outside the range: `τ_lim = k · violation` (spring, points back toward
  the range) plus `−damping · qdot` (only when moving further outside).
  v0 uses this penalty model; v1 will add real constraints solved with
  the contact solver.

## joint kind: `Slide`

```text
JointKind::Slide {
    axis: Vec3,               // unit axis in the joint frame
    range: Option<(f32, f32)>, // (low, high) displacement limits (m)
    damping: f32,             // F_damp = -damping * qdot
    armature: f32,            // reflected translational inertia (kg)
    limit: JointLimit { stiffness, damping },
}
```

Prismatic (1-DOF translation) joint. Displacement `q` in meters along the
axis. The child body frame stays identically oriented to the parent
(slide never rotates), so ABA's Xup transform for a slide is
`(rot = I, translation = r_jc − r_pj − axis · q)` and the joint subspace
is a pure translation `S = (0, axis)` at the child COM. Range limits use
the same penalty spring-damper model as hinge; units follow the DOF
(N per m of violation for slide vs. N·m per rad for hinge). Armature is
a reflected translational inertia (kg) added on the ABA diagonal —
motor-style, e.g. a rack-and-pinion. Actuator plumbing is unchanged:
[`PdServo`](../src/actuator.rs) actuates hinges *and* slides through the
same 1-DOF `qfrc_applied` slot.

## joint kind: `Ball`

```text
JointKind::Ball {
    damping: f32,   // isotropic angular damping (τ = -damping * ω_body)
    armature: f32,  // rotor inertia added on each rotational axis (kg·m²)
}
```

3-DOF spherical joint. Position is a 4-component child-relative-to-parent
quaternion (`nq = 4`); velocity is a 3-component body-frame angular
velocity (`nv = 3`). ABA's joint subspace is a 3-column S matrix
`S_k = (e_k, r_jc × e_k)` for `k ∈ {0, 1, 2}` (analogous to the hinge
column but for every body axis). The pass-2 articulated-inertia block
`D = Sᵀ IA S + armature · I₃` is a 3×3 matrix; the ABA `IA − U D⁻¹ Uᵀ`
rank-3 update takes the place of the hinge's scalar `1/d`. The
quaternion is renormalized at the end of each RK4 step (same as the
free root — mid-stage renorm breaks the linearity RK4 relies on).

**Ball limits are deferred.** A physically correct 3-DOF orientation
limit (cone, swing/twist) needs the real constraint solver landing in
the next ticket (v1 tier 2 — PGS over solref/solimp). The
[`crate::model`] loader rejects a `range` field on a ball joint with an
error pointing at that deferral, so a user who tries to add limits gets
a clear message instead of a silent no-op.

## ABA in body-frame coordinates

every per-link spatial vector is expressed in that link's body frame at
the COM. Plücker transforms `Xup[i]` move motion from parent body frame
to child body frame; force pull-backs use `Xup.transpose_force` (see
[`spatial::Xform::transpose_force`](../src/spatial.rs)).

Featherstone (Rigid Body Dynamics Algorithms, 2008), algorithm 7.1:

1. **pass 1** (root → leaves): forward kinematics; `v[i] = Xup[i] *
   v[parent] + S[i] · qdot[i]`; Coriolis bias `c[i] = v[i] × (S[i] · qdot[i])`.
2. **pass 2** (leaves → root): initialize `IA[i] = I[i]`, `pA[i] = v[i] ×*
   (I[i] · v[i]) − f_ext[i]`. for each hinge child (bottom-up):
   `d = Sᵀ IA S + armature`, `u_stage = τ − Sᵀ (pA + IA · c)`,
   `pA_reduced = pA + IA · c + IA S · u_stage / d`,
   `IA_reduced = IA − (IA S)(IA S)ᵀ / d`. propagate `IA_parent +=
   Xupᵀ IA_reduced Xup`, `pA_parent += Xupᵀ pA_reduced`.
3. **pass 3** (root → leaves): solve the root's spatial acceleration (6x6
   solve for free root; zero for fixed root). then `qddot[i] = (τ − Sᵀ
   (IA · (Xup · a[parent] + c) + pA)) / d`, `a[i] = Xup · a[parent] + S ·
   qddot[i] + c[i]`.

gravity is added as an external world-frame force `m_i · g` at each link's
COM (rotated into body coords). contact wrenches from tier 2 enter the
same way. this keeps the free-root path symmetric with the fixed-root
path — no special "base acceleration = −g" trick.

### the u_stage form (why not just `τ − Sᵀ pA`?)

the textbook one-liner is `pA_reduced = pA + I_a · c + U · u / d` with
`I_a = IA − U D⁻¹ Uᵀ` (the "reduced" articulated inertia). expanding
`I_a · c` and grouping gives the equivalent form we use:

```text
p_stage  = pA + IA · c
u_stage  = τ − Sᵀ p_stage       // = u − Sᵀ IA c
pA_reduced = p_stage + IA S · u_stage / d
```

both are algebraically identical; the `u_stage` form avoids materializing
`I_a` separately (rank-1-updated matrix) which would double the work per
link. the code carries the derivation in a comment above the update.

a subtle regression this converged on during development: an earlier draft
used `u = τ − Sᵀ pA` (missing the `Sᵀ IA c` term) directly in the
pA-update. that is correct when velocities are zero (`c = 0`) but drifts
under any joint motion — the double-pendulum vs Lagrangian anchor caught
it within 2 s.

## RK4 on generalized coordinates

`tree::rk4_step` runs four sub-stages of the standard Runge-Kutta 4
scheme on `(q, qdot)`. between sub-stages the tree's `q`/`qdot` are
advanced with the derivatives; forward kinematics and ABA are re-run at
each sub-stage. free-root quaternion is renormalized once at step end
(matches tier 1 — mid-stage renorm breaks the linearity RK4 relies on).

`World::step` runs `step_bodies` (unchanged tier-2 path, bit-identical
for tier-1/2 goldens when no tree links are in play) and then
`step_trees` (one `rk4_step` per tree, feeding contact wrenches via a
closure). v0 does NOT support cross-integration between free bodies and
tree links within one sub-stage — a link-vs-body contact would need both
sides interpolated together. link-vs-static (the chain-onto-plane demo)
works out of the box.

## anchor tests

| file | pin |
|------|-----|
| `tests/joints_pendulum.rs::small_amplitude_pendulum_period_matches_analytic` | single hinge; measured swing period within 1% of `2π√(L/g)` — hand-computed reference, not from a second engine call. |
| `tests/joints_pendulum.rs::pendulum_forward_kinematics_traces_a_circular_arc` | FK: COM sweeps the circle of radius L in the swing plane, `x = 0`. |
| `tests/joints_double_pendulum.rs::double_pendulum_matches_hand_lagrangian_over_two_seconds` | two uniform rods vs the Lagrangian mass-matrix formulation coded inline in the test (independent RK4). the primary ABA-coupling anchor. tolerance 0.02 rad on both angles over 2 s. |
| `tests/joints_double_pendulum.rs::double_pendulum_conserves_energy_over_ten_seconds` | K + V drift < 5e-3 over 10 s (`dt = 1 ms`), no damping. catches integrator drift a point-wise trajectory match can miss. |
| `tests/joints_chain_energy.rs::three_link_chain_energy_conserved_over_10k_steps` | 3-link chain with mixed masses/lengths (symmetry broken), gravity on, no damping, no contacts. energy drift < 5e-3 over 10 k steps. |
| `tests/joints_damping_armature_limits.rs::damped_pendulum_peaks_are_strictly_decreasing` | joint damping produces strictly decreasing swing peaks; first peak strictly below release angle. |
| `tests/joints_damping_armature_limits.rs::armature_scales_static_angular_acceleration_by_hand_ratio` | applied torque under armature `A` gives α = τ / (I_pivot + A). ratio matches hand-computed value. |
| `tests/joints_damping_armature_limits.rs::hinge_range_limit_confines_release_from_outside` | released above the upper limit, pendulum ends inside/near range; limit-violation envelope shrinks late-run vs early-run (no energy gain across bounces). |
| `tests/joints_floating_base_momentum.rs::floating_base_conserves_linear_and_angular_momentum` | free-root box + swinging arm, no gravity, no wrenches. `Σ p` and `Σ L` about world origin drift < 5e-4 / 5e-3 over 2 s. **primary anchor for tree-pass math** — a wrong Xup / wrong-force-pull-back bug would silently dump momentum into the "wall" with a fixed root but not with a free root. |
| `tests/joints_floating_base_momentum.rs::free_root_alone_gravity_preserves_body_frame_free_fall` | single free-root link; body-frame linear accel = `ori⁻¹ · gravity_world`. exercises the free-root 6x6 solve independent of any joint chain. |
| `tests/joints_golden.rs::joints_golden_trajectory_is_byte_identical` | 3-link chain + floating-base scene, `(q, qdot)` serialized at steps 0/100/1000, byte-compared against `tests/goldens/joints_chain_and_floating.bin`. |
| `tests/joints_slide.rs::slide_free_fall_matches_closed_form_gravity_acceleration` | mass on a vertical frictionless slide under gravity vs closed-form `q0 + v0 t − ½ g t²`; non-zero initial displacement + rate. |
| `tests/joints_slide.rs::slide_damped_free_fall_approaches_terminal_velocity` | damped slide `v(t) = v_∞ (1 − e^{-(c/m) t})` matches at 1-s intervals through 10 s (5 τ). |
| `tests/joints_slide.rs::slide_range_limit_confines_release_from_outside` | slide analog of the hinge range-limit test — released above the upper limit, settles inside, no violation growth over 6 s. |
| `tests/joints_slide.rs::slide_armature_scales_static_acceleration_by_hand_ratio` | slide armature enters ABA's diagonal as `Sᵀ IA S + armature`; applied force α ratio matches `1 / (1 + A/m)`. |
| `tests/joints_slide.rs::slide_damping_decelerates_a_coasting_slider` | zero-gravity coast: `v(t) = v0 e^{-(c/m) t}` matches at 2 s. |
| `tests/joints_cartpole.rs::cartpole_matches_hand_lagrangian_over_two_seconds` | THE primary slide+hinge coupling anchor. Cart on slide + pole on hinge vs the hand-derived Lagrangian EOM (independent RK4 twin, scalar math only). 2 s window, 5 ms dt, drift < 5e-3 in both `x` and `θ`. |
| `tests/joints_cartpole.rs::cartpole_energy_conservation_no_damping` | mechanical energy drift < 5e-3 over 2 s — catches Coriolis / pA-update sign errors the twin match could hide if both had the same latent bug. |
| `tests/joints_ball.rs::spherical_pendulum_conserves_energy_and_vertical_angular_momentum` | ball joint under gravity: total energy AND `L_z` about the pivot conserved (gravity torque about the pivot has zero z-component). Non-planar ICs → genuinely 3D motion. |
| `tests/joints_ball.rs::ball_joint_with_pivot_at_com_reproduces_torque_free_free_body` | ball joint with anchor at child COM, no gravity → pure torque-free rotation. Compared against tier-1 `Body` with identical inertia + ω. |
| `tests/joints_mixed_golden.rs::joints_mixed_golden_is_byte_identical` | Free + Hinge + Slide + Ball in one tree, symmetry broken on every joint; `(q, qdot)` serialized at 0/100/500/1000 steps, byte-compared against `tests/goldens/joints_mixed.bin`. Cross-platform pin for the whole v1 tier-1 joint set. |

### symmetry-breaking notes (per the tier-2 lesson)

symmetric scenes hide lever-arm-class bugs. every joints anchor scene has
at least one broken symmetry:

- double pendulum: `M1 ≠ M2`, `L1 ≠ L2`; nonzero initial rates in the
  energy variant.
- 3-link chain: mixed masses `[1.1, 0.7, 1.3]`, mixed lengths `[0.6, 0.4,
  0.5]`, staggered initial angles `[0.4, -0.3, 0.2]`.
- floating base: hinge axis `(1, 0.3, 0).normalize()` — not aligned with
  any principal axis of the child; root orientation `Rot((1, 0.4, -0.3),
  0.6)` — not identity.
- golden scene: composes the above (chain + floating base in the same
  binary), also non-zero initial rates on both.

### mutation coverage summary

| mutation | test that catches it |
|----------|----------------------|
| wrong `Sᵀ IA c` in the `pA` update (using `u` instead of `u_stage`) | `joints_double_pendulum::*` — zero-velocity ICs pass, nonzero-velocity ICs drift within 2 s |
| flipped hinge axis sign | `joints_pendulum::small_amplitude_...` — measured period would drift + FK arc trace would land in wrong direction |
| wrong Coriolis bias `c[i]` | `three_link_chain_energy_conserved_...` — drift grows past 5e-3 |
| missing damping torque | `damped_pendulum_peaks_are_strictly_decreasing` — peaks stop decreasing |
| armature missing from `d = Sᵀ IA S` | `armature_scales_static_angular_acceleration_by_hand_ratio` — ratio fails |
| range-limit spring on the wrong side | `hinge_range_limit_confines_release_from_outside` — final angle escapes range |
| any grounded-only bug that dumps momentum "into the wall" | `floating_base_conserves_linear_and_angular_momentum` — direct momentum drift measure |
| any orientation/normal / Plücker sign issue that flips at snapshot 2 | `joints_golden_trajectory_is_byte_identical` — file mismatch |

## running the demos

double pendulum (traces the tip):

```sh
cargo run --release --example pendulum -- --frames 900 --out /tmp/pendulum.ppm --size 640x360
sips -s format png /tmp/pendulum.ppm --out /tmp/pendulum.png
```

5-link chain (settles onto plane; joints + contacts):

```sh
cargo run --release --example chain -- --frames 6000 --out /tmp/chain.ppm --size 640x360
sips -s format png /tmp/chain.ppm --out /tmp/chain.png
```

cart-pole (slide + hinge; PD servo on the slide holds the cart while the
pole swings):

```sh
cargo run --release --example cartpole -- --frames 1200 --out /tmp/cartpole.ppm --size 640x360
sips -s format png /tmp/cartpole.ppm --out /tmp/cartpole.png
```

both write PPMs via chimy2's `Framebuffer`; convert to PNG on macOS with
`sips`, or open the PPM directly in most image viewers.

## regenerating the goldens (macOS-only)

```sh
cargo test --test joints_golden       regenerate_joints_golden       -- --ignored --nocapture
cargo test --test joints_mixed_golden regenerate_mixed_joints_golden -- --ignored --nocapture
```

both ignored tests guard on `target_os = "macos"` + `target_arch =
"aarch64"` and panic on any other host so accidental `--ignored` runs
cannot silently swap the reference. CI on ubuntu-latest re-runs the
normal tests; a bytes mismatch means the scalar policy is being violated
somewhere (probably a new libm call in the engine — the
`libm-free` CI gate catches this too).

## verification

fmt, clippy, tests, libm-free grep — same quad as tiers 1 and 2:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
grep -rnE '\.(sin|cos|tan|exp|ln|powf)\(' newt/src/ \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'
```

the grep must print nothing. `.github/workflows/newt.yml` runs the same
four checks on every push and PR.
