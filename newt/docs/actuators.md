# newt actuators (tier 4)

PD position servos, motor-style direct joint torques, and per-link
world-frame external wrenches. Builds on tier 3's [joints](joints.md).
the tier's showpiece demo is **arm** — a 3-link commanded arm executing a
three-waypoint reach sequence (up → sideways → settle).

design of record:
[superpowers/specs/2026-08-13-newt-physics-design.md](superpowers/specs/2026-08-13-newt-physics-design.md).

## the three actuation channels

every controlled input funnels through one of these; all three sum inside
ABA before integrating.

| channel | data | where it enters |
|---------|------|-----------------|
| PD position servo | `Tree::actuators: Vec<PdServo>` | pass 2 hinge branch, added to `tau_scalar` |
| direct joint torque | `Tree::qfrc_applied: Vec<f32>` | pass 2 hinge branch, part of `tau_scalar` |
| external 6D wrench | `Tree::applied_wrenches: Vec<(Vec3, Vec3)>` | pass 2 preamble, summed with caller-supplied `external_wrenches` (contacts) before rotation into body coords |

## PD position servo (`actuator::PdServo`)

```text
τ_raw = kp · (target − q) − kd · qdot
τ     = clamp(τ_raw, −force_range, +force_range)      // force_range ≤ 0 → no clamp
```

- **kp** — position gain (N·m per rad of error).
- **kd** — velocity gain (N·m per rad/s).
- **force_range** — symmetric torque clamp. `0` or negative disables it.
- **target** — desired joint angle (rad); settable per step via
  [`Tree::set_actuator_target`](../src/tree.rs).

### deriving `kd` from a damping ratio (`from_dampratio`)

a single hinge with reflected inertia `I_ref` under PD control obeys

```text
I_ref · q̈ = kp · (target − q) − kd · qdot
```

a linear second-order system with natural frequency `ωₙ = √(kp / I_ref)`
and damping ratio `ζ = kd / (2·√(kp · I_ref))`. solving for `kd`:

```text
kd = 2 · ζ · √(kp · I_ref)
```

behavior by ratio:

- **ζ = 1** — critical damping. fastest settle with no overshoot.
- **ζ < 1** — underdamped. analytic overshoot `Mp = exp(−ζ·π/√(1−ζ²))`.
  at ζ = 0.2 the peak sits ≈ 53 % above the target.
- **ζ > 1** — overdamped. no overshoot, slower settle.

this mirrors MuJoCo's `<position kp="…" dampratio="…"/>` idiom used in
the biped work.

### why the CALLER supplies `I_ref`

`Sᵀ IA S` at a hinge is available at ABA pass 2, but the true diagonal
mass-matrix element requires the FULL articulated inertia at the joint,
which depends on the entire subtree (recouples the actuator to
topology). instead the servo carries an `I_ref` fixed at construction:

- for a hinge that swings a point-mass at length `L` with rotor armature
  `A`, [`reflected_inertia_point_mass(m, L, A)`](../src/actuator.rs)
  returns `m·L² + A`.
- for a uniform rod pivoted at one end (`I_pivot = m·L²/3`), pass
  `length = L/√3` to that helper, or use raw [`PdServo::new`] with your
  own `kd`.

field tuning is straightforward: measure the effective joint inertia
once, plug in, iterate on `kp` and `ζ` from there. the biped model uses
`kp` 45–80, `ζ` ≈ 1, `force_range` 45–80 N — plugged into the same
formulas.

## direct joint torque (motor-style)

when the outer loop wants to inject a raw torque (feedforward, gravity
compensation, custom control law), write into `qfrc_applied` — either
directly, or via the convenience wrapper that clamps for you:

```rust
tree.set_joint_torque_clamped(link_idx, torque, force_range);
tree.clear_qfrc_applied();
```

`qfrc_applied` persists across steps. call `set_joint_torque_clamped`
each step to update, or `clear_qfrc_applied` to zero every slot at once.

## external 6D wrenches on links

set a persistent per-link world-frame wrench (force at COM, torque about
COM) that stays live for as long as you want:

```rust
tree.set_link_wrench(link_idx, force_world, torque_world);
tree.clear_applied_wrenches();
```

inside [`aba`](../src/tree.rs), the applied wrench sums with the
caller-supplied `external_wrenches` (contacts) BEFORE rotation into body
coords:

```text
force_world  = force_world_ext + force_world_applied
torque_world = torque_world_ext + torque_world_applied
force_body   = ori⁻¹ · (force_world + gravity · mass)
torque_body  = ori⁻¹ ·  torque_world
```

so articulated links see the wrench through EXACTLY the same
`f_ext_body` path free-root bodies do — no special case. this is
verified by
[`constant_wrench_on_tip_link_produces_hand_derived_static_deflection`](../tests/actuators_wrench.rs).

## anchor tests

| file | pin |
|------|-----|
| `tests/actuators_pd.rs::servo_critical_damping_step_no_overshoot` | ζ=1 step: observed max angle ≤ target within 1e-4. |
| `tests/actuators_pd.rs::servo_underdamped_step_overshoots` | ζ=0.2 step: observed peak in [0.72, 0.81] for target 0.5 (analytic Mp ≈ 0.5269). |
| `tests/actuators_pd.rs::servo_p_only_steady_state_error_under_gravity` | pendulum with P-only servo, target = π/6: Newton-solved q_ss ≈ 0.4785 matches within 5e-4 rad. |
| `tests/actuators_pd.rs::servo_clamp_bounds_the_effective_torque` | `aba` α at rest equals `clamped_τ / I` exactly, not `kp·error / I`. |
| `tests/actuators_pd.rs::direct_joint_torque_and_clamp_flow_into_aba` | `set_joint_torque_clamped` and raw `qfrc_applied` produce the same qddot. |
| `tests/actuators_pd.rs::servo_holding_pendulum_against_gravity_settles_no_growth` | swing amplitude late-run ≤ early-run — catches sign-flipped actuator torque. |
| `tests/actuators_wrench.rs::constant_wrench_on_tip_link_produces_hand_derived_static_deflection` | 2-link chain, world-y force at tip COM: `q1 = (L₁+L₂/2)·F/kp`, `q2 = L·F/(2·kp)`. |
| `tests/actuators_wrench.rs::wrench_clear_zeros_all_links` | `clear_applied_wrenches` resets every slot. |
| `tests/actuators_golden.rs::actuators_golden_trajectory_is_byte_identical` | 3-link arm, two-waypoint sequence with servos + persistent motor torque + persistent wrench — `(q, qdot, qfrc_applied)` serialized at steps 0/500/1000/1500, byte-compared against `tests/goldens/actuators_arm_waypoints.bin`. |

### mutation coverage summary

| mutation | test that catches it |
|----------|----------------------|
| sign flip on actuator torque | `servo_holding_pendulum_against_gravity_settles_no_growth` — unstable pole, amplitude grows |
| clamp dropped or wrong sign | `servo_clamp_bounds_the_effective_torque` — α wrong |
| wrong `qdot` in servo (e.g. positional) | `servo_critical_damping_step_no_overshoot` — kd term dead → overshoots for ζ=1 |
| `dampratio` formula off by a factor | `servo_underdamped_step_overshoots` — peak in wrong bracket |
| actuator routed to wrong link | `actuators_golden_trajectory_is_byte_identical` — snapshot 1+ shifts |
| wrench summed into wrong body / dropped for articulated links | `constant_wrench_on_tip_link_produces_hand_derived_static_deflection` — angle blows |
| wrench applied only to free-root (not articulated) | same, plus `actuators_golden_trajectory_is_byte_identical` — snapshot bytes shift |
| direct-torque clamp missing | `direct_joint_torque_and_clamp_flow_into_aba` — mismatch vs hand-clamped |

## running the demo

```sh
cargo run --release --example arm -- --frames 1800 --out /tmp/arm.ppm --size 640x360
sips -s format png /tmp/arm.ppm --out /tmp/arm.png
```

renders the arm at its final pose (three coloured rods stacked vertically
at rest) plus a cool-to-warm tip trail that traces the reach sequence:
up → sideways → settle. base pivot marked with a white cross.

## regenerating the golden (macOS-only)

```sh
cargo test --test actuators_golden regenerate_actuators_golden -- --ignored --nocapture
```

the ignored test guards on `target_os = "macos"` + `target_arch =
"aarch64"` and panics on any other host — accidental `--ignored` runs
cannot silently swap the reference. CI on ubuntu-latest re-runs the
normal test; a bytes mismatch means the scalar policy is being violated
somewhere (probably a new libm call in the engine — the `libm-free` CI
gate catches this too).

## verification

the standard fmt / clippy / test / libm-free grep quad — same as tiers
1-3:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
grep -rnE '\.(sin|cos|tan|exp|ln|powf)\(' newt/src/ \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*//'
```

the grep must print nothing. `.github/workflows/newt.yml` runs the same
four checks on every push and PR.
