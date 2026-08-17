# newt actuators (v2 tier 2)

MuJoCo's general actuator model: a per-actuator `(gain, bias, dyn,
transmission, clamps)` composition. Every shorthand — `<motor>`,
`<position>`, `<velocity>` — is a specific parameterization of the same
formula, and shares the same `ctrl → activation → force → clamp`
pipeline. Builds on tier 3's [joints](joints.md); the arm and cartpole
demos are the visual pins.

design of record:
[superpowers/specs/2026-08-13-newt-physics-design.md](superpowers/specs/2026-08-13-newt-physics-design.md).

## the general model

```text
u      = clamp(ctrl, ctrl_range)                        (1)
act'   = (u - act) / tau       (only for dyn=filter)    (2)
signal = if dyn=filter { act } else { u }               (3)
gain   = gainprm[0]                              // gaintype=fixed
       | gainprm[0] + gainprm[1]*len + gainprm[2]*vel   // gaintype=affine
bias   = 0                                       // biastype=none
       | biasprm[0] + biasprm[1]*len + biasprm[2]*vel   // biastype=affine
F_raw  = gain * signal + bias                            (4)
τ      = clamp(F_raw * gear, force_range)                (5)
```

`(len, vel)` are the **transmission-space** position and rate — for a
joint transmission with scalar `gear`, `len = gear · q` and
`vel = gear · qdot`. This matches MuJoCo's `actuator_length` /
`actuator_velocity` semantics: a `gear!=1` actuator reads its own
scaled state, not the raw joint coordinate. `Fixed` gain and
`None` bias do not sample `(len, vel)` and so are unaffected by
`gear`, but the affine branches DO — a bug there would silently
under- or over-drive the actuator by a factor of `gear` on the
sampled term. Anchor:
`tests/actuators_general.rs::general_affine_bias_samples_transmission_space`.

Clamping order (ordered because the two clamps are different constraint
kinds and MuJoCo defines this):
- **ctrl clamp** binds FIRST (step 1). The activation filter (step 2)
  sees the clamped `u`, not the raw `ctrl`.
- **force clamp** binds LAST (step 5). It bounds the final joint force
  after gear multiplication.

## activation dynamics (`dyntype=filter`)

Filter dynamics low-pass the control input through a first-order ODE
with time constant `tau = dynprm[0]`:

```text
act' = (u - act) / tau
```

Integrated with **forward Euler at each step boundary**:

```text
act ← act + (dt/tau) · (u - act)
```

ZOH within the step: all four RK4 sub-stages of `Tree::step` see `act`
frozen at its step-start value; the update runs ONCE per step at the
end of [`rk4_step`](../src/tree.rs) after mechanical state has
integrated. This is the same convention already used for `ctrl`,
`qfrc_applied`, and `applied_wrenches`, and mirrors MuJoCo's Euler
activation integrator for `integrator=Euler`.

For `integrator=RK4`, real MuJoCo also integrates activation through
the same RK4 stages, so newt's forward-Euler act picks up a small,
bounded, documented residual in the differential scenario
`filtered_motor_pendulum` (see
[differential.md](differential.md)). Keeping the activation update at
the step boundary avoids the alternative — a 4-stage RK4 on `act`
that would need to re-evaluate `ctrl` inside each sub-stage — for
which newt has no interior `ctrl` snapshot machinery in v2 tier 2.

Edge cases:
- **`tau < dt` (alpha > 1)**: forward Euler overshoots (documented).
  With `alpha=1` exactly, `act` jumps to `u` in one step. With
  `alpha ∈ (1, 2)`, `act` overshoots then oscillates back. With
  `alpha ≥ 2`, the recurrence is unstable — pick `tau > dt`.
  `Actuator::general` rejects `tau <= 0`.
- **`actearly=true`** (MuJoCo): NOT supported. The MJCF loader rejects
  it with a message pointing here. `actearly` evaluates gain/bias at
  NEXT-step forward-kinematics values; wiring this in requires
  deferring the activation update past the mechanical step, which is
  invasive enough to warrant its own tier.

## shorthand mapping (position/velocity/motor)

Every shorthand is a `<general>` under the hood. `Actuator` stores a
`flavor` enum that keeps the evaluation path fast AND, for `Position`,
byte-identical to the v0 `PdServo`. The mapping:

| Shorthand    | gaintype | gainprm       | biastype | biasprm            | gear | dyn  | Torque expression         |
|--------------|----------|---------------|----------|--------------------|------|------|---------------------------|
| `<motor>`    | fixed    | `[1, 0, 0]`   | none     | `[0, 0, 0]`        | gear | none | `gear · ctrl`             |
| `<position>` | fixed    | `[kp, 0, 0]`  | affine   | `[0, -kp, -kv]`    | 1    | none | `kp·(ctrl − q) − kv·qdot` |
| `<velocity>` | fixed    | `[kv, 0, 0]`  | affine   | `[0, 0, -kv]`      | 1    | none | `kv·(ctrl − qdot)`        |

`<general>` accepts every field explicitly. See MJCF/JSON details below.

## muscle actuators

`<muscle>` and muscle-typed `<general>` actuators use MuJoCo's shared
force-length-velocity functions. The same formulas serve joint and tendon
transmissions.

For normalized length `L` and normalized velocity `V`, active force is:

```text
FL(L) = 0                         L < lmin or L > lmax
       = 0.5*x²                   lmin ≤ L ≤ a, x=(L-lmin)/(a-lmin)
       = 1 - 0.5*x²               a < L ≤ 1, x=(1-L)/(1-a)
       = 1 - 0.5*x²               1 < L ≤ b, x=(L-1)/(b-1)
       = 0.5*x²                   b < L ≤ lmax, x=(lmax-L)/(lmax-b)

FV(V) = 0                         V ≤ -1
       = (V+1)²                    -1 < V ≤ 0
       = fvmax - (fvmax-1-V)²/(fvmax-1)  0 < V ≤ fvmax-1
       = fvmax                     V > fvmax-1

gain = -F0 · FL(L) · FV(V)
```

Here `a=(lmin+1)/2`, `b=(1+lmax)/2`,
`L=range0 + (length-lengthrange0)/L0`,
`V=velocity/(L0·vmax)`, and
`L0=(lengthrange1-lengthrange0)/(range1-range0)`. Passive bias is zero
through `L=1`, then follows the quadratic-to-linear curve controlled by
`fpmax`. `F0=force` when force is nonnegative; force `-1` uses
`scale/max(MINVAL, acc0)`.

Muscle activation uses `dynprm=[tau_act,tau_deact,tausmooth]`:

```text
tau_act   = tau_act0 · (0.5 + 1.5·clamp(act,0,1))
tau_deact = tau_deact0 / (0.5 + 1.5·clamp(act,0,1))
dctrl     = clamp(ctrl,0,1) - act
act'      = dctrl / tau
```

With `tausmooth=0`, `tau` is `tau_act` when `dctrl>0`, else
`tau_deact`. Smoothing blends them with MuJoCo's clamped quintic sigmoid.
Euler updates muscle state at the step boundary. RK4 updates it at each
mechanical stage.

The supported auto-default subset uses MuJoCo's default curve and time
parameters. `lengthrange` is required because this loader does not run
MuJoCo's simulation-based compiler search. Omitted `acc0` uses the explicit
subset default `1`; provide `acc0` with `force=-1` for a compiled-model
match. Missing values never disappear silently.

Muscles may target a joint or tendon. Wrapped tendon length and Jacobian
calculation stays in the existing tendon path. Muscle velocity terms enter
`implicitfast` only for joint transmissions. Tendon velocity terms remain
explicit because their `JᵀJ` contribution is dense.

### muscle MJCF and JSON

```xml
<muscle name="m" joint="hinge" lengthrange="0 1"
        timeconst="0.01 0.04" tausmooth="0"/>
<general name="g" joint="hinge" gaintype="muscle"
         biastype="muscle" dyntype="muscle" lengthrange="0 1"
         gainprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2"
         biasprm="0.75 1.05 -1 200 0.5 1.6 1.5 1.3 1.2"
         dynprm="0.01 0.04 0"/>
```

JSON uses `type:"muscle"` with `lengthrange`, curve fields, `timeconst`,
and optional `gear`, `ctrlrange`, `forcerange`, `ctrllimited`, `forcelimited`,
and `acc0`. A muscle general actuator requires explicit `gaintype`,
`biastype`, `dyntype`, `gainprm`, and `biasprm`; omitted `dynprm` compiles as
`[1,0,0]`. A range is active by default when its matching limited flag is
omitted. Both loaders reject unknown fields and missing muscle-general
parameters.

### byte-identity: Position ↔ v0 PdServo

The `Position` flavor is stored with `(kp, kv)` and evaluated as
`kp*(ctrl-len) - kv*vel` with `clamp_symmetric` — literally the ops
v0's `PdServo::torque_unclamped` used. Every position anchor in
`tests/actuators_pd.rs` and the arm golden
`tests/goldens/actuators_arm_waypoints.bin` still passes bit-equal
after the migration.

A `<general>` actuator with position-shaped parameters
(`gainprm=[kp,0,0]`, `biasprm=[0,-kp,-kv]`) is algebraically the same
but NOT bit-equal — the expansion is `kp*ctrl + (-kp*len) + (-kv*vel)`,
different associativity. `tests/actuators_general.rs::
general_position_shape_matches_shorthand_arm` bounds that residual at
< 1e-4 over 500 RK4 steps on the arm scene. If you need bit-parity
with the v0 PD path, use the shorthand.

### the `<motor>` semantics change (v1 → v2)

The v1 MJCF loader stubbed `<motor>` as a PD servo with `kp=0, kd=0`
— the actuator emitted zero torque and required the caller to write
into `qfrc_applied` directly. v2 tier 2 fixes this: `<motor gear="G">`
now correctly emits `torque = gear · ctrl`, and `ctrl` is set via
`Tree::set_actuator_target`. Existing
`<motor>` fixtures that expected zero torque and drove torque
externally will now feel actuator torque; the v1 subset had no such
fixtures.

## MJCF surface

```xml
<actuator>
  <position name="j_pos" joint="j" kp="200" kv="14.14"  forcerange="-60 60"/>
  <velocity name="j_vel" joint="j" kv="15"              forcerange="-30 30"/>
  <motor    name="j_mot" joint="j" gear="10"            forcerange="-25 25"/>
  <general  name="j_gen" joint="j"
            gaintype="fixed" gainprm="1 0 0"
            biastype="none"  biasprm="0 0 0"
            dyntype="filter" dynprm="0.05"
            gear="1" forcerange="-5 5" ctrlrange="-1 1"/>
</actuator>
```

Attributes accepted per element (v2 tier 2 subset):

- **`<position>`** — `kp` (required), one of `kv | dampratio`,
  `forcerange`, `ctrlrange` (enforced — clamps `ctrl` before torque
  eval, MuJoCo parity), `class`, `ctrllimited`/`forcelimited`
  (parsed then accepted), `target` (newt extension — initial ctrl at
  load time). `dampratio` derives `kv = 2·dr·√(kp·1)` — see [the
  derivation below](#deriving-kv-from-a-damping-ratio-dampratio).
- **`<velocity>`** — `kv` (required), `forcerange`, `ctrlrange`
  (enforced — clamps `ctrl` before torque eval), `class`,
  `ctrllimited`/`forcelimited`.
- **`<motor>`** — `gear` (scalar; MuJoCo 6-vector accepted, only the
  first entry honored for 1-DOF joints), `forcerange`, `ctrlrange`
  (enforced — clamps `ctrl` before torque eval), `class`,
  `ctrllimited`/`forcelimited`. Malformed `ctrlrange`/`forcerange`
  values (wrong count, `lo >= hi`) are rejected at load.
- **`<general>`** — `gaintype ∈ {fixed, affine}`, `gainprm` (1..=3
  numbers; missing tail defaults `[1,0,0]`), `biastype ∈ {none,
  affine}`, `biasprm` (1..=3 numbers; missing tail defaults `[0,0,0]`),
  `dyntype ∈ {none, filter}`, `dynprm` (first number is `tau`),
  `gear` (scalar; 6-vector accepted with only-first-honored),
  `forcerange` `"lo hi"`, `ctrlrange` `"lo hi"`, `class`,
  `ctrllimited`/`forcelimited`. `actearly="true"` is rejected.

`<cylinder>`, `<damper>`, `<muscle>`, `<intvelocity>` remain outside
the subset.

## JSON surface

Types (`"position" | "velocity" | "motor" | "general"`) live under
`actuators[i].type`. Every entry needs `name`, `tree`, `link`. Per-
type fields:

- **position** — `kp`, one of `kd | (dampratio + reflected_inertia)`,
  optional `clamp` (symmetric magnitude), `target`.
- **velocity** — `kv`, optional `clamp`, `target` (initial ctrl).
- **motor** — optional `gear` (default 1), optional `clamp`, `target`.
- **general** — `gaintype`, `gainprm` (3-array), `biastype`,
  `biasprm` (3-array), `gear`, `dyntype`, `dynprm` (scalar),
  `ctrlrange` (2-array), `forcerange` (2-array), `target`.

See `docs/model-format.md` for the full schema and the arm/pendulum
examples in `models/`.

## deriving `kv` from a damping ratio (`dampratio`)

A single hinge with reflected inertia `I_ref` under PD control obeys

```text
I_ref · q̈ = kp · (target − q) − kv · qdot
```

a linear second-order system with natural frequency `ωₙ = √(kp / I_ref)`
and damping ratio `ζ = kv / (2·√(kp · I_ref))`. Solving for `kv`:

```text
kv = 2 · ζ · √(kp · I_ref)
```

behavior by ratio:

- **ζ = 1** — critical damping. fastest settle with no overshoot.
- **ζ < 1** — underdamped. analytic overshoot `Mp = exp(−ζ·π/√(1−ζ²))`.
  at ζ = 0.2 the peak sits ≈ 53 % above the target.
- **ζ > 1** — overdamped. no overshoot, slower settle.

`Actuator::position_from_dampratio(link, kp, ζ, I_ref, force_range)`
computes it. The MJCF `<position dampratio="...">` shorthand assumes
`I_ref = 1` (matches MuJoCo's `meaninertia` approximation for a fresh
scene — documented in `docs/mjcf.md`).

### why the CALLER supplies `I_ref`

`Sᵀ IA S` at a hinge is available at ABA pass 2, but the true diagonal
mass-matrix element requires the FULL articulated inertia at the joint,
which depends on the entire subtree (recouples the actuator to
topology). instead the actuator carries an `I_ref` fixed at construction:

- for a hinge that swings a point-mass at length `L` with rotor armature
  `A`, [`reflected_inertia_point_mass(m, L, A)`](../src/actuator.rs)
  returns `m·L² + A`.
- for a uniform rod pivoted at one end (`I_pivot = m·L²/3`), pass
  `length = L/√3` to that helper, or use raw [`Actuator::position`]
  with your own `kv`.

## timing: ZOH-per-step ctrl + activation, re-evaluated per RK4 stage

Every actuator's `ctrl` is a **zero-order hold** across a whole
[`rk4_step`](../src/tree.rs): the value at step start is what every
sub-stage of that step sees. Write a new `ctrl` between steps via
[`Tree::set_actuator_target`](../src/tree.rs) — the change takes
effect on the NEXT step,
matching MuJoCo's discrete-time control convention.

Each actuator's **torque**, on the other hand, is **re-evaluated at
every RK4 sub-stage** using that stage's interpolated `(q, qdot)`.
This is what makes the PD/velocity/filter terms act as genuine
continuous-time forcing functions rather than explicit-Euler kicks at
step start.

**Activation state** `act` is ZOH within the step (like `ctrl`) and
updated once at step end via forward Euler on the filter ODE. See the
"activation dynamics" section above for edge cases.

## direct joint torque (motor-style)

still supported alongside the actuator channel. writes go directly into
`qfrc_applied` — either raw, or via the convenience wrapper that
clamps for you:

```rust
tree.set_joint_torque_clamped(link_idx, torque, force_range);
tree.clear_qfrc_applied();
```

`qfrc_applied` persists across steps. call `set_joint_torque_clamped`
each step to update, or `clear_qfrc_applied` to zero every slot at once.
An actuator's output adds to `qfrc_applied` inside ABA; they compose.

## external 6D wrenches on links

Set a persistent per-link world-frame wrench (force at COM, torque about
COM) that stays live for as long as you want:

```rust
tree.set_link_wrench(link_idx, force_world, torque_world);
tree.clear_applied_wrenches();
```

Inside [`aba`](../src/tree.rs), the applied wrench sums with the
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
| `tests/actuators_general.rs::general_position_shape_matches_shorthand_arm` | Migration anchor: general actuator with position-shape params tracks position shorthand within 1e-4 over 500 RK4 steps on the arm. |
| `tests/actuators_general.rs::velocity_actuator_tracks_target_rate_under_gravity` | velocity actuator lag envelope `\|ctrl - qdot\| ≤ (m·g·L + damping·qdot) / kv` holds. |
| `tests/actuators_general.rs::filter_step_response_matches_forward_euler_closed_form` | filter act at n=50/100/250/500 steps matches `1 - (1 - dt/tau)^n` closed form. |
| `tests/actuators_general.rs::ctrl_clamp_binds_before_force_clamp` | motor with tight ctrl clamp saturates at gear·ctrl_hi, NOT force_hi. |
| `tests/actuators_general.rs::affine_bias_zero_at_hand_derived_equilibrium` | bias `[k, -k, 0]` produces zero at `len=1`; hand-computed cases at 0.5 / 1.0 / 2.0. |
| `tests/actuators_general.rs::activation_integrates_once_per_rk4_step` | filter act after 50 rk4_steps on a hinge matches the closed-form Euler recurrence. |
| `tests/differential.rs::differential_velocity_cartpole` | velocity actuator on slide vs real MuJoCo — L∞ divergence at f32-quant scale (1e-6 qpos/qvel). |
| `tests/differential.rs::differential_filtered_motor_pendulum` | filter-motor step response vs real MuJoCo — bounded divergence 2.56e-4 rad qpos, 1.34e-3 rad/s qvel (documented; forward-Euler vs MuJoCo RK4 activation gap). |

### mutation coverage summary

| mutation | test that catches it |
|----------|----------------------|
| sign flip on actuator torque | `servo_holding_pendulum_against_gravity_settles_no_growth` — unstable pole, amplitude grows |
| clamp dropped or wrong sign | `servo_clamp_bounds_the_effective_torque` — α wrong |
| clamp binds on P only, not full P+D | `servo_clamp_binds_on_full_pd_expression_not_p_only` — with `qdot ≠ 0`, α splits +2 (correct) vs −18 (mutant) |
| wrong `qdot` in servo (e.g. positional) | `servo_critical_damping_step_no_overshoot` — kv term dead → overshoots for ζ=1 |
| `dampratio` formula off by a factor | `servo_underdamped_step_overshoots` — peak in wrong bracket |
| actuator routed to wrong link | `actuators_golden_trajectory_is_byte_identical` — snapshot 1+ shifts |
| wrench summed into wrong body / dropped for articulated links | `constant_wrench_on_tip_link_produces_hand_derived_static_deflection` — angle blows |
| wrench applied only to free-root (not articulated) | same, plus `actuators_golden_trajectory_is_byte_identical` — snapshot bytes shift |
| direct-torque clamp missing | `direct_joint_torque_and_clamp_flow_into_aba` — mismatch vs hand-clamped |
| general velocity flavor: bias sign flipped | `differential_velocity_cartpole` — MJ trajectory diverges to O(m) |
| general filter: signal fed from ctrl not act | `filter_signal_is_act_not_ctrl` (in actuator.rs unit tests) |
| force clamp bound before ctrl clamp | `ctrl_clamp_binds_before_force_clamp` — swap of 20 vs 25 |
| activation integrated with wrong `alpha` | `filter_step_response_matches_forward_euler_closed_form` — wrong n=50/100/250 values |

## running the demos

```sh
# Arm (three PD position servos).
cargo run --release --example arm -- --frames 1800 --out /tmp/arm.ppm --size 640x360
# Cartpole, position mode (default) — PD holds cart at x=0, pole swings.
cargo run --release --example cartpole -- --frames 900 --out /tmp/cartpole.ppm
# Cartpole, velocity mode — cart tracks 0.6·sin(2π·t/1.5s) m/s.
cargo run --release --example cartpole -- --velocity --frames 900 --out /tmp/cartpole_vel.ppm
sips -s format png /tmp/arm.ppm --out /tmp/arm.png
```

The velocity-mode caption prints `final cart_v` — the sample of the
last cart rate — matched against the sinusoidal profile at `t = 4.5 s`.

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
