# Differential parity: newt vs real MuJoCo

Status: NEWT-14 (v2 tier 1). Both v1 open findings — box_stack
collapse and the 179 μm sphere-drop steady-state penetration
offset — are closed here. Capture tool at
`tools/capture_mujoco.py`; Rust suite at `tests/differential.rs`;
shared scenario manifest at `tests/references/scenarios.json`.

This document is the honest map of where newt stands against real
MuJoCo (the reference implementation) on a fixed battery of scenarios.
Each row of the scorecard is the MEASURED maximum divergence over the
sample points of the trajectory, the tolerance the CI test asserts,
and the verdict.

The rule (from the arc's integrity standard, verbatim in NEWT-13's
contract): a divergence beyond physical reasonableness is a FINDING
to report, not to hide. Bounds are set from measurement plus ~2×
headroom; open findings get their own section below and are called
out at the top of the PR body.

## How the harness works

- **Reference** — `tools/capture_mujoco.py` loads each scenario's
  MJCF under the biped-venv MuJoCo, configures the CLOSEST comparable
  settings (RK4, PGS, pyramidal cone, matching timestep and iteration
  count — set in the MJCF's `<option>` block so both engines read the
  same numbers), steps N times, samples `qpos` / `qvel` every `stride`
  steps, and writes a compact binary fixture with a single-line
  provenance header (mujoco version, capture date, settings). The
  fixtures live under `tests/references/*.bin` and ARE committed —
  CI needs no Python or MuJoCo.
- **Comparison** — `tests/differential.rs` reads the same MJCF via
  newt's loader, applies the same initial `qpos` / `qvel` overrides
  and actuator targets from `scenarios.json`, steps newt with its
  own PGS solver, extracts state into MuJoCo layout, and asserts the
  L∞ component divergence at each sampled step is within tolerance.
  MuJoCo values arrive as f64 and stay f64; newt values are widened
  from f32 for the comparison.
- **Regeneration** — run `python newt/tools/capture_mujoco.py`
  (or `--force` if MuJoCo has been upgraded — the tool refuses to
  overwrite fixtures captured with a different MuJoCo version
  otherwise, mirroring the golden regen guard). Fixtures diff cleanly
  so a regeneration always shows exactly which trajectories moved.

## Scorecard

Numbers are the observed max component divergence over all sample
points of one full trajectory. `qpos` errors are in the scenario's
native units (m for translations, dimensionless for quaternion
components, rad for hinge angles); `qvel` errors are per-second of
those units.

| Scenario           | Observed max qpos | Bound qpos | Observed max qvel | Bound qvel | Verdict            |
|--------------------|-------------------|------------|-------------------|------------|--------------------|
| ballistic          | 1.24e-5           | 3.0e-5     | 3.71e-5           | 8.0e-5     | parity             |
| tumble             | 1.59e-3           | 4.0e-3     | 2.03e-2           | 5.0e-2     | parity             |
| double_pendulum    | 1.33e-7           | 5.0e-7     | 6.23e-7           | 2.0e-6     | parity             |
| servo_arm          | 7.03e-4           | 2.0e-3     | 3.06e-2           | 8.0e-2     | parity             |
| floating_base      | 2.64e-6           | 6.0e-6     | 2.77e-6           | 6.0e-6     | parity             |
| sphere_drop_stiff  | 5.24e-2           | 8.0e-2     | 6.25e-1           | 8.0e-1     | bounded divergence |
| sphere_drop        | 5.69e-3           | 1.0e-2     | 4.72e-1           | 1.0e+0     | bounded divergence |
| sphere_drop_soft   | 4.55e-2           | 7.0e-2     | 3.80e-1           | 8.0e-1     | bounded divergence |
| box_stack          | 1.11e-2           | 5.0e-2     | 7.68e-2           | 5.0e-1     | parity (recovered) |
| joint_limit_swing  | 1.08e-1           | 1.5e-1     | 9.10e-1           | 1.2e+0     | bounded divergence |
| velocity_cartpole  | 2.70e-7           | 1.0e-6     | 5.77e-7           | 2.0e-6     | parity             |
| filtered_motor_pendulum | 2.56e-4      | 6.0e-4     | 1.34e-3           | 3.0e-3     | bounded divergence |
| mocap_rangefinder | 0                  | 1.0e-6     | 0                  | 1.0e-6     | parity; sensors 2.4e-8 |

`mocap_rangefinder` also compares all seven `sensordata` values at each
sample. The maximum direct sensor error is `2.4e-8`, below the `2.0e-6`
sensor bound. It covers rangefinder ray casting, mocap site attachment,
velocimeter output, and magnetic-field frame conversion.

### sphere_drop solref sweep (steady-state penetration)

The load-bearing NEWT-14 signal from Finding 2: after the
split-α reference-term fix, newt's steady-state penetration
tracks MuJoCo **within the fitted window `tc ∈ [0.010, 0.050]`**
at dampratio = 1. Numbers are z-center at t=3 s (well past
bounce and settle); pen = 0.1 − z. Fixtures:
`sphere_drop_stiff.xml` (tc=0.010), `sphere_drop.xml` (tc=0.020,
default), `sphere_drop_soft.xml` (tc=0.050). The dedicated test
that guards this is
`sphere_drop_steady_state_penetration_matches_mujoco` in
`tests/differential.rs`.

| tc     | mj_z         | newt_z       | mj_pen (μm) | newt_pen (μm) | gap (μm) |
|--------|--------------|--------------|-------------|---------------|----------|
| 0.010  | 0.09994287   | 0.09994562   | 57.1        | 54.4          | 2.8      |
| 0.020  | 0.09978356   | 0.09979220   | 216.4       | 207.8         | 8.6      |
| 0.050  | 0.09925940   | 0.09923524   | 740.6       | 764.8         | 24.2     |

Pre-NEWT-14 newt at tc=0.020 sat at 216 μm below the same MuJoCo
point (395 μm penetration where MJ was 216 μm). See the NEWT-14
Finding 2 note below for the derivation and the empirical fit.

**Outside the fitted window the fit does not extrapolate — see
the "MuJoCo k_impedance functional form outside fitted window"
open finding below for numbers and scope.**

### Energy scorecard (long-horizon)

For chaotic scenarios where component-wise state comparison is not
meaningful over long horizons, we compare TOTAL mechanical energy
drift on both sides. `newt_drift` = `max_t |E_newt(t) - E_newt(0)|`;
`mujoco_drift` = same for MuJoCo's reference trajectory; `gap` =
`max_t | |ΔE_newt(t)| - |ΔE_mj(t)| |`. All three must sit under
their per-scenario bounds.

| Scenario                  | Horizon | newt_drift | mujoco_drift | gap      | Bound (each) | Verdict |
|---------------------------|---------|------------|--------------|----------|--------------|---------|
| double_pendulum_energy    | 5.0 s   | 2.76e-6    | 3.38e-8      | 2.74e-6  | 6e-6 / 8e-8 / 6e-6 | parity  |

## Per-scenario notes

### ballistic (parity)

Single free body under gravity, 2 s horizon, no contacts. Nonzero
initial linear velocity `(3.0, -1.5, 5.0)` breaks axis symmetry so
a bug that zeroed one component would be caught. Observed maxima are
at f32 quantization scale — newt state is f32, widened to f64 for
the comparison; MuJoCo is f64 throughout. No physics disagreement.

### tumble (parity)

Torque-free rotation of an inertia-asymmetric brick over 4 s. Set up
for the intermediate-axis (Dzhanibekov) instability with initial
`ω_body = (0.1, 8.0, 0.1)`. Both engines flip on schedule; component
divergence tracks the different quaternion-renormalization schedules
each side uses (both hit RK4 mid-stage renorm; details of when the
free-root quaternion gets clamped-and-renormed differ).

### double_pendulum (parity)

Two-hinge chain on a fixed anchor. Horizon is deliberately short
(0.4 s) — beyond ~1 s the two trajectories diverge macroscopically
because the system IS chaotic. Sub-microsecond divergence over this
short window says the ABA and RK4 paths agree to machine precision
scaled by f32-vs-f64.

### floating_base (parity)

Free-root base with one hinge child under zero gravity, initial
`(0.5, -0.2, 0.1)` m/s linear velocity in the WORLD frame,
`(2.0, 0.5, -1.0)` rad/s body-frame angular velocity, and a
`3.0 rad/s` hinge rate. This is the coverage scenario for the
per-joint qpos/qvel remap between MuJoCo's `(qw, qx, qy, qz)` +
world-frame free-root linear velocity and newt's `(qx, qy, qz, qw)` +
body-frame free-root linear velocity slot. A wrong quat-slot order
or a missed frame rotation would drive divergence to the scale of the
initial velocities themselves (0.5 m/s+); the observed ~1e-6 says the
remap is empirically correct across all six free-root slots plus the
hinge scalar.

### double_pendulum_energy (parity, long-horizon)

Same 2-hinge chain as `double_pendulum` but sampled for 5.0 s. Per-
sample state divergence is not asserted (chaotic beyond ~1 s); the
assertion is that both engines conserve total mechanical energy over
the run and by roughly the same amount. MuJoCo's f64 RK4 conserves
energy to ~3e-8; newt's f32 RK4 with quaternion renormalization at
stage boundaries drifts to ~3e-6 (both are microscopic on the ~31 J
scale of the scene). The `gap` metric bounds the difference between
the two drifts, so a NEWT bug that inflated the drift dramatically
would fail even if MuJoCo's reference happened to move.

### servo_arm (parity)

Three-hinge arm driven by position servos to a fixed pose from rest.
`kv` values in the MJCF are precomputed from `dampratio=1` so both
loaders arrive at the same PD gains (documented in the fixture MJCF
and mirrored from `models/arm.xml`). The 3 cm/s peak `qvel`
divergence appears once, at the first sample (0.2 s in — during the
initial acceleration burst) and shrinks after; steady-state pose
error is well under a millimetre.

### sphere_drop, sphere_drop_stiff, sphere_drop_soft (bounded divergence)

Single sphere dropped from 0.6 m onto a plane over 3 s. The three
scenarios share every parameter except the solref timeconst
(`sphere_drop_stiff` at tc=0.010, `sphere_drop` default at tc=0.020,
`sphere_drop_soft` at tc=0.050) so together they span a 5x solref
sweep. Post-NEWT-14:

- **Steady-state penetration** matches MuJoCo to ~10 μm at the
  stiff and default points and to ~24 μm at the soft point (see
  the sweep table above). The v1 open finding — newt sitting
  ~180 μm shallower than MuJoCo — is closed here.
- **First-bounce transient** dominates the component-wise
  divergence measured over the full trajectory. Because the
  NEWT-14 split-α formula (see the NEWT-14 note below) reshapes
  the impulse profile at impact, the bounce trajectory shifts a
  few centimetres relative to MuJoCo's for one or two sample
  windows before both sides settle. The steady-state signal is
  the load-bearing one; the transient's residual is the noise
  ceiling of the discretization difference between the two
  engines' PGS pipelines. A dedicated test
  (`sphere_drop_steady_state_penetration_matches_mujoco` in
  `tests/differential.rs`) guards the steady-state gap
  independently of the transient tolerance.

Verdict is "bounded divergence" for all three scenarios: the
steady-state match is the parity signal, and the transient stays
bounded and predictable (no unbounded drift, no lost contact, no
sign flips).

### box_stack (parity — recovered)

Three cubes (bottom, middle, top) settling on a plane over 4 s.
Middle is offset by 2 cm in x to break perfect axial symmetry.
`friction=0.6` on all surfaces; `iterations=20` on the PGS solver
matching newt's default.

Post-NEWT-14 (Finding 1): the stack holds. Real MuJoCo parks the
top box at ≈ `(-6.7e-5, 2.4e-4, 1.7497)`; newt now settles at ≈
`(+1.1e-3, +8.6e-4, +1.7469)` — a 3 mm drop below MJ's final
position and a millimetre-scale horizontal drift. Component-wise
divergence peaks at 1.11 cm (transient) and 7.7 cm/s (transient);
the stack is stable through the full 4 s capture.

The v1 investigation hypothesized three possible causes (PGS
under-convergence, contact ordering, box-box narrow-phase
manifold). NEWT-14 instrumented both engines with per-step
contact dumps and found the root cause: **newt's box-box narrow
phase emits only 2 diagonal contact points for a tilted face-face
stack while MuJoCo emits 4 face-clipped points**. With only 2
contact points on the top-middle interface, the friction moment is
under-determined and the top box tips off. The fix routes the PGS
pipeline through a `narrow_phase_solver` that always runs SAT
face-clipping for box-box (`src/contact.rs::box_box_full_manifold`);
the penalty pipeline continues to use the vertex-vs-face primary
so its goldens stay byte-identical.

An iteration sweep (20 / 50 / 100 / 200 / 400) confirmed the
old collapse was NOT solver-convergence-limited — even 400 PGS
iterations still let the stack fall — so the "under-converging PGS"
hypothesis was wrong. The full evidence trace lives in this
ticket's PR body.

### velocity_cartpole (parity)

Cart on a slide + free-swinging pole; a `<velocity kv="15">` actuator
holds the cart at a constant 0.4 m/s target rate over 4 s while
gravity dumps the pole into random-ish swings that couple back into
the cart through Coriolis reaction. Purpose: validate the v2 tier 2
`<velocity>` semantics (gain=fixed(kv), bias=affine(0,0,-kv)) against
real MuJoCo end-to-end.

Both engines evaluate the actuator identically inside RK4, so
component divergence sits at f32-quant scale — 2.7e-7 qpos and
5.8e-7 qvel over the full trajectory. A semantic mismatch (wrong bias
sign, missed gear, ctrl-vs-vel swap) would blow divergence to
O(0.1 m or rad).

### filtered_motor_pendulum (bounded divergence)

Single-hinge pendulum driven by a `<general dyntype="filter">`
actuator with `tau=0.05 s`, `ctrl=0.7 N·m` from `t=0`. The activation
ODE low-passes the step so the applied torque ramps in over
~0.15 s. Purpose: validate the general actuator's filter dynamics
against real MuJoCo.

Newt integrates activation with **forward Euler** at each RK4 step
boundary (ZOH within the step — see `docs/actuators.md`). Real
MuJoCo, running under `integrator=RK4`, integrates activation through
the same RK4 stages as the mechanical state. The residual is bounded
and steady-state after the filter settles:

| metric | observed | bound |
|--------|----------|-------|
| qpos (rad) | 2.56e-4 | 6.0e-4 |
| qvel (rad/s) | 1.34e-3 | 3.0e-3 |

The gap scales with `dt/tau` and the ctrl-step amplitude; it is much
smaller than the bounded divergences on `sphere_drop_*` and
`joint_limit_swing`, and it does not accumulate over the horizon.

### joint_limit_swing (bounded divergence)

Single-hinge pendulum with a `±0.6 rad` range limit, initial
`qdot = 3.5 rad/s`, 2 s horizon. The pendulum drives into the limit
repeatedly. Newt's PGS limit path uses `JointLimit::DEFAULT` (solref
defaults, solimp defaults — MJCF does not expose per-limit
solref/solimp in newt's v1 subset and MuJoCo defaults are equivalent
scalar-wise). The per-impact impulse profile is not bit-identical
between the two solvers, and the residual velocity after each limit
collision accumulates as a phase drift over ~4 swings.

Magnitude: up to ~0.11 rad phase (about 6 degrees) and ~0.9 rad/s
peak velocity error, both concentrated in the moments immediately
after a limit hit. No divergence in position over the free-swing
half-cycles.

Verdict "bounded divergence" — the trajectories share qualitative
behaviour (both hit the limit, both bounce back, both eventually
lose energy at similar rates) but do not agree component-wise across
a limit event. A future ticket may want to expose solref-limit /
solimp-limit through the MJCF subset so both sides can be
constrained to exactly matching parameters.

## NEWT-14 fixes and remaining bounded divergences

Both v1 open findings are closed:

1. **box_stack instability** — CLOSED. Root cause: newt's box-box
   narrow phase emitted 2 diagonal contact points (vertex-vs-face)
   where MuJoCo emits 4 face-clipped points. Fix:
   `narrow_phase_solver` routes box-box through
   `box_box_full_manifold` (always SAT face-clipping). Penalty
   pipeline unchanged. See the `box_stack` note above.
2. **sphere_drop 179 μm steady-state offset** — CLOSED. Root cause:
   newt's PGS bias assembly scaled the reference acceleration by
   `d(r)` (impedance), producing a steady-state penetration of
   `r_ss = g(1−d) / (d²·k)` — about `2/d ≈ 2.1×` MuJoCo's. Fix:
   split-α scaling on the CONTACT-NORMAL reference term only:
   damping scalar `α_b = 1` (keeps the bias multiplier on
   `v_current` stable at large `dt·b`), stiffness scalar `α_k = 2`
   (empirical fit — makes the steady state `g(1−d) / (2·d·k)`,
   matching MuJoCo across the sweep to residuals of `alpha ∈
   {1.905, 1.914, 1.996, 2.006}`). Equality and joint-limit rows
   keep the pre-NEWT-14 impedance-scaled reference so their tests
   stay green. Constants in `src/solver.rs::CONTACT_AREF_ALPHA_*`.

Remaining bounded divergences (documented per-scenario above):

- **sphere_drop transient**: the split-α impulse reshape moves the
  bounce trajectory by a few cm for one or two sample windows
  before settling. The scorecard tolerance is chosen to survive
  the new transient shape; the steady-state signal is what the
  ticket contract asserts.
- **joint_limit_swing**: unchanged from v1 — joint-limit rows still
  use the pre-NEWT-14 reference formula, and MuJoCo's PGS limit
  impulse profile still differs slightly from newt's. ~0.11 rad
  peak phase drift; scorecard row and bound unchanged.

Both findings are documented in the NEWT-14 PR body.

### NEW OPEN FINDING: MuJoCo `k_impedance` functional form outside `tc ∈ [0.010, 0.050]`

The `α_k = 2` fit that closes the in-window sphere_drop finding
was measured on six solref timeconst points at dampratio = 1
(`tc ∈ {0.010, 0.015, 0.020, 0.030, 0.050, 0.100}`; the largest
was already borderline). Round-2 out-of-sample probes at
`tc ∈ {0.005, 0.070, 0.100}` show the fit does NOT extrapolate —
newt over-penetrates real MuJoCo by 0.5 to 1.2 mm outside
`[0.010, 0.050]`:

| tc     | mj_z        | newt_z      | mj_pen (μm) | newt_pen (μm) | gap (μm) |
|--------|-------------|-------------|-------------|---------------|----------|
| 0.005  | 0.099986    | 0.098745    | 14.3        | 1254.9        | 1240.6   |
| 0.070  | 0.098739    | 0.098216    | 1261.3      | 1784.5        | 523.1    |
| 0.100  | 0.097426    | 0.096381    | 2574.1      | 3618.8        | 1044.7   |

Iteration count was ruled out (bumped 20 → 200; no material
change at these tc). The pre-NEWT-14 code was uniformly worse
across the entire tc range (2/d ratio); the round-1 fix
correctly closes the fitted window while ALSO improving the
tc=0.100 case from 5476 μm to 3619 μm, but does not match
MuJoCo bit-for-bit outside `[0.010, 0.050]`.

Interpretation. The true MuJoCo `k_impedance` functional form
almost certainly is NOT the single-scalar `α_k = 2` shape that
happens to match in-window. Deriving it would need either
(a) reading the MuJoCo source's `mj_makeConstraint` /
`mj_softConstraint` directly to extract the actual `k` and
`b` expressions, or (b) fitting a more expressive form
(e.g., `α_k(tc, d)` with tc-dependent scaling, or a proper
midpoint-stabilization term) against a denser sweep.

**Scope for NEWT-14: report, do not paper over.** The scorecard
verdict for the in-window sweep is "parity (fitted window)"; the
out-of-window numbers are called out here and are NOT asserted
by any test (that would either force us to widen the tolerance
into meaninglessness or lie about the fit). A future ticket
should either extend the fit or replace `α_k = 2` with a
tc-dependent expression informed by MuJoCo's source.

## Regeneration

```bash
# From the newt/ directory. Requires the biped venv to have mujoco>=3.11.
python tools/capture_mujoco.py                       # regen ALL fixtures
python tools/capture_mujoco.py sphere_drop           # regen one scenario
python tools/capture_mujoco.py --force               # override mujoco-version guard
python tools/capture_mujoco.py --list                # print scenario names
# Matched Newton / Euler captures for the v3 solver rows:
python tools/capture_mujoco.py box_stack sphere_drop joint_limit_swing \
  --solver Newton --integrator Euler --suffix _newton_euler --force
```

## NEWT-21 Newton rows

The Newton rows use MuJoCo 3.11.0, `solver=Newton`, `integrator=Euler`,
`cone=pyramidal`, and 20 iterations. Newt loads the same MJCF, selects
`SolverMode::Newton`, and uses the matched Euler path.

| scenario | observed max qpos | bound | observed max qvel | bound |
|----------|------------------:|------:|------------------:|------:|
| sphere_drop | 7.220840e-3 | 2.0e-2 | 4.196461e-1 | 1.2 |
| box_stack | 9.693845e-3 | 5.0e-2 | 2.274836e-1 | 1.0 |
| joint_limit_swing | 9.167274e-2 | 2.0e-1 | 9.246982e-1 | 1.5 |

These bounds are measured from the committed `_newton_euler` fixtures.
The Newton path is also covered by stack and incline byte goldens.

## NEWT-22 tree-contact rows

Tree contacts now use the selected PGS or Newton row system. Penalty mode stays
on its old callback, so penalty goldens remain byte-identical. The new tree
anchors cover a free-root foot on a plane, a force-free contact before an
active contact, and a mixed tree/free-body pair.

The assisted biped uses `SolverMode::Newton`, `Integrator::Euler`, and the
source-faithful controller for 5,000 steps. No controller gains changed. The
measured result is:

| scenario | steps | distance | cadence | mean step | clearance | self-contact steps |
|---|---:|---:|---:|---:|---:|---:|
| assisted biped, Newton tree contacts | 5,000 | `2.5138 m` | `117.60 bpm` | `0.3251 m` | `0.2059 m` | `0` |

The 2,000-step solver smoke rows measured `1.2456 m` for Euler PGS and
`1.2491 m` for Euler Newton. The PGS and Newton tree anchor forces both
settled at `9.81 N` for a unit-mass free-root sphere.

The symmetry-broken tree PGS/Newton anchor agrees within `8.94e-8` in q and
`9.54e-7` in qdot over 100 steps. The test prints these maxima and keeps
`1.0e-6` and `2.0e-6` bounds.

The matched articulated-chain fixtures use MuJoCo 3.11.0, Euler, pyramidal
cones, and 20 iterations. They cover the first ground-contact transition:

| scenario | observed max qpos | bound | observed max qvel | bound |
|---|---:|---:|---:|---:|
| falling chain, PGS | `6.396024e-2` | `7.0e-2` | `6.105601e-1` | `7.0e-1` |
| falling chain, Newton | `7.089735e-2` | `8.0e-2` | `1.127755e0` | `1.3` |
| assisted biped, Newton | `3.546789e-1` | `4.0e-1` | `2.552028e0` | `3.0` |

The chain fixtures are `tree_chain_contact_pgs_euler.bin` and
`tree_chain_contact_newton_euler.bin`. Recapture them with the matching
`tools/capture_mujoco.py` commands. The assisted biped fixture is
`biped_assisted_walk_newton_euler.bin`; recapture it with
`tools/capture_biped_mujoco.py`. Its 5,000-step run uses the source-faithful
controller, `assist_scale=0.8`, and no debug state correction.

## Debugging a failing tolerance

Set `NEWT_DIFFERENTIAL_DUMP=1` when running the tests to print the
per-sample per-component error tape (qpos and qvel). Useful when a
tolerance needs to be understood or re-set after a fixture regen:

```bash
NEWT_DIFFERENTIAL_DUMP=1 cargo test --test differential -- --nocapture
NEWT_DIFFERENTIAL_DUMP=1 cargo test --test differential sphere_drop -- --nocapture
```

The dump goes to stderr; combine with `--nocapture` to see it.

If the MuJoCo version in the venv differs from the version recorded
in any existing fixture's provenance header, the tool refuses to
overwrite. Add `--force` to overwrite, and mirror the moved numbers
into this scorecard's "Observed max" column and into the TOLERANCES
table in `tests/differential.rs`. Never loosen a bound without
recording WHY it moved and updating the verdict.

## Adding a new scenario

1. Author the shared MJCF under `tests/references/<name>.xml`. Set
   `<option>` values (`timestep`, `integrator`, `solver`, `cone`,
   `iterations`, `gravity`) EXPLICITLY so both engines read the same
   numbers.
2. Add an entry to `tests/references/scenarios.json`. `init_qpos`
   and `init_qvel` are optional overrides in MuJoCo layout;
   `actuator_targets` is a name→ctrl map. Set `"check_kind":
   "energy"` for a long-horizon energy-drift scenario instead of the
   default per-sample state comparison.
3. Run `python tools/capture_mujoco.py <name>` to write the fixture
   (and, for energy scenarios, the `<name>_energy.bin` sidecar). Set
   `"compare_sensors": true` to also write `<name>_sensors.bin`.
4. Add a `differential_<name>` test in `tests/differential.rs` and
   a `Tolerance` (or `EnergyTolerance` for energy scenarios) entry
   in the match table. Set the tolerance from the observed max × ~2
   for a clean scenario, or a wider ratio with a note if the scenario
   has a bounded divergence. Add a scorecard row here.
