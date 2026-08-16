# Differential parity: newt vs real MuJoCo

Status: NEWT-23 (v2 tier 1). The exact solref reference-row finding
is closed. Capture tool at
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
  MJCF under the biped-venv MuJoCo, configures matching timestep,
  solver, cone, and iteration settings, then writes a compact binary
  fixture. The `--row-diagnostics` and `--solref-sweep` modes write JSON
  evidence for factor and matched-integrator checks.
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
| sphere_drop_stiff  | 2.97e-4           | 8.0e-2     | 2.00e-2           | 8.0e-1     | bounded RK4 residual |
| sphere_drop        | 5.67e-4           | 1.0e-2     | 2.05e-2           | 1.0e+0     | bounded RK4 residual |
| sphere_drop_soft   | 1.38e-3           | 7.0e-2     | 9.68e-3           | 8.0e-1     | bounded RK4 residual |
| box_stack          | 5.14e-3           | 5.0e-2     | 1.23e-1           | 5.0e-1     | parity (recovered) |
| joint_limit_swing  | 6.56e-2           | 1.5e-1     | 3.09e-1           | 1.2e+0     | bounded divergence |
| velocity_cartpole  | 2.70e-7           | 1.0e-6     | 5.77e-7           | 2.0e-6     | parity             |
| filtered_motor_pendulum | 2.56e-4      | 6.0e-4     | 1.34e-3           | 3.0e-3     | bounded divergence |
| mocap_rangefinder | 0                  | 1.0e-6     | 0                  | 1.0e-6     | parity; sensors 2.4e-8 |

`mocap_rangefinder` also compares all seven `sensordata` values at each
sample. The maximum direct sensor error is `2.4e-8`, below the `2.0e-6`
sensor bound. It covers rangefinder ray casting, mocap site attachment,
velocimeter output, and magnetic-field frame conversion.

### sphere_drop solref sweep

The old fit is removed. Under RK4, the remaining transient is an
integration-semantics residual: newt holds solver forces across the RK4
step, while MuJoCo reevaluates constraints at each stage. Soft contacts
show the largest gap because their force changes more within one step.
Matched Euler evidence is in the NEWT-23 section below.

Verdict is "bounded integration residual". It is not an impedance-form
finding.

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

### sphere_drop, sphere_drop_stiff, sphere_drop_soft (bounded RK4 residual)

Single sphere dropped from 0.6 m onto a plane over 3 s. The three
scenarios share every parameter except the solref timeconst
(`sphere_drop_stiff` at tc=0.010, `sphere_drop` default at tc=0.020,
`sphere_drop_soft` at tc=0.050) so together they span a 5x solref
The exact reference-row form is source-derived. The remaining RK4 gap is
an integration-semantics residual: newt holds solver forces across the RK4
step, while MuJoCo reevaluates constraints at each stage. Soft contacts
show the largest gap because their force changes more within one step.

Verdict is "bounded integration residual". It is not an impedance-form
finding. Matched Euler evidence is in the NEWT-23 section below.

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

## NEWT-23 exact reference rows

The old `α_k = 2` contact fit is deleted. Newt now follows MuJoCo's
source-derived `mj_makeImpedance` and `mj_referenceConstraint` equations.
The path is shared by contacts, equalities, limits, and both solver modes.

### first-contact factor parity

`tools/capture_mujoco.py --row-diagnostics` captures the first contact
state for six rows. The control row is `tc=0.020, dr=1`. Newt uses the
same `qpos` and `qvel` in its row assembly. Facet rows are compared after
the MuJoCo pyramid ordering is applied.

| factor | result | evidence |
|---|---|---|
| `efc_pos` | match | signed distance `-0.002272`; margin `0` |
| `efc_vel` / `efc_J` | match | facet rows use `N ± μT`; max velocity delta `1.3e-5` |
| `k`, `b` | match | standard and `tc=0.005` clamp use source formulas |
| `efc_R` / `efc_D` | match | `R=0.2209684211`; `D=4.5255335366` |
| `efc_aref` | match | max delta `5.0e-3 m/s²` from f32 state input |

The `tc=0.005` row uses MuJoCo's `timeconst >= 2*dt` clamp. The impedance
clamps are also reproduced: `dmin`, `dmax`, and midpoint stay in
`[1e-4, 0.9999]`, width is non-negative, and power is at least one.

The same capture includes `qacc` and `qfrc_constraint`. The matched first
step has these normal-force comparisons:

| tc / dr | MuJoCo `qfrc_constraint[z]` | newt | delta |
|---|---:|---:|---:|
| 0.005 / 0.5 | 720.169 | 720.611 | 0.442 |
| 0.020 / 1 | 162.712 | 162.812 | 0.100 |
| 0.100 / 1 | 36.793 | 36.816 | 0.023 |
| 0.200 / 2 | 20.787 | 20.800 | 0.013 |

These small deltas come from the finite PGS sweep and f32 state. They are
not factor mismatches. The old hundreds-of-m/s² `aref` error is gone.

### matched Euler sweep

MuJoCo and newt both use Euler, PGS, pyramidal cones, `dt=0.002`, and
20 iterations. The seven-point sweep uses 1,500 steps. Stable-contact rows
now match at the same tier across and outside the old fitted window:

| tc / dr | MuJoCo penetration (μm) | newt (μm) | gap (μm) |
|---|---:|---:|---:|
| 0.005 / 0.5 | 3.585 | 3.582 | 0.003 |
| 0.010 / 1 | 57.133 | 57.144 | 0.011 |
| 0.020 / 1 | 216.440 | 216.408 | 0.032 |
| 0.050 / 1 | 740.598 | 735.468 | 5.130 |
| 0.070 / 1 | 1261.329 | 1232.369 | 28.960 |
| 0.100 / 1 | 2574.144 | 2514.167 | 59.977 |

The `0.200 / 2` row has no stable contact equilibrium. Both systems pass
through the plane under the very soft reference, so its final position is
not a penetration-parity signal. Its first-step force is included above.

Verdict: the impedance finding is CLOSED. RK4 rows remain honest rows with
measured bounds. Their residual is the documented constraint-ZOH versus
per-stage reevaluation gap; see `docs/integrators.md`.

The solver and equality goldens changed because their shared row bias now
uses the source acceleration equation. The affected files are the six
equality/condim goldens and the six PGS/Newton solver goldens. Penalty-mode
goldens remain byte-identical.

## Regeneration

```bash
# From the newt/ directory. Requires the biped venv to have mujoco>=3.11.
python tools/capture_mujoco.py                       # regen ALL fixtures
python tools/capture_mujoco.py sphere_drop           # regen one scenario
python tools/capture_mujoco.py --force               # override mujoco-version guard
python tools/capture_mujoco.py --list                # print scenario names
python tools/capture_mujoco.py --row-diagnostics /tmp/newt-23-row-diagnostics.json
python tools/capture_mujoco.py --solref-sweep /tmp/newt-23-euler-sweep.json
python tools/capture_mujoco.py sphere_drop sphere_drop_stiff sphere_drop_soft \
  --integrator Euler --suffix _euler --force
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
