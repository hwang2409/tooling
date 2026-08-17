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

NEWT-26 targets the Euler PGS/Newton configuration. Those paths now detect
contacts once at step start and solve that current set. Penalty RK4 keeps live
per-stage collision evaluation. RK4 PGS/Newton keeps constraint zero-order
hold; that residual is separate from Euler phase ordering.

The rule (from the arc's integrity standard, verbatim in NEWT-13's
contract): a divergence beyond physical reasonableness is a FINDING
to report, not to hide. Bounds are set from measurement plus ~2×
headroom; open findings get their own section below and are called
out at the top of the PR body.

## How the harness works

- **Reference** — `tools/capture_mujoco.py` loads each scenario's
  MJCF under the biped-venv MuJoCo, configures matching timestep,
  solver, cone, and iteration settings, then writes a compact binary
  fixture. The `--row-diagnostics`, `--solref-sweep`, and `--rk4-sweep`
  modes write committed JSON evidence for factor and integrator checks.
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
| box_stack          | 1.459283e-2       | 5.0e-2     | 1.882446e-1       | 5.0e-1     | bounded divergence; plane-manifold residual |
| joint_limit_swing  | 6.56e-2           | 1.5e-1     | 3.09e-1           | 1.2e+0     | bounded divergence |
| velocity_cartpole  | 2.70e-7           | 1.0e-6     | 5.77e-7           | 2.0e-6     | parity             |
| filtered_motor_pendulum | 2.56e-4      | 6.0e-4     | 1.34e-3           | 3.0e-3     | bounded divergence |
| tendon_cylinder_lift | 8.83e-8 | 2.0e-7 | 4.58e-7 | 1.0e-6 | parity; matched Euler PGS |
| tendon_pulley_2to1 | 1.95e-7 | 4.0e-7 | 1.18e-6 | 3.0e-6 | parity; matched Euler PGS |
| tendon_mixed_wrap | 1.06e-7 | 3.0e-7 | 5.04e-7 | 1.0e-6 | parity; matched Euler PGS |
| mocap_rangefinder | 0                  | 1.0e-6     | 0                  | 1.0e-6     | parity; sensors 2.4e-8 |
| hfield_sphere_ramp | 3.211479e-1       | 5.0e-1     | 1.101810     | 2.0        | bounded hfield terrain divergence |
| hfield_box_terrain | 1.906255           | 2.5        | 43.93162     | 55.0       | bounded multi-cell terrain divergence |
| hfield_capsule_waves | 4.643577        | 5.5        | 90.03303     | 120.0      | bounded wavy-terrain divergence |

### NEWT-27 spatial tendon probes

The cylinder and pulley extension has committed hand and finite-difference
probes in `tests/tendon_wrap_cylinder.rs`. The probes cover tangent length,
shortest-side selection, divisor scaling, and the analytic Jacobian.

Three matched MuJoCo 3.11.0 Euler/PGS captures are committed in
`tests/references/`: `tendon_cylinder_lift`, `tendon_pulley_2to1`, and
`tendon_mixed_wrap`. The measured maxima and asserted bounds are in the
scorecard above. Each scenario has a permanent differential test.

The finite-difference probes report maximum relative Jacobian errors of
`2.883e-3` for the cylinder wrap and `1.195e-3` for the pulley branches.

### NEWT-28 heightfield captures

MuJoCo 3.11.0 Euler/PGS captures are committed in
`tests/references/hfield_*.bin`, with matching XML provenance. The three
scenes cover a sphere on a ramp, a box on uneven terrain, and a capsule on
waves. The measured maxima are listed above. The wide bounds are explicit:
the current prism-top candidate model keeps the contact set bounded, but it
does not claim trajectory parity for rolling and multi-cell transitions yet.
The fixtures remain required regression evidence until those residuals close.
The matching Newton/Euler captures measure qpos/qvel maxima of
`(0.650158, 5.043927)`, `(1.871275, 63.14738)`, and `(2.437249,
109.1951)` for sphere, box, and capsule respectively. Their asserted bounds
are `(1.0, 8.0)`, `(2.5, 75.0)`, and `(3.0, 130.0)`.

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

### box_stack (bounded divergence; finding reopened)

Three cubes (bottom, middle, top) settling on a plane over 4 s.
Middle is offset by 2 cm in x to break perfect axial symmetry.
`friction=0.6` on all surfaces; `iterations=20` on the PGS solver
matching newt's default.

The stack remains bounded, but the exact MuJoCo box-plane manifold reopens
the differential residual. The current PGS capture measures `1.459283e-2 m`
qpos and `1.882446e-1 m/s` qvel. The matched Euler capture measures
`8.167121e-3 m` and `2.795157e-1 m/s`. The Newton/Euler capture measures
`1.729087e-2 m` and `1.179916e-1 m/s`.

The first large PGS qvel residual appears at sample `2`, component `15`; the
maximum qpos residual appears at sample `20`, component `14`. These are top-box
state slots. The exact plane rule changes the contact scan order, midpoint
point, and penetration anchor for box-plane contacts. That changes the
finite-step normal and friction moment during settling. The box-box clipping
fix remains in place; this is a separate plane-contact effect.

The stack does not collapse. The scorecard now reports this as bounded
divergence, not recovered parity.

The Euler/Newton remeasurement is `1.729087e-2 m` qpos and
`1.179916e-1 m/s` qvel. The phase change did not tighten this hypothesis.
The biped 0.4 sweep also remains an Euler/Newton result and did not close its
fall-step gap.

The v1 investigation hypothesized three possible causes (PGS
under-convergence, contact ordering, box-box narrow-phase
manifold). NEWT-14 instrumented both engines with per-step
contact dumps and found the box-box root cause: **newt's box-box narrow
phase emits only 2 diagonal contact points for a tilted face-face
stack while MuJoCo emits 4 face-clipped points**. With only 2
contact points on the top-middle interface, the friction moment is
under-determined and the top box tips off. The fix routes the PGS
pipeline through a `narrow_phase_solver` that always runs SAT
face-clipping for box-box (`src/contact.rs::box_box_full_manifold`);
the penalty pipeline continues to use the vertex-vs-face primary for box-box.
NEWT-25 still changes penalty stack goldens through the shared plane manifold.

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
state for six sphere rows and one tree row. The committed fixture is
`tests/references/constraint_factor_diagnostics.json`, captured with
MuJoCo 3.11.0. The control row is `tc=0.020, dr=1`. Newt uses the same
`qpos` and `qvel` in its row assembly. Facet rows use MuJoCo's
`N ± μT` ordering.

The maximum absolute factor deltas across the six sphere rows are:

| tc / dr | pos | vel | k_eff | b | imp | R | aref |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.005 / 0.5 | 2.399e-9 | 1.666e-7 | 2.547e-2 | 1.122e-5 | 1.190e-8 | 6.385e-8 | 4.965e-4 |
| 0.010 / 1 | 2.399e-9 | 7.779e-8 | 1.591e-3 | 5.611e-6 | 1.190e-8 | 6.385e-8 | 7.706e-5 |
| 0.020 / 1 | 2.399e-9 | 1.819e-7 | 3.986e-4 | 2.805e-6 | 1.190e-8 | 6.385e-8 | 1.824e-5 |
| 0.050 / 1 | 2.399e-9 | 2.076e-7 | 1.928e-5 | 3.979e-7 | 1.190e-8 | 6.385e-8 | 1.716e-5 |
| 0.100 / 1 | 2.399e-9 | 1.475e-7 | 4.795e-6 | 1.989e-7 | 1.190e-8 | 6.385e-8 | 6.292e-6 |
| 0.200 / 2 | 2.399e-9 | 1.450e-7 | 3.014e-7 | 9.947e-8 | 1.190e-8 | 6.385e-8 | 4.259e-6 |

The permanent Rust test checks `pos`, `vel`, `k_eff`, `b`, `imp`, `R`,
and `aref` against these committed MuJoCo factors.

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

The tree record also stores `efc_diagA`, `efc_R`, and `efc_KBIP` for all
four facets. The permanent test checks the source relation
`Rpy = 2*mu²*Rnormal`. Newt now computes tree regularization from the
source body-level `diagApprox`, then applies the shared pyramid rule.

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
| 0.200 / 2 | no stable equilibrium | no stable equilibrium | 2652.394 final-z μm |

The `0.200 / 2` row has no stable contact equilibrium. Both systems pass
through the plane under the very soft reference, so its final position is
not a penetration-parity signal. Its first-step force is included above.

The committed `solref_sweep_rk4.json` fixture keeps separate RK4 evidence
for the two out-of-window rows. The measured final-z gaps are `59.977 μm`
for `0.100 / 1` and `3031.035 μm` for `0.200 / 2`. The Rust test uses
separate bounds of `100 μm` and `4000 μm`. These rows remain honest
integration-semantics rows, not impedance-form evidence.

Verdict: the impedance finding is CLOSED. RK4 rows remain honest rows with
measured bounds. Their residual is the documented constraint-ZOH versus
per-stage reevaluation gap; see `docs/integrators.md`.

Earlier row-bias work changed the solver and equality goldens. NEWT-25 also
changes the six PGS/Newton solver goldens, the penalty stack goldens, and the
packaged `model_stack.bin`: all use plane contacts. This is an expected
behavior change from the exact MuJoCo manifold, not a hidden solver tweak.

## Regeneration

```bash
# From the newt/ directory. Requires the biped venv to have mujoco>=3.11.
python tools/capture_mujoco.py                       # regen ALL fixtures
python tools/capture_mujoco.py sphere_drop           # regen one scenario
python tools/capture_mujoco.py --force               # override mujoco-version guard
python tools/capture_mujoco.py --list                # print scenario names
python tools/capture_mujoco.py --row-diagnostics /tmp/newt-23-row-diagnostics.json
python tools/capture_mujoco.py --solref-sweep /tmp/newt-23-euler-sweep.json
python tools/capture_mujoco.py --rk4-sweep /tmp/newt-23-rk4-sweep.json
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
| box_stack | 1.729087e-2 | 5.0e-2 | 1.179916e-1 | 1.0 |
| joint_limit_swing | 9.167274e-2 | 2.0e-1 | 9.246982e-1 | 1.5 |

These bounds are measured from the committed `_newton_euler` fixtures.
The Newton path is also covered by stack and incline byte goldens.

The exact source `k` changes the finite-iteration PGS/Newton residual in
the condim-4 torsional anchor. The measured maxima are `8.931686729e-2`
qpos, `3.880491853e-1` qvel, and `3.008949041` rad/s spin. A 10x PGS
probe (`300` versus the default `30` iterations) produced the same values.
The gap is not PGS iteration starvation. The test keeps bounds of `0.1`,
`0.5`, and `3.2`, with this error budget disclosed here and in the test.

## NEWT-22 tree-contact rows

Tree contacts now use the selected PGS or Newton row system. Penalty mode stays
on its old callback, but shared plane-manifold changes can move penalty
goldens. The new tree
anchors cover a free-root foot on a plane, a force-free contact before an
active contact, and a mixed tree/free-body pair.

The assisted biped uses `SolverMode::Newton`, `Integrator::Euler`, and the
source-faithful controller for 5,000 steps. No controller gains changed. The
measured result is:

| scenario | steps | distance | cadence | mean step | clearance | self-contact steps |
|---|---:|---:|---:|---:|---:|---:|
| assisted biped, Newton tree contacts | 5,000 | `2.523389 m` | `117.60 bpm` | `0.372897 m` | `0.200563 m` | `0` |

These values are a disclosed re-measurement after the exact-bias and shared
tree-regularization changes. CI records bands of `2.50..2.55 m` distance,
`116..119 bpm` cadence, `0.36..0.39 m` mean step length, and
`0.19..0.21 m` clearance. A future shift outside a band fails the walk test.

The 2,000-step solver smoke rows measured `1.2456 m` for Euler PGS and
`1.2491 m` for Euler Newton. The PGS and Newton tree anchor forces both
settled at `9.81 N` for a unit-mass free-root sphere.

The shared `diagApprox` and pyramid `Rpy` construction changes the
symmetry-broken tree PGS/Newton residual. The measured maxima are
`1.365253e-2` in q and `4.467820e-1` in qdot over 100 steps. A 10x PGS
probe (`400` versus the default `40` iterations) produced `1.364952e-2`
and `4.467788e-1`. This also rules out iteration starvation. The test
keeps measured-plus-headroom bounds of `2.0e-2` and `5.0e-1`.

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

## NEWT-25 biped v3 sweep and solver-phase finding

The biped acceptance oracle uses MuJoCo `3.11.0`, Euler, Newton, pyramidal
cones, 20 iterations, and `dt=0.005`. The source controller stays unchanged.
The four binary fixtures are `biped_walk_oracle_v3_assist_080.bin`,
`_040.bin`, `_020.bin`, and `_000.bin`. They contain one qpos/qvel checkpoint
per step and the measured gait metrics.

The normal test runs a 120-step representative sweep at assist `0.8` and
`0.0`. The ignored full sweep compares each source trace with newt through the
newt fall step. These are current-gap regression bounds, not parity claims.
Run the full sweep with:

```text
cargo test --manifest-path newt/Cargo.toml --test biped_walk_acceptance v3_full_sweep -- --ignored --nocapture
```

| assist | source outcome | newt outcome | source fall | newt fall | compared steps | max qpos gap | max qvel gap |
|---:|---|---|---:|---:|---:|---:|---:|
| `0.8` | complete | complete | — | — | `5,000` | `3.54552e-1` | `4.66813e0` |
| `0.4` | fallen | fallen | `756` | `551` | `551` | `1.15457e0` | `6.60413e0` |
| `0.2` | fallen | fallen | `492` | `472` | `472` | `9.07494e-1` | `6.89277e0` |
| `0.0` | fallen | fallen | `578` | `439` | `439` | `1.16138e0` | `6.44864e0` |

The outcome class matches at all four levels. The `0.4` row remains the
acceptance failure: the fall-step gap is `756 - 551 = 205`, versus `203`
before the manifold update. The no-assist oracle also falls, so stable
no-assist walking is not a valid target for this source controller.

status: the isolated manifold finding is closed. Euler PGS/Newton now use one
current-position contact pass for free-body and tree rows. Visual support masks
and solver contact masks are stored separately. The fresh parsed structural
window is `[25,35]`: onset-boundary drift changes the live contact set by the
time the engines cross the foot threshold.

At solver step `25`, source and newt constraint-force maxima are `112.489` and
`0.000`; row reference-acceleration maxima are `124.015` and `0.000`. At step
`36`, the maxima are `503.445` and `574.786`; row reference-acceleration
maxima are `155.270` and `152.956`. The first solver-phase qvel residual
bound exceeds at step `26` (`1.636116`). The per-step trail in
`docs/biped-walk.md` is the next ticket's comparison specification.

The complete measured solver structural mismatch set through step `36` is
`[25,35]`. The acceptance test asserts this set verbatim. A state-injection
probe places MuJoCo's exact step-25 and step-35 qpos into newt before
step-start collision. It matches contact count, geom pair, row mapping, and
contact depth within `1.933e-6`; position gap is at most `2.716e-4 m`.
This closes the phase finding. The remaining live mismatch is trajectory drift
at a contact-onset boundary, not prior-step contact latency. The source-side
contact and row records are in
`tests/references/biped_walk_v3_diagnostics.json`. Records cover step `0`
through `40`, including solver-phase and post-step records. The source
capture records solver-phase state before `mj_step` and keeps post-step
`mj_forward` geometry separate. The selected newt records are in
`tests/references/biped_walk_v3_newt_diagnostics.json`; they are solver-phase
records for steps `0` through `40`. Each parsed contact stores the geom pair,
point, depth, and row mapping needed by the parity comparison.
The newt diagnostic runner is `examples/capture_biped_diagnostics`; it writes
the parsed fixture. The human diagnostic runner is
`examples/biped_walk_diagnostics`; it prints
steps `0..12`, `17`, `25`, and `36` with the same geom-level fields. It calls
`solver::solve_tree_contacts`, the NEWT-23 row assembly path.

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
