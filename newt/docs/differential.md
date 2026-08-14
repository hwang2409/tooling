# Differential parity: newt vs real MuJoCo

Status: NEWT-13, v1 tier 8 closer. Capture tool at
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
| sphere_drop        | 4.10e-3           | 1.0e-2     | 1.67e-1           | 3.0e-1     | bounded divergence |
| box_stack          | 1.45e+0           | 2.0e+0     | 3.89e+0           | 5.0e+0     | OPEN FINDING       |
| joint_limit_swing  | 1.08e-1           | 1.5e-1     | 9.10e-1           | 1.2e+0     | bounded divergence |

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

### sphere_drop (bounded divergence)

Single sphere dropped from 0.6 m onto a plane over 3 s. Two things
diverge:

- **First-bounce transient (t ≈ 0.2 s)** — newt is slightly higher
  and slower than MuJoCo during the first ~30 ms of contact, by up
  to 4.1 mm in z position and 0.17 m/s in vertical velocity. Both
  sides recover.
- **Steady-state penetration** — after settlement (t ≥ 0.5 s), newt
  parks the sphere centre at z ≈ 0.09996 m (bottom of sphere ≈ 37 μm
  below the plane); MuJoCo settles at z ≈ 0.09978 m (bottom of
  sphere ≈ 216 μm below the plane). That is a persistent 179 μm
  disagreement.

This is the specific NEWT-9-related finding the arc predicted a home
for. Newt's PGS uses a softened d-scaling of the effective Baumgarte
coefficient (see `src/solver.rs` docstring on `SolImp`) that
suppresses steady-state penetration compared with real MuJoCo's
default profile. The direction is CONSISTENT with the softened-d
claim (newt penetrates less because our effective d saturates
sooner); the magnitude (~180 μm) is the number to remember.

Verdict is "bounded divergence" — no engine bug, but the
solref/solimp semantics do not match MuJoCo bit-for-bit and a future
ticket may want to bring them in line. Not in scope for NEWT-13.

### box_stack (OPEN FINDING)

Three cubes (bottom, middle, top) settling on a plane over 4 s.
Middle is offset by 2 cm in x to break perfect axial symmetry.
`friction=0.6` on all surfaces; `iterations=20` on the PGS solver
matching newt's default.

**Finding: the top block does not stay stacked in newt — the stack
fully collapses to the ground.** Real MuJoCo parks the top box at
position ≈ `(-6.7e-5, 2.4e-4, 1.7497)` — a clean stack at rest.
Newt lets the top block slide off (mostly in −x, some +y) and it
drops all the way to the ground; by t = 4 s it is at rest at
≈ `(-1.41, +0.44, +0.35)` — z = 0.35 is the on-ground rest height
of a 0.35 m half-extent cube on top of the plane. Divergence peaks
at t ≈ 3.0 s (top box at `(-1.45, +0.44, +0.39)`; qvel component
peak 3.89 m/s at t ≈ 2.6 s) then converges as the block settles on
the ground away from the stack. Failure mode is fully-collapsed-
stack vs stable-stack, NOT a small tip or phase drift — the
follow-up investigator should hunt a full collapse, not a partial
lean.

Cause hypotheses (not verified in this ticket; that is follow-up
work):

- The 20-iteration PGS solve may be under-converging for the
  three-body-frictional-cone contact structure newt sees, letting
  small tangential impulses accumulate; MuJoCo's implementation
  may add a slip-projection pass or a warm-start that we do not.
- Our contact-ordering totalization is `(min, max)` lexicographic
  on geom indices, which for stacked boxes ends up ordering the
  bottom-of-middle-vs-top-of-bottom contact vs the bottom-of-top-
  vs-top-of-middle contact in an order that biases friction the
  wrong way; MuJoCo's order comes out of its collision builder
  and may differ.
- The box-box narrow phase (added in NEWT-7 as part of the
  MuJoCo-parity contact set) uses SAT-based clipping; MuJoCo uses
  MPR by default and the contact-point locations may differ enough
  to matter for a friction-critical stack.

The test tolerance is set to just above the observed max so the
scenario STILL asserts and any FURTHER regression fails CI. The
scorecard row will move as this is investigated in a follow-up
ticket.

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

## Open findings (rolled up)

1. **box_stack instability** — newt's three-box stack fully
   collapses at the current default settings while real MuJoCo's
   holds. The top block slides off (mostly −x) and lands on the
   ground; end state is ≈ `(-1.41, +0.44, +0.35)` vs MuJoCo's
   `(-6.7e-5, 2.4e-4, 1.7497)`. Peak divergence at t ≈ 3.0 s. Not a
   NEWT-13 fix; filed as an open finding for a follow-up ticket to
   investigate contact ordering, iteration count, and box-box
   narrow-phase contact-point placement.
2. **sphere_drop steady-state penetration** — newt sits ~180 μm
   shallower than MuJoCo under identical `solref`/`solimp`/
   `iterations`. Consistent with the softened-d-scaling claim in
   `src/solver.rs`. May warrant a follow-up to bring the effective
   Baumgarte coefficient into bit-agreement with MuJoCo.

Both findings are documented in the PR body as required by the
NEWT-13 contract.

## Regeneration

```bash
# From the newt/ directory. Requires the biped venv to have mujoco>=3.11.
python tools/capture_mujoco.py                       # regen ALL fixtures
python tools/capture_mujoco.py sphere_drop           # regen one scenario
python tools/capture_mujoco.py --force               # override mujoco-version guard
python tools/capture_mujoco.py --list                # print scenario names
```

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
   (and, for energy scenarios, the `<name>_energy.bin` sidecar).
4. Add a `differential_<name>` test in `tests/differential.rs` and
   a `Tolerance` (or `EnergyTolerance` for energy scenarios) entry
   in the match table. Set the tolerance from the observed max × ~2
   for a clean scenario, or a wider ratio with a note if the scenario
   has a bounded divergence. Add a scorecard row here.
