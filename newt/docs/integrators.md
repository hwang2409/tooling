# integrators

newt exposes three fixed-step integrators through `World::integrator`.
`Rk4` is the default. Existing scenes keep this value and keep their old
trajectory.

| integrator | state sample | force and constraint sample | position update |
| --- | --- | --- | --- |
| `Rk4` | four interpolated stages | penalty contacts at every stage; PGS/Newton rows once at step start and held constant | weighted stage sum |
| `Euler` | start of step | forward kinematics, collision, and PGS/Newton rows once at the current state | `qvel += dt*qacc`; integrate `qpos` from new `qvel` |
| `ImplicitFast` | start of step | same one-solve pipeline as Euler | same velocity-first update, with joint and joint-actuator velocity terms folded into the solve |

## euler pipeline

For `Euler`, one world step runs in this order:

1. compute forward kinematics for trees and geom poses;
2. detect the current contact set;
3. solve the current PGS constraints, when PGS is enabled;
4. evaluate forward dynamics at the current `q` and `qvel`;
5. update generalized velocity;
6. integrate generalized position from that new velocity;
7. evaluate sensors on the post-step state.

Free bodies use the same ordering. Their updated world linear velocity drives
the position update. Their updated body angular velocity drives a right-
multiplied quaternion exponential map, so the orientation stays on the unit
quaternion manifold.

Contacts, limits, and equality rows are fresh for every Euler step. The RK4
constraint ZOH is not used by Euler.

## tree-contact impulse entry

For PGS and Newton, the world assembles tree-involved contact rows before it
advances either pool. A tree link point Jacobian maps the contact direction to
joint-space `qfrc`. The equal-and-opposite body or tree participant receives
the same impulse in the shared solve.

RK4 adds the recovered tree `qfrc` to `qfrc_applied` for the step. The value
stays constant through all four ABA stages, then the temporary addition is
removed. The tree penalty contact callback is disabled for that step, so a
solver contact cannot be applied twice.

Euler and implicitfast add the same recovered `qfrc` before their one ABA
call. Their free-velocity term uses the matching tree ABA mode: explicit ABA
for RK4, implicit joint damping for Euler, and implicit joint plus
joint-transmitted actuator velocity terms for implicitfast.

Penalty mode does not enter this path. It keeps the old per-stage RK4 and
single-stage Euler contact wrenches.

## implicit joint damping

MuJoCo's Euler path treats joint damping implicitly. Let `M` be the
configuration-space mass matrix, `B` the diagonal joint damping matrix, and
`r` all other generalized forces. The damping force uses the new velocity:

```text
M qacc = r - B qvel_new
qvel_new = qvel + dt qacc
```

Substitution gives the solve used by newt:

```text
(M + dt B) qacc = r - B qvel
```

The articulated-body implementation applies this without building a dense
matrix. For a scalar hinge or slide, it adds `dt*damping` to
`Sᵀ IA S + armature`. For a ball joint, it adds the same value to each
diagonal of its three-by-three joint block. The right-hand side still uses
`-damping*qvel` from the start of the step.

A free root uses the same scalar damping coefficient on all six velocity
slots. Its root `6×6` articulated inertia gets `dt*damping` on every diagonal,
and its right-hand side gets `-damping*qvel`. JSON and MJCF root joints accept
that scalar as `damping`.

This fold prevents the explicit factor `1 - dt*damping/I` from becoming
unstable at the biped timestep. The permanent stability anchor uses
`dt=0.005` and `damping=1000`: the explicit probe grows by a factor of four
per step, while shipped Euler decays by the implicit factor `1/6`.

## implicitfast scope

`ImplicitFast` uses the same solve and adds the negative velocity derivative
of joint-transmitted actuator force. For a force law `tau(q, qvel)`, newt adds

```text
B_act = -d tau / d qvel
```

to the diagonal fold. This covers the `kv` term in position and velocity
actuators and the velocity terms in a general actuator's affine gain/bias.
It includes actuator gear factors in transmission space.

This ticket does not include Coriolis derivatives. It also does not
differentiate force clamps. Tendon-actuator velocity derivatives are also
excluded: their correct fold is the dense, non-diagonal `kv*Jᵀ*J` term. This
ticket evaluates those tendon forces explicitly; the full coupled fold arrives
with the Newton ticket's dense machinery. That is the known delta from full
MuJoCo `implicitfast`; the supported implicit scope is joint damping plus
joint-transmitted actuator velocity terms only. At force saturation, the
review probe measured `qacc=-1` with the full clamp derivative and `qacc=-0.167`
with this intentional clamp-derivative omission.

## selection

native JSON selects the mode at the world root:

```json
{"timestep": 0.005, "integrator": "Euler"}
```

Accepted names are `RK4`, `Euler`, and `implicitfast`. `implicit` is an alias
for newt's same documented implicitfast scope. Lower-case aliases exist for
the first two. Unknown names fail with a path-aware loader error.

MJCF uses the MuJoCo option:

```xml
<option timestep="0.005" integrator="Euler"/>
```

`RK4`, `Euler`, and `implicitfast` map to `Integrator::Rk4`,
`Integrator::Euler`, and `Integrator::ImplicitFast`. The public default is
`Integrator::Rk4`.

## verification

The selection suite includes three byte goldens. The RK4 golden runs on the
same default path as the pre-v3 engine. Each golden has a symmetry-broken
pose, mixed inertia, and non-zero linear and angular velocity.

The matched MuJoCo captures use MuJoCo 3.11.0, captured on 2026-08-15 with
the XML integrator overridden to Euler. The rows below report the observed
maximum absolute error, followed by the stated bound:

| scenario | qpos observed / bound | qvel observed / bound |
| --- | ---: | ---: |
| ballistic | `1.24e-5 / 3.0e-5` | `3.61e-5 / 8.0e-5` |
| double pendulum | `1.71e-7 / 4.0e-7` | `4.29e-7 / 1.0e-6` |
| sphere drop | `7.22e-3 / 1.5e-2` | `4.20e-1 / 9.0e-1` |
| box stack | `9.76e-3 / 2.0e-2` | `2.25e-1 / 5.0e-1` |

The matched rows are tighter than cross-integrator comparisons because both
sides now use Euler. The damping-heavy filtered pendulum also has an
`implicitfast` MuJoCo capture. Its observed errors are `1.38e-7` qpos and
`3.71e-7` qvel, with bounds `3.0e-7` and `1.0e-6`.

The tumbling energy anchor records the integrator order without claiming
conservation. Over 2000 steps at `dt=0.005`, the measured initial, Euler,
and RK4 kinetic energies are `64.019996643`, `77.718658447`, and
`64.020149231`. This torque-free free-body case gains energy under the
semi-implicit qvel update; RK4 remains near-conservative. The result is
kept explicit because a blanket claim that Euler always dissipates would be
false for this anchor.

The assisted biped Euler smoke rollout ran for 2000 steps with
`assist_scale=0.8`. It measured `1.1560 m` forward distance, `102.00 bpm`,
`0.4287 m` mean step length, `0.1674 m` foot clearance, and zero self-contact
steps. The test uses lower gates than the RK4 acceptance run because Euler
is a parity mode, not a claim that it beats the tuned RK4 controller.
