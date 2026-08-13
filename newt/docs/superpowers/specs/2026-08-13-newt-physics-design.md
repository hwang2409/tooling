# newt — Rigid-Body Physics Engine Design

Date: 2026-08-13 (revised same day: MuJoCo parity is the north star, versioned roadmap)
Status: approved by Henry (design conversation, tooling-dev orchestrator session)
Sibling: [chimy2](../../../../chimy2/) — the software rasterizer that visualizes newt
Reference implementation: MuJoCo. Prior art for validation: `~/me/fun/biped`
(Python/MuJoCo bipedal walker — ONE v2 validation milestone, not the scope)

## Purpose

Recreate MuJoCo from scratch in Rust: articulated dynamics in generalized
coordinates, soft contacts, actuators, tendons, sensors, MJCF-compatible model
loading — versioned v0/v1/v2/v3 because this is a long project. chimy2 renders
it. The endgame is genuine MuJoCo feature parity.

## Relationship to chimy2 (the architecture split)

Same split as MuJoCo core vs its viewer. newt is a pure computation library:
zero dependencies, no rendering, runs headless and native. chimy2 is the
viewer: native demo binaries render newt worlds. Demo binaries couple the
two; the engine crate never links the renderer.

wasm is NOT a requirement (Henry, 2026-08-13). chimy2 compiles to wasm because
Henry demos it on his website; newt does not need to. The zero-dependency rule
happens to keep the door open, and a browser sim may become a nice-to-have
later — but no ticket should carry wasm work or wasm CI for newt unless Henry
asks.

## Non-goals (permanent)

- No physics, math, collision, or linear-algebra crates. Everything is
  hand-written (the chimy rule, inherited).
- No GPU compute. CPU only.
- No real-time guarantees beyond "the demos are interactive".

## Dependency rule

Zero runtime dependencies for the engine crate. Demo binaries may dev-depend
on chimy2. If a wasm build ever happens, it follows chimy2's pattern
(`wasm32-unknown-unknown`, scalar FFI, no wasm-bindgen) — but see above: not
a requirement.

## Architecture: generalized coordinates

The world is a set of kinematic trees (free/fixed roots, joint chains with
branching). State is `q`/`qdot` in joint space, exactly as MuJoCo lays it out
(quaternion+translation for free joints, scalars for hinges/slides, quaternion
for ball joints from v1).

Forward dynamics via Featherstone's Articulated Body Algorithm (O(n) per
tree); CRB + RNE join in v1 for mass-matrix and inverse-dynamics paths. The
spatial-algebra 6-vector machinery is a dedicated hand-written module.

Pipeline per step:
1. forward kinematics — body/geom/site poses from `q`
2. collision detection — filtered pairs, narrow phase
3. constraint/contact forces — v0: penalty; v1+: MuJoCo's soft-constraint
   model (solref/solimp) with a real solver
4. actuator forces + user-applied wrenches
5. forward dynamics — `qddot`
6. integrate — RK4 (v0); Euler and implicit-in-velocity join later

Determinism doctrine (inherited from chimy2, load-bearing):
- no platform libm in any production or golden path; trig/exp come from
  hand-written range-reduced polynomials; `+ - * / sqrt` are IEEE-exact and
  allowed
- no unordered iteration affecting results; contact ordering total and
  index-tie-broken
- fixed timestep only; no wall-clock coupling
- golden trajectories: serialized `q`/`qdot` at fixed step counts,
  byte-compared macOS vs Linux in CI

## Versioned roadmap

### v0 — rigid-body mini (current arc)

Enough engine to make articulated toys real and visible end to end.

1. **core** — math + spatial algebra + free bodies + RK4 + gravity;
   tumbling demo (NEWT-1)
2. **contacts** — plane/sphere/box/capsule geoms, penalty normal force,
   pyramidal friction; stacking and rolling demos
3. **joints** — hinge + free root, ABA on trees; pendulum and chain demos
4. **actuators** — PD position servos with clamps, external wrenches;
   commanded 3-link arm demo
5. **model format** — JSON robots (hand-written parser, chimy2 json.rs
   style), sites, contact filtering
(A former "web/wasm" tier was cut from v0 on 2026-08-13: wasm is an optional
future nice-to-have, not a requirement.)

### v1 — MuJoCo core parity (kinematics, contacts, constraints)

- joints: ball, slide; joint limits as constraints
- geoms: cylinder, ellipsoid, convex mesh; margin/gap
- MuJoCo's actual contact model: solref/solimp parameterization, condim
  1/3/4/6 (frictionless, sliding, torsional, rolling), elliptic and pyramidal
  cones, solved with PGS over the soft-constraint formulation (the v0 penalty
  path stays as a debug/reference mode)
- equality constraints: connect, weld, joint coupling, distance
- CRB mass matrix + RNE inverse-dynamics building blocks
- sensors, first battery: jointpos/jointvel, framepos/framequat, accelerometer,
  gyro, touch, force/torque
- MJCF loader for a documented subset — real MuJoCo XML models load (defaults/
  classes, compiler basics); our JSON stays as the native format

### v2 — actuation + robotics

- general actuator model (gain/bias), types: motor, position, velocity,
  cylinder; activation dynamics
- tendons: fixed, then spatial with wrapping; pulleys
- inverse dynamics and Jacobian APIs
- keyframes, mocap bodies, full sensor battery (rangefinder, magnetometer...)
- **biped validation milestone**: port the walker model + parametric gait
  controller from `~/me/fun/biped`; reproduce its recorded gait metrics
  (cadence > 80 bpm, step length > 0.05 m, foot clearance > 0.06 m, ~0.68 m
  forward per 1000 steps no-assist, zero self-contact)

### v3 — frontier

- integrators: Euler, implicit-in-velocity
- Newton solver; sparse factorization performance work
- heightfields, SDF geoms; muscles; maybe flex/deformables
- analytic derivatives
- (optional, only if Henry asks) wasm build and a browser sim/studio

Version boundaries are checkpoints, not contracts — features can move when a
tier teaches us something. Each version ships as fleet tickets with the
standing gates: CI green, goldens, demo rendered and inspected at gate time,
review round, orchestrator self-merge.

## Testing doctrine

- Analytic anchors: closed-form pendulum period, projectile arc, energy
  conservation of torque-free tumbling (bounded drift), static friction
  thresholds, gyroscopic (Dzhanibekov) flip.
- From v1: differential testing against reference MuJoCo trajectories
  captured offline from the real MuJoCo (via the biped repo's environment) —
  tolerance-based, never byte-based; MuJoCo is the oracle for parity claims.
- Golden trajectories: byte-identical serialized states cross-platform in CI.
- Independent-twin rule: no test compares the implementation against itself.
- Reviewer exact probes become permanent tests.

## Reference notes from the biped survey (v2 milestone requirements)

10 hinge joints (5/leg), free-joint pelvis, capsule thighs/shins + box
feet/torso, per-joint damping 2–8, armature 0.02–0.05, position actuators
kp 45–80 with force bounds 45–80 N, friction 1.2, timestep 5 ms, RK4,
contact-force extraction, external balance wrenches, site queries
(heels/toes), robot-ground contacts with self-collision disabled.
