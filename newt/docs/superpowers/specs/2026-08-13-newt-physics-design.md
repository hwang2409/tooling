# newt — Rigid-Body Physics Engine Design

Date: 2026-08-13
Status: approved by Henry (design conversation, tooling-dev orchestrator session)
Sibling: [chimy2](../../../../chimy2/) — the software rasterizer that visualizes newt
Prior art: `~/me/fun/biped` — Python/MuJoCo bipedal walker (the reproduction target)

## Purpose

Build a rigid-body physics engine from scratch in Rust: articulated dynamics in
generalized coordinates, soft contacts, PD actuators. chimy2 renders it. The
endgame is an interactive robot simulation running in the browser via wasm,
culminating in a port of the biped walker validated against its recorded gait
metrics.

## Non-goals

- No physics, math, collision, or linear-algebra crates. Everything is
  hand-written (the chimy rule, inherited).
- No LCP/convex-optimization contact solver. Contacts are penalty-based
  (MuJoCo-style soft contacts), which the biped project proves sufficient.
- No maximal-coordinate constraint solver. Joints are structural (reduced
  coordinates), not constraints to be enforced iteratively.
- No soft bodies, fluids, or deformables.
- No real-time guarantees beyond "the demos are interactive".

## Dependency rule

Zero runtime dependencies for the engine crate. Demo binaries may depend on
chimy2 (which itself keeps its winit+softbuffer-only rule). The wasm build
follows chimy2's pattern: `wasm32-unknown-unknown`, scalar FFI, no wasm-bindgen.

## Architecture: generalized coordinates

The world is a set of kinematic trees. Each tree has a root joint (free 6-DOF
or fixed) and a chain/branching of hinge joints (more joint types later).
State is `q` (generalized positions: root quaternion+translation, joint
angles) and `qdot` (generalized velocities).

Forward dynamics via Featherstone's Articulated Body Algorithm (ABA):
O(n) per tree, no global mass matrix inversion. The spatial-algebra
6-vector/6-matrix machinery is hand-written in a dedicated module.

Pipeline per step (RK4 over the whole state):
1. forward kinematics — body poses from `q`
2. collision detection — geoms vs world/each other (filtered pairs)
3. contact forces — penalty normal + friction, applied as external wrenches
4. actuator forces — PD position control, clamped, plus user wrenches
5. ABA — `qddot` from forces
6. integrate — RK4, quaternion renormalization at step end

Determinism doctrine (inherited from chimy2, load-bearing):
- no platform libm in any production or golden path — trig/exp/sqrt come from
  the same deterministic routines chimy2 uses (extracted or reimplemented in
  newt's math module; byte-identical macOS/Linux is CI-enforced)
- no unordered iteration affecting results; contact ordering is total and
  index-tie-broken
- fixed timestep only (default 5 ms, matching biped); no wall-clock coupling
- golden trajectories: serialized `q`/`qdot` snapshots at fixed step counts,
  byte-compared in CI

## Components

- `math` — Vec3/Quat/Mat3, spatial vectors (motion/force), spatial inertia,
  deterministic scalar functions
- `body` — rigid body: mass, COM, inertia tensor, attached geoms, sites
- `joint` — free, hinge (axis, range, damping, armature); later: slide, ball
- `tree` — kinematic tree topology, `q`/`qdot` layout, forward kinematics
- `collide` — geom types (sphere, capsule, box, plane), pair filtering
  (self-collision off by default within a tree, per biped's Phase 51 lesson),
  narrow-phase returning contact points/normals/depths
- `contact` — soft contact model: penetration-spring normal force
  (solref/solimp-style parameterization), pyramidal friction approximating an
  elliptic cone
- `actuator` — PD position servos with gain, damping ratio, force clamp;
  external 6D wrenches on any body
- `dynamics` — ABA implementation
- `integrate` — RK4, energy accounting (kinetic/potential for tests)
- `model` — JSON model format (same hand-written parser style as chimy2's
  `src/json.rs`): bodies, joints, geoms, sites, actuators, contact params
- `viz` — adapter producing chimy2 scene submissions from world state (demo
  binaries only)

## Testing doctrine

- Analytic anchors: single pendulum period vs closed form, projectile arc,
  energy conservation of a free tumbling body (drift bounded), block-on-plane
  static friction threshold.
- Golden trajectories: byte-identical serialized states after N steps,
  cross-platform via CI.
- Independent-twin rule (inherited): no test may compare the implementation
  against itself; anchors come from closed-form math or hand-computed values.
- Reviewer exact probes become permanent tests.
- Every tier lands a chimy2-rendered demo; the orchestrator renders and
  inspects at gate time.

## Tier ladder

1. **core** — math + spatial algebra + single free body + RK4 + gravity;
   tumbling-bodies demo
2. **contacts** — geoms, ground plane, penalty contacts, friction; stacking
   and rolling demos
3. **joints** — hinge + free root, ABA for trees; pendulum, N-link chain,
   swinging demos
4. **actuators** — PD servos, wrenches; commanded 3-link arm demo
5. **model format** — JSON robots, sites, contact filtering; arm loaded from
   file
6. **web** — wasm build, browser demo page integration (newt scenes in the
   chimy2 demo app), interactive control
7. **biped** — port the walker model + parametric gait controller; validate
   against biped's recorded metrics (cadence > 80 bpm, step length > 0.05 m,
   foot clearance > 0.06 m, forward distance ~0.68 m per 1000 steps no-assist,
   zero self-contact)

Each tier ships as fleet tickets with the standing gates: CI green, goldens,
demo inspected, review round, orchestrator self-merge.

## Reference: what the biped port needs (from the survey)

10 hinge joints (5/leg: hip roll, hip, knee, ankle roll, ankle), free-joint
pelvis, capsule thighs/shins + box feet/torso, per-joint damping 2–8 and
armature 0.02–0.05, position actuators kp 45–80 with force bounds 45–80 N,
friction coefficient 1.2, timestep 5 ms, RK4, contact-force extraction for
metrics, external balance wrenches, site queries (heels/toes), robot-ground
contacts with robot-self collisions disabled.
