//! newt — a from-scratch rigid-body physics engine.
//!
//! # Modules
//!
//! - [`math`] — deterministic Vec3, Mat3, Quat, and hand-written sin/cos/tan.
//! - [`spatial`] — Featherstone-style 6-vectors, spatial inertia, Plücker
//!   transforms. Load-bearing for tier 3+ (ABA); tier 1 uses only the inertia
//!   primitives.
//! - [`body`] — free rigid body with mass, body-frame inertia, pose, twist.
//! - [`world`] — N free bodies under uniform gravity, RK4 integration.
//!
//! # Determinism
//!
//! Zero runtime dependencies. No platform libm (sin, cos, tan, exp, ln, powf)
//! anywhere in the engine. Fixed timestep. `Vec<Body>` iteration order is
//! deterministic. See `docs/superpowers/specs/2026-08-13-newt-physics-design.md`
//! for the load-bearing determinism doctrine and tier ladder.
//!
//! # Tier scope
//!
//! - **Tier 1** (`math`, `spatial`, `body`, `world`) — core dynamics: math,
//!   spatial algebra, single free bodies, RK4, gravity. See `docs/core.md`.
//! - **Tier 2** (`geom`, `contact`, contacts in `world`) — collision
//!   primitives + penalty contact model + pyramidal friction. See
//!   `docs/contacts.md`.
//! - **Tier 3 / v1 tier 1** (`joint`, `tree`) — kinematic trees, hinge /
//!   slide / ball joints with limits/damping/armature (ball limits deferred
//!   to the v1 solver), Featherstone's ABA for O(n) forward dynamics, RK4
//!   on generalized coordinates. See `docs/joints.md`.
//! - **Tier 4** (`actuator`, tree extensions) — PD position servos with
//!   damping-ratio parameterization and force clamp, motor-style direct
//!   joint torques, and world-frame external wrenches on articulated
//!   links. See `docs/actuators.md`.
//! - **Tier 5** (`json`, `model`) — native JSON scene format:
//!   hand-written parser, strict-by-default loader with JSON-path errors,
//!   packaged model files (`newt/models/*.json`), sites with world-pose
//!   query. See `docs/model-format.md`.
//! - **v1 tier 3** (`dynamics`) — CRB mass matrix `M(q)`, RNE inverse
//!   dynamics `τ(q, qdot, qddot)`, bias vector `h(q, qdot)`, and a hand-
//!   rolled dense Cholesky factor + solve. Building blocks the v1 soft-
//!   constraint solver consumes. See `docs/dynamics.md`.
//! - **v1 tier 4** (`solver`) — MuJoCo soft-constraint contact model
//!   (5-parameter SolImp, SolRef reference acceleration, regularized
//!   dual) solved with fixed-iteration PGS. Ships condim 1 / 3, both
//!   pyramidal and elliptic friction cones, and constraint-based joint
//!   limits for hinge/slide. Opt-in via `world.solver.mode = Pgs`;
//!   penalty is still default and preserves every existing golden.
//!   See `docs/solver.md`.
//! - **v1 tier 5** (`equality`, solver/model extensions) — equality
//!   constraints (connect, weld, joint coupling, distance) as bilateral
//!   PGS rows with per-constraint `(solref, solimp)`; contact `condim`
//!   extended to `4` (torsional friction about the normal) and `6`
//!   (rolling friction about the two tangents), both pyramidal and
//!   elliptic. See `docs/solver.md` (equality-constraint rows + condim
//!   4/6 block).
//! - **v1 tier 6** (`sensor`) — first battery of MuJoCo-parity sensors:
//!   jointpos/jointvel, ballquat/ballangvel, framepos/framequat, gyro,
//!   proper-acceleration accelerometer, touch, force/torque. Declared on
//!   the [`World`], evaluated after each `step` into a flat deterministic
//!   `sensordata` vector, no perturbation to the simulation state. See
//!   `docs/sensors.md`.
//! - **v2 tier 3** (`tendon`) — fixed and spatial tendons with sphere
//!   wrap. Fixed tendons sum scalar joint coordinates; spatial tendons
//!   chain sites with the envelope-theorem Jacobian. Passive springs
//!   / dampers feed the tree's `tau` via `Jᵀ · F` inside ABA. PGS
//!   solver rows enforce length limits per tendon; actuators may
//!   target a tendon (motor / general etc.) with the same transmission-
//!   space convention as joint transmissions. Cylinder wrap and pulley
//!   branches are rejected loudly at load time. See `docs/tendons.md`.

pub mod actuator;
pub mod body;
pub mod contact;
pub mod dynamics;
pub mod equality;
pub mod geom;
pub mod jacobian;
pub mod joint;
pub mod json;
pub mod math;
pub mod mjcf;
pub mod model;
pub mod sensor;
pub mod solver;
pub mod spatial;
pub mod tendon;
pub mod tree;
pub mod world;
pub mod xml;
