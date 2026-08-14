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
//! - **Tier 3** (`joint`, `tree`) — kinematic trees, hinge joints with
//!   limits/damping/armature, Featherstone's ABA for O(n) forward dynamics,
//!   RK4 on generalized coordinates. See `docs/joints.md`.
//! - **Tier 4** (`actuator`, tree extensions) — PD position servos with
//!   damping-ratio parameterization and force clamp, motor-style direct
//!   joint torques, and world-frame external wrenches on articulated
//!   links. See `docs/actuators.md`.
//! - **Tier 5** (`json`, `model`) — native JSON scene format:
//!   hand-written parser, strict-by-default loader with JSON-path errors,
//!   packaged model files (`newt/models/*.json`), sites with world-pose
//!   query. See `docs/model-format.md`.

pub mod actuator;
pub mod body;
pub mod contact;
pub mod geom;
pub mod joint;
pub mod json;
pub mod math;
pub mod model;
pub mod spatial;
pub mod tree;
pub mod world;
