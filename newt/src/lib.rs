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
//! # Tier 1 scope
//!
//! Core dynamics: math, spatial algebra, single free bodies, RK4, gravity.
//! No joints, no contacts, no actuators. Later tiers add those; see
//! `docs/core.md` for how to run the tumbling demo and the anchor tests.

pub mod body;
pub mod contact;
pub mod geom;
pub mod math;
pub mod spatial;
pub mod world;
