//! World: free rigid bodies under gravity, integrated with RK4.
//!
//! Tier 1 scope: N free bodies with no joints, no contacts, uniform gravity as
//! the only external force. Fixed timestep `dt`. See
//! `docs/superpowers/specs/2026-08-13-newt-physics-design.md`.

use crate::body::Body;
use crate::math::{Quat, Vec3};

/// Simulation world.
#[derive(Clone, Debug, PartialEq)]
pub struct World {
    /// Fixed integration timestep. Default 5 ms (matches biped).
    pub dt: f32,
    /// Uniform gravity vector applied to every body's COM. defaults to
    /// `(0, 0, -9.81)` m/s².
    pub gravity: Vec3,
    /// Bodies. index-stable across the whole simulation.
    pub bodies: Vec<Body>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            dt: 0.005,
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: Vec::new(),
        }
    }

    /// Adds a body and returns its stable index.
    pub fn add_body(&mut self, body: Body) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Advance the whole world by one fixed-dt RK4 step.
    ///
    /// The RK4 is applied to each body independently (they are all free and
    /// tier 1 has no coupling). Bodies iterate in insertion order.
    pub fn step(&mut self) {
        for body in &mut self.bodies {
            *body = rk4_step(*body, self.gravity, self.dt);
        }
    }
}

/// Body-state derivative under gravity only (no applied torque or force other
/// than gravity acting at the COM). The gravity wrench produces zero torque
/// about the COM.
#[derive(Clone, Copy, Debug)]
struct Deriv {
    dposition: Vec3,
    dlinear_velocity: Vec3,
    /// Quaternion time derivative (not necessarily unit; renormalize once per
    /// step after integrating).
    dorientation: Quat,
    dangular_velocity_body: Vec3,
}

fn evaluate(state: Body, gravity: Vec3) -> Deriv {
    // Linear: dv/dt = g. Gravity acts at COM → no torque contribution.
    let dlin = gravity;

    // Angular: Euler's equation in body frame with no external torque.
    // I * dω/dt = -ω × (I ω).
    let iw = state.inertia_body * state.angular_velocity_body;
    let gyroscopic = -state.angular_velocity_body.cross(iw);
    let dang_body = state.inertia_body_inverse * gyroscopic;

    // Orientation: dq/dt = 0.5 * q * (ω_body, 0).
    let dq = state.orientation.derivative(state.angular_velocity_body);

    // Position: dp/dt = linear velocity.
    let dp = state.linear_velocity;

    Deriv {
        dposition: dp,
        dlinear_velocity: dlin,
        dorientation: dq,
        dangular_velocity_body: dang_body,
    }
}

fn advance(state: Body, deriv: Deriv, dt: f32) -> Body {
    // Note: we do NOT renormalize the quaternion in intermediate stages —
    // that would introduce a nonlinearity between stages and break RK4's
    // formal order. Renormalization happens once at step end, in rk4_step.
    Body {
        position: state.position + deriv.dposition * dt,
        linear_velocity: state.linear_velocity + deriv.dlinear_velocity * dt,
        orientation: state.orientation + deriv.dorientation * dt,
        angular_velocity_body: state.angular_velocity_body + deriv.dangular_velocity_body * dt,
        ..state
    }
}

fn rk4_step(state: Body, gravity: Vec3, dt: f32) -> Body {
    let k1 = evaluate(state, gravity);
    let k2 = evaluate(advance(state, k1, dt * 0.5), gravity);
    let k3 = evaluate(advance(state, k2, dt * 0.5), gravity);
    let k4 = evaluate(advance(state, k3, dt), gravity);

    let sixth = 1.0 / 6.0;
    let dp = (k1.dposition + k2.dposition * 2.0 + k3.dposition * 2.0 + k4.dposition) * sixth;
    let dv = (k1.dlinear_velocity
        + k2.dlinear_velocity * 2.0
        + k3.dlinear_velocity * 2.0
        + k4.dlinear_velocity)
        * sixth;
    let dq =
        (k1.dorientation + k2.dorientation * 2.0 + k3.dorientation * 2.0 + k4.dorientation) * sixth;
    let dw = (k1.dangular_velocity_body
        + k2.dangular_velocity_body * 2.0
        + k3.dangular_velocity_body * 2.0
        + k4.dangular_velocity_body)
        * sixth;

    let position = state.position + dp * dt;
    let linear_velocity = state.linear_velocity + dv * dt;
    let orientation = (state.orientation + dq * dt).renormalize();
    let angular_velocity_body = state.angular_velocity_body + dw * dt;

    Body {
        position,
        linear_velocity,
        orientation,
        angular_velocity_body,
        ..state
    }
}
