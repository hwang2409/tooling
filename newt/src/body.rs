//! Free rigid body: state, inertia, and pose update.
//!
//! # State layout
//!
//! - `position` — center of mass in world coordinates.
//! - `orientation` — unit quaternion mapping body-frame vectors to world.
//! - `linear_velocity` — d(position)/dt in world coordinates.
//! - `angular_velocity_body` — angular velocity ω expressed in the **body**
//!   frame. body-frame ω keeps the inertia tensor constant (Euler's equations
//!   of motion take their simple form).
//!
//! # Inertia
//!
//! `mass` and `inertia_body` describe the body about its COM in body-frame
//! axes. `inertia_body_inverse` is precomputed since it's needed every RK4
//! stage; it is invariant so the caller pays the cost once.

use crate::math::{Mat3, Quat, Vec3};

/// Deterministic rigid body state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub mass: f32,
    /// Inertia tensor about the COM, in body-frame axes.
    pub inertia_body: Mat3,
    /// Precomputed inverse; must equal `inertia_body.inverse()`.
    pub inertia_body_inverse: Mat3,

    /// World-frame COM position.
    pub position: Vec3,
    /// World-frame linear velocity of the COM.
    pub linear_velocity: Vec3,

    /// Body-to-world orientation.
    pub orientation: Quat,
    /// Body-frame angular velocity.
    pub angular_velocity_body: Vec3,
}

impl Body {
    /// Construct with an explicit inertia tensor about the COM in body axes.
    ///
    /// panics if the inertia is singular.
    pub fn new(mass: f32, inertia_body: Mat3, position: Vec3, orientation: Quat) -> Self {
        let inertia_body_inverse = inertia_body
            .inverse()
            .expect("body inertia tensor must be invertible");
        Self {
            mass,
            inertia_body,
            inertia_body_inverse,
            position,
            linear_velocity: Vec3::ZERO,
            orientation,
            angular_velocity_body: Vec3::ZERO,
        }
    }

    /// Diagonal-inertia constructor (principal-axis body frame, COM at origin).
    pub fn principal_axis(
        mass: f32,
        ixx: f32,
        iyy: f32,
        izz: f32,
        position: Vec3,
        orientation: Quat,
    ) -> Self {
        Self::new(mass, Mat3::diag(ixx, iyy, izz), position, orientation)
    }

    /// Uniform-density solid box of half-extents `hx, hy, hz` and mass `m`:
    /// principal moments `(m/3)(hy²+hz²), ...` (Featherstone Appendix A.1).
    pub fn solid_box(mass: f32, half_extents: Vec3, position: Vec3, orientation: Quat) -> Self {
        Self::new(
            mass,
            crate::geom::solid_box_inertia(mass, half_extents),
            position,
            orientation,
        )
    }

    /// Uniform-density solid sphere of radius `r` and mass `m`. `I = (2/5) m r²`.
    pub fn solid_sphere(mass: f32, radius: f32, position: Vec3, orientation: Quat) -> Self {
        Self::new(
            mass,
            crate::geom::solid_sphere_inertia(mass, radius),
            position,
            orientation,
        )
    }

    /// Uniform-density solid capsule with axis along the body's local Z. See
    /// [`crate::geom::solid_capsule_inertia`] for the derivation.
    pub fn solid_capsule(
        mass: f32,
        radius: f32,
        half_height: f32,
        position: Vec3,
        orientation: Quat,
    ) -> Self {
        Self::new(
            mass,
            crate::geom::solid_capsule_inertia(mass, radius, half_height),
            position,
            orientation,
        )
    }

    /// Angular velocity re-expressed in world coordinates.
    pub fn angular_velocity_world(&self) -> Vec3 {
        self.orientation.rotate(self.angular_velocity_body)
    }

    /// Kinetic energy `½ m v² + ½ ω · (I ω)`.
    pub fn kinetic_energy(&self) -> f32 {
        let lin = 0.5 * self.mass * self.linear_velocity.dot(self.linear_velocity);
        let iw = self.inertia_body * self.angular_velocity_body;
        let rot = 0.5 * self.angular_velocity_body.dot(iw);
        lin + rot
    }

    /// Angular momentum about the COM in world coordinates.
    pub fn angular_momentum_world(&self) -> Vec3 {
        let iw_body = self.inertia_body * self.angular_velocity_body;
        self.orientation.rotate(iw_body)
    }
}
