//! Spatial algebra (Featherstone).
//!
//! # Convention
//!
//! We follow Featherstone (Rigid Body Dynamics Algorithms, 2008) with **motion
//! vectors on top of force vectors** — the standard M/F split. A motion vector
//! stacks angular over linear parts; a force vector stacks torque over linear
//! force. All 6-vectors here are in a single coordinate frame; the transform
//! between frames is a Plücker matrix ([`Xform`]).
//!
//! ```text
//! Motion m = [ω]   ⋅   Force f = [τ]
//!            [v]              [F]
//! ```
//!
//! Spatial dot product pairs motion with force: `m · f = ω·τ + v·F`. Spatial
//! cross products come in two flavors:
//!   - `crm(m1) * m2`: motion-cross-motion (bracket for two rigid velocities).
//!   - `crf(m1) * f2 = -(crm(m1))ᵀ * f2`: motion-cross-force (used in the
//!     gyroscopic bias term of Newton-Euler and in ABA).
//!
//! # Non-goals of tier 1
//!
//! ABA and articulated forward dynamics are tier 3+. Tier 1 uses this module
//! only for the spatial inertia and Newton-Euler on a **single free body** —
//! we still want the algebra pinned down so tier 3 doesn't have to redefine
//! it.

use crate::math::{Mat3, Vec3};
use core::ops::{Add, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// spatial vectors
// ---------------------------------------------------------------------------

/// Spatial motion vector: 3D angular part on top of 3D linear part.
///
/// Ordered so that `dot(SpatialMotion, SpatialForce) = ω·τ + v·F`, the power
/// exchanged when the body moves at this twist under this wrench.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpatialMotion {
    pub angular: Vec3,
    pub linear: Vec3,
}

impl SpatialMotion {
    pub const ZERO: Self = Self {
        angular: Vec3::ZERO,
        linear: Vec3::ZERO,
    };

    pub const fn new(angular: Vec3, linear: Vec3) -> Self {
        Self { angular, linear }
    }

    /// Motion-motion cross bracket: `crm(a) * b = [a ×] b` in spatial form,
    /// i.e. the Lie bracket `[a, b]` of two velocities.
    ///
    /// Featherstone (2.15): `[a, b] = ([a_ω × b_ω], [a_ω × b_v + a_v × b_ω])`.
    pub fn cross_motion(self, other: Self) -> Self {
        Self::new(
            self.angular.cross(other.angular),
            self.angular.cross(other.linear) + self.linear.cross(other.angular),
        )
    }

    /// Motion-force cross: `crf(m) * f`. Equal to `-(crm(m))ᵀ * f`, expanded
    /// below to avoid materializing the matrix.
    ///
    /// Featherstone (2.16): `crf(a) * f = ([a_ω × f_τ + a_v × f_F], [a_ω × f_F])`.
    pub fn cross_force(self, force: SpatialForce) -> SpatialForce {
        SpatialForce::new(
            self.angular.cross(force.torque) + self.linear.cross(force.linear),
            self.angular.cross(force.linear),
        )
    }
}

impl Add for SpatialMotion {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(self.angular + r.angular, self.linear + r.linear)
    }
}
impl Sub for SpatialMotion {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        Self::new(self.angular - r.angular, self.linear - r.linear)
    }
}
impl Neg for SpatialMotion {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.angular, -self.linear)
    }
}
impl Mul<f32> for SpatialMotion {
    type Output = Self;
    fn mul(self, r: f32) -> Self {
        Self::new(self.angular * r, self.linear * r)
    }
}

/// Spatial force (wrench): 3D torque on top of 3D linear force.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct SpatialForce {
    pub torque: Vec3,
    pub linear: Vec3,
}

impl SpatialForce {
    pub const ZERO: Self = Self {
        torque: Vec3::ZERO,
        linear: Vec3::ZERO,
    };

    pub const fn new(torque: Vec3, linear: Vec3) -> Self {
        Self { torque, linear }
    }
}

impl Add for SpatialForce {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self::new(self.torque + r.torque, self.linear + r.linear)
    }
}
impl Sub for SpatialForce {
    type Output = Self;
    fn sub(self, r: Self) -> Self {
        Self::new(self.torque - r.torque, self.linear - r.linear)
    }
}
impl Neg for SpatialForce {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.torque, -self.linear)
    }
}
impl Mul<f32> for SpatialForce {
    type Output = Self;
    fn mul(self, r: f32) -> Self {
        Self::new(self.torque * r, self.linear * r)
    }
}

/// Scalar dot product `m · f = ω·τ + v·F`. Represents power exchange.
pub fn spatial_dot(m: SpatialMotion, f: SpatialForce) -> f32 {
    m.angular.dot(f.torque) + m.linear.dot(f.linear)
}

// ---------------------------------------------------------------------------
// spatial inertia
// ---------------------------------------------------------------------------

/// Spatial inertia of a rigid body about a fixed reference frame.
///
/// Stored in the classical block form so `f = I * m` (where `m` is a spatial
/// motion vector representing acceleration) gives the wrench needed to
/// produce that acceleration for a body with mass `mass`, center of mass at
/// `com` (in the reference frame), and body-frame inertia tensor `inertia_com`
/// about the COM.
///
/// The 6x6 matrix form is
/// ```text
/// [ I_com + m [c]×[c]×ᵀ    m [c]× ]
/// [       m [c]×ᵀ            m 1  ]
/// ```
/// but tier 1 only needs `apply_to_motion` and `newton_euler_bias`, so we keep
/// the primitives instead of materializing the 6x6.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpatialInertia {
    pub mass: f32,
    /// Center of mass relative to the reference frame origin, in reference
    /// frame coordinates.
    pub com: Vec3,
    /// Inertia tensor **about the COM**, expressed in the reference frame's
    /// orientation.
    pub inertia_com: Mat3,
}

impl SpatialInertia {
    pub const fn new(mass: f32, com: Vec3, inertia_com: Mat3) -> Self {
        Self {
            mass,
            com,
            inertia_com,
        }
    }

    /// Diagonal body-frame inertia (typical when the reference frame is a
    /// principal-axis body frame with the COM at the origin).
    pub fn principal_axis(mass: f32, ixx: f32, iyy: f32, izz: f32) -> Self {
        Self::new(mass, Vec3::ZERO, Mat3::diag(ixx, iyy, izz))
    }

    /// Inertia tensor about the reference-frame origin (parallel-axis shift).
    ///
    /// `I_origin = I_com + m * ([c]×ᵀ [c]×) = I_com + m * (‖c‖² 1 − c cᵀ)`
    pub fn inertia_about_origin(self) -> Mat3 {
        let c = self.com;
        let sk = Mat3::skew(c);
        // [c]×ᵀ [c]× = -[c]× [c]× (skew transpose is negation).
        // Materialize both to avoid depending on Mat3 negation composition.
        let sk_t = sk.transpose();
        self.inertia_com + (sk_t * sk) * self.mass
    }

    /// Spatial inertia times a spatial motion. Interpreted as
    /// momentum-per-velocity (call with a twist to get the spatial momentum)
    /// or as wrench-per-acceleration (call with an acceleration to get the
    /// required wrench, ignoring the gyroscopic bias which the caller adds).
    ///
    /// Derivation in COM-frame quantities. Let `v_com = v + ω × c` be the
    /// COM linear velocity for reference-frame velocity `v` and offset `c`.
    /// Linear momentum is `p = m * v_com`. Angular momentum about the
    /// reference origin is `L = I_com * ω + c × p` (parallel-axis for the
    /// angular part). The returned SpatialForce packs `(L, p)` in the
    /// `(torque, linear)` layout, matching the M/F pairing convention.
    pub fn times_motion(self, motion: SpatialMotion) -> SpatialForce {
        let v_com = motion.linear + motion.angular.cross(self.com);
        let p = v_com * self.mass;
        let l = self.inertia_com * motion.angular + self.com.cross(p);
        SpatialForce::new(l, p)
    }
}

// ---------------------------------------------------------------------------
// Plücker transform between frames
// ---------------------------------------------------------------------------

/// Plücker transform between coordinate frames.
///
/// Applied to a motion vector `m_a` expressed in frame A, it produces the
/// same physical motion expressed in frame B. The transform is parameterized
/// by [`Xform::rot_a_to_b`] (the linear map from A-coords to B-coords) and
/// [`Xform::translation_a_in_b`] (the origin of A expressed in B).
///
/// For motion vectors:
/// ```text
/// m_b.ω = R_a→b * m_a.ω
/// m_b.v = R_a→b * m_a.v + p × (R_a→b * m_a.ω)
/// ```
/// For force vectors:
/// ```text
/// f_b.F = R_a→b * f_a.F
/// f_b.τ = R_a→b * f_a.τ + p × (R_a→b * f_a.F)
/// ```
///
/// The force transform is the transpose-inverse of the motion transform;
/// the block structure above is Featherstone's Plücker matrix `X*` for force.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xform {
    /// Linear map from A-coordinates to B-coordinates: `v_b = rot_a_to_b * v_a`.
    pub rot_a_to_b: Mat3,
    /// Origin of A expressed in B coordinates.
    pub translation_a_in_b: Vec3,
}

impl Xform {
    pub const IDENTITY: Self = Self {
        rot_a_to_b: Mat3::IDENTITY,
        translation_a_in_b: Vec3::ZERO,
    };

    pub const fn new(rot_a_to_b: Mat3, translation_a_in_b: Vec3) -> Self {
        Self {
            rot_a_to_b,
            translation_a_in_b,
        }
    }

    /// Push a motion vector from frame A to frame B.
    pub fn motion(self, m: SpatialMotion) -> SpatialMotion {
        let w = self.rot_a_to_b * m.angular;
        let v = self.rot_a_to_b * m.linear + self.translation_a_in_b.cross(w);
        SpatialMotion::new(w, v)
    }

    /// Push a force vector from frame A to frame B.
    pub fn force(self, f: SpatialForce) -> SpatialForce {
        let fl = self.rot_a_to_b * f.linear;
        let tau = self.rot_a_to_b * f.torque + self.translation_a_in_b.cross(fl);
        SpatialForce::new(tau, fl)
    }
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{FRAC_PI_2, Quat};

    #[inline]
    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn approx_vec(a: Vec3, b: Vec3, tol: f32) -> bool {
        approx(a.x, b.x, tol) && approx(a.y, b.y, tol) && approx(a.z, b.z, tol)
    }

    #[test]
    fn spatial_dot_matches_power() {
        // Hand-computed: m · f = 1*4 + 2*5 + 3*6 + 7*10 + 8*11 + 9*12.
        let m = SpatialMotion::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(7.0, 8.0, 9.0));
        let f = SpatialForce::new(Vec3::new(4.0, 5.0, 6.0), Vec3::new(10.0, 11.0, 12.0));
        assert_eq!(spatial_dot(m, f), 4.0 + 10.0 + 18.0 + 70.0 + 88.0 + 108.0);
    }

    #[test]
    fn cross_motion_is_lie_bracket() {
        // [a, b] = -[b, a]. Hand-computed axis pair.
        let a = SpatialMotion::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let b = SpatialMotion::new(Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0));
        let ab = a.cross_motion(b);
        let ba = b.cross_motion(a);
        assert!(approx_vec(ab.angular, -ba.angular, 1e-6));
        assert!(approx_vec(ab.linear, -ba.linear, 1e-6));
        // hand-computed: ω = (1,0,0)×(0,0,1) = (0,-1,0);
        // v = (1,0,0)×(1,0,0) + (0,1,0)×(0,0,1) = (0,0,0) + (1,0,0) = (1,0,0).
        assert!(approx_vec(ab.angular, Vec3::new(0.0, -1.0, 0.0), 1e-6));
        assert!(approx_vec(ab.linear, Vec3::new(1.0, 0.0, 0.0), 1e-6));
    }

    #[test]
    fn cross_force_equals_negative_crm_transpose() {
        // crf(a) f = -(crm(a))ᵀ f — verify with two independent expansions.
        let a = SpatialMotion::new(Vec3::new(0.5, -1.0, 2.0), Vec3::new(3.0, 0.5, -0.75));
        let f = SpatialForce::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(-4.0, 5.0, -6.0));
        let via_method = a.cross_force(f);
        // hand form: torque = a.ω × f.τ + a.v × f.F, linear = a.ω × f.F.
        let expected_torque = a.angular.cross(f.torque) + a.linear.cross(f.linear);
        let expected_linear = a.angular.cross(f.linear);
        assert!(approx_vec(via_method.torque, expected_torque, 1e-6));
        assert!(approx_vec(via_method.linear, expected_linear, 1e-6));
    }

    #[test]
    fn parallel_axis_shifts_inertia() {
        // Point mass m at distance r on the x-axis: I about origin = m r²
        // about the y and z axes (and zero about x, since a point on the axis
        // has no moment arm to it). I_com = 0. hand-computed.
        let m = 2.5;
        let r = 3.0;
        let inertia = SpatialInertia::new(m, Vec3::new(r, 0.0, 0.0), Mat3::ZERO);
        let about_o = inertia.inertia_about_origin();
        assert!(approx(about_o.get(0, 0), 0.0, 1e-5));
        assert!(approx(about_o.get(1, 1), m * r * r, 1e-5));
        assert!(approx(about_o.get(2, 2), m * r * r, 1e-5));
        // symmetry.
        assert!(approx(about_o.get(0, 1), about_o.get(1, 0), 1e-6));
        assert!(approx(about_o.get(0, 2), about_o.get(2, 0), 1e-6));
    }

    #[test]
    fn times_motion_reproduces_momentum_of_pure_translation() {
        // Rigid body with unit mass, unit diagonal inertia, COM at origin,
        // translating at v = (1, 2, 3) with zero ω. Expected p = (1, 2, 3),
        // L = 0. hand-computed.
        let i = SpatialInertia::principal_axis(1.0, 1.0, 1.0, 1.0);
        let m = SpatialMotion::new(Vec3::ZERO, Vec3::new(1.0, 2.0, 3.0));
        let f = i.times_motion(m);
        assert!(approx_vec(f.linear, Vec3::new(1.0, 2.0, 3.0), 1e-6));
        assert!(approx_vec(f.torque, Vec3::ZERO, 1e-6));
    }

    #[test]
    fn times_motion_reproduces_angular_momentum_about_offset_com() {
        // Rigid body mass m, inertia I_com=diag(1,2,3), COM at c=(1,0,0),
        // spinning at ω=(0,0,ω0) about origin, v=0.
        // L_origin = I_com ω + c × (m (v + ω × c)) = (0,0, 3 ω0) + m * (1,0,0)×(0,ω0,0)
        //           = (0,0,3ω0) + m*(0,0,ω0*0? ) let's compute:
        //  ω × c = (0,0,ω0) × (1,0,0) = (0, ω0, 0)  [since z×x = y]
        //  v_com = 0 + (0, ω0, 0)
        //  p = m v_com = (0, m ω0, 0)
        //  c × p = (1,0,0) × (0, m ω0, 0) = (0, 0, m ω0)
        //  L = (0, 0, 3 ω0) + (0, 0, m ω0) = (0, 0, (3 + m) ω0)
        let mass = 2.0;
        let inertia_com = Mat3::diag(1.0, 2.0, 3.0);
        let com = Vec3::new(1.0, 0.0, 0.0);
        let omega0 = 1.5;
        let si = SpatialInertia::new(mass, com, inertia_com);
        let m = SpatialMotion::new(Vec3::new(0.0, 0.0, omega0), Vec3::ZERO);
        let f = si.times_motion(m);
        assert!(approx_vec(
            f.linear,
            Vec3::new(0.0, mass * omega0, 0.0),
            1e-5
        ));
        assert!(approx_vec(
            f.torque,
            Vec3::new(0.0, 0.0, (3.0 + mass) * omega0),
            1e-5
        ));
    }

    #[test]
    fn xform_motion_is_inverse_of_reverse_xform() {
        // A rotation by 90° around z, translated by (2, 0, 0).
        let q = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        let rot = q.to_mat3();
        let xf = Xform::new(rot, Vec3::new(2.0, 0.0, 0.0));

        // Motion in A frame: pure rotation about x with linear velocity.
        let m_a = SpatialMotion::new(Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 3.0, 0.0));

        // Push forward, then push back with the reverse transform.
        let m_b = xf.motion(m_a);
        // Reverse transform: rotation of B relative to A is Rᵀ, and origin of
        // B in A is -Rᵀ * translation.
        let inv_rot = rot.transpose();
        let inv_trans = -(inv_rot * xf.translation_a_in_b);
        let xf_inv = Xform::new(inv_rot, inv_trans);
        let m_back = xf_inv.motion(m_b);
        assert!(approx_vec(m_back.angular, m_a.angular, 1e-5));
        assert!(approx_vec(m_back.linear, m_a.linear, 1e-5));
    }

    #[test]
    fn xform_preserves_power_between_frames() {
        // The power m · f is a scalar invariant under a Plücker change of
        // frame. A famous property; if this fails, the transforms are wrong.
        let q = Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.7);
        let xf = Xform::new(q.to_mat3(), Vec3::new(0.5, -1.2, 3.4));

        let m_a = SpatialMotion::new(Vec3::new(0.3, -0.5, 1.1), Vec3::new(-2.0, 0.0, 4.0));
        let f_a = SpatialForce::new(Vec3::new(1.0, -2.0, 0.5), Vec3::new(0.75, 3.0, -1.0));

        let p_a = spatial_dot(m_a, f_a);
        let p_b = spatial_dot(xf.motion(m_a), xf.force(f_a));
        assert!(approx(p_a, p_b, 1e-4), "power {p_a} vs {p_b}");
    }
}
