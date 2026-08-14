//! Actuators and applied forces (tier 4).
//!
//! Two flavors of controlled input in v0:
//!
//! - [`PdServo`] — PD position servo for a single 1-DOF joint (hinge or,
//!   from v1 tier 1, slide). Computes `τ = kp·(target − q) − kd·qdot`,
//!   clamped to a symmetric force bound. For a hinge `τ` is a torque
//!   (N·m) and `q` is an angle (rad); for a slide `τ` is a force (N) and
//!   `q` is a displacement (m). Same 1-DOF plumbing either way. The `kd`
//!   gain is derived at construction from a damping ratio and a caller-
//!   supplied reflected inertia estimate — see [`PdServo::from_dampratio`].
//! - Direct joint torques — go through the existing `Tree::qfrc_applied`
//!   slot. Convenience helpers on [`Tree`] (`set_joint_torque_clamped`,
//!   `add_joint_torque`) write into it with an optional symmetric clamp.
//!
//! External 6D wrenches on links live on `Tree::applied_wrenches` and get
//! summed into ABA's `external_wrenches` inside [`crate::tree::aba`], so
//! they act on articulated links exactly the same way contacts do — same
//! `f_ext_body` path, no special case.
//!
//! # PD damping model (why `2·ζ·√(kp·I_ref)`)
//!
//! A single hinge with reflected inertia `I_ref` under PD control obeys
//! `I_ref · q̈ = kp·(target − q) − kd·qdot`, i.e. a linear second-order
//! system with natural frequency `ωₙ = √(kp / I_ref)` and damping ratio
//! `ζ = kd / (2·√(kp·I_ref))`. Solving for `kd`:
//!
//! ```text
//! kd = 2 · ζ · √(kp · I_ref)
//! ```
//!
//! At `ζ = 1` the response is critically damped: fastest settling with no
//! overshoot. At `ζ < 1` the response overshoots; at `ζ > 1` it settles
//! slowly without overshoot. This matches MuJoCo's `position` actuator
//! with `dampratio` (reference: `mjc_bias`, MuJoCo source, and the
//! MuJoCo XML `<position dampratio="1"/>` idiom the biped work uses).
//!
//! The caller supplies `I_ref` because ABA does not materialize the joint's
//! diagonal mass-matrix element — the actual `Sᵀ IA S` at a hinge depends
//! on the whole subtree's articulated inertia and would recouple the
//! actuator to the topology. For the arm demo and typical robot joints,
//! [`reflected_inertia_point_mass`] gives a reasonable estimate; users
//! calibrating in the field can measure the effective inertia and pass it
//! in directly.

/// PD position servo for a 1-DOF joint (hinge or slide).
///
/// Generalized force at a joint state `(q, qdot)` is
///
/// ```text
/// τ_raw = kp·(target − q) − kd·qdot
/// τ     = clamp(τ_raw, −force_range, +force_range)
/// ```
///
/// with `force_range <= 0` interpreted as "no clamp". `target` is settable
/// per step through [`crate::tree::Tree::set_actuator_target`]. Units follow
/// the joint kind (hinge: rad / rad·s / N·m; slide: m / m·s / N) — the
/// servo does not need to know which; `qfrc_applied` is the same 1-DOF
/// generalized-force slot either way.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PdServo {
    /// Link index in the containing tree. Must reference a hinge or slide.
    pub link_idx: usize,
    /// Position gain (N·m per rad of error).
    pub kp: f32,
    /// Velocity gain (N·m per rad/s). Derived from a damping ratio via
    /// [`PdServo::from_dampratio`] when using that constructor; can also be
    /// set directly.
    pub kd: f32,
    /// Symmetric torque clamp magnitude. `<= 0` disables the clamp.
    pub force_range: f32,
    /// Desired joint angle (rad).
    pub target: f32,
}

impl PdServo {
    /// Build a PD servo with explicit `kp` and `kd`.
    pub const fn new(link_idx: usize, kp: f32, kd: f32, force_range: f32, target: f32) -> Self {
        Self {
            link_idx,
            kp,
            kd,
            force_range,
            target,
        }
    }

    /// Build a PD servo with `kd = 2·dampratio·√(kp·reflected_inertia)`.
    ///
    /// `reflected_inertia` is the effective moment of inertia the joint's
    /// motor sees — for a hinge that swings a single link,
    /// [`reflected_inertia_point_mass`] approximates it well.
    ///
    /// The starting `target` is 0.
    pub fn from_dampratio(
        link_idx: usize,
        kp: f32,
        dampratio: f32,
        reflected_inertia: f32,
        force_range: f32,
    ) -> Self {
        let kd = 2.0 * dampratio * (kp * reflected_inertia).sqrt();
        Self {
            link_idx,
            kp,
            kd,
            force_range,
            target: 0.0,
        }
    }

    /// Torque at the given joint state, after clamping.
    #[inline]
    pub fn torque(&self, q: f32, qdot: f32) -> f32 {
        let raw = self.kp * (self.target - q) - self.kd * qdot;
        clamp_symmetric(raw, self.force_range)
    }

    /// Torque before clamping. Exposed for tests and diagnostics.
    #[inline]
    pub fn torque_unclamped(&self, q: f32, qdot: f32) -> f32 {
        self.kp * (self.target - q) - self.kd * qdot
    }
}

/// Effective moment of inertia of a hinge that swings a point-mass at
/// distance `length` from the pivot, plus rotor `armature`:
///
/// ```text
/// I_ref = mass · length² + armature
/// ```
///
/// For a uniform rod pivoted at one end, `I_pivot = m·L²/3`; call with
/// `mass, length` set so `mass·length² = m·L²/3` (e.g. `length = L/√3`),
/// or pass the pivot inertia directly through the raw [`PdServo::new`].
#[inline]
pub fn reflected_inertia_point_mass(mass: f32, length: f32, armature: f32) -> f32 {
    mass * length * length + armature
}

/// Symmetric clamp with pass-through when `cap <= 0`.
#[inline]
pub(crate) fn clamp_symmetric(x: f32, cap: f32) -> f32 {
    if cap <= 0.0 {
        x
    } else if x > cap {
        cap
    } else if x < -cap {
        -cap
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn dampratio_recovers_critical_damping_formula() {
        // ωn = √(kp/I) = 10, kd_critical = 2·1·√(kp·I) = 2·√100 = 20.
        let kp = 100.0f32;
        let i_ref = 1.0f32;
        let servo = PdServo::from_dampratio(0, kp, 1.0, i_ref, -1.0);
        assert!(approx(servo.kd, 20.0, 1e-5));
        // At ζ = 0.5, kd is half of critical.
        let underdamped = PdServo::from_dampratio(0, kp, 0.5, i_ref, -1.0);
        assert!(approx(underdamped.kd, 10.0, 1e-5));
    }

    #[test]
    fn torque_clamps_symmetrically() {
        let mut servo = PdServo::new(0, 100.0, 0.0, 5.0, 0.0);
        servo.target = 10.0; // huge error → raw = 1000
        assert!(approx(servo.torque(0.0, 0.0), 5.0, 1e-6));
        servo.target = -10.0;
        assert!(approx(servo.torque(0.0, 0.0), -5.0, 1e-6));
        // force_range = 0 disables clamp.
        servo.force_range = 0.0;
        servo.target = 10.0;
        assert!(approx(servo.torque(0.0, 0.0), 1000.0, 1e-6));
    }

    #[test]
    fn clamp_symmetric_passes_negative_cap_through() {
        // cap <= 0 is documented to mean "no clamp".
        assert!(approx(clamp_symmetric(42.0, -1.0), 42.0, 1e-6));
        assert!(approx(clamp_symmetric(-42.0, 0.0), -42.0, 1e-6));
    }

    #[test]
    fn reflected_inertia_point_mass_matches_formula() {
        // 2 kg at 0.5 m, armature 0.05: I = 2·0.25 + 0.05 = 0.55.
        let i = reflected_inertia_point_mass(2.0, 0.5, 0.05);
        assert!(approx(i, 0.55, 1e-6));
    }
}
