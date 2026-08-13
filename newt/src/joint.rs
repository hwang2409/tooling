//! Joints connect a child link to its parent (or to the world at a tree's
//! root). Each joint kind declares how many position (`nq`) and velocity
//! (`nv`) slots it contributes to the tree's generalized-coordinate vectors,
//! MuJoCo-style.
//!
//! # Layout
//!
//! - [`JointKind::Free`] — 6-DOF root joint. `nq = 7`: `(px, py, pz, qx, qy,
//!   qz, qw)`. `nv = 6`: spatial velocity `(ωx, ωy, ωz, vx, vy, vz)` in the
//!   link's own body frame (matches tier-1 `angular_velocity_body` and
//!   world-frame `linear_velocity` semantics via composition below).
//! - [`JointKind::Fixed`] — rigid attachment to parent (or to the world if
//!   this is a root). `nq = 0`, `nv = 0`. The initial pose is baked into the
//!   link's [`joint_offset_in_parent`](crate::tree::Link::joint_offset_in_parent).
//! - [`JointKind::Hinge`] — single axis rotation. `nq = 1`, `nv = 1`. State is
//!   the joint angle (radians) and joint rate (rad/s).
//!
//! Ball and slide joints are on the v1 roadmap.
//!
//! # Free-root velocity frame
//!
//! For a free root, the 6 velocity slots are the components of a spatial
//! motion vector expressed in the LINK's body frame, angular-then-linear.
//! Tier-1 free bodies store `angular_velocity_body` (body coords) and
//! `linear_velocity` (world coords). The tree->free-body bridge translates
//! at construction and inspection time; it is not exercised in ABA proper
//! since ABA never mixes free-body-only code paths with articulated trees.

use crate::math::Vec3;

/// Kind of joint connecting a link to its parent.
///
/// See the module docs for the `nq`/`nv` contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind {
    /// 6-DOF root joint. Only valid for a link with no parent.
    Free,
    /// Rigid attachment. `nq = nv = 0`.
    Fixed,
    /// Single-axis hinge. Axis is expressed in the joint frame (which
    /// coincides with the child body frame at `q = 0`, given the standard
    /// convention that `joint_offset_in_child.orientation = IDENTITY`).
    Hinge {
        /// Unit-length hinge axis in the joint frame.
        axis: Vec3,
        /// Optional `(low, high)` joint angle limits, radians. Enforced as a
        /// smooth spring-damper penalty torque outside the range; see
        /// [`HingeLimit`].
        range: Option<(f32, f32)>,
        /// Joint damping torque coefficient: `τ_damp = -damping * qdot`.
        damping: f32,
        /// Rotor inertia added on the hinge axis (kg·m²). Enters the
        /// articulated-inertia scalar `Sᵀ IA S + armature` in ABA and shows
        /// up as the classical MuJoCo reflected inertia.
        armature: f32,
        /// Penalty parameters for the range limit; see [`HingeLimit`].
        limit: HingeLimit,
    },
}

/// Spring-damper parameters for enforcing a hinge's range limit. Interpreted
/// per side (low and high) as a one-sided spring: force is zero inside the
/// range and grows linearly with the violation depth outside.
///
/// This is the v0 model. v1 will replace it with a real constraint solved by
/// the same solver that handles contacts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HingeLimit {
    /// Spring stiffness per unit violation (N·m/rad).
    pub stiffness: f32,
    /// Damping coefficient applied to the joint rate WHEN the joint is
    /// outside the range on the corresponding side. `2 sqrt(k I_eff)` is
    /// critical damping; we default to a comfortably damped value.
    pub damping: f32,
}

impl HingeLimit {
    /// Reasonable stiff-but-stable defaults for a single unit-mass link with
    /// `I ≈ 1` about the hinge axis: `k = 1000 N·m/rad` gives a limit
    /// frequency of ≈ 31 rad/s (period ≈ 200 ms) — much slower than the
    /// dt = 5 ms integrator so RK4 stays stable, and much faster than
    /// typical motion so the limit feels rigid on human timescales.
    /// Damping is ≈ critical for that spring/inertia.
    pub const DEFAULT: Self = Self {
        stiffness: 1000.0,
        damping: 60.0,
    };

    pub const fn new(stiffness: f32, damping: f32) -> Self {
        Self { stiffness, damping }
    }
}

impl JointKind {
    /// Number of position (q) slots this joint contributes.
    pub const fn nq(&self) -> usize {
        match self {
            JointKind::Free => 7,
            JointKind::Fixed => 0,
            JointKind::Hinge { .. } => 1,
        }
    }

    /// Number of velocity (qdot) slots this joint contributes.
    pub const fn nv(&self) -> usize {
        match self {
            JointKind::Free => 6,
            JointKind::Fixed => 0,
            JointKind::Hinge { .. } => 1,
        }
    }

    /// Convenience constructor for a hinge with default penalty limits and
    /// zero damping/armature.
    pub fn hinge(axis: Vec3) -> Self {
        JointKind::Hinge {
            axis,
            range: None,
            damping: 0.0,
            armature: 0.0,
            limit: HingeLimit::DEFAULT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dof_counts_match_spec() {
        assert_eq!(JointKind::Free.nq(), 7);
        assert_eq!(JointKind::Free.nv(), 6);
        assert_eq!(JointKind::Fixed.nq(), 0);
        assert_eq!(JointKind::Fixed.nv(), 0);
        let h = JointKind::hinge(Vec3::X);
        assert_eq!(h.nq(), 1);
        assert_eq!(h.nv(), 1);
    }
}
