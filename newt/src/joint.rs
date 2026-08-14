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
//! - [`JointKind::Slide`] — single axis prismatic. `nq = 1`, `nv = 1`. State
//!   is displacement along the axis (meters) and rate (m/s).
//! - [`JointKind::Ball`] — 3-DOF rotation. `nq = 4` (child-frame quaternion
//!   `(qx, qy, qz, qw)`, renormalized at step end like the free root). `nv =
//!   3` (body-frame ω). Ball joint limits are deferred to the v1 constraint
//!   solver; see the field docs on [`JointKind::Ball`].
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
use crate::solver::SolImp;

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
        /// [`JointLimit`].
        range: Option<(f32, f32)>,
        /// Joint damping torque coefficient: `τ_damp = -damping * qdot`.
        damping: f32,
        /// Rotor inertia added on the hinge axis (kg·m²). Enters the
        /// articulated-inertia scalar `Sᵀ IA S + armature` in ABA and shows
        /// up as the classical MuJoCo reflected inertia.
        armature: f32,
        /// Penalty parameters for the range limit; see [`JointLimit`].
        limit: JointLimit,
    },
    /// Single-axis prismatic (slide) joint. `nq = nv = 1`; the coordinate is
    /// a displacement `q` in meters along the axis. The axis is in the joint
    /// frame, which coincides with the child body frame at `q = 0`; since a
    /// slide keeps the two frames identically oriented, the axis is fixed in
    /// both parent and child at all `q`. The ABA joint subspace is a pure
    /// translation `S = (0, axis)` at the child COM.
    Slide {
        /// Unit-length slide axis in the joint frame.
        axis: Vec3,
        /// Optional `(low, high)` slide displacement limits (meters). Same
        /// penalty spring-damper model as hinge; see [`JointLimit`].
        range: Option<(f32, f32)>,
        /// Linear damping coefficient: `F_damp = -damping * qdot`.
        damping: f32,
        /// Reflected translational inertia added on the axis (kg). Enters
        /// ABA's diagonal as `Sᵀ IA S + armature`. Analogous to a motor
        /// rotor mass reflected through a rack-and-pinion.
        armature: f32,
        /// Penalty parameters for the range limit; see [`JointLimit`].
        limit: JointLimit,
    },
    /// Ball (3-DOF spherical) joint. `nq = 4` — a child-frame quaternion
    /// `(qx, qy, qz, qw)` representing the child body's rotation relative to
    /// the parent (right-multiplied: `child_ori = parent_ori * q_ball`).
    /// `nv = 3` — body-frame angular velocity `(ωx, ωy, ωz)` in the child's
    /// body frame at the COM. The ABA joint subspace is 3 columns
    /// `S_k = (e_k, r_jc × e_k)` for `k ∈ {0, 1, 2}` (the joint anchor in
    /// child body coords is `r_jc`).
    ///
    /// # Deferred: joint limits
    ///
    /// Ball joints do NOT support range limits in v0/v1-tier-1. A physically
    /// correct 3-DOF orientation limit (cone, swing/twist) needs the real
    /// constraint solver landing in the next ticket (v1 tier 2 — PGS over
    /// solref/solimp). The [`crate::model`] loader rejects a `range` field
    /// on a ball joint with an error pointing at that deferral.
    Ball {
        /// Isotropic angular damping: `τ_damp = -damping * ω_body` on each
        /// rotational axis (added as a 3-vector into `qfrc_applied`'s
        /// effective torque during ABA pass 2).
        damping: f32,
        /// Reflected rotor inertia (kg·m²) added on the diagonal of the ball
        /// joint's articulated-inertia block `D = Sᵀ IA S + armature · I₃`.
        /// Uniform across the three rotational axes.
        armature: f32,
    },
}

/// Spring-damper parameters for enforcing a single-DOF joint's range limit
/// (hinge or slide). Interpreted per side (low and high) as a one-sided
/// spring: the generalized force is zero inside the range and grows linearly
/// with the violation depth outside.
///
/// `stiffness` and `damping` drive the v0/penalty pathway. `solref` and
/// `solimp` drive the v1 constraint solver ([`crate::solver::solve_tree_limits`]);
/// both default to `None`, which the solver reads as
/// [`crate::geom::SolRef::DEFAULT`] / [`crate::solver::SolImp::DEFAULT`].
/// Populated overrides tune the limit's constraint-solver behavior per joint,
/// mirroring how contacts already carry per-geom SolRef/SolImp.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointLimit {
    /// Penalty spring stiffness per unit violation (N·m/rad for hinge,
    /// N/m for slide). Only consulted under [`crate::solver::SolverMode::Penalty`].
    pub stiffness: f32,
    /// Penalty damping coefficient applied to the joint rate WHEN outside
    /// the range on the violated side. Only consulted under
    /// [`crate::solver::SolverMode::Penalty`]. `2 sqrt(k I_eff)` is
    /// critical damping.
    pub damping: f32,
    /// Optional per-limit override for the PGS constraint's SolRef.
    /// `None` → use [`crate::geom::SolRef::DEFAULT`]. Only consulted
    /// under [`crate::solver::SolverMode::Pgs`].
    pub solref: Option<crate::geom::SolRef>,
    /// Optional per-limit override for the PGS constraint's SolImp.
    /// `None` → use [`SolImp::DEFAULT`]. Only consulted under
    /// [`crate::solver::SolverMode::Pgs`].
    pub solimp: Option<SolImp>,
}

impl JointLimit {
    /// Stiff-but-stable defaults for a single unit-mass link with
    /// `I ≈ 1` about the hinge axis (or an equivalent single unit mass on a
    /// slide): `k = 1000` gives a limit frequency of ≈ 31 rad/s (period ≈
    /// 200 ms) — much slower than the `dt = 5 ms` integrator so RK4 stays
    /// stable, and much faster than typical motion so the limit feels rigid
    /// on human timescales. Damping is ≈ critical for that spring/inertia.
    /// Both solver overrides default to `None` so the solver falls back
    /// to `SolRef::DEFAULT` / `SolImp::DEFAULT`.
    pub const DEFAULT: Self = Self {
        stiffness: 1000.0,
        damping: 60.0,
        solref: None,
        solimp: None,
    };

    pub const fn new(stiffness: f32, damping: f32) -> Self {
        Self {
            stiffness,
            damping,
            solref: None,
            solimp: None,
        }
    }

    /// Override the PGS solver's SolRef for this limit (builder-style).
    pub const fn with_solref(mut self, solref: crate::geom::SolRef) -> Self {
        self.solref = Some(solref);
        self
    }

    /// Override the PGS solver's SolImp for this limit (builder-style).
    pub const fn with_solimp(mut self, solimp: SolImp) -> Self {
        self.solimp = Some(solimp);
        self
    }
}

/// Historical alias — the v0 code called this `HingeLimit`; v1 promoted the
/// type to cover slide joints too (same math, different units on the DOF).
/// Kept so tier-3 tests that reference `HingeLimit` compile unchanged.
pub type HingeLimit = JointLimit;

impl JointKind {
    /// Number of position (q) slots this joint contributes.
    pub const fn nq(&self) -> usize {
        match self {
            JointKind::Free => 7,
            JointKind::Fixed => 0,
            JointKind::Hinge { .. } => 1,
            JointKind::Slide { .. } => 1,
            JointKind::Ball { .. } => 4,
        }
    }

    /// Number of velocity (qdot) slots this joint contributes.
    pub const fn nv(&self) -> usize {
        match self {
            JointKind::Free => 6,
            JointKind::Fixed => 0,
            JointKind::Hinge { .. } => 1,
            JointKind::Slide { .. } => 1,
            JointKind::Ball { .. } => 3,
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
            limit: JointLimit::DEFAULT,
        }
    }

    /// Convenience constructor for a slide with default penalty limits and
    /// zero damping/armature.
    pub fn slide(axis: Vec3) -> Self {
        JointKind::Slide {
            axis,
            range: None,
            damping: 0.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        }
    }

    /// Convenience constructor for a ball joint with zero damping/armature.
    pub fn ball() -> Self {
        JointKind::Ball {
            damping: 0.0,
            armature: 0.0,
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
        let s = JointKind::slide(Vec3::Z);
        assert_eq!(s.nq(), 1);
        assert_eq!(s.nv(), 1);
        let b = JointKind::ball();
        assert_eq!(b.nq(), 4);
        assert_eq!(b.nv(), 3);
    }
}
