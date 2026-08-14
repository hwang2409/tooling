//! Equality constraints (v1 tier 5).
//!
//! Four bilateral constraint kinds, each solved as extra rows in the same
//! PGS sweep the contact solver already runs:
//!
//! - [`Equality::Connect`] — 3-DOF point coincidence between two body anchor
//!   points (or body-world). Three linear rows along the world basis
//!   `(x, y, z)`.
//! - [`Equality::Weld`] — 6-DOF pose lock. Three linear rows (anchor
//!   coincidence) plus three angular rows (relative-orientation error as a
//!   quaternion difference vector `2 · imag(q_err)`).
//! - [`Equality::JointCoupling`] — scalar equation on two 1-DOF joints of the
//!   SAME tree: `q_a = c0 + c1·q_b + c2·q_b²`. One scalar row on the tree's
//!   `nv` vector. Cross-tree coupling is intentionally out of scope this
//!   tier (would require a world-level A matrix).
//! - [`Equality::Distance`] — `|p_a − p_b| = d0`. One linear row along the
//!   current separation unit vector. Degenerate when the two anchors
//!   coincide — the row is elided that step (documented) and picked up
//!   again as soon as the separation exceeds the guard.
//!
//! Every equality carries its own `(solref, solimp)`. MuJoCo folds these
//! through the same regularized-dual machinery as contacts and limits:
//! `R = ((1 − d)/d) · A_ii`, `bias = J qdot_free + d · a_ref · dt`. Since
//! equalities are bilateral, the PGS update projects a delta impulse
//! WITHOUT a non-negativity clamp — the impulse can push in either
//! direction to drive the residual to zero.
//!
//! # Determinism
//!
//! Equality rows are appended AFTER contact rows (and, for tree DOFs, after
//! limit rows). Within one `Equality`, rows are emitted in a fixed sub-
//! order: connect → `(x, y, z)`; weld → `(x, y, z, rot_x, rot_y, rot_z)`;
//! distance → 1 row; joint coupling → 1 row. Two equalities of the same
//! kind emit in declaration order.

use crate::geom::SolRef;
use crate::math::{Quat, Vec3};
use crate::solver::SolImp;

/// One equality constraint. See the module docs for the four kinds and how
/// each maps to PGS rows.
#[derive(Clone, Debug, PartialEq)]
pub enum Equality {
    /// Anchor points on two bodies (or a body and the world) coincide in
    /// world coordinates. Three linear rows.
    ///
    /// `body_a = None` (or `body_b = None`) means "attach to the world at
    /// this world-frame point"; the corresponding `anchor` is then read as
    /// a world-frame position rather than a body-local offset.
    Connect {
        body_a: Option<usize>,
        body_b: Option<usize>,
        anchor_a: Vec3,
        anchor_b: Vec3,
        solref: SolRef,
        solimp: SolImp,
    },
    /// Pose lock between two bodies: anchor points coincide AND relative
    /// orientation stays at `relative_orientation` (child-B-in-A frame).
    /// Six rows: three linear (the anchor equation, same as `Connect`)
    /// then three angular. The orientation error is
    /// `2 · imag(q_err)` in world coordinates, with
    /// `q_err = q_A · q_target · q_B_conj` (chosen so `q_err = identity`
    /// when the target relative pose is met, i.e.
    /// `q_B = q_A · q_target`), sign-canonicalized (multiply by −1 when
    /// `q_err.w < 0` to take the shortest rotation).
    ///
    /// A `body_a = None` means the world; then `q_A` is treated as
    /// identity. When both bodies are `None` the constraint is degenerate
    /// (two world anchors have no relative motion) — the loader rejects
    /// that.
    Weld {
        body_a: Option<usize>,
        body_b: Option<usize>,
        anchor_a: Vec3,
        anchor_b: Vec3,
        /// Locked child-B-in-A relative orientation. The convention is
        /// `q_B_world = q_A_world · relative_orientation`, so
        /// `relative_orientation = Quat::IDENTITY` locks the two body
        /// frames parallel.
        relative_orientation: Quat,
        solref: SolRef,
        solimp: SolImp,
    },
    /// Polynomial coupling of two 1-DOF joints on the SAME tree:
    /// `q_a = c0 + c1·q_b + c2·q_b²`.
    ///
    /// Both `link_a` and `link_b` must be a hinge or slide (`nv = 1`) in
    /// tree `tree`. `polycoef = [c0, c1, c2]`; higher-order terms are not
    /// supported this tier (MuJoCo's `polycoef` extends to degree 4 —
    /// deferred).
    JointCoupling {
        tree: usize,
        link_a: usize,
        link_b: usize,
        polycoef: [f32; 3],
        solref: SolRef,
        solimp: SolImp,
    },
    /// Fixed distance between two anchor points: `|p_a − p_b| = distance`.
    /// One linear row along the current separation direction. When
    /// `|p_a − p_b|` drops below [`DISTANCE_DEGENERATE_EPS`] the row is
    /// elided that step (no direction is well-defined); the constraint
    /// re-engages as soon as the separation grows past the guard.
    Distance {
        body_a: Option<usize>,
        body_b: Option<usize>,
        anchor_a: Vec3,
        anchor_b: Vec3,
        distance: f32,
        solref: SolRef,
        solimp: SolImp,
    },
}

/// Below this current separation, [`Equality::Distance`] elides its row for
/// one step (no direction is well-defined). Chosen small enough to trigger
/// only at genuine coincidence; large enough not to divide by an
/// underflowed norm.
pub const DISTANCE_DEGENERATE_EPS: f32 = 1.0e-6;

impl Equality {
    /// Structural validation. Ranges (positive distance, valid solimp) —
    /// body/link/tree indices are checked separately by the world/loader
    /// (they need access to the world to resolve indices).
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Equality::Connect { solimp, .. } | Equality::Weld { solimp, .. } => {
                solimp.validate().map_err(|m| format!("solimp: {m}"))
            }
            Equality::JointCoupling {
                polycoef, solimp, ..
            } => {
                solimp.validate().map_err(|m| format!("solimp: {m}"))?;
                if !polycoef.iter().all(|c| c.is_finite()) {
                    return Err("polycoef entries must be finite".to_string());
                }
                Ok(())
            }
            Equality::Distance {
                distance, solimp, ..
            } => {
                if !distance.is_finite() {
                    return Err("distance must be finite".to_string());
                }
                if *distance < 0.0 {
                    return Err(format!("distance must be ≥ 0, got {distance}"));
                }
                solimp.validate().map_err(|m| format!("solimp: {m}"))
            }
        }
    }

    /// Which tree this equality touches, if any. Used by the tree solver
    /// to gather joint-coupling equalities per tree.
    pub fn tree_index(&self) -> Option<usize> {
        match self {
            Equality::JointCoupling { tree, .. } => Some(*tree),
            _ => None,
        }
    }

    /// Whether this equality is body-space (contributes to the free-body
    /// PGS solver rather than the per-tree limit solver).
    pub fn is_free_body(&self) -> bool {
        !matches!(self, Equality::JointCoupling { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_rejects_negative() {
        let eq = Equality::Distance {
            body_a: Some(0),
            body_b: Some(1),
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            distance: -0.1,
            solref: SolRef::DEFAULT,
            solimp: SolImp::DEFAULT,
        };
        assert!(eq.validate().is_err());
    }

    #[test]
    fn distance_rejects_nan() {
        let eq = Equality::Distance {
            body_a: Some(0),
            body_b: Some(1),
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            distance: f32::NAN,
            solref: SolRef::DEFAULT,
            solimp: SolImp::DEFAULT,
        };
        assert!(eq.validate().is_err());
    }

    #[test]
    fn coupling_rejects_nonfinite_polycoef() {
        let eq = Equality::JointCoupling {
            tree: 0,
            link_a: 1,
            link_b: 2,
            polycoef: [0.0, f32::INFINITY, 0.0],
            solref: SolRef::DEFAULT,
            solimp: SolImp::DEFAULT,
        };
        assert!(eq.validate().is_err());
    }

    #[test]
    fn is_free_body_maps_correctly() {
        let connect = Equality::Connect {
            body_a: Some(0),
            body_b: Some(1),
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            solref: SolRef::DEFAULT,
            solimp: SolImp::DEFAULT,
        };
        assert!(connect.is_free_body());
        let coupling = Equality::JointCoupling {
            tree: 0,
            link_a: 1,
            link_b: 2,
            polycoef: [0.0, 2.0, 0.0],
            solref: SolRef::DEFAULT,
            solimp: SolImp::DEFAULT,
        };
        assert!(!coupling.is_free_body());
        assert_eq!(coupling.tree_index(), Some(0));
    }
}
