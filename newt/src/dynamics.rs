//! Inverse dynamics and mass matrix for kinematic trees.
//!
//! Two building blocks the v1 soft-constraint solver consumes:
//!
//! - [`mass_matrix`] — dense `nv × nv` joint-space inertia `M(q)` via the
//!   Composite Rigid Body algorithm (Featherstone RBDA §6, Table 6.2).
//! - [`inverse_dynamics`] — generalized force `τ` needed to produce a given
//!   `(q, qdot, qddot)` under gravity and external wrenches, via Recursive
//!   Newton-Euler (Featherstone RBDA §5, Table 5.1).
//! - [`bias_forces`] — `inverse_dynamics` with `qddot = 0` and no external
//!   wrenches; the Coriolis + centrifugal + gravity vector `h(q, qdot)`.
//!
//! Also included: a hand-rolled dense Cholesky factor + solve
//! ([`cholesky`], [`cholesky_solve`]) so tests and the future PGS solver
//! can invert `M` without a linear-algebra crate. No LAPACK, no libm — the
//! only transcendental used is `f32::sqrt`, which is IEEE-exact and
//! deterministic across every platform newt targets.
//!
//! # Contract: what `bias`/`inverse_dynamics` INCLUDE and EXCLUDE
//!
//! The equation of motion for the tree in joint space is
//!
//! ```text
//! M(q) · qddot + h(q, qdot) = τ_applied
//! ```
//!
//! where `τ_applied` is the sum of every generalized force the solver, the
//! actuators, damping, joint limits, `qfrc_applied`, and user wrenches
//! deliver into the joint's DOF. The functions in this module handle only
//! the rigid-body left-hand side:
//!
//! - `M(q)` includes each hinge/slide `armature` scalar on its diagonal and
//!   each ball joint's `armature · I₃` on its 3×3 block. Free-root has no
//!   armature slot, so the top-left 6×6 block is exactly `Ic[0]` — the
//!   composite spatial inertia of the whole tree at the root, materialized
//!   in `Mat6` form.
//! - The CRB pass fills each off-diagonal `(i, j)` cell and mirrors it to
//!   `(j, i)` in the same loop iteration — so a "M is symmetric" unit check
//!   is non-discriminating for that class of mutation. The load-bearing
//!   anchors for CRB correctness are (a) the ABA-solve round-trip
//!   `M · qddot ≈ τ` for a random `τ`, (b) the kinetic energy identity
//!   `KE = ½ qdotᵀ M qdot` cross-checked against a per-link sum, and
//!   (c) the PD-servo closed-loop response matching an independent
//!   Lagrangian twin. Those live in `newt/tests/dynamics_crb_rne.rs`.
//! - `h(q, qdot)` (== `bias_forces`) is `C(q, qdot) · qdot + g(q)`:
//!   Coriolis + centrifugal + gravity torques, per Newton-Euler. Gravity
//!   is applied as a per-link body-force `m_i · g_world` at each COM
//!   (rotated to the link's body frame), the same convention
//!   [`crate::tree::aba`] uses — so the two algorithms share only the
//!   spatial-algebra primitives and can validate each other.
//! - `inverse_dynamics` = `M(q) · qddot + h(q, qdot) - S_generalized(f_ext)`,
//!   i.e. it accepts an [`crate::tree::ExternalWrenches`] parameter
//!   identical in shape and semantics to ABA's. Wrenches enter through the
//!   same body-frame path; a nonzero `f_ext` reduces the residual `τ` the
//!   caller would otherwise have to supply.
//!
//! Everything else — `damping`, joint `range` limits, `actuators`,
//! `qfrc_applied`, `tree.applied_wrenches` — belongs on the RHS
//! `τ_applied` and is NOT consulted by these functions. This matches
//! MuJoCo's `qfrc_bias` semantics (bias = Coriolis + gravity, no passive
//! or actuator terms). Actuators/damping/limits/etc. sum into
//! `τ_applied` at whatever level the caller assembles the RHS.
//!
//! # The round-trip identity
//!
//! `ABA` and `RNE` are two independent algorithms that share only the
//! spatial-algebra primitives. If either has a bug in its per-link
//! recursion, the round-trip
//!
//! ```text
//! qddot = ABA(q, qdot, τ_input)
//! τ_out = RNE(q, qdot, qddot)
//! assert τ_out ≈ τ_input   (tight f32 tolerance)
//! ```
//!
//! will drift. Because the two algorithms share no per-link scratch, no
//! outer-recursion structure, and no `IA` update path, a mutation in
//! either that produces a physically wrong torque will not silently be
//! "confirmed" by the other. This is why the docs call this the primary
//! anchor.
//!
//! For the identity to hold, the test setup must eliminate every
//! `τ_applied` term the model in this crate delivers outside the RNE
//! contract:
//!
//! - `damping = 0`, no joint `range`, no actuators, no `qfrc_applied`
//!   contribution beyond the `τ_input` under test, and no
//!   `tree.applied_wrenches` (or set `f_ext` in ABA/RNE to sum them).
//! - `armature` may be any value — it is on the M diagonal and RNE adds
//!   `armature · qddot_slot` per DOF, so the identity holds.
//!
//! # Fixed-root gravity / external wrenches: a physical no-op
//!
//! For a tree whose root is `JointKind::Fixed`, the root link is pinned to
//! the world — `nq = nv = 0` at that slot. Any body-force applied there
//! (gravity `m_root · g`, or an external wrench passed via `f_ext[0]`) has
//! no DOF to attribute itself to: the RNE Pass-2 walk-up terminates at the
//! root with the accumulated wrench dropped on the floor, and `M(q)` has no
//! root row/column for it to enter. This mirrors what MuJoCo does with
//! `worldbody` and is not a bug — it is what a "fixed to the world" root
//! MEANS. Users who want a floating base should use `JointKind::Free` on
//! the root; users who want to sense the reaction wrench at a fixed base
//! will get it from a follow-up sensor/anchor-constraint ticket, not from
//! RNE.

use crate::joint::JointKind;
use crate::math::{Quat, Vec3};
use crate::spatial::{Mat6, SpatialForce, SpatialMotion, Xform};
use crate::tree::{
    ExternalWrenches, Tree, forward_kinematics, spatial_dot_ms, xup_for_link, xup_for_link_ball,
    xup_for_link_hinge, xup_for_link_slide,
};

// ---------------------------------------------------------------------------
// Shared pass-1 kinematics: compute per-link Xup, v, a, joint subspace, and
// the ExtBody wrench (gravity + external in body frame at COM). RNE consumes
// (v, a, f_ext) directly; CRB consumes only Xup.
// ---------------------------------------------------------------------------

/// Per-link spatial-motion / joint-subspace bundle needed by RNE.
///
/// Only the joint-kind-appropriate `s`/`s3` slot is populated; the other is
/// left at its default and never read on that link's arms. This mirrors
/// [`crate::tree::aba`]'s workspace layout.
struct RneScratch {
    xup: Vec<Xform>,
    v: Vec<SpatialMotion>,
    a: Vec<SpatialMotion>,
    f: Vec<SpatialForce>,
    /// 1-column subspace for hinge/slide.
    s: Vec<SpatialMotion>,
    /// 3-column subspace for ball.
    s3: Vec<[SpatialMotion; 3]>,
}

impl RneScratch {
    fn new(n: usize) -> Self {
        Self {
            xup: vec![Xform::IDENTITY; n],
            v: vec![SpatialMotion::ZERO; n],
            a: vec![SpatialMotion::ZERO; n],
            f: vec![SpatialForce::ZERO; n],
            s: vec![SpatialMotion::ZERO; n],
            s3: vec![[SpatialMotion::ZERO; 3]; n],
        }
    }
}

/// Compute `Xup[i]` for every link at the tree's current `q`. Callers that
/// need velocity/acceleration state add their own pass; CRB needs only this.
fn compute_xup(tree: &Tree) -> Vec<Xform> {
    let n = tree.links.len();
    let mut xup = vec![Xform::IDENTITY; n];
    for (i, link) in tree.links.iter().enumerate() {
        xup[i] = match link.joint {
            JointKind::Free => Xform::IDENTITY,
            JointKind::Fixed => xup_for_link(link, 0.0),
            JointKind::Hinge { axis, .. } => {
                let q_angle = tree.q[tree.q_offset[i]];
                xup_for_link_hinge(link, axis, q_angle)
            }
            JointKind::Slide { axis, .. } => {
                let q_slide = tree.q[tree.q_offset[i]];
                xup_for_link_slide(link, axis, q_slide)
            }
            JointKind::Ball { .. } => {
                let off = tree.q_offset[i];
                let q_ball = Quat::new(
                    tree.q[off],
                    tree.q[off + 1],
                    tree.q[off + 2],
                    tree.q[off + 3],
                );
                xup_for_link_ball(link, q_ball)
            }
        };
    }
    xup
}

/// Joint subspace `S` for a single-DOF hinge/slide, in child-body frame at
/// COM. `r_jc = joint_offset_in_child.translation`.
///
/// Hinge: `S = (axis, r_jc × axis)`. Slide: `S = (0, axis)`.
fn subspace_single(link: &crate::tree::Link) -> SpatialMotion {
    match link.joint {
        JointKind::Hinge { axis, .. } => {
            let r_jc = link.joint_offset_in_child.0;
            SpatialMotion::new(axis, r_jc.cross(axis))
        }
        JointKind::Slide { axis, .. } => SpatialMotion::new(Vec3::ZERO, axis),
        _ => SpatialMotion::ZERO,
    }
}

/// Joint subspace S for a ball joint (3 columns in child-body frame at COM).
/// `S_k = (e_k, r_jc × e_k)`.
fn subspace_ball(link: &crate::tree::Link) -> [SpatialMotion; 3] {
    let r_jc = link.joint_offset_in_child.0;
    [
        SpatialMotion::new(Vec3::X, r_jc.cross(Vec3::X)),
        SpatialMotion::new(Vec3::Y, r_jc.cross(Vec3::Y)),
        SpatialMotion::new(Vec3::Z, r_jc.cross(Vec3::Z)),
    ]
}

// ---------------------------------------------------------------------------
// RNE — inverse dynamics
// ---------------------------------------------------------------------------

/// Compute the generalized force vector `τ` (length `tree.nv()`) needed to
/// produce the given `qddot` at the tree's current `(q, qdot)` under
/// `gravity` and per-link `external_wrenches` (world-frame at each link's
/// COM — same shape as [`crate::tree::aba`]).
///
/// See the module docs for exactly what `τ` includes and excludes. In brief:
/// τ = `M(q) · qddot + h(q, qdot) − S_generalized(f_ext)`, where `f_ext`
/// combines `gravity` and `external_wrenches` in each link's body frame.
///
/// `qddot` layout matches [`crate::tree::Tree::qdot`]:
///
/// | joint | slot count | payload |
/// |-------|-----------|---------|
/// | Free  | 6 | spatial acceleration at COM in root's body frame, angular-then-linear |
/// | Fixed | 0 | — |
/// | Hinge | 1 | scalar angular acceleration (rad/s²) |
/// | Slide | 1 | scalar linear acceleration (m/s²) |
/// | Ball  | 3 | body-frame angular acceleration (rad/s²) |
///
/// The returned `τ` has the same layout. For the free root, `τ[0..6]` is a
/// body-frame spatial force at COM `(τ_ang, τ_lin)` = `(torque, linear)`.
pub fn inverse_dynamics(
    tree: &Tree,
    qddot: &[f32],
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
) -> Vec<f32> {
    let n = tree.links.len();
    assert_eq!(
        qddot.len(),
        tree.nv(),
        "qddot length {} != tree.nv() {}",
        qddot.len(),
        tree.nv()
    );
    assert_eq!(
        external_wrenches.len(),
        n,
        "external_wrenches length must equal number of links"
    );
    let poses = forward_kinematics(tree);
    let mut w = RneScratch::new(n);

    // --- Pass 1: root → leaves. compute Xup, v, a, f. ---
    for i in 0..n {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Free => {
                let voff = tree.v_offset[i];
                w.xup[i] = Xform::IDENTITY;
                w.v[i] = SpatialMotion::new(
                    Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]),
                    Vec3::new(
                        tree.qdot[voff + 3],
                        tree.qdot[voff + 4],
                        tree.qdot[voff + 5],
                    ),
                );
                w.a[i] = SpatialMotion::new(
                    Vec3::new(qddot[voff], qddot[voff + 1], qddot[voff + 2]),
                    Vec3::new(qddot[voff + 3], qddot[voff + 4], qddot[voff + 5]),
                );
            }
            JointKind::Fixed => {
                let xup = xup_for_link(link, 0.0);
                w.xup[i] = xup;
                let parent = link.parent;
                let v_parent = parent.map(|p| w.v[p]).unwrap_or(SpatialMotion::ZERO);
                let a_parent = parent.map(|p| w.a[p]).unwrap_or(SpatialMotion::ZERO);
                w.v[i] = xup.motion(v_parent);
                w.a[i] = xup.motion(a_parent);
            }
            JointKind::Hinge { axis, .. } => {
                let parent = link.parent.expect("hinge must have parent");
                let q_angle = tree.q[tree.q_offset[i]];
                let xup = xup_for_link_hinge(link, axis, q_angle);
                w.xup[i] = xup;
                let s = subspace_single(link);
                w.s[i] = s;
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let qddot_i = qddot[tree.v_offset[i]];
                let s_qdot = s * qdot_i;
                let s_qddot = s * qddot_i;
                let v_parent = xup.motion(w.v[parent]);
                let a_parent = xup.motion(w.a[parent]);
                w.v[i] = v_parent + s_qdot;
                let c = w.v[i].cross_motion(s_qdot);
                w.a[i] = a_parent + s_qddot + c;
            }
            JointKind::Slide { axis, .. } => {
                let parent = link.parent.expect("slide must have parent");
                let q_slide = tree.q[tree.q_offset[i]];
                let xup = xup_for_link_slide(link, axis, q_slide);
                w.xup[i] = xup;
                let s = subspace_single(link);
                w.s[i] = s;
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let qddot_i = qddot[tree.v_offset[i]];
                let s_qdot = s * qdot_i;
                let s_qddot = s * qddot_i;
                let v_parent = xup.motion(w.v[parent]);
                let a_parent = xup.motion(w.a[parent]);
                w.v[i] = v_parent + s_qdot;
                let c = w.v[i].cross_motion(s_qdot);
                w.a[i] = a_parent + s_qddot + c;
            }
            JointKind::Ball { .. } => {
                let parent = link.parent.expect("ball must have parent");
                let off = tree.q_offset[i];
                let q_ball = Quat::new(
                    tree.q[off],
                    tree.q[off + 1],
                    tree.q[off + 2],
                    tree.q[off + 3],
                );
                let xup = xup_for_link_ball(link, q_ball);
                w.xup[i] = xup;
                let s3 = subspace_ball(link);
                w.s3[i] = s3;
                let voff = tree.v_offset[i];
                let omega = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let alpha = Vec3::new(qddot[voff], qddot[voff + 1], qddot[voff + 2]);
                let s_qdot = s3[0] * omega.x + s3[1] * omega.y + s3[2] * omega.z;
                let s_qddot = s3[0] * alpha.x + s3[1] * alpha.y + s3[2] * alpha.z;
                let v_parent = xup.motion(w.v[parent]);
                let a_parent = xup.motion(w.a[parent]);
                w.v[i] = v_parent + s_qdot;
                let c = w.v[i].cross_motion(s_qdot);
                w.a[i] = a_parent + s_qddot + c;
            }
        }

        // f[i] = I[i] * a[i] + v[i] ×* (I[i] * v[i]) - f_ext[i]
        let si = link.spatial_inertia();
        let i_mat = Mat6::from_spatial_inertia(si);
        let i_a = i_mat.times_motion(w.a[i]);
        let iv = i_mat.times_motion(w.v[i]);
        let bias = w.v[i].cross_force(iv);

        // External wrench, world-frame at COM, rotated to body frame. Gravity
        // is an additive world-frame force at COM (m_i · g_world).
        let (_pos, ori) = poses[i];
        let (force_ext, torque_ext) = external_wrenches[i];
        let force_world_total = force_ext + gravity * link.mass;
        let force_body = ori.inverse_rotate(force_world_total);
        let torque_body = ori.inverse_rotate(torque_ext);
        let f_ext_body = SpatialForce::new(torque_body, force_body);

        w.f[i] = i_a + bias - f_ext_body;
    }

    // --- Pass 2: leaves → root. accumulate f into parents, extract τ. ---
    let mut tau = vec![0.0f32; tree.nv()];
    for i in (1..n).rev() {
        let link = &tree.links[i];
        let voff = tree.v_offset[i];
        match link.joint {
            JointKind::Fixed => {
                // No DOF; just propagate the wrench to the parent.
                let parent = link.parent.expect("fixed non-root must have parent");
                let pulled = w.xup[i].transpose_force(w.f[i]);
                w.f[parent] = w.f[parent] + pulled;
            }
            JointKind::Hinge { armature, .. } | JointKind::Slide { armature, .. } => {
                let parent = link.parent.unwrap();
                let s = w.s[i];
                let s_dot_f = spatial_dot_ms(s, w.f[i]);
                let qdd = qddot[voff];
                tau[voff] = s_dot_f + armature * qdd;
                let pulled = w.xup[i].transpose_force(w.f[i]);
                w.f[parent] = w.f[parent] + pulled;
            }
            JointKind::Ball { armature, .. } => {
                let parent = link.parent.unwrap();
                let s3 = w.s3[i];
                let s_dot_f = Vec3::new(
                    spatial_dot_ms(s3[0], w.f[i]),
                    spatial_dot_ms(s3[1], w.f[i]),
                    spatial_dot_ms(s3[2], w.f[i]),
                );
                let alpha = Vec3::new(qddot[voff], qddot[voff + 1], qddot[voff + 2]);
                tau[voff] = s_dot_f.x + armature * alpha.x;
                tau[voff + 1] = s_dot_f.y + armature * alpha.y;
                tau[voff + 2] = s_dot_f.z + armature * alpha.z;
                let pulled = w.xup[i].transpose_force(w.f[i]);
                w.f[parent] = w.f[parent] + pulled;
            }
            JointKind::Free => unreachable!("Free joint only allowed at root"),
        }
    }

    // Root DOFs. Free root: τ[0..6] = f[0] as body-frame spatial force
    // (torque, linear). Fixed root: no slots. Others rejected by push_link.
    if let JointKind::Free = tree.links[0].joint {
        let voff = tree.v_offset[0];
        tau[voff] = w.f[0].torque.x;
        tau[voff + 1] = w.f[0].torque.y;
        tau[voff + 2] = w.f[0].torque.z;
        tau[voff + 3] = w.f[0].linear.x;
        tau[voff + 4] = w.f[0].linear.y;
        tau[voff + 5] = w.f[0].linear.z;
    }

    tau
}

/// Coriolis + centrifugal + gravity vector `h(q, qdot)` for the tree at its
/// current `(q, qdot)`. Same as [`inverse_dynamics`] with `qddot = 0` and
/// no external wrenches. Length: `tree.nv()`.
pub fn bias_forces(tree: &Tree, gravity: Vec3) -> Vec<f32> {
    let qddot = vec![0.0f32; tree.nv()];
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()];
    inverse_dynamics(tree, &qddot, gravity, &ext)
}

/// Dense derivatives of the tree's explicit forward dynamics.
///
/// The acceleration is evaluated by [`crate::tree::aba`] with the current
/// tree state and external wrenches. Matrices use row-major storage:
/// `qacc_q` is `nv × nq`, `qacc_qvel` is `nv × nv`, and `qacc_ctrl` is
/// `nv × nu`, where `nu` is `tree.actuators.len()`.
///
/// The velocity and control paths differentiate the rigid-body bias and
/// actuator transmission directly. Position derivatives use the same
/// independent ABA evaluation at either side of each position coordinate.
/// This is the safe fallback at contact and quaternion branch boundaries;
/// callers should use a small perturbation around smooth states.
#[derive(Clone, Debug, PartialEq)]
pub struct Derivatives {
    /// Forward acceleration at the evaluated state.
    pub qacc: Vec<f32>,
    /// `∂qacc/∂q`, row-major `nv × nq`.
    pub qacc_q: Vec<f32>,
    /// `∂qacc/∂qvel`, row-major `nv × nv`.
    pub qacc_qvel: Vec<f32>,
    /// `∂qacc/∂ctrl`, row-major `nv × nu`.
    pub qacc_ctrl: Vec<f32>,
}

impl Derivatives {
    /// Number of generalized acceleration rows.
    pub fn nv(&self) -> usize {
        self.qacc.len()
    }

    /// Number of position columns.
    pub fn nq(&self) -> usize {
        self.qacc_q.len() / self.nv().max(1)
    }

    /// Number of actuator control columns.
    pub fn nu(&self) -> usize {
        self.qacc_ctrl.len() / self.nv().max(1)
    }
}

/// Compute dense explicit forward-dynamics derivatives for one tree.
pub fn derivatives(
    tree: &Tree,
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
) -> Derivatives {
    assert_eq!(
        external_wrenches.len(),
        tree.links.len(),
        "external_wrenches length must equal number of links"
    );
    let poses = forward_kinematics(tree);
    let qacc = crate::tree::aba(tree, &poses, gravity, external_wrenches);
    let nv = tree.nv();

    let mass = mass_matrix(tree);
    let qacc_qvel_force = qacc_qvel_force_jacobian(tree, &poses);
    let qacc_qvel = solve_mass_columns(&mass, nv, &qacc_qvel_force);
    let qacc_ctrl_force = qacc_ctrl_force_jacobian(tree, &poses);
    let qacc_ctrl = solve_mass_columns(&mass, nv, &qacc_ctrl_force);
    let qacc_q = finite_difference_qacc_q(tree, gravity, external_wrenches);

    Derivatives {
        qacc,
        qacc_q,
        qacc_qvel,
        qacc_ctrl,
    }
}

/// Central-difference fallback for position coordinates. This remains
/// independent of the velocity and control derivative paths above, so a
/// contact or quaternion branch can only affect its own matrix.
fn finite_difference_qacc_q(
    tree: &Tree,
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
) -> Vec<f32> {
    let nv = tree.nv();
    let nq = tree.nq();
    let mut out = vec![0.0; nv * nq];
    for column in 0..nq {
        let mut plus = tree.clone();
        let mut minus = tree.clone();
        let step = 1.0e-4 * plus.q[column].abs().max(1.0);
        plus.q[column] += step;
        minus.q[column] -= step;
        let plus_poses = forward_kinematics(&plus);
        let minus_poses = forward_kinematics(&minus);
        let plus_acc = crate::tree::aba(&plus, &plus_poses, gravity, external_wrenches);
        let minus_acc = crate::tree::aba(&minus, &minus_poses, gravity, external_wrenches);
        for row in 0..nv {
            out[row * nq + column] = (plus_acc[row] - minus_acc[row]) / (2.0 * step);
        }
    }
    out
}

/// Build `dτ/dqdot` for the explicit force balance
/// `M qacc = τ - h`. RNE supplies the analytic `dh/dqdot` term.
fn qacc_qvel_force_jacobian(tree: &Tree, poses: &[(Vec3, Quat)]) -> Vec<f32> {
    let nv = tree.nv();
    let mut force = vec![0.0; nv * nv];
    let dh = bias_velocity_jacobian(tree);
    for row in 0..nv {
        for column in 0..nv {
            force[row * nv + column] = -dh[row * nv + column];
        }
    }

    for (i, link) in tree.links.iter().enumerate() {
        let offset = tree.v_offset[i];
        match link.joint {
            JointKind::Free => {
                for slot in offset..offset + 6 {
                    force[slot * nv + slot] -= link.free_damping;
                }
            }
            JointKind::Fixed => {}
            JointKind::Hinge {
                damping,
                range,
                limit,
                ..
            }
            | JointKind::Slide {
                damping,
                range,
                limit,
                ..
            } => {
                force[offset * nv + offset] -= damping;
                if !tree.disable_penalty_limits {
                    force[offset * nv + offset] += joint_limit_velocity_derivative(
                        tree.q[tree.q_offset[i]],
                        tree.qdot[offset],
                        range,
                        limit,
                    );
                }
                let q = tree.q[tree.q_offset[i]];
                let qdot = tree.qdot[offset];
                for act in &tree.actuators {
                    if act.tendon_target.is_none() && act.link_idx == i {
                        force[offset * nv + offset] -= act.velocity_damping(q, qdot);
                    }
                }
            }
            JointKind::Ball { damping, .. } => {
                for slot in offset..offset + 3 {
                    force[slot * nv + slot] -= damping;
                }
            }
        }
    }

    if !tree.tendons.is_empty() {
        for (tendon_idx, tendon) in tree.tendons.iter().enumerate() {
            let kin = crate::tendon::tendon_kinematics(tendon, tree, poses);
            for row in 0..nv {
                for column in 0..nv {
                    let jj = kin.jacobian[row] * kin.jacobian[column];
                    if tendon.damping > 0.0 {
                        force[row * nv + column] -= tendon.damping * jj;
                    }
                    for act in &tree.actuators {
                        if act.tendon_target == Some(tendon_idx) {
                            force[row * nv + column] -=
                                act.velocity_damping(kin.length, kin.velocity) * jj;
                        }
                    }
                }
            }
        }
    }
    force
}

/// Build the generalized force columns from actuator controls.
fn qacc_ctrl_force_jacobian(tree: &Tree, poses: &[(Vec3, Quat)]) -> Vec<f32> {
    let nv = tree.nv();
    let nu = tree.actuators.len();
    let mut force = vec![0.0; nv * nu];
    let tendon_kinematics: Vec<_> = tree
        .tendons
        .iter()
        .map(|tendon| crate::tendon::tendon_kinematics(tendon, tree, poses))
        .collect();
    for (control, act) in tree.actuators.iter().enumerate() {
        let (len, vel) = if let Some(tid) = act.tendon_target {
            let kin = &tendon_kinematics[tid];
            (kin.length, kin.velocity)
        } else {
            let q = tree.q[tree.q_offset[act.link_idx]];
            let qdot = tree.qdot[tree.v_offset[act.link_idx]];
            (q, qdot)
        };
        let derivative = act.control_derivative(len, vel);
        if let Some(tid) = act.tendon_target {
            for (row, &jacobian) in tendon_kinematics[tid].jacobian.iter().enumerate() {
                force[row * nu + control] += jacobian * derivative;
            }
        } else {
            force[tree.v_offset[act.link_idx] * nu + control] += derivative;
        }
    }
    force
}

/// Solve `M X = rhs` for a row-major matrix whose columns are derivative
/// force directions. The returned matrix has the same column count.
fn solve_mass_columns(mass: &[f32], nv: usize, rhs: &[f32]) -> Vec<f32> {
    if nv == 0 {
        return Vec::new();
    }
    let columns = rhs.len() / nv;
    let factor = cholesky(mass, nv).expect("tree mass matrix must be positive definite");
    let mut out = vec![0.0; rhs.len()];
    for column in 0..columns {
        let mut b = vec![0.0; nv];
        for row in 0..nv {
            b[row] = rhs[row * columns + column];
        }
        let x = cholesky_solve(&factor, nv, &b);
        for row in 0..nv {
            out[row * columns + column] = x[row];
        }
    }
    out
}

/// Analytic velocity derivative of the RNE bias vector.
fn bias_velocity_jacobian(tree: &Tree) -> Vec<f32> {
    let nv = tree.nv();
    let n = tree.links.len();
    let xup = compute_xup(tree);
    let mut out = vec![0.0; nv * nv];
    for column in 0..nv {
        let mut v = vec![SpatialMotion::ZERO; n];
        let mut a = vec![SpatialMotion::ZERO; n];
        let mut dv = vec![SpatialMotion::ZERO; n];
        let mut da = vec![SpatialMotion::ZERO; n];
        let mut f = vec![SpatialForce::ZERO; n];
        let mut df = vec![SpatialForce::ZERO; n];
        for i in 0..n {
            let link = &tree.links[i];
            match link.joint {
                JointKind::Free => {
                    let offset = tree.v_offset[i];
                    v[i] = SpatialMotion::new(
                        Vec3::new(
                            tree.qdot[offset],
                            tree.qdot[offset + 1],
                            tree.qdot[offset + 2],
                        ),
                        Vec3::new(
                            tree.qdot[offset + 3],
                            tree.qdot[offset + 4],
                            tree.qdot[offset + 5],
                        ),
                    );
                    dv[i] = SpatialMotion::new(
                        Vec3::new(
                            basis(column, offset),
                            basis(column, offset + 1),
                            basis(column, offset + 2),
                        ),
                        Vec3::new(
                            basis(column, offset + 3),
                            basis(column, offset + 4),
                            basis(column, offset + 5),
                        ),
                    );
                }
                JointKind::Fixed => {
                    if let Some(parent) = link.parent {
                        v[i] = xup[i].motion(v[parent]);
                        dv[i] = xup[i].motion(dv[parent]);
                    }
                }
                JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                    let parent = link.parent.expect("single-DOF joint parent");
                    let s = subspace_single(link);
                    let qdot = tree.qdot[tree.v_offset[i]];
                    let sq = s * qdot;
                    v[i] = xup[i].motion(v[parent]) + sq;
                    let dq = if column == tree.v_offset[i] { 1.0 } else { 0.0 };
                    dv[i] = xup[i].motion(dv[parent]) + s * dq;
                    a[i] = xup[i].motion(a[parent]) + v[i].cross_motion(sq);
                    da[i] = xup[i].motion(da[parent])
                        + dv[i].cross_motion(sq)
                        + v[i].cross_motion(s * dq);
                }
                JointKind::Ball { .. } => {
                    let parent = link.parent.expect("ball joint parent");
                    let s3 = subspace_ball(link);
                    let offset = tree.v_offset[i];
                    let omega = Vec3::new(
                        tree.qdot[offset],
                        tree.qdot[offset + 1],
                        tree.qdot[offset + 2],
                    );
                    let sq = s3[0] * omega.x + s3[1] * omega.y + s3[2] * omega.z;
                    let domega = Vec3::new(
                        basis(column, offset),
                        basis(column, offset + 1),
                        basis(column, offset + 2),
                    );
                    let dsq = s3[0] * domega.x + s3[1] * domega.y + s3[2] * domega.z;
                    v[i] = xup[i].motion(v[parent]) + sq;
                    dv[i] = xup[i].motion(dv[parent]) + dsq;
                    a[i] = xup[i].motion(a[parent]) + v[i].cross_motion(sq);
                    da[i] =
                        xup[i].motion(da[parent]) + dv[i].cross_motion(sq) + v[i].cross_motion(dsq);
                }
            }
            let inertia = Mat6::from_spatial_inertia(link.spatial_inertia());
            let iv = inertia.times_motion(v[i]);
            let div = inertia.times_motion(dv[i]);
            f[i] = inertia.times_motion(a[i]) + v[i].cross_force(iv);
            df[i] = inertia.times_motion(da[i]) + dv[i].cross_force(iv) + v[i].cross_force(div);
        }
        let mut column_out = vec![0.0; nv];
        for i in (1..n).rev() {
            let link = &tree.links[i];
            match link.joint {
                JointKind::Fixed => {
                    let parent = link.parent.expect("fixed joint parent");
                    df[parent] = df[parent] + xup[i].transpose_force(df[i]);
                }
                JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                    let offset = tree.v_offset[i];
                    let parent = link.parent.expect("single-DOF joint parent");
                    column_out[offset] = spatial_dot_ms(subspace_single(link), df[i]);
                    df[parent] = df[parent] + xup[i].transpose_force(df[i]);
                }
                JointKind::Ball { .. } => {
                    let offset = tree.v_offset[i];
                    let parent = link.parent.expect("ball joint parent");
                    let s3 = subspace_ball(link);
                    for k in 0..3 {
                        column_out[offset + k] = spatial_dot_ms(s3[k], df[i]);
                    }
                    df[parent] = df[parent] + xup[i].transpose_force(df[i]);
                }
                JointKind::Free => unreachable!("free joint only allowed at root"),
            }
        }
        if matches!(tree.links[0].joint, JointKind::Free) {
            let offset = tree.v_offset[0];
            column_out[offset] = df[0].torque.x;
            column_out[offset + 1] = df[0].torque.y;
            column_out[offset + 2] = df[0].torque.z;
            column_out[offset + 3] = df[0].linear.x;
            column_out[offset + 4] = df[0].linear.y;
            column_out[offset + 5] = df[0].linear.z;
        }
        for row in 0..nv {
            out[row * nv + column] = column_out[row];
        }
    }
    out
}

fn joint_limit_velocity_derivative(
    q: f32,
    qdot: f32,
    range: Option<(f32, f32)>,
    limit: crate::joint::JointLimit,
) -> f32 {
    let Some((lo, hi)) = range else { return 0.0 };
    if (q < lo && qdot < 0.0) || (q > hi && qdot > 0.0) {
        -limit.damping
    } else {
        0.0
    }
}

#[inline]
fn basis(column: usize, slot: usize) -> f32 {
    if column == slot { 1.0 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// CRB — composite rigid body mass matrix
// ---------------------------------------------------------------------------

/// Dense `nv × nv` joint-space mass matrix `M(q)` at the tree's current `q`.
///
/// Row-major: `M[row * nv + col]`. Symmetric, positive-definite for any
/// physically valid tree (positive masses, positive-definite link inertias,
/// nonnegative armature, no zero-length root inertia). Includes each
/// hinge/slide `armature` on the scalar diagonal and each ball joint's
/// `armature · I₃` on its 3×3 block. Free-root: top-left 6×6 = `Ic[0]` as
/// [`Mat6`].
///
/// Featherstone RBDA §6, Table 6.2 (CRB algorithm). Pass 1 accumulates
/// composite spatial inertias bottom-up:
///
/// ```text
/// Ic[i] = I_link[i]  (initial)
/// for i in (n-1)..=1:  Ic[parent[i]] += Xup[i]^T Ic[i] Xup[i]
/// ```
///
/// Pass 2 fills each column of `M`:
///
/// ```text
/// for each joint i:
///     for each subspace column k of S[i]:
///         F = Ic[i] * S[i, k]
///         M[i_slot+k, i_slot+k'] = S[i, k']^T F   (diagonal block)
///         j = i
///         while j has parent:
///             F = Xup[j]^T F
///             j = parent[j]
///             M[j_slot+k'', i_slot+k] = S[j, k'']^T F   (off-diagonal)
///             M[i_slot+k, j_slot+k''] = same   (symmetry)
/// ```
///
/// For a free root, the walk-up terminates at `j = 0` with `S[0]` = the six
/// spatial basis vectors — the 6-row block of `F` (already in root body
/// frame after the pull-backs) is copied straight into
/// `M[0..6, i_slot+k]`.
pub fn mass_matrix(tree: &Tree) -> Vec<f32> {
    let n = tree.links.len();
    let nv = tree.nv();
    let mut m = vec![0.0f32; nv * nv];
    if nv == 0 {
        return m;
    }
    let xup = compute_xup(tree);

    // Composite inertias. Start with each link's rigid-body Mat6.
    let mut ic: Vec<Mat6> = (0..n)
        .map(|i| Mat6::from_spatial_inertia(tree.links[i].spatial_inertia()))
        .collect();
    // Pull children up into parents, bottom-up. Order: highest index first
    // (topological sort guarantees children have higher indices than parents).
    for i in (1..n).rev() {
        let link = &tree.links[i];
        if let Some(parent) = link.parent {
            let pulled = ic[i].pull_back(xup[i]);
            ic[parent] = ic[parent].plus(pulled);
        }
    }

    // Root's diagonal block. Free root: Ic[0] materialized as 6×6.
    if let JointKind::Free = tree.links[0].joint {
        let voff0 = tree.v_offset[0];
        for r in 0..6 {
            for c in 0..6 {
                m[(voff0 + r) * nv + (voff0 + c)] = ic[0].rows[r][c];
            }
        }
    }

    // Per-joint column fill. Walk each link's subspace columns up the parent
    // chain, filling M row-column pairs symmetrically. `i` indexes multiple
    // parallel structures (ic, tree.links, tree.v_offset) so an iterator over
    // just `ic` would double the noise here.
    #[allow(clippy::needless_range_loop)]
    for i in 1..n {
        let link = &tree.links[i];
        let voff_i = tree.v_offset[i];
        let subspace_cols: Vec<SpatialMotion> = match link.joint {
            JointKind::Fixed => continue,
            JointKind::Hinge { .. } | JointKind::Slide { .. } => vec![subspace_single(link)],
            JointKind::Ball { .. } => subspace_ball(link).to_vec(),
            JointKind::Free => unreachable!(),
        };
        let armature = joint_armature(&link.joint);

        for (k, s_ik) in subspace_cols.iter().copied().enumerate() {
            // F starts as Ic[i] * S[i, k] in link i's body frame.
            let mut f = ic[i].times_motion(s_ik);

            // Diagonal block within link i.
            for (k_prime, &s_ik_prime) in subspace_cols.iter().enumerate() {
                let val = spatial_dot_ms(s_ik_prime, f);
                m[(voff_i + k_prime) * nv + (voff_i + k)] = val;
            }
            // Add armature to the diagonal for hinge/slide/ball.
            m[(voff_i + k) * nv + (voff_i + k)] += armature;

            // Walk up parents, pulling F back through each Xup on the way.
            let mut j_child = i;
            while let Some(j) = tree.links[j_child].parent {
                f = xup[j_child].transpose_force(f);
                j_child = j;
                let voff_j = tree.v_offset[j];
                let parent_link = &tree.links[j];
                match parent_link.joint {
                    JointKind::Fixed => {
                        // No DOFs at this ancestor; just keep walking.
                    }
                    JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                        let s_jk = subspace_single(parent_link);
                        let val = spatial_dot_ms(s_jk, f);
                        m[(voff_j) * nv + (voff_i + k)] = val;
                        m[(voff_i + k) * nv + voff_j] = val;
                    }
                    JointKind::Ball { .. } => {
                        let s3 = subspace_ball(parent_link);
                        for (kk, s_jkk) in s3.iter().enumerate() {
                            let val = spatial_dot_ms(*s_jkk, f);
                            m[(voff_j + kk) * nv + (voff_i + k)] = val;
                            m[(voff_i + k) * nv + (voff_j + kk)] = val;
                        }
                    }
                    JointKind::Free => {
                        // Root DOFs = spatial basis. F is a spatial force in
                        // root body frame; the 6 dot products with the basis
                        // are exactly the 6 packed components.
                        let f_components = [
                            f.torque.x, f.torque.y, f.torque.z, f.linear.x, f.linear.y, f.linear.z,
                        ];
                        for (kk, &val) in f_components.iter().enumerate() {
                            m[(voff_j + kk) * nv + (voff_i + k)] = val;
                            m[(voff_i + k) * nv + (voff_j + kk)] = val;
                        }
                    }
                }
            }
        }
    }

    m
}

/// Return the joint's armature scalar (0 for Free/Fixed).
fn joint_armature(kind: &JointKind) -> f32 {
    match kind {
        JointKind::Hinge { armature, .. } => *armature,
        JointKind::Slide { armature, .. } => *armature,
        JointKind::Ball { armature, .. } => *armature,
        JointKind::Free | JointKind::Fixed => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Cholesky factor + solve
// ---------------------------------------------------------------------------

/// Cholesky factor `L` (lower triangular) of a symmetric positive-definite
/// `n × n` matrix stored row-major in `a`. Returns `None` if the matrix is
/// not positive-definite (a diagonal pivot is non-positive).
///
/// Output layout: the returned `L` is `n × n` row-major with zeros in the
/// strict upper triangle; `L L^T = A`. Purely hand-rolled (Cholesky-Banachiewicz
/// form): `L[i][j] = (A[i][j] - Σ_{k<j} L[i][k] L[j][k]) / L[j][j]`,
/// `L[i][i] = sqrt(A[i][i] - Σ_{k<i} L[i][k]²)`.
pub fn cholesky(a: &[f32], n: usize) -> Option<Vec<f32>> {
    assert_eq!(a.len(), n * n, "cholesky: matrix len {} != n²", a.len());
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L^T x = b` for `x`, given the Cholesky factor `L` (row-major,
/// lower triangular). Length `n`; `b.len() == n` required.
pub fn cholesky_solve(l: &[f32], n: usize, b: &[f32]) -> Vec<f32> {
    assert_eq!(l.len(), n * n);
    assert_eq!(b.len(), n);
    // Forward: L y = b.
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }
    // Back: L^T x = y.
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn cholesky_recovers_identity() {
        let a = vec![
            4.0, 12.0, -16.0, //
            12.0, 37.0, -43.0, //
            -16.0, -43.0, 98.0,
        ];
        let l = cholesky(&a, 3).unwrap();
        // Reference from RBDA / Wikipedia: L = [[2,0,0],[6,1,0],[-8,5,3]]
        assert!(approx(l[0], 2.0, 1e-5));
        assert!(approx(l[3], 6.0, 1e-5));
        assert!(approx(l[4], 1.0, 1e-5));
        assert!(approx(l[6], -8.0, 1e-5));
        assert!(approx(l[7], 5.0, 1e-5));
        assert!(approx(l[8], 3.0, 1e-5));
        // solve A x = b for b = [1,2,3]; verify by A x back.
        let b = vec![1.0, 2.0, 3.0];
        let x = cholesky_solve(&l, 3, &b);
        // A x should equal b.
        let n = 3;
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * x[k];
            }
            assert!(approx(s, b[i], 1e-3), "row {i}: {s} vs {}", b[i]);
        }
    }

    #[test]
    fn cholesky_rejects_non_pd() {
        // Diagonal has a zero: 0-pivot triggers None.
        let a = vec![0.0, 0.0, 0.0, 1.0];
        assert!(cholesky(&a, 2).is_none());
        // Negative diagonal.
        let a = vec![-1.0, 0.0, 0.0, 1.0];
        assert!(cholesky(&a, 2).is_none());
    }

    /// The initial diagonal check catches a plainly non-PD matrix. This one
    /// slips past that — every diagonal entry is +1 > 0 — but the FIRST
    /// Schur update makes the (1, 1) pivot negative:
    /// `A[1][1] − L[1][0]² = 1 − 2² = −3`.
    /// A cholesky that only guarded `A[i][i] > 0` up front would sqrt a
    /// negative and produce a NaN factor; the loop's per-row `sum <= 0`
    /// guard is what makes this case return `None`.
    #[test]
    fn cholesky_rejects_negative_schur_pivot() {
        let a = vec![
            1.0, 2.0, //
            2.0, 1.0,
        ];
        assert!(
            cholesky(&a, 2).is_none(),
            "matrix with positive diag but negative Schur pivot must be rejected"
        );
    }
}
