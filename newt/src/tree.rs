//! Kinematic tree: articulated multi-body system with generalized
//! coordinates, forward kinematics, and Featherstone's Articulated Body
//! Algorithm (ABA) for O(n) forward dynamics.
//!
//! # Model
//!
//! A [`Tree`] holds an ordered list of [`Link`]s. Link 0 is always the root
//! (its `parent` is `None`) and carries a [`crate::joint::JointKind::Free`]
//! or [`crate::joint::JointKind::Fixed`] joint. Non-root links have a
//! [`crate::joint::JointKind::Hinge`] joint connecting them to a parent
//! whose index must be strictly less than their own — the tree is stored
//! topologically sorted so a single pass in the vector order visits the
//! root before any of its children. Branching is allowed (multiple children
//! per parent); loops are NOT (no closed kinematic chains in v0).
//!
//! Each link's body frame origin sits at the link's COM (same convention as
//! tier-1 `Body`), and the inertia tensor `inertia_body` is expressed in
//! that frame about the COM. Joint offsets place the joint anchor relative
//! to the parent's and child's body-frame origins.
//!
//! # State layout (MuJoCo-style dense q / qdot)
//!
//! `q` and `qdot` are dense vectors laid out by joint order. Slot counts:
//!   - Free root: `nq = 7` (px, py, pz, qx, qy, qz, qw), `nv = 6`
//!     (angular_body, linear_body). Angular components are body-frame; the
//!     linear components are body-frame COM velocity too so the whole 6-vector
//!     is a spatial motion at the COM expressed in the root's body frame.
//!   - Fixed root: `nq = nv = 0`. Root pose is fixed at
//!     `joint_offset_in_parent` (used as the world-frame anchor pose).
//!   - Hinge: `nq = nv = 1`. Angle (rad) / rate (rad/s).
//!
//! `qfrc_applied` is a dense `nv`-vector of user-applied generalized forces
//! (tier-4 actuators plug in here). Zero-initialized.
//!
//! # ABA in body-frame coordinates
//!
//! Every per-link spatial vector is expressed in that link's own body frame
//! at the COM. Plücker transforms `Xup[i]` move motion from the parent
//! body frame to child body frame; force pull-backs use `Xup[i].transpose_force`
//! (see [`crate::spatial::Xform::transpose_force`]).
//!
//! - Pass 1 (root → leaves): forward kinematics + `v[i] = Xup[i] * v[parent]
//!   + S[i] * qdot[i]`; joint-motion bias `c[i] = v[i] × (S[i] * qdot[i])`.
//! - Pass 2 (leaves → root): articulated inertia `IA[i]` and bias `pA[i]`
//!   accumulated bottom-up with the classical `IA -= (IA S)(Sᵀ IA S + arm)⁻¹
//!   (IA S)ᵀ` rank-1 update.
//! - Pass 3 (root → leaves): solve for the root spatial acceleration (6x6
//!   solve for free root; zero for fixed root), then propagate `qddot[i] =
//!   (Sᵀ IA S + arm)⁻¹ (τ[i] − Sᵀ (IA (Xup a[parent] + c[i]) + pA[i]))` and
//!   `a[i] = Xup a[parent] + S qddot[i] + c[i]`.
//!
//! Gravity is applied as an external world-frame force `m_i * g` at each
//! link's COM (rotated into body coords before entering `pA`). Contact
//! wrenches from tier 2 enter the same way. This keeps the free-root path
//! symmetric with the fixed-root path — no special "base-acceleration =
//! −g" trick.
//!
//! # Determinism
//!
//! Links are stored in topological order (parent index strictly less than
//! child index). All passes iterate that order deterministically; per-parent
//! contribution accumulations are additive so the child order does not
//! affect the result.

use crate::actuator::{PdServo, clamp_symmetric};
use crate::joint::{HingeLimit, JointKind};
use crate::math::{Mat3, Quat, Vec3};
use crate::spatial::{Mat6, SpatialForce, SpatialInertia, SpatialMotion, Xform};

/// One link in a kinematic tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    /// Parent link index in the containing [`Tree`]. Must be strictly less
    /// than this link's own index (topological order). `None` iff this is
    /// the root (index 0).
    pub parent: Option<usize>,

    /// Joint connecting this link to its parent (or to the world at the
    /// root). Must be [`JointKind::Free`] or [`JointKind::Fixed`] when
    /// `parent` is `None`; must be [`JointKind::Hinge`] (or `Fixed`, in
    /// principle) when `parent` is `Some`.
    pub joint: JointKind,

    /// Joint anchor pose in the parent's body frame. For the root with a
    /// `Fixed` joint, this is the world-frame anchor. For the root with a
    /// `Free` joint, this is ignored (initial pose comes from `q`).
    ///
    /// v0 convention: `orientation = IDENTITY` (joint frame aligns with
    /// parent body axes at `q = 0`). Non-identity orientation is on the
    /// v1 roadmap.
    pub joint_offset_in_parent: (Vec3, Quat),

    /// Joint anchor pose in this link's body frame. v0 convention:
    /// `orientation = IDENTITY`.
    pub joint_offset_in_child: (Vec3, Quat),

    /// Mass in kg.
    pub mass: f32,

    /// Inertia tensor about the COM in body-frame axes.
    pub inertia_body: Mat3,

    /// Precomputed inverse of `inertia_body`.
    pub inertia_body_inverse: Mat3,
}

impl Link {
    /// Build a link with an explicit inertia tensor. Panics if the inertia
    /// is not invertible.
    pub fn new(
        parent: Option<usize>,
        joint: JointKind,
        joint_offset_in_parent: (Vec3, Quat),
        joint_offset_in_child: (Vec3, Quat),
        mass: f32,
        inertia_body: Mat3,
    ) -> Self {
        // v0 convention for joint-frame anchors: the joint anchor frame in
        // both parent and child body coords must have IDENTITY orientation.
        // Non-identity orientations are silently ignored by `xup_for_link`
        // (it reads only the translation components), so a caller who passes
        // a rotated offset would get subtly wrong dynamics. The v1 lift will
        // thread these quats into `Xup` and drop this assert.
        //
        // Exemption: at the root, `joint_offset_in_parent` is the world-frame
        // anchor pose (see `push_link` — for Free it becomes q[0..7]; for
        // Fixed it is the fixed world pose per the field docs), not a
        // joint-frame offset, so any orientation is meaningful there.
        let is_root = parent.is_none();
        debug_assert!(
            is_root || joint_offset_in_parent.1 == Quat::IDENTITY,
            "v0: joint_offset_in_parent.orientation must be IDENTITY (v1 will lift this)"
        );
        debug_assert!(
            joint_offset_in_child.1 == Quat::IDENTITY,
            "v0: joint_offset_in_child.orientation must be IDENTITY (v1 will lift this)"
        );
        let inertia_body_inverse = inertia_body
            .inverse()
            .expect("link inertia tensor must be invertible");
        Self {
            parent,
            joint,
            joint_offset_in_parent,
            joint_offset_in_child,
            mass,
            inertia_body,
            inertia_body_inverse,
        }
    }

    /// Body-frame spatial inertia at the COM (v0 convention: link origin =
    /// COM, so the spatial-inertia COM offset is zero).
    pub fn spatial_inertia(&self) -> SpatialInertia {
        SpatialInertia::new(self.mass, Vec3::ZERO, self.inertia_body)
    }
}

/// Kinematic tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Tree {
    /// Links in topological order. `links[0]` is the root.
    pub links: Vec<Link>,

    /// Offsets into `q` for each link. `q_offset[i]` gives the first slot of
    /// link `i`'s joint state in `q`.
    pub q_offset: Vec<usize>,
    /// Offsets into `qdot`/`qfrc_applied` for each link.
    pub v_offset: Vec<usize>,

    /// Generalized position vector; layout follows [`JointKind::nq`].
    pub q: Vec<f32>,
    /// Generalized velocity vector; layout follows [`JointKind::nv`].
    pub qdot: Vec<f32>,
    /// User-applied generalized forces; layout matches `qdot`. Persists
    /// across steps until the caller changes it — the tier-4 direct-torque
    /// helpers (see [`Tree::set_joint_torque_clamped`]) write here.
    pub qfrc_applied: Vec<f32>,

    /// PD position servos (tier 4). Persistent; targets settable per step via
    /// [`Tree::set_actuator_target`]. Each is bound to a hinge link.
    pub actuators: Vec<PdServo>,
    /// Per-link user-applied world-frame wrenches at each link's COM,
    /// `(force_world, torque_world)`. Length equals `links.len()`; grows
    /// automatically on [`Tree::push_link`]. Sums with contact wrenches
    /// inside [`aba`] — no special-casing per link kind.
    pub applied_wrenches: Vec<(Vec3, Vec3)>,
}

impl Tree {
    /// Empty tree (no links). Add the root first via [`Tree::push_link`].
    pub fn new() -> Self {
        Self {
            links: Vec::new(),
            q_offset: Vec::new(),
            v_offset: Vec::new(),
            q: Vec::new(),
            qdot: Vec::new(),
            qfrc_applied: Vec::new(),
            actuators: Vec::new(),
            applied_wrenches: Vec::new(),
        }
    }

    /// Append a link and grow `q`/`qdot`/`qfrc_applied` accordingly. Returns
    /// the new link's index. Root joints (`Free`/`Fixed`) get their default
    /// `q` written into the position slots.
    pub fn push_link(&mut self, link: Link) -> usize {
        let idx = self.links.len();
        if idx == 0 {
            assert!(
                link.parent.is_none(),
                "root link (index 0) must have parent = None"
            );
            // Only Free / Fixed are valid at the root — hinge or any future
            // joint kind at the root would reach an `unreachable!` deep inside
            // `aba`, so reject it here where the failure message points at
            // the actual bug.
            assert!(
                matches!(link.joint, JointKind::Free | JointKind::Fixed),
                "root joint must be Free or Fixed (got {:?})",
                link.joint
            );
        } else {
            let parent = link.parent.expect("non-root link must have Some(parent)");
            assert!(
                parent < idx,
                "parent index {parent} must be < link index {idx} (topological order)"
            );
        }
        let nv = link.joint.nv();
        let q_off = self.q.len();
        let v_off = self.qdot.len();
        self.q_offset.push(q_off);
        self.v_offset.push(v_off);
        // Extend q with default position for this joint.
        match link.joint {
            JointKind::Free => {
                // Position = joint_offset_in_parent.translation, orientation
                // = joint_offset_in_parent.orientation. Free-root initial
                // pose comes from the same field for convenience.
                let (p, ori) = link.joint_offset_in_parent;
                self.q
                    .extend_from_slice(&[p.x, p.y, p.z, ori.x, ori.y, ori.z, ori.w]);
            }
            JointKind::Fixed => {}
            JointKind::Hinge { .. } => {
                self.q.push(0.0);
            }
        }
        self.qdot.extend(std::iter::repeat_n(0.0, nv));
        self.qfrc_applied.extend(std::iter::repeat_n(0.0, nv));
        self.applied_wrenches.push((Vec3::ZERO, Vec3::ZERO));
        self.links.push(link);
        idx
    }

    /// Attach a PD position actuator to a hinge link. Returns the actuator's
    /// stable index (usable with [`Tree::set_actuator_target`]). Panics if
    /// `servo.link_idx` is out of range or does not reference a hinge.
    pub fn add_actuator(&mut self, servo: PdServo) -> usize {
        assert!(
            servo.link_idx < self.links.len(),
            "actuator link out of range"
        );
        assert!(
            matches!(self.links[servo.link_idx].joint, JointKind::Hinge { .. }),
            "PD servo can only actuate a Hinge joint (link {} is {:?})",
            servo.link_idx,
            self.links[servo.link_idx].joint
        );
        let idx = self.actuators.len();
        self.actuators.push(servo);
        idx
    }

    /// Set the target angle of a previously-added actuator. Panics on
    /// out-of-range index.
    pub fn set_actuator_target(&mut self, actuator_idx: usize, target: f32) {
        self.actuators[actuator_idx].target = target;
    }

    /// Directly write a generalized joint torque into `qfrc_applied` for a
    /// hinge link, symmetric-clamped to `force_range` (pass `0.0` or a
    /// negative value to disable the clamp). This is the motor-style input
    /// tier-4 exposes on top of the raw `qfrc_applied` buffer. Persists
    /// across steps — call again to update, or use
    /// [`Tree::clear_qfrc_applied`] to zero every slot.
    ///
    /// Panics if `link_idx` is not a hinge.
    pub fn set_joint_torque_clamped(&mut self, link_idx: usize, torque: f32, force_range: f32) {
        assert!(
            matches!(self.links[link_idx].joint, JointKind::Hinge { .. }),
            "set_joint_torque_clamped requires a Hinge link"
        );
        self.qfrc_applied[self.v_offset[link_idx]] = clamp_symmetric(torque, force_range);
    }

    /// Zero every entry of `qfrc_applied`.
    pub fn clear_qfrc_applied(&mut self) {
        for x in &mut self.qfrc_applied {
            *x = 0.0;
        }
    }

    /// Set the user-applied external wrench on a link, expressed in world
    /// coordinates at the link's COM. Overwrites any prior value at that
    /// link. Persists across steps — call [`Tree::clear_applied_wrenches`]
    /// to reset every link to zero.
    pub fn set_link_wrench(&mut self, link_idx: usize, force_world: Vec3, torque_world: Vec3) {
        self.applied_wrenches[link_idx] = (force_world, torque_world);
    }

    /// Zero every link's applied wrench.
    pub fn clear_applied_wrenches(&mut self) {
        for w in &mut self.applied_wrenches {
            *w = (Vec3::ZERO, Vec3::ZERO);
        }
    }

    /// Total number of position slots.
    pub fn nq(&self) -> usize {
        self.q.len()
    }

    /// Total number of velocity/force slots.
    pub fn nv(&self) -> usize {
        self.qdot.len()
    }

    /// Overwrite the hinge angle for link `i`. Panics if `i`'s joint is not
    /// a hinge.
    pub fn set_hinge_angle(&mut self, i: usize, angle: f32) {
        assert!(matches!(self.links[i].joint, JointKind::Hinge { .. }));
        let off = self.q_offset[i];
        self.q[off] = angle;
    }

    /// Overwrite the hinge rate for link `i`.
    pub fn set_hinge_rate(&mut self, i: usize, rate: f32) {
        assert!(matches!(self.links[i].joint, JointKind::Hinge { .. }));
        let off = self.v_offset[i];
        self.qdot[off] = rate;
    }

    /// Read the hinge angle at link `i`.
    pub fn hinge_angle(&self, i: usize) -> f32 {
        assert!(matches!(self.links[i].joint, JointKind::Hinge { .. }));
        self.q[self.q_offset[i]]
    }

    /// Read the hinge rate at link `i`.
    pub fn hinge_rate(&self, i: usize) -> f32 {
        assert!(matches!(self.links[i].joint, JointKind::Hinge { .. }));
        self.qdot[self.v_offset[i]]
    }

    /// Set the free-root pose (position + orientation) for link 0. Panics if
    /// the root joint is not Free.
    pub fn set_free_root_pose(&mut self, position: Vec3, orientation: Quat) {
        assert!(matches!(self.links[0].joint, JointKind::Free));
        self.q[0] = position.x;
        self.q[1] = position.y;
        self.q[2] = position.z;
        self.q[3] = orientation.x;
        self.q[4] = orientation.y;
        self.q[5] = orientation.z;
        self.q[6] = orientation.w;
    }

    /// Set the free-root spatial velocity for link 0 (body-frame,
    /// angular-then-linear at COM).
    pub fn set_free_root_velocity(&mut self, twist_body: SpatialMotion) {
        assert!(matches!(self.links[0].joint, JointKind::Free));
        self.qdot[0] = twist_body.angular.x;
        self.qdot[1] = twist_body.angular.y;
        self.qdot[2] = twist_body.angular.z;
        self.qdot[3] = twist_body.linear.x;
        self.qdot[4] = twist_body.linear.y;
        self.qdot[5] = twist_body.linear.z;
    }

    /// World pose of a link. Position is COM in world; orientation is body →
    /// world. Computed via a single walk over the topological order.
    pub fn link_pose(&self, i: usize) -> (Vec3, Quat) {
        forward_kinematics(self)[i]
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Forward kinematics
// ---------------------------------------------------------------------------

/// World-frame COM position and body → world orientation for every link.
/// The returned vector is indexed by link index.
pub fn forward_kinematics(tree: &Tree) -> Vec<(Vec3, Quat)> {
    let mut out: Vec<(Vec3, Quat)> = Vec::with_capacity(tree.links.len());
    for (i, link) in tree.links.iter().enumerate() {
        let pose = match link.joint {
            JointKind::Free => {
                assert!(link.parent.is_none(), "free joint only supported on root");
                let off = tree.q_offset[i];
                let p = Vec3::new(tree.q[off], tree.q[off + 1], tree.q[off + 2]);
                let ori = Quat::new(
                    tree.q[off + 3],
                    tree.q[off + 4],
                    tree.q[off + 5],
                    tree.q[off + 6],
                );
                (p, ori)
            }
            JointKind::Fixed => {
                let (offset_p, offset_o) = link.joint_offset_in_parent;
                let (offset_c_p, offset_c_o) = link.joint_offset_in_child;
                let (parent_pos, parent_ori) = match link.parent {
                    Some(p) => out[p],
                    None => (Vec3::ZERO, Quat::IDENTITY),
                };
                // Joint anchor in world:
                let joint_pos_world = parent_pos + parent_ori.rotate(offset_p);
                let joint_ori_world = parent_ori * offset_o;
                // Child body pose: COM offset such that offset_c is the joint
                // anchor position in child body coords.
                let child_ori = joint_ori_world * offset_c_o.conjugate();
                let child_pos = joint_pos_world - child_ori.rotate(offset_c_p);
                (child_pos, child_ori)
            }
            JointKind::Hinge { axis, .. } => {
                let parent_idx = link.parent.expect("hinge joint must have a parent");
                let (parent_pos, parent_ori) = out[parent_idx];
                let q_angle = tree.q[tree.q_offset[i]];
                let (offset_p, _offset_p_o) = link.joint_offset_in_parent;
                let (offset_c_p, _offset_c_o) = link.joint_offset_in_child;
                // Joint frame in world at q=0: same as parent-anchor pose.
                let joint_pos_world = parent_pos + parent_ori.rotate(offset_p);
                // Apply hinge rotation about the axis (axis expressed in the
                // parent body frame; equivalently in the joint frame with
                // R_pj = IDENTITY).
                let joint_rot = Quat::from_axis_angle(axis, q_angle);
                let child_ori = parent_ori * joint_rot;
                // Position child so its joint anchor coincides with the
                // world joint position:
                let child_pos = joint_pos_world - child_ori.rotate(offset_c_p);
                (child_pos, child_ori)
            }
        };
        out.push(pose);
    }
    out
}

// ---------------------------------------------------------------------------
// ABA — Featherstone's Articulated Body Algorithm
// ---------------------------------------------------------------------------

/// Per-link scratch used by the ABA passes.
struct AbaWorkspace {
    /// Motion transform from parent body frame to this link's body frame.
    xup: Vec<Xform>,
    /// Joint subspace basis (child-body-frame spatial motion per unit qdot).
    /// Only meaningful for hinge/free joints; empty for fixed.
    s: Vec<SpatialMotion>,
    /// Per-link spatial velocity in body frame at COM.
    v: Vec<SpatialMotion>,
    /// Per-link coriolis bias `c[i] = v[i] × (S[i] * qdot[i])`.
    c: Vec<SpatialMotion>,
    /// Articulated-body inertia at each link (body frame at COM).
    ia: Vec<Mat6>,
    /// Articulated-body bias force at each link.
    pa: Vec<SpatialForce>,
    /// Per-link `IA[i] * S[i]` (cached from pass 2 for reuse in pass 3).
    ia_s: Vec<SpatialForce>,
    /// Per-link `Sᵀ IA S + armature` (scalar for hinge).
    d: Vec<f32>,
    /// Per-link joint scalar torque applied at pass-2 (damping, armature-related,
    /// limits, external qfrc_applied). Cached for pass 3.
    tau: Vec<f32>,
    /// Per-link joint qddot (computed in pass 3).
    qddot_joint: Vec<f32>,
    /// Per-link spatial acceleration (computed in pass 3, body frame at COM).
    a: Vec<SpatialMotion>,
}

impl AbaWorkspace {
    fn new(n: usize) -> Self {
        Self {
            xup: vec![Xform::IDENTITY; n],
            s: vec![SpatialMotion::ZERO; n],
            v: vec![SpatialMotion::ZERO; n],
            c: vec![SpatialMotion::ZERO; n],
            ia: vec![Mat6::ZERO; n],
            pa: vec![SpatialForce::ZERO; n],
            ia_s: vec![SpatialForce::ZERO; n],
            d: vec![0.0; n],
            tau: vec![0.0; n],
            qddot_joint: vec![0.0; n],
            a: vec![SpatialMotion::ZERO; n],
        }
    }
}

/// External wrench on each link, in WORLD coordinates at the link's COM.
/// `ext[i] = (force_world, torque_world_about_com)`.
pub type ExternalWrenches = Vec<(Vec3, Vec3)>;

/// Compute the generalized acceleration `qddot` for the tree at its current
/// state `(q, qdot, qfrc_applied)` under gravity + `external_wrenches`.
///
/// `external_wrenches[i]` is expressed in world coordinates at link `i`'s
/// COM (this matches the format contacts already produce for free bodies).
/// The returned vector has length `tree.nv()`.
pub fn aba(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
) -> Vec<f32> {
    let n = tree.links.len();
    assert_eq!(poses.len(), n);
    assert_eq!(external_wrenches.len(), n);
    let mut w = AbaWorkspace::new(n);

    // --- Pass 1: compute Xup, S, v, c bottom-down. ---
    for i in 0..n {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Free => {
                // Root free joint: v[0] is the stored twist (body frame at
                // COM). Xup is identity (no parent transform), S is unused
                // since the root's contribution comes from solving the 6x6
                // in pass 3.
                let off = tree.v_offset[i];
                let twist = SpatialMotion::new(
                    Vec3::new(tree.qdot[off], tree.qdot[off + 1], tree.qdot[off + 2]),
                    Vec3::new(tree.qdot[off + 3], tree.qdot[off + 4], tree.qdot[off + 5]),
                );
                w.xup[i] = Xform::IDENTITY;
                w.v[i] = twist;
                w.c[i] = SpatialMotion::ZERO;
            }
            JointKind::Fixed => {
                let parent = link.parent;
                let xup = xup_for_link(link, 0.0);
                w.xup[i] = xup;
                let parent_v = parent.map(|p| w.v[p]).unwrap_or(SpatialMotion::ZERO);
                w.v[i] = xup.motion(parent_v);
                w.c[i] = SpatialMotion::ZERO;
            }
            JointKind::Hinge { axis, .. } => {
                let parent = link.parent.expect("hinge must have parent");
                let q_angle = tree.q[tree.q_offset[i]];
                let xup = xup_for_link(link, q_angle);
                w.xup[i] = xup;
                // Joint subspace in child body frame at COM: (axis, r_jc × axis)
                // where r_jc = joint_offset_in_child.translation.
                let r_jc = link.joint_offset_in_child.0;
                let s = SpatialMotion::new(axis, r_jc.cross(axis));
                w.s[i] = s;
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let s_qdot = s * qdot_i;
                let v_parent = xup.motion(w.v[parent]);
                w.v[i] = v_parent + s_qdot;
                w.c[i] = w.v[i].cross_motion(s_qdot);
            }
        }
    }

    // --- Pass 2: leaves→root, accumulate IA and pA. ---
    // Initialize each link's IA = spatial inertia and pA = velocity-product bias.
    let mut ext_body: Vec<SpatialForce> = Vec::with_capacity(n);
    for i in 0..n {
        let link = &tree.links[i];
        let si = link.spatial_inertia();
        let ia_i = Mat6::from_spatial_inertia(si);
        w.ia[i] = ia_i;
        // pA = v × I v − f_ext. f_ext includes gravity + external wrench
        // (contacts) + persistent link-attached wrench (`Tree::applied_wrenches`,
        // tier-4). All expressed in body frame at COM. Both wrench channels
        // are world-frame at the link's COM, so they sum trivially before
        // the frame rotation.
        let (_pos, ori) = poses[i];
        let (force_world_ext, torque_world_ext) = external_wrenches[i];
        let (force_world_applied, torque_world_applied) = tree.applied_wrenches[i];
        let force_world = force_world_ext + force_world_applied;
        let torque_world = torque_world_ext + torque_world_applied;
        // Add gravity as world-frame force at COM.
        let force_world_total = force_world + gravity * link.mass;
        // Rotate world-frame wrench into body frame.
        let force_body = ori.inverse_rotate(force_world_total);
        let torque_body = ori.inverse_rotate(torque_world);
        // Package as body-frame spatial force at COM (torque, linear).
        let f_ext_body = SpatialForce::new(torque_body, force_body);
        ext_body.push(f_ext_body);

        let iv = ia_i.times_motion(w.v[i]);
        let bias = w.v[i].cross_force(iv);
        w.pa[i] = bias - f_ext_body;
    }
    // Walk leaves→root. Because links are topologically sorted, iterating
    // from the highest index down covers children before parents.
    for i in (1..n).rev() {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Fixed => {
                // No joint DOF: propagate IA and pA to parent unchanged.
                let parent = link.parent.expect("fixed non-root must have parent");
                let ia_parent_contrib = w.ia[i].pull_back(w.xup[i]);
                w.ia[parent] = w.ia[parent].plus(ia_parent_contrib);
                // pA parent contribution: Xᵀ * (pA + IA * c[i]) — c is zero
                // for fixed joints so this reduces to Xᵀ * pA.
                let pa_child_total = w.pa[i] + w.ia[i].times_motion(w.c[i]);
                let pa_parent_contrib = w.xup[i].transpose_force(pa_child_total);
                w.pa[parent] = w.pa[parent] + pa_parent_contrib;
            }
            JointKind::Hinge {
                damping,
                armature,
                range,
                limit,
                ..
            } => {
                let parent = link.parent.expect("hinge must have parent");
                let s = w.s[i];
                let ia_s = w.ia[i].times_motion(s);
                // Sᵀ (IA S) + armature = scalar for a hinge.
                let d_scalar = spatial_dot_ms(s, ia_s) + armature;
                // τ_effective = qfrc_applied − damping * qdot + limit penalty
                //                + Σ actuator torques on this link.
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let q_i = tree.q[tree.q_offset[i]];
                let tau_lim = hinge_limit_torque(q_i, qdot_i, range, limit);
                // Sum every PD actuator bound to this link. Small linear scan;
                // v0 tree sizes are tiny (biped ≈ 10 hinges). Deterministic —
                // order-independent because it's a sum of scalars.
                let mut tau_act = 0.0;
                for act in &tree.actuators {
                    if act.link_idx == i {
                        tau_act += act.torque(q_i, qdot_i);
                    }
                }
                let tau_scalar =
                    tree.qfrc_applied[tree.v_offset[i]] - damping * qdot_i + tau_lim + tau_act;
                // Featherstone's reduced-inertia form: pA_reduced = pA + I_a c
                // + U u / D with I_a = IA - U D⁻¹ Uᵀ. Expanding I_a c and
                // grouping gives the equivalent form we use here (avoids
                // materializing I_a * c separately):
                //
                //     p_stage = pA + IA c
                //     u_stage = τ − Sᵀ p_stage    (= u − Sᵀ IA c)
                //     pA_reduced = p_stage + IA S · u_stage / D
                let ia_c = w.ia[i].times_motion(w.c[i]);
                let p_stage = w.pa[i] + ia_c;
                let s_dot_p_stage = spatial_dot_ms(s, p_stage);
                let u_stage = tau_scalar - s_dot_p_stage;
                let pa_full = p_stage + ia_s * (u_stage / d_scalar);

                // IA update within child: IA -= (IA S)(IA S)ᵀ / d
                let outer = Mat6::outer(ia_s, s_force_to_motion(ia_s));
                let ia_full = w.ia[i].minus(scale_mat6(outer, 1.0 / d_scalar));

                // Cache for pass 3. `tau` is the raw applied joint torque
                // (damping + limits + qfrc_applied). Pass 3 recomputes u
                // relative to the parent-driven acceleration a′ so it uses
                // this raw τ, NOT `u_stage`.
                w.ia_s[i] = ia_s;
                w.d[i] = d_scalar;
                w.tau[i] = tau_scalar;

                // Propagate to parent via Xup pull-back.
                let ia_parent_contrib = ia_full.pull_back(w.xup[i]);
                w.ia[parent] = w.ia[parent].plus(ia_parent_contrib);
                let pa_parent_contrib = w.xup[i].transpose_force(pa_full);
                w.pa[parent] = w.pa[parent] + pa_parent_contrib;
            }
            JointKind::Free => unreachable!("free joint only allowed at root"),
        }
    }

    // --- Pass 3: root → leaves. ---
    // Solve for root acceleration a[0].
    let mut qddot = vec![0.0; tree.nv()];
    match tree.links[0].joint {
        JointKind::Free => {
            // At root: IA[0] a[0] = -pA[0]. Solve 6x6.
            let rhs = SpatialForce::new(-w.pa[0].torque, -w.pa[0].linear);
            let a0 = w.ia[0]
                .solve(rhs)
                .expect("root articulated inertia is singular — degenerate mass distribution?");
            w.a[0] = a0;
            // Store the 6 free-root accelerations (body-frame at COM) into
            // qddot slots 0..6.
            qddot[0] = a0.angular.x;
            qddot[1] = a0.angular.y;
            qddot[2] = a0.angular.z;
            qddot[3] = a0.linear.x;
            qddot[4] = a0.linear.y;
            qddot[5] = a0.linear.z;
        }
        JointKind::Fixed => {
            w.a[0] = SpatialMotion::ZERO;
        }
        JointKind::Hinge { .. } => unreachable!("hinge cannot be root"),
    }

    for i in 1..n {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Fixed => {
                let parent = link.parent.unwrap();
                // No DOF; a[i] = Xup a[parent] + c[i] (c is zero).
                w.a[i] = w.xup[i].motion(w.a[parent]);
            }
            JointKind::Hinge { .. } => {
                let parent = link.parent.unwrap();
                let a_parent_at_child = w.xup[i].motion(w.a[parent]);
                let s = w.s[i];
                let ia = w.ia[i];
                let pa = w.pa[i];
                // qddot[i] = (τ − Sᵀ (IA (a_parent + c) + pA)) / d
                let acc_prime = a_parent_at_child + w.c[i];
                let inner = ia.times_motion(acc_prime) + pa;
                let s_inner = spatial_dot_ms(s, inner);
                let qdd = (w.tau[i] - s_inner) / w.d[i];
                w.qddot_joint[i] = qdd;
                w.a[i] = acc_prime + s * qdd;
                qddot[tree.v_offset[i]] = qdd;
            }
            JointKind::Free => unreachable!(),
        }
    }

    qddot
}

/// Motion transform from parent body frame to link i's body frame at joint
/// angle `q_angle` (ignored for `Fixed`). Assumes joint orientation offsets
/// are identity in both parent and child (v0 convention).
fn xup_for_link(link: &Link, q_angle: f32) -> Xform {
    let (r_pj, _o_pj) = link.joint_offset_in_parent;
    let (r_jc, _o_jc) = link.joint_offset_in_child;
    let rot_c_from_p = match link.joint {
        JointKind::Fixed => Mat3::IDENTITY,
        JointKind::Hinge { axis, .. } => {
            // Rotation from parent body coords to child body coords is
            // Rot(axis, -q). Encode via the quaternion, then to matrix.
            Quat::from_axis_angle(axis, -q_angle).to_mat3()
        }
        JointKind::Free => Mat3::IDENTITY,
    };
    let t_p_in_c = r_jc - rot_c_from_p * r_pj;
    Xform::new(rot_c_from_p, t_p_in_c)
}

/// Range-limit spring-damper torque. Zero inside `range` (or unconditionally
/// if `range = None`). Outside, a one-sided spring pulls the joint back
/// toward the limit; a damping term on qdot activates too (also one-sided so
/// it doesn't resist motion INTO the allowed range).
fn hinge_limit_torque(q: f32, qdot: f32, range: Option<(f32, f32)>, limit: HingeLimit) -> f32 {
    let (lo, hi) = match range {
        Some(pair) => pair,
        None => return 0.0,
    };
    if q < lo {
        let violation = lo - q; // positive
        // Spring pushes toward +q (into allowed range).
        let spring = limit.stiffness * violation;
        // Damping resists motion further into violation (qdot negative).
        // Only apply when moving away from the range (qdot < 0). When
        // moving back toward the range (qdot > 0), don't oppose it.
        let damp = if qdot < 0.0 {
            -limit.damping * qdot
        } else {
            0.0
        };
        spring + damp
    } else if q > hi {
        let violation = q - hi; // positive
        // Spring pushes toward -q.
        let spring = -limit.stiffness * violation;
        // Damping opposes further increase (qdot > 0).
        let damp = if qdot > 0.0 {
            -limit.damping * qdot
        } else {
            0.0
        };
        spring + damp
    } else {
        0.0
    }
}

/// Scalar dot of a spatial motion with a spatial force: `ω·τ + v·F`. Wrapper
/// around [`crate::spatial::spatial_dot`] for readability inside ABA.
#[inline]
fn spatial_dot_ms(m: SpatialMotion, f: SpatialForce) -> f32 {
    m.angular.dot(f.torque) + m.linear.dot(f.linear)
}

/// Reinterpret a spatial force's `(torque, linear)` components as a spatial
/// motion's `(angular, linear)` for use as the row vector in an outer
/// product. This is purely a packing convenience — the outer product
/// produces a 6x6 matrix regardless of which type is called "row" or
/// "column". Using [`Mat6::outer`] directly with two `SpatialForce`s would
/// require another wrapper.
#[inline]
fn s_force_to_motion(f: SpatialForce) -> SpatialMotion {
    SpatialMotion::new(f.torque, f.linear)
}

/// Scalar multiply of a `Mat6`.
fn scale_mat6(mut m: Mat6, s: f32) -> Mat6 {
    for r in 0..6 {
        for c in 0..6 {
            m.rows[r][c] *= s;
        }
    }
    m
}

// ---------------------------------------------------------------------------
// RK4 integration on a tree
// ---------------------------------------------------------------------------

/// One RK4 step on the tree.
///
/// `compute_ext_wrenches(state)` is invoked at each of the four sub-stages;
/// it receives a temporary snapshot of the tree (with intermediate `q` and
/// `qdot`) and returns per-link external wrenches (contacts, etc.). Gravity
/// is added by [`aba`] directly; do NOT include it here.
pub fn rk4_step<F>(tree: &mut Tree, gravity: Vec3, dt: f32, mut compute_ext_wrenches: F)
where
    F: FnMut(&Tree) -> ExternalWrenches,
{
    let s0 = tree.clone();

    // k1
    let poses1 = forward_kinematics(&s0);
    let ext1 = compute_ext_wrenches(&s0);
    let k1 = aba(&s0, &poses1, gravity, &ext1);
    let (dq1, dv1) = tree_deriv(&s0, &k1);

    // k2 at s0 + k1 * dt/2
    let s1 = tree_advance(&s0, &dq1, &dv1, dt * 0.5);
    let poses2 = forward_kinematics(&s1);
    let ext2 = compute_ext_wrenches(&s1);
    let k2 = aba(&s1, &poses2, gravity, &ext2);
    let (dq2, dv2) = tree_deriv(&s1, &k2);

    // k3 at s0 + k2 * dt/2
    let s2 = tree_advance(&s0, &dq2, &dv2, dt * 0.5);
    let poses3 = forward_kinematics(&s2);
    let ext3 = compute_ext_wrenches(&s2);
    let k3 = aba(&s2, &poses3, gravity, &ext3);
    let (dq3, dv3) = tree_deriv(&s2, &k3);

    // k4 at s0 + k3 * dt
    let s3 = tree_advance(&s0, &dq3, &dv3, dt);
    let poses4 = forward_kinematics(&s3);
    let ext4 = compute_ext_wrenches(&s3);
    let k4 = aba(&s3, &poses4, gravity, &ext4);
    let (dq4, dv4) = tree_deriv(&s3, &k4);

    // Combine and write back into tree.
    let sixth = 1.0 / 6.0;
    for j in 0..s0.q.len() {
        tree.q[j] = s0.q[j] + (dq1[j] + 2.0 * dq2[j] + 2.0 * dq3[j] + dq4[j]) * (dt * sixth);
    }
    for j in 0..s0.qdot.len() {
        tree.qdot[j] = s0.qdot[j] + (dv1[j] + 2.0 * dv2[j] + 2.0 * dv3[j] + dv4[j]) * (dt * sixth);
    }
    // Renormalize free-root quaternion once at step end (mirrors tier 1).
    if let JointKind::Free = tree.links[0].joint {
        let q = Quat::new(tree.q[3], tree.q[4], tree.q[5], tree.q[6]).renormalize();
        tree.q[3] = q.x;
        tree.q[4] = q.y;
        tree.q[5] = q.z;
        tree.q[6] = q.w;
    }
}

/// Compute the position and velocity derivatives for every DOF of the tree,
/// given the joint acceleration vector `qddot` from [`aba`].
///
/// Returns `(dq_dt, dv_dt)`. `dv_dt = qddot`. For hinges `dq_dt = qdot`
/// directly; for a free root the quaternion derivative is
/// `0.5 * q * (ω_body, 0)` (matches tier 1) and the position derivative is
/// `orientation.rotate(v_body)` — the world-frame linear velocity from
/// body-frame components.
fn tree_deriv(tree: &Tree, qddot: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut dq = vec![0.0f32; tree.q.len()];
    let dv = qddot.to_vec();
    for (i, link) in tree.links.iter().enumerate() {
        match link.joint {
            JointKind::Free => {
                let qoff = tree.q_offset[i];
                let voff = tree.v_offset[i];
                let ori = Quat::new(
                    tree.q[qoff + 3],
                    tree.q[qoff + 4],
                    tree.q[qoff + 5],
                    tree.q[qoff + 6],
                );
                let w_body = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let v_body = Vec3::new(
                    tree.qdot[voff + 3],
                    tree.qdot[voff + 4],
                    tree.qdot[voff + 5],
                );
                let v_world = ori.rotate(v_body);
                dq[qoff] = v_world.x;
                dq[qoff + 1] = v_world.y;
                dq[qoff + 2] = v_world.z;
                let dqori = ori.derivative(w_body);
                dq[qoff + 3] = dqori.x;
                dq[qoff + 4] = dqori.y;
                dq[qoff + 5] = dqori.z;
                dq[qoff + 6] = dqori.w;
            }
            JointKind::Fixed => {}
            JointKind::Hinge { .. } => {
                let qoff = tree.q_offset[i];
                let voff = tree.v_offset[i];
                dq[qoff] = tree.qdot[voff];
            }
        }
    }
    (dq, dv)
}

/// Return a new tree state = `origin + (dq, dv) * dt`. Does not renormalize
/// the quaternion (mid-RK4 stages preserve linearity).
fn tree_advance(origin: &Tree, dq: &[f32], dv: &[f32], dt: f32) -> Tree {
    let mut out = origin.clone();
    for (j, slot) in out.q.iter_mut().enumerate() {
        *slot = origin.q[j] + dq[j] * dt;
    }
    for (j, slot) in out.qdot.iter_mut().enumerate() {
        *slot = origin.qdot[j] + dv[j] * dt;
    }
    out
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{FRAC_PI_2, PI};

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn free_root_only_zero_wrench_stays_at_rest() {
        // A single free-root link with no gravity and no external wrench
        // should stay put — this exercises the free-root 6x6 solve.
        let mut tree = Tree::new();
        let inertia = Mat3::diag(1.0, 2.0, 3.0);
        tree.push_link(Link::new(
            None,
            JointKind::Free,
            (Vec3::new(0.0, 0.0, 5.0), Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            inertia,
        ));
        let poses = forward_kinematics(&tree);
        let ext = vec![(Vec3::ZERO, Vec3::ZERO); 1];
        let qddot = aba(&tree, &poses, Vec3::ZERO, &ext);
        for &x in &qddot {
            assert!(approx(x, 0.0, 1e-6), "expected zero accel, got {x}");
        }
    }

    #[test]
    fn free_root_gravity_produces_free_fall_accel() {
        // Free root, gravity −g on z. Linear accel of the body's COM in the
        // BODY frame equals R^T * (0,0,−g). With orientation = IDENTITY,
        // this is exactly (0,0,−g).
        let mut tree = Tree::new();
        tree.push_link(Link::new(
            None,
            JointKind::Free,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            2.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        let poses = forward_kinematics(&tree);
        let ext = vec![(Vec3::ZERO, Vec3::ZERO); 1];
        let qddot = aba(&tree, &poses, Vec3::new(0.0, 0.0, -9.81), &ext);
        // qddot[0..3] = angular_body accel (should be ~0), qddot[3..6] =
        // linear_body accel = (0, 0, -9.81).
        assert!(approx(qddot[0], 0.0, 1e-5));
        assert!(approx(qddot[1], 0.0, 1e-5));
        assert!(approx(qddot[2], 0.0, 1e-5));
        assert!(approx(qddot[3], 0.0, 1e-5));
        assert!(approx(qddot[4], 0.0, 1e-5));
        assert!(approx(qddot[5], -9.81, 1e-4));
    }

    #[test]
    fn single_pendulum_at_horizontal_matches_analytic_accel() {
        // Point-mass pendulum (approximated by a solid sphere of small
        // radius) hanging from a hinge. At angle = π/2 (horizontal), the
        // gravitational torque about the pivot is m g L, and the moment of
        // inertia about the pivot is m L² for a point mass. So α = g / L.
        // Use a very small sphere so its own I_com is negligible.
        let mut tree = Tree::new();
        // Root fixed at world origin.
        tree.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        // Pendulum link: hinge about x, joint anchor in child at
        // (0, 0, L) so the COM sits L below the pivot.
        let l = 1.0f32;
        let mass = 1.0f32;
        // Tiny inertia to approximate a point mass at the COM.
        let inertia = Mat3::diag(1e-6, 1e-6, 1e-6);
        tree.push_link(Link::new(
            Some(0),
            JointKind::hinge(Vec3::X),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, l), Quat::IDENTITY),
            mass,
            inertia,
        ));
        // Set the joint angle to π/2 (horizontal).
        tree.set_hinge_angle(1, FRAC_PI_2);
        let poses = forward_kinematics(&tree);
        let ext = vec![(Vec3::ZERO, Vec3::ZERO); 2];
        let qddot = aba(&tree, &poses, Vec3::new(0.0, 0.0, -9.81), &ext);
        // Expected α: the effective inertia about the pivot is m L² = 1
        // (point-mass); torque about pivot = m g L = 9.81. But hinge axis
        // is +x; positive angle = rotation about +x by right-hand rule.
        // At q=π/2, the COM sits at (0, +L, 0) (rotated from (0, 0, -L) by
        // +π/2 about x). Gravity is (0, 0, -mg). Torque about pivot:
        // r × F = (0, L, 0) × (0, 0, -mg) = (L * -mg, 0, 0) = (-mgL, 0, 0).
        // So expected qddot for hinge-about-x = -g/L = -9.81.
        let alpha_expected = -9.81 / l;
        assert!(
            approx(qddot[0], alpha_expected, 5e-3),
            "expected α={alpha_expected}, got {}",
            qddot[0]
        );
    }

    #[test]
    fn hinge_at_angle_zero_has_zero_accel_when_pointing_down() {
        // Pendulum hanging straight down (q=0). Gravity torque about the
        // hinge is zero → α = 0. Also verifies forward kinematics: COM
        // should sit at (0, 0, -L).
        let l = 1.0f32;
        let mut tree = Tree::new();
        tree.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        tree.push_link(Link::new(
            Some(0),
            JointKind::hinge(Vec3::X),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, l), Quat::IDENTITY),
            1.0,
            Mat3::diag(1e-6, 1e-6, 1e-6),
        ));
        let poses = forward_kinematics(&tree);
        assert!(approx(poses[1].0.x, 0.0, 1e-6));
        assert!(approx(poses[1].0.y, 0.0, 1e-6));
        assert!(approx(poses[1].0.z, -l, 1e-6));
        let ext = vec![(Vec3::ZERO, Vec3::ZERO); 2];
        let qddot = aba(&tree, &poses, Vec3::new(0.0, 0.0, -9.81), &ext);
        assert!(approx(qddot[0], 0.0, 1e-4));
    }

    #[test]
    fn hinge_axis_alignment_matches_forward_kinematics() {
        // Rotate a hinge by π/2 about z. Child COM at (L, 0, 0) in child
        // body frame becomes (0, L, 0) in world.
        let l = 1.0f32;
        let mut tree = Tree::new();
        tree.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        tree.push_link(Link::new(
            Some(0),
            JointKind::hinge(Vec3::Z),
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::new(-l, 0.0, 0.0), Quat::IDENTITY),
            1.0,
            Mat3::diag(1e-6, 1e-6, 1e-6),
        ));
        tree.set_hinge_angle(1, FRAC_PI_2);
        let poses = forward_kinematics(&tree);
        // Child anchor at (-l, 0, 0) in child means the COM sits at (l, 0, 0)
        // in child (COM = origin, anchor is offset by -l on x from COM;
        // equivalently anchor - COM = (-l, 0, 0), so COM - anchor = (l, 0, 0)).
        // After Rot_z(π/2), (l, 0, 0) → (0, l, 0).
        assert!(approx(poses[1].0.x, 0.0, 1e-5));
        assert!(approx(poses[1].0.y, l, 1e-5));
    }

    #[test]
    fn xup_inverse_composes_to_identity_on_motions() {
        // Hinge link at angle π/6 with a nontrivial anchor offset.
        let axis = Vec3::new(1.0, 2.0, -0.5).normalize();
        let joint_p = Vec3::new(0.2, -0.1, 0.3);
        let joint_c = Vec3::new(-0.4, 0.05, 0.15);
        let link = Link::new(
            Some(0),
            JointKind::hinge(axis),
            (joint_p, Quat::IDENTITY),
            (joint_c, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        );
        let q = PI / 6.0;
        let xup = xup_for_link(&link, q);
        let xdn = xup.inverse();
        let m = SpatialMotion::new(Vec3::new(0.7, -1.1, 0.3), Vec3::new(0.2, 0.4, -0.9));
        let round = xdn.motion(xup.motion(m));
        assert!((round.angular - m.angular).length() < 1e-5);
        assert!((round.linear - m.linear).length() < 1e-5);
    }
}
