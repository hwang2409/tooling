//! Kinematic tree: articulated multi-body system with generalized
//! coordinates, forward kinematics, and Featherstone's Articulated Body
//! Algorithm (ABA) for O(n) forward dynamics.
//!
//! # Model
//!
//! A [`Tree`] holds an ordered list of [`Link`]s. Link 0 is always the root
//! (its `parent` is `None`) and carries a [`crate::joint::JointKind::Free`]
//! or [`crate::joint::JointKind::Fixed`] joint. Non-root links carry a
//! [`crate::joint::JointKind::Hinge`], [`crate::joint::JointKind::Slide`],
//! [`crate::joint::JointKind::Ball`], or [`crate::joint::JointKind::Fixed`]
//! joint (v1 tier 1) and reference a parent whose index is strictly less
//! than their own — the tree is stored topologically sorted so a single pass
//! in the vector order visits the root before any of its children. Branching
//! is allowed (multiple children per parent); loops are NOT (no closed
//! kinematic chains in v0).
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
//!   - Slide: `nq = nv = 1`. Displacement (m) / rate (m/s) along the axis.
//!   - Ball: `nq = 4` (child-frame quaternion, renormalized at step end),
//!     `nv = 3` (body-frame ω). Ball limits are deferred to the v1 solver.
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

use crate::aba::{self, GForce, GVec3, JointForces, SharedAbaWorkspace};
use crate::actuator::{Actuator, clamp_symmetric};
use crate::joint::{JointKind, JointLimit};
use crate::math::{Mat3, Quat, Vec3};
use crate::spatial::{SpatialForce, SpatialInertia, SpatialMotion, Xform};
use crate::tendon::Tendon;

/// One link in a kinematic tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    /// Parent link index in the containing [`Tree`]. Must be strictly less
    /// than this link's own index (topological order). `None` iff this is
    /// the root (index 0).
    pub parent: Option<usize>,

    /// Joint connecting this link to its parent (or to the world at the
    /// root). Must be [`JointKind::Free`] or [`JointKind::Fixed`] when
    /// `parent` is `None`; must be one of [`JointKind::Hinge`],
    /// [`JointKind::Slide`], [`JointKind::Ball`], or [`JointKind::Fixed`]
    /// when `parent` is `Some`.
    pub joint: JointKind,

    /// Scalar damping applied to every DOF of a free root joint. MuJoCo's
    /// free-joint damping uses one coefficient for its three angular and
    /// three linear velocity slots. This stays separate from `JointKind` so
    /// existing programmatic `JointKind::Free` callers keep their API.
    pub free_damping: f32,

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

    /// A mocap link is posed by the caller and is never integrated by the
    /// dynamics solver. Only root mocap links are supported in this tier.
    pub mocap: bool,
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
            free_damping: 0.0,
            joint_offset_in_parent,
            joint_offset_in_child,
            mass,
            inertia_body,
            inertia_body_inverse,
            mocap: false,
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

    /// Actuators attached to hinge/slide joints (tier 4, generalized in
    /// v2 tier 2 — see [`Actuator`]). Controls settable per step via
    /// [`Tree::set_actuator_target`].
    /// Filter activation is integrated once per step at the end of
    /// [`rk4_step`]. Muscle activation follows the RK4 mechanical stages.
    pub actuators: Vec<Actuator>,
    /// Per-link user-applied world-frame wrenches at each link's COM,
    /// `(force_world, torque_world)`. Length equals `links.len()`; grows
    /// automatically on [`Tree::push_link`]. Sums with contact wrenches
    /// inside [`aba`] — no special-casing per link kind.
    pub applied_wrenches: Vec<(Vec3, Vec3)>,

    /// When `true`, [`aba`] skips the tier-3 penalty limit torque
    /// contribution for every hinge/slide range. Owned by
    /// [`crate::world::World`], which sets this before each solver step when
    /// `SolverMode::Pgs` or `SolverMode::Newton` is active so the selected
    /// limit constraint (see
    /// [`crate::solver::solve_tree_limits`]) is the sole limit
    /// enforcer — mirroring how contacts already switch. Default `false`
    /// preserves every pre-v1-tier-4 golden and any direct
    /// [`aba`]/[`rk4_step`] caller under the penalty pathway.
    pub disable_penalty_limits: bool,

    /// Tendons in this tree (v2 tier 3). Fixed and spatial — see
    /// [`crate::tendon`]. Passive spring/damper forces are computed
    /// inside every ABA call and added to the effective `tau`. Tendon-
    /// attached actuators (`Actuator::tendon_target = Some(_)`) route
    /// their scalar force through the same tendon Jacobian instead of
    /// the joint per-link path. Empty by default, so every pre-v2-tier-3
    /// scene runs unchanged.
    pub tendons: Vec<Tendon>,
    /// User-supplied world-frame velocity for a mocap root. It is used for
    /// contact relative velocity and is not integrated.
    pub mocap_linear_velocity: Vec3,
    pub mocap_angular_velocity: Vec3,
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
            disable_penalty_limits: false,
            tendons: Vec::new(),
            mocap_linear_velocity: Vec3::ZERO,
            mocap_angular_velocity: Vec3::ZERO,
        }
    }

    /// Append a tendon to this tree. Returns its stable index (usable as
    /// `Actuator::on_tendon(idx)` or `tree.tendons[idx]`). Validated at
    /// call time — panics on a bad tendon (loader routes structured
    /// errors instead).
    pub fn add_tendon(&mut self, tendon: Tendon) -> usize {
        tendon.validate(self).expect(
            "Tree::add_tendon: tendon failed validation (use loader for structured errors)",
        );
        let idx = self.tendons.len();
        self.tendons.push(tendon);
        idx
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
            JointKind::Slide { .. } => {
                self.q.push(0.0);
            }
            JointKind::Ball { .. } => {
                // Default ball q = IDENTITY quaternion (no rotation between
                // parent and child at t=0).
                self.q.extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            }
        }
        self.qdot.extend(std::iter::repeat_n(0.0, nv));
        self.qfrc_applied.extend(std::iter::repeat_n(0.0, nv));
        self.applied_wrenches.push((Vec3::ZERO, Vec3::ZERO));
        self.links.push(link);
        idx
    }

    /// Attach an actuator to a hinge or slide link. Returns the
    /// actuator's stable index (usable with [`Tree::set_actuator_target`]).
    /// Panics if `actuator.link_idx` is out of range or does not reference a
    /// hinge or slide.
    ///
    /// Whichever flavor (position/velocity/motor/general), the actuator
    /// contributes a scalar joint force to the 1-DOF slot — hinges see it
    /// as N·m, slides as N — so the plumbing does not need to know which
    /// kind of joint it hangs off.
    pub fn add_actuator(&mut self, actuator: Actuator) -> usize {
        // Tendon-mode actuators skip the joint kind check — their scalar
        // force enters via the tendon Jacobian, not a joint-slot torque.
        if let Some(tid) = actuator.tendon_target {
            assert!(
                tid < self.tendons.len(),
                "tendon-mode actuator: tendon index {tid} out of range ({} tendons)",
                self.tendons.len()
            );
        } else {
            assert!(
                actuator.link_idx < self.links.len(),
                "actuator link out of range"
            );
            assert!(
                matches!(
                    self.links[actuator.link_idx].joint,
                    JointKind::Hinge { .. } | JointKind::Slide { .. }
                ),
                "actuators only attach to Hinge or Slide joints (link {} is {:?})",
                actuator.link_idx,
                self.links[actuator.link_idx].joint
            );
        }
        let idx = self.actuators.len();
        self.actuators.push(actuator);
        idx
    }

    /// Set an actuator's control input (`ctrl`) — for position this is the
    /// target setpoint; for velocity the target rate; for motor the raw
    /// command scale; for general the input to `gain*ctrl + bias` (or the
    /// driving input to the activation filter). Panics on out-of-range
    /// index. Retains the v0 name for API stability; MuJoCo would call
    /// this `data.ctrl[i]`.
    pub fn set_actuator_target(&mut self, actuator_idx: usize, ctrl: f32) {
        self.actuators[actuator_idx].ctrl = ctrl;
    }

    /// Integrate every actuator's boundary activation state forward by `dt`.
    /// Euler uses this for filters and muscles. RK4 uses it only for filters;
    /// muscle activation is integrated by the RK4 stages.
    pub fn integrate_activations(&mut self, dt: f32) {
        for a in &mut self.actuators {
            a.integrate_activation(dt);
        }
    }

    /// Integrate only filter activations at the end of an RK4 step.
    /// Muscle activations are integrated by the RK4 stages themselves.
    fn integrate_filter_activations(&mut self, dt: f32) {
        for a in &mut self.actuators {
            if matches!(a.dyn_type, crate::actuator::DynType::Filter) {
                a.integrate_activation(dt);
            }
        }
    }

    /// Directly write a generalized joint force into `qfrc_applied` for a
    /// hinge or slide link, symmetric-clamped to `force_range` (pass `0.0`
    /// or a negative value to disable the clamp). This is the motor-style
    /// input tier-4 exposes on top of the raw `qfrc_applied` buffer.
    /// Persists across steps — call again to update, or use
    /// [`Tree::clear_qfrc_applied`] to zero every slot.
    ///
    /// Panics if `link_idx` is not a hinge or slide.
    pub fn set_joint_torque_clamped(&mut self, link_idx: usize, torque: f32, force_range: f32) {
        assert!(
            matches!(
                self.links[link_idx].joint,
                JointKind::Hinge { .. } | JointKind::Slide { .. }
            ),
            "set_joint_torque_clamped requires a Hinge or Slide link"
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

    /// Overwrite the slide displacement for link `i`. Panics if `i`'s joint
    /// is not a slide.
    pub fn set_slide_position(&mut self, i: usize, position: f32) {
        assert!(matches!(self.links[i].joint, JointKind::Slide { .. }));
        let off = self.q_offset[i];
        self.q[off] = position;
    }

    /// Overwrite the slide rate for link `i`.
    pub fn set_slide_rate(&mut self, i: usize, rate: f32) {
        assert!(matches!(self.links[i].joint, JointKind::Slide { .. }));
        let off = self.v_offset[i];
        self.qdot[off] = rate;
    }

    /// Read the slide displacement at link `i`.
    pub fn slide_position(&self, i: usize) -> f32 {
        assert!(matches!(self.links[i].joint, JointKind::Slide { .. }));
        self.q[self.q_offset[i]]
    }

    /// Read the slide rate at link `i`.
    pub fn slide_rate(&self, i: usize) -> f32 {
        assert!(matches!(self.links[i].joint, JointKind::Slide { .. }));
        self.qdot[self.v_offset[i]]
    }

    /// Read the ball joint's child-relative-to-parent orientation at link
    /// `i`. Panics if `i`'s joint is not a ball.
    pub fn ball_orientation(&self, i: usize) -> Quat {
        assert!(matches!(self.links[i].joint, JointKind::Ball { .. }));
        let off = self.q_offset[i];
        Quat::new(
            self.q[off],
            self.q[off + 1],
            self.q[off + 2],
            self.q[off + 3],
        )
    }

    /// Overwrite the ball joint's orientation at link `i` (renormalized).
    pub fn set_ball_orientation(&mut self, i: usize, orientation: Quat) {
        assert!(matches!(self.links[i].joint, JointKind::Ball { .. }));
        let q = orientation.renormalize();
        let off = self.q_offset[i];
        self.q[off] = q.x;
        self.q[off + 1] = q.y;
        self.q[off + 2] = q.z;
        self.q[off + 3] = q.w;
    }

    /// Read the ball joint's body-frame angular velocity at link `i`.
    pub fn ball_omega(&self, i: usize) -> Vec3 {
        assert!(matches!(self.links[i].joint, JointKind::Ball { .. }));
        let off = self.v_offset[i];
        Vec3::new(self.qdot[off], self.qdot[off + 1], self.qdot[off + 2])
    }

    /// Overwrite the ball joint's body-frame angular velocity at link `i`.
    pub fn set_ball_omega(&mut self, i: usize, omega: Vec3) {
        assert!(matches!(self.links[i].joint, JointKind::Ball { .. }));
        let off = self.v_offset[i];
        self.qdot[off] = omega.x;
        self.qdot[off + 1] = omega.y;
        self.qdot[off + 2] = omega.z;
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

    /// Dense world-frame Jacobian for a link COM.
    pub fn link_jacobian(&self, link: usize) -> crate::jacobian::Jacobian {
        crate::jacobian::link_jacobian(self, link)
    }

    /// Dense world-frame Jacobian for a point in a link's body frame.
    pub fn point_jacobian(&self, link: usize, point_local: Vec3) -> crate::jacobian::Jacobian {
        crate::jacobian::point_jacobian(self, link, point_local)
    }

    /// Mark a root link as mocap. Mocap links are kinematic and cannot carry
    /// a movable child chain in this tier.
    pub fn set_mocap(&mut self, link: usize, mocap: bool) {
        assert_eq!(link, 0, "only a tree root can be mocap in this tier");
        assert!(
            matches!(self.links[link].joint, JointKind::Free | JointKind::Fixed),
            "mocap root must use a free or fixed joint"
        );
        if mocap
            && self
                .links
                .iter()
                .skip(1)
                .any(|child| !matches!(child.joint, JointKind::Fixed))
        {
            panic!("mocap root cannot have movable descendants");
        }
        self.links[link].mocap = mocap;
    }

    /// Set the world pose of a mocap root. A free root stores the pose in q;
    /// a fixed root stores it in its world anchor.
    pub fn set_mocap_pose(&mut self, position: Vec3, orientation: Quat) {
        assert!(self.links.first().is_some_and(|link| link.mocap));
        match self.links[0].joint {
            JointKind::Free => self.set_free_root_pose(position, orientation),
            JointKind::Fixed => {
                self.links[0].joint_offset_in_parent = (position, orientation.renormalize());
            }
            _ => unreachable!(),
        }
    }

    /// Set the world-frame velocity used by contacts against a mocap root.
    pub fn set_mocap_velocity(&mut self, linear: Vec3, angular: Vec3) {
        assert!(self.links.first().is_some_and(|link| link.mocap));
        self.mocap_linear_velocity = linear;
        self.mocap_angular_velocity = angular;
    }

    /// Dense joint-space mass matrix `M(q)` (row-major, `nv × nv`). See
    /// [`crate::dynamics::mass_matrix`] for the algorithm and layout.
    pub fn mass_matrix(&self) -> Vec<f32> {
        crate::dynamics::mass_matrix(self)
    }

    /// Dense joint-space response matrix for the selected implicit velocity
    /// solve. This is `M + dt*B`, matching the damping and actuator terms
    /// that [`aba_implicit`] folds into its articulated-inertia pivots.
    pub fn implicit_mass_matrix(&self, dt: f32, implicit_fast: bool) -> Vec<f32> {
        let mut matrix = self.mass_matrix();
        let nv = self.nv();
        if nv == 0 || dt == 0.0 {
            return matrix;
        }
        for (link_idx, link) in self.links.iter().enumerate() {
            let (offset, damping): (usize, Vec<f32>) = match link.joint {
                JointKind::Free => (
                    self.v_offset[link_idx],
                    vec![link.free_damping; link.joint.nv()],
                ),
                JointKind::Hinge { damping, .. } | JointKind::Slide { damping, .. } => {
                    let actuator_damping = if implicit_fast {
                        let q = self.q[self.q_offset[link_idx]];
                        let qdot = self.qdot[self.v_offset[link_idx]];
                        self.actuators
                            .iter()
                            .filter(|act| act.tendon_target.is_none() && act.link_idx == link_idx)
                            .map(|act| act.velocity_damping(q, qdot))
                            .sum()
                    } else {
                        0.0
                    };
                    (self.v_offset[link_idx], vec![damping + actuator_damping])
                }
                JointKind::Ball { damping, .. } => {
                    (self.v_offset[link_idx], vec![damping; link.joint.nv()])
                }
                JointKind::Fixed => continue,
            };
            for (slot, damping) in (offset..offset + link.joint.nv()).zip(damping) {
                matrix[slot * nv + slot] += dt * damping;
            }
        }
        matrix
    }

    /// Coriolis + centrifugal + gravity torques `h(q, qdot)` — RNE with
    /// `qddot = 0` and no external wrenches. See
    /// [`crate::dynamics::bias_forces`].
    pub fn bias_forces(&self, gravity: Vec3) -> Vec<f32> {
        crate::dynamics::bias_forces(self, gravity)
    }

    /// Dense explicit forward-dynamics derivatives at the current state.
    ///
    /// `external_wrenches` uses the same world-frame, per-link format as
    /// [`crate::tree::aba`]. See [`crate::dynamics::Derivatives`] for matrix
    /// layouts and the supported analytic paths.
    pub fn derivatives(
        &self,
        gravity: Vec3,
        external_wrenches: &ExternalWrenches,
    ) -> crate::dynamics::Derivatives {
        crate::dynamics::derivatives(self, gravity, external_wrenches)
    }

    /// Dense derivatives for constrained dynamics.
    ///
    /// The callback rebuilds world-frame contact or friction wrenches for
    /// every perturbed state. Use this entry point when the external wrench
    /// depends on `q`, `qdot`, or actuator controls. Fixed-wrench derivatives
    /// should use [`Tree::derivatives`].
    pub fn constrained_derivatives<F>(
        &self,
        gravity: Vec3,
        external_wrenches: F,
    ) -> crate::dynamics::Derivatives
    where
        F: Fn(&Tree) -> ExternalWrenches,
    {
        crate::dynamics::constrained_derivatives(self, gravity, external_wrenches)
    }

    /// Inverse dynamics: generalized force required to produce `qddot`
    /// under `gravity` and `external_wrenches`. See
    /// [`crate::dynamics::inverse_dynamics`].
    pub fn inverse_dynamics(
        &self,
        qddot: &[f32],
        gravity: Vec3,
        external_wrenches: &ExternalWrenches,
    ) -> Vec<f32> {
        crate::dynamics::inverse_dynamics(self, qddot, gravity, external_wrenches)
    }

    /// Inverse dynamics for an explicit state. The input vectors are copied
    /// into a temporary tree, so this call does not change the live state.
    /// Damping, armature, limits, actuators, and `qfrc_applied` stay outside
    /// the RNE contract; only gravity and `external_wrenches` enter here.
    pub fn inverse_dynamics_at(
        &self,
        q: &[f32],
        qdot: &[f32],
        qddot: &[f32],
        gravity: Vec3,
        external_wrenches: &ExternalWrenches,
    ) -> Vec<f32> {
        assert_eq!(q.len(), self.nq(), "q length must match tree.nq()");
        assert_eq!(qdot.len(), self.nv(), "qdot length must match tree.nv()");
        let mut state = self.clone();
        state.q.copy_from_slice(q);
        state.qdot.copy_from_slice(qdot);
        state.inverse_dynamics(qddot, gravity, external_wrenches)
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
    forward_kinematics_into(tree, &mut out);
    out
}

pub(crate) fn forward_kinematics_into(tree: &Tree, out: &mut Vec<(Vec3, Quat)>) {
    out.clear();
    out.reserve(tree.links.len().saturating_sub(out.capacity()));
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
            JointKind::Slide { axis, .. } => {
                let parent_idx = link.parent.expect("slide joint must have a parent");
                let (parent_pos, parent_ori) = out[parent_idx];
                let q_slide = tree.q[tree.q_offset[i]];
                let (offset_p, _offset_p_o) = link.joint_offset_in_parent;
                let (offset_c_p, _offset_c_o) = link.joint_offset_in_child;
                // Child orientation = parent orientation (slide never rotates).
                let child_ori = parent_ori;
                // Joint anchor in world at q=0: parent-anchor pose. Slide
                // displaces the child body by axis*q along the (child = parent
                // frame) axis direction, rotated into world.
                let joint_pos_world = parent_pos + parent_ori.rotate(offset_p);
                let slide_world = parent_ori.rotate(axis * q_slide);
                let child_pos = joint_pos_world + slide_world - child_ori.rotate(offset_c_p);
                (child_pos, child_ori)
            }
            JointKind::Ball { .. } => {
                let parent_idx = link.parent.expect("ball joint must have a parent");
                let (parent_pos, parent_ori) = out[parent_idx];
                let off = tree.q_offset[i];
                let q_ball = Quat::new(
                    tree.q[off],
                    tree.q[off + 1],
                    tree.q[off + 2],
                    tree.q[off + 3],
                );
                let (offset_p, _offset_p_o) = link.joint_offset_in_parent;
                let (offset_c_p, _offset_c_o) = link.joint_offset_in_child;
                let joint_pos_world = parent_pos + parent_ori.rotate(offset_p);
                let child_ori = parent_ori * q_ball;
                let child_pos = joint_pos_world - child_ori.rotate(offset_c_p);
                (child_pos, child_ori)
            }
        };
        out.push(pose);
    }
}

/// Compute one link pose without allocating a pose vector.
pub(crate) fn link_pose_nonalloc(tree: &Tree, target: usize) -> (Vec3, Quat) {
    fn pose_at(tree: &Tree, index: usize) -> (Vec3, Quat) {
        let link = &tree.links[index];
        let parent = link
            .parent
            .map(|parent| pose_at(tree, parent))
            .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
        let (parent_pos, parent_ori) = parent;
        match link.joint {
            JointKind::Free => {
                let off = tree.q_offset[index];
                (
                    Vec3::new(tree.q[off], tree.q[off + 1], tree.q[off + 2]),
                    Quat::new(
                        tree.q[off + 3],
                        tree.q[off + 4],
                        tree.q[off + 5],
                        tree.q[off + 6],
                    ),
                )
            }
            JointKind::Fixed => {
                let (offset_p, offset_o) = link.joint_offset_in_parent;
                let (offset_c_p, offset_c_o) = link.joint_offset_in_child;
                let joint_pos = parent_pos + parent_ori.rotate(offset_p);
                let joint_ori = parent_ori * offset_o;
                let child_ori = joint_ori * offset_c_o.conjugate();
                (joint_pos - child_ori.rotate(offset_c_p), child_ori)
            }
            JointKind::Hinge { axis, .. } => {
                let q_angle = tree.q[tree.q_offset[index]];
                let (offset_p, _) = link.joint_offset_in_parent;
                let (offset_c_p, _) = link.joint_offset_in_child;
                let joint_pos = parent_pos + parent_ori.rotate(offset_p);
                let child_ori = parent_ori * Quat::from_axis_angle(axis, q_angle);
                (joint_pos - child_ori.rotate(offset_c_p), child_ori)
            }
            JointKind::Slide { axis, .. } => {
                let q_slide = tree.q[tree.q_offset[index]];
                let (offset_p, _) = link.joint_offset_in_parent;
                let (offset_c_p, _) = link.joint_offset_in_child;
                let joint_pos = parent_pos + parent_ori.rotate(offset_p);
                let child_pos = joint_pos + parent_ori.rotate(axis * q_slide);
                (child_pos - parent_ori.rotate(offset_c_p), parent_ori)
            }
            JointKind::Ball { .. } => {
                let off = tree.q_offset[index];
                let q_ball = Quat::new(
                    tree.q[off],
                    tree.q[off + 1],
                    tree.q[off + 2],
                    tree.q[off + 3],
                );
                let (offset_p, _) = link.joint_offset_in_parent;
                let (offset_c_p, _) = link.joint_offset_in_child;
                let joint_pos = parent_pos + parent_ori.rotate(offset_p);
                let child_ori = parent_ori * q_ball;
                (joint_pos - child_ori.rotate(offset_c_p), child_ori)
            }
        }
    }

    pose_at(tree, target)
}

// ---------------------------------------------------------------------------
// ABA — Featherstone's Articulated Body Algorithm
// ---------------------------------------------------------------------------

/// Scratch reused by the shared ABA pass.
#[derive(Clone, Debug)]
pub(crate) struct AbaWorkspace {
    /// Tendon-generated generalized force scratch (passive spring/damper +
    /// tendon-actuator). Length = tree.nv(). Reused across ABA calls to
    /// avoid an allocation per call.
    tendon_qfrc: Vec<f32>,
    /// Scalar joint-force scratch, reused across ABA calls.
    joint_force_scalar: Vec<f32>,
    /// Ball-joint force scratch, reused across ABA calls.
    joint_force_ball: Vec<Vec3>,
    /// Shared scalar ABA workspace.
    shared: SharedAbaWorkspace<f32>,
    /// Per-link external forces in the link body frame.
    external_body: Vec<GForce<f32>>,
    /// Shared ball-joint force view.
    shared_ball: Vec<GVec3<f32>>,
    /// Per-link implicit damping mass.
    damping_mass: Vec<f32>,
}

impl AbaWorkspace {
    pub(crate) fn new(n: usize) -> Self {
        Self::with_nv(n, 0)
    }

    fn with_nv(n: usize, nv: usize) -> Self {
        Self {
            tendon_qfrc: vec![0.0; nv],
            joint_force_scalar: vec![0.0; n],
            joint_force_ball: vec![Vec3::ZERO; n],
            shared: SharedAbaWorkspace::new(n),
            external_body: vec![GForce::zero(); n],
            shared_ball: vec![
                GVec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0
                };
                n
            ],
            damping_mass: vec![0.0; n],
        }
    }

    pub(crate) fn ensure_len(&mut self, n: usize) {
        if self.shared_ball.len() != n {
            *self = Self::new(n);
        }
    }

    /// Ensure the nv-sized scratch buffers match `nv`. Called at the start of
    /// each ABA pass because `n` (link count) and `nv` are set independently
    /// and callers only pass link count to `new`/`ensure_len`.
    fn ensure_nv(&mut self, nv: usize) {
        if self.tendon_qfrc.len() != nv {
            self.tendon_qfrc.resize(nv, 0.0);
        }
    }
}

/// External wrench on each link, in WORLD coordinates at the link's COM.
/// `ext[i] = (force_world, torque_world_about_com)`.
pub type ExternalWrenches = Vec<(Vec3, Vec3)>;

#[derive(Clone, Copy, Debug, PartialEq)]
enum VelocityImplicit {
    Explicit,
    JointDamping { dt: f32 },
    JointDampingAndActuators { dt: f32 },
}

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
    aba_with_velocity_implicit(
        tree,
        poses,
        gravity,
        external_wrenches,
        VelocityImplicit::Explicit,
    )
}

/// Compute generalized acceleration with the same velocity-implicit fold as
/// [`euler_step`]. Constraint assembly uses this to form the free velocity
/// before a tree contact impulse enters the selected solver.
pub fn aba_implicit(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
    dt: f32,
    implicit_fast: bool,
) -> Vec<f32> {
    let mode = if implicit_fast {
        VelocityImplicit::JointDampingAndActuators { dt }
    } else {
        VelocityImplicit::JointDamping { dt }
    };
    aba_with_velocity_implicit(tree, poses, gravity, external_wrenches, mode)
}

fn implicit_mass_damping(
    tree: &Tree,
    link_idx: usize,
    q: f32,
    qdot: f32,
    joint_damping: f32,
    mode: VelocityImplicit,
) -> f32 {
    let (dt, actuator_damping) = match mode {
        VelocityImplicit::Explicit => return 0.0,
        VelocityImplicit::JointDamping { dt } => (dt, 0.0),
        VelocityImplicit::JointDampingAndActuators { dt } => {
            // Tendon actuator velocity derivatives are dense JᵀJ terms. Do
            // not fold them into these scalar joint denominators; this
            // ticket evaluates tendon velocity force explicitly.
            let damping = tree
                .actuators
                .iter()
                .filter(|act| act.tendon_target.is_none() && act.link_idx == link_idx)
                .map(|act| act.velocity_damping(q, qdot))
                .sum();
            (dt, damping)
        }
    };
    dt * (joint_damping + actuator_damping)
}

fn aba_with_velocity_implicit(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
    velocity_implicit: VelocityImplicit,
) -> Vec<f32> {
    let mut workspace = AbaWorkspace::new(tree.links.len());
    aba_with_velocity_implicit_workspace(
        tree,
        poses,
        gravity,
        external_wrenches,
        velocity_implicit,
        &mut workspace,
    )
}

fn aba_with_velocity_implicit_workspace(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
    velocity_implicit: VelocityImplicit,
    workspace: &mut AbaWorkspace,
) -> Vec<f32> {
    let n = tree.links.len();
    let nv = tree.nv();
    assert_eq!(poses.len(), n);
    assert_eq!(external_wrenches.len(), n);
    workspace.ensure_len(n);
    workspace.ensure_nv(nv);

    let mut tendon_qfrc = std::mem::take(&mut workspace.tendon_qfrc);
    tendon_qfrc.clear();
    tendon_qfrc.resize(nv, 0.0);
    if !tree.tendons.is_empty() {
        let mut tendon_state =
            crate::tendon::accumulate_tendon_passive(tree, poses, &mut tendon_qfrc);
        crate::tendon::accumulate_tendon_actuator_qfrc(tree, &mut tendon_state, &mut tendon_qfrc);
    }
    let free = crate::forces::assemble_joint_forces(
        tree,
        &tendon_qfrc,
        &mut workspace.joint_force_scalar,
        &mut workspace.joint_force_ball,
    );
    for (i, pose) in poses.iter().enumerate() {
        workspace.shared_ball[i] = GVec3 {
            x: workspace.joint_force_ball[i].x,
            y: workspace.joint_force_ball[i].y,
            z: workspace.joint_force_ball[i].z,
        };
        let (force_world, torque_world) = external_wrenches[i];
        let (force_applied, torque_applied) = tree.applied_wrenches[i];
        let force_body = pose
            .1
            .inverse_rotate(force_world + force_applied + gravity * tree.links[i].mass);
        let torque_body = pose.1.inverse_rotate(torque_world + torque_applied);
        workspace.external_body[i] = GForce {
            torque: GVec3::from_vec3(torque_body),
            linear: GVec3::from_vec3(force_body),
        };
        workspace.damping_mass[i] = match tree.links[i].joint {
            JointKind::Hinge { damping, .. } | JointKind::Slide { damping, .. } => {
                let q = tree.q[tree.q_offset[i]];
                let qdot = tree.qdot[tree.v_offset[i]];
                implicit_mass_damping(tree, i, q, qdot, damping, velocity_implicit)
            }
            JointKind::Ball { damping, .. } => match velocity_implicit {
                VelocityImplicit::Explicit => 0.0,
                VelocityImplicit::JointDamping { dt }
                | VelocityImplicit::JointDampingAndActuators { dt } => dt * damping,
            },
            JointKind::Free => match velocity_implicit {
                VelocityImplicit::Explicit => 0.0,
                VelocityImplicit::JointDamping { dt }
                | VelocityImplicit::JointDampingAndActuators { dt } => {
                    dt * tree.links[i].free_damping
                }
            },
            JointKind::Fixed => 0.0,
        };
    }
    let forces = JointForces {
        scalar: &workspace.joint_force_scalar,
        ball: &workspace.shared_ball,
        free: GForce {
            torque: GVec3 {
                x: free.torque.x,
                y: free.torque.y,
                z: free.torque.z,
            },
            linear: GVec3 {
                x: free.linear.x,
                y: free.linear.y,
                z: free.linear.z,
            },
        },
    };
    let mut qacc = vec![0.0; nv];
    aba::run(
        tree,
        &tree.q,
        &tree.qdot,
        &workspace.external_body,
        forces,
        &workspace.damping_mass,
        &mut workspace.shared,
        &mut qacc,
    );
    workspace.tendon_qfrc = tendon_qfrc;
    qacc
}

/// Motion transform from parent body frame to a `Fixed`-jointed child's body
/// frame. The relative joint orientation matches `forward_kinematics`.
pub(crate) fn xup_for_link(link: &Link, _unused: f32) -> Xform {
    let (r_pj, q_pj) = link.joint_offset_in_parent;
    let (r_jc, q_jc) = link.joint_offset_in_child;
    let relative = q_pj * q_jc.conjugate();
    let rot_c_from_p = relative.conjugate().to_mat3();
    let t_p_in_c = r_jc - rot_c_from_p * r_pj;
    Xform::new(rot_c_from_p, t_p_in_c)
}

/// Xup for a hinge at angle `q_angle`. `rot_c_from_p = Rot(axis, -q)`.
pub(crate) fn xup_for_link_hinge(link: &Link, axis: Vec3, q_angle: f32) -> Xform {
    let (r_pj, _) = link.joint_offset_in_parent;
    let (r_jc, _) = link.joint_offset_in_child;
    let rot_c_from_p = Quat::from_axis_angle(axis, -q_angle).to_mat3();
    let t_p_in_c = r_jc - rot_c_from_p * r_pj;
    Xform::new(rot_c_from_p, t_p_in_c)
}

/// Xup for a slide at displacement `q_slide`. The child body frame stays
/// identically oriented to the parent (slide never rotates), so
/// `rot_c_from_p = I`. The translation part carries the extra `-axis*q`
/// term for the displacement.
pub(crate) fn xup_for_link_slide(link: &Link, axis: Vec3, q_slide: f32) -> Xform {
    let (r_pj, _) = link.joint_offset_in_parent;
    let (r_jc, _) = link.joint_offset_in_child;
    let rot_c_from_p = Mat3::IDENTITY;
    // parent origin in child coords = r_jc - r_pj - axis * q  (child ori =
    // parent ori, so no rotation shift).
    let t_p_in_c = r_jc - r_pj - axis * q_slide;
    Xform::new(rot_c_from_p, t_p_in_c)
}

/// Xup for a ball at orientation `q_ball` (child relative to parent).
/// `rot_c_from_p = R(q_ball)ᵀ`.
pub(crate) fn xup_for_link_ball(link: &Link, q_ball: Quat) -> Xform {
    let (r_pj, _) = link.joint_offset_in_parent;
    let (r_jc, _) = link.joint_offset_in_child;
    let rot_c_from_p = q_ball.conjugate().to_mat3();
    let t_p_in_c = r_jc - rot_c_from_p * r_pj;
    Xform::new(rot_c_from_p, t_p_in_c)
}

/// Pass-2 update shared by all single-DOF joints (hinge, slide). Consumes
/// the cached `w.s[i]` from pass 1 plus the caller-computed scalar τ, and
/// mutates `w.ia[parent]` / `w.pa[parent]` / caches for pass 3.
/// Range-limit spring-damper generalized force for a single-DOF joint
/// (hinge or slide). Zero inside `range` (or unconditionally if
/// `range = None`). Outside, a one-sided spring pulls the joint back
/// toward the limit; a damping term on `qdot` activates too (also one-
/// sided so it doesn't resist motion INTO the allowed range).
pub(crate) fn joint_limit_scalar_force(
    q: f32,
    qdot: f32,
    range: Option<(f32, f32)>,
    limit: JointLimit,
) -> f32 {
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
pub(crate) fn spatial_dot_ms(m: SpatialMotion, f: SpatialForce) -> f32 {
    m.angular.dot(f.torque) + m.linear.dot(f.linear)
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
pub fn rk4_step<F>(tree: &mut Tree, gravity: Vec3, dt: f32, compute_ext_wrenches: F)
where
    F: FnMut(&Tree) -> ExternalWrenches,
{
    let mut workspace = AbaWorkspace::new(tree.links.len());
    rk4_step_with_workspace(tree, gravity, dt, compute_ext_wrenches, &mut workspace);
}

pub(crate) fn rk4_step_with_workspace<F>(
    tree: &mut Tree,
    gravity: Vec3,
    dt: f32,
    mut compute_ext_wrenches: F,
    workspace: &mut AbaWorkspace,
) where
    F: FnMut(&Tree) -> ExternalWrenches,
{
    // Save only the state fields that RK4 sub-stages mutate: generalized
    // position, velocity, and per-muscle activation. The rest of the tree
    // (links, offsets, tendons, actuator params) is invariant across the
    // step, so cloning it four times (as `tree.clone()` +
    // `tree_advance().clone()` used to do) burned per-stage allocations
    // scaling with tree size. Reusing the same working tree and swapping
    // just these buffers preserves every read the callback / ABA path
    // makes, and keeps the arithmetic identical.
    let q0 = tree.q.clone();
    let qdot0 = tree.qdot.clone();
    let act0: Vec<f32> = tree.actuators.iter().map(|actuator| actuator.act).collect();

    // k1 — tree is already at s0.
    let poses1 = forward_kinematics(tree);
    let ext1 = compute_ext_wrenches(tree);
    let k1 = aba_with_velocity_implicit_workspace(
        tree,
        &poses1,
        gravity,
        &ext1,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq1, dv1) = tree_deriv(tree, &k1);
    let da1 = muscle_activation_deriv(tree);

    // Advance tree in-place to s0 + k1 * dt/2 for stage 2.
    advance_tree_state(tree, &q0, &qdot0, &act0, &dq1, &dv1, &da1, dt * 0.5);
    let poses2 = forward_kinematics(tree);
    let ext2 = compute_ext_wrenches(tree);
    let k2 = aba_with_velocity_implicit_workspace(
        tree,
        &poses2,
        gravity,
        &ext2,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq2, dv2) = tree_deriv(tree, &k2);
    let da2 = muscle_activation_deriv(tree);

    // Advance tree in-place to s0 + k2 * dt/2 for stage 3.
    advance_tree_state(tree, &q0, &qdot0, &act0, &dq2, &dv2, &da2, dt * 0.5);
    let poses3 = forward_kinematics(tree);
    let ext3 = compute_ext_wrenches(tree);
    let k3 = aba_with_velocity_implicit_workspace(
        tree,
        &poses3,
        gravity,
        &ext3,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq3, dv3) = tree_deriv(tree, &k3);
    let da3 = muscle_activation_deriv(tree);

    // Advance tree in-place to s0 + k3 * dt for stage 4.
    advance_tree_state(tree, &q0, &qdot0, &act0, &dq3, &dv3, &da3, dt);
    let poses4 = forward_kinematics(tree);
    let ext4 = compute_ext_wrenches(tree);
    let k4 = aba_with_velocity_implicit_workspace(
        tree,
        &poses4,
        gravity,
        &ext4,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq4, dv4) = tree_deriv(tree, &k4);
    let da4 = muscle_activation_deriv(tree);

    // Combine and write back into tree. All reads use the saved
    // start-of-step state so the arithmetic matches the original
    // `s0 + (k1 + 2 k2 + 2 k3 + k4) / 6 · dt` combination exactly.
    let sixth = 1.0 / 6.0;
    for j in 0..q0.len() {
        tree.q[j] = q0[j] + (dq1[j] + 2.0 * dq2[j] + 2.0 * dq3[j] + dq4[j]) * (dt * sixth);
    }
    for j in 0..qdot0.len() {
        tree.qdot[j] = qdot0[j] + (dv1[j] + 2.0 * dv2[j] + 2.0 * dv3[j] + dv4[j]) * (dt * sixth);
    }
    for (i, actuator) in tree.actuators.iter_mut().enumerate() {
        if matches!(actuator.dyn_type, crate::actuator::DynType::Muscle) {
            actuator.act = act0[i] + (da1[i] + 2.0 * da2[i] + 2.0 * da3[i] + da4[i]) * (dt * sixth);
        }
    }
    if tree.links.first().is_some_and(|link| link.mocap) {
        let root_nq = tree.links[0].joint.nq();
        let root_nv = tree.links[0].joint.nv();
        tree.q[..root_nq].copy_from_slice(&q0[..root_nq]);
        tree.qdot[..root_nv].copy_from_slice(&qdot0[..root_nv]);
    }
    // Renormalize free-root and ball-joint quaternions once at step end
    // (mirrors tier 1; mid-RK4 renormalization would break the linearity the
    // integrator relies on).
    if let JointKind::Free = tree.links[0].joint {
        let q = Quat::new(tree.q[3], tree.q[4], tree.q[5], tree.q[6]).renormalize();
        tree.q[3] = q.x;
        tree.q[4] = q.y;
        tree.q[5] = q.z;
        tree.q[6] = q.w;
    }
    for i in 0..tree.links.len() {
        if let JointKind::Ball { .. } = tree.links[i].joint {
            let off = tree.q_offset[i];
            let q = Quat::new(
                tree.q[off],
                tree.q[off + 1],
                tree.q[off + 2],
                tree.q[off + 3],
            )
            .renormalize();
            tree.q[off] = q.x;
            tree.q[off + 1] = q.y;
            tree.q[off + 2] = q.z;
            tree.q[off + 3] = q.w;
        }
    }
    // Filter actuators keep the existing end-of-step Euler update. Muscle
    // activations were advanced at each RK4 stage and are already final.
    tree.integrate_filter_activations(dt);
}

/// One semi-implicit Euler step on a tree.
///
/// The forward-kinematics and external-wrench callback observe the state at
/// the start of the step. ABA computes acceleration, then the new velocity
/// integrates generalized positions. Joint damping is folded into the mass
/// solve for both modes. `implicit_fast` additionally folds actuator velocity
/// derivatives and does not differentiate Coriolis terms.
pub fn euler_step<F>(
    tree: &mut Tree,
    gravity: Vec3,
    dt: f32,
    implicit_fast: bool,
    compute_ext_wrenches: F,
) where
    F: FnMut(&Tree) -> ExternalWrenches,
{
    let mut workspace = AbaWorkspace::new(tree.links.len());
    euler_step_with_workspace(
        tree,
        gravity,
        dt,
        implicit_fast,
        compute_ext_wrenches,
        &mut workspace,
    );
}

pub(crate) fn euler_step_with_workspace<F>(
    tree: &mut Tree,
    gravity: Vec3,
    dt: f32,
    implicit_fast: bool,
    mut compute_ext_wrenches: F,
    workspace: &mut AbaWorkspace,
) where
    F: FnMut(&Tree) -> ExternalWrenches,
{
    let poses = forward_kinematics(tree);
    let ext = compute_ext_wrenches(tree);
    let mode = if implicit_fast {
        VelocityImplicit::JointDampingAndActuators { dt }
    } else {
        VelocityImplicit::JointDamping { dt }
    };
    let qddot = aba_with_velocity_implicit_workspace(tree, &poses, gravity, &ext, mode, workspace);
    let mocap_root = tree.links.first().is_some_and(|link| link.mocap);
    for (i, qdot) in tree.qdot.iter_mut().enumerate() {
        let in_mocap_root = mocap_root && i < tree.links[0].joint.nv();
        if !in_mocap_root {
            *qdot += qddot[i] * dt;
        }
    }
    integrate_tree_positions(tree, dt, mocap_root);
    tree.integrate_activations(dt);
}

fn integrate_tree_positions(tree: &mut Tree, dt: f32, mocap_root: bool) {
    for (i, link) in tree.links.iter().enumerate() {
        if mocap_root && i == 0 {
            continue;
        }
        match link.joint {
            JointKind::Free => {
                let qoff = tree.q_offset[i];
                let voff = tree.v_offset[i];
                let orientation = Quat::new(
                    tree.q[qoff + 3],
                    tree.q[qoff + 4],
                    tree.q[qoff + 5],
                    tree.q[qoff + 6],
                );
                let omega = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let velocity = Vec3::new(
                    tree.qdot[voff + 3],
                    tree.qdot[voff + 4],
                    tree.qdot[voff + 5],
                );
                let position_delta = orientation.rotate(velocity) * dt;
                tree.q[qoff] += position_delta.x;
                tree.q[qoff + 1] += position_delta.y;
                tree.q[qoff + 2] += position_delta.z;
                let next = orientation.integrate_body_angular_velocity(omega, dt);
                tree.q[qoff + 3] = next.x;
                tree.q[qoff + 4] = next.y;
                tree.q[qoff + 5] = next.z;
                tree.q[qoff + 6] = next.w;
            }
            JointKind::Fixed => {}
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                tree.q[tree.q_offset[i]] += tree.qdot[tree.v_offset[i]] * dt;
            }
            JointKind::Ball { .. } => {
                let qoff = tree.q_offset[i];
                let voff = tree.v_offset[i];
                let orientation = Quat::new(
                    tree.q[qoff],
                    tree.q[qoff + 1],
                    tree.q[qoff + 2],
                    tree.q[qoff + 3],
                );
                let omega = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let next = orientation.integrate_body_angular_velocity(omega, dt);
                tree.q[qoff] = next.x;
                tree.q[qoff + 1] = next.y;
                tree.q[qoff + 2] = next.z;
                tree.q[qoff + 3] = next.w;
            }
        }
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
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                let qoff = tree.q_offset[i];
                let voff = tree.v_offset[i];
                dq[qoff] = tree.qdot[voff];
            }
            JointKind::Ball { .. } => {
                let qoff = tree.q_offset[i];
                let voff = tree.v_offset[i];
                let q_ball = Quat::new(
                    tree.q[qoff],
                    tree.q[qoff + 1],
                    tree.q[qoff + 2],
                    tree.q[qoff + 3],
                );
                let omega_body =
                    Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                // dq/dt = 0.5 * q * (omega_body, 0)  (same body-frame right-
                // multiplication as the free root; see math::Quat::derivative).
                let dqori = q_ball.derivative(omega_body);
                dq[qoff] = dqori.x;
                dq[qoff + 1] = dqori.y;
                dq[qoff + 2] = dqori.z;
                dq[qoff + 3] = dqori.w;
            }
        }
    }
    (dq, dv)
}

/// Advance the tree's mutable state fields in-place to
/// `origin + (dq, dv, da) * dt`. The original `tree_advance` returned a
/// freshly cloned `Tree` so RK4 stages could hold four owned trees; this
/// path mutates the working tree so all four stages share one allocation
/// footprint. Does not renormalize the quaternion (mid-RK4 stages preserve
/// linearity).
#[allow(clippy::too_many_arguments)]
fn advance_tree_state(
    tree: &mut Tree,
    q0: &[f32],
    qdot0: &[f32],
    act0: &[f32],
    dq: &[f32],
    dv: &[f32],
    da: &[f32],
    dt: f32,
) {
    for j in 0..q0.len() {
        tree.q[j] = q0[j] + dq[j] * dt;
    }
    for j in 0..qdot0.len() {
        tree.qdot[j] = qdot0[j] + dv[j] * dt;
    }
    for (i, actuator) in tree.actuators.iter_mut().enumerate() {
        if matches!(actuator.dyn_type, crate::actuator::DynType::Muscle) {
            actuator.act = act0[i] + da[i] * dt;
        }
    }
}

fn muscle_activation_deriv(tree: &Tree) -> Vec<f32> {
    tree.actuators
        .iter()
        .map(|actuator| {
            if matches!(actuator.dyn_type, crate::actuator::DynType::Muscle) {
                crate::actuator::muscle_dynamics(
                    actuator.clamped_ctrl(),
                    actuator.act,
                    actuator.muscle_dyn_prm,
                )
            } else {
                0.0
            }
        })
        .collect()
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
