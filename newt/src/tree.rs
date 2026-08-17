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

use crate::actuator::{Actuator, clamp_symmetric};
use crate::joint::{JointKind, JointLimit};
use crate::math::{Mat3, Quat, Vec3};
use crate::spatial::{Mat6, SpatialForce, SpatialInertia, SpatialMotion, Xform};
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
    out
}

// ---------------------------------------------------------------------------
// ABA — Featherstone's Articulated Body Algorithm
// ---------------------------------------------------------------------------

/// Per-link scratch used by the ABA passes.
///
/// The scalar slots (`s`, `ia_s`, `d`, `tau`, `qddot_joint`) carry the state
/// for single-DOF joints (Hinge, Slide). The ball-joint slots (`s3`,
/// `ia_s3`, `d3_inv`, `tau3`, `qddot3`) carry the 3-column parallel state
/// for the 3-DOF ball joint. Only one set is meaningful per link, decided
/// by the link's `JointKind`; the other is left at its default and never
/// read on that link's ABA arms.
#[derive(Clone, Debug)]
pub(crate) struct AbaWorkspace {
    /// Motion transform from parent body frame to this link's body frame.
    xup: Vec<Xform>,
    /// Joint subspace basis (child-body-frame spatial motion per unit qdot)
    /// for single-DOF joints. Meaningful for hinge/slide; unused for
    /// fixed/free/ball.
    s: Vec<SpatialMotion>,
    /// 3-column joint subspace for ball joints (columns k=0..3 give the
    /// spatial motion per unit `qdot_k` in the child body frame at COM).
    s3: Vec<[SpatialMotion; 3]>,
    /// Per-link spatial velocity in body frame at COM.
    v: Vec<SpatialMotion>,
    /// Per-link coriolis bias `c[i] = v[i] × (S[i] * qdot[i])` (spatial
    /// motion). For a ball joint, `S[i] * qdot[i]` is `Σ_k S_k * qdot_k`
    /// (the 3-DOF joint velocity in body frame).
    c: Vec<SpatialMotion>,
    /// Articulated-body inertia at each link (body frame at COM).
    ia: Vec<Mat6>,
    /// Articulated-body bias force at each link.
    pa: Vec<SpatialForce>,
    /// Per-link `IA[i] * S[i]` for single-DOF joints (hinge/slide).
    ia_s: Vec<SpatialForce>,
    /// Per-link `IA[i] * S_k[i]` for ball joints (3 spatial forces).
    ia_s3: Vec<[SpatialForce; 3]>,
    /// Per-link `Sᵀ IA S + armature` (scalar) for single-DOF joints.
    d: Vec<f32>,
    /// Per-link inverse of `Sᵀ IA S + armature·I₃` (3x3 Mat3) for ball joints.
    d3_inv: Vec<Mat3>,
    /// Per-link joint scalar torque applied at pass-2 for single-DOF joints
    /// (damping, armature-related, limits, external qfrc_applied, actuators).
    /// Cached for pass 3.
    tau: Vec<f32>,
    /// Per-link 3-vector joint torque for ball joints (damping + qfrc_applied).
    tau3: Vec<Vec3>,
    /// Per-link joint qddot (single-DOF, computed in pass 3).
    qddot_joint: Vec<f32>,
    /// Per-link joint qddot (ball, 3-vector).
    qddot_ball: Vec<Vec3>,
    /// Per-link spatial acceleration (computed in pass 3, body frame at COM).
    a: Vec<SpatialMotion>,
}

impl AbaWorkspace {
    pub(crate) fn new(n: usize) -> Self {
        let zero_s3 = [SpatialMotion::ZERO; 3];
        let zero_f3 = [SpatialForce::ZERO; 3];
        Self {
            xup: vec![Xform::IDENTITY; n],
            s: vec![SpatialMotion::ZERO; n],
            s3: vec![zero_s3; n],
            v: vec![SpatialMotion::ZERO; n],
            c: vec![SpatialMotion::ZERO; n],
            ia: vec![Mat6::ZERO; n],
            pa: vec![SpatialForce::ZERO; n],
            ia_s: vec![SpatialForce::ZERO; n],
            ia_s3: vec![zero_f3; n],
            d: vec![0.0; n],
            d3_inv: vec![Mat3::ZERO; n],
            tau: vec![0.0; n],
            tau3: vec![Vec3::ZERO; n],
            qddot_joint: vec![0.0; n],
            qddot_ball: vec![Vec3::ZERO; n],
            a: vec![SpatialMotion::ZERO; n],
        }
    }

    pub(crate) fn ensure_len(&mut self, n: usize) {
        if self.xup.len() != n {
            *self = Self::new(n);
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
    assert_eq!(poses.len(), n);
    assert_eq!(external_wrenches.len(), n);
    workspace.ensure_len(n);
    let w = workspace;

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
                let xup = xup_for_link_hinge(link, axis, q_angle);
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
            JointKind::Slide { axis, .. } => {
                let parent = link.parent.expect("slide must have parent");
                let q_slide = tree.q[tree.q_offset[i]];
                let xup = xup_for_link_slide(link, axis, q_slide);
                w.xup[i] = xup;
                // Slide joint subspace at COM: pure translation along axis.
                // Angular part = 0; linear part = axis (in child body coords).
                let s = SpatialMotion::new(Vec3::ZERO, axis);
                w.s[i] = s;
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let s_qdot = s * qdot_i;
                let v_parent = xup.motion(w.v[parent]);
                w.v[i] = v_parent + s_qdot;
                w.c[i] = w.v[i].cross_motion(s_qdot);
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
                // 3-column joint subspace at COM: S_k = (e_k, r_jc × e_k).
                let r_jc = link.joint_offset_in_child.0;
                let sx = SpatialMotion::new(Vec3::X, r_jc.cross(Vec3::X));
                let sy = SpatialMotion::new(Vec3::Y, r_jc.cross(Vec3::Y));
                let sz = SpatialMotion::new(Vec3::Z, r_jc.cross(Vec3::Z));
                w.s3[i] = [sx, sy, sz];
                let voff = tree.v_offset[i];
                let omega = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let s_qdot = sx * omega.x + sy * omega.y + sz * omega.z;
                let v_parent = xup.motion(w.v[parent]);
                w.v[i] = v_parent + s_qdot;
                w.c[i] = w.v[i].cross_motion(s_qdot);
            }
        }
    }

    // Tendon contributions: compute passive spring/damper AND tendon-
    // attached actuator forces into a per-DOF buffer that pass 2 folds
    // into `tau_scalar` per link. Cached across the whole aba call.
    let mut tendon_qfrc = vec![0.0f32; tree.nv()];
    if !tree.tendons.is_empty() {
        let tendon_state = crate::tendon::accumulate_tendon_passive(tree, poses, &mut tendon_qfrc);
        crate::tendon::accumulate_tendon_actuator_qfrc(tree, &tendon_state, &mut tendon_qfrc);
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
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let q_i = tree.q[tree.q_offset[i]];
                let tau_lim = if tree.disable_penalty_limits {
                    0.0
                } else {
                    joint_limit_scalar_force(q_i, qdot_i, range, limit)
                };
                let mut tau_act = 0.0;
                for act in &tree.actuators {
                    // Tendon-mode actuators are dispatched by
                    // `accumulate_tendon_actuator_qfrc` (fed into
                    // `tendon_qfrc` before pass 2) and MUST NOT double-
                    // count via the joint-link scan.
                    if act.tendon_target.is_none() && act.link_idx == i {
                        tau_act += act.torque(q_i, qdot_i);
                    }
                }
                let tau_scalar = tree.qfrc_applied[tree.v_offset[i]]
                    + tendon_qfrc[tree.v_offset[i]]
                    - damping * qdot_i
                    + tau_lim
                    + tau_act;
                let damping_mass =
                    implicit_mass_damping(tree, i, q_i, qdot_i, damping, velocity_implicit);
                single_dof_pass2(w, tree, i, parent, armature, damping_mass, tau_scalar);
            }
            JointKind::Slide {
                damping,
                armature,
                range,
                limit,
                ..
            } => {
                // Slide's pass-2 arithmetic is IDENTICAL to hinge's — both
                // are 1-DOF with a scalar `Sᵀ IA S + armature`. The only
                // difference is the semantic of `q_i` (radians vs meters)
                // and the sign convention on the limit, both encoded in
                // `joint_limit_scalar_force` and `single_dof_pass2` which
                // work off `w.s[i]` cached from pass 1.
                let parent = link.parent.expect("slide must have parent");
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let q_i = tree.q[tree.q_offset[i]];
                let tau_lim = if tree.disable_penalty_limits {
                    0.0
                } else {
                    joint_limit_scalar_force(q_i, qdot_i, range, limit)
                };
                let mut tau_act = 0.0;
                for act in &tree.actuators {
                    if act.tendon_target.is_none() && act.link_idx == i {
                        tau_act += act.torque(q_i, qdot_i);
                    }
                }
                let tau_scalar = tree.qfrc_applied[tree.v_offset[i]]
                    + tendon_qfrc[tree.v_offset[i]]
                    - damping * qdot_i
                    + tau_lim
                    + tau_act;
                let damping_mass =
                    implicit_mass_damping(tree, i, q_i, qdot_i, damping, velocity_implicit);
                single_dof_pass2(w, tree, i, parent, armature, damping_mass, tau_scalar);
            }
            JointKind::Ball { damping, armature } => {
                let parent = link.parent.expect("ball must have parent");
                let s3 = w.s3[i];
                // Compute IA * S_k for k = 0..3.
                let ia_s3 = [
                    w.ia[i].times_motion(s3[0]),
                    w.ia[i].times_motion(s3[1]),
                    w.ia[i].times_motion(s3[2]),
                ];
                // D = Sᵀ IA S + armature * I₃  (3x3 symmetric).
                // Written out column-major (matches Mat3's layout) so clippy's
                // needless_range_loop lint doesn't fire on nested index loops.
                let damping_mass = match velocity_implicit {
                    VelocityImplicit::Explicit => 0.0,
                    VelocityImplicit::JointDamping { dt }
                    | VelocityImplicit::JointDampingAndActuators { dt } => dt * damping,
                };
                let d_mat = Mat3::new([
                    spatial_dot_ms(s3[0], ia_s3[0]) + armature + damping_mass,
                    spatial_dot_ms(s3[1], ia_s3[0]),
                    spatial_dot_ms(s3[2], ia_s3[0]),
                    spatial_dot_ms(s3[0], ia_s3[1]),
                    spatial_dot_ms(s3[1], ia_s3[1]) + armature + damping_mass,
                    spatial_dot_ms(s3[2], ia_s3[1]),
                    spatial_dot_ms(s3[0], ia_s3[2]),
                    spatial_dot_ms(s3[1], ia_s3[2]),
                    spatial_dot_ms(s3[2], ia_s3[2]) + armature + damping_mass,
                ]);
                let d_inv = d_mat
                    .inverse()
                    .expect("ball articulated-inertia block is singular");
                // Ball joint torque: qfrc_applied - damping * omega (isotropic).
                let voff = tree.v_offset[i];
                let omega = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let tau3 = Vec3::new(
                    tree.qfrc_applied[voff] + tendon_qfrc[voff] - damping * omega.x,
                    tree.qfrc_applied[voff + 1] + tendon_qfrc[voff + 1] - damping * omega.y,
                    tree.qfrc_applied[voff + 2] + tendon_qfrc[voff + 2] - damping * omega.z,
                );

                // p_stage = pA + IA c
                let ia_c = w.ia[i].times_motion(w.c[i]);
                let p_stage = w.pa[i] + ia_c;
                // u_stage = tau3 - Sᵀ p_stage  (3-vector)
                let sp = Vec3::new(
                    spatial_dot_ms(s3[0], p_stage),
                    spatial_dot_ms(s3[1], p_stage),
                    spatial_dot_ms(s3[2], p_stage),
                );
                let u_stage = tau3 - sp;
                // qddot_stage = D⁻¹ u_stage.
                let qdd_stage = d_inv * u_stage;
                // pA_full = p_stage + Σ_k (IA S_k) * qdd_stage[k].
                let pa_full = p_stage
                    + ia_s3[0] * qdd_stage.x
                    + ia_s3[1] * qdd_stage.y
                    + ia_s3[2] * qdd_stage.z;

                // IA_full = IA - (IA S) D⁻¹ (IA S)ᵀ.
                // Write A_col_k = Σ_j (IA S_j) * D_inv[j, k]; then update
                // is Σ_k outer(A_col_k, (IA S_k) as motion).
                let a_col = [
                    ia_s3[0] * d_inv.get(0, 0)
                        + ia_s3[1] * d_inv.get(1, 0)
                        + ia_s3[2] * d_inv.get(2, 0),
                    ia_s3[0] * d_inv.get(0, 1)
                        + ia_s3[1] * d_inv.get(1, 1)
                        + ia_s3[2] * d_inv.get(2, 1),
                    ia_s3[0] * d_inv.get(0, 2)
                        + ia_s3[1] * d_inv.get(1, 2)
                        + ia_s3[2] * d_inv.get(2, 2),
                ];
                let ia_full = w.ia[i]
                    .minus(Mat6::outer(a_col[0], s_force_to_motion(ia_s3[0])))
                    .minus(Mat6::outer(a_col[1], s_force_to_motion(ia_s3[1])))
                    .minus(Mat6::outer(a_col[2], s_force_to_motion(ia_s3[2])));

                // Cache for pass 3.
                w.ia_s3[i] = ia_s3;
                w.d3_inv[i] = d_inv;
                w.tau3[i] = tau3;

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
            // At root: IA[0] a[0] = tau_free_gen - pA[0]. The generalized
            // free-root solver force is a spatial force in body-frame-at-COM
            // coordinates conjugate to the 6 slot layout (ω_body, v_body).
            // The solver path sets `disable_penalty_limits`, which gates this
            // channel on. Penalty callers keep the old free-root behavior.
            let applied_free = tree.disable_penalty_limits;
            let free_force = |slot: usize| {
                if applied_free {
                    tree.qfrc_applied[slot]
                } else {
                    0.0
                }
            };
            let tau_free = SpatialForce::new(
                Vec3::new(
                    free_force(0) + tendon_qfrc[0],
                    free_force(1) + tendon_qfrc[1],
                    free_force(2) + tendon_qfrc[2],
                ),
                Vec3::new(
                    free_force(3) + tendon_qfrc[3],
                    free_force(4) + tendon_qfrc[4],
                    free_force(5) + tendon_qfrc[5],
                ),
            );
            let rhs = SpatialForce::new(
                tau_free.torque - w.pa[0].torque,
                tau_free.linear - w.pa[0].linear,
            );
            let root_damping = tree.links[0].free_damping;
            let a0 = if root_damping == 0.0 {
                w.ia[0]
                    .solve(rhs)
                    .expect("root articulated inertia is singular — degenerate mass distribution?")
            } else {
                let voff = tree.v_offset[0];
                let qdot = SpatialMotion::new(
                    Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]),
                    Vec3::new(
                        tree.qdot[voff + 3],
                        tree.qdot[voff + 4],
                        tree.qdot[voff + 5],
                    ),
                );
                let damped_rhs = rhs
                    - SpatialForce::new(qdot.angular * root_damping, qdot.linear * root_damping);
                let damping_mass = match velocity_implicit {
                    VelocityImplicit::Explicit => 0.0,
                    VelocityImplicit::JointDamping { dt }
                    | VelocityImplicit::JointDampingAndActuators { dt } => dt * root_damping,
                };
                let mut root_ia = w.ia[0];
                for i in 0..6 {
                    root_ia.rows[i][i] += damping_mass;
                }
                root_ia
                    .solve(damped_rhs)
                    .expect("root articulated inertia is singular — degenerate mass distribution?")
            };
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
        JointKind::Hinge { .. } | JointKind::Slide { .. } | JointKind::Ball { .. } => {
            unreachable!("only Free/Fixed joints are valid at the root — push_link rejects others")
        }
    }

    for i in 1..n {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Fixed => {
                let parent = link.parent.unwrap();
                // No DOF; a[i] = Xup a[parent] + c[i] (c is zero).
                w.a[i] = w.xup[i].motion(w.a[parent]);
            }
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
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
            JointKind::Ball { .. } => {
                let parent = link.parent.unwrap();
                let a_parent_at_child = w.xup[i].motion(w.a[parent]);
                let s3 = w.s3[i];
                let ia = w.ia[i];
                let pa = w.pa[i];
                let acc_prime = a_parent_at_child + w.c[i];
                let inner = ia.times_motion(acc_prime) + pa;
                let s_inner = Vec3::new(
                    spatial_dot_ms(s3[0], inner),
                    spatial_dot_ms(s3[1], inner),
                    spatial_dot_ms(s3[2], inner),
                );
                let u = w.tau3[i] - s_inner;
                let qdd3 = w.d3_inv[i] * u;
                w.qddot_ball[i] = qdd3;
                w.a[i] = acc_prime + s3[0] * qdd3.x + s3[1] * qdd3.y + s3[2] * qdd3.z;
                let voff = tree.v_offset[i];
                qddot[voff] = qdd3.x;
                qddot[voff + 1] = qdd3.y;
                qddot[voff + 2] = qdd3.z;
            }
            JointKind::Free => unreachable!(),
        }
    }

    qddot
}

/// Motion transform from parent body frame to a `Fixed`-jointed child's body
/// frame. Assumes joint orientation offsets are identity (v0 convention).
pub(crate) fn xup_for_link(link: &Link, _unused: f32) -> Xform {
    let (r_pj, _) = link.joint_offset_in_parent;
    let (r_jc, _) = link.joint_offset_in_child;
    let rot_c_from_p = Mat3::IDENTITY;
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
fn single_dof_pass2(
    w: &mut AbaWorkspace,
    _tree: &Tree,
    i: usize,
    parent: usize,
    armature: f32,
    damping_mass: f32,
    tau_scalar: f32,
) {
    let s = w.s[i];
    let ia_s = w.ia[i].times_motion(s);
    let d_scalar = spatial_dot_ms(s, ia_s) + armature + damping_mass;
    // Featherstone's reduced-inertia form; see the derivation comment in
    // docs/joints.md ("the u_stage form").
    let ia_c = w.ia[i].times_motion(w.c[i]);
    let p_stage = w.pa[i] + ia_c;
    let s_dot_p_stage = spatial_dot_ms(s, p_stage);
    let u_stage = tau_scalar - s_dot_p_stage;
    let pa_full = p_stage + ia_s * (u_stage / d_scalar);

    let outer = Mat6::outer(ia_s, s_force_to_motion(ia_s));
    let ia_full = w.ia[i].minus(scale_mat6(outer, 1.0 / d_scalar));

    w.ia_s[i] = ia_s;
    w.d[i] = d_scalar;
    w.tau[i] = tau_scalar;

    let ia_parent_contrib = ia_full.pull_back(w.xup[i]);
    w.ia[parent] = w.ia[parent].plus(ia_parent_contrib);
    let pa_parent_contrib = w.xup[i].transpose_force(pa_full);
    w.pa[parent] = w.pa[parent] + pa_parent_contrib;
}

/// Range-limit spring-damper generalized force for a single-DOF joint
/// (hinge or slide). Zero inside `range` (or unconditionally if
/// `range = None`). Outside, a one-sided spring pulls the joint back
/// toward the limit; a damping term on `qdot` activates too (also one-
/// sided so it doesn't resist motion INTO the allowed range).
fn joint_limit_scalar_force(
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

/// Reinterpret a spatial force's `(torque, linear)` components as a spatial
/// motion's `(angular, linear)` for use as the row vector in an outer
/// product. This is purely a packing convenience — the outer product
/// produces a 6x6 matrix regardless of which type is called "row" or
/// "column". Using [`Mat6::outer`] directly with two `SpatialForce`s would
/// require another wrapper.
#[inline]
pub(crate) fn s_force_to_motion(f: SpatialForce) -> SpatialMotion {
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
    let s0 = tree.clone();

    // k1
    let poses1 = forward_kinematics(&s0);
    let ext1 = compute_ext_wrenches(&s0);
    let k1 = aba_with_velocity_implicit_workspace(
        &s0,
        &poses1,
        gravity,
        &ext1,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq1, dv1) = tree_deriv(&s0, &k1);
    let da1 = muscle_activation_deriv(&s0);

    // k2 at s0 + k1 * dt/2
    let mut s1 = tree_advance(&s0, &dq1, &dv1, dt * 0.5);
    advance_muscle_activation(&mut s1, &s0, &da1, dt * 0.5);
    let poses2 = forward_kinematics(&s1);
    let ext2 = compute_ext_wrenches(&s1);
    let k2 = aba_with_velocity_implicit_workspace(
        &s1,
        &poses2,
        gravity,
        &ext2,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq2, dv2) = tree_deriv(&s1, &k2);
    let da2 = muscle_activation_deriv(&s1);

    // k3 at s0 + k2 * dt/2
    let mut s2 = tree_advance(&s0, &dq2, &dv2, dt * 0.5);
    advance_muscle_activation(&mut s2, &s0, &da2, dt * 0.5);
    let poses3 = forward_kinematics(&s2);
    let ext3 = compute_ext_wrenches(&s2);
    let k3 = aba_with_velocity_implicit_workspace(
        &s2,
        &poses3,
        gravity,
        &ext3,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq3, dv3) = tree_deriv(&s2, &k3);
    let da3 = muscle_activation_deriv(&s2);

    // k4 at s0 + k3 * dt
    let mut s3 = tree_advance(&s0, &dq3, &dv3, dt);
    advance_muscle_activation(&mut s3, &s0, &da3, dt);
    let poses4 = forward_kinematics(&s3);
    let ext4 = compute_ext_wrenches(&s3);
    let k4 = aba_with_velocity_implicit_workspace(
        &s3,
        &poses4,
        gravity,
        &ext4,
        VelocityImplicit::Explicit,
        workspace,
    );
    let (dq4, dv4) = tree_deriv(&s3, &k4);
    let da4 = muscle_activation_deriv(&s3);

    // Combine and write back into tree.
    let sixth = 1.0 / 6.0;
    for j in 0..s0.q.len() {
        tree.q[j] = s0.q[j] + (dq1[j] + 2.0 * dq2[j] + 2.0 * dq3[j] + dq4[j]) * (dt * sixth);
    }
    for j in 0..s0.qdot.len() {
        tree.qdot[j] = s0.qdot[j] + (dv1[j] + 2.0 * dv2[j] + 2.0 * dv3[j] + dv4[j]) * (dt * sixth);
    }
    for (i, actuator) in tree.actuators.iter_mut().enumerate() {
        if matches!(actuator.dyn_type, crate::actuator::DynType::Muscle) {
            actuator.act = s0.actuators[i].act
                + (da1[i] + 2.0 * da2[i] + 2.0 * da3[i] + da4[i]) * (dt * sixth);
        }
    }
    if s0.links.first().is_some_and(|link| link.mocap) {
        let root_nq = s0.links[0].joint.nq();
        let root_nv = s0.links[0].joint.nv();
        tree.q[..root_nq].copy_from_slice(&s0.q[..root_nq]);
        tree.qdot[..root_nv].copy_from_slice(&s0.qdot[..root_nv]);
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

fn advance_muscle_activation(tree: &mut Tree, origin: &Tree, deriv: &[f32], dt: f32) {
    for (i, actuator) in tree.actuators.iter_mut().enumerate() {
        if matches!(actuator.dyn_type, crate::actuator::DynType::Muscle) {
            actuator.act = origin.actuators[i].act + deriv[i] * dt;
        }
    }
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
