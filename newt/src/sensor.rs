//! Sensor battery (v1 tier 6).
//!
//! Sensors are declared on the [`crate::world::World`] and evaluated AFTER
//! each [`crate::world::World::step`]. Each sensor produces a fixed-width
//! contribution to a flat `sensordata: Vec<f32>` vector; offsets are pinned
//! by declaration order, so a scene's sensor layout is stable across steps
//! and across platforms.
//!
//! # Types
//!
//! | Kind | Dim | Semantics |
//! |------|-----|-----------|
//! | [`SensorKind::JointPos`] | 1 | Hinge angle (rad) or slide displacement (m). |
//! | [`SensorKind::JointVel`] | 1 | Hinge rate (rad/s) or slide rate (m/s). |
//! | [`SensorKind::BallQuat`] | 4 | Ball joint's child-relative-to-parent quaternion `(x, y, z, w)`. |
//! | [`SensorKind::BallAngVel`] | 3 | Ball joint's body-frame angular velocity ω (rad/s). |
//! | [`SensorKind::FramePos`] | 3 | World-frame position of the site anchor point. |
//! | [`SensorKind::FrameQuat`] | 4 | Site frame's orientation in world `(x, y, z, w)`. |
//! | [`SensorKind::Gyro`] | 3 | Site-frame angular velocity of the site's parent link. |
//! | [`SensorKind::Accelerometer`] | 3 | Proper acceleration in the site frame (linear specific force). |
//! | [`SensorKind::Touch`] | 1 | Sum of penalty normal-force magnitudes on the designated geom. |
//! | [`SensorKind::Force`] | 3 | Force transmitted through a link's parent joint, in the child body frame. |
//! | [`SensorKind::Torque`] | 3 | Torque transmitted through a link's parent joint, at the joint anchor, in the child body frame. |
//!
//! # Evaluation timing (no perturbation)
//!
//! Sensor evaluation is invoked at the end of [`crate::world::World::step`]
//! when `sensors` is non-empty. The evaluation is **strictly observational**
//! — it reads `bodies`, `trees`, `geoms`, and recomputes derived quantities
//! (contacts, wrenches, `qddot`, spatial accelerations) at the post-step
//! state, then writes into `world.sensordata`. It does NOT modify any
//! simulation state, so a scene with sensors declared vs one without takes
//! byte-identical trajectories under the same inputs. When `sensors` is
//! empty, the sensor pipeline is not invoked at all, keeping every pre-v1-
//! tier-6 golden byte-identical.
//!
//! # Accelerometer semantics
//!
//! The accelerometer reports the specific force at the site — what a real
//! IMU would measure. In classical form:
//!
//! ```text
//! a_site_world  = a_com_world + α_world × r + ω_world × (ω_world × r)
//! a_proper      = a_site_world − g_world
//! reading_site  = R_site→world^T · a_proper
//! ```
//!
//! where `r` is the world-frame vector from the parent-body COM to the site
//! anchor, `α_world` and `ω_world` are the link's world-frame angular
//! acceleration and velocity, `a_com_world` is the world-frame linear
//! acceleration of the COM, and `g_world` is the world gravity vector. A
//! static body sitting still under gravity has `a_com_world = 0` (contact
//! cancels weight) so `a_proper = −g_world = (0, 0, +9.81)` — the classic
//! `+g` upward reading in a world-aligned site frame.
//!
//! The post-step accelerations are computed by re-running the same
//! wrench-assembly and forward-dynamics code paths [`crate::world::World::step`]
//! uses at each RK4 sub-stage. For solver-mode `Penalty` that means one
//! `compute_wrenches` per free body + one ABA per tree; for `Pgs` that means
//! one `compute_solver_wrenches` + one ABA per tree. This mirrors MuJoCo's
//! `mj_sensor` running after `mj_forward` on the post-step state.
//!
//! # Force / torque semantics
//!
//! [`SensorKind::Force`] and [`SensorKind::Torque`] report the interaction
//! wrench that the parent link must exert on the specified child link
//! through the joint to produce the observed acceleration. Both are
//! expressed in the child's body frame. Force is translation-invariant
//! (same value at COM and at the joint anchor); torque is reported AT
//! THE JOINT ANCHOR, translated from RNE's per-link COM torque via
//! `τ_joint = τ_com − r_com_to_joint × F`. The sensor recomputes this
//! via RNE on the post-step `(q, qdot, qddot)`.
//!
//! # Touch semantics
//!
//! The touch sensor sums the penalty normal-force magnitude
//! `f_n = max(0, k · pen_eff − c · v_n)` over every contact that involves
//! the designated geom, at the post-step state. This is the same formula
//! [`crate::world::World`] uses for penalty contact wrenches. In `Pgs`
//! mode the PGS solver's own impulse magnitudes may differ instantaneously,
//! but at equilibrium both modes agree with the weight-balance reading
//! (`sum |f_n| ≈ m · g`), which is the classic anchor. Contact detection
//! is unchanged.

use crate::body::Body;
use crate::contact::Contact;
use crate::geom::{Geom, GeomAttach, GeomShape, geom_world_pose};
use crate::joint::JointKind;
use crate::math::{Quat, Vec3};
use crate::spatial::{SpatialMotion, Xform};
use crate::tree::{
    ExternalWrenches, Tree, aba, forward_kinematics, xup_for_link, xup_for_link_ball,
    xup_for_link_hinge, xup_for_link_slide,
};

// ---------------------------------------------------------------------------
// Site frame + attachment
// ---------------------------------------------------------------------------

/// Where a sensor's parent frame lives — a free body or a tree link.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensorAttach {
    /// Free body index in [`crate::world::World::bodies`].
    Body(usize),
    /// `(tree_idx, link_idx)` into [`crate::world::World::trees`].
    Link(usize, usize),
}

/// A site-style local frame anchored to a body or link.
///
/// `local_offset` is expressed in the parent body frame (link COM frame for
/// links, body COM frame for free bodies). `local_orientation` is a
/// quaternion such that a vector in the site's local frame maps to the
/// parent body frame via `v_body = local_orientation · v_site` (the same
/// convention [`crate::model::Site::local_orientation`] uses, so a loaded
/// [`crate::model::Scene`] can hand its `Site` straight into a sensor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SiteFrame {
    pub attach: SensorAttach,
    pub local_offset: Vec3,
    pub local_orientation: Quat,
}

impl SiteFrame {
    /// World-frame `(position, orientation)` of the site given the current
    /// bodies/trees state.
    pub fn world_pose(&self, bodies: &[Body], trees: &[Tree]) -> (Vec3, Quat) {
        let (parent_pos, parent_ori) = parent_pose(self.attach, bodies, trees);
        let pos = parent_pos + parent_ori.rotate(self.local_offset);
        let ori = parent_ori * self.local_orientation;
        (pos, ori)
    }
}

fn parent_pose(attach: SensorAttach, bodies: &[Body], trees: &[Tree]) -> (Vec3, Quat) {
    match attach {
        SensorAttach::Body(i) => (bodies[i].position, bodies[i].orientation),
        SensorAttach::Link(t, l) => forward_kinematics(&trees[t])[l],
    }
}

// ---------------------------------------------------------------------------
// SensorKind + Sensor
// ---------------------------------------------------------------------------

/// Sensor kind. See module docs for semantics per kind.
///
/// Joint sensors are limited to the joint types listed in each variant's
/// docs; the loader / [`crate::world::World::add_sensor`] validates the
/// attachment at construction time so the runtime is index-safe.
#[derive(Clone, Debug, PartialEq)]
pub enum SensorKind {
    /// Hinge angle or slide displacement, one scalar.
    JointPos { tree: usize, link: usize },
    /// Hinge rate or slide rate, one scalar.
    JointVel { tree: usize, link: usize },
    /// Ball joint's `(x, y, z, w)` quaternion — four scalars.
    BallQuat { tree: usize, link: usize },
    /// Ball joint's body-frame angular velocity — three scalars.
    BallAngVel { tree: usize, link: usize },
    /// World-frame position of the site anchor.
    FramePos(SiteFrame),
    /// Site frame's orientation in world — four scalars `(x, y, z, w)`.
    FrameQuat(SiteFrame),
    /// Site-frame angular velocity of the site's parent body.
    Gyro(SiteFrame),
    /// Proper (specific-force) acceleration in the site frame.
    Accelerometer(SiteFrame),
    /// Sum of penalty normal-force magnitudes on this geom.
    Touch { geom: usize },
    /// Interaction force at the child link's parent-joint connection,
    /// in the child body frame (translation-invariant).
    Force { tree: usize, link: usize },
    /// Interaction torque at the child link's parent-joint connection,
    /// reported at the joint anchor in the child body frame.
    Torque { tree: usize, link: usize },
    /// Tendon length `L` — one scalar per sensor. Consumes
    /// [`crate::tendon::tendon_kinematics`] on the referenced tendon at
    /// the post-step state.
    TendonPos { tree: usize, tendon: usize },
    /// Tendon rate `Ldot` — one scalar per sensor.
    TendonVel { tree: usize, tendon: usize },
    /// Site linear velocity expressed in the site frame.
    Velocimeter(SiteFrame),
    /// Global magnetic field expressed in the site frame.
    Magnetometer(SiteFrame),
    /// Distance from the site along local +Z to the nearest geom, or -1.
    Rangefinder(SiteFrame),
    /// World-frame COM of the subtree rooted at a link.
    SubtreeCom { tree: usize, link: usize },
    /// World-frame linear velocity of a site.
    FrameLinVel(SiteFrame),
    /// World-frame angular velocity of a site parent body.
    FrameAngVel(SiteFrame),
}

impl SensorKind {
    /// Number of scalars this sensor contributes to `sensordata`.
    pub fn dim(&self) -> usize {
        match self {
            SensorKind::JointPos { .. }
            | SensorKind::JointVel { .. }
            | SensorKind::Touch { .. }
            | SensorKind::TendonPos { .. }
            | SensorKind::TendonVel { .. } => 1,
            SensorKind::BallAngVel { .. }
            | SensorKind::FramePos(_)
            | SensorKind::Gyro(_)
            | SensorKind::Accelerometer(_)
            | SensorKind::Force { .. }
            | SensorKind::Torque { .. }
            | SensorKind::Velocimeter(_)
            | SensorKind::Magnetometer(_)
            | SensorKind::SubtreeCom { .. }
            | SensorKind::FrameLinVel(_)
            | SensorKind::FrameAngVel(_) => 3,
            SensorKind::Rangefinder(_) => 1,
            SensorKind::BallQuat { .. } | SensorKind::FrameQuat(_) => 4,
        }
    }
}

/// A named sensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Sensor {
    pub name: String,
    pub kind: SensorKind,
}

/// A validation error for sensor construction. Returned by
/// [`Sensor::validate`] and by [`crate::world::World::add_sensor`] when the
/// caller passes an ill-formed sensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SensorError(pub String);

impl std::fmt::Display for SensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for SensorError {}

impl Sensor {
    /// Validate this sensor against the given world state. Called by
    /// [`crate::world::World::add_sensor`] and by the JSON loader.
    pub fn validate(
        &self,
        bodies: &[Body],
        trees: &[Tree],
        geoms: &[Geom],
    ) -> Result<(), SensorError> {
        match &self.kind {
            SensorKind::JointPos { tree, link } | SensorKind::JointVel { tree, link } => {
                let l = check_tree_link(*tree, *link, trees)?;
                match l.joint {
                    JointKind::Hinge { .. } | JointKind::Slide { .. } => Ok(()),
                    other => Err(SensorError(format!(
                        "jointpos/jointvel requires a hinge or slide (got {other:?}); \
                         use ballquat/ballangvel for ball joints"
                    ))),
                }
            }
            SensorKind::BallQuat { tree, link } | SensorKind::BallAngVel { tree, link } => {
                let l = check_tree_link(*tree, *link, trees)?;
                match l.joint {
                    JointKind::Ball { .. } => Ok(()),
                    other => Err(SensorError(format!(
                        "ballquat/ballangvel requires a ball joint (got {other:?})"
                    ))),
                }
            }
            SensorKind::FramePos(s)
            | SensorKind::FrameQuat(s)
            | SensorKind::Gyro(s)
            | SensorKind::Accelerometer(s)
            | SensorKind::Velocimeter(s)
            | SensorKind::Magnetometer(s)
            | SensorKind::Rangefinder(s)
            | SensorKind::FrameLinVel(s)
            | SensorKind::FrameAngVel(s) => check_site(s, bodies, trees),
            SensorKind::Touch { geom } => {
                if *geom >= geoms.len() {
                    return Err(SensorError(format!(
                        "touch geom index {geom} out of range ({} geoms)",
                        geoms.len()
                    )));
                }
                Ok(())
            }
            SensorKind::Force { tree, link } | SensorKind::Torque { tree, link } => {
                let l = check_tree_link(*tree, *link, trees)?;
                if l.parent.is_none() {
                    return Err(SensorError(
                        "force/torque sensor requires a non-root link (needs a parent joint)"
                            .to_string(),
                    ));
                }
                Ok(())
            }
            SensorKind::TendonPos { tree, tendon } | SensorKind::TendonVel { tree, tendon } => {
                if *tree >= trees.len() {
                    return Err(SensorError(format!(
                        "tendon sensor tree {tree} out of range ({} trees)",
                        trees.len()
                    )));
                }
                if *tendon >= trees[*tree].tendons.len() {
                    return Err(SensorError(format!(
                        "tendon sensor: tendon index {tendon} out of range for tree {tree} \
                         ({} tendons)",
                        trees[*tree].tendons.len()
                    )));
                }
                Ok(())
            }
            SensorKind::SubtreeCom { tree, link } => {
                check_tree_link(*tree, *link, trees).map(|_| ())
            }
        }
    }
}

fn check_tree_link(
    tree: usize,
    link: usize,
    trees: &[Tree],
) -> Result<&crate::tree::Link, SensorError> {
    if tree >= trees.len() {
        return Err(SensorError(format!(
            "tree index {tree} out of range ({} trees)",
            trees.len()
        )));
    }
    if link >= trees[tree].links.len() {
        return Err(SensorError(format!(
            "link index {link} out of range for tree {tree} ({} links)",
            trees[tree].links.len()
        )));
    }
    Ok(&trees[tree].links[link])
}

fn check_site(s: &SiteFrame, bodies: &[Body], trees: &[Tree]) -> Result<(), SensorError> {
    match s.attach {
        SensorAttach::Body(i) => {
            if i >= bodies.len() {
                return Err(SensorError(format!(
                    "site body index {i} out of range ({} bodies)",
                    bodies.len()
                )));
            }
        }
        SensorAttach::Link(t, l) => {
            check_tree_link(t, l, trees)?;
        }
    }
    // Reject zero-norm orientation up-front (rotate() on a zero quat is
    // undefined and would silently produce garbage sensor readings).
    if s.local_orientation.norm_squared() < 1e-12 {
        return Err(SensorError(
            "site local_orientation must be a non-zero quaternion".to_string(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sensor bank on the world — offsets + data
// ---------------------------------------------------------------------------

/// Sensor registry attached to a [`crate::world::World`]. Holds the ordered
/// [`Sensor`] list, the flat `sensordata` output vector, and the per-sensor
/// offsets into that vector. Layout is fixed by declaration order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SensorBank {
    pub sensors: Vec<Sensor>,
    /// Start index of each sensor's contribution in `data`. Same length as
    /// `sensors`; `offsets[i] + sensors[i].kind.dim() == offsets[i+1]` (or
    /// `data.len()` for the last).
    pub offsets: Vec<usize>,
    /// Flat sensor output vector. Length = sum of every sensor's `dim()`.
    /// Zeroed at construction; filled by [`evaluate`].
    pub data: Vec<f32>,
}

impl SensorBank {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a validated sensor to the bank. Assumes the caller (World)
    /// has already run [`Sensor::validate`]. Extends `data` with zeros.
    pub fn push(&mut self, sensor: Sensor) {
        let off = self.data.len();
        let dim = sensor.kind.dim();
        self.offsets.push(off);
        self.sensors.push(sensor);
        self.data.extend(std::iter::repeat_n(0.0, dim));
    }

    /// Slice of `data` for one sensor, or `None` if the index is bad.
    pub fn slice(&self, idx: usize) -> Option<&[f32]> {
        let off = *self.offsets.get(idx)?;
        let dim = self.sensors[idx].kind.dim();
        self.data.get(off..off + dim)
    }
}

// ---------------------------------------------------------------------------
// Post-step evaluation
// ---------------------------------------------------------------------------

/// Bundle of world state used by [`evaluate`]. Kept as a struct so callers
/// don't have to thread six borrows through the entrypoint (and so we can
/// grow the input in future tiers — e.g. equalities, mocap — without
/// churning the signature).
pub struct SensorInputs<'a> {
    pub bodies: &'a [Body],
    pub trees: &'a [Tree],
    pub geoms: &'a [Geom],
    pub meshes: &'a [crate::geom::ConvexMesh],
    pub gravity: Vec3,
    pub magnetic_field: Vec3,
    pub dt: f32,
    /// Per-free-body external world-frame wrench `(force, torque_about_com)`.
    /// The world assembles this the same way it does for its RK4 sub-stages
    /// (penalty contacts or PGS solver wrenches). Length == `bodies.len()`.
    pub body_wrenches: Vec<(Vec3, Vec3)>,
    /// Per-tree per-link external world-frame wrench. Same layout as ABA's
    /// `ExternalWrenches`. Trees[i] uses `tree_wrenches[i]`.
    pub tree_wrenches: Vec<ExternalWrenches>,
    /// Post-step contact list — same enumeration order the world's step
    /// pipeline uses. Passed in so `evaluate` doesn't have to redetect.
    pub contacts: Vec<Contact>,
    /// Per-contact normal force magnitude (N). Under `Penalty` this is the
    /// penalty-formula reading; under `Pgs` it is the PGS-computed contact
    /// impulse divided by `dt`. Same length and order as `contacts`.
    pub contact_normal_forces: Vec<f32>,
}

/// Evaluate every sensor in `bank` against the state described by `inputs`
/// and write the readings into `bank.data`. Never mutates `bodies`,
/// `trees`, `geoms`, or `meshes`.
pub fn evaluate(bank: &mut SensorBank, inputs: &SensorInputs<'_>) {
    // Pre-compute per-tree link poses (world) and per-tree qddot only when
    // sensors need them. Every kind that reads spatial state needs poses;
    // accelerometer / force / torque need qddot too.
    let n_trees = inputs.trees.len();
    let tree_poses: Vec<Vec<(Vec3, Quat)>> = (0..n_trees)
        .map(|t| forward_kinematics(&inputs.trees[t]))
        .collect();

    // Compute qddot per tree only when a sensor needs the *acceleration*
    // (accelerometer / force / torque). Gyro and framepos live on `v`
    // alone, so a scene with only those still runs a `compute_link_va`
    // pass with zero qddot below.
    let needs_qddot = bank.sensors.iter().any(|s| {
        matches!(
            s.kind,
            SensorKind::Accelerometer(_) | SensorKind::Force { .. } | SensorKind::Torque { .. }
        )
    });
    let needs_va = needs_qddot
        || bank.sensors.iter().any(|s| {
            matches!(
                s.kind,
                SensorKind::Gyro(_)
                    | SensorKind::Velocimeter(_)
                    | SensorKind::FrameLinVel(_)
                    | SensorKind::FrameAngVel(_)
            )
        });
    let tree_qddot: Vec<Vec<f32>> = if needs_qddot {
        (0..n_trees)
            .map(|t| {
                aba(
                    &inputs.trees[t],
                    &tree_poses[t],
                    inputs.gravity,
                    &inputs.tree_wrenches[t],
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    // Per-link spatial motion (v, a) in body frame at COM. When `needs_va`
    // but not `needs_qddot`, we pass a zero-qddot vector (a[i] then falls
    // out as zero but v[i] is still correct — v only depends on qdot).
    let tree_va: Vec<Vec<(SpatialMotion, SpatialMotion)>> = if needs_va {
        (0..n_trees)
            .map(|t| {
                let qdd_ref = if needs_qddot {
                    tree_qddot[t].clone()
                } else {
                    vec![0.0f32; inputs.trees[t].nv()]
                };
                compute_link_va(&inputs.trees[t], &qdd_ref)
            })
            .collect()
    } else {
        Vec::new()
    };
    // Per-link RNE wrench `f[i]` (child-body-frame at COM) — the interaction
    // wrench transmitted from parent to child. Computed only when a force/
    // torque sensor asks. RNE also produces `tau` but we don't need it.
    let tree_link_f: Vec<Vec<crate::spatial::SpatialForce>> = if bank
        .sensors
        .iter()
        .any(|s| matches!(s.kind, SensorKind::Force { .. } | SensorKind::Torque { .. }))
    {
        (0..n_trees)
            .map(|t| {
                compute_link_wrenches(
                    &inputs.trees[t],
                    &tree_poses[t],
                    &tree_qddot[t],
                    inputs.gravity,
                    &inputs.tree_wrenches[t],
                )
            })
            .collect()
    } else {
        Vec::new()
    };

    // Per-free-body classical (linear_world at COM, angular_body) acceleration
    // — from Newton's 2nd law + Euler's equations under the assembled wrench.
    let body_accels: Vec<(Vec3, Vec3)> = if needs_qddot {
        inputs
            .bodies
            .iter()
            .enumerate()
            .map(|(i, b)| {
                let (f_ext, tau_ext) = inputs.body_wrenches[i];
                let a_com_world = inputs.gravity + f_ext / b.mass;
                let tau_body = b.orientation.inverse_rotate(tau_ext);
                let iw = b.inertia_body * b.angular_velocity_body;
                let gyroscopic = -b.angular_velocity_body.cross(iw);
                let alpha_body = b.inertia_body_inverse * (tau_body + gyroscopic);
                (a_com_world, alpha_body)
            })
            .collect()
    } else {
        Vec::new()
    };

    // Contact list + per-contact normal forces are supplied by the caller
    // (the world computes them once from the solver-mode-appropriate
    // pipeline). Sensor eval reads directly.

    // Evaluate each sensor.
    for (i, sensor) in bank.sensors.iter().enumerate() {
        let off = bank.offsets[i];
        let dim = sensor.kind.dim();
        let out = &mut bank.data[off..off + dim];
        match &sensor.kind {
            SensorKind::JointPos { tree, link } => {
                let t = &inputs.trees[*tree];
                out[0] = t.q[t.q_offset[*link]];
            }
            SensorKind::JointVel { tree, link } => {
                let t = &inputs.trees[*tree];
                out[0] = t.qdot[t.v_offset[*link]];
            }
            SensorKind::BallQuat { tree, link } => {
                let t = &inputs.trees[*tree];
                let o = t.q_offset[*link];
                out[0] = t.q[o];
                out[1] = t.q[o + 1];
                out[2] = t.q[o + 2];
                out[3] = t.q[o + 3];
            }
            SensorKind::BallAngVel { tree, link } => {
                let t = &inputs.trees[*tree];
                let o = t.v_offset[*link];
                out[0] = t.qdot[o];
                out[1] = t.qdot[o + 1];
                out[2] = t.qdot[o + 2];
            }
            SensorKind::FramePos(s) => {
                let (p, _) = site_world_pose(s, inputs.bodies, &tree_poses);
                out[0] = p.x;
                out[1] = p.y;
                out[2] = p.z;
            }
            SensorKind::FrameQuat(s) => {
                let (_, q) = site_world_pose(s, inputs.bodies, &tree_poses);
                out[0] = q.x;
                out[1] = q.y;
                out[2] = q.z;
                out[3] = q.w;
            }
            SensorKind::Gyro(s) => {
                let (omega_body_parent, _) =
                    parent_body_velocities(s.attach, inputs.bodies, &tree_va);
                // omega vector rotated into site frame: v_site = q^T v_body.
                let omega_site = s.local_orientation.inverse_rotate(omega_body_parent);
                out[0] = omega_site.x;
                out[1] = omega_site.y;
                out[2] = omega_site.z;
            }
            SensorKind::Accelerometer(s) => {
                let reading = accelerometer_reading(
                    s,
                    inputs.bodies,
                    inputs.trees,
                    &tree_poses,
                    &tree_va,
                    &body_accels,
                    inputs.gravity,
                );
                out[0] = reading.x;
                out[1] = reading.y;
                out[2] = reading.z;
            }
            SensorKind::Touch { geom } => {
                out[0] = touch_reading(*geom, &inputs.contacts, &inputs.contact_normal_forces);
            }
            SensorKind::Force { tree, link } => {
                // Force is translation-invariant, so the value at COM
                // equals the value at the joint anchor. Report directly.
                let f = tree_link_f[*tree][*link];
                out[0] = f.linear.x;
                out[1] = f.linear.y;
                out[2] = f.linear.z;
            }
            SensorKind::Torque { tree, link } => {
                // Translate the RNE per-link wrench from COM to the joint
                // anchor so a static horizontal arm reads the classical
                // gravity-moment τ = m · g · (rod half-length) about the
                // hinge axis. Anchor = `joint_offset_in_child.0` (child
                // body coords), so
                //   τ_at_joint = τ_at_COM − r_com_to_joint × F.
                let f = tree_link_f[*tree][*link];
                let r = inputs.trees[*tree].links[*link].joint_offset_in_child.0;
                let tau_joint = f.torque - r.cross(f.linear);
                out[0] = tau_joint.x;
                out[1] = tau_joint.y;
                out[2] = tau_joint.z;
            }
            SensorKind::TendonPos { tree, tendon } => {
                let t = &inputs.trees[*tree];
                let kin =
                    crate::tendon::tendon_kinematics(&t.tendons[*tendon], t, &tree_poses[*tree]);
                out[0] = kin.length;
            }
            SensorKind::TendonVel { tree, tendon } => {
                let t = &inputs.trees[*tree];
                let kin =
                    crate::tendon::tendon_kinematics(&t.tendons[*tendon], t, &tree_poses[*tree]);
                out[0] = kin.velocity;
            }
            SensorKind::Velocimeter(s) => {
                let (linear, _) = site_velocity(s, inputs.bodies, &tree_poses, &tree_va);
                let (_, site_orientation) = site_world_pose(s, inputs.bodies, &tree_poses);
                let value = site_orientation.inverse_rotate(linear);
                out[0] = value.x;
                out[1] = value.y;
                out[2] = value.z;
            }
            SensorKind::Magnetometer(s) => {
                let (_, site_orientation) = site_world_pose(s, inputs.bodies, &tree_poses);
                let value = site_orientation.inverse_rotate(inputs.magnetic_field);
                out[0] = value.x;
                out[1] = value.y;
                out[2] = value.z;
            }
            SensorKind::Rangefinder(s) => {
                let (origin, orientation) = site_world_pose(s, inputs.bodies, &tree_poses);
                let direction = orientation.rotate(Vec3::Z);
                out[0] = rangefinder_reading(
                    origin,
                    direction,
                    inputs.bodies,
                    inputs.geoms,
                    inputs.meshes,
                    &tree_poses,
                );
            }
            SensorKind::SubtreeCom { tree, link } => {
                let (position, _) = subtree_com(*tree, *link, inputs.trees, &tree_poses);
                out[0] = position.x;
                out[1] = position.y;
                out[2] = position.z;
            }
            SensorKind::FrameLinVel(s) => {
                let (linear, _) = site_velocity(s, inputs.bodies, &tree_poses, &tree_va);
                out[0] = linear.x;
                out[1] = linear.y;
                out[2] = linear.z;
            }
            SensorKind::FrameAngVel(s) => {
                let (_, angular) = site_velocity(s, inputs.bodies, &tree_poses, &tree_va);
                out[0] = angular.x;
                out[1] = angular.y;
                out[2] = angular.z;
            }
        }
    }
}

fn site_velocity(
    site: &SiteFrame,
    bodies: &[Body],
    tree_poses: &[Vec<(Vec3, Quat)>],
    tree_va: &[Vec<(SpatialMotion, SpatialMotion)>],
) -> (Vec3, Vec3) {
    match site.attach {
        SensorAttach::Body(index) => {
            let body = &bodies[index];
            let angular = body.orientation.rotate(body.angular_velocity_body);
            let offset_world = body.orientation.rotate(site.local_offset);
            (body.linear_velocity + angular.cross(offset_world), angular)
        }
        SensorAttach::Link(tree, link) => {
            let (v, _) = tree_va[tree][link];
            let (_, orientation) = tree_poses[tree][link];
            let angular = orientation.rotate(v.angular);
            let linear = orientation.rotate(v.linear + v.angular.cross(site.local_offset));
            (linear, angular)
        }
    }
}

fn subtree_com(
    tree: usize,
    root: usize,
    trees: &[Tree],
    tree_poses: &[Vec<(Vec3, Quat)>],
) -> (Vec3, f32) {
    let model = &trees[tree];
    let mut weighted = Vec3::ZERO;
    let mut mass = 0.0;
    for (link_idx, (position, _)) in tree_poses[tree].iter().enumerate().take(model.links.len()) {
        let mut current = Some(link_idx);
        let mut included = false;
        while let Some(index) = current {
            if index == root {
                included = true;
                break;
            }
            current = model.links[index].parent;
        }
        if included {
            weighted += *position * model.links[link_idx].mass;
            mass += model.links[link_idx].mass;
        }
    }
    if mass == 0.0 {
        (Vec3::ZERO, 0.0)
    } else {
        (weighted / mass, mass)
    }
}

fn rangefinder_reading(
    origin: Vec3,
    direction: Vec3,
    bodies: &[Body],
    geoms: &[Geom],
    meshes: &[crate::geom::ConvexMesh],
    tree_poses: &[Vec<(Vec3, Quat)>],
) -> f32 {
    let mut nearest = f32::MAX;
    for geom in geoms {
        let pose = match geom.attachment() {
            GeomAttach::Static => geom_world_pose(geom, Vec3::ZERO, Quat::IDENTITY),
            GeomAttach::Body(body) => {
                geom_world_pose(geom, bodies[body].position, bodies[body].orientation)
            }
            GeomAttach::Link(tree, link) => {
                let (position, orientation) = tree_poses[tree][link];
                geom_world_pose(geom, position, orientation)
            }
        };
        let local_origin = pose.orientation.inverse_rotate(origin - pose.position);
        let local_direction = pose.orientation.inverse_rotate(direction);
        if let Some(distance) = ray_shape_hit(geom.shape, local_origin, local_direction, meshes) {
            if distance >= 0.0 && distance < nearest {
                nearest = distance;
            }
        }
    }
    if nearest == f32::MAX { -1.0 } else { nearest }
}

fn ray_shape_hit(
    shape: GeomShape,
    origin: Vec3,
    direction: Vec3,
    meshes: &[crate::geom::ConvexMesh],
) -> Option<f32> {
    match shape {
        GeomShape::Plane => {
            if direction.z.abs() < 1.0e-8 {
                None
            } else {
                positive_hit(-origin.z / direction.z)
            }
        }
        GeomShape::Sphere { radius } => ray_sphere(origin, direction, Vec3::ZERO, radius),
        GeomShape::Box { half_extents } => ray_box(origin, direction, half_extents),
        GeomShape::Capsule {
            radius,
            half_height,
        } => {
            let mut hit = ray_cylinder(origin, direction, radius, half_height);
            for center in [
                Vec3::new(0.0, 0.0, -half_height),
                Vec3::new(0.0, 0.0, half_height),
            ] {
                hit = min_hit(hit, ray_sphere(origin, direction, center, radius));
            }
            hit
        }
        GeomShape::Cylinder {
            radius,
            half_height,
        } => ray_cylinder(origin, direction, radius, half_height),
        GeomShape::Ellipsoid { semi_axes } => ray_ellipsoid(origin, direction, semi_axes),
        GeomShape::Mesh { mesh_id } => ray_mesh(origin, direction, &meshes[mesh_id]),
    }
}

fn positive_hit(value: f32) -> Option<f32> {
    if value >= 0.0 { Some(value) } else { None }
}

fn min_hit(a: Option<f32>, b: Option<f32>) -> Option<f32> {
    match (a, b) {
        (Some(x), Some(y)) => Some(if x <= y { x } else { y }),
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(y),
        (None, None) => None,
    }
}

fn ray_sphere(origin: Vec3, direction: Vec3, center: Vec3, radius: f32) -> Option<f32> {
    let offset = origin - center;
    let a = direction.dot(direction);
    if a <= 0.0 {
        return None;
    }
    let half_b = offset.dot(direction);
    let c = offset.dot(offset) - radius * radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = discriminant.sqrt();
    let first = (-half_b - root) / a;
    if first >= 0.0 {
        Some(first)
    } else {
        positive_hit((-half_b + root) / a)
    }
}

fn ray_box(origin: Vec3, direction: Vec3, half: Vec3) -> Option<f32> {
    let mut near = 0.0;
    let mut far = f32::MAX;
    for (o, d, h) in [
        (origin.x, direction.x, half.x),
        (origin.y, direction.y, half.y),
        (origin.z, direction.z, half.z),
    ] {
        if d.abs() < 1.0e-8 {
            if o < -h || o > h {
                return None;
            }
        } else {
            let mut a = (-h - o) / d;
            let mut b = (h - o) / d;
            if a > b {
                core::mem::swap(&mut a, &mut b);
            }
            if a > near {
                near = a;
            }
            if b < far {
                far = b;
            }
            if near > far {
                return None;
            }
        }
    }
    if near >= 0.0 {
        Some(near)
    } else {
        positive_hit(far)
    }
}

fn ray_cylinder(origin: Vec3, direction: Vec3, radius: f32, half_height: f32) -> Option<f32> {
    let mut best = None;
    let a = direction.x * direction.x + direction.y * direction.y;
    if a > 1.0e-8 {
        let half_b = origin.x * direction.x + origin.y * direction.y;
        let c = origin.x * origin.x + origin.y * origin.y - radius * radius;
        let disc = half_b * half_b - a * c;
        if disc >= 0.0 {
            let root = disc.sqrt();
            for t in [(-half_b - root) / a, (-half_b + root) / a] {
                if t >= 0.0 {
                    let z = origin.z + direction.z * t;
                    if z >= -half_height && z <= half_height {
                        best = min_hit(best, Some(t));
                    }
                }
            }
        }
    }
    if direction.z.abs() > 1.0e-8 {
        for z in [-half_height, half_height] {
            let t = (z - origin.z) / direction.z;
            if t >= 0.0 {
                let x = origin.x + direction.x * t;
                let y = origin.y + direction.y * t;
                if x * x + y * y <= radius * radius {
                    best = min_hit(best, Some(t));
                }
            }
        }
    }
    best
}

fn ray_ellipsoid(origin: Vec3, direction: Vec3, axes: Vec3) -> Option<f32> {
    let ox = origin.x / axes.x;
    let oy = origin.y / axes.y;
    let oz = origin.z / axes.z;
    let dx = direction.x / axes.x;
    let dy = direction.y / axes.y;
    let dz = direction.z / axes.z;
    let a = dx * dx + dy * dy + dz * dz;
    let half_b = ox * dx + oy * dy + oz * dz;
    let c = ox * ox + oy * oy + oz * oz - 1.0;
    let disc = half_b * half_b - a * c;
    if disc < 0.0 || a <= 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let first = (-half_b - root) / a;
    if first >= 0.0 {
        Some(first)
    } else {
        positive_hit((-half_b + root) / a)
    }
}

fn ray_mesh(origin: Vec3, direction: Vec3, mesh: &crate::geom::ConvexMesh) -> Option<f32> {
    let mut nearest = None;
    for face in &mesh.faces {
        let a = mesh.vertices[face[0] as usize];
        let b = mesh.vertices[face[1] as usize];
        let c = mesh.vertices[face[2] as usize];
        let normal = (b - a).cross(c - a);
        let denom = normal.dot(direction);
        if denom.abs() < 1.0e-8 {
            continue;
        }
        let t = normal.dot(a - origin) / denom;
        if t < 0.0 {
            continue;
        }
        let point = origin + direction * t;
        let e0 = b - a;
        let e1 = c - b;
        let e2 = a - c;
        if normal.dot((point - a).cross(e0)) >= -1.0e-6
            && normal.dot((point - b).cross(e1)) >= -1.0e-6
            && normal.dot((point - c).cross(e2)) >= -1.0e-6
        {
            nearest = min_hit(nearest, Some(t));
        }
    }
    nearest
}

fn site_world_pose(
    s: &SiteFrame,
    bodies: &[Body],
    tree_poses: &[Vec<(Vec3, Quat)>],
) -> (Vec3, Quat) {
    let (parent_pos, parent_ori) = match s.attach {
        SensorAttach::Body(i) => (bodies[i].position, bodies[i].orientation),
        SensorAttach::Link(t, l) => tree_poses[t][l],
    };
    let pos = parent_pos + parent_ori.rotate(s.local_offset);
    let ori = parent_ori * s.local_orientation;
    (pos, ori)
}

/// Return `(omega_body_parent, v_com_body_parent)` for the site's parent
/// body (free body or link). For a free body the body-frame omega is
/// stored directly; for a link it comes from the ABA-propagated `w.v[i]`
/// pre-computed by [`compute_link_va`] and passed in via `tree_va`.
fn parent_body_velocities(
    attach: SensorAttach,
    bodies: &[Body],
    tree_va: &[Vec<(SpatialMotion, SpatialMotion)>],
) -> (Vec3, Vec3) {
    match attach {
        SensorAttach::Body(i) => {
            let b = &bodies[i];
            let v_body = b.orientation.inverse_rotate(b.linear_velocity);
            (b.angular_velocity_body, v_body)
        }
        SensorAttach::Link(t, l) => {
            let (v, _) = tree_va[t][l];
            (v.angular, v.linear)
        }
    }
}

/// Compute per-link spatial `(v, a)` in each link's body frame at COM, given
/// the tree's current `(q, qdot)` and the joint acceleration vector `qddot`
/// from [`aba`]. Mirrors the top-down pass 1 of RNE without the wrench
/// accumulation. `Xup` is derived from the tree's own `q`.
fn compute_link_va(tree: &Tree, qddot: &[f32]) -> Vec<(SpatialMotion, SpatialMotion)> {
    let n = tree.links.len();
    assert_eq!(qddot.len(), tree.nv());
    let mut out = vec![(SpatialMotion::ZERO, SpatialMotion::ZERO); n];
    for i in 0..n {
        let link = &tree.links[i];
        match link.joint {
            JointKind::Free => {
                let voff = tree.v_offset[i];
                let v = SpatialMotion::new(
                    Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]),
                    Vec3::new(
                        tree.qdot[voff + 3],
                        tree.qdot[voff + 4],
                        tree.qdot[voff + 5],
                    ),
                );
                let a = SpatialMotion::new(
                    Vec3::new(qddot[voff], qddot[voff + 1], qddot[voff + 2]),
                    Vec3::new(qddot[voff + 3], qddot[voff + 4], qddot[voff + 5]),
                );
                out[i] = (v, a);
            }
            JointKind::Fixed => {
                let xup = xup_for_link(link, 0.0);
                let (v_p, a_p) = link
                    .parent
                    .map(|p| out[p])
                    .unwrap_or((SpatialMotion::ZERO, SpatialMotion::ZERO));
                out[i] = (xup.motion(v_p), xup.motion(a_p));
            }
            JointKind::Hinge { axis, .. } => {
                let (v, a) = propagate_single_dof(
                    tree,
                    i,
                    link,
                    &out,
                    xup_for_link_hinge(link, axis, tree.q[tree.q_offset[i]]),
                    subspace_hinge(link, axis),
                    tree.qdot[tree.v_offset[i]],
                    qddot[tree.v_offset[i]],
                );
                out[i] = (v, a);
            }
            JointKind::Slide { axis, .. } => {
                let (v, a) = propagate_single_dof(
                    tree,
                    i,
                    link,
                    &out,
                    xup_for_link_slide(link, axis, tree.q[tree.q_offset[i]]),
                    subspace_slide(axis),
                    tree.qdot[tree.v_offset[i]],
                    qddot[tree.v_offset[i]],
                );
                out[i] = (v, a);
            }
            JointKind::Ball { .. } => {
                let off = tree.q_offset[i];
                let q_ball = Quat::new(
                    tree.q[off],
                    tree.q[off + 1],
                    tree.q[off + 2],
                    tree.q[off + 3],
                );
                let xup = xup_for_link_ball(link, q_ball);
                let parent = link.parent.expect("ball must have parent");
                let (v_p, a_p) = out[parent];
                let s3 = subspace_ball(link);
                let voff = tree.v_offset[i];
                let omega = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let alpha = Vec3::new(qddot[voff], qddot[voff + 1], qddot[voff + 2]);
                let s_qdot = s3[0] * omega.x + s3[1] * omega.y + s3[2] * omega.z;
                let s_qddot = s3[0] * alpha.x + s3[1] * alpha.y + s3[2] * alpha.z;
                let v = xup.motion(v_p) + s_qdot;
                let c = v.cross_motion(s_qdot);
                let a = xup.motion(a_p) + s_qddot + c;
                out[i] = (v, a);
            }
        }
    }
    out
}

fn subspace_hinge(link: &crate::tree::Link, axis: Vec3) -> SpatialMotion {
    let r_jc = link.joint_offset_in_child.0;
    SpatialMotion::new(axis, r_jc.cross(axis))
}
fn subspace_slide(axis: Vec3) -> SpatialMotion {
    SpatialMotion::new(Vec3::ZERO, axis)
}
fn subspace_ball(link: &crate::tree::Link) -> [SpatialMotion; 3] {
    let r_jc = link.joint_offset_in_child.0;
    [
        SpatialMotion::new(Vec3::X, r_jc.cross(Vec3::X)),
        SpatialMotion::new(Vec3::Y, r_jc.cross(Vec3::Y)),
        SpatialMotion::new(Vec3::Z, r_jc.cross(Vec3::Z)),
    ]
}

#[allow(clippy::too_many_arguments)]
fn propagate_single_dof(
    _tree: &Tree,
    _i: usize,
    link: &crate::tree::Link,
    va: &[(SpatialMotion, SpatialMotion)],
    xup: Xform,
    s: SpatialMotion,
    qdot_i: f32,
    qddot_i: f32,
) -> (SpatialMotion, SpatialMotion) {
    let parent = link.parent.expect("non-root must have parent");
    let (v_p, a_p) = va[parent];
    let s_qdot = s * qdot_i;
    let s_qddot = s * qddot_i;
    let v = xup.motion(v_p) + s_qdot;
    let c = v.cross_motion(s_qdot);
    let a = xup.motion(a_p) + s_qddot + c;
    (v, a)
}

/// Compute per-link wrench `f[i]` accumulated by RNE — the interaction
/// wrench the parent joint transmits to child link `i`, expressed in the
/// child body frame at COM. This is exactly the intermediate `w.f[i]` from
/// [`crate::dynamics::inverse_dynamics`] after the leaves-to-root pass.
///
/// We re-run inverse_dynamics but capture the per-link f instead of the
/// scalar τ. Keeping the primary RNE entry point unchanged (it returns
/// only `τ`) means the sensor path pays one extra RNE per step when at
/// least one force/torque sensor is registered.
fn compute_link_wrenches(
    tree: &Tree,
    poses: &[(Vec3, Quat)],
    qddot: &[f32],
    gravity: Vec3,
    external_wrenches: &ExternalWrenches,
) -> Vec<crate::spatial::SpatialForce> {
    let _ = poses; // reserved for future extensions
    let n = tree.links.len();
    // Independent implementation of RNE's per-link f — computes v, a, and
    // f from scratch so the sensor doesn't depend on private state inside
    // `dynamics::inverse_dynamics`. Matches that function's pass structure
    // byte-for-byte; see docs/dynamics.md.
    let va = compute_link_va(tree, qddot);
    let poses_local = forward_kinematics(tree);
    let mut f: Vec<crate::spatial::SpatialForce> = vec![crate::spatial::SpatialForce::ZERO; n];
    // Per-link initial wrench: f[i] = I a + v ×* (I v) − f_ext (body frame).
    for i in 0..n {
        let link = &tree.links[i];
        let (_pos, ori) = poses_local[i];
        let si = link.spatial_inertia();
        let i_mat = crate::spatial::Mat6::from_spatial_inertia(si);
        let (v, a) = va[i];
        let i_a = i_mat.times_motion(a);
        let iv = i_mat.times_motion(v);
        let bias = v.cross_force(iv);
        let (force_ext, torque_ext) = external_wrenches[i];
        let force_world_total = force_ext + gravity * link.mass;
        let force_body = ori.inverse_rotate(force_world_total);
        let torque_body = ori.inverse_rotate(torque_ext);
        let f_ext_body = crate::spatial::SpatialForce::new(torque_body, force_body);
        f[i] = i_a + bias - f_ext_body;
    }
    // Leaves-to-root: pull each child's wrench into its parent.
    let mut xup_cache: Vec<Xform> = Vec::with_capacity(n);
    for (i, link) in tree.links.iter().enumerate() {
        let x = match link.joint {
            JointKind::Free => Xform::IDENTITY,
            JointKind::Fixed => xup_for_link(link, 0.0),
            JointKind::Hinge { axis, .. } => {
                xup_for_link_hinge(link, axis, tree.q[tree.q_offset[i]])
            }
            JointKind::Slide { axis, .. } => {
                xup_for_link_slide(link, axis, tree.q[tree.q_offset[i]])
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
        xup_cache.push(x);
    }
    for i in (1..n).rev() {
        let parent = tree.links[i].parent.expect("non-root has parent");
        let pulled = xup_cache[i].transpose_force(f[i]);
        f[parent] = f[parent] + pulled;
    }
    f
}

fn accelerometer_reading(
    s: &SiteFrame,
    bodies: &[Body],
    _trees: &[Tree],
    tree_poses: &[Vec<(Vec3, Quat)>],
    tree_va: &[Vec<(SpatialMotion, SpatialMotion)>],
    body_accels: &[(Vec3, Vec3)],
    gravity: Vec3,
) -> Vec3 {
    // Extract world-frame classical kinematics of the site's parent body:
    // (parent_pos, parent_ori, omega_world, alpha_world, a_com_world).
    let (parent_pos, parent_ori, omega_world, alpha_world, a_com_world) = match s.attach {
        SensorAttach::Body(i) => {
            let b = &bodies[i];
            let omega_world = b.orientation.rotate(b.angular_velocity_body);
            let (a_com_w, alpha_body) = body_accels[i];
            let alpha_world = b.orientation.rotate(alpha_body);
            (b.position, b.orientation, omega_world, alpha_world, a_com_w)
        }
        SensorAttach::Link(t, l) => {
            let (com, ori) = tree_poses[t][l];
            let (v_body, a_body) = tree_va[t][l];
            let omega_world = ori.rotate(v_body.angular);
            let alpha_world = ori.rotate(a_body.angular);
            // Featherstone's spatial `a` is the body-frame derivative of
            // the spatial velocity, NOT the classical inertial-frame COM
            // acceleration. The two differ by the frame-carrying
            // `ω × v_body` term:
            //   a_com_body_classical = a_body.linear + ω_body × v_body.linear
            // Rotating that into world gives the world-frame COM
            // acceleration the site formula needs. Missing this term made
            // link-attached accelerometers overreport by exactly that
            // correction (reviewer probe: gravity-only spinning +
            // translating free-root link read (-0.96, -0.51, 0.39) instead
            // of the true (0, 0, 0)).
            let a_com_body_classical = a_body.linear + v_body.angular.cross(v_body.linear);
            let a_com_world = ori.rotate(a_com_body_classical);
            (com, ori, omega_world, alpha_world, a_com_world)
        }
    };
    // `parent_pos` intentionally unused — site anchor offset already folded
    // via `r_world` below.
    let _ = parent_pos;
    // Site anchor position world-frame offset from parent COM:
    let r_world = parent_ori.rotate(s.local_offset);
    // Classical point acceleration at the site anchor:
    let a_site_world =
        a_com_world + alpha_world.cross(r_world) + omega_world.cross(omega_world.cross(r_world));
    // Proper (specific-force) acceleration in world:
    let a_proper = a_site_world - gravity;
    // Rotate into site frame. Site orientation in world = parent_ori · local_orientation.
    let site_ori_world = parent_ori * s.local_orientation;
    site_ori_world.inverse_rotate(a_proper)
}

/// World-frame `(v_com_body, ω_body)` for a link at its current `(q, qdot)`,
/// treating the tree as if `qddot = 0` (only `v` matters — `a` is discarded).
/// Consumed by [`crate::world::penalty_normal_force`] so link-attached
/// contact touch readings pick up the same point velocity the world's own
/// wrench assembly uses.
pub fn link_body_frame_vw_at_rest(tree: &Tree, target: usize) -> (Vec3, Vec3) {
    let qddot = vec![0.0f32; tree.nv()];
    let va = compute_link_va(tree, &qddot);
    let (v, _) = va[target];
    (v.linear, v.angular)
}

/// Sum per-contact normal-force magnitudes over every contact that
/// involves `geom_idx`. Force values come from the caller
/// ([`SensorInputs::contact_normal_forces`]) so the reading matches the
/// world's solver mode: penalty formula under
/// [`crate::solver::SolverMode::Penalty`], PGS constraint impulse over
/// `dt` under [`crate::solver::SolverMode::Pgs`].
fn touch_reading(geom_idx: usize, contacts: &[Contact], normal_forces: &[f32]) -> f32 {
    let mut sum: f32 = 0.0;
    for (i, c) in contacts.iter().enumerate() {
        if c.geom_a == geom_idx || c.geom_b == geom_idx {
            let f = normal_forces.get(i).copied().unwrap_or(0.0);
            sum += f.abs();
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Mat3;

    #[test]
    fn dim_matches_written_slice_width() {
        let mut bank = SensorBank::new();
        bank.push(Sensor {
            name: "a".into(),
            kind: SensorKind::JointPos { tree: 0, link: 1 },
        });
        bank.push(Sensor {
            name: "b".into(),
            kind: SensorKind::FramePos(SiteFrame {
                attach: SensorAttach::Body(0),
                local_offset: Vec3::ZERO,
                local_orientation: Quat::IDENTITY,
            }),
        });
        bank.push(Sensor {
            name: "c".into(),
            kind: SensorKind::FrameQuat(SiteFrame {
                attach: SensorAttach::Body(0),
                local_offset: Vec3::ZERO,
                local_orientation: Quat::IDENTITY,
            }),
        });
        assert_eq!(bank.offsets, vec![0, 1, 4]);
        assert_eq!(bank.data.len(), 8);
        assert_eq!(bank.slice(0).unwrap().len(), 1);
        assert_eq!(bank.slice(1).unwrap().len(), 3);
        assert_eq!(bank.slice(2).unwrap().len(), 4);
    }

    #[test]
    fn free_body_site_pose_composes_offset_and_orientation() {
        // Body at world (1, 2, 3) rotated 90° about z. Site at local offset
        // (1, 0, 0) → world (1, 3, 3). Site's world orientation = body's.
        let mut b = Body::new(
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
            Vec3::new(1.0, 2.0, 3.0),
            Quat::from_axis_angle(Vec3::Z, crate::math::FRAC_PI_2),
        );
        b.linear_velocity = Vec3::ZERO;
        let bodies = vec![b];
        let trees: Vec<Tree> = Vec::new();
        let site = SiteFrame {
            attach: SensorAttach::Body(0),
            local_offset: Vec3::new(1.0, 0.0, 0.0),
            local_orientation: Quat::IDENTITY,
        };
        let (p, q) = site.world_pose(&bodies, &trees);
        assert!((p.x - 1.0).abs() < 1e-5, "x = {}", p.x);
        assert!((p.y - 3.0).abs() < 1e-5, "y = {}", p.y);
        assert!((p.z - 3.0).abs() < 1e-5, "z = {}", p.z);
        // Orientation: 90° about z remains 90° about z.
        assert!((q.z - bodies[0].orientation.z).abs() < 1e-5);
        assert!((q.w - bodies[0].orientation.w).abs() < 1e-5);
    }
}
