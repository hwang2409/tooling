//! World: free rigid bodies + geoms, contacts, and fixed-step integration.
//!
//! # Scope (tier 2)
//!
//! Free bodies under uniform gravity plus contact forces from the selected
//! penalty or soft-constraint model.
//! Bodies, geoms, and contact pairs are index-stable across the simulation.
//!
//! # Contact model
//!
//! Full derivation lives in `docs/contacts.md`. In brief:
//!
//! - **Normal**: `f_n = max(0, k pen + c (−v_n))` where `v_n = (v_a − v_b) · n`
//!   is the relative velocity of the contact point along the normal. `k, c`
//!   come from the pair's [`crate::geom::SolRef`] and the reduced mass. `f_n`
//!   is clamped non-negative — the contact cannot pull.
//! - **Friction**: pyramidal, two tangent axes `(t1, t2)` picked
//!   deterministically from `n`. Per-axis force `f_ti = clamp(-c_t v_ti,
//!   −μ|f_n|, +μ|f_n|)` with `c_t = c_normal`. This is a viscous-with-Coulomb-
//!   clamp formulation: at rest, viscous force is small; under slip, it
//!   saturates at the Coulomb cap. The static/kinetic distinction is
//!   *effective*: with the pair's default stiffness, tangential drift at rest
//!   under a subcritical tangent load is small enough for the incline anchor.
//!
//! Contact forces are applied as world-frame wrenches at the contact point
//! and are recomputed at every RK4 sub-stage — that is the correct RK4 form
//! for a forced ODE. The tier-1 empty-geom path is preserved bit-for-bit
//! because every added term is `+ 0` when no contacts exist.
//!
//! # Determinism
//!
//! - Pair enumeration is total: sorted `(min, max)` index pairs, no HashMap
//!   iteration in the hot path.
//! - Narrow-phase output is fixed-order per pair (see [`crate::contact`]).
//! - Only `+ − * / sqrt` and hand-written transcendentals from
//!   [`crate::math`] appear in the compute path.

use crate::body::Body;
use crate::contact::{Contact, is_pair_supported, narrow_phase, narrow_phase_solver};
use crate::equality::Equality;
use crate::geom::{
    ConvexMesh, Geom, GeomAttach, GeomPose, GeomShape, combine_solref, geom_world_pose,
    solref_to_kc,
};
use crate::joint::JointKind;
use crate::math::{Quat, Vec3};
use crate::sensor::{Sensor, SensorBank, SensorError, SensorInputs};
use crate::solver::{
    ConstraintRowDiagnostic, SolverConfig, SolverMode, TreeContactSolution, solve_free_bodies,
};
use crate::tree::{
    Tree, euler_step as tree_euler_step, forward_kinematics as tree_forward_kinematics,
    rk4_step as tree_rk4_step,
};

/// Fixed-step integration schemes supported by [`World::step`].
///
/// `Rk4` remains the default to preserve every pre-v3 trajectory. `Euler`
/// is MuJoCo's semi-implicit Euler path with implicit joint damping.
/// `ImplicitFast` adds the velocity derivative of actuator forces to the same
/// mass-matrix fold. It intentionally does not include Coriolis derivatives.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Integrator {
    /// MuJoCo-style semi-implicit Euler.
    Euler,
    /// MuJoCo's documented implicit-in-velocity approximation.
    ImplicitFast,
    /// The original four-stage integrator. This is the default.
    #[default]
    Rk4,
}

impl Integrator {
    /// Compatibility spelling for callers that use MuJoCo's name.
    #[allow(non_upper_case_globals)]
    pub const RK4: Self = Self::Rk4;
    /// Compatibility alias for MuJoCo's full implicit spelling. Newt uses
    /// the documented implicitfast scope for both names.
    #[allow(non_upper_case_globals)]
    pub const Implicit: Self = Self::ImplicitFast;
}

/// Simulation world.
#[derive(Clone, Debug)]
pub struct World {
    /// Fixed integration timestep. Default 5 ms (matches biped).
    pub dt: f32,
    /// Integration scheme. Defaults to [`Integrator::Rk4`] for trajectory
    /// compatibility with all pre-v3 scenes.
    pub integrator: Integrator,
    /// Uniform gravity vector applied to every body's COM.
    pub gravity: Vec3,
    /// Global magnetic field in world coordinates for magnetometer sensors.
    pub magnetic_field: Vec3,
    /// Free bodies. Index-stable. Tier-1 style (no joints).
    pub bodies: Vec<Body>,
    /// Kinematic trees (tier 3). Index-stable. Free bodies and trees can
    /// coexist in the same world; contacts see both.
    pub trees: Vec<Tree>,
    /// Geoms. Index-stable. A geom's attachment (body, tree link, or static)
    /// determines which state drives its world pose.
    pub geoms: Vec<Geom>,
    /// Convex-mesh assets, indexed by [`GeomShape::Mesh::mesh_id`]. Empty
    /// when no mesh geoms are in play.
    pub meshes: Vec<ConvexMesh>,
    /// Optional explicit pair list `(geom_a, geom_b)` with `a < b`. When
    /// `None`, contact detection enumerates every unordered geom pair whose
    /// two geoms don't share a body/link and aren't both static; the
    /// resulting order is `(min, max)` lexicographic.
    pub pair_list: Option<Vec<(usize, usize)>>,
    /// Constraint solver configuration. Default is
    /// [`SolverConfig::DEFAULT`] — `SolverMode::Penalty`, the legacy force
    /// path. Set to `SolverMode::Pgs` or `SolverMode::Newton` to switch on a
    /// MuJoCo soft-constraint solver.
    pub solver: SolverConfig,
    /// Equality constraints (v1 tier 5). Only active when
    /// `solver.mode == Pgs`. Free-body equalities (connect / weld /
    /// distance) contribute rows to the free-body PGS solve; tree
    /// joint-couplings contribute one row each to their tree's PGS solve.
    /// Empty by default so every pre-v1-tier-5 golden and every scene
    /// that does not declare an equality is bit-for-bit unchanged.
    pub equalities: Vec<Equality>,
    /// Sensor bank (v1 tier 6). Sensors are declared here and evaluated at
    /// the end of every [`Self::step`] into `sensors.data`. Empty by
    /// default — a scene with no sensors skips the whole sensor pipeline,
    /// so every pre-v1-tier-6 golden and every scene without sensors is
    /// bit-for-bit unchanged.
    pub sensors: SensorBank,
    /// Named generalized-state snapshots.
    pub keyframes: Vec<Keyframe>,
    /// Cached pair-support fingerprint from the last successful validation.
    /// Encoded as `(geoms.len() << 32) | pair_list_encoded` where
    /// `pair_list_encoded` is `(pair_list.len() as u32) + 1` when
    /// `pair_list` is `Some` else `0`. `Cell<u64>::default() == 0` marks
    /// "never validated" (the empty world has zero geoms and no explicit
    /// pair list, whose encoding is `(0 << 32) | 0 = 0` — same as the
    /// default; harmless because that scene has no pairs to fail on).
    /// This is a cheap change detector so [`Self::step`] is O(1) after
    /// the first check; it is NOT authoritative — mutating a geom's
    /// `shape` in place bypasses detection. Callers that do so should
    /// call [`Self::invalidate_pair_check`].
    #[doc(hidden)]
    checked_pairs: std::cell::Cell<u64>,
    #[doc(hidden)]
    solver_phase_capture: bool,
    #[doc(hidden)]
    last_solver_phase: Option<SolverPhaseDiagnostics>,
}

/// State and contacts consumed by the most recent solver phase.
///
/// This is a diagnostic surface. It records the state before integration, not
/// the post-step geometry returned by [`World::detect_contacts`]. Enable it
/// with [`World::set_solver_phase_capture`].
#[derive(Clone, Debug)]
pub struct SolverPhaseDiagnostics {
    /// Flattened native tree generalized positions at the solver phase.
    pub qpos: Vec<f32>,
    /// Flattened native tree generalized velocities at the solver phase.
    pub qvel: Vec<f32>,
    /// Tree contacts consumed by the solver in deterministic order.
    pub contacts: Vec<Contact>,
    /// Solver row to original contact mapping.
    pub row_to_contact: Vec<usize>,
    /// Per-row reference diagnostics from the solver phase.
    pub row_diagnostics: Vec<ConstraintRowDiagnostic>,
    /// Tree generalized contact forces from the solver phase.
    pub tree_qfrc: Vec<Vec<f32>>,
}

/// A named generalized-state snapshot. Vectors flatten trees in world tree
/// order. `act` and `ctrl` flatten actuators in the same order.
#[derive(Clone, Debug, PartialEq)]
pub struct Keyframe {
    pub name: String,
    pub q: Vec<f32>,
    pub qdot: Vec<f32>,
    pub act: Vec<f32>,
    pub ctrl: Vec<f32>,
}

/// Keyframe lookup or dimension validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyframeError(pub String);

impl std::fmt::Display for KeyframeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for KeyframeError {}

// Manual PartialEq: the pair-check cache is not part of logical world state.
// Two worlds with identical bodies/trees/geoms/meshes/pair_list are equal
// regardless of whether either has run the pair check.
impl PartialEq for World {
    fn eq(&self, other: &Self) -> bool {
        self.dt == other.dt
            && self.integrator == other.integrator
            && self.gravity == other.gravity
            && self.magnetic_field == other.magnetic_field
            && self.bodies == other.bodies
            && self.trees == other.trees
            && self.geoms == other.geoms
            && self.meshes == other.meshes
            && self.pair_list == other.pair_list
            && self.solver == other.solver
            && self.equalities == other.equalities
            && self.sensors == other.sensors
            && self.keyframes == other.keyframes
    }
}

/// One entry returned by [`World::validate_supported_pairs`]: a geom index
/// pair whose shape combination is not implemented by
/// [`crate::contact::narrow_phase`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedPair {
    pub geom_a: usize,
    pub geom_b: usize,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    pub fn new() -> Self {
        Self {
            dt: 0.005,
            integrator: Integrator::default(),
            gravity: Vec3::new(0.0, 0.0, -9.81),
            magnetic_field: Vec3::new(0.0, -0.5, 0.0),
            bodies: Vec::new(),
            trees: Vec::new(),
            geoms: Vec::new(),
            meshes: Vec::new(),
            pair_list: None,
            solver: SolverConfig::DEFAULT,
            equalities: Vec::new(),
            sensors: SensorBank::new(),
            keyframes: Vec::new(),
            checked_pairs: std::cell::Cell::new(0),
            solver_phase_capture: false,
            last_solver_phase: None,
        }
    }

    /// Enable or disable capture of the state and contacts used by
    /// [`World::step`]. Disabled by default to avoid diagnostic allocations.
    pub fn set_solver_phase_capture(&mut self, enabled: bool) {
        self.solver_phase_capture = enabled;
        if !enabled {
            self.last_solver_phase = None;
        }
    }

    /// Return the most recent pre-integration solver-phase capture.
    pub fn solver_phase_diagnostics(&self) -> Option<&SolverPhaseDiagnostics> {
        self.last_solver_phase.as_ref()
    }

    /// Capture the current state through the same solver assembly used at the
    /// start of [`World::step`], without integrating. This supports an initial
    /// step-zero diagnostic record.
    pub fn capture_solver_phase(&mut self) {
        self.solver
            .validate()
            .unwrap_or_else(|message| panic!("{message}"));
        self.assert_pairs_supported();
        let pairs = match &self.pair_list {
            Some(p) => p.clone(),
            None => self.auto_pairs(),
        };
        let state = self.solver_phase_state();
        let solution = self.solver_phase_solution(&pairs);
        self.record_solver_phase(state, solution.as_ref(), &pairs);
    }

    /// Add a sensor to the world's sensor bank. Validates the sensor's
    /// references against the current bodies/trees/geoms and returns the
    /// sensor's stable index. Fails with a [`SensorError`] if the
    /// reference is out of range or the joint kind doesn't match the sensor
    /// (see [`Sensor::validate`]).
    ///
    /// The returned index also indexes `self.sensors.offsets` and picks out
    /// this sensor's slice via [`SensorBank::slice`].
    pub fn add_sensor(&mut self, sensor: Sensor) -> Result<usize, SensorError> {
        sensor.validate(&self.bodies, &self.trees, &self.geoms)?;
        let idx = self.sensors.sensors.len();
        self.sensors.push(sensor);
        Ok(idx)
    }

    /// Add a named keyframe after checking every vector against the current
    /// model dimensions.
    pub fn add_keyframe(
        &mut self,
        name: impl Into<String>,
        q: Vec<f32>,
        qdot: Vec<f32>,
        act: Vec<f32>,
        ctrl: Vec<f32>,
    ) -> Result<(), KeyframeError> {
        let name = name.into();
        if self.keyframes.iter().any(|key| key.name == name) {
            return Err(KeyframeError(format!("duplicate keyframe name {name:?}")));
        }
        let expected = self.keyframe_dimensions();
        for (label, actual, wanted) in [
            ("q", q.len(), expected.0),
            ("qdot", qdot.len(), expected.1),
            ("act", act.len(), expected.2),
            ("ctrl", ctrl.len(), expected.2),
        ] {
            if actual != wanted {
                return Err(KeyframeError(format!(
                    "keyframe {name:?} {label} dimension mismatch: expected {wanted}, got {actual}"
                )));
            }
        }
        self.keyframes.push(Keyframe {
            name,
            q,
            qdot,
            act,
            ctrl,
        });
        Ok(())
    }

    /// Reset generalized positions, velocities, activations, and controls to
    /// a named keyframe. The model and integration settings stay unchanged.
    pub fn reset_to_keyframe(&mut self, name: &str) -> Result<(), KeyframeError> {
        let key = self
            .keyframes
            .iter()
            .find(|key| key.name == name)
            .cloned()
            .ok_or_else(|| KeyframeError(format!("unknown keyframe {name:?}")))?;
        let mut q = 0;
        let mut qdot = 0;
        let mut actuator = 0;
        for tree in &mut self.trees {
            let nq = tree.nq();
            let nv = tree.nv();
            tree.q.copy_from_slice(&key.q[q..q + nq]);
            tree.qdot.copy_from_slice(&key.qdot[qdot..qdot + nv]);
            q += nq;
            qdot += nv;
            for item in &mut tree.actuators {
                item.act = key.act[actuator];
                item.ctrl = key.ctrl[actuator];
                actuator += 1;
            }
        }
        Ok(())
    }

    /// Apply MuJoCo's qpos layout to this world's state.
    ///
    /// This is shared by the differential harness and the MJCF keyframe
    /// loader. MuJoCo stores quaternions as `(w, x, y, z)`; free-root qvel
    /// stores world-frame linear velocity before body-frame angular velocity.
    pub fn apply_mujoco_qpos(&mut self, qpos: &[f32]) {
        let mut cursor = 0usize;
        for body in &mut self.bodies {
            body.position = Vec3::new(qpos[cursor], qpos[cursor + 1], qpos[cursor + 2]);
            body.orientation = Quat::new(
                qpos[cursor + 4],
                qpos[cursor + 5],
                qpos[cursor + 6],
                qpos[cursor + 3],
            )
            .renormalize();
            cursor += 7;
        }
        for tree in &mut self.trees {
            for i in 0..tree.links.len() {
                let q_off = tree.q_offset[i];
                match tree.links[i].joint {
                    JointKind::Free => {
                        tree.q[q_off..q_off + 3].copy_from_slice(&qpos[cursor..cursor + 3]);
                        let q = Quat::new(
                            qpos[cursor + 4],
                            qpos[cursor + 5],
                            qpos[cursor + 6],
                            qpos[cursor + 3],
                        )
                        .renormalize();
                        tree.q[q_off + 3..q_off + 7].copy_from_slice(&[q.x, q.y, q.z, q.w]);
                        cursor += 7;
                    }
                    JointKind::Ball { .. } => {
                        let q = Quat::new(
                            qpos[cursor + 1],
                            qpos[cursor + 2],
                            qpos[cursor + 3],
                            qpos[cursor],
                        )
                        .renormalize();
                        tree.q[q_off..q_off + 4].copy_from_slice(&[q.x, q.y, q.z, q.w]);
                        cursor += 4;
                    }
                    JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                        tree.q[q_off] = qpos[cursor];
                        cursor += 1;
                    }
                    JointKind::Fixed => {}
                }
            }
        }
        assert_eq!(cursor, qpos.len(), "MuJoCo qpos has trailing values");
    }

    /// Apply MuJoCo's qvel layout to this world's state.
    pub fn apply_mujoco_qvel(&mut self, qvel: &[f32]) {
        let mut cursor = 0usize;
        for body in &mut self.bodies {
            body.linear_velocity = Vec3::new(qvel[cursor], qvel[cursor + 1], qvel[cursor + 2]);
            body.angular_velocity_body =
                Vec3::new(qvel[cursor + 3], qvel[cursor + 4], qvel[cursor + 5]);
            cursor += 6;
        }
        for tree in &mut self.trees {
            for i in 0..tree.links.len() {
                let v_off = tree.v_offset[i];
                match tree.links[i].joint {
                    JointKind::Free => {
                        let q_off = tree.q_offset[i];
                        let orientation = Quat::new(
                            tree.q[q_off + 3],
                            tree.q[q_off + 4],
                            tree.q[q_off + 5],
                            tree.q[q_off + 6],
                        );
                        let v_body = orientation.inverse_rotate(Vec3::new(
                            qvel[cursor],
                            qvel[cursor + 1],
                            qvel[cursor + 2],
                        ));
                        tree.qdot[v_off..v_off + 6].copy_from_slice(&[
                            qvel[cursor + 3],
                            qvel[cursor + 4],
                            qvel[cursor + 5],
                            v_body.x,
                            v_body.y,
                            v_body.z,
                        ]);
                        cursor += 6;
                    }
                    JointKind::Ball { .. } => {
                        tree.qdot[v_off..v_off + 3].copy_from_slice(&qvel[cursor..cursor + 3]);
                        cursor += 3;
                    }
                    JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                        tree.qdot[v_off] = qvel[cursor];
                        cursor += 1;
                    }
                    JointKind::Fixed => {}
                }
            }
        }
        assert_eq!(cursor, qvel.len(), "MuJoCo qvel has trailing values");
    }

    /// Convert a tree-only MuJoCo keyframe into newt's dense tree state.
    /// Free bodies are not part of the current keyframe vector contract.
    pub fn mujoco_tree_keyframe_state(
        &self,
        qpos: &[f32],
        qvel: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), KeyframeError> {
        if !self.bodies.is_empty() {
            return Err(KeyframeError(
                "MJCF keyframes with free bodies are deferred; use tree state only".into(),
            ));
        }
        let mut state = self.clone();
        state.apply_mujoco_qpos(qpos);
        state.apply_mujoco_qvel(qvel);
        let q = state
            .trees
            .iter()
            .flat_map(|tree| tree.q.iter().copied())
            .collect();
        let qdot = state
            .trees
            .iter()
            .flat_map(|tree| tree.qdot.iter().copied())
            .collect();
        Ok((q, qdot))
    }

    fn keyframe_dimensions(&self) -> (usize, usize, usize) {
        (
            self.trees.iter().map(Tree::nq).sum(),
            self.trees.iter().map(Tree::nv).sum(),
            self.trees.iter().map(|tree| tree.actuators.len()).sum(),
        )
    }

    /// Read-only slice of the latest sensor reading for sensor `idx`, or
    /// `None` when `idx` is out of range.
    pub fn sensor(&self, idx: usize) -> Option<&[f32]> {
        self.sensors.slice(idx)
    }

    /// Invalidate the pair-support cache, forcing the next [`Self::step`]
    /// to re-run [`Self::validate_supported_pairs`]. Call this after any
    /// in-place mutation of a geom's `shape` field (which the length-based
    /// change detector cannot see).
    pub fn invalidate_pair_check(&self) {
        self.checked_pairs.set(0);
    }

    /// Encode the current geom/pair fingerprint. Zero encoding is reserved
    /// for "never validated"; the empty scene has no pairs so a spurious
    /// match against zero is harmless.
    fn pair_fingerprint(&self) -> u64 {
        let g = self.geoms.len() as u64;
        let p = match &self.pair_list {
            Some(v) => (v.len() as u64) + 1,
            None => 0,
        };
        (g << 32) | p
    }

    /// Panic if any ACTIVE contact pair (auto-generated or explicit) targets
    /// a shape combination not implemented by
    /// [`crate::contact::narrow_phase`]. The message names the offending
    /// geom indices and shape kinds so the caller can find the
    /// misconfiguration. Cached: after the first successful check, this is
    /// O(1) until the geom count or pair-list length changes (see
    /// [`Self::invalidate_pair_check`] for the shape-mutation case).
    ///
    /// This is the "engine-level, loudest form" enforcement of the
    /// documented no-silent-no-op rule (see `docs/contacts.md`) and the
    /// direct response to the tier-2 `stack.json` incident.
    fn assert_pairs_supported(&self) {
        let sig = self.pair_fingerprint();
        if self.checked_pairs.get() == sig {
            return;
        }
        let unsupported = self.validate_supported_pairs();
        if let Some(bad) = unsupported.first() {
            let ga = &self.geoms[bad.geom_a];
            let gb = &self.geoms[bad.geom_b];
            panic!(
                "contact pair {:?}(geom {}) x {:?}(geom {}) is not supported by \
                 newt's narrow phase — see docs/contacts.md support matrix. Restrict \
                 `world.pair_list` to a supported subset or defer this configuration.",
                shape_name(ga.shape),
                bad.geom_a,
                shape_name(gb.shape),
                bad.geom_b,
            );
        }
        self.checked_pairs.set(sig);
    }

    /// Register a convex mesh asset and return its stable id. Use the id in
    /// [`GeomShape::Mesh`]. Panics if the mesh fails
    /// [`ConvexMesh::validate`] — the loader validates model-format meshes;
    /// programmatic scenes get the same safety net.
    pub fn add_mesh(&mut self, mesh: ConvexMesh) -> usize {
        mesh.validate()
            .expect("convex mesh failed structural validation");
        let idx = self.meshes.len();
        self.meshes.push(mesh);
        idx
    }

    /// Return the list of geom-index pairs (drawn from `pair_list` if
    /// present, else the auto-enumerated pairs) whose shape combination is
    /// NOT implemented by [`crate::contact::narrow_phase`]. Callers should
    /// treat a non-empty result as a configuration bug: the pair will
    /// silently produce zero contacts at runtime and the two geoms will
    /// pass through each other.
    ///
    /// This is the "reject or warn at pair-construction time" guarantee
    /// documented in the v1-tier-2 spec — it prevents the tier-2 stack.json
    /// incident (an unsupported box-sphere pair silently no-op'd, letting
    /// bodies fall through the ground) from recurring for the new
    /// cylinder/ellipsoid/mesh combinations.
    pub fn validate_supported_pairs(&self) -> Vec<UnsupportedPair> {
        let pairs = match &self.pair_list {
            Some(p) => p.clone(),
            None => self.auto_pairs(),
        };
        let mut out = Vec::new();
        for (a, b) in pairs {
            let sa = self.geoms[a].shape;
            let sb = self.geoms[b].shape;
            if !is_pair_supported(sa, sb) {
                out.push(UnsupportedPair {
                    geom_a: a,
                    geom_b: b,
                });
            }
        }
        out
    }

    /// Adds a free body and returns its stable index.
    pub fn add_body(&mut self, body: Body) -> usize {
        let idx = self.bodies.len();
        self.bodies.push(body);
        idx
    }

    /// Adds a tree and returns its stable index.
    pub fn add_tree(&mut self, tree: Tree) -> usize {
        let idx = self.trees.len();
        self.trees.push(tree);
        idx
    }

    /// Adds a geom and returns its stable index.
    pub fn add_geom(&mut self, geom: Geom) -> usize {
        let idx = self.geoms.len();
        self.geoms.push(geom);
        idx
    }

    /// Enumerate all valid contact pairs in canonical `(min, max)` order.
    /// Used when `pair_list` is `None`. Pairs are dropped when both geoms
    /// share the same attachment (same body or same tree link, including
    /// static-vs-static).
    fn auto_pairs(&self) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let n = self.geoms.len();
        for a in 0..n {
            for b in (a + 1)..n {
                let att_a = self.geoms[a].attachment();
                let att_b = self.geoms[b].attachment();
                if att_a == att_b {
                    // Same body, same tree link, or two statics — no
                    // meaningful pair.
                    continue;
                }
                out.push((a, b));
            }
        }
        out
    }

    /// Read-only accessor: current world-frame COM position + orientation of
    /// a tree's link. Convenience for demos and tests that inspect state
    /// without doing a full forward-kinematics walk themselves.
    pub fn tree_link_pose(&self, tree_idx: usize, link_idx: usize) -> (Vec3, Quat) {
        tree_forward_kinematics(&self.trees[tree_idx])[link_idx]
    }

    /// Dense world-frame Jacobian for a tree link COM.
    pub fn tree_link_jacobian(
        &self,
        tree_idx: usize,
        link_idx: usize,
    ) -> crate::jacobian::Jacobian {
        self.trees[tree_idx].link_jacobian(link_idx)
    }

    /// Dense world-frame Jacobian for a point in a tree link's body frame.
    pub fn tree_point_jacobian(
        &self,
        tree_idx: usize,
        link_idx: usize,
        point_local: Vec3,
    ) -> crate::jacobian::Jacobian {
        self.trees[tree_idx].point_jacobian(link_idx, point_local)
    }

    /// Dense world-frame Jacobian for a free-body point in body coordinates.
    pub fn body_point_jacobian(
        &self,
        body_idx: usize,
        point_local: Vec3,
    ) -> crate::jacobian::Jacobian {
        let body = &self.bodies[body_idx];
        crate::jacobian::free_body_jacobian(body.position, body.orientation, point_local)
    }

    /// Dense world-frame Jacobian for a site frame attachment.
    pub fn site_jacobian(&self, site: &crate::sensor::SiteFrame) -> crate::jacobian::Jacobian {
        match site.attach {
            crate::sensor::SensorAttach::Body(body) => {
                self.body_point_jacobian(body, site.local_offset)
            }
            crate::sensor::SensorAttach::Link(tree, link) => {
                self.tree_point_jacobian(tree, link, site.local_offset)
            }
        }
    }

    /// Set a mocap root pose by tree index.
    pub fn set_mocap_pose(&mut self, tree_idx: usize, position: Vec3, orientation: Quat) {
        self.trees[tree_idx].set_mocap_pose(position, orientation);
    }

    /// Joint-space mass matrix `M(q)` for the tree at index `tree_idx`.
    /// Convenience wrapper around [`Tree::mass_matrix`] that also picks up
    /// the world's gravity semantics (mass matrix itself does not use
    /// gravity, but keeping the API co-located avoids the caller having to
    /// juggle borrows across the world and its trees).
    pub fn mass_matrix(&self, tree_idx: usize) -> Vec<f32> {
        self.trees[tree_idx].mass_matrix()
    }

    /// Bias forces `h(q, qdot)` for a tree under the world's gravity.
    pub fn bias_forces(&self, tree_idx: usize) -> Vec<f32> {
        self.trees[tree_idx].bias_forces(self.gravity)
    }

    /// Inverse dynamics for a tree with the world's gravity. Callers pass
    /// their own `external_wrenches` (matching [`Tree::inverse_dynamics`]);
    /// contact wrenches from the world's pipeline are not automatically
    /// re-derived here.
    pub fn inverse_dynamics(
        &self,
        tree_idx: usize,
        qddot: &[f32],
        external_wrenches: &crate::tree::ExternalWrenches,
    ) -> Vec<f32> {
        self.trees[tree_idx].inverse_dynamics(qddot, self.gravity, external_wrenches)
    }

    /// World-level inverse dynamics for an explicit tree state.
    pub fn inverse_dynamics_at(
        &self,
        tree_idx: usize,
        q: &[f32],
        qdot: &[f32],
        qddot: &[f32],
        external_wrenches: &crate::tree::ExternalWrenches,
    ) -> Vec<f32> {
        self.trees[tree_idx].inverse_dynamics_at(q, qdot, qddot, self.gravity, external_wrenches)
    }

    /// Advance the whole world by one fixed-dt step.
    ///
    /// Contact forces are recomputed at each RK4 sub-stage from the
    /// interpolated body states. This is the standard RK4 treatment for a
    /// forced ODE (Butcher, *Numerical Methods for ODEs*, §3.1) and is
    /// stable for the stiffness range v0 targets (`SolRef::DEFAULT` gives
    /// a contact period ≈ 125 ms; `dt = 5 ms` is 25 samples per period).
    /// Very stiff underdamped contacts still show some parasitic damping
    /// (the intermediate stages sample deeper penetrations than the true
    /// continuous solution reaches); the bouncing anchor picks parameters
    /// that keep the effective restitution comfortably above zero. With no
    /// geoms, this collapses to tier-1 gravity-only RK4 bit-for-bit, and
    /// the golden `tumbling_3_body.bin` still passes.
    pub fn step(&mut self) {
        self.solver
            .validate()
            .unwrap_or_else(|message| panic!("{message}"));
        // Loud engine-level enforcement: the first step after any pair-list
        // or geom-count change panics if any ACTIVE pair falls in the
        // deferred bucket. Prevents a stack.json-style silent no-op.
        self.assert_pairs_supported();
        let pairs = match &self.pair_list {
            Some(p) => p.clone(),
            None => self.auto_pairs(),
        };
        let solver_phase_state = self.solver_phase_capture.then(|| self.solver_phase_state());
        let tree_contact_solution = self.solver_phase_solution(&pairs);
        if let Some(state) = solver_phase_state {
            self.record_solver_phase(state, tree_contact_solution.as_ref(), &pairs);
        }
        match self.integrator {
            Integrator::Rk4 => {
                self.step_bodies(&pairs, tree_contact_solution.as_ref());
                self.step_trees(&pairs, tree_contact_solution.as_ref());
            }
            Integrator::Euler => {
                self.step_trees_euler(&pairs, false, tree_contact_solution.as_ref());
                self.step_bodies_euler(&pairs, tree_contact_solution.as_ref());
            }
            Integrator::ImplicitFast => {
                self.step_trees_euler(&pairs, true, tree_contact_solution.as_ref());
                self.step_bodies_euler(&pairs, tree_contact_solution.as_ref());
            }
        }
        // Sensor evaluation runs strictly on post-step state — no
        // perturbation. Skipped when no sensors are declared so every
        // pre-v1-tier-6 golden path is bit-for-bit untouched.
        if !self.sensors.sensors.is_empty() {
            self.evaluate_sensors(&pairs);
        }
    }

    /// Recompute every sensor reading against the current world state and
    /// store the results in `self.sensors.data`. Called automatically by
    /// [`Self::step`] when the sensor bank is non-empty; exposed publicly
    /// so tests / callers who bypass `step` (e.g. loading a scene and
    /// wanting a snapshot before the first integration) can force an
    /// evaluation.
    pub fn evaluate_sensors(&mut self, pairs: &[(usize, usize)]) {
        // Take the sensor bank out temporarily so `build_sensor_inputs`
        // can borrow the rest of `self` immutably without conflicting
        // with the &mut we need for the writeback. A scope guard restores
        // the bank on Drop so a panic inside `evaluate` doesn't leave the
        // world with an empty SensorBank.
        struct Restore<'w> {
            world: &'w mut World,
            bank: SensorBank,
        }
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                std::mem::swap(&mut self.world.sensors, &mut self.bank);
            }
        }
        let bank_taken = std::mem::take(&mut self.sensors);
        let mut guard = Restore {
            world: self,
            bank: bank_taken,
        };
        let inputs = guard.world.build_sensor_inputs(pairs);
        crate::sensor::evaluate(&mut guard.bank, &inputs);
        // Guard's Drop restores the bank into `self.sensors`.
    }

    /// Build the [`SensorInputs`] bundle for the post-step state. Uses the
    /// solver-mode-appropriate wrench source (penalty vs PGS) so an
    /// accelerometer or touch reading sees the SAME contact forces the
    /// integrator did.
    fn build_sensor_inputs<'a>(&'a self, pairs: &'a [(usize, usize)]) -> SensorInputs<'a> {
        // Detect the full contact set from the same pipeline `step` uses.
        // Split into free-body-only and per-tree lists so we can reuse the
        // solver/penalty machinery downstream unchanged.
        let mut free_pairs: Vec<(usize, usize)> = Vec::new();
        for &(a, b) in pairs {
            let att_a = self.geoms[a].attachment();
            let att_b = self.geoms[b].attachment();
            if !matches!(att_a, GeomAttach::Link(_, _)) && !matches!(att_b, GeomAttach::Link(_, _))
            {
                free_pairs.push((a, b));
            }
        }
        // Match the narrow-phase manifold to the solver mode so sensor
        // readings (touch, contact forces) see the same contact set the
        // wrench pathway used this step.
        let manifold = match self.solver.mode {
            SolverMode::Pgs | SolverMode::Newton => ContactManifold::Full,
            SolverMode::Penalty => ContactManifold::Legacy,
        };
        let free_body_contacts = collect_contacts(
            &self.bodies,
            &self.geoms,
            &self.meshes,
            &free_pairs,
            manifold,
        );

        // Body wrenches + per-contact normal forces (touch sensor input).
        let (mut body_wrenches, free_body_contact_forces): (Vec<(Vec3, Vec3)>, Vec<f32>) =
            match self.solver.mode {
                SolverMode::Penalty => {
                    let w = self.compute_wrenches(&self.bodies, pairs);
                    let f: Vec<f32> = free_body_contacts
                        .iter()
                        .map(|c| penalty_normal_force(c, &self.bodies, &self.trees, &self.geoms))
                        .collect();
                    (w, f)
                }
                SolverMode::Pgs => crate::solver::solve_free_bodies_diag(
                    &self.bodies,
                    &self.geoms,
                    &free_body_contacts,
                    &self.equalities,
                    self.gravity,
                    self.dt,
                    self.solver.cone,
                    self.solver.iterations,
                ),
                SolverMode::Newton => crate::solver::solve_free_bodies_newton_diag(
                    &self.bodies,
                    &self.geoms,
                    &free_body_contacts,
                    &self.equalities,
                    self.gravity,
                    self.dt,
                    self.solver.cone,
                    self.solver.iterations,
                ),
            };

        // Keep one original-indexed tree contact list. The solver solution
        // uses this exact order, so touch sensors cannot drift when a gap
        // contact is omitted from the compact row system.
        let tree_contacts: Vec<Contact> =
            collect_contacts_full(&self.bodies, &self.trees, &self.geoms, &self.meshes, pairs)
                .into_iter()
                .filter(|contact| {
                    matches!(
                        self.geoms[contact.geom_a].attachment(),
                        GeomAttach::Link(_, _)
                    ) || matches!(
                        self.geoms[contact.geom_b].attachment(),
                        GeomAttach::Link(_, _)
                    )
                })
                .collect();
        let tree_contact_solution = match self.solver.mode {
            SolverMode::Penalty => None,
            SolverMode::Pgs => Some(self.solve_tree_contact_sensor_solution(&tree_contacts, false)),
            SolverMode::Newton => {
                Some(self.solve_tree_contact_sensor_solution(&tree_contacts, true))
            }
        };
        let (tree_wrenches, tree_contact_forces) =
            if let Some(solution) = tree_contact_solution.as_ref() {
                (
                    solution.tree_wrenches.clone(),
                    solution.contact_normal_forces.clone(),
                )
            } else {
                let mut wrenches = Vec::with_capacity(self.trees.len());
                for (ti, tree) in self.trees.iter().enumerate() {
                    let tree_pairs: Vec<(usize, usize)> = pairs
                        .iter()
                        .copied()
                        .filter(|&(a, b)| {
                            matches!(self.geoms[a].attachment(), GeomAttach::Link(t, _) if t == ti)
                                || matches!(
                                    self.geoms[b].attachment(),
                                    GeomAttach::Link(t, _) if t == ti
                                )
                        })
                        .collect();
                    wrenches.push(tree_wrenches_from_contacts(
                        tree,
                        ti,
                        &self.bodies,
                        &self.geoms,
                        &self.meshes,
                        &tree_pairs,
                    ));
                }
                let forces = tree_contacts
                    .iter()
                    .map(|contact| {
                        penalty_normal_force(contact, &self.bodies, &self.trees, &self.geoms)
                    })
                    .collect();
                (wrenches, forces)
            };
        if let Some(solution) = tree_contact_solution.as_ref() {
            for (wrench, solved) in body_wrenches.iter_mut().zip(&solution.body_wrenches) {
                wrench.0 += solved.0;
                wrench.1 += solved.1;
            }
        }

        // Merge free-body + tree contacts (preserving order) so touch
        // sensors on either attachment kind read from one list.
        let mut contacts: Vec<Contact> = free_body_contacts;
        contacts.extend(tree_contacts);
        let mut contact_normal_forces = free_body_contact_forces;
        contact_normal_forces.extend(tree_contact_forces);

        SensorInputs {
            bodies: &self.bodies,
            trees: &self.trees,
            geoms: &self.geoms,
            meshes: &self.meshes,
            gravity: self.gravity,
            magnetic_field: self.magnetic_field,
            dt: self.dt,
            body_wrenches,
            tree_wrenches,
            contacts,
            contact_normal_forces,
        }
    }

    /// Advance only the free bodies. Preserves the tier-1/2 behavior
    /// bit-for-bit when no tree links are in play.
    fn step_bodies(
        &mut self,
        pairs: &[(usize, usize)],
        tree_contact_solution: Option<&TreeContactSolution>,
    ) {
        let s0 = self.bodies.clone();

        // Solver mode dispatch:
        // - `Penalty` recomputes contact wrenches at each RK4 sub-stage
        //   (the tier-2 path, unchanged).
        // - `Pgs` solves the constraint system ONCE at s0 and holds those
        //   per-body wrenches constant (zero-order hold) across all four
        //   sub-stages. See newt/docs/solver.md, "Once-per-step under RK4",
        //   for the rationale + tradeoffs.
        let solver_zoh: Option<Vec<(Vec3, Vec3)>> = match self.solver.mode {
            SolverMode::Penalty => None,
            SolverMode::Pgs | SolverMode::Newton => {
                Some(self.compute_solver_wrenches(&s0, pairs, tree_contact_solution))
            }
        };
        let sample_wrenches = |state: &[Body], pairs: &[(usize, usize)]| -> Vec<(Vec3, Vec3)> {
            match &solver_zoh {
                Some(w) => w.clone(),
                None => self.compute_wrenches(state, pairs),
            }
        };

        let ext1 = sample_wrenches(&s0, pairs);
        let k1 = evaluate_all(&s0, self.gravity, &ext1);

        let s1 = advance_all(&s0, &k1, self.dt * 0.5);
        let ext2 = sample_wrenches(&s1, pairs);
        let k2 = evaluate_all(&s1, self.gravity, &ext2);

        let s2 = advance_all(&s0, &k2, self.dt * 0.5);
        let ext3 = sample_wrenches(&s2, pairs);
        let k3 = evaluate_all(&s2, self.gravity, &ext3);

        let s3 = advance_all(&s0, &k3, self.dt);
        let ext4 = sample_wrenches(&s3, pairs);
        let k4 = evaluate_all(&s3, self.gravity, &ext4);

        for i in 0..self.bodies.len() {
            let state = s0[i];
            let sixth = 1.0 / 6.0;
            let dp =
                (k1[i].dposition + k2[i].dposition * 2.0 + k3[i].dposition * 2.0 + k4[i].dposition)
                    * sixth;
            let dv = (k1[i].dlinear_velocity
                + k2[i].dlinear_velocity * 2.0
                + k3[i].dlinear_velocity * 2.0
                + k4[i].dlinear_velocity)
                * sixth;
            let dq = (k1[i].dorientation
                + k2[i].dorientation * 2.0
                + k3[i].dorientation * 2.0
                + k4[i].dorientation)
                * sixth;
            let dw = (k1[i].dangular_velocity_body
                + k2[i].dangular_velocity_body * 2.0
                + k3[i].dangular_velocity_body * 2.0
                + k4[i].dangular_velocity_body)
                * sixth;

            self.bodies[i] = Body {
                position: state.position + dp * self.dt,
                linear_velocity: state.linear_velocity + dv * self.dt,
                orientation: (state.orientation + dq * self.dt).renormalize(),
                angular_velocity_body: state.angular_velocity_body + dw * self.dt,
                ..state
            };
        }
    }

    /// Advance free bodies with MuJoCo's semi-implicit Euler ordering.
    /// Contact and PGS forces are sampled once from the current state. The
    /// updated velocity drives both the position and quaternion updates.
    fn step_bodies_euler(
        &mut self,
        pairs: &[(usize, usize)],
        tree_contact_solution: Option<&TreeContactSolution>,
    ) {
        let ext = match self.solver.mode {
            SolverMode::Penalty => self.compute_wrenches(&self.bodies, pairs),
            SolverMode::Pgs | SolverMode::Newton => {
                self.compute_solver_wrenches(&self.bodies, pairs, tree_contact_solution)
            }
        };
        let accel = evaluate_all(&self.bodies, self.gravity, &ext);
        let dt = self.dt;
        for (body, d) in self.bodies.iter_mut().zip(accel) {
            let linear_velocity = body.linear_velocity + d.dlinear_velocity * dt;
            let angular_velocity_body = body.angular_velocity_body + d.dangular_velocity_body * dt;
            body.position += linear_velocity * dt;
            body.orientation = body
                .orientation
                .integrate_body_angular_velocity(angular_velocity_body, dt);
            body.linear_velocity = linear_velocity;
            body.angular_velocity_body = angular_velocity_body;
        }
    }

    /// Advance the kinematic trees under gravity + contact wrenches.
    ///
    /// Penalty mode keeps the legacy independent-tree contact callback.
    /// PGS and Newton use one start-of-step world contact solve, then add the
    /// returned joint forces here. Contacts touching only free bodies are
    /// handled by [`Self::step_bodies`]. Trees are integrated in stable index
    /// order after the shared solve.
    fn step_trees(
        &mut self,
        pairs: &[(usize, usize)],
        tree_contact_solution: Option<&TreeContactSolution>,
    ) {
        if self.trees.is_empty() {
            return;
        }
        let n_trees = self.trees.len();
        // Snapshot scalars before we start borrowing the vector fields.
        let dt = self.dt;
        let gravity = self.gravity;
        let solver_mode = self.solver.mode;
        let solver_iterations = self.solver.iterations;
        for ti in 0..n_trees {
            // Filter pairs to those touching this tree (immutable borrow
            // of self.geoms, released before the mem::take below).
            let mut tree_pairs: Vec<(usize, usize)> = Vec::new();
            for &(a, b) in pairs {
                let att_a = self.geoms[a].attachment();
                let att_b = self.geoms[b].attachment();
                let a_ours = matches!(att_a, GeomAttach::Link(t, _) if t == ti);
                let b_ours = matches!(att_b, GeomAttach::Link(t, _) if t == ti);
                if a_ours || b_ours {
                    tree_pairs.push((a, b));
                }
            }
            // Move the current tree out so the closure below can borrow
            // the rest of `self` immutably without conflicting with the
            // `&mut tree` that tree_rk4_step wants. std::mem::take
            // replaces the slot with `Tree::default()`; we overwrite
            // that with the stepped tree at the end. No clone of
            // bodies/geoms/meshes — the closure captures those as
            // borrows, whose lifetime ends before we mutate self.trees
            // again.
            let mut tree = std::mem::take(&mut self.trees[ti]);
            // Solver mode: compute per-DOF limit force ONCE at s0 and
            // hold it constant across the RK4 stages via qfrc_applied.
            // Preserves the pre-step qfrc_applied so user-set torques
            // remain in effect (the solver term is added on top and
            // subtracted back after the step). ALSO: disable the tier-3
            // penalty limit torque for this step so the PGS constraint
            // is the sole limit authority — mirrors how contacts already
            // switch on solver mode.
            let mut solver_qfrc_delta: Vec<f32> = Vec::new();
            let prior_disable = tree.disable_penalty_limits;
            if matches!(solver_mode, SolverMode::Pgs | SolverMode::Newton) {
                solver_qfrc_delta = match solver_mode {
                    SolverMode::Pgs => crate::solver::solve_tree_limits(
                        &tree,
                        ti,
                        &self.equalities,
                        dt,
                        solver_iterations,
                    ),
                    SolverMode::Newton => crate::solver::solve_tree_limits_newton(
                        &tree,
                        ti,
                        &self.equalities,
                        dt,
                        solver_iterations,
                    ),
                    SolverMode::Penalty => unreachable!(),
                };
                if let Some(solution) = tree_contact_solution {
                    for (slot, delta) in solution.tree_qfrc[ti].iter().enumerate() {
                        solver_qfrc_delta[slot] += delta;
                    }
                }
                for (slot, &delta) in solver_qfrc_delta.iter().enumerate() {
                    tree.qfrc_applied[slot] += delta;
                }
                tree.disable_penalty_limits = true;
            }
            {
                // Scope the immutable borrows so the closure lifetime
                // ends before we mutate self.trees[ti] on the next line.
                let bodies_ref = &self.bodies;
                let geoms_ref = &self.geoms;
                let meshes_ref = &self.meshes;
                tree_rk4_step(&mut tree, gravity, dt, |t| {
                    if matches!(solver_mode, SolverMode::Pgs | SolverMode::Newton) {
                        vec![(Vec3::ZERO, Vec3::ZERO); t.links.len()]
                    } else {
                        tree_wrenches_from_contacts(
                            t,
                            ti,
                            bodies_ref,
                            geoms_ref,
                            meshes_ref,
                            &tree_pairs,
                        )
                    }
                });
            }
            // Roll back the ZOH limit torque + penalty-limit gate so
            // neither accumulates across steps (the solver recomputes
            // both fresh at each step start).
            for (slot, &delta) in solver_qfrc_delta.iter().enumerate() {
                tree.qfrc_applied[slot] -= delta;
            }
            tree.disable_penalty_limits = prior_disable;
            self.trees[ti] = tree;
        }
    }

    /// Advance trees with one MuJoCo-style semi-implicit step. The contact
    /// callback is called once at the current state, and the tree ABA folds
    /// joint damping into `M + dt*B`. `implicit_fast` also folds the negative
    /// velocity derivative of joint-transmitted actuator forces.
    fn step_trees_euler(
        &mut self,
        pairs: &[(usize, usize)],
        implicit_fast: bool,
        tree_contact_solution: Option<&TreeContactSolution>,
    ) {
        if self.trees.is_empty() {
            return;
        }
        let dt = self.dt;
        let gravity = self.gravity;
        let solver_mode = self.solver.mode;
        let solver_iterations = self.solver.iterations;
        for ti in 0..self.trees.len() {
            let mut tree_pairs = Vec::new();
            for &(a, b) in pairs {
                let att_a = self.geoms[a].attachment();
                let att_b = self.geoms[b].attachment();
                if matches!(att_a, GeomAttach::Link(t, _) if t == ti)
                    || matches!(att_b, GeomAttach::Link(t, _) if t == ti)
                {
                    tree_pairs.push((a, b));
                }
            }

            let mut tree = std::mem::take(&mut self.trees[ti]);
            let prior_disable = tree.disable_penalty_limits;
            let mut solver_qfrc_delta = Vec::new();
            if matches!(solver_mode, SolverMode::Pgs | SolverMode::Newton) {
                solver_qfrc_delta = match solver_mode {
                    SolverMode::Pgs => crate::solver::solve_tree_limits(
                        &tree,
                        ti,
                        &self.equalities,
                        dt,
                        solver_iterations,
                    ),
                    SolverMode::Newton => crate::solver::solve_tree_limits_newton(
                        &tree,
                        ti,
                        &self.equalities,
                        dt,
                        solver_iterations,
                    ),
                    SolverMode::Penalty => unreachable!(),
                };
                if let Some(solution) = tree_contact_solution {
                    for (slot, delta) in solution.tree_qfrc[ti].iter().enumerate() {
                        solver_qfrc_delta[slot] += delta;
                    }
                }
                for (slot, &delta) in solver_qfrc_delta.iter().enumerate() {
                    tree.qfrc_applied[slot] += delta;
                }
                tree.disable_penalty_limits = true;
            }
            {
                let bodies_ref = &self.bodies;
                let geoms_ref = &self.geoms;
                let meshes_ref = &self.meshes;
                tree_euler_step(&mut tree, gravity, dt, implicit_fast, |state| {
                    if matches!(solver_mode, SolverMode::Pgs | SolverMode::Newton) {
                        vec![(Vec3::ZERO, Vec3::ZERO); state.links.len()]
                    } else {
                        tree_wrenches_from_contacts(
                            state,
                            ti,
                            bodies_ref,
                            geoms_ref,
                            meshes_ref,
                            &tree_pairs,
                        )
                    }
                });
            }
            for (slot, &delta) in solver_qfrc_delta.iter().enumerate() {
                tree.qfrc_applied[slot] -= delta;
            }
            tree.disable_penalty_limits = prior_disable;
            self.trees[ti] = tree;
        }
    }

    /// Public: detect all contacts against the current body/tree state.
    /// Useful for tests that need to inspect contact geometry.
    pub fn detect_contacts(&self) -> Vec<Contact> {
        let pairs = match &self.pair_list {
            Some(p) => p.clone(),
            None => self.auto_pairs(),
        };
        collect_contacts_full(&self.bodies, &self.trees, &self.geoms, &self.meshes, &pairs)
    }

    /// Compute per-body external wrench arrays for solver mode.
    ///
    /// Runs the PGS or Newton solve at `state` (typically s0 — start of RK4 step) and
    /// returns per-body `(force_world, torque_world_at_com)` to hold
    /// constant across all four RK4 sub-stages. Tree-involved contact rows
    /// are added by the shared world solve before this result is returned.
    fn compute_solver_wrenches(
        &self,
        state: &[Body],
        pairs: &[(usize, usize)],
        tree_contact_solution: Option<&TreeContactSolution>,
    ) -> Vec<(Vec3, Vec3)> {
        // Filter pairs to free-body-only ones (both sides Body or
        // Static). `solve_free_bodies` returns per-body zero wrenches
        // when there are no contacts AND no free-body equalities, so
        // the outer fast path is redundant.
        let mut free_pairs: Vec<(usize, usize)> = Vec::with_capacity(pairs.len());
        for &(a, b) in pairs {
            let att_a = self.geoms[a].attachment();
            let att_b = self.geoms[b].attachment();
            if matches!(att_a, GeomAttach::Link(_, _)) || matches!(att_b, GeomAttach::Link(_, _)) {
                continue;
            }
            free_pairs.push((a, b));
        }
        let contacts = collect_contacts(
            state,
            &self.geoms,
            &self.meshes,
            &free_pairs,
            ContactManifold::Full,
        );
        let mut wrenches = match self.solver.mode {
            SolverMode::Pgs => solve_free_bodies(
                state,
                &self.geoms,
                &contacts,
                &self.equalities,
                self.gravity,
                self.dt,
                self.solver.cone,
                self.solver.iterations,
            ),
            SolverMode::Newton => crate::solver::solve_free_bodies_newton(
                state,
                &self.geoms,
                &contacts,
                &self.equalities,
                self.gravity,
                self.dt,
                self.solver.cone,
                self.solver.iterations,
            ),
            SolverMode::Penalty => unreachable!("penalty does not call compute_solver_wrenches"),
        };
        // Mocap contacts are kinematic rows in the shared tree solve. Their
        // recovered force reaches this body pool through `body_wrenches`, so
        // the legacy penalty mocap callback must not run here.
        if let Some(solution) = tree_contact_solution {
            for (wrench, solved) in wrenches.iter_mut().zip(&solution.body_wrenches) {
                wrench.0 += solved.0;
                wrench.1 += solved.1;
            }
        }
        wrenches
    }

    /// Assemble the full solver contact set for rows that touch a tree.
    /// Contacts stay in the original narrow-phase order so sensor force
    /// readings can index the result without a compact-row offset.
    fn compute_tree_contact_solution(
        &self,
        pairs: &[(usize, usize)],
        use_newton: bool,
    ) -> Option<TreeContactSolution> {
        let contacts =
            collect_contacts_full(&self.bodies, &self.trees, &self.geoms, &self.meshes, pairs);
        let tree_contacts: Vec<Contact> = contacts
            .into_iter()
            .filter(|contact| {
                matches!(
                    self.geoms[contact.geom_a].attachment(),
                    GeomAttach::Link(_, _)
                ) || matches!(
                    self.geoms[contact.geom_b].attachment(),
                    GeomAttach::Link(_, _)
                )
            })
            .collect();
        Some(self.solve_tree_contact_sensor_solution(&tree_contacts, use_newton))
    }

    fn solver_phase_solution(&self, pairs: &[(usize, usize)]) -> Option<TreeContactSolution> {
        match self.solver.mode {
            SolverMode::Penalty => None,
            SolverMode::Pgs => self.compute_tree_contact_solution(pairs, false),
            SolverMode::Newton => self.compute_tree_contact_solution(pairs, true),
        }
    }

    fn solver_phase_state(&self) -> (Vec<f32>, Vec<f32>) {
        let qpos = self
            .trees
            .iter()
            .flat_map(|tree| tree.q.iter().copied())
            .collect();
        let qvel = self
            .trees
            .iter()
            .flat_map(|tree| tree.qdot.iter().copied())
            .collect();
        (qpos, qvel)
    }

    fn record_solver_phase(
        &mut self,
        state: (Vec<f32>, Vec<f32>),
        solution: Option<&TreeContactSolution>,
        pairs: &[(usize, usize)],
    ) {
        let (contacts, row_to_contact, row_diagnostics, tree_qfrc) = match solution {
            Some(solution) => (
                solution.contacts.clone(),
                solution.row_to_contact.clone(),
                solution.row_diagnostics.clone(),
                solution.tree_qfrc.clone(),
            ),
            None => {
                let contacts: Vec<Contact> = collect_contacts_full(
                    &self.bodies,
                    &self.trees,
                    &self.geoms,
                    &self.meshes,
                    pairs,
                )
                .into_iter()
                .filter(|contact| {
                    matches!(
                        self.geoms[contact.geom_a].attachment(),
                        GeomAttach::Link(_, _)
                    ) || matches!(
                        self.geoms[contact.geom_b].attachment(),
                        GeomAttach::Link(_, _)
                    )
                })
                .collect();
                (
                    contacts,
                    Vec::new(),
                    Vec::new(),
                    self.trees.iter().map(|tree| vec![0.0; tree.nv()]).collect(),
                )
            }
        };
        self.last_solver_phase = Some(SolverPhaseDiagnostics {
            qpos: state.0,
            qvel: state.1,
            contacts,
            row_to_contact,
            row_diagnostics,
            tree_qfrc,
        });
    }

    fn solve_tree_contact_sensor_solution(
        &self,
        tree_contacts: &[Contact],
        use_newton: bool,
    ) -> TreeContactSolution {
        let tree_implicit = match self.integrator {
            Integrator::Rk4 => None,
            Integrator::Euler => Some(false),
            Integrator::ImplicitFast => Some(true),
        };
        crate::solver::solve_tree_contacts(
            &self.bodies,
            &self.trees,
            &self.geoms,
            tree_contacts,
            self.gravity,
            self.dt,
            self.solver.cone,
            self.solver.iterations,
            use_newton,
            tree_implicit,
        )
    }

    /// Compute per-body external wrench arrays for a given body-state vector.
    ///
    /// Returns `(force_world, torque_world_about_com)` for each body. Zero for
    /// bodies with no active contacts and (crucially) all-zero when
    /// `self.geoms` is empty, which preserves the tier-1 golden.
    fn compute_wrenches(&self, state: &[Body], pairs: &[(usize, usize)]) -> Vec<(Vec3, Vec3)> {
        let n = state.len();
        let mut out = vec![(Vec3::ZERO, Vec3::ZERO); n];
        if self.geoms.is_empty() {
            return out;
        }
        let contacts = collect_contacts(
            state,
            &self.geoms,
            &self.meshes,
            pairs,
            ContactManifold::Legacy,
        );
        for c in &contacts {
            apply_contact_wrench(&mut out, state, &self.geoms, c);
        }
        self.apply_mocap_wrenches(&mut out, state, pairs);
        out
    }

    fn apply_mocap_wrenches(
        &self,
        out: &mut [(Vec3, Vec3)],
        bodies: &[Body],
        pairs: &[(usize, usize)],
    ) {
        if !self
            .trees
            .iter()
            .any(|tree| tree.links.first().is_some_and(|link| link.mocap))
        {
            return;
        }
        let contacts = collect_contacts_full(bodies, &self.trees, &self.geoms, &self.meshes, pairs);
        let poses: Vec<Vec<(Vec3, Quat)>> =
            self.trees.iter().map(tree_forward_kinematics).collect();
        for contact in contacts {
            let a = self.geoms[contact.geom_a].attachment();
            let b = self.geoms[contact.geom_b].attachment();
            let (body_idx, mocap_tree) = match (a, b) {
                (GeomAttach::Body(body), GeomAttach::Link(tree, _)) => (body, tree),
                (GeomAttach::Link(tree, _), GeomAttach::Body(body)) => (body, tree),
                _ => continue,
            };
            if !self.trees[mocap_tree].links[0].mocap {
                continue;
            }
            apply_one_mocap_wrench(MocapWrenchInput {
                out,
                bodies,
                tree: &self.trees[mocap_tree],
                tree_idx: mocap_tree,
                link_poses: &poses[mocap_tree],
                geoms: &self.geoms,
                contact: &contact,
                body_idx,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// contact assembly and force application
// ---------------------------------------------------------------------------

/// Which narrow-phase dispatch to use when enumerating contacts. Penalty
/// keeps the legacy vertex-vs-face primary for box-box pairs;
/// [`ContactManifold::Full`] routes box-box through SAT face-clipping so
/// tilted face-face stacks see the 4-corner manifold instead of the
/// 2-diagonal degenerate one (see NEWT-14 evidence in
/// `docs/differential.md`). Plane colliders use the shared source-parity
/// primitive rules in both paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContactManifold {
    Legacy,
    Full,
}

fn collect_contacts(
    state: &[Body],
    geoms: &[Geom],
    meshes: &[ConvexMesh],
    pairs: &[(usize, usize)],
    manifold: ContactManifold,
) -> Vec<Contact> {
    let mut out = Vec::new();
    // Pre-compute world poses for every geom in stable index order.
    let poses: Vec<GeomPose> = geoms
        .iter()
        .map(|g| match g.attachment() {
            GeomAttach::Body(i) => geom_world_pose(g, state[i].position, state[i].orientation),
            // Link-attached geoms cannot participate in the body-only path;
            // return a static-style pose (unused because `apply_contact_wrench`
            // ignores link geoms).
            GeomAttach::Link(_, _) | GeomAttach::Static => {
                geom_world_pose(g, Vec3::ZERO, Quat::IDENTITY)
            }
        })
        .collect();

    for &(a, b) in pairs {
        // Skip pairs where either side is a tree link — those are handled
        // in `step_trees`. Body-only path stays bit-identical to tier 2.
        let att_a = geoms[a].attachment();
        let att_b = geoms[b].attachment();
        if matches!(att_a, GeomAttach::Link(_, _)) || matches!(att_b, GeomAttach::Link(_, _)) {
            continue;
        }
        let buf = match manifold {
            ContactManifold::Legacy => {
                narrow_phase(a, &geoms[a], &poses[a], b, &geoms[b], &poses[b], meshes)
            }
            ContactManifold::Full => {
                narrow_phase_solver(a, &geoms[a], &poses[a], b, &geoms[b], &poses[b], meshes)
            }
        };
        for c in buf.as_slice() {
            out.push(*c);
        }
    }
    out
}

/// Contact enumeration that considers both free bodies AND trees. Used by
/// [`World::detect_contacts`] as a diagnostic surface.
fn collect_contacts_full(
    bodies: &[Body],
    trees: &[Tree],
    geoms: &[Geom],
    meshes: &[ConvexMesh],
    pairs: &[(usize, usize)],
) -> Vec<Contact> {
    let mut out = Vec::new();
    // Cache each tree's link poses so we don't redo forward kinematics per
    // geom.
    let tree_poses: Vec<Vec<(Vec3, Quat)>> = trees.iter().map(tree_forward_kinematics).collect();

    let poses: Vec<GeomPose> = geoms
        .iter()
        .map(|g| match g.attachment() {
            GeomAttach::Static => geom_world_pose(g, Vec3::ZERO, Quat::IDENTITY),
            GeomAttach::Body(i) => geom_world_pose(g, bodies[i].position, bodies[i].orientation),
            GeomAttach::Link(t, l) => {
                let (p, o) = tree_poses[t][l];
                geom_world_pose(g, p, o)
            }
        })
        .collect();

    for &(a, b) in pairs {
        let buf = narrow_phase(a, &geoms[a], &poses[a], b, &geoms[b], &poses[b], meshes);
        for c in buf.as_slice() {
            out.push(*c);
        }
    }
    out
}

/// Compute per-link external wrenches for one tree in penalty mode.
/// Iterates the pairs that touch this tree, resolves the OTHER side of each
/// pair (a body, static, or another tree link), and applies the same
/// penalty/friction contact model as tier 2, but records forces only on the
/// tree's own links (Newton's-third-law reactions on free bodies or other
/// trees are dropped — see the v0 simplification note on `step_trees`).
fn tree_wrenches_from_contacts(
    tree: &Tree,
    tree_idx: usize,
    bodies: &[Body],
    geoms: &[Geom],
    meshes: &[ConvexMesh],
    pairs: &[(usize, usize)],
) -> Vec<(Vec3, Vec3)> {
    let n_links = tree.links.len();
    let mut out = vec![(Vec3::ZERO, Vec3::ZERO); n_links];
    if pairs.is_empty() {
        return out;
    }
    // Forward kinematics for this tree (sub-stage state).
    let link_poses = tree_forward_kinematics(tree);
    // Geom world poses restricted to geoms mentioned in `pairs`.
    let pose_of = |g: &Geom| -> GeomPose {
        match g.attachment() {
            GeomAttach::Static => geom_world_pose(g, Vec3::ZERO, Quat::IDENTITY),
            GeomAttach::Body(i) => geom_world_pose(g, bodies[i].position, bodies[i].orientation),
            GeomAttach::Link(t, l) => {
                if t == tree_idx {
                    let (p, o) = link_poses[l];
                    geom_world_pose(g, p, o)
                } else {
                    // Other-tree pose from its stored state (step-start).
                    // v0 does not support cross-tree contacts anyway; return
                    // a static-style pose so a narrow-phase call is well-
                    // defined but likely produces no penetration in the
                    // demos we care about.
                    geom_world_pose(g, Vec3::ZERO, Quat::IDENTITY)
                }
            }
        }
    };
    for &(a, b) in pairs {
        let ga = &geoms[a];
        let gb = &geoms[b];
        let pose_a = pose_of(ga);
        let pose_b = pose_of(gb);
        let buf = narrow_phase(a, ga, &pose_a, b, gb, &pose_b, meshes);
        for c in buf.as_slice() {
            apply_tree_contact_wrench(&mut out, tree, tree_idx, &link_poses, bodies, geoms, c);
        }
    }
    out
}

/// Apply one penalty contact's wrench to a link of the given tree, following the
/// same penalty / pyramidal-friction model as the free-body path. The
/// "other side" of the contact contributes only its point velocity for the
/// relative-normal-velocity term; equal-opposite reaction on the other side
/// is discarded (v0 simplification — see `step_trees`).
fn apply_tree_contact_wrench(
    ext: &mut [(Vec3, Vec3)],
    tree: &Tree,
    tree_idx: usize,
    link_poses: &[(Vec3, Quat)],
    bodies: &[Body],
    geoms: &[Geom],
    contact: &Contact,
) {
    let ga = &geoms[contact.geom_a];
    let gb = &geoms[contact.geom_b];
    let normal = contact.normal_world;

    // Effective mass: for a link-vs-static contact, use the link's own mass.
    // Cross-tree/body-link cases fall back to the link's own mass (v0).
    let link_mass =
        |t: usize, l: usize| -> f32 { tree.links[l].mass * (t == tree_idx) as u8 as f32 };
    let m_eff = {
        let ma = match ga.attachment() {
            GeomAttach::Link(t, l) => link_mass(t, l),
            GeomAttach::Body(i) => bodies[i].mass,
            GeomAttach::Static => 0.0,
        };
        let mb = match gb.attachment() {
            GeomAttach::Link(t, l) => link_mass(t, l),
            GeomAttach::Body(i) => bodies[i].mass,
            GeomAttach::Static => 0.0,
        };
        if ma > 0.0 && mb > 0.0 {
            ma * mb / (ma + mb)
        } else if ma > 0.0 {
            ma
        } else if mb > 0.0 {
            mb
        } else {
            return;
        }
    };

    let solref = combine_solref(ga.solref, gb.solref);
    let (k, c) = solref_to_kc(solref, m_eff);
    let c_tangent = c;
    // Point velocities.
    let (v_a, _wa, r_a) = point_velocity_generic(
        ga,
        contact.position_world,
        tree,
        tree_idx,
        link_poses,
        bodies,
    );
    let (v_b, _wb, r_b) = point_velocity_generic(
        gb,
        contact.position_world,
        tree,
        tree_idx,
        link_poses,
        bodies,
    );
    let v_rel = v_a - v_b;
    let v_n = v_rel.dot(normal);
    // Gap subtract: force only applies once the shifted penetration exceeds
    // the gap; sensing-only contacts (pen ≤ gap) fire in the contact list
    // but contribute zero wrench.
    let pen_eff = contact.penetration - contact.gap;
    if pen_eff <= 0.0 {
        return;
    }
    let f_n_raw = k * pen_eff - c * v_n;
    let f_n = if f_n_raw > 0.0 { f_n_raw } else { 0.0 };
    if f_n <= 0.0 {
        return;
    }
    let (t1, t2) = tangent_basis(normal);
    let v_t = v_rel - normal * v_n;
    let v_t1 = v_t.dot(t1);
    let v_t2 = v_t.dot(t2);
    let cap = contact.friction * f_n;
    let f_t1 = clamp_symmetric(-c_tangent * v_t1, cap);
    let f_t2 = clamp_symmetric(-c_tangent * v_t2, cap);
    let force_on_a = normal * f_n + t1 * f_t1 + t2 * f_t2;

    if let GeomAttach::Link(t, l) = ga.attachment() {
        if t == tree_idx {
            let (f, tau) = &mut ext[l];
            *f += force_on_a;
            *tau += r_a.cross(force_on_a);
        }
    }
    if let GeomAttach::Link(t, l) = gb.attachment() {
        if t == tree_idx {
            let force_on_b = -force_on_a;
            let (f, tau) = &mut ext[l];
            *f += force_on_b;
            *tau += r_b.cross(force_on_b);
        }
    }
}

/// Point velocity + angular velocity + moment-arm at a world position for a
/// geom attached to a link/body/static. Returns `(v_world, ω_world, r_arm)`
/// where `r_arm` is the vector from the anchor's COM to the contact point.
fn point_velocity_generic(
    geom: &Geom,
    contact_pos_world: Vec3,
    tree: &Tree,
    tree_idx: usize,
    link_poses: &[(Vec3, Quat)],
    bodies: &[Body],
) -> (Vec3, Vec3, Vec3) {
    match geom.attachment() {
        GeomAttach::Static => (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),
        GeomAttach::Body(i) => {
            let body = &bodies[i];
            let r = contact_pos_world - body.position;
            let w = body.angular_velocity_world();
            (body.linear_velocity + w.cross(r), w, r)
        }
        GeomAttach::Link(t, l) => {
            if t == tree_idx {
                let (com, _ori) = link_poses[l];
                let (v_world, w_world) = link_world_velocity(tree, l, link_poses);
                let r = contact_pos_world - com;
                (v_world + w_world.cross(r), w_world, r)
            } else {
                (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO)
            }
        }
    }
}

/// World-frame (linear-at-COM, angular) velocity of a link in the given
/// tree. Computed by walking the tree's spatial-velocity recursion from
/// the root — same layout as ABA pass 1 but keeping only what the contact
/// code needs.
fn link_world_velocity(tree: &Tree, target: usize, link_poses: &[(Vec3, Quat)]) -> (Vec3, Vec3) {
    use crate::joint::JointKind;
    let n = tree.links.len();
    // Ancestor chain root → target.
    let mut chain = vec![target];
    let mut cur = target;
    while let Some(p) = tree.links[cur].parent {
        chain.push(p);
        cur = p;
    }
    chain.reverse();
    let mut v_world = vec![Vec3::ZERO; n];
    let mut w_world = vec![Vec3::ZERO; n];
    // Seed root.
    let root = chain[0];
    if tree.links[root].mocap {
        w_world[root] = tree.mocap_angular_velocity;
        v_world[root] = tree.mocap_linear_velocity;
    } else if tree.links[root].joint == JointKind::Free {
        let (_pos, ori) = link_poses[root];
        let wb = Vec3::new(tree.qdot[0], tree.qdot[1], tree.qdot[2]);
        let vb = Vec3::new(tree.qdot[3], tree.qdot[4], tree.qdot[5]);
        w_world[root] = ori.rotate(wb);
        v_world[root] = ori.rotate(vb);
    }
    // Walk down the chain.
    for &i in chain.iter().skip(1) {
        let parent = tree.links[i].parent.unwrap();
        let (child_pos, _child_ori) = link_poses[i];
        let (parent_pos, parent_ori) = link_poses[parent];
        match tree.links[i].joint {
            JointKind::Hinge { axis, .. } => {
                let axis_world = parent_ori.rotate(axis);
                let qdot_i = tree.hinge_rate(i);
                w_world[i] = w_world[parent] + axis_world * qdot_i;
                let joint_world =
                    parent_pos + parent_ori.rotate(tree.links[i].joint_offset_in_parent.0);
                let v_parent_at_child_com =
                    v_world[parent] + w_world[parent].cross(child_pos - parent_pos);
                v_world[i] =
                    v_parent_at_child_com + (axis_world * qdot_i).cross(child_pos - joint_world);
            }
            JointKind::Slide { axis, .. } => {
                // Slide never rotates; child ω = parent ω. Child COM adds
                // the joint's linear velocity (axis * qdot) to the parent's
                // point velocity at the child COM.
                let axis_world = parent_ori.rotate(axis);
                let qdot_i = tree.slide_rate(i);
                w_world[i] = w_world[parent];
                let v_parent_at_child_com =
                    v_world[parent] + w_world[parent].cross(child_pos - parent_pos);
                v_world[i] = v_parent_at_child_com + axis_world * qdot_i;
            }
            JointKind::Ball { .. } => {
                // Body-frame ω on the child; rotate into world by the child
                // orientation (equivalent to parent_ori * q_ball, but we
                // already computed it in link_poses).
                let (_, child_ori) = link_poses[i];
                let omega_body = tree.ball_omega(i);
                let omega_child_world = child_ori.rotate(omega_body);
                w_world[i] = w_world[parent] + omega_child_world;
                // The ball's joint anchor coincides with parent's anchor —
                // the ball doesn't translate; only rotates.
                let joint_world =
                    parent_pos + parent_ori.rotate(tree.links[i].joint_offset_in_parent.0);
                let v_parent_at_child_com =
                    v_world[parent] + w_world[parent].cross(child_pos - parent_pos);
                v_world[i] =
                    v_parent_at_child_com + omega_child_world.cross(child_pos - joint_world);
            }
            JointKind::Fixed => {
                w_world[i] = w_world[parent];
                v_world[i] = v_world[parent] + w_world[parent].cross(child_pos - parent_pos);
            }
            JointKind::Free => {
                // A non-root free joint isn't allowed in v0.
            }
        }
    }
    (v_world[target], w_world[target])
}

/// Apply one contact's wrench to the appropriate body/bodies. Static geoms
/// (no owning body) simply absorb the reaction.
fn apply_contact_wrench(
    ext: &mut [(Vec3, Vec3)],
    state: &[Body],
    geoms: &[Geom],
    contact: &Contact,
) {
    let ga = &geoms[contact.geom_a];
    let gb = &geoms[contact.geom_b];
    let normal = contact.normal_world;

    // Reduced mass. Static geoms behave as infinite mass.
    let m_eff = match (ga.body, gb.body) {
        (Some(ia), Some(ib)) => {
            let ma = state[ia].mass;
            let mb = state[ib].mass;
            ma * mb / (ma + mb)
        }
        (Some(ia), None) => state[ia].mass,
        (None, Some(ib)) => state[ib].mass,
        (None, None) => return, // static-static: ignore (auto_pairs already skips)
    };

    let solref = combine_solref(ga.solref, gb.solref);
    let (k, c) = solref_to_kc(solref, m_eff);
    let c_tangent = c;
    // Point velocities at the contact.
    let (v_a, w_a_world, r_a) = point_velocity(state, ga, contact.position_world);
    let (v_b, w_b_world, r_b) = point_velocity(state, gb, contact.position_world);
    let _ = (w_a_world, w_b_world); // returned for completeness
    let v_rel = v_a - v_b;

    let v_n = v_rel.dot(normal);
    // Gap subtract: force only applies once the shifted penetration exceeds
    // the gap. See `Contact::gap` in `contact.rs`.
    let pen_eff = contact.penetration - contact.gap;
    if pen_eff <= 0.0 {
        return;
    }
    // Normal force magnitude: spring + damping opposing closing motion.
    // Clamped at zero — contacts cannot pull.
    let f_n_raw = k * pen_eff - c * v_n;
    let f_n = if f_n_raw > 0.0 { f_n_raw } else { 0.0 };
    if f_n <= 0.0 {
        return;
    }

    // Deterministic tangent basis.
    let (t1, t2) = tangent_basis(normal);
    let v_t = v_rel - normal * v_n;
    let v_t1 = v_t.dot(t1);
    let v_t2 = v_t.dot(t2);

    let cap = contact.friction * f_n;
    let f_t1 = clamp_symmetric(-c_tangent * v_t1, cap);
    let f_t2 = clamp_symmetric(-c_tangent * v_t2, cap);

    let force_on_a = normal * f_n + t1 * f_t1 + t2 * f_t2;

    if let Some(ia) = ga.body {
        let (f, tau) = &mut ext[ia];
        *f += force_on_a;
        *tau += r_a.cross(force_on_a);
    }
    if let Some(ib) = gb.body {
        let force_on_b = -force_on_a;
        let (f, tau) = &mut ext[ib];
        *f += force_on_b;
        *tau += r_b.cross(force_on_b);
    }
}

struct MocapWrenchInput<'a> {
    out: &'a mut [(Vec3, Vec3)],
    bodies: &'a [Body],
    tree: &'a Tree,
    tree_idx: usize,
    link_poses: &'a [(Vec3, Quat)],
    geoms: &'a [Geom],
    contact: &'a Contact,
    body_idx: usize,
}

fn apply_one_mocap_wrench(input: MocapWrenchInput<'_>) {
    let MocapWrenchInput {
        out,
        bodies,
        tree,
        tree_idx,
        link_poses,
        geoms,
        contact,
        body_idx,
    } = input;
    let ga = &geoms[contact.geom_a];
    let gb = &geoms[contact.geom_b];
    let (v_a, _, r_a) = point_velocity_generic(
        ga,
        contact.position_world,
        tree,
        tree_idx,
        link_poses,
        bodies,
    );
    let (v_b, _, r_b) = point_velocity_generic(
        gb,
        contact.position_world,
        tree,
        tree_idx,
        link_poses,
        bodies,
    );
    let body_is_a = matches!(ga.attachment(), GeomAttach::Body(index) if index == body_idx);
    let body_mass = bodies[body_idx].mass;
    let (k, damping) = solref_to_kc(combine_solref(ga.solref, gb.solref), body_mass);
    let v_rel = v_a - v_b;
    let normal = contact.normal_world;
    let v_n = v_rel.dot(normal);
    let penetration = contact.penetration - contact.gap;
    if penetration <= 0.0 {
        return;
    }
    let normal_force = (k * penetration - damping * v_n).max(0.0);
    if normal_force <= 0.0 {
        return;
    }
    let (t1, t2) = tangent_basis(normal);
    let tangent_velocity = v_rel - normal * v_n;
    let cap = contact.friction * normal_force;
    let force_on_a = normal * normal_force
        + t1 * clamp_symmetric(-damping * tangent_velocity.dot(t1), cap)
        + t2 * clamp_symmetric(-damping * tangent_velocity.dot(t2), cap);
    let force_on_body = if body_is_a { force_on_a } else { -force_on_a };
    let moment_arm = if body_is_a { r_a } else { r_b };
    out[body_idx].0 += force_on_body;
    out[body_idx].1 += moment_arm.cross(force_on_body);
}

/// World-frame linear velocity of the contact point on a geom's parent body.
/// Returns `(v_point_world, w_world, r_arm_world)` where `r_arm_world` is
/// the vector from the body's COM to the contact point. For static geoms
/// returns zero vectors and a zero arm.
fn point_velocity(state: &[Body], geom: &Geom, contact_pos_world: Vec3) -> (Vec3, Vec3, Vec3) {
    match geom.body {
        Some(i) => {
            let body = &state[i];
            let r = contact_pos_world - body.position;
            let w_world = body.angular_velocity_world();
            (body.linear_velocity + w_world.cross(r), w_world, r)
        }
        None => (Vec3::ZERO, Vec3::ZERO, Vec3::ZERO),
    }
}

/// Deterministic orthonormal tangent basis `(t1, t2)` perpendicular to a unit
/// normal `n`. Picks the reference axis (X or Z) whose alignment with `n` is
/// weakest so the cross product does not underflow.
pub fn tangent_basis(n: Vec3) -> (Vec3, Vec3) {
    // Pick the world reference axis less aligned with `n`.
    let use_z_ref = crate::math::abs(n.z) < 0.9;
    let reference = if use_z_ref { Vec3::Z } else { Vec3::X };
    let t1 = reference.cross(n).normalize();
    let t2 = n.cross(t1);
    (t1, t2)
}

/// Human-readable shape name for panic messages.
fn shape_name(s: GeomShape) -> &'static str {
    match s {
        GeomShape::Plane => "plane",
        GeomShape::Sphere { .. } => "sphere",
        GeomShape::Box { .. } => "box",
        GeomShape::Capsule { .. } => "capsule",
        GeomShape::Cylinder { .. } => "cylinder",
        GeomShape::Ellipsoid { .. } => "ellipsoid",
        GeomShape::Mesh { .. } => "mesh",
    }
}

/// Penalty-formula normal-force magnitude for one contact: the same
/// `max(0, k · pen_eff − c · v_n)` [`World`] uses when computing per-body
/// contact wrenches under [`SolverMode::Penalty`]. Used by the touch
/// sensor input assembly so a touch reading in Penalty mode reports the
/// force the world would actually apply at this contact.
fn penalty_normal_force(c: &Contact, bodies: &[Body], trees: &[Tree], geoms: &[Geom]) -> f32 {
    let ga = &geoms[c.geom_a];
    let gb = &geoms[c.geom_b];
    // Mass resolution mirrors `apply_tree_contact_wrench`: body geoms use
    // their body mass; link geoms use their link mass; static geoms count
    // as infinite (treated as zero here so the pair reduces to the
    // dynamic side alone). Missing the link branch was the touch-on-link
    // blocker: link-attached feet always read 0 N.
    let mass_for = |att: GeomAttach| -> f32 {
        match att {
            GeomAttach::Body(i) => bodies[i].mass,
            GeomAttach::Link(t, l) => trees[t].links[l].mass,
            GeomAttach::Static => 0.0,
        }
    };
    let ma = mass_for(ga.attachment());
    let mb = mass_for(gb.attachment());
    let m_eff = if ma > 0.0 && mb > 0.0 {
        ma * mb / (ma + mb)
    } else if ma > 0.0 {
        ma
    } else if mb > 0.0 {
        mb
    } else {
        return 0.0;
    };
    let solref = combine_solref(ga.solref, gb.solref);
    let (k, c_damp) = solref_to_kc(solref, m_eff);
    let pen_eff = c.penetration - c.gap;
    if pen_eff <= 0.0 {
        return 0.0;
    }
    // World-frame point velocity at the contact for each side. Body geoms
    // read directly from `Body`; link geoms walk the tree's velocity
    // recursion via the sensor helper (avoids duplicating the ω → v_world
    // machinery). Static geoms contribute zero.
    let point_v = |att: GeomAttach| -> Vec3 {
        match att {
            GeomAttach::Static => Vec3::ZERO,
            GeomAttach::Body(i) => {
                let b = &bodies[i];
                let r = c.position_world - b.position;
                b.linear_velocity + b.angular_velocity_world().cross(r)
            }
            GeomAttach::Link(t, l) => {
                let tree = &trees[t];
                let poses = tree_forward_kinematics(tree);
                let (com, ori) = poses[l];
                let (v_lin_body, omega_body) = crate::sensor::link_body_frame_vw_at_rest(tree, l);
                let v_com_world = ori.rotate(v_lin_body);
                let omega_world = ori.rotate(omega_body);
                let r = c.position_world - com;
                v_com_world + omega_world.cross(r)
            }
        }
    };
    let v_a = point_v(ga.attachment());
    let v_b = point_v(gb.attachment());
    let v_n = (v_a - v_b).dot(c.normal_world);
    let f = k * pen_eff - c_damp * v_n;
    if f > 0.0 { f } else { 0.0 }
}

fn clamp_symmetric(x: f32, cap: f32) -> f32 {
    if x > cap {
        cap
    } else if x < -cap {
        -cap
    } else {
        x
    }
}

// ---------------------------------------------------------------------------
// RK4 stage helpers
// ---------------------------------------------------------------------------

/// Body-state derivative under gravity + one external world-frame wrench.
#[derive(Clone, Copy, Debug)]
struct Deriv {
    dposition: Vec3,
    dlinear_velocity: Vec3,
    dorientation: Quat,
    dangular_velocity_body: Vec3,
}

/// Body derivative given gravity and a world-frame external wrench.
///
/// Bit-identical to tier-1 `evaluate` when the wrench is `(ZERO, ZERO)`:
/// `Vec3::ZERO / mass = ZERO`, `gravity + ZERO = gravity`, and
/// `Vec3::ZERO − X = −X` per component in IEEE arithmetic.
fn evaluate(state: Body, gravity: Vec3, ext_force_world: Vec3, ext_torque_world: Vec3) -> Deriv {
    let dlin = gravity + ext_force_world / state.mass;

    let iw = state.inertia_body * state.angular_velocity_body;
    let gyroscopic = -state.angular_velocity_body.cross(iw);
    let tau_body = state.orientation.inverse_rotate(ext_torque_world);
    let dang_body = state.inertia_body_inverse * (tau_body + gyroscopic);

    let dq = state.orientation.derivative(state.angular_velocity_body);
    let dp = state.linear_velocity;

    Deriv {
        dposition: dp,
        dlinear_velocity: dlin,
        dorientation: dq,
        dangular_velocity_body: dang_body,
    }
}

fn evaluate_all(states: &[Body], gravity: Vec3, ext: &[(Vec3, Vec3)]) -> Vec<Deriv> {
    let mut out = Vec::with_capacity(states.len());
    for (i, &s) in states.iter().enumerate() {
        let (f, tau) = ext[i];
        out.push(evaluate(s, gravity, f, tau));
    }
    out
}

/// Advance a whole slice of bodies by `origin + deriv * dt`. Used at each
/// RK4 sub-stage with `origin = s0` (the step's start state).
fn advance_all(origin: &[Body], deriv: &[Deriv], dt: f32) -> Vec<Body> {
    let mut out = Vec::with_capacity(origin.len());
    for (i, &state) in origin.iter().enumerate() {
        let d = deriv[i];
        out.push(Body {
            position: state.position + d.dposition * dt,
            linear_velocity: state.linear_velocity + d.dlinear_velocity * dt,
            orientation: state.orientation + d.dorientation * dt,
            angular_velocity_body: state.angular_velocity_body + d.dangular_velocity_body * dt,
            ..state
        });
    }
    out
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_basis_is_orthonormal_for_various_normals() {
        let normals = [
            Vec3::Z,
            -Vec3::Z,
            Vec3::X,
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(0.1, 0.7, 0.7).normalize(),
        ];
        for &n in &normals {
            let (t1, t2) = tangent_basis(n);
            assert!((t1.length() - 1.0).abs() < 1e-5, "t1 not unit for {n:?}");
            assert!((t2.length() - 1.0).abs() < 1e-5, "t2 not unit for {n:?}");
            assert!(t1.dot(n).abs() < 1e-5, "t1 not ⟂ n for {n:?}");
            assert!(t2.dot(n).abs() < 1e-5, "t2 not ⟂ n for {n:?}");
            assert!(t1.dot(t2).abs() < 1e-5, "t1 not ⟂ t2 for {n:?}");
        }
    }

    #[test]
    fn empty_world_step_matches_tier1_single_body_free_fall() {
        // Confirm the refactor did not perturb the gravity-only path. Single
        // body under gravity for 100 steps: y' = v0 * t + ½ g t².
        let mut w = World::new();
        w.dt = 0.005;
        w.gravity = Vec3::new(0.0, 0.0, -10.0);
        let b = Body::solid_box(1.0, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY);
        w.add_body(b);
        for _ in 0..100 {
            w.step();
        }
        // t = 0.5 s. Expected z = ½ * (-10) * 0.25 = -1.25.
        let z = w.bodies[0].position.z;
        assert!((z - (-1.25)).abs() < 1e-3, "free-fall z {z}");
    }

    #[test]
    fn auto_pairs_skip_same_body_and_static_static() {
        let mut w = World::new();
        let bi = w.add_body(Body::solid_sphere(1.0, 0.5, Vec3::ZERO, Quat::IDENTITY));
        // Two static planes (both body=None) should not pair with each other.
        let sa = w.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5));
        let sb = w.add_geom(Geom::static_plane(Vec3::new(0.0, 0.0, 1.0), Vec3::Z, 0.5));
        // Two spheres on the same body should not pair with each other.
        let g1 = w.add_geom(Geom::sphere(bi, 0.1, Vec3::new(-0.2, 0.0, 0.0), 0.5));
        let g2 = w.add_geom(Geom::sphere(bi, 0.1, Vec3::new(0.2, 0.0, 0.0), 0.5));
        let pairs = w.auto_pairs();
        assert!(!pairs.contains(&(sa, sb)));
        assert!(!pairs.contains(&(g1, g2)));
        // sphere-vs-plane pairs must be present.
        assert!(pairs.contains(&(sa, g1)));
    }

    #[test]
    fn evaluate_matches_tier1_when_wrench_zero() {
        // Discriminating case: an asymmetric-inertia body with nontrivial ω;
        // evaluate with zero external wrench must reproduce the tier-1 dω/dt
        // exactly (bit-identical is not required in a mutation test — a tight
        // numerical bound is — but the derivation is that they *are*
        // arithmetically identical up to `+ 0` and `0 − X = −X`).
        let mut b = Body::new(
            1.0,
            crate::math::Mat3::diag(1.0, 2.0, 3.0),
            Vec3::ZERO,
            Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.3), 0.7),
        );
        b.angular_velocity_body = Vec3::new(0.5, 1.3, -0.7);
        b.linear_velocity = Vec3::new(0.1, 0.2, 0.3);
        let d_new = evaluate(b, Vec3::new(0.0, 0.0, -9.81), Vec3::ZERO, Vec3::ZERO);
        // Recreate tier-1 form directly:
        let iw = b.inertia_body * b.angular_velocity_body;
        let gyro = -b.angular_velocity_body.cross(iw);
        let dang_ref = b.inertia_body_inverse * gyro;
        let diff = d_new.dangular_velocity_body - dang_ref;
        assert!(diff.length() < 1e-6, "wrench-free path diverged");
        assert_eq!(d_new.dlinear_velocity, Vec3::new(0.0, 0.0, -9.81));
    }
}
