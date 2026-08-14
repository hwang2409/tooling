//! World: free rigid bodies + geoms, penalty contacts, RK4 integration.
//!
//! # Scope (tier 2)
//!
//! Free bodies under uniform gravity plus contact forces from a penalty model.
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
use crate::contact::{Contact, is_pair_supported, narrow_phase};
use crate::geom::{
    ConvexMesh, Geom, GeomAttach, GeomPose, GeomShape, combine_solref, geom_world_pose,
    solref_to_kc,
};
use crate::math::{Quat, Vec3};
use crate::solver::{SolverConfig, SolverMode, solve_free_bodies};
use crate::tree::{Tree, forward_kinematics as tree_forward_kinematics, rk4_step as tree_rk4_step};

/// Simulation world.
#[derive(Clone, Debug)]
pub struct World {
    /// Fixed integration timestep. Default 5 ms (matches biped).
    pub dt: f32,
    /// Uniform gravity vector applied to every body's COM.
    pub gravity: Vec3,
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
    /// Constraint solver configuration (v1 tier 4). Default is
    /// [`SolverConfig::DEFAULT`] — `SolverMode::Penalty`, which keeps every
    /// pre-v1-tier-4 golden byte-identical. Set to
    /// `SolverMode::Pgs` to switch on the MuJoCo soft-constraint solver.
    pub solver: SolverConfig,
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
}

// Manual PartialEq: the pair-check cache is not part of logical world state.
// Two worlds with identical bodies/trees/geoms/meshes/pair_list are equal
// regardless of whether either has run the pair check.
impl PartialEq for World {
    fn eq(&self, other: &Self) -> bool {
        self.dt == other.dt
            && self.gravity == other.gravity
            && self.bodies == other.bodies
            && self.trees == other.trees
            && self.geoms == other.geoms
            && self.meshes == other.meshes
            && self.pair_list == other.pair_list
            && self.solver == other.solver
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
            gravity: Vec3::new(0.0, 0.0, -9.81),
            bodies: Vec::new(),
            trees: Vec::new(),
            geoms: Vec::new(),
            meshes: Vec::new(),
            pair_list: None,
            solver: SolverConfig::DEFAULT,
            checked_pairs: std::cell::Cell::new(0),
        }
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

    /// Advance the whole world by one fixed-dt RK4 step.
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
        // Loud engine-level enforcement: the first step after any pair-list
        // or geom-count change panics if any ACTIVE pair falls in the
        // deferred bucket. Prevents a stack.json-style silent no-op.
        self.assert_pairs_supported();
        let pairs = match &self.pair_list {
            Some(p) => p.clone(),
            None => self.auto_pairs(),
        };
        self.step_bodies(&pairs);
        self.step_trees(&pairs);
    }

    /// Advance only the free bodies. Preserves the tier-1/2 behavior
    /// bit-for-bit when no tree links are in play.
    fn step_bodies(&mut self, pairs: &[(usize, usize)]) {
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
            SolverMode::Pgs => Some(self.compute_solver_wrenches(&s0, pairs)),
        };
        let sample_wrenches = |state: &[Body], pairs: &[(usize, usize)]| -> Vec<(Vec3, Vec3)> {
            match &solver_zoh {
                Some(w) => w.clone(),
                None => self.compute_wrenches(state, pairs),
            }
        };

        let ext1 = sample_wrenches(&s0, pairs);
        let k1 = evaluate_all(&s0, self.gravity, &ext1);

        let s1 = advance_all(&s0, &s0, &k1, self.dt * 0.5);
        let ext2 = sample_wrenches(&s1, pairs);
        let k2 = evaluate_all(&s1, self.gravity, &ext2);

        let s2 = advance_all(&s0, &s0, &k2, self.dt * 0.5);
        let ext3 = sample_wrenches(&s2, pairs);
        let k3 = evaluate_all(&s2, self.gravity, &ext3);

        let s3 = advance_all(&s0, &s0, &k3, self.dt);
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

    /// Advance the kinematic trees under gravity + contact wrenches.
    ///
    /// v0 simplification: cross-integration between free bodies and tree
    /// links within a single sub-stage is NOT modeled. Contacts that touch
    /// a tree link generate a wrench for the link's tree only; the other
    /// side (a free body or a static geom) is treated as a "wall" for the
    /// tree. Contacts that touch only free bodies are handled by
    /// [`Self::step_bodies`]. Trees are integrated independently of each
    /// other in stable index order.
    fn step_trees(&mut self, pairs: &[(usize, usize)]) {
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
            if solver_mode == SolverMode::Pgs {
                solver_qfrc_delta = crate::solver::solve_tree_limits(&tree, dt, solver_iterations);
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
                    tree_wrenches_from_contacts(
                        t,
                        ti,
                        bodies_ref,
                        geoms_ref,
                        meshes_ref,
                        &tree_pairs,
                    )
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
    /// Runs the PGS solve at `state` (typically s0 — start of RK4 step) and
    /// returns per-body `(force_world, torque_world_at_com)` to hold
    /// constant across all four RK4 sub-stages. Contacts touching tree
    /// links are dropped (v1-tier-4 scope: cross-tree/body contacts remain
    /// on the penalty pathway; see newt/docs/solver.md).
    fn compute_solver_wrenches(
        &self,
        state: &[Body],
        pairs: &[(usize, usize)],
    ) -> Vec<(Vec3, Vec3)> {
        let n = state.len();
        if self.geoms.is_empty() {
            return vec![(Vec3::ZERO, Vec3::ZERO); n];
        }
        // Filter pairs to free-body-only ones (both sides Body or Static).
        let mut free_pairs: Vec<(usize, usize)> = Vec::with_capacity(pairs.len());
        for &(a, b) in pairs {
            let att_a = self.geoms[a].attachment();
            let att_b = self.geoms[b].attachment();
            if matches!(att_a, GeomAttach::Link(_, _)) || matches!(att_b, GeomAttach::Link(_, _)) {
                continue;
            }
            free_pairs.push((a, b));
        }
        let contacts = collect_contacts(state, &self.geoms, &self.meshes, &free_pairs);
        solve_free_bodies(
            state,
            &self.geoms,
            &contacts,
            self.gravity,
            self.dt,
            self.solver.cone,
            self.solver.iterations,
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
        let contacts = collect_contacts(state, &self.geoms, &self.meshes, pairs);
        for c in &contacts {
            apply_contact_wrench(&mut out, state, &self.geoms, c);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// contact assembly and force application
// ---------------------------------------------------------------------------

fn collect_contacts(
    state: &[Body],
    geoms: &[Geom],
    meshes: &[ConvexMesh],
    pairs: &[(usize, usize)],
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
        let buf = narrow_phase(a, &geoms[a], &poses[a], b, &geoms[b], &poses[b], meshes);
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

/// Compute per-link external wrenches for one tree at its current state.
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

/// Apply one contact's wrench to a link of the given tree, following the
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
    if tree.links[root].joint == JointKind::Free {
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

/// Advance a whole slice of bodies by (from_state + deriv * dt). `origin` is
/// the RK4 stage anchor — always `s0` in our loop.
fn advance_all(origin: &[Body], _from: &[Body], deriv: &[Deriv], dt: f32) -> Vec<Body> {
    // NOTE on `_from`: kept for signature symmetry; we always advance from
    // the anchor `origin` = s0 (standard RK4). The parameter is unused today
    // but reserved for future integrators that treat the stage state as a
    // proper linearization point.
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
