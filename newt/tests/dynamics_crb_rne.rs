//! Anchor tests for the CRB mass matrix and RNE inverse dynamics.
//!
//! Primary anchor: the ABA↔RNE round-trip identity on a mixed-joint tree.
//! ABA and RNE share only the spatial-algebra primitives (`Xform`,
//! `SpatialInertia`, `Mat6`); a mutation that produces the wrong per-link
//! torque in either algorithm cannot be silently confirmed by the other,
//! so `RNE(ABA(τ)) ≡ τ` is a strong cross-validation. See
//! `newt/docs/dynamics.md` for the derivation.

use newt::dynamics::{bias_forces, cholesky, cholesky_solve, inverse_dynamics, mass_matrix};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, aba, forward_kinematics};

// ---------------------------------------------------------------------------
// Deterministic scalar PRNG — a small linear congruential generator (Numerical
// Recipes / Knuth's constants). Purely hand-rolled so newt's zero-dep policy
// stays intact and the tests are byte-identical across platforms without
// depending on the platform libm.
// ---------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u32(&mut self) -> u32 {
        // Knuth's MMIX LCG constants (a = 6364136223846793005, c = 1442695040888963407).
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.state >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        // 24-bit uniform in [0, 1).
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }
    /// Uniform in `[-a, a]`.
    fn sym(&mut self, a: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * a
    }
}

// ---------------------------------------------------------------------------
// Mixed-joint scene builders. Symmetry is broken in every anchor per the
// tier-2 lesson (a symmetric tree can hide a zero-lever-arm bug: mutants
// still produce bit-identical trajectories under a mislabeled inertia arm).
// ---------------------------------------------------------------------------

fn box_inertia(mass: f32, hx: f32, hy: f32, hz: f32) -> Mat3 {
    let hx2 = hx * hx;
    let hy2 = hy * hy;
    let hz2 = hz * hz;
    Mat3::diag(
        (mass / 3.0) * (hy2 + hz2),
        (mass / 3.0) * (hx2 + hz2),
        (mass / 3.0) * (hx2 + hy2),
    )
}

/// Fixed root + hinge + slide + ball. Every link has broken symmetry: the
/// hinge axis is oblique, the slide axis is not axis-aligned, and the ball
/// joint's anchor sits off-COM. Non-zero anchors in both parent and child
/// frames so the joint-offset lever arms actually load-bear.
fn mixed_fixed_root() -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.1, -0.2, 0.35), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Link 1: hinge on oblique axis.
    let axis1 = Vec3::new(1.0, 0.3, -0.2).normalize();
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: axis1,
            range: None,
            damping: 0.0,
            armature: 0.02,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::new(0.05, 0.0, -0.1), Quat::IDENTITY),
        (Vec3::new(-0.15, 0.0, 0.3), Quat::IDENTITY),
        0.9,
        box_inertia(0.9, 0.15, 0.1, 0.3),
    ));
    // Link 2: slide on axis not aligned with the previous.
    let axis2 = Vec3::new(0.2, 1.0, 0.4).normalize();
    tree.push_link(Link::new(
        Some(1),
        JointKind::Slide {
            axis: axis2,
            range: None,
            damping: 0.0,
            armature: 0.05,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::new(0.0, 0.0, -0.3), Quat::IDENTITY),
        (Vec3::new(0.1, -0.2, 0.15), Quat::IDENTITY),
        0.7,
        box_inertia(0.7, 0.1, 0.2, 0.15),
    ));
    // Link 3: ball with non-COM anchor.
    tree.push_link(Link::new(
        Some(2),
        JointKind::Ball {
            damping: 0.0,
            armature: 0.01,
        },
        (Vec3::new(0.1, -0.2, -0.15), Quat::IDENTITY),
        (Vec3::new(-0.05, 0.1, 0.2), Quat::IDENTITY),
        0.5,
        Mat3::diag(0.02, 0.03, 0.04),
    ));
    tree
}

/// Free root + hinge + slide + ball, mirroring `mixed_fixed_root` but with a
/// floating base. Used to pin the free-root path.
fn mixed_free_root() -> Tree {
    let mut tree = Tree::new();
    let q0 = Quat::from_axis_angle(Vec3::new(0.5, 1.0, -0.3).normalize(), 0.7);
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(0.4, -0.5, 1.2), q0),
        (Vec3::ZERO, Quat::IDENTITY),
        2.0,
        box_inertia(2.0, 0.2, 0.15, 0.3),
    ));
    let axis1 = Vec3::new(1.0, 0.2, 0.1).normalize();
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: axis1,
            range: None,
            damping: 0.0,
            armature: 0.03,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::new(0.0, 0.0, -0.3), Quat::IDENTITY),
        (Vec3::new(0.05, -0.1, 0.2), Quat::IDENTITY),
        0.8,
        box_inertia(0.8, 0.12, 0.08, 0.22),
    ));
    let axis2 = Vec3::new(0.3, 0.6, 1.0).normalize();
    tree.push_link(Link::new(
        Some(1),
        JointKind::Slide {
            axis: axis2,
            range: None,
            damping: 0.0,
            armature: 0.04,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::new(-0.1, 0.0, -0.22), Quat::IDENTITY),
        (Vec3::new(0.0, 0.1, 0.15), Quat::IDENTITY),
        0.6,
        box_inertia(0.6, 0.1, 0.15, 0.15),
    ));
    tree.push_link(Link::new(
        Some(2),
        JointKind::Ball {
            damping: 0.0,
            armature: 0.02,
        },
        (Vec3::new(0.05, -0.1, -0.15), Quat::IDENTITY),
        (Vec3::new(-0.1, 0.05, 0.18), Quat::IDENTITY),
        0.4,
        Mat3::diag(0.02, 0.025, 0.03),
    ));
    tree
}

/// Populate `tree.q` and `tree.qdot` with deterministic pseudo-random state
/// respecting the ball / free-root quaternion normalization.
fn scramble_state(tree: &mut Tree, seed: u64) {
    let mut rng = Lcg::new(seed);
    let n = tree.links.len();
    for i in 0..n {
        let link = &tree.links[i];
        let qoff = tree.q_offset[i];
        let voff = tree.v_offset[i];
        match link.joint {
            JointKind::Free => {
                for k in 0..3 {
                    tree.q[qoff + k] = rng.sym(0.5);
                }
                let axis = Vec3::new(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0)).normalize();
                let angle = rng.sym(1.5);
                let q = Quat::from_axis_angle(axis, angle);
                tree.q[qoff + 3] = q.x;
                tree.q[qoff + 4] = q.y;
                tree.q[qoff + 5] = q.z;
                tree.q[qoff + 6] = q.w;
                for k in 0..6 {
                    tree.qdot[voff + k] = rng.sym(0.8);
                }
            }
            JointKind::Fixed => {}
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                tree.q[qoff] = rng.sym(0.7);
                tree.qdot[voff] = rng.sym(1.2);
            }
            JointKind::Ball { .. } => {
                let axis = Vec3::new(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0)).normalize();
                let angle = rng.sym(1.5);
                let q = Quat::from_axis_angle(axis, angle);
                tree.q[qoff] = q.x;
                tree.q[qoff + 1] = q.y;
                tree.q[qoff + 2] = q.z;
                tree.q[qoff + 3] = q.w;
                for k in 0..3 {
                    tree.qdot[voff + k] = rng.sym(1.0);
                }
            }
        }
    }
}

/// Set `qfrc_applied` for every hinge/slide/ball slot to a deterministic
/// pseudo-random torque. Free-root slots are left zero (see the module docs
/// on how the round-trip handles floating bases via `applied_wrenches`
/// instead).
fn scramble_tau_internal(tree: &mut Tree, seed: u64) {
    let mut rng = Lcg::new(seed);
    let n = tree.links.len();
    for i in 0..n {
        let link = &tree.links[i];
        let voff = tree.v_offset[i];
        match link.joint {
            JointKind::Free | JointKind::Fixed => {}
            JointKind::Hinge { .. } | JointKind::Slide { .. } => {
                tree.qfrc_applied[voff] = rng.sym(0.5);
            }
            JointKind::Ball { .. } => {
                for k in 0..3 {
                    tree.qfrc_applied[voff + k] = rng.sym(0.3);
                }
            }
        }
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ---------------------------------------------------------------------------
// Anchor 1: THE round-trip identity — the primary cross-validation between
// ABA and RNE. Fixed root, all four joint kinds, five seeded states.
// ---------------------------------------------------------------------------

#[test]
fn round_trip_aba_rne_reconstructs_tau_fixed_root() {
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    // Five seeds → five distinct scrambled (q, qdot, τ_input) triples. If the
    // identity holds across all five, the algorithms agree along a whole
    // slice of configuration space, not one lucky corner.
    for seed in [
        0x5eed_0001,
        0xd15c_0010,
        0xfeed_0100,
        0xbead_1000,
        0xc0de_1234,
    ] {
        let mut tree = mixed_fixed_root();
        scramble_state(&mut tree, seed);
        scramble_tau_internal(&mut tree, seed ^ 0xa5a5_a5a5);

        let ext = vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()];
        let poses = forward_kinematics(&tree);
        let qddot = aba(&tree, &poses, gravity, &ext);
        let tau_reconstructed = inverse_dynamics(&tree, &qddot, gravity, &ext);

        let max_err = max_abs(&tau_reconstructed, &tree.qfrc_applied);
        assert!(
            max_err < 5e-4,
            "seed {seed:#x}: max |τ_out - τ_in| = {max_err:e} (τ_in {:?}, τ_out {:?})",
            tree.qfrc_applied,
            tau_reconstructed
        );
    }
}

/// Free-root variant. `qfrc_applied` for the 6 root slots is not consumed by
/// ABA (see `Tree::qfrc_applied` docs); the round-trip reconstructs the root
/// wrench required to hold the given qddot, which — since we apply no root
/// wrench in ABA either — must equal zero.
#[test]
fn round_trip_aba_rne_reconstructs_tau_free_root() {
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    for seed in [0x5ee1_0001, 0xd15d_0010, 0xfee1_0100] {
        let mut tree = mixed_free_root();
        scramble_state(&mut tree, seed);
        scramble_tau_internal(&mut tree, seed ^ 0x1234_5678);

        let ext = vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()];
        let poses = forward_kinematics(&tree);
        let qddot = aba(&tree, &poses, gravity, &ext);
        let tau_out = inverse_dynamics(&tree, &qddot, gravity, &ext);

        // First 6 slots: root wrench should reconstruct to ~zero (no root
        // wrench was applied by ABA — free root's qfrc_applied is ignored).
        for (k, &tau_k) in tau_out.iter().take(6).enumerate() {
            assert!(
                tau_k.abs() < 5e-4,
                "seed {seed:#x}: τ_root[{k}] = {tau_k} (expected ~0)"
            );
        }
        // Remaining slots: reconstruct input.
        for (k, (&tau_k, &tau_in_k)) in tau_out
            .iter()
            .zip(tree.qfrc_applied.iter())
            .enumerate()
            .skip(6)
        {
            let err = (tau_k - tau_in_k).abs();
            assert!(
                err < 5e-4,
                "seed {seed:#x}: τ_out[{k}] = {tau_k} vs τ_in = {tau_in_k} (Δ = {err})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Anchor 2: M symmetry + positive-definite.
// ---------------------------------------------------------------------------

#[test]
fn mass_matrix_is_symmetric_and_positive_definite() {
    for scene_kind in ["fixed", "free"] {
        for seed in [0xa11c_0001u64, 0xb225_0010, 0xc339_0100] {
            let mut tree = if scene_kind == "fixed" {
                mixed_fixed_root()
            } else {
                mixed_free_root()
            };
            scramble_state(&mut tree, seed);
            let nv = tree.nv();
            let m = mass_matrix(&tree);

            // Symmetry: |M - M^T| ≤ 1e-6 * max|M|.
            let m_max = m.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let mut max_asym = 0.0f32;
            for r in 0..nv {
                for c in 0..nv {
                    max_asym = max_asym.max((m[r * nv + c] - m[c * nv + r]).abs());
                }
            }
            let asym_tol = 1e-6 * m_max.max(1.0);
            assert!(
                max_asym < asym_tol,
                "{scene_kind} seed {seed:#x}: max |M-Mᵀ| = {max_asym:e} > {asym_tol:e}"
            );

            // Cholesky succeeds ⇒ M is positive-definite.
            let l = cholesky(&m, nv);
            assert!(
                l.is_some(),
                "{scene_kind} seed {seed:#x}: Cholesky failed — M not PD"
            );
            // Belt & braces: 0.5 * qdot^T M qdot > 0 for a random nonzero qdot.
            let mut probe = vec![0.0f32; nv];
            let mut rng = Lcg::new(seed ^ 0xdead_beef);
            for x in &mut probe {
                *x = rng.sym(1.0);
            }
            let ke = 0.5 * quadratic_form(&m, &probe, nv);
            assert!(
                ke > 0.0,
                "{scene_kind} seed {seed:#x}: qdot^T M qdot = {} not positive",
                ke * 2.0
            );
        }
    }
}

fn quadratic_form(m: &[f32], v: &[f32], n: usize) -> f32 {
    let mut s = 0.0;
    for r in 0..n {
        let mut row_dot = 0.0f32;
        for c in 0..n {
            row_dot += m[r * n + c] * v[c];
        }
        s += v[r] * row_dot;
    }
    s
}

// ---------------------------------------------------------------------------
// Anchor 3: M vs ABA consistency — ABA(τ) == solve(M, τ - bias).
// ---------------------------------------------------------------------------

#[test]
fn mass_matrix_solve_matches_aba() {
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    for seed in [0xa5a5_0001u64, 0x5a5a_0010, 0x1234_0100] {
        let mut tree = mixed_fixed_root();
        scramble_state(&mut tree, seed);
        scramble_tau_internal(&mut tree, seed ^ 0xf00d_f00d);

        let ext = vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()];
        let poses = forward_kinematics(&tree);
        let qddot_aba = aba(&tree, &poses, gravity, &ext);

        let m = mass_matrix(&tree);
        let bias = bias_forces(&tree, gravity);
        let nv = tree.nv();
        let mut rhs = vec![0.0f32; nv];
        for k in 0..nv {
            rhs[k] = tree.qfrc_applied[k] - bias[k];
        }
        let l = cholesky(&m, nv).expect("M is positive-definite");
        let qddot_solve = cholesky_solve(&l, nv, &rhs);
        let err = max_abs(&qddot_aba, &qddot_solve);
        assert!(
            err < 5e-4,
            "seed {seed:#x}: max |ABA - M^-1(τ - bias)| = {err:e}"
        );
    }
}

// ---------------------------------------------------------------------------
// Anchor 4: M vs energy — KE from Newton-Euler summation equals
// 0.5 qdot^T M qdot. Independent path to the same scalar.
//
// Use armature = 0 so the two definitions coincide (armature stores in the M
// diagonal but does NOT contribute to the rigid-body KE computed from link
// twists).
// ---------------------------------------------------------------------------

fn mixed_fixed_no_armature() -> Tree {
    let mut tree = mixed_fixed_root();
    for i in 0..tree.links.len() {
        let link = &tree.links[i];
        let new_joint = match link.joint {
            JointKind::Hinge {
                axis,
                range,
                damping,
                armature: _,
                limit,
            } => JointKind::Hinge {
                axis,
                range,
                damping,
                armature: 0.0,
                limit,
            },
            JointKind::Slide {
                axis,
                range,
                damping,
                armature: _,
                limit,
            } => JointKind::Slide {
                axis,
                range,
                damping,
                armature: 0.0,
                limit,
            },
            JointKind::Ball { damping, .. } => JointKind::Ball {
                damping,
                armature: 0.0,
            },
            other => other,
        };
        tree.links[i] = Link::new(
            link.parent,
            new_joint,
            link.joint_offset_in_parent,
            link.joint_offset_in_child,
            link.mass,
            link.inertia_body,
        );
    }
    tree
}

/// Rigid-body kinetic energy of a tree, from per-link world-frame velocities.
/// Independent of CRB (uses only forward kinematics + spatial inertias) so a
/// bug in one algorithm cannot "confirm" the same bug in the other.
fn kinetic_energy_from_twists(tree: &Tree) -> f32 {
    let poses = forward_kinematics(tree);
    let n = tree.links.len();
    // Per-link world-frame angular velocity and COM linear velocity, walked
    // root → leaves so parent state is ready when its children are visited.
    let mut v_world_ang = vec![Vec3::ZERO; n];
    let mut v_world_lin_com = vec![Vec3::ZERO; n];
    for i in 0..n {
        let link = &tree.links[i];
        let (com_i, ori_i) = poses[i];
        match link.joint {
            JointKind::Free => {
                let voff = tree.v_offset[i];
                let w_body = Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                let v_body = Vec3::new(
                    tree.qdot[voff + 3],
                    tree.qdot[voff + 4],
                    tree.qdot[voff + 5],
                );
                v_world_ang[i] = ori_i.rotate(w_body);
                v_world_lin_com[i] = ori_i.rotate(v_body);
            }
            JointKind::Fixed => match link.parent {
                Some(p) => {
                    let (com_p, _) = poses[p];
                    let r = com_i - com_p;
                    v_world_ang[i] = v_world_ang[p];
                    v_world_lin_com[i] = v_world_lin_com[p] + v_world_ang[p].cross(r);
                }
                None => {
                    v_world_ang[i] = Vec3::ZERO;
                    v_world_lin_com[i] = Vec3::ZERO;
                }
            },
            JointKind::Hinge { axis, .. } => {
                let p = link.parent.unwrap();
                let (com_p, ori_p) = poses[p];
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let axis_world = ori_p.rotate(axis);
                let joint_pos_world = com_p + ori_p.rotate(link.joint_offset_in_parent.0);
                let r_from_p = com_i - com_p;
                let r_from_joint = com_i - joint_pos_world;
                v_world_ang[i] = v_world_ang[p] + axis_world * qdot_i;
                v_world_lin_com[i] = v_world_lin_com[p]
                    + v_world_ang[p].cross(r_from_p)
                    + (axis_world * qdot_i).cross(r_from_joint);
            }
            JointKind::Slide { axis, .. } => {
                let p = link.parent.unwrap();
                let (com_p, ori_p) = poses[p];
                let qdot_i = tree.qdot[tree.v_offset[i]];
                let axis_world = ori_p.rotate(axis);
                let r_from_p = com_i - com_p;
                v_world_ang[i] = v_world_ang[p];
                v_world_lin_com[i] =
                    v_world_lin_com[p] + v_world_ang[p].cross(r_from_p) + axis_world * qdot_i;
            }
            JointKind::Ball { .. } => {
                let p = link.parent.unwrap();
                let (com_p, ori_p) = poses[p];
                let voff = tree.v_offset[i];
                let omega_body =
                    Vec3::new(tree.qdot[voff], tree.qdot[voff + 1], tree.qdot[voff + 2]);
                // ω is body-frame of the CHILD link; child ori = poses[i].1.
                let omega_world = ori_i.rotate(omega_body);
                let joint_pos_world = com_p + ori_p.rotate(link.joint_offset_in_parent.0);
                let r_from_p = com_i - com_p;
                let r_from_joint = com_i - joint_pos_world;
                // Body-frame ω contributes velocity at COM as ω × (com - joint).
                v_world_ang[i] = v_world_ang[p] + omega_world;
                v_world_lin_com[i] = v_world_lin_com[p]
                    + v_world_ang[p].cross(r_from_p)
                    + omega_world.cross(r_from_joint);
            }
        }
    }
    let mut ke = 0.0;
    for i in 0..n {
        let link = &tree.links[i];
        let (_, ori_i) = poses[i];
        let ke_lin = 0.5 * link.mass * v_world_lin_com[i].length_squared();
        let ang_body = ori_i.inverse_rotate(v_world_ang[i]);
        let i_omega_body = link.inertia_body * ang_body;
        let ke_ang = 0.5 * ang_body.dot(i_omega_body);
        ke += ke_lin + ke_ang;
    }
    ke
}

#[test]
fn kinetic_energy_matches_half_qdot_m_qdot() {
    for scene_kind in ["fixed", "free"] {
        for seed in [0xe1e1_0001u64, 0xe2e2_0010, 0xe3e3_0100] {
            let mut tree = if scene_kind == "fixed" {
                mixed_fixed_no_armature()
            } else {
                // Free-root variant with armature stripped.
                let mut t = mixed_free_root();
                for i in 0..t.links.len() {
                    let link = &t.links[i];
                    let new_joint = match link.joint {
                        JointKind::Hinge {
                            axis,
                            range,
                            damping,
                            armature: _,
                            limit,
                        } => JointKind::Hinge {
                            axis,
                            range,
                            damping,
                            armature: 0.0,
                            limit,
                        },
                        JointKind::Slide {
                            axis,
                            range,
                            damping,
                            armature: _,
                            limit,
                        } => JointKind::Slide {
                            axis,
                            range,
                            damping,
                            armature: 0.0,
                            limit,
                        },
                        JointKind::Ball { damping, .. } => JointKind::Ball {
                            damping,
                            armature: 0.0,
                        },
                        other => other,
                    };
                    t.links[i] = Link::new(
                        link.parent,
                        new_joint,
                        link.joint_offset_in_parent,
                        link.joint_offset_in_child,
                        link.mass,
                        link.inertia_body,
                    );
                }
                t
            };
            scramble_state(&mut tree, seed);

            let ke_twists = kinetic_energy_from_twists(&tree);
            let m = mass_matrix(&tree);
            let ke_crb = 0.5 * quadratic_form(&m, &tree.qdot, tree.nv());

            let tol = 5e-4 * ke_twists.abs().max(1.0);
            assert!(
                (ke_twists - ke_crb).abs() < tol,
                "{scene_kind} seed {seed:#x}: KE(twists) = {ke_twists}, 0.5 qdotᵀ M qdot = {ke_crb} (Δ = {})",
                (ke_twists - ke_crb).abs()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Anchor 5: single-pendulum closed form. M reduces to I_pivot + armature.
// ---------------------------------------------------------------------------

#[test]
fn single_pendulum_mass_matrix_and_gravity_bias_match_hand_form() {
    // Point-mass approximation: sphere of tiny I_com at distance L below the
    // pivot. Hinge axis = +X, mass = 1, L = 1, armature = 0.05.
    let l = 1.0f32;
    let mass = 1.0f32;
    let armature = 0.05f32;
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let inertia = Mat3::diag(1e-6, 1e-6, 1e-6);
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 0.0,
            armature,
            limit: newt::joint::JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, l), Quat::IDENTITY),
        mass,
        inertia,
    ));

    // At q=0 the COM sits at (0,0,-L). I_pivot (about X, through joint) via
    // parallel-axis: I_com_xx + m·L² ≈ 0 + 1·1 = 1. Add armature → 1.05.
    let m = mass_matrix(&tree);
    assert!(
        (m[0] - (mass * l * l + armature + 1e-6)).abs() < 1e-5,
        "M[0][0] = {} vs expected ~{}",
        m[0],
        mass * l * l + armature + 1e-6
    );

    // Static bias with gravity: q=0 (pointing down), gravity torque about
    // pivot is zero. So bias = 0.
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let bias0 = bias_forces(&tree, gravity);
    assert!(
        bias0[0].abs() < 1e-5,
        "bias at q=0 = {} (expected 0)",
        bias0[0]
    );

    // Move to horizontal: q = π/2. COM at (0, +L, 0). Gravity (0,0,-mg).
    // Gravity torque about +X pivot: (0,L,0)×(0,0,-mg) = (-mgL, 0, 0), i.e.
    // gravity pulls the pendulum back toward hanging-down (decreasing q).
    // bias(q, 0) is the τ_applied needed to KEEP qddot=0 — it must oppose
    // gravity, so bias = +mgL.
    tree.set_hinge_angle(1, std::f32::consts::FRAC_PI_2);
    let bias_h = bias_forces(&tree, gravity);
    let expected = mass * 9.81 * l; // +mgL
    assert!(
        (bias_h[0] - expected).abs() < 5e-3,
        "bias at q=π/2 = {} vs expected {}",
        bias_h[0],
        expected
    );
}

// ---------------------------------------------------------------------------
// Anchor 6: RNE gravity-compensation on a 2-link arm. Static (qdot=qddot=0)
// bias should equal the hand-computed gravitational torque at each hinge.
// ---------------------------------------------------------------------------

#[test]
fn rne_static_two_link_arm_matches_hand_gravity_comp() {
    // Two rods, both hinges about +X, both angles 0 initially → arm hangs
    // straight down along -Z. l1, l2, m1, m2 broken from equality.
    let l1 = 0.8f32;
    let l2 = 0.5f32;
    let m1 = 1.2f32;
    let m2 = 0.7f32;
    let mut tree = Tree::new();
    // Fixed root at origin.
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Link 1: hinge about +X at origin, COM at -L1/2.
    let i1 = m1 * l1 * l1 / 12.0;
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, l1 * 0.5), Quat::IDENTITY),
        m1,
        Mat3::diag(i1, i1, 1e-6),
    ));
    // Link 2: hinge about +X at bottom of link 1, COM at -L2/2 from that
    // hinge.
    let i2 = m2 * l2 * l2 / 12.0;
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.0, 0.0, -l1 * 0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, l2 * 0.5), Quat::IDENTITY),
        m2,
        Mat3::diag(i2, i2, 1e-6),
    ));

    // Rotate both joints so the arm isn't straight — pick angles where the
    // gravity torques are non-trivial. Use q1 = 0.3, q2 = -0.4 (radians).
    tree.set_hinge_angle(1, 0.3);
    tree.set_hinge_angle(2, -0.4);
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let bias = bias_forces(&tree, gravity);

    // Hand-compute gravity-compensation torques.
    // World-frame COM positions:
    //   COM1 = R_x(q1) * (0, 0, -L1/2)
    //   Anchor2 world = R_x(q1) * (0, 0, -L1)
    //   R2 = R_x(q1) * R_x(q2) = R_x(q1+q2)
    //   COM2 = Anchor2 + R2 * (0, 0, -L2/2)
    let s1 = f32::sin(0.3);
    let c1 = f32::cos(0.3);
    let s12 = f32::sin(0.3 + -0.4);
    let c12 = f32::cos(0.3 + -0.4);
    // Rotation about +X: (0,y,z) → (0, cy·y − sy·z, sy·y + cy·z).
    // COM1 at (0, 0, -L1/2) rotated by R_x(q1) → (0, s1*L1/2, -c1*L1/2)
    let com1 = Vec3::new(0.0, s1 * l1 * 0.5, -c1 * l1 * 0.5);
    let anchor2 = Vec3::new(0.0, s1 * l1, -c1 * l1);
    let com2 = anchor2 + Vec3::new(0.0, s12 * l2 * 0.5, -c12 * l2 * 0.5);

    // Gravity torque on joint 1 (about +X at world origin):
    //   τ_grav_1 · X = (Σ_i m_i · COM_i × g_world) · X
    // Gravity torque on joint 2 (about +X at anchor2):
    //   τ_grav_2 · X = (m_2 · (COM_2 − anchor2) × g_world) · X
    // bias(q, 0) is the τ_applied needed to hold qddot=0, i.e. the OPPOSITE
    // of the gravity torque (holding the arm still requires countering
    // gravity's pull).
    let g_world = gravity;
    let torque_grav_1 = com1.cross(g_world * m1) + com2.cross(g_world * m2);
    let torque_grav_2 = (com2 - anchor2).cross(g_world * m2);
    let tau1_expected = -torque_grav_1.x;
    let tau2_expected = -torque_grav_2.x;
    assert!(
        (bias[0] - tau1_expected).abs() < 5e-3,
        "bias[0] = {} vs expected {} (Δ = {})",
        bias[0],
        tau1_expected,
        (bias[0] - tau1_expected).abs()
    );
    assert!(
        (bias[1] - tau2_expected).abs() < 5e-3,
        "bias[1] = {} vs expected {} (Δ = {})",
        bias[1],
        tau2_expected,
        (bias[1] - tau2_expected).abs()
    );
}
