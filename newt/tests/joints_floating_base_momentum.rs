//! Floating-base momentum conservation.
//!
//! A free-root body with a swinging hinge link, no gravity, no external
//! wrenches. All forces are internal (through the hinge, equal-and-opposite
//! at the same point) so total linear momentum `P = Σ m_i v_i` and total
//! angular momentum about the world origin `L = Σ (r_i × m_i v_i + I_i^world
//! ω_i)` must be conserved.
//!
//! This test is the primary catch for wrong tree-pass math that grounded-
//! pendulum tests miss: with a fixed root, the world absorbs any reaction
//! wrench silently, so wrong `X_up` transforms or wrong pull-back-of-force
//! math can look correct at the joint level while quietly dumping momentum
//! into the "wall". A free root cannot hide those bugs.
//!
//! Symmetry break: non-diagonal-ish inertia (asymmetric box half-extents),
//! initial box orientation NOT identity, hinge axis not aligned with a
//! principal axis of the child link.

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::spatial::SpatialMotion;
use newt::tree::{Link, Tree, forward_kinematics, rk4_step};

fn build_floating_arm() -> Tree {
    let mut tree = Tree::new();
    // Root: free-floating box. Nontrivial initial orientation to catch
    // frame errors.
    let hx = 0.3;
    let hy = 0.2;
    let hz = 0.15;
    let m0 = 2.0f32;
    let box_i = {
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        let ixx = (m0 / 3.0) * (hy2 + hz2);
        let iyy = (m0 / 3.0) * (hx2 + hz2);
        let izz = (m0 / 3.0) * (hx2 + hy2);
        Mat3::diag(ixx, iyy, izz)
    };
    let q0 = Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.3), 0.6);
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(0.0, 0.0, 5.0), q0),
        (Vec3::ZERO, Quat::IDENTITY),
        m0,
        box_i,
    ));
    // Rod link attached to the bottom of the box. Hinge axis is deliberately
    // non-axis-aligned to break symmetry.
    let l = 0.8f32;
    let m1 = 0.7f32;
    let i_perp = (1.0 / 12.0) * m1 * l * l;
    let axis = Vec3::new(1.0, 0.3, 0.0).normalize();
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(axis),
        (Vec3::new(0.0, 0.0, -hz), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
        m1,
        Mat3::diag(i_perp, i_perp, 1e-4),
    ));
    // Give the hinge a nontrivial rate. The root velocity stays zero
    // initially — this way total P starts non-zero (rod is swinging) and
    // the box must start moving to conserve momentum.
    tree.set_hinge_rate(1, 4.0);
    tree
}

/// Total linear and angular momentum about the world origin.
fn total_momentum(tree: &Tree) -> (Vec3, Vec3) {
    let poses = forward_kinematics(tree);
    let n = tree.links.len();
    // Rebuild world-frame COM velocities and body-frame angular velocities
    // from the tree's spatial-velocity walk. Same layout as the chain-energy
    // test but the free root now contributes v0.
    let mut v_lin_world = vec![Vec3::ZERO; n];
    let mut w_body = vec![Vec3::ZERO; n];
    let mut w_world = vec![Vec3::ZERO; n];
    // Free root: qdot slots (0..3) = ω_body, (3..6) = v_body.
    match tree.links[0].joint {
        JointKind::Free => {
            let wb = Vec3::new(tree.qdot[0], tree.qdot[1], tree.qdot[2]);
            let vb = Vec3::new(tree.qdot[3], tree.qdot[4], tree.qdot[5]);
            let (_pos, ori) = poses[0];
            w_body[0] = wb;
            w_world[0] = ori.rotate(wb);
            v_lin_world[0] = ori.rotate(vb);
        }
        JointKind::Fixed => {}
        JointKind::Hinge { .. } => unreachable!(),
    }
    for i in 1..n {
        let link = &tree.links[i];
        let parent = link.parent.unwrap();
        let (child_pos, child_ori) = poses[i];
        let (parent_pos, parent_ori) = poses[parent];
        match link.joint {
            JointKind::Hinge { axis, .. } => {
                let axis_world = parent_ori.rotate(axis);
                let qdot_i = tree.hinge_rate(i);
                let w_i_world = w_world[parent] + axis_world * qdot_i;
                w_world[i] = w_i_world;
                w_body[i] = child_ori.inverse_rotate(w_i_world);
                let joint_world = parent_pos + parent_ori.rotate(link.joint_offset_in_parent.0);
                let v_parent_at_child_com =
                    v_lin_world[parent] + w_world[parent].cross(child_pos - parent_pos);
                v_lin_world[i] =
                    v_parent_at_child_com + (axis_world * qdot_i).cross(child_pos - joint_world);
            }
            JointKind::Fixed => {
                w_world[i] = w_world[parent];
                w_body[i] = child_ori.inverse_rotate(w_world[i]);
                v_lin_world[i] =
                    v_lin_world[parent] + w_world[parent].cross(child_pos - parent_pos);
            }
            JointKind::Free => unreachable!(),
        }
    }
    let mut p_total = Vec3::ZERO;
    let mut l_total = Vec3::ZERO;
    for i in 0..n {
        let link = &tree.links[i];
        let (com, ori) = poses[i];
        let p_i = v_lin_world[i] * link.mass;
        p_total += p_i;
        // Angular momentum about world origin = r × p + I_world ω_world.
        // I_world = R I_body R^T; applied to ω_world: R (I_body (R^T ω_world))
        //         = R (I_body ω_body).
        let l_orbital = com.cross(p_i);
        let iw_body = link.inertia_body * w_body[i];
        let l_spin_world = ori.rotate(iw_body);
        l_total += l_orbital + l_spin_world;
    }
    (p_total, l_total)
}

#[test]
fn floating_base_conserves_linear_and_angular_momentum() {
    let mut tree = build_floating_arm();
    let (p0, l0) = total_momentum(&tree);
    // Sanity: the initial momentum must be NONZERO on both counts —
    // otherwise a bug that constant-returns zero would slip through.
    assert!(
        p0.length() > 1e-3,
        "initial linear momentum near zero — test not exercising ABA"
    );
    assert!(
        l0.length() > 1e-3,
        "initial angular momentum near zero — test not exercising ABA"
    );

    let dt = 0.001f32;
    let steps = 2_000usize;
    let mut max_p_rel: f32 = 0.0;
    let mut max_l_rel: f32 = 0.0;
    for _ in 0..steps {
        rk4_step(&mut tree, Vec3::ZERO, dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 2]
        });
        let (p, l) = total_momentum(&tree);
        let pr = (p - p0).length() / p0.length();
        let lr = (l - l0).length() / l0.length();
        if pr > max_p_rel {
            max_p_rel = pr;
        }
        if lr > max_l_rel {
            max_l_rel = lr;
        }
    }
    assert!(
        max_p_rel < 5.0e-4,
        "linear momentum drift {max_p_rel} exceeded 5e-4 over 2s"
    );
    assert!(
        max_l_rel < 5.0e-3,
        "angular momentum drift {max_l_rel} exceeded 5e-3 over 2s"
    );
}

#[test]
fn free_root_alone_gravity_preserves_body_frame_free_fall() {
    // A single free-root link with gravity. Body-frame linear acceleration
    // should equal ori^-1 * gravity_world exactly. This exercises the free-
    // root 6x6 solve path without any child links.
    let mut tree = Tree::new();
    let q0 = Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3);
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::ZERO, q0),
        (Vec3::ZERO, Quat::IDENTITY),
        1.5,
        Mat3::diag(1.0, 2.0, 3.0),
    ));
    let poses = forward_kinematics(&tree);
    let g_world = Vec3::new(0.0, 0.0, -9.81);
    let expected_body = q0.inverse_rotate(g_world);
    let qddot = newt::tree::aba(&tree, &poses, g_world, &vec![(Vec3::ZERO, Vec3::ZERO); 1]);
    let ang_accel_body = Vec3::new(qddot[0], qddot[1], qddot[2]);
    let lin_accel_body = Vec3::new(qddot[3], qddot[4], qddot[5]);
    assert!(
        ang_accel_body.length() < 1e-5,
        "free-root angular accel under pure gravity should be zero; got {ang_accel_body:?}"
    );
    let diff = lin_accel_body - expected_body;
    assert!(
        diff.length() < 1e-4,
        "free-root linear accel {lin_accel_body:?} != expected {expected_body:?}"
    );
    // Consume the workspace symbol so unused-import warnings don't fire.
    let _ = SpatialMotion::ZERO;
}
