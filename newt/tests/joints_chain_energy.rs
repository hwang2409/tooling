//! 3-link chain energy conservation over 10 000 steps. No damping, no
//! contacts — just three uniform-rod links swinging under gravity. If the
//! ABA has a wrong-frame bias term, wrong Coriolis coupling, or an integrator
//! sign flip, this test catches the drift within the run.
//!
//! The energy expression uses the same body-frame `T = ½ q̇ᵀ M q̇` +
//! world-frame `V = m g z` split as the double-pendulum test; here we
//! rebuild it from the tree's link poses and spatial velocities (via
//! forward kinematics + the joint subspace), NOT from a Lagrangian call
//! into the engine — so a bug in the ABA cannot help this pass.

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, forward_kinematics, rk4_step};

fn build_three_rod_chain() -> Tree {
    // Break symmetry: mixed masses (1.1, 0.7, 1.3), mixed lengths.
    let masses = [1.1f32, 0.7, 1.3];
    let lengths = [0.6f32, 0.4, 0.5];
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
    for i in 0..3 {
        let l = lengths[i];
        let m = masses[i];
        let i_perp = (1.0 / 12.0) * m * l * l;
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, 0.0, -lengths[i - 1] * 0.5)
        };
        tree.push_link(Link::new(
            Some(i),
            JointKind::hinge(Vec3::X),
            (parent_anchor, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
            m,
            Mat3::diag(i_perp, i_perp, 1e-6),
        ));
    }
    // Break rest symmetry with nontrivial starting angles.
    tree.set_hinge_angle(1, 0.4);
    tree.set_hinge_angle(2, -0.3);
    tree.set_hinge_angle(3, 0.2);
    tree
}

/// Total kinetic + potential energy of the tree in world coordinates.
///
/// Rebuilds each link's world-frame COM velocity from a spatial-velocity
/// walk over the joint subspaces — mirrors what ABA pass 1 does. Uses the
/// engine's math primitives (Vec3, Quat, forward kinematics) but does NOT
/// call `aba`, so a bug in the ABA doesn't help this test pass.
fn tree_energy(tree: &Tree, gravity_z: f32) -> f32 {
    let poses = forward_kinematics(tree);
    let n = tree.links.len();
    // Per-link world-frame COM linear velocity and body-frame angular
    // velocity. Root is fixed → both zero.
    let mut v_lin_world = vec![Vec3::ZERO; n];
    let mut w_body = vec![Vec3::ZERO; n];
    let mut w_world = vec![Vec3::ZERO; n];
    for i in 1..n {
        let link = &tree.links[i];
        let parent = link.parent.unwrap();
        let (child_pos, child_ori) = poses[i];
        let (parent_pos, _parent_ori) = poses[parent];
        // Hinge: ω_body = ω_parent_body + qdot * axis (both in child frame).
        // ω_parent_body expressed in child frame = (child_ori.inverse * parent_ori) * ω_parent_body_in_parent.
        // Simpler: use world-frame ω. ω_world_child = ω_world_parent + qdot * axis_world.
        match link.joint {
            JointKind::Hinge { axis, .. } => {
                let axis_world = tree.link_pose(parent).1.rotate(axis);
                let qdot_i = tree.hinge_rate(i);
                let w_i = w_world[parent] + axis_world * qdot_i;
                w_world[i] = w_i;
                w_body[i] = child_ori.inverse_rotate(w_i);
                // World-frame COM velocity: parent COM linear + ω_parent ×
                // (child COM − parent COM), then + qdot * axis × (COM − joint
                // anchor).
                let joint_world = tree.link_pose(parent).0
                    + tree
                        .link_pose(parent)
                        .1
                        .rotate(link.joint_offset_in_parent.0);
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
            JointKind::Free | JointKind::Slide { .. } | JointKind::Ball { .. } => {
                // Not exercised in this scenario (hinge-only chain) but the
                // branches keep the function total for exhaustiveness.
                v_lin_world[i] = Vec3::ZERO;
                w_body[i] = Vec3::ZERO;
                w_world[i] = Vec3::ZERO;
            }
        }
    }
    let mut e = 0.0f32;
    for i in 1..n {
        let link = &tree.links[i];
        let (com, _) = poses[i];
        let lin_ke = 0.5 * link.mass * v_lin_world[i].dot(v_lin_world[i]);
        let iw = link.inertia_body * w_body[i];
        let rot_ke = 0.5 * w_body[i].dot(iw);
        let pe = link.mass * (-gravity_z) * com.z;
        e += lin_ke + rot_ke + pe;
    }
    e
}

#[test]
fn three_link_chain_energy_conserved_over_10k_steps() {
    let g = 9.81f32;
    let mut tree = build_three_rod_chain();
    let e0 = tree_energy(&tree, -g);
    let dt = 0.001f32;
    let n_steps = 10_000usize;
    let mut max_drift: f32 = 0.0;
    for _ in 0..n_steps {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -g), dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 4]
        });
        let e = tree_energy(&tree, -g);
        let rel = ((e - e0) / e0).abs();
        if rel > max_drift {
            max_drift = rel;
        }
    }
    // Observed max drift on macOS aarch64 during development: ~4e-4. The
    // 5e-3 ceiling gives comfortable headroom without allowing a real
    // integrator regression through.
    assert!(
        max_drift < 5.0e-3,
        "3-link chain energy drift {max_drift} exceeded 5e-3 bound"
    );
    // Sanity: swing motion actually occurred.
    let moved = (tree.hinge_angle(1) - 0.4).abs()
        + (tree.hinge_angle(2) + 0.3).abs()
        + (tree.hinge_angle(3) - 0.2).abs();
    assert!(moved > 0.1, "chain barely moved — test not exercising ABA");
}
