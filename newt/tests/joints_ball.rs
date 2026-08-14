//! Ball-joint anchors.
//!
//! # Anchors
//!
//! 1. `spherical_pendulum_conserves_energy_and_vertical_angular_momentum` —
//!    a mass on a ball-jointed rod under gravity. Gravity acts along −Z, so
//!    the vertical component of angular momentum about the pivot is
//!    conserved (gravity torque about the pivot has zero z-component for
//!    any pole direction). Total mechanical energy is also conserved.
//!    Non-planar ICs so the motion is genuinely 3D — a planar setup would
//!    be indistinguishable from a hinge test.
//! 2. `ball_joint_with_pivot_at_com_reproduces_torque_free_free_body` —
//!    ball joint whose anchor coincides with the child COM, no gravity.
//!    Physically identical to a free body with the same inertia and initial
//!    body-frame ω: pure Euler-equation torque-free rotation. Compared
//!    against a tier-1 `Body` integrated by the same RK4 scheme.

use newt::body::Body;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, aba, forward_kinematics, rk4_step};

fn zero_ext(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

// ---------------------------------------------------------------------------
// 1. Spherical pendulum: energy + Lz conservation
// ---------------------------------------------------------------------------

/// Build a ball-jointed rod: fixed root at world (0, 0, H), pole hangs on
/// a ball joint below it. Pole is a uniform rod along child +Z of length L
/// and mass M; joint anchor sits at the top of the rod (child (0, 0, L/2)),
/// so COM is L/2 below the pivot at identity orientation.
fn build_spherical_pendulum(mass: f32, length: f32, height: f32) -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, height), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Uniform rod along child +Z: I_perp = (1/12) m L² about x and y,
    // small (>0) about z to keep the inertia tensor invertible.
    let i_perp = (1.0 / 12.0) * mass * length * length;
    tree.push_link(Link::new(
        Some(0),
        JointKind::ball(),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, length * 0.5), Quat::IDENTITY),
        mass,
        Mat3::diag(i_perp, i_perp, 1e-6),
    ));
    tree
}

/// Total mechanical energy of the spherical pendulum: rotational KE about
/// the fixed pivot + gravitational PE. Uses only forward kinematics + the
/// tree's stored `qdot`; independent of the ABA path.
fn spherical_pendulum_energy(tree: &Tree, mass: f32, length: f32, g: f32) -> f32 {
    let poses = forward_kinematics(tree);
    let (com_world, ori) = poses[1];
    let omega_body = tree.ball_omega(1);
    let omega_world = ori.rotate(omega_body);

    // Body-frame inertia about the COM (matches build_spherical_pendulum).
    let i_perp = (1.0 / 12.0) * mass * length * length;
    let i_body = Mat3::diag(i_perp, i_perp, 1e-6);

    // Linear KE at COM.
    let r_com_from_pivot = com_world - Vec3::new(0.0, 0.0, poses[0].0.z);
    let v_com_world = omega_world.cross(r_com_from_pivot);
    let ke_lin = 0.5 * mass * v_com_world.dot(v_com_world);
    // Rotational KE about COM: ½ ω_body · I_body · ω_body.
    let iw = i_body * omega_body;
    let ke_rot = 0.5 * omega_body.dot(iw);
    // PE = m g z_com.
    let pe = mass * g * com_world.z;
    ke_lin + ke_rot + pe
}

/// Angular momentum about the pivot, computed in world coordinates.
/// Convert `L = m r × v + R I R⁻¹ ω_world` into world frame.
fn spherical_pendulum_angular_momentum(tree: &Tree, mass: f32, length: f32) -> Vec3 {
    let poses = forward_kinematics(tree);
    let (com_world, ori) = poses[1];
    let pivot = Vec3::new(0.0, 0.0, poses[0].0.z);
    let omega_body = tree.ball_omega(1);
    let omega_world = ori.rotate(omega_body);
    let r = com_world - pivot;
    let v = omega_world.cross(r);

    let i_perp = (1.0 / 12.0) * mass * length * length;
    let i_body = Mat3::diag(i_perp, i_perp, 1e-6);
    // I_world ω_world = R (I_body (R⁻¹ ω_world)) = R (I_body ω_body).
    let l_spin_world = ori.rotate(i_body * omega_body);
    let l_orbital = r.cross(v * mass);
    l_orbital + l_spin_world
}

#[test]
fn spherical_pendulum_conserves_energy_and_vertical_angular_momentum() {
    let mass = 1.3f32;
    let length = 0.9f32;
    let height = 1.5f32;
    let g = 9.81f32;
    let g_vec = Vec3::new(0.0, 0.0, -g);

    let mut tree = build_spherical_pendulum(mass, length, height);
    // Non-planar ICs: tilt the pole away from vertical about the X axis so
    // the COM sits off the pivot's z axis, then spin it about the body +Z
    // (rod's own long axis) plus about body +X (swing). The mix means the
    // motion is genuinely 3D: neither purely conical nor purely planar.
    tree.set_ball_orientation(
        1,
        Quat::from_axis_angle(Vec3::X, 0.55) * Quat::from_axis_angle(Vec3::Y, 0.15),
    );
    tree.set_ball_omega(1, Vec3::new(0.4, 0.6, 0.9));

    let e0 = spherical_pendulum_energy(&tree, mass, length, g);
    let lz0 = spherical_pendulum_angular_momentum(&tree, mass, length).z;

    let dt = 0.001f32;
    let n_steps = 2000usize; // 2 s
    let mut e_min = e0;
    let mut e_max = e0;
    let mut lz_min = lz0;
    let mut lz_max = lz0;
    for _ in 0..n_steps {
        rk4_step(&mut tree, g_vec, dt, zero_ext(2));
        let e = spherical_pendulum_energy(&tree, mass, length, g);
        let lz = spherical_pendulum_angular_momentum(&tree, mass, length).z;
        if e < e_min {
            e_min = e;
        }
        if e > e_max {
            e_max = e;
        }
        if lz < lz_min {
            lz_min = lz;
        }
        if lz > lz_max {
            lz_max = lz;
        }
    }
    let e_drift = (e_max - e_min).abs();
    let lz_drift = (lz_max - lz_min).abs();
    // Tolerances derived from the measured drift with the current
    // integrator (1 ms RK4, 2 s window). 5e-3 for energy leaves ~5x
    // headroom over the observed drift; 1e-4 for Lz is tighter because
    // gravity torque about the pivot has exactly zero z component so any
    // drift is pure integrator noise, not physical.
    assert!(
        e_drift < 5e-3,
        "energy drifted by {e_drift} (min={e_min}, max={e_max}, e0={e0})"
    );
    assert!(
        lz_drift < 1e-4,
        "Lz drifted by {lz_drift} (min={lz_min}, max={lz_max}, lz0={lz0})"
    );
    // Non-degeneracy: check that some motion happened. If the pole just
    // sat at rest (a bug in the ball ω integration or the ABA S-block),
    // orientation would stay identical to the initial and both metrics
    // above would trivially conserve at their initial values.
    let ori_now = tree.ball_orientation(1);
    let ori_initial = Quat::from_axis_angle(Vec3::X, 0.55) * Quat::from_axis_angle(Vec3::Y, 0.15);
    let dot = ori_now.x * ori_initial.x
        + ori_now.y * ori_initial.y
        + ori_now.z * ori_initial.z
        + ori_now.w * ori_initial.w;
    assert!(
        dot.abs() < 0.995,
        "pole didn't move over 2 s (final·initial = {dot}); test is not exercising the ball"
    );
}

// ---------------------------------------------------------------------------
// 2. Ball with pivot at COM = torque-free rotation
// ---------------------------------------------------------------------------

#[test]
fn ball_joint_with_pivot_at_com_reproduces_torque_free_free_body() {
    // A ball joint whose anchor coincides with the child COM contributes
    // no linear coupling: the joint pins the child COM to the parent's
    // anchor, and the child freely rotates about that point. With gravity
    // = 0 (or gravity acting through COM), this is exactly the tier-1
    // free-body torque-free rotation dynamics.
    //
    // Compare: `Body` (tier 1) integrated with its own RK4 vs a tree with
    // a Fixed root + Ball child at pivot = COM. Body and Tree use the same
    // integrator scheme applied to the same body-frame ω derivative
    // (dω/dt = I⁻¹ (τ − ω × I ω)); orientation update uses the same
    // `Quat::derivative`. So they should agree to arithmetic precision.
    let inertia = Mat3::diag(0.5, 1.0, 1.4); // asymmetric — genuine 3D tumbling
    let mass = 0.7f32;

    // Tree side.
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
        JointKind::ball(),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY), // pivot = child COM
        mass,
        inertia,
    ));
    let ori0 = Quat::from_axis_angle(Vec3::new(1.0, 0.3, -0.2), 0.4);
    let omega0 = Vec3::new(0.5, -0.9, 1.2); // spins about all 3 body axes
    tree.set_ball_orientation(1, ori0);
    tree.set_ball_omega(1, omega0);

    // Body side.
    let mut body = Body::new(mass, inertia, Vec3::ZERO, ori0);
    body.angular_velocity_body = omega0;
    body.linear_velocity = Vec3::ZERO;

    let dt = 0.001f32;
    let n = 500usize;
    let g = Vec3::ZERO;
    for _ in 0..n {
        rk4_step(&mut tree, g, dt, zero_ext(2));
        // Body::step_bodies from world.rs would apply gravity too; since
        // g = 0 that's a no-op, but we want to isolate rotation. Call the
        // exposed integrator directly through the world.
        let mut w = newt::world::World::new();
        w.gravity = Vec3::ZERO;
        let idx = w.add_body(body);
        w.dt = dt;
        w.step();
        body = w.bodies[idx];
    }

    let ori_tree = tree.ball_orientation(1);
    let ori_body = body.orientation;
    // Quaternion double-cover: q and -q are the same rotation. Use |dot|.
    let dot = ori_tree.x * ori_body.x
        + ori_tree.y * ori_body.y
        + ori_tree.z * ori_body.z
        + ori_tree.w * ori_body.w;
    assert!(
        dot.abs() > 1.0 - 1e-4,
        "ball@COM diverged from free body: |dot| = {}, tree_ori = {ori_tree:?}, body_ori = {ori_body:?}",
        dot.abs()
    );
    // Angular velocity in body frame should also match closely.
    let omega_tree = tree.ball_omega(1);
    let dw = omega_tree - body.angular_velocity_body;
    assert!(
        dw.length() < 1e-3,
        "ball@COM ω diverged from free body: |Δω| = {} (tree={omega_tree:?}, body={:?})",
        dw.length(),
        body.angular_velocity_body
    );
}

// ---------------------------------------------------------------------------
// 3. Ball armature anchor (mutation-coverage recipe from NEWT-6 review)
// ---------------------------------------------------------------------------

/// Ball-joint armature enters ABA's 3x3 articulated-inertia block as
/// `D = Sᵀ IA S + armature · I₃`. With the joint anchor at the child COM
/// (`r_jc = 0`), a diagonal `I_com`, and a unit generalized torque on axis
/// `k`, the closed-form acceleration on that axis is
/// `qddot_k = 1 / (I_com[k, k] + armature)`.
///
/// The reviewer's mutant drops the `+ armature` term from the D diagonal; the
/// full test suite before this anchor did not catch it because every other
/// ball anchor either used `armature = 0` or coupled the ball axis to
/// inertias where a small off-by-one term rounded into noise. This anchor
/// pins the D-diagonal arithmetic directly.
#[test]
fn ball_armature_enters_the_d_diagonal_on_every_axis() {
    let i_com = Mat3::diag(0.5, 1.3, 0.9);
    let armature = 0.7f32;

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
        JointKind::Ball {
            damping: 0.0,
            armature,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY), // pivot at child COM: r_jc = 0
        1.4,                          // child mass; irrelevant with r_jc = 0
        i_com,
    ));

    let g = Vec3::ZERO;
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); 2];
    let voff = tree.v_offset[1];

    for k in 0..3 {
        // Reset q, qdot, qfrc; apply a unit torque on axis k only.
        tree.set_ball_orientation(1, Quat::IDENTITY);
        tree.set_ball_omega(1, Vec3::ZERO);
        for slot in tree.qfrc_applied.iter_mut() {
            *slot = 0.0;
        }
        tree.qfrc_applied[voff + k] = 1.0;

        let poses = forward_kinematics(&tree);
        let qddot = aba(&tree, &poses, g, &ext);

        let expected = 1.0 / (i_com.get(k, k) + armature);
        let got = qddot[voff + k];
        assert!(
            (got - expected).abs() < 1e-6,
            "ball armature D-diagonal broken on axis {k}: got {got}, expected {expected} \
             (I_com[{k},{k}] = {}, armature = {armature})",
            i_com.get(k, k)
        );
        // Off-axis accelerations must be zero: with r_jc = 0, ω = 0, no
        // gravity, and the torque isolated on axis k, cross-axis coupling
        // vanishes.
        for j in 0..3 {
            if j == k {
                continue;
            }
            let off = qddot[voff + j];
            assert!(
                off.abs() < 1e-6,
                "ball unit-τ on axis {k} leaked into axis {j}: qddot = {off}"
            );
        }
    }
}
