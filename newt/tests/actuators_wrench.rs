//! External-wrench anchors for tier-4 actuation.
//!
//! Verifies that `Tree::set_link_wrench` flows into ABA's `f_ext_body` path
//! for articulated links (not just free-root bodies) by checking a
//! hand-derived static equilibrium. A wrench mis-plumbed to only the
//! free-root branch would leave articulated hinges unaffected and blow
//! this anchor.
//!
//! Setup: 2-link chain, both hinges about `+x`, both length `L`. Gravity
//! is disabled to isolate the wrench effect. Both hinges carry a PD servo
//! with target = 0 and moderate `kd` for convergence. A constant world-y
//! force is applied at the tip link's COM; the analytic steady-state
//! angles come from a torque-balance argument computed in the test.

use newt::actuator::Actuator;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, aba, forward_kinematics, rk4_step};

fn build_two_link_chain(length: f32, mass: f32) -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Link 1: hinge_x at world origin, COM at (0, 0, -L/2) when q=0. Uniform
    // rod inertia about the perpendicular axes: I = m·L²/12.
    let i_perp = (1.0 / 12.0) * mass * length * length;
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, length * 0.5), Quat::IDENTITY),
        mass,
        Mat3::diag(i_perp, i_perp, 1e-6),
    ));
    // Link 2: hinge_x at the bottom of link 1 (parent-frame anchor at
    // (0, 0, -L/2) relative to link-1 COM). COM at (0, 0, -L/2) below its
    // own anchor.
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.0, 0.0, -length * 0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, length * 0.5), Quat::IDENTITY),
        mass,
        Mat3::diag(i_perp, i_perp, 1e-6),
    ));
    tree
}

fn zero_wrenches(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

#[test]
fn constant_wrench_on_tip_link_produces_hand_derived_static_deflection() {
    // Setup — see module docs.
    //
    // Analytic static balance (linearized for small angles):
    //   • Servo 2 (elbow) sees external torque about the elbow axis:
    //       τ_ext(q_2)  = (r_com_2_from_elbow) × F  → about +x = L/2 · F
    //     Balance:  L/2 · F = kp · q2_ss   →   q2_ss = L·F / (2·kp)
    //   • Consider the WHOLE chain about hinge 1: the elbow's internal
    //     servo torque cancels between link 1 and link 2, so only external
    //     force and hinge-1 servo remain:
    //       τ_ext_about_hinge1 = (0, 0, -L₁ - L₂/2) × (0, F, 0) → +x =
    //         (L₁ + L₂/2)·F
    //     Balance:  (L₁ + L₂/2)·F = kp · q1_ss
    //       → q1_ss = (L₁ + L₂/2)·F / kp = 3L/2 · F / kp     (L₁=L₂=L)
    let l = 1.0f32;
    let m = 1.0f32;
    let kp = 100.0f32;
    let f_y = 5.0f32;

    let mut tree = build_two_link_chain(l, m);
    // ζ = 1 for fast, non-oscillating convergence. I_ref is a rough estimate
    // (I = m·L² for the tip mass, m·L²/3 for the pivoting rod at the base
    // — anything in the ballpark gives good damping).
    let servo1 = Actuator::position_from_dampratio(1, kp, 1.0, m * l * l, 0.0);
    let servo2 = Actuator::position_from_dampratio(2, kp, 1.0, m * l * l, 0.0);
    tree.add_actuator(servo1);
    tree.add_actuator(servo2);

    // Apply constant world-y force at tip link's (link 2) COM. Zero torque.
    tree.set_link_wrench(2, Vec3::new(0.0, f_y, 0.0), Vec3::ZERO);

    let dt = 0.001f32;
    // Long-enough settle: 5 s at dt=1ms (5000 steps). The servo damping
    // time constant is 1/ωn ≈ 0.1 s so this is 50 τ.
    for _ in 0..5000 {
        rk4_step(&mut tree, Vec3::ZERO, dt, zero_wrenches(3));
    }
    let q1 = tree.hinge_angle(1);
    let q2 = tree.hinge_angle(2);
    let q1_expected = (l + l * 0.5) * f_y / kp; // 3L/2 · F / kp = 1.5 · 5 / 100 = 0.075
    let q2_expected = l * f_y / (2.0 * kp); //          L/2 · F / kp = 0.5 · 5 / 100 = 0.025
    assert!(
        (q1 - q1_expected).abs() < 5e-4,
        "q1 static balance: got {q1}, expected {q1_expected}"
    );
    assert!(
        (q2 - q2_expected).abs() < 5e-4,
        "q2 static balance: got {q2}, expected {q2_expected}"
    );
}

#[test]
fn hinge_z_at_prerotated_pose_catches_world_vs_body_frame_mutant() {
    // Semantic discriminator for the world→body wrench transform. The
    // world-vs-body-frame mutant swaps `ori.inverse_rotate` with
    // `ori.rotate`, which flips the sign of any rotation-about-z applied to
    // a horizontal wrench. The existing hinge_x + world_y test cannot see
    // that mutant: for a rotation about x, the y-component of a
    // (0, F, 0) wrench is `F·cos(q)` — an EVEN function of q — so the
    // hinge-x contribution `L·F_body.y` is identical for correct and
    // mutant. The recipe below breaks that symmetry:
    //
    //   • hinge_z (child rotates about z)
    //   • COM offset in child body frame: (0, L, 0)  → r_jc = (0, -L, 0)
    //   • world force at COM: (0, F, 0)   →  wrench sits in the plane
    //     perpendicular to the hinge axis
    //   • pre-rotate the hinge by q0 = π/4
    //
    // Correct body-frame force:   F_body = Rot_z(-q0)·(0, F, 0)
    //                                    = ( F·sin q0,  F·cos q0, 0)
    // Mutant body-frame force:    F_body = Rot_z(+q0)·(0, F, 0)
    //                                    = (-F·sin q0,  F·cos q0, 0)
    //
    // Joint subspace for hinge_z with r_jc = (0, -L, 0):
    //   S = (axis_z, r_jc × axis_z) = ((0,0,1), (-L, 0, 0))
    // The wrench contribution to `Sᵀ·f_ext` is `-L · F_body.x`, so:
    //   correct:  contribution = -L · F·sin q0
    //   mutant:   contribution = +L · F·sin q0
    //
    // These land on opposite-signed joint accelerations. The test asserts
    // the correct sign and a value near the hand-derived α.
    let l = 1.0f32;
    let m = 1.0f32;
    let f_y = 4.0f32;
    let q0 = std::f32::consts::FRAC_PI_4;

    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Uniform-rod-ish inertia; the z-axis moment about COM is small but
    // non-zero so `M_zz = I_zz_com + m·L²` is well-defined. Values don't
    // change the sign of α — only its magnitude.
    let i_perp = 0.05f32;
    let i_zz = 0.01f32;
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::Z),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, -l, 0.0), Quat::IDENTITY),
        m,
        Mat3::diag(i_perp, i_perp, i_zz),
    ));
    tree.set_hinge_angle(1, q0);
    tree.set_link_wrench(1, Vec3::new(0.0, f_y, 0.0), Vec3::ZERO);

    let poses = forward_kinematics(&tree);
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()];
    let qddot = aba(&tree, &poses, Vec3::ZERO, &ext);
    let alpha = qddot[0];

    // Hand-derived expected α = -L·F·sin(q0) / (I_zz_com + m·L²).
    // sin(π/4) = √2/2. Written as `SQRT_2 * 0.5` so the constant path is
    // recognisably the exact one and clippy's approx_constant check is
    // happy.
    let sin_q0 = std::f32::consts::SQRT_2 * 0.5;
    let m_zz = i_zz + m * l * l;
    let alpha_expected = -l * f_y * sin_q0 / m_zz;
    assert!(
        (alpha - alpha_expected).abs() < 1e-3,
        "hinge_z pre-rotated wrench: expected α ≈ {alpha_expected}, got {alpha}"
    );
    // The mutant would land on +|alpha_expected|; assert we are firmly on
    // the correct side of zero so a sign-only mutation is caught.
    assert!(
        alpha < 0.0,
        "world-vs-body-frame sign: α must be negative here, got {alpha}"
    );
}

#[test]
fn wrench_clear_zeros_all_links() {
    // Confidence check for the API: set, then clear, must produce zero at
    // every slot regardless of how many links were touched.
    let mut tree = build_two_link_chain(1.0, 1.0);
    tree.set_link_wrench(1, Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3));
    tree.set_link_wrench(2, Vec3::new(-1.0, -2.0, -3.0), Vec3::ZERO);
    tree.clear_applied_wrenches();
    for w in &tree.applied_wrenches {
        assert_eq!(*w, (Vec3::ZERO, Vec3::ZERO));
    }
}
