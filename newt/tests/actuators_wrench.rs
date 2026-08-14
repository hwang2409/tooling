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

use newt::actuator::PdServo;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, rk4_step};

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
    let servo1 = PdServo::from_dampratio(1, kp, 1.0, m * l * l, 0.0);
    let servo2 = PdServo::from_dampratio(2, kp, 1.0, m * l * l, 0.0);
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
