//! Small-amplitude pendulum period anchor.
//!
//! A point-mass pendulum of length L under gravity g has small-amplitude
//! period T = 2π√(L/g). We build the closest engine analogue: a single
//! hinge link with an inertia tensor so tiny that the effective inertia
//! about the pivot is dominated by m·L² (the parallel-axis contribution).
//!
//! Independent-twin rule: the expected period is HARD-CODED from the
//! closed-form formula, not derived from a second call into the engine.
//! Symmetry-break: the mass is 1.7 kg (not 1) and the hinge axis is
//! (1, 0, 0) explicitly (so a mutation that mixes up x/y/z axes shows up).

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, forward_kinematics, rk4_step};

#[test]
fn small_amplitude_pendulum_period_matches_analytic() {
    let l = 1.0f32;
    let g = 9.81f32;
    let expected_period = 2.0 * std::f32::consts::PI * (l / g).sqrt();
    // Small starting angle so the linearized approximation is valid.
    let theta0 = 0.05_f32;

    let mut tree = Tree::new();
    // Root fixed at origin.
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Pendulum: hinge about x, COM L below pivot.
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, l), Quat::IDENTITY),
        1.7,
        Mat3::diag(1e-6, 1e-6, 1e-6),
    ));
    tree.set_hinge_angle(1, theta0);

    let dt = 0.001_f32;
    let steps = (3.0 * expected_period / dt) as usize;
    // Track zero-crossings of the angle to measure period.
    let mut prev_angle = tree.hinge_angle(1);
    let mut crossings: Vec<f32> = Vec::new();
    let mut t = 0.0f32;
    for _ in 0..steps {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -g), dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 2]
        });
        t += dt;
        let angle = tree.hinge_angle(1);
        // Detect down-going zero-crossings (positive → negative). Linear
        // interpolate to sub-step precision.
        if prev_angle > 0.0 && angle <= 0.0 {
            let alpha = prev_angle / (prev_angle - angle);
            crossings.push(t - dt + alpha * dt);
        }
        prev_angle = angle;
    }
    assert!(
        crossings.len() >= 2,
        "expected at least 2 down-going zero crossings; got {}",
        crossings.len()
    );
    // First zero crossing is at ~T/4, second at ~T + T/4, so their gap is T.
    let measured_period = crossings[1] - crossings[0];
    let rel_err = (measured_period - expected_period).abs() / expected_period;
    assert!(
        rel_err < 0.01,
        "measured period {measured_period}, expected {expected_period}, \
         relative error {rel_err} > 1%"
    );
}

#[test]
fn pendulum_forward_kinematics_traces_a_circular_arc() {
    // As the pendulum swings, the COM trajectory must lie on the circle of
    // radius L centered at the pivot. This catches an FK bug (wrong axis
    // sign, dropped joint offset) independently of ABA correctness.
    let l = 0.7f32;
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
    for i in 0..20 {
        let angle = i as f32 * (std::f32::consts::PI / 10.0);
        tree.set_hinge_angle(1, angle);
        let poses = forward_kinematics(&tree);
        let com = poses[1].0;
        // COM must sit on the circle of radius L in the y-z plane, x = 0.
        let r = com.length();
        assert!(
            (r - l).abs() < 1e-5,
            "COM {com:?} not on circle of radius {l} (r={r})"
        );
        assert!(com.x.abs() < 1e-5);
    }
}
