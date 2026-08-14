//! Slide-joint anchors — closed-form free-fall and damped terminal-velocity
//! approach. Independent-twin: the reference is derived from Newton's law by
//! hand and evaluated in this test file (using platform libm is fine here —
//! the CI grep gate is `newt/src/` only). No `aba` call ever ends up on the
//! reference side.
//!
//! Both anchors use a fixed root at the origin plus a single link connected
//! by a slide along +Z. The child COM sits on the slide axis so `q_slide` is
//! literally the child's world-frame z coordinate (relative to the root).

use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, rk4_step};

fn zero_ext(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

fn build_vertical_slide(mass: f32, damping: f32) -> Tree {
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
        JointKind::Slide {
            axis: Vec3::Z,
            range: None,
            damping,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        mass,
        // Inertia is irrelevant for a pure slider (no rotation) but must
        // be positive-definite; keep small isotropic values.
        Mat3::diag(1e-4, 1e-4, 1e-4),
    ));
    tree
}

#[test]
fn slide_free_fall_matches_closed_form_gravity_acceleration() {
    // Mass on a frictionless vertical slide under gravity: qddot = -g
    // exactly (independent of mass — same free-fall as tier-1 projectile).
    // Reference is the closed-form q(t) = q0 + v0·t + 0.5·(-g)·t².
    let mass = 2.3f32; // non-unit mass to break "1.0 makes everything look right"
    let mut tree = build_vertical_slide(mass, 0.0);
    // Non-zero ICs so any wire-through-zero bug diverges immediately.
    let q0 = 0.5f32;
    let v0 = 0.7f32;
    tree.set_slide_position(1, q0);
    tree.set_slide_rate(1, v0);

    let g = 9.81f32;
    let dt = 0.001f32;
    let n_steps = 1000usize;
    for _ in 0..n_steps {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -g), dt, zero_ext(2));
    }
    let t = dt * n_steps as f32;
    let q_expected = q0 + v0 * t - 0.5 * g * t * t;
    let v_expected = v0 - g * t;
    let q_got = tree.slide_position(1);
    let v_got = tree.slide_rate(1);
    // RK4 is exact on a constant-acceleration ODE up to arithmetic; be tight.
    assert!(
        (q_got - q_expected).abs() < 1e-4,
        "slide q after {t}s: got {q_got}, expected {q_expected}"
    );
    assert!(
        (v_got - v_expected).abs() < 1e-4,
        "slide qdot after {t}s: got {v_got}, expected {v_expected}"
    );
}

#[test]
fn slide_damped_free_fall_approaches_terminal_velocity() {
    // m qddot = -m g - c qdot  →  qddot = -g - (c/m) qdot.
    // Terminal velocity v_∞ = -m g / c. Closed-form velocity for v0 = 0:
    //     v(t) = v_∞ · (1 − exp(-(c/m) · t))
    // With v0 ≠ 0:
    //     v(t) = v_∞ + (v0 − v_∞) · exp(-(c/m) · t)
    let mass = 1.5f32;
    let damping = 0.75f32; // c > 0
    let g = 9.81f32;
    let v_terminal = -mass * g / damping;

    let mut tree = build_vertical_slide(mass, damping);
    tree.set_slide_position(1, 2.0);
    tree.set_slide_rate(1, 0.0);

    let dt = 0.001f32;
    // τ = m/c = 2 s. Run 10 s → 5 τ → 99.3% approach so the terminal-
    // velocity check has meaningful headroom.
    let n_steps = 10_000usize;
    let mut samples: Vec<(f32, f32)> = Vec::new();
    for step in 1..=n_steps {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -g), dt, zero_ext(2));
        // Record velocity at t = 1..10 s in whole-second increments.
        if step % 1000 == 0 {
            let t = dt * step as f32;
            samples.push((t, tree.slide_rate(1)));
        }
    }
    for (t, v) in samples {
        let v_expected = v_terminal + (0.0 - v_terminal) * (-(damping / mass) * t).exp();
        assert!(
            (v - v_expected).abs() < 5e-3,
            "damped slide v at t={t}: got {v}, expected {v_expected} (v_∞ = {v_terminal})"
        );
    }
    // Final velocity within 1% of terminal after 5 τ.
    let v_final = tree.slide_rate(1);
    let approach_frac = (v_final - v_terminal).abs() / v_terminal.abs();
    assert!(
        approach_frac < 0.01,
        "final v {v_final} not close to terminal {v_terminal} (fraction {approach_frac})"
    );
}

#[test]
fn slide_range_limit_confines_release_from_outside() {
    // Mirrors the hinge_range_limit test: release the slider ABOVE its
    // upper limit and verify it settles inside/near the limit and does not
    // gain violation over time.
    let mass = 1.0f32;
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let lo = -0.5f32;
    let hi = 0.5f32;
    tree.push_link(Link::new(
        Some(0),
        JointKind::Slide {
            axis: Vec3::Z,
            range: Some((lo, hi)),
            damping: 0.2,
            armature: 0.0,
            limit: JointLimit::new(2000.0, 40.0),
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        mass,
        Mat3::diag(1e-4, 1e-4, 1e-4),
    ));
    tree.set_slide_position(1, hi + 0.25);
    tree.set_slide_rate(1, 0.0);
    let dt = 0.001f32;
    let mut violation_early_max = 0f32;
    let mut violation_late_max = 0f32;
    for step in 0..6_000usize {
        // No gravity → the only restoring force is the limit spring.
        rk4_step(&mut tree, Vec3::ZERO, dt, zero_ext(2));
        let q = tree.slide_position(1);
        let vio_hi = (q - hi).max(0.0);
        let vio_lo = (lo - q).max(0.0);
        let v = vio_hi.max(vio_lo);
        if step < 1500 {
            violation_early_max = violation_early_max.max(v);
        }
        if step > 4500 {
            violation_late_max = violation_late_max.max(v);
        }
    }
    let q_final = tree.slide_position(1);
    assert!(
        q_final >= lo - 0.05 && q_final <= hi + 0.05,
        "final q {q_final} outside limits [{lo}, {hi}]"
    );
    assert!(
        violation_late_max <= violation_early_max,
        "limit violation grew from early={violation_early_max} to late={violation_late_max}"
    );
}

#[test]
fn slide_armature_scales_static_acceleration_by_hand_ratio() {
    // Zero gravity. Apply a fixed force τ via qfrc_applied. F = (m + A) qddot
    // where A is the armature (kg). Compare α with A=0 vs A=1.
    let mass = 1.0f32;
    let armature = 1.0f32;
    let mut plain = build_vertical_slide(mass, 0.0);
    let mut armed_tree = Tree::new();
    armed_tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    armed_tree.push_link(Link::new(
        Some(0),
        JointKind::Slide {
            axis: Vec3::Z,
            range: None,
            damping: 0.0,
            armature,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        mass,
        Mat3::diag(1e-4, 1e-4, 1e-4),
    ));
    let force = 4.0f32;
    plain.qfrc_applied[plain.v_offset[1]] = force;
    armed_tree.qfrc_applied[armed_tree.v_offset[1]] = force;

    // Zero gravity so only τ matters.
    let g = Vec3::ZERO;
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); 2];
    let poses_plain = newt::tree::forward_kinematics(&plain);
    let poses_armed = newt::tree::forward_kinematics(&armed_tree);
    let a_plain = newt::tree::aba(&plain, &poses_plain, g, &ext);
    let a_armed = newt::tree::aba(&armed_tree, &poses_armed, g, &ext);
    let expected_plain = force / mass;
    let expected_armed = force / (mass + armature);
    assert!(
        (a_plain[0] - expected_plain).abs() < 1e-5,
        "plain slide α: {} vs {expected_plain}",
        a_plain[0]
    );
    let ratio_expected = expected_armed / expected_plain;
    let ratio_got = a_armed[0] / a_plain[0];
    assert!(
        (ratio_got - ratio_expected).abs() < 1e-5,
        "armature ratio {ratio_got} vs expected {ratio_expected}"
    );
}

#[test]
fn slide_damping_decelerates_a_coasting_slider() {
    // Zero gravity, no gravity force, damping only. m qddot = -c qdot →
    // v(t) = v0 · exp(-(c/m) t). Compare after ~2 s.
    let mass = 1.2f32;
    let damping = 0.6f32;
    let mut tree = build_vertical_slide(mass, damping);
    tree.set_slide_rate(1, 1.5);
    let dt = 0.001f32;
    let n = 2000usize;
    for _ in 0..n {
        rk4_step(&mut tree, Vec3::ZERO, dt, zero_ext(2));
    }
    let t = dt * n as f32;
    let v_expected = 1.5 * (-(damping / mass) * t).exp();
    let v_got = tree.slide_rate(1);
    assert!(
        (v_got - v_expected).abs() < 5e-4,
        "damped coast v at t={t}: got {v_got}, expected {v_expected}"
    );
}
