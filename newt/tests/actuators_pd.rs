//! PD servo anchors.
//!
//! Every expected value is derived from the linear second-order response
//! `I·q̈ = kp·(target − q) − kd·qdot` — a closed-form damped oscillator.
//! The engine's servo torque is the SAME formula (see [`newt::actuator`]),
//! so the discriminating power comes from routing that torque through ABA
//! and RK4 correctly. Mutations that mis-sign the actuator term, use the
//! wrong `qdot`, drop the clamp, or wire the torque to the wrong link
//! will land on one of these anchors.
//!
//! Setup convention: a single hinge link swinging a point-mass at
//! `L = 1 m` from the pivot with mass `m = 1 kg`, so the effective inertia
//! about the hinge axis is `m·L² = 1 kg·m²`. All hinges are about `+x`,
//! matching the tier-3 pendulum test.

use newt::actuator::Actuator;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, aba, forward_kinematics, rk4_step};

/// Build a "point-mass on a rod" tree: fixed root + one hinge_x link whose
/// COM sits at `L` below the pivot (`joint_offset_in_child = (0, 0, L)`).
/// Effective inertia about the hinge axis is `m·L²` up to the small
/// `Mat3::diag(1e-6, 1e-6, 1e-6)` COM-inertia.
fn build_hinge(mass: f32, length: f32) -> Tree {
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
        (Vec3::new(0.0, 0.0, length), Quat::IDENTITY),
        mass,
        Mat3::diag(1e-6, 1e-6, 1e-6),
    ));
    tree
}

fn zero_wrenches(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

#[test]
fn servo_critical_damping_step_no_overshoot() {
    // ωn = √(kp/I) = 10 rad/s, ζ = 1.0 → critically damped step. The
    // analytic response `q(t) = target·(1 − (1 + ωn·t)·exp(−ωn·t))` is
    // strictly monotone increasing toward `target`, so the maximum angle
    // observed during simulation should equal target within a tiny
    // integration slack.
    let mass = 1.0f32;
    let length = 1.0f32;
    let mut tree = build_hinge(mass, length);
    let servo = Actuator::position_from_dampratio(
        1, /*kp*/ 100.0, /*ζ*/ 1.0, /*I_ref*/ 1.0, /*clamp*/ 0.0,
    );
    let a = tree.add_actuator(servo);
    tree.set_actuator_target(a, 0.5);

    let dt = 0.001f32;
    let mut max_q = f32::NEG_INFINITY;
    for _ in 0..2000 {
        rk4_step(&mut tree, Vec3::ZERO, dt, zero_wrenches(2));
        let q = tree.hinge_angle(1);
        if q > max_q {
            max_q = q;
        }
    }
    // Critically damped: analytic peak = target exactly. Allow a whisker of
    // slack for RK4 rounding across 2 s of simulation.
    assert!(
        max_q <= 0.5 + 1e-4,
        "critical damping should not overshoot, max_q={max_q}"
    );
    // And the response should actually reach ~target (not stall).
    let q_final = tree.hinge_angle(1);
    assert!(
        (q_final - 0.5).abs() < 1e-3,
        "critical damping should settle to target, got {q_final}"
    );
}

#[test]
fn servo_underdamped_step_overshoots() {
    // ζ = 0.2 → underdamped. Analytic overshoot for a unit step:
    //   Mp = exp(-ζπ/√(1-ζ²)) = exp(-0.2·π/√0.96) ≈ exp(-0.6410) ≈ 0.5269
    // so peak ≈ target·(1 + 0.5269) = 0.5·1.5269 ≈ 0.7635 rad. Peak occurs
    // at t_peak = π/ωd where ωd = ωn·√(1-ζ²) ≈ 9.798 rad/s → t_peak ≈ 0.32 s.
    let mass = 1.0f32;
    let length = 1.0f32;
    let mut tree = build_hinge(mass, length);
    let servo = Actuator::position_from_dampratio(
        1, /*kp*/ 100.0, /*ζ*/ 0.2, /*I_ref*/ 1.0, 0.0,
    );
    let a = tree.add_actuator(servo);
    tree.set_actuator_target(a, 0.5);

    let dt = 0.001f32;
    let mut max_q = f32::NEG_INFINITY;
    for _ in 0..2000 {
        rk4_step(&mut tree, Vec3::ZERO, dt, zero_wrenches(2));
        let q = tree.hinge_angle(1);
        if q > max_q {
            max_q = q;
        }
    }
    // Peak should sit near 0.7635. Broad bracket around the analytic value.
    assert!(
        (0.72..0.81).contains(&max_q),
        "underdamped ζ=0.2 peak should be ≈ 0.7635, got {max_q}"
    );
}

#[test]
fn servo_p_only_steady_state_error_under_gravity() {
    // Gravity present, hinge_x, servo is P-only (kd=0). To settle the
    // transient into a steady state, add joint damping directly on the
    // hinge (10 N·m·s/rad) — that's an ANALYSIS aid, not part of the
    // servo model.
    //
    // Static balance (see docs/actuators.md derivation):
    //   0 = kp·(target − q_ss) − m·g·L·sin(q_ss)
    // For kp = 100, m·g·L = 9.81 (m=1, L=1, g=9.81), target = π/6 (0.5236),
    // Newton-solved q_ss ≈ 0.4785 rad. Sign convention: q > 0 with hinge_x
    // rotates the COM into +y (see tier-3 single_pendulum_at_horizontal
    // anchor), and gravity restores toward q = 0, so the equilibrium sits
    // *below* the target.
    let mass = 1.0f32;
    let length = 1.0f32;
    // Build a damped hinge directly (bypass `build_hinge` since we need
    // joint-level damping in the JointKind).
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
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 10.0,
            armature: 0.0,
            limit: newt::joint::HingeLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, length), Quat::IDENTITY),
        mass,
        Mat3::diag(1e-6, 1e-6, 1e-6),
    ));

    // P-only servo: kd = 0.
    let mut servo = Actuator::position(
        1, /*kp*/ 100.0, /*kd*/ 0.0, /*clamp*/ 0.0, 0.0,
    );
    servo.ctrl = std::f32::consts::PI / 6.0;
    tree.add_actuator(servo);

    let dt = 0.001f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    for _ in 0..10_000 {
        rk4_step(&mut tree, g, dt, zero_wrenches(2));
    }
    let q_ss = tree.hinge_angle(1);
    // Newton-derived expected value (five iterations from linearized guess,
    // hand-computed in the docstring above).
    let q_expected = 0.4785_f32;
    assert!(
        (q_ss - q_expected).abs() < 5e-4,
        "P-only steady state q_ss={q_ss}, expected {q_expected}"
    );
    // No residual libm check here — the q_ss assertion within 5e-4 rad
    // already implies the static-balance residual is small, and the engine
    // tests stay libm-free by policy.
}

#[test]
fn servo_clamp_bounds_the_effective_torque() {
    // Servo with a tight force clamp: the joint acceleration at rest with a
    // large target error must equal `clamped_torque / I` exactly, NOT the
    // unclamped `kp·error / I`.
    let mut tree = build_hinge(1.0, 1.0);
    // kp=1000, target=1, clamp=2 → unclamped τ = 1000, clamped τ = 2.
    // I = m·L² = 1, so expected α at q=0, qdot=0 is 2 rad/s².
    let mut servo = Actuator::position(1, 1000.0, 0.0, /*clamp*/ 2.0, 0.0);
    servo.ctrl = 1.0;
    tree.add_actuator(servo);
    let poses = forward_kinematics(&tree);
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()];
    let qddot = aba(&tree, &poses, Vec3::ZERO, &ext);
    let alpha = qddot[0]; // hinge angular accel (only DOF).
    assert!(
        (alpha - 2.0).abs() < 1e-4,
        "clamped α expected 2.0, got {alpha}"
    );

    // Flip the sign: negative target → negative clamped torque → negative α.
    let mut tree = build_hinge(1.0, 1.0);
    let mut servo = Actuator::position(1, 1000.0, 0.0, 2.0, 0.0);
    servo.ctrl = -1.0;
    tree.add_actuator(servo);
    let poses = forward_kinematics(&tree);
    let qddot = aba(&tree, &poses, Vec3::ZERO, &ext);
    let alpha = qddot[0];
    assert!(
        (alpha - -2.0).abs() < 1e-4,
        "clamped negative α expected -2.0, got {alpha}"
    );
}

#[test]
fn servo_clamp_binds_on_full_pd_expression_not_p_only() {
    // Discriminator between the intended `clamp(P + D)` and a bug where the
    // clamp only binds on the P term (`clamp(P) + D`). Same hinge_x + point
    // mass at L, kp = 1000, kd = 1, target = 1, clamp = ±2. Set an initial
    // hinge rate `qdot0 = 20` so the D contribution is large enough to
    // discriminate.
    //
    //   raw = kp·(target − q) − kd·qdot = 1000·1 − 1·20 = 980
    //   correct:      τ = clamp(980, ±2)      = +2
    //   mutant P-clamp: τ = clamp(1000, ±2) − 1·20 = 2 − 20 = −18
    //
    // I = m·L² = 1 → α_correct = +2 rad/s², α_mutant = −18 rad/s².
    let mut tree = build_hinge(1.0, 1.0);
    let mut servo = Actuator::position(1, 1000.0, 1.0, /*clamp*/ 2.0, 0.0);
    servo.ctrl = 1.0;
    tree.add_actuator(servo);
    tree.set_hinge_rate(1, 20.0);
    let poses = forward_kinematics(&tree);
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); tree.links.len()];
    let qddot = aba(&tree, &poses, Vec3::ZERO, &ext);
    let alpha = qddot[0];
    assert!(
        (alpha - 2.0).abs() < 1e-4,
        "clamp must bind on the full PD expression: expected α ≈ +2, got {alpha}"
    );
    // Sanity-check the mutant hypothesis is well outside the tolerance.
    assert!(
        (alpha - (-18.0)).abs() > 1.0,
        "mutant α (-18) should be far from observed α ({alpha})"
    );
}

#[test]
fn direct_joint_torque_and_clamp_flow_into_aba() {
    // set_joint_torque_clamped writes into qfrc_applied for a hinge slot;
    // aba should treat it exactly like the raw qfrc_applied path. Sanity-
    // check by comparing to a direct qfrc_applied assignment.
    let mut tree_a = build_hinge(1.0, 1.0);
    tree_a.set_joint_torque_clamped(1, 100.0, /*clamp*/ 3.0);
    let poses_a = forward_kinematics(&tree_a);
    let ext = vec![(Vec3::ZERO, Vec3::ZERO); 2];
    let qddot_a = aba(&tree_a, &poses_a, Vec3::ZERO, &ext);

    let mut tree_b = build_hinge(1.0, 1.0);
    tree_b.qfrc_applied[0] = 3.0; // clamped by hand
    let qddot_b = aba(&tree_b, &poses_a, Vec3::ZERO, &ext);
    assert!(
        (qddot_a[0] - qddot_b[0]).abs() < 1e-6,
        "clamped set = raw set: {} vs {}",
        qddot_a[0],
        qddot_b[0]
    );
    // Effective α = 3 / (m·L²) = 3 rad/s².
    assert!((qddot_a[0] - 3.0).abs() < 1e-4);

    // clear_qfrc_applied resets everything to zero.
    tree_a.clear_qfrc_applied();
    let qddot = aba(&tree_a, &poses_a, Vec3::ZERO, &ext);
    assert!(qddot[0].abs() < 1e-6);
}

#[test]
fn servo_holding_pendulum_against_gravity_settles_no_growth() {
    // Servo holding a pendulum at a target angle above rest. Even when the
    // servo has enough gain to hold, gravity kicks the joint each RK4 stage
    // and a subtle sign error in the actuator torque would show up as
    // amplitude growth over many periods. Assert that the swing amplitude
    // (max deviation from steady state) is strictly smaller in the last
    // second of a 10 s run than in the first second post-settle.
    let mass = 1.0f32;
    let length = 1.0f32;
    let mut tree = build_hinge(mass, length);
    let servo = Actuator::position_from_dampratio(
        1, /*kp*/ 200.0, /*ζ*/ 0.7, /*I_ref*/ 1.0, 0.0,
    );
    let a = tree.add_actuator(servo);
    tree.set_actuator_target(a, 0.6);

    let dt = 0.001f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    // Warm up: let the initial transient decay.
    for _ in 0..2000 {
        rk4_step(&mut tree, g, dt, zero_wrenches(2));
    }
    let q_settled = tree.hinge_angle(1);

    // Sample amplitude in the two windows.
    let mut amp_early = 0.0f32;
    for _ in 0..1000 {
        rk4_step(&mut tree, g, dt, zero_wrenches(2));
        let d = (tree.hinge_angle(1) - q_settled).abs();
        if d > amp_early {
            amp_early = d;
        }
    }
    for _ in 0..7000 {
        rk4_step(&mut tree, g, dt, zero_wrenches(2));
    }
    let mut amp_late = 0.0f32;
    for _ in 0..1000 {
        rk4_step(&mut tree, g, dt, zero_wrenches(2));
        let d = (tree.hinge_angle(1) - q_settled).abs();
        if d > amp_late {
            amp_late = d;
        }
    }
    // Late amplitude must not exceed early amplitude — a sign flip on the
    // actuator torque would drive an unstable pole and blow this up.
    assert!(
        amp_late <= amp_early + 1e-4,
        "amplitude grew: early={amp_early}, late={amp_late}"
    );
}
