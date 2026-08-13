//! Double-pendulum trajectory anchor.
//!
//! Two uniform rods connected by a hinge. This test implements the textbook
//! Lagrangian equations of motion DIRECTLY in the file (hand-derived from
//! the standard `T = ½ q̇ᵀ M(q) q̇`, `V(q)` derivation) and integrates them
//! with a stand-alone RK4 that uses only scalar math from `newt::math`. The
//! engine is stepped in parallel with the same dt, and the two trajectories
//! are compared over ~2 seconds.
//!
//! The independent-twin rule is strict here: nothing in the reference RK4
//! calls into `newt::tree` or `newt::body` — the reference IS the Lagrangian,
//! the engine IS the ABA implementation. If they agree to a stated tolerance
//! for a chaotic-but-short horizon, they are almost certainly both right.
//!
//! Symmetry-break: mass and length asymmetry (m1 ≠ m2, L1 ≠ L2), and the
//! hinge axis is +x rather than an axis aligned with any pattern of symmetry.

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3, cos, sin};
use newt::tree::{Link, Tree, rk4_step};

// --- rod parameters (broken symmetry: masses and lengths differ) ---
const M1: f32 = 1.3;
const M2: f32 = 0.7;
const L1: f32 = 0.9;
const L2: f32 = 0.6;
const G: f32 = 9.81;

/// Reference double-pendulum accelerations (θ1'', θ2'') from the Lagrangian
/// mass matrix M(q) and RHS vector b(q, q̇). Derivation is in comments in
/// the source above the function; the equations are for two uniform rods
/// hinged at their ends with gravity along −z and angles measured about
/// the +x axis.
///
/// θ1 = link 1 angle from vertical (positive = link tips toward +y).
/// θ2 = link 2 angle relative to link 1.
fn lagrangian_accel(theta1: f32, theta2: f32, w1: f32, w2: f32) -> (f32, f32) {
    let s2 = sin(theta2);
    let c2 = cos(theta2);
    let k = M2 * L1 * L2;

    let m11 = (1.0 / 3.0) * M1 * L1 * L1 + M2 * L1 * L1 + (1.0 / 3.0) * M2 * L2 * L2 + k * c2;
    let m12 = (1.0 / 3.0) * M2 * L2 * L2 + 0.5 * k * c2;
    let m22 = (1.0 / 3.0) * M2 * L2 * L2;

    let rhs1 = k * s2 * w1 * w2 + 0.5 * k * s2 * w2 * w2
        - G * L1 * (0.5 * M1 + M2) * sin(theta1)
        - 0.5 * M2 * G * L2 * sin(theta1 + theta2);
    let rhs2 = -0.5 * k * s2 * w1 * w1 - 0.5 * M2 * G * L2 * sin(theta1 + theta2);

    let det = m11 * m22 - m12 * m12;
    let a1 = (m22 * rhs1 - m12 * rhs2) / det;
    let a2 = (-m12 * rhs1 + m11 * rhs2) / det;
    (a1, a2)
}

/// One RK4 step for the reference (θ1, θ2, w1, w2) 4-vector.
fn ref_rk4_step(state: &mut [f32; 4], dt: f32) {
    let (t1, t2, w1, w2) = (state[0], state[1], state[2], state[3]);

    let d1 = [
        w1,
        w2,
        lagrangian_accel(t1, t2, w1, w2).0,
        lagrangian_accel(t1, t2, w1, w2).1,
    ];

    let s2 = [
        t1 + 0.5 * dt * d1[0],
        t2 + 0.5 * dt * d1[1],
        w1 + 0.5 * dt * d1[2],
        w2 + 0.5 * dt * d1[3],
    ];
    let (a1, a2) = lagrangian_accel(s2[0], s2[1], s2[2], s2[3]);
    let d2 = [s2[2], s2[3], a1, a2];

    let s3 = [
        t1 + 0.5 * dt * d2[0],
        t2 + 0.5 * dt * d2[1],
        w1 + 0.5 * dt * d2[2],
        w2 + 0.5 * dt * d2[3],
    ];
    let (a1, a2) = lagrangian_accel(s3[0], s3[1], s3[2], s3[3]);
    let d3 = [s3[2], s3[3], a1, a2];

    let s4 = [
        t1 + dt * d3[0],
        t2 + dt * d3[1],
        w1 + dt * d3[2],
        w2 + dt * d3[3],
    ];
    let (a1, a2) = lagrangian_accel(s4[0], s4[1], s4[2], s4[3]);
    let d4 = [s4[2], s4[3], a1, a2];

    for i in 0..4 {
        state[i] += (d1[i] + 2.0 * d2[i] + 2.0 * d3[i] + d4[i]) * (dt / 6.0);
    }
}

fn build_engine_tree(theta1: f32, theta2: f32) -> Tree {
    let mut tree = Tree::new();
    // Root: fixed to world at origin.
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Link 1: uniform rod along local z. COM at rod middle. I_xx = I_yy =
    // (1/12) m L². I_zz small (thin rod).
    let i_perp1 = (1.0 / 12.0) * M1 * L1 * L1;
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, L1 * 0.5), Quat::IDENTITY),
        M1,
        Mat3::diag(i_perp1, i_perp1, 1e-6),
    ));
    // Link 2: uniform rod. Joint 2 anchor in link 1: at bottom of rod 1,
    // i.e. (0, 0, -L1/2) in link 1's body frame (below its COM).
    let i_perp2 = (1.0 / 12.0) * M2 * L2 * L2;
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.0, 0.0, -L1 * 0.5), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, L2 * 0.5), Quat::IDENTITY),
        M2,
        Mat3::diag(i_perp2, i_perp2, 1e-6),
    ));
    tree.set_hinge_angle(1, theta1);
    tree.set_hinge_angle(2, theta2);
    tree
}

#[test]
fn double_pendulum_matches_hand_lagrangian_over_two_seconds() {
    let theta1_0 = 0.5f32;
    let theta2_0 = 0.3f32;

    let mut tree = build_engine_tree(theta1_0, theta2_0);
    let mut ref_state = [theta1_0, theta2_0, 0.0f32, 0.0f32];

    let dt = 0.001f32;
    let total = 2.0f32;
    let n_steps = (total / dt) as usize;

    let mut max_err_theta1 = 0.0f32;
    let mut max_err_theta2 = 0.0f32;
    // Sample every 100 steps (0.1 s) so we characterize drift over the run.
    for step in 0..n_steps {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -G), dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 3]
        });
        ref_rk4_step(&mut ref_state, dt);
        if step % 100 == 0 {
            let e1 = (tree.hinge_angle(1) - ref_state[0]).abs();
            let e2 = (tree.hinge_angle(2) - ref_state[1]).abs();
            if e1 > max_err_theta1 {
                max_err_theta1 = e1;
            }
            if e2 > max_err_theta2 {
                max_err_theta2 = e2;
            }
        }
    }

    // Tolerance: 0.02 rad (~1.1°). Both are RK4 at dt=1ms; a well-formed ABA
    // should stay within this over 2 s of nontrivial swing. Chaotic
    // divergence for a double pendulum grows exponentially; if this ever
    // seems tight, shorten `total` before loosening the bound.
    assert!(
        max_err_theta1 < 0.02,
        "θ1 drift {max_err_theta1} rad exceeded tolerance over 2s"
    );
    assert!(
        max_err_theta2 < 0.02,
        "θ2 drift {max_err_theta2} rad exceeded tolerance over 2s"
    );
    // Also verify the run was NON-TRIVIAL: the pendulum must actually swing
    // (a bug where both stay near zero would sail through a tolerance check).
    assert!(
        ref_state[0].abs() + ref_state[1].abs() > 0.1,
        "reference stayed near equilibrium — test is not exercising ABA"
    );
}

#[test]
fn double_pendulum_conserves_energy_over_ten_seconds() {
    // Same rig, no damping; run 10s and verify total energy K + V stays
    // bounded. Independent of the Lagrangian trajectory match — this test
    // catches integrator drift that would still let trajectories agree
    // point-wise for a short horizon.
    let mut tree = build_engine_tree(0.4, 0.6);
    tree.set_hinge_rate(1, 0.3); // small nonzero initial velocities
    tree.set_hinge_rate(2, -0.2);

    let dt = 0.001f32;
    let steps = 10_000usize;

    // Energy of the engine's tree state via analytic formulas — using the
    // SAME hand-derived T and V as the Lagrangian, applied to the engine's
    // (θ, θ̇) values. If ABA had a sign bug, energy would run away here.
    let energy = |t1: f32, t2: f32, w1: f32, w2: f32| -> f32 {
        let c2 = cos(t2);
        let m11 = (1.0 / 3.0) * M1 * L1 * L1
            + M2 * L1 * L1
            + (1.0 / 3.0) * M2 * L2 * L2
            + M2 * L1 * L2 * c2;
        let m12 = (1.0 / 3.0) * M2 * L2 * L2 + 0.5 * M2 * L1 * L2 * c2;
        let m22 = (1.0 / 3.0) * M2 * L2 * L2;
        let t = 0.5 * m11 * w1 * w1 + m12 * w1 * w2 + 0.5 * m22 * w2 * w2;
        let v =
            -0.5 * M1 * G * L1 * cos(t1) - M2 * G * L1 * cos(t1) - 0.5 * M2 * G * L2 * cos(t1 + t2);
        t + v
    };
    let e0 = energy(
        tree.hinge_angle(1),
        tree.hinge_angle(2),
        tree.hinge_rate(1),
        tree.hinge_rate(2),
    );

    let mut max_drift: f32 = 0.0;
    for _ in 0..steps {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -G), dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 3]
        });
        let e = energy(
            tree.hinge_angle(1),
            tree.hinge_angle(2),
            tree.hinge_rate(1),
            tree.hinge_rate(2),
        );
        let rel = ((e - e0) / e0).abs();
        if rel > max_drift {
            max_drift = rel;
        }
    }
    assert!(
        max_drift < 5.0e-3,
        "energy drift {max_drift} over 10s exceeded bound"
    );
    // Sanity: swing must actually move (not stuck at rest at the top of the
    // energy well — a degenerate case that would auto-pass).
    let final_disp = tree.hinge_angle(1).abs() + tree.hinge_angle(2).abs();
    assert!(final_disp > 0.05, "double pendulum stayed near rest?");
}
