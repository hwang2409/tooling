//! Joint-coupling equality anchor. Two sibling hinges on a fixed-root
//! tree, coupled 2:1 via `q_a = 2·q_b`. A PD servo drives `q_b` to
//! ±0.5 rad; the coupling row must drive `q_a` to track `2·q_b` inside
//! a tight bound.
//!
//! Paired no-op mutant: with the coupling equality removed, `q_a`
//! stays at its rest state (~0) — the servo only actuates `q_b`, so
//! without the coupling, `q_a` has nothing to move it.

use newt::actuator::PdServo;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::World;

fn build_scene(with_coupling: bool) -> (World, usize, usize, usize) {
    // Root fixed at world origin. Two hinges as sibling children of the
    // root, both about the x axis, with small inertia to make dynamics
    // fast.
    let mut tree = Tree::new();
    let root = tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let link_a = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        (Vec3::new(-0.3, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    let link_b = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.3, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    // Actuator on B: PD target = 0.5 rad, aggressive gains.
    // Soft gains — kp small enough to stay well inside the RK4/dt=5ms
    // stability envelope for a 0.01 kg·m² hinge. Critical damping is
    // ≈0.45 N·m·s/rad; kd=1.0 is ≈2× critical.
    tree.add_actuator(PdServo::new(link_b, 5.0, 1.0, 100.0, 0.5));

    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::ZERO;
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    let tree_idx = w.add_tree(tree);
    if with_coupling {
        w.equalities.push(newt::equality::Equality::JointCoupling {
            tree: tree_idx,
            link_a,
            link_b,
            polycoef: [0.0, 2.0, 0.0], // q_a = 2 * q_b
            solref: newt::geom::SolRef::new(0.01, 1.0),
            solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
        });
    }
    (w, tree_idx, link_a, link_b)
}

#[test]
fn coupling_slaves_hinge_a_to_twice_hinge_b() {
    let (mut w, ti, link_a, link_b) = build_scene(true);
    // Warm-up: let servo drive B toward 0.5 rad and both hinges reach
    // steady state (slower servo needs a longer settle window).
    for _ in 0..2000 {
        w.step();
    }
    // Now sweep target so B tracks a slow sinusoid; sample tracking
    // error. 0.5 rad/s is well inside the servo's response bandwidth
    // (natural ω ≈ 22 rad/s for kp=5, I=0.01).
    let mut max_err: f32 = 0.0;
    for step in 0..2000 {
        let t = step as f32 * w.dt;
        let target = 0.5 * (0.5 * t).sin();
        w.trees[ti].set_actuator_target(0, target);
        w.step();
        let q_a = w.trees[ti].hinge_angle(link_a);
        let q_b = w.trees[ti].hinge_angle(link_b);
        let coupling_err = (q_a - 2.0 * q_b).abs();
        if coupling_err > max_err {
            max_err = coupling_err;
        }
    }
    // Coupling error stays small — it's proportional to the
    // instantaneous constraint load (servo torque), which is at most a
    // few N·m in this scene.
    assert!(
        max_err < 0.05,
        "coupling error over 2000 tracked steps: {max_err}"
    );
}

#[test]
fn coupling_no_op_mutant_leaves_hinge_a_at_rest() {
    let (mut w, ti, link_a, link_b) = build_scene(false);
    for _ in 0..2000 {
        w.step();
    }
    let q_a = w.trees[ti].hinge_angle(link_a);
    let q_b = w.trees[ti].hinge_angle(link_b);
    // B tracked to 0.5 rad by its PD servo; A never received any
    // command and stays at 0.
    assert!(
        (q_b - 0.5).abs() < 0.05,
        "servo should hold B near 0.5: q_b = {q_b}"
    );
    assert!(
        q_a.abs() < 0.01,
        "without coupling A should stay at 0: q_a = {q_a}"
    );
}

#[test]
fn coupling_jacobian_probe_matches_chain_rule_at_multiple_q_b() {
    // Direct chain-rule mutant catch: independently derive
    // k = c1 + 2·c2·q_b at several q_b probe states and compare
    // against the SOLVER's row Jacobian returned by
    // [`tree_coupling_jacobian_probe`]. Any mutation of the k formula
    // in the shared row-builder (`build_tree_solver_rows`) shows up
    // here — the probe reads the LIVE code path, not a mirror. Fails
    // for k_wrong = c1 (chain-rule dropped) and for
    // k_wrong = c1 + c2·q_b (2× factor dropped).
    for &(q_a, q_b) in &[
        (0.0f32, 0.0f32),
        (0.0, 0.3),
        (0.4, -0.5),
        // Positive residual too — exercises the escape-convention
        // sign flip on both coefficients.
        (1.2, 0.4),
    ] {
        let mut tree = Tree::new();
        let root = tree.push_link(Link::new(
            None,
            JointKind::Fixed,
            (Vec3::ZERO, Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(1.0, 1.0, 1.0),
        ));
        let link_a = tree.push_link(Link::new(
            Some(root),
            JointKind::hinge(Vec3::X),
            (Vec3::new(-0.3, 0.0, 0.0), Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(0.01, 0.01, 0.01),
        ));
        let link_b = tree.push_link(Link::new(
            Some(root),
            JointKind::hinge(Vec3::X),
            (Vec3::new(0.3, 0.0, 0.0), Quat::IDENTITY),
            (Vec3::ZERO, Quat::IDENTITY),
            1.0,
            Mat3::diag(0.01, 0.01, 0.01),
        ));
        tree.set_hinge_angle(link_a, q_a);
        tree.set_hinge_angle(link_b, q_b);
        let mut w = World::new();
        w.gravity = Vec3::ZERO;
        w.solver = SolverConfig {
            mode: SolverMode::Pgs,
            iterations: 20,
            cone: ConeKind::Pyramidal,
        };
        let ti = w.add_tree(tree);
        // Nontrivial c0, c1, c2 so the chain-rule term is a distinct
        // fraction of the total k at every probe q_b.
        let polycoef = [0.15f32, 0.5, 0.4];
        w.equalities.push(newt::equality::Equality::JointCoupling {
            tree: ti,
            link_a,
            link_b,
            polycoef,
            solref: newt::geom::SolRef::new(0.01, 1.0),
            solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
        });

        let probes = newt::solver::tree_coupling_jacobian_probe(&w.trees[ti], ti, &w.equalities);
        assert_eq!(probes.len(), 1);
        let probe = probes[0];
        // Hand-derived chain-rule Jacobian.
        let expected_k = polycoef[1] + 2.0 * polycoef[2] * q_b;
        let expected_r_signed = q_a - (polycoef[0] + polycoef[1] * q_b + polycoef[2] * q_b * q_b);
        // Escape-convention flip: +1/-k when r < 0, -1/+k when r >= 0.
        let expected_coeff_a: f32 = if expected_r_signed >= 0.0 { -1.0 } else { 1.0 };
        let expected_coeff_b: f32 = -expected_k * expected_coeff_a;
        // (When flip = -1, coeff_a = -1 → signum = -1 → coeff_b = -k·(-1) = k;
        // and the row's -k*flip = -k*-1 = k. Both agree.)
        assert!(
            (probe.coeff_a - expected_coeff_a).abs() < 1e-6,
            "coeff_a at (q_a={q_a}, q_b={q_b}): expected {expected_coeff_a}, got {}",
            probe.coeff_a
        );
        assert!(
            (probe.coeff_b - expected_coeff_b).abs() < 1e-6,
            "coeff_b at (q_a={q_a}, q_b={q_b}): expected {expected_coeff_b} \
             (from k = c1 + 2·c2·q_b = {expected_k}), got {}",
            probe.coeff_b
        );
        assert!(
            (probe.violation_signed - expected_r_signed).abs() < 1e-6,
            "violation_signed at (q_a={q_a}, q_b={q_b}): expected \
             {expected_r_signed}, got {}",
            probe.violation_signed
        );
    }
}

#[test]
fn coupling_tracks_fast_time_varying_q_b_under_quadratic_polycoef() {
    // Physical chain-rule mutant catch: with c2 ≠ 0 and q_b changing
    // at nonzero rate, the constraint velocity `J·qdot` picks up a
    // `2·c2·q_b·qdot_b` term through the chain rule. A mutant using
    // `k_wrong = c1` reports the wrong constraint velocity, so the
    // solver applies bogus damping that fights actual polynomial
    // tracking under fast motion. Sinusoidal servo drive at ω = 3
    // rad/s (period 2 s) keeps qdot_b large enough that a wrong-J
    // mutant blows past the tolerance.
    let mut tree = Tree::new();
    let root = tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let link_a = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        (Vec3::new(-0.3, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    let link_b = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.3, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    tree.add_actuator(PdServo::new(link_b, 5.0, 1.0, 100.0, 0.0));
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::ZERO;
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    let ti = w.add_tree(tree);
    // c2 = 0.4 chosen so 2·c2·q_b at q_b ≈ 0.4 is 0.32, doubling
    // the c1 = 0.3 baseline — mutant that drops this term is a ~50%
    // Jacobian error at peak amplitude.
    let polycoef = [0.0f32, 0.3, 0.4];
    w.equalities.push(newt::equality::Equality::JointCoupling {
        tree: ti,
        link_a,
        link_b,
        polycoef,
        solref: newt::geom::SolRef::new(0.01, 1.0),
        solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
    });
    // Warm-up so the servo settles before sampling.
    for _ in 0..1200 {
        w.step();
    }
    let mut max_err: f32 = 0.0;
    for step in 0..1200 {
        let t = step as f32 * w.dt;
        // ω = 3 rad/s, A = 0.4 rad → peak qdot_b = 1.2 rad/s.
        let target = 0.4 * (3.0 * t).sin();
        w.trees[ti].set_actuator_target(0, target);
        w.step();
        let q_a = w.trees[ti].hinge_angle(link_a);
        let q_b = w.trees[ti].hinge_angle(link_b);
        let expected = polycoef[0] + polycoef[1] * q_b + polycoef[2] * q_b * q_b;
        let err = (q_a - expected).abs();
        if err > max_err {
            max_err = err;
        }
    }
    // Correct Jacobian holds tracking below ~0.0003 rad on this
    // machine. Bound picked so k_wrong = c1 (drops the chain-rule
    // term entirely) blows past ~0.006 rad — a 20× degradation —
    // and both plausible chain-rule mutants (drop c2 entirely, or
    // drop the factor 2) fail this cleanly.
    assert!(
        max_err < 0.002,
        "chain-rule tracking error over 6 s of sinusoidal q_b: {max_err} rad"
    );
}

#[test]
fn coupling_polycoef_higher_order_terms_track() {
    // q_a = 0.1 + 0.5·q_b + 0.3·q_b². Drive B, verify A tracks polynomial.
    let mut tree = Tree::new();
    let root = tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let link_a = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        (Vec3::new(-0.3, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    let link_b = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.3, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    tree.add_actuator(PdServo::new(link_b, 5.0, 1.0, 100.0, 0.3));
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::ZERO;
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    let ti = w.add_tree(tree);
    w.equalities.push(newt::equality::Equality::JointCoupling {
        tree: ti,
        link_a,
        link_b,
        polycoef: [0.1, 0.5, 0.3],
        solref: newt::geom::SolRef::new(0.01, 1.0),
        solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
    });
    for _ in 0..3000 {
        w.step();
    }
    let q_a = w.trees[ti].hinge_angle(link_a);
    let q_b = w.trees[ti].hinge_angle(link_b);
    let expected = 0.1 + 0.5 * q_b + 0.3 * q_b * q_b;
    assert!(
        (q_a - expected).abs() < 0.05,
        "polynomial coupling: q_a = {q_a}, expected {expected} (q_b = {q_b})"
    );
}
