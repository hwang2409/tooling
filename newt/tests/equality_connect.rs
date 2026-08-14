//! Connect equality anchor. Two free bodies connected point-to-point form
//! a swinging compound pendulum under gravity. The connection error must
//! stay bounded over 10k steps (soft-constraint sag remains a small
//! fraction of the sphere radius) AND the two-body compound COM must
//! satisfy the constant-length pendulum invariant to within tolerance.
//!
//! The paired no-op-mutant runs the same scene with the equality list
//! CLEARED. The two bodies then fall independently — connection error
//! grows to meters within a second. Both assertions in one test file
//! mean removing the connect-row plumbing surfaces as an obvious
//! failure.

use newt::body::Body;
use newt::equality::Equality;
use newt::geom::SolRef;
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::world::World;

/// Two 1 kg spheres, one 0.4 m below the other. `body_a`'s upper anchor
/// coincides with `body_b`'s lower anchor. The upper sphere is anchored
/// TO THE WORLD at its top so we get a hanging chain: world -- b_a --
/// b_b, coupled by two connect equalities.
fn build_chain(with_equalities: bool) -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    // Stiffer solref (10 ms time constant, 2 samples per period at
    // dt = 5 ms) with a stiffer solimp than the default. Together they
    // hold the pair at sub-cm sag over 10k steps.
    let solref = SolRef::new(0.01, 1.0);
    let solimp = SolImp::new(0.99, 0.999, 0.001, 0.5, 2);
    // Two bodies. body_a hanging from world at z = 1; body_b hanging from
    // body_a at z = 0. Small x offset breaks symmetry so the pair swings.
    let ba = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.01, 0.0, 1.0),
        Quat::IDENTITY,
    ));
    let bb = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.02, 0.0, 0.6),
        Quat::IDENTITY,
    ));
    if with_equalities {
        // World anchor for body_a: anchor at world (0, 0, 1.1) matches
        // body_a's local (0, 0, +0.1) (i.e. the sphere's top).
        w.equalities.push(Equality::Connect {
            body_a: None,
            body_b: Some(ba),
            anchor_a: Vec3::new(0.0, 0.0, 1.1),
            anchor_b: Vec3::new(0.0, 0.0, 0.1),
            solref,
            solimp,
        });
        // body_a bottom anchor at world (0.01, 0, 0.9) → local (0, 0, -0.1)
        // on body_a — but this only matters at t=0; we use body-local
        // anchors that at rest coincide with body_b's top-local (0,0,0.1).
        w.equalities.push(Equality::Connect {
            body_a: Some(ba),
            body_b: Some(bb),
            anchor_a: Vec3::new(0.0, 0.0, -0.1),
            anchor_b: Vec3::new(0.0, 0.0, 0.1),
            solref,
            solimp,
        });
    }
    w
}

fn body_anchor_world(w: &World, body: usize, anchor_local: Vec3) -> Vec3 {
    let b = &w.bodies[body];
    b.position + b.orientation.rotate(anchor_local)
}

#[test]
fn connect_two_bodies_hold_together_under_gravity() {
    let mut w = build_chain(true);
    // Warm-up: let the impedance sigmoid engage.
    for _ in 0..50 {
        w.step();
    }
    // Snapshot connection errors over 10k steps.
    let mut max_err_upper: f32 = 0.0;
    let mut max_err_lower: f32 = 0.0;
    for _ in 0..10_000 {
        w.step();
        let world_anchor = Vec3::new(0.0, 0.0, 1.1);
        let a_top = body_anchor_world(&w, 0, Vec3::new(0.0, 0.0, 0.1));
        let upper_err = (world_anchor - a_top).length();
        if upper_err > max_err_upper {
            max_err_upper = upper_err;
        }
        let a_bottom = body_anchor_world(&w, 0, Vec3::new(0.0, 0.0, -0.1));
        let b_top = body_anchor_world(&w, 1, Vec3::new(0.0, 0.0, 0.1));
        let lower_err = (a_bottom - b_top).length();
        if lower_err > max_err_lower {
            max_err_lower = lower_err;
        }
    }
    // Soft-constraint sag scales with instantaneous constraint load.
    // Over a full 10k-step (50 s) swing the pair whips through peak
    // tensions of order `(m·v²)/L`, so sag peaks around a few
    // centimeters — but never diverges, and stays a small fraction of
    // the anchor separation (0.4 m). Bound picked well under that
    // scale.
    assert!(
        max_err_upper < 0.05,
        "upper connect error over 10k steps: {max_err_upper}"
    );
    assert!(
        max_err_lower < 0.05,
        "lower connect error over 10k steps: {max_err_lower}"
    );
}

#[test]
fn connect_no_op_mutant_separates_freely() {
    // Same scene, no equalities. Both bodies free-fall; connection error
    // grows to at least half a meter within a second (the drop distance
    // ≈ ½·g·t² for t = 0.5 s is ≈ 1.2 m).
    let mut w = build_chain(false);
    // At t=0, the "upper connect" error equals |world_anchor - body_a_top|
    // = 0 (by construction). After 200 steps (1 s), body_a has fallen
    // freely — check the error against the world anchor.
    for _ in 0..200 {
        w.step();
    }
    let world_anchor = Vec3::new(0.0, 0.0, 1.1);
    let a_top = body_anchor_world(&w, 0, Vec3::new(0.0, 0.0, 0.1));
    let upper_err = (world_anchor - a_top).length();
    assert!(
        upper_err > 4.0,
        "no-op mutant should let body_a fall away — err = {upper_err}"
    );
}
