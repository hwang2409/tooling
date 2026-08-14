//! Distance equality anchor. Two free bodies at fixed distance orbit
//! each other about their COM. The separation stays clamped to `d0`
//! over 10k steps. A no-op mutant runs the same scene without the
//! distance equality: the bodies fly apart along their initial
//! tangential velocities.

use newt::body::Body;
use newt::equality::Equality;
use newt::geom::SolRef;
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::world::World;

/// Two 1-kg spheres 1 m apart, spinning about their COM with opposite
/// tangential velocities. No gravity. Anchor points at each sphere's
/// centre. Distance equality holds them at 1 m separation.
fn build_pair(with_equality: bool) -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::ZERO;
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    let ba = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(-0.5, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    let bb = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.5, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    // Tangential velocities: A moves +y, B moves -y. Together they orbit
    // about origin.
    w.bodies[ba].linear_velocity = Vec3::new(0.0, 0.3, 0.0);
    w.bodies[bb].linear_velocity = Vec3::new(0.0, -0.3, 0.0);
    if with_equality {
        w.equalities.push(Equality::Distance {
            body_a: Some(ba),
            body_b: Some(bb),
            anchor_a: Vec3::ZERO,
            anchor_b: Vec3::ZERO,
            distance: 1.0,
            solref: SolRef::new(0.01, 1.0),
            solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
        });
    }
    w
}

#[test]
fn distance_pair_orbits_without_drift() {
    let mut w = build_pair(true);
    for _ in 0..50 {
        w.step();
    }
    let mut max_dev: f32 = 0.0;
    for _ in 0..10_000 {
        w.step();
        let sep = (w.bodies[0].position - w.bodies[1].position).length();
        let dev = (sep - 1.0).abs();
        if dev > max_dev {
            max_dev = dev;
        }
    }
    assert!(
        max_dev < 0.005,
        "distance deviation over 10k steps: {max_dev}"
    );
}

#[test]
fn distance_no_op_mutant_flies_apart() {
    let mut w = build_pair(false);
    // Without a distance equality, tangential velocities carry the two
    // bodies apart at 0.6 m/s (relative). After 600 steps (3 s),
    // separation should grow well past 1.5 m — no restoring force,
    // linear drift dominates. Wider margin (3 s + 1.5 m) so minor
    // drift from a partially-broken constraint can't sneak past.
    for _ in 0..600 {
        w.step();
    }
    let sep = (w.bodies[0].position - w.bodies[1].position).length();
    assert!(
        sep > 1.5,
        "no-op mutant should let bodies drift apart to sep > 1.5 over 3 s: got {sep}"
    );
}
