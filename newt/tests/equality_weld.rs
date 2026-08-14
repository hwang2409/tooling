//! Weld equality anchor. Two bodies welded pose-to-pose fall as one
//! rigid body under gravity. Anchor: after N steps, the compound COM's
//! z position matches the analytic composite free-fall
//! `p_z(t) = p_z(0) + ½ g t²`, AND their relative pose is preserved to
//! within a tight bound.
//!
//! Paired no-op mutant: with the equality removed, the two bodies fall
//! at the same rate but their relative pose is UNCONSTRAINED — any
//! initial angular momentum on one but not the other separates them.

use newt::body::Body;
use newt::equality::Equality;
use newt::geom::SolRef;
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::world::World;

/// Two 1-kg spheres 0.5 m apart on x, welded pose-to-pose. Body B has
/// initial angular velocity about y — with the weld, the two rotate as
/// one; without, they drift apart in orientation.
fn build_pair(with_equality: bool) -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    let ba = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(-0.25, 0.0, 1.0),
        Quat::IDENTITY,
    ));
    let bb = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.25, 0.0, 1.0),
        Quat::IDENTITY,
    ));
    // Give body B a spin about world +y. The weld must equalize both
    // bodies' angular velocities.
    w.bodies[bb].angular_velocity_body = Vec3::new(0.0, 1.5, 0.0);
    if with_equality {
        w.equalities.push(Equality::Weld {
            body_a: Some(ba),
            body_b: Some(bb),
            // Anchor: midpoint between the two bodies (world 0, 0, 1).
            // In body-A local: (0.25, 0, 0). In body-B local: (-0.25, 0, 0).
            anchor_a: Vec3::new(0.25, 0.0, 0.0),
            anchor_b: Vec3::new(-0.25, 0.0, 0.0),
            // Locked relative orientation = identity (bodies parallel).
            relative_orientation: Quat::IDENTITY,
            solref: SolRef::new(0.01, 1.0),
            solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
        });
    }
    w
}

#[test]
fn weld_pair_falls_as_composite_and_orientations_stay_locked() {
    let mut w = build_pair(true);
    // Compound COM (2 bodies, each 1 kg) starts at world (0, 0, 1).
    let com0_z = 1.0;
    // Fall for 200 steps (1 s).
    for _ in 0..200 {
        w.step();
    }
    let t = 0.005 * 200.0;
    let expected_com_z = com0_z + 0.5 * (-9.81) * t * t;
    let com_z = 0.5 * (w.bodies[0].position.z + w.bodies[1].position.z);
    assert!(
        (com_z - expected_com_z).abs() < 0.02,
        "compound COM z = {com_z}, expected {expected_com_z}"
    );
    // Relative angular velocity should be tiny (weld drives them to the
    // same body-frame angular velocity — at IDENTITY relative
    // orientation the world-frame ω's must also agree).
    let w_a_world = w.bodies[0]
        .orientation
        .rotate(w.bodies[0].angular_velocity_body);
    let w_b_world = w.bodies[1]
        .orientation
        .rotate(w.bodies[1].angular_velocity_body);
    let rel_w = (w_a_world - w_b_world).length();
    assert!(
        rel_w < 0.1,
        "welded pair relative ω after 1 s: {rel_w} rad/s"
    );
    // Anchor coincidence.
    let anchor_a_world =
        w.bodies[0].position + w.bodies[0].orientation.rotate(Vec3::new(0.25, 0.0, 0.0));
    let anchor_b_world =
        w.bodies[1].position + w.bodies[1].orientation.rotate(Vec3::new(-0.25, 0.0, 0.0));
    let anchor_err = (anchor_a_world - anchor_b_world).length();
    assert!(anchor_err < 0.02, "weld anchor error: {anchor_err}");
}

#[test]
fn weld_no_op_mutant_bodies_spin_independently() {
    let mut w = build_pair(false);
    // Without the weld, body A does not pick up body B's spin. After 1 s
    // their world-frame ω's differ by ~1.5 rad/s (body B's initial
    // spin, minus zero).
    for _ in 0..200 {
        w.step();
    }
    let w_a_world = w.bodies[0]
        .orientation
        .rotate(w.bodies[0].angular_velocity_body);
    let w_b_world = w.bodies[1]
        .orientation
        .rotate(w.bodies[1].angular_velocity_body);
    let rel_w = (w_a_world - w_b_world).length();
    assert!(
        rel_w > 1.0,
        "no-op mutant should preserve independent spin: rel_w = {rel_w}"
    );
}
