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
    // Hand-derived composite:
    //   Each sphere: m = 1 kg, r = 0.1, I_sphere = 2/5·m·r² = 0.004 kg·m²
    //   (isotropic).
    //   Sphere COMs at (∓0.25, 0, 1) in world; composite COM at
    //   (0, 0, 1). Distance from composite COM to each sphere COM along
    //   +x is d = 0.25 m.
    //   Composite inertia about world +y axis (perpendicular to x, the
    //   axis connecting the sphere COMs) via parallel-axis theorem:
    //     I_composite_yy = 2 · (I_sphere_yy + m · d²)
    //                    = 2 · (0.004 + 1 · 0.0625)
    //                    = 0.133 kg·m².
    //   Initial angular momentum about composite COM:
    //     body_a: ω = 0 → L_a = 0.
    //     body_b: ω_body = (0, 1.5, 0), lin_vel = 0.
    //       body_b's I_body = 0.004·I₃; r_from_composite_com = (0.25,0,0).
    //       L_b = I_body · ω_b + m · r × v = (0, 0.006, 0) + 0 =
    //             (0, 0.006, 0) N·m·s.
    //   Total L about composite COM = (0, 0.006, 0) N·m·s. Gravity is
    //   uniform → net torque about composite COM = 0 → angular
    //   momentum conserved through the fall AND through the weld's
    //   internal impulses (equal-opposite pair at a single anchor
    //   point → zero net torque). Expected steady composite angular
    //   velocity:
    //     ω_composite_y = L_y / I_composite_yy = 0.006 / 0.133
    //                   ≈ 0.04511 rad/s.
    let mut w = build_pair(true);
    let com0_z = 1.0;
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
    // Analytic steady composite ω about +y from the derivation above.
    let expected_omega_y = 0.006 / 0.133_f32;
    let w_a_world = w.bodies[0]
        .orientation
        .rotate(w.bodies[0].angular_velocity_body);
    let w_b_world = w.bodies[1]
        .orientation
        .rotate(w.bodies[1].angular_velocity_body);
    // Both bodies rotate at the composite ω (weld drives them to the
    // same rigid-body twist).
    assert!(
        (w_a_world.y - expected_omega_y).abs() < 0.01,
        "body_a ω_y at composite steady state: {}, expected {expected_omega_y}",
        w_a_world.y
    );
    assert!(
        (w_b_world.y - expected_omega_y).abs() < 0.01,
        "body_b ω_y at composite steady state: {}, expected {expected_omega_y}",
        w_b_world.y
    );
    // Off-axis ω components should stay near zero (initial L has only
    // a y component; angular momentum conservation keeps the ω vector
    // parallel to +y since I_composite is diagonal about +y).
    assert!(
        w_a_world.x.abs() < 0.02 && w_a_world.z.abs() < 0.02,
        "body_a off-axis ω: x={}, z={}",
        w_a_world.x,
        w_a_world.z
    );
    let rel_w = (w_a_world - w_b_world).length();
    assert!(
        rel_w < 0.1,
        "welded pair relative ω after 1 s: {rel_w} rad/s"
    );
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
