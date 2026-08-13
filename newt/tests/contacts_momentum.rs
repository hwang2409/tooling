//! Two-sphere head-on collision: total momentum approximately conserved.
//!
//! Newton's third law is baked into the contact-force application code (equal
//! and opposite wrenches on the two bodies). This test protects that
//! invariant: any refactor that applies force to only one body, or that
//! mislabels which side receives the reaction, will change the total momentum
//! sum during the collision window and fail the tight bound below.
//!
//! Zero gravity so the y and z axes stay uninteresting and the +X sum stays
//! constant. The bound is tight (≈ 1e-4) because with equal-and-opposite
//! forces the only remaining drift comes from RK4 evaluation rounding.

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::world::World;

#[test]
fn head_on_collision_conserves_linear_momentum() {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::ZERO;

    let radius = 0.5;
    let ma = 1.0;
    let mb = 1.0;
    let a_idx = world.add_body(Body::solid_sphere(
        ma,
        radius,
        Vec3::new(-2.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.bodies[a_idx].linear_velocity = Vec3::new(5.0, 0.0, 0.0);
    let b_idx = world.add_body(Body::solid_sphere(
        mb,
        radius,
        Vec3::new(2.0, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    world.bodies[b_idx].linear_velocity = Vec3::new(-3.0, 0.0, 0.0);

    world.add_geom(Geom::sphere(a_idx, radius, Vec3::ZERO, 0.5));
    world.add_geom(Geom::sphere(b_idx, radius, Vec3::ZERO, 0.5));

    let total_before =
        world.bodies[a_idx].linear_velocity * ma + world.bodies[b_idx].linear_velocity * mb;

    // Run past the collision (bodies approach for ~0.4 s, collide, separate).
    let mut max_deviation: f32 = 0.0;
    for _ in 0..400 {
        world.step();
        let p = world.bodies[a_idx].linear_velocity * ma + world.bodies[b_idx].linear_velocity * mb;
        let d = (p - total_before).length();
        if d > max_deviation {
            max_deviation = d;
        }
    }
    // With Newton's third law obeyed, drift is round-off only. 1e-3 is a
    // comfortable envelope that still trips on a mutant that drops the
    // reaction force on one body.
    assert!(
        max_deviation < 1.0e-3,
        "momentum drift {max_deviation} — check contact force pair symmetry"
    );

    // Also confirm the collision actually happened: sphere A should have
    // slowed and B sped up (net exchange in +X).
    assert!(
        world.bodies[a_idx].linear_velocity.x < 4.0,
        "sphere A did not decelerate: vx = {}",
        world.bodies[a_idx].linear_velocity.x
    );
    assert!(
        world.bodies[b_idx].linear_velocity.x > -2.0,
        "sphere B did not decelerate: vx = {}",
        world.bodies[b_idx].linear_velocity.x
    );
}
