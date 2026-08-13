//! Rolling anchor: a sphere released on a plane with pure linear velocity
//! transitions toward rolling without slipping under friction.
//!
//! Slip is `|v_x − r ω_y|` (with the +X motion / +Y angular convention). We
//! do not require it to reach exactly zero — with a viscous-with-Coulomb-clamp
//! friction the terminal state has residual slip proportional to the ratio of
//! rolling drag to viscous tangent coefficient. Instead we assert the slip
//! shrinks substantially and steadily.
//!
//! Mutation coverage: a mutant that mislabels tangent basis axes so the
//! friction points along y instead of x would fail to spin the sphere (ω_y
//! stays near zero → slip barely decreases). A missing-friction mutant would
//! never spin up the sphere either.

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::world::World;

#[test]
fn sphere_on_plane_transitions_toward_rolling() {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    let radius = 0.5;
    let mass = 1.0;
    let body_idx = world.add_body(Body::solid_sphere(
        mass,
        radius,
        Vec3::new(0.0, 0.0, radius + 0.02),
        Quat::IDENTITY,
    ));
    // Kick it forward: pure translation, no spin.
    world.bodies[body_idx].linear_velocity = Vec3::new(5.0, 0.0, 0.0);

    // High friction so we see the transition fast.
    let mu = 1.0;
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, mu));
    world.add_geom(Geom::sphere(body_idx, radius, Vec3::ZERO, mu));

    // Measure slip immediately (no prewarm): the sphere is just above the
    // plane and hasn't yet transitioned to rolling. Initial slip is the
    // full 5 m/s of pure translation.
    let body = &world.bodies[body_idx];
    let initial_slip = (body.linear_velocity.x - radius * body.angular_velocity_body.y).abs();
    assert!(
        initial_slip > 4.5,
        "initial slip should be near 5 m/s; got {initial_slip}"
    );

    // Long enough to enter and then dwell in the rolling regime. Analytic
    // slip-decay time for a solid sphere under Coulomb friction μ = 1 is
    // ≈ 2 v_0 / (7 μ g) ≈ 0.15 s, so 500 steps (2.5 s) is comfortably past
    // the transition even accounting for the penalty softness.
    for _ in 0..500 {
        world.step();
    }
    let body = &world.bodies[body_idx];
    let final_slip = (body.linear_velocity.x - radius * body.angular_velocity_body.y).abs();

    // Slip must shrink to a small fraction of its initial value.
    assert!(
        final_slip < 0.1,
        "slip failed to collapse: initial {initial_slip} → final {final_slip}"
    );
    // Sphere must actually be spinning about +Y (rolling forward). A
    // mutant that dropped tangent friction would leave ω_y ≈ 0.
    assert!(
        body.angular_velocity_body.y > 1.0,
        "expected ω_y > 1 rad/s after rolling transition, got {}",
        body.angular_velocity_body.y
    );
    // And the sphere kept moving forward — didn't get stopped by an
    // overweight tangential damper.
    assert!(
        body.linear_velocity.x > 0.5,
        "expected forward motion to persist; vx = {}",
        body.linear_velocity.x
    );
}
