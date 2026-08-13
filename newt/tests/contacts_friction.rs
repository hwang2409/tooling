//! Static-friction anchor: box on a tilted-gravity "incline".
//!
//! We tilt the gravity vector instead of the plane so the plane geom stays
//! horizontal — the tangent direction is then unambiguously +X. Two cases
//! bracket the Coulomb friction angle `atan(μ)`:
//!   - `θ_low = 10°` (well below `atan(0.5) ≈ 26.6°`): box stays effectively
//!     put. The viscous-with-Coulomb-clamp model produces a tiny drift
//!     proportional to `m g sin θ / c_tangent`; with default stiffness this
//!     is a few centimeters over the test window.
//!   - `θ_high = 45°` (well above): the tangent load exceeds the clamp, the
//!     box slides accelerating; drift is meters.
//!
//! The gap between the two drift bounds is over an order of magnitude —
//! designed to survive without retuning across integrator round-off changes.
//!
//! Mutation coverage: friction applied ALONG THE NORMAL would make the box
//! shoot away from the plane instead of resisting slide; both cases would
//! see huge z-motion and this test's tangential drift assertions would fail
//! (the low-angle case would slide freely, the high case would not slide in
//! the expected direction). Missing damping is checked elsewhere.

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3, cos, sin};
use newt::world::World;

fn incline_world(angle_rad: f32, mu: f32) -> World {
    let mut world = World::new();
    world.dt = 0.005;
    // Tilt gravity: tangent component in +X, normal against +Z.
    let g = 9.81;
    world.gravity = Vec3::new(g * sin(angle_rad), 0.0, -g * cos(angle_rad));

    let half = Vec3::splat(0.2);
    let body_idx = world.add_body(Body::solid_box(
        1.0,
        half,
        // Position so the box just touches the plane at rest.
        Vec3::new(0.0, 0.0, 0.2),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, mu));
    world.add_geom(Geom::r#box(body_idx, half, Vec3::ZERO, Quat::IDENTITY, mu));
    world
}

#[test]
fn box_below_friction_angle_stays_put() {
    let mu = 0.5;
    let angle = 10.0 * newt::math::PI / 180.0;
    let mut world = incline_world(angle, mu);
    let x0 = world.bodies[0].position.x;
    for _ in 0..800 {
        world.step();
    }
    let x = world.bodies[0].position.x;
    let drift = (x - x0).abs();
    // Empirical steady-state drift is ~0.03 m over 4 s at this angle; 0.15 m
    // leaves comfortable room while still failing a mutant that removes the
    // Coulomb cap entirely (drift then grows linearly to ~0.3 m in the same
    // window).
    assert!(
        drift < 0.15,
        "below-angle drift {drift} exceeded static-friction bound"
    );
}

#[test]
fn box_above_friction_angle_slides_downslope() {
    let mu = 0.5;
    let angle = 45.0 * newt::math::PI / 180.0;
    let mut world = incline_world(angle, mu);
    let x0 = world.bodies[0].position.x;
    for _ in 0..800 {
        world.step();
    }
    let x = world.bodies[0].position.x;
    let drift = x - x0;
    // With tan(45°) - μ = 0.5 net tangent, acceleration ≈ g/2 ≈ 4.9 m/s² —
    // over 4 s, drift ≈ 0.5 * 4.9 * 16 = ~39 m. Assert firmly above the low
    // case's ceiling of 0.15 m.
    assert!(
        drift > 1.0,
        "above-angle sliding drift {drift} unexpectedly small"
    );
    // Direction: gravity's +X tangent component drives motion +X.
    assert!(drift > 0.0, "sliding went the wrong way; drift {drift}");
}

#[test]
fn friction_coefficient_zero_removes_static_hold() {
    // Additional discrimination: with μ = 0, even the low-angle case must
    // slide — this catches a mutant that ignores μ (e.g. hard-coded cap
    // large enough to always clamp).
    let angle = 10.0 * newt::math::PI / 180.0;
    let mut world = incline_world(angle, 0.0);
    let x0 = world.bodies[0].position.x;
    for _ in 0..800 {
        world.step();
    }
    let drift = (world.bodies[0].position.x - x0).abs();
    // With g sin(10°) ≈ 1.7 m/s² unopposed, drift ≈ 0.5 * 1.7 * 16 ≈ 13.6 m.
    assert!(drift > 1.0, "μ = 0 should slide freely; drift only {drift}");
}
