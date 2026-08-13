//! Static-friction anchor: box on a tilted-gravity "incline".
//!
//! We tilt the gravity vector instead of the plane so the plane geom stays
//! horizontal — the tangent direction is then unambiguously +X. Two cases
//! bracket the Coulomb friction angle `atan(0.5) ≈ 26.6°`, tightly:
//!   - `θ_low = 22°` (below threshold, `tan 22° ≈ 0.404 < 0.5`): the
//!     Coulomb cap holds; the box stays effectively put (viscous drift
//!     only).
//!   - `θ_high = 32°` (above threshold, `tan 32° ≈ 0.625 > 0.5`): the
//!     tangent load exceeds the cap and the box slides accelerating.
//!
//! The 22°/32° bracket pins the cap firmly: a mutant that mis-scales μ by
//! ±20% would move the effective threshold off one side of the bracket and
//! flip an assertion. The wider 10°/45° bracket used earlier could hide such
//! a mutant.
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
    // 22° is 4.6° below the atan(0.5) threshold — tan(22°)/μ ≈ 0.81,
    // enough margin for the RK4/viscous residual without inviting a mutant
    // to escape.
    let angle = 22.0 * newt::math::PI / 180.0;
    let mut world = incline_world(angle, mu);
    let x0 = world.bodies[0].position.x;
    for _ in 0..800 {
        world.step();
    }
    let x = world.bodies[0].position.x;
    let drift = (x - x0).abs();
    // Empirical steady-state drift is ~0.07 m over 4 s at 22°; 0.15 m is
    // still a comfortable envelope. A mutant that shrinks the cap
    // significantly would slide much further within the same window.
    assert!(
        drift < 0.15,
        "below-angle drift {drift} exceeded static-friction bound"
    );
}

#[test]
fn box_above_friction_angle_slides_downslope() {
    let mu = 0.5;
    // 32° is 5.4° above atan(0.5) — tan(32°)/μ ≈ 1.25, so the cap binds
    // and the box slides with acceleration g(sin θ − μ cos θ) ≈ 1.05 m/s².
    // Over 4 s that predicts ≈ 8.4 m of drift; we require > 1 m.
    let angle = 32.0 * newt::math::PI / 180.0;
    let mut world = incline_world(angle, mu);
    let x0 = world.bodies[0].position.x;
    for _ in 0..800 {
        world.step();
    }
    let x = world.bodies[0].position.x;
    let drift = x - x0;
    assert!(
        drift > 1.0,
        "above-angle sliding drift {drift} unexpectedly small"
    );
    // Direction: gravity's +X tangent component drives motion +X.
    assert!(drift > 0.0, "sliding went the wrong way; drift {drift}");
}

#[test]
fn friction_coefficient_zero_removes_static_hold() {
    // Additional discrimination: with μ = 0, even the (previously-)below-
    // threshold case must slide. Catches a mutant that ignores μ (e.g.
    // hard-coded cap large enough to always clamp).
    let angle = 22.0 * newt::math::PI / 180.0;
    let mut world = incline_world(angle, 0.0);
    let x0 = world.bodies[0].position.x;
    for _ in 0..800 {
        world.step();
    }
    let drift = (world.bodies[0].position.x - x0).abs();
    // With g sin(22°) ≈ 3.67 m/s² unopposed, drift ≈ 0.5 * 3.67 * 16 ≈ 29 m.
    assert!(drift > 1.0, "μ = 0 should slide freely; drift only {drift}");
}
