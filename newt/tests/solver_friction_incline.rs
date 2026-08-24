//! Friction incline anchors under solver mode.
//!
//! Same bracket as `tests/contacts_friction.rs`:
//! - θ = 22° (below atan(0.5)): cone holds, drift bounded.
//! - θ = 32° (above): cone saturates, box slides down-slope.
//!
//! Runs BOTH cone kinds (pyramidal, elliptic) — a mutant that swaps in a
//! per-axis pyramidal projection while calling it elliptic would fail the
//! symmetric-friction discrimination.

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3, cos, sin};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::world::World;

fn incline_world(angle_rad: f32, mu: f32, cone: ConeKind) -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        pgs_tolerance: 0.0,
        cone,
    };
    let g = 9.81;
    world.gravity = Vec3::new(g * sin(angle_rad), 0.0, -g * cos(angle_rad));

    let half = Vec3::splat(0.2);
    let body_idx = world.add_body(Body::solid_box(
        1.0,
        half,
        Vec3::new(0.0, 0.0, 0.2),
        Quat::IDENTITY,
    ));
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, mu));
    world.add_geom(Geom::r#box(body_idx, half, Vec3::ZERO, Quat::IDENTITY, mu));
    world
}

#[test]
fn solver_pyramidal_holds_at_22deg_slides_at_32deg() {
    let mu = 0.5;
    let below = 22.0 * newt::math::PI / 180.0;
    let above = 32.0 * newt::math::PI / 180.0;
    // 22°: static cone should hold. Some drift permitted (solver soft
    // impedance + RK4 + 5ms dt).
    {
        let mut w = incline_world(below, mu, ConeKind::Pyramidal);
        let x0 = w.bodies[0].position.x;
        for _ in 0..800 {
            w.step();
        }
        let drift = (w.bodies[0].position.x - x0).abs();
        assert!(
            drift < 0.20,
            "pyramidal 22°: drift {drift} exceeds static-cone bound"
        );
    }
    // 32°: cone saturates → box slides > 1 m in 4 s.
    {
        let mut w = incline_world(above, mu, ConeKind::Pyramidal);
        let x0 = w.bodies[0].position.x;
        for _ in 0..800 {
            w.step();
        }
        let drift = w.bodies[0].position.x - x0;
        assert!(
            drift > 1.0,
            "pyramidal 32°: sliding drift {drift} unexpectedly small"
        );
    }
}

#[test]
fn solver_elliptic_holds_at_22deg_slides_at_32deg() {
    let mu = 0.5;
    let below = 22.0 * newt::math::PI / 180.0;
    let above = 32.0 * newt::math::PI / 180.0;
    {
        let mut w = incline_world(below, mu, ConeKind::Elliptic);
        let x0 = w.bodies[0].position.x;
        for _ in 0..800 {
            w.step();
        }
        let drift = (w.bodies[0].position.x - x0).abs();
        assert!(
            drift < 0.20,
            "elliptic 22°: drift {drift} exceeds static-cone bound"
        );
    }
    {
        let mut w = incline_world(above, mu, ConeKind::Elliptic);
        let x0 = w.bodies[0].position.x;
        for _ in 0..800 {
            w.step();
        }
        let drift = w.bodies[0].position.x - x0;
        assert!(
            drift > 1.0,
            "elliptic 32°: sliding drift {drift} unexpectedly small"
        );
    }
}
