//! condim 6 (rolling friction) anchor. A sphere rolling without
//! slipping on a plane. Under condim 6 with a non-zero rolling
//! coefficient, the sphere decelerates monotonically and comes to
//! rest. Under condim 3 (mutant), sliding friction alone leaves the
//! sphere rolling indefinitely — rolling-without-slipping means the
//! contact point has zero tangential velocity, so the sliding rows
//! have no work to do.
//!
//! Anchor: paired assertion in one file so removing the rolling row
//! from the solver fails a test.

use newt::body::Body;
use newt::geom::{Geom, SolRef};
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::world::World;

fn build_roller(condim: u8) -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let solref = SolRef::new(0.02, 1.0);
    let solimp = SolImp::DEFAULT;
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.9);
    plane.solref = solref;
    plane.solimp = solimp;
    plane.condim = condim;
    plane.torsional_friction = 0.0;
    plane.rolling_friction = 0.1;
    w.add_geom(plane);
    let radius = 0.1;
    let bi = w.add_body(Body::solid_sphere(
        1.0,
        radius,
        Vec3::new(0.0, 0.0, radius),
        Quat::IDENTITY,
    ));
    // Rolling without slipping: v = ω × r. Give initial linear
    // velocity +x and angular velocity about y so contact point has
    // zero tangential velocity.
    // v = 1 m/s along +x. r_contact_from_com = (0, 0, -r).
    // omega × r_contact = (0, ωy, 0) × (0, 0, -r) = (-ωy·r, 0, 0).
    // For contact velocity = v + omega × r = (1 - ωy·r, 0, 0) = 0 ⇒
    // ωy = 1/r = 10 rad/s.
    w.bodies[bi].linear_velocity = Vec3::new(1.0, 0.0, 0.0);
    w.bodies[bi].angular_velocity_body = Vec3::new(0.0, 10.0, 0.0);
    let mut sphere = Geom::sphere(bi, radius, Vec3::ZERO, 0.9);
    sphere.solref = solref;
    sphere.solimp = solimp;
    sphere.condim = condim;
    sphere.torsional_friction = 0.0;
    sphere.rolling_friction = 0.1;
    w.add_geom(sphere);
    w
}

fn linear_speed(w: &World) -> f32 {
    let v = w.bodies[0].linear_velocity;
    (v.x * v.x + v.y * v.y).sqrt()
}

#[test]
fn condim_6_rolling_ball_decelerates_monotonic() {
    let mut w = build_roller(6);
    // Settle onto the plane.
    for _ in 0..50 {
        w.step();
    }
    let v0 = linear_speed(&w);
    // Sample speed over 800 steps (4 s). Rolling friction pumps energy
    // out monotonically — the ball slows down.
    let mut last = v0;
    for chunk in 0..8 {
        for _ in 0..100 {
            w.step();
        }
        let now = linear_speed(&w);
        assert!(
            now <= last + 0.02,
            "condim 6 speed should decay monotonically; chunk {chunk} \
             now={now} > last={last}"
        );
        last = now;
    }
    let v_end = linear_speed(&w);
    assert!(
        v_end < 0.5 * v0,
        "condim 6 should halve rolling speed within 4 s: v0 = {v0}, end = {v_end}"
    );
}

#[test]
fn condim_3_mutant_ball_keeps_rolling() {
    let mut w = build_roller(3);
    for _ in 0..50 {
        w.step();
    }
    let v0 = linear_speed(&w);
    // Over the same 4 s window, condim 3 (sliding only) can't damp
    // rolling — the contact point has near-zero tangential velocity
    // (rolling-without-slipping), so the sliding cone doesn't engage.
    for _ in 0..800 {
        w.step();
    }
    let v_end = linear_speed(&w);
    assert!(
        v_end > 0.5 * v0,
        "condim 3 mutant should keep rolling: v0 = {v0}, end = {v_end}"
    );
}
