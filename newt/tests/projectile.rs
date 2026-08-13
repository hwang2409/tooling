//! Projectile anchor: a body launched with initial velocity under uniform
//! gravity must follow the closed-form parabola `p(t) = p0 + v0 t + ½ g t²`
//! to within a tight tolerance after many RK4 steps. RK4 on a linear (in
//! velocity) system with constant `g` is exact up to floating-point
//! rounding; we allow a small margin.

use newt::body::Body;
use newt::math::{Quat, Vec3};
use newt::world::World;

#[test]
fn free_fall_matches_closed_form_after_1000_steps() {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    let p0 = Vec3::new(2.0, -1.0, 5.0);
    let v0 = Vec3::new(1.5, 2.0, 4.0);
    let mut body = Body::solid_box(1.0, Vec3::new(0.1, 0.1, 0.1), p0, Quat::IDENTITY);
    body.linear_velocity = v0;
    let idx = world.add_body(body);

    let steps = 1000usize;
    for _ in 0..steps {
        world.step();
    }
    let t = steps as f32 * world.dt;

    let expected = p0 + v0 * t + world.gravity * (0.5 * t * t);
    let actual = world.bodies[idx].position;

    let err = (actual - expected).length();
    // Under exact arithmetic RK4 on this linear-in-v system reproduces the
    // parabola exactly; f32 rounding across 1000 steps stays well under 1 mm.
    assert!(err < 5.0e-4, "projectile drift {err} m after {steps} steps");

    // Velocity should also match to first order.
    let expected_v = v0 + world.gravity * t;
    let actual_v = world.bodies[idx].linear_velocity;
    let v_err = (actual_v - expected_v).length();
    // f32 rounding across 1000 * 4 = 4000 stage evaluations at v≈49 m/s
    // accumulates ~3e-4 m/s. Bound generously; this is round-off, not model
    // error (RK4 on constant `g` should be closed-form-exact in real arithmetic).
    assert!(v_err < 1.0e-3, "velocity drift {v_err} m/s");
}
