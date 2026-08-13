//! Torque-free tumbling anchors.
//!
//! A rigid body with an asymmetric inertia tensor, spinning under zero
//! external torque (turn gravity off), must conserve:
//!  - kinetic energy `E = ½ ω · I ω`
//!  - angular momentum in the WORLD frame `L = R I ω`
//!
//! We do not assert exact equality (RK4 is only 4th-order and quaternion
//! renormalization introduces tiny drift). Instead we assert bounded
//! *relative* drift and record the observed magnitude for later regression.

use newt::body::Body;
use newt::math::{Mat3, Quat, Vec3};
use newt::world::World;

fn free_tumbler(angular_body: Vec3) -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::ZERO; // torque-free requires no gravity torque either.
    //  gravity acts at COM so it contributes zero
    //  torque, but zeroing keeps the state simple.
    let inertia = Mat3::diag(1.0, 2.0, 3.0);
    let mut body = Body::new(1.0, inertia, Vec3::ZERO, Quat::IDENTITY);
    body.angular_velocity_body = angular_body;
    world.add_body(body);
    world
}

#[test]
fn energy_conserved_within_bound_over_10000_steps() {
    let mut world = free_tumbler(Vec3::new(0.3, 1.4, 0.2));
    let e0 = world.bodies[0].kinetic_energy();
    assert!(e0 > 0.0);

    let mut max_rel_drift: f32 = 0.0;
    for _ in 0..10_000 {
        world.step();
        let e = world.bodies[0].kinetic_energy();
        let rel = ((e - e0) / e0).abs();
        if rel > max_rel_drift {
            max_rel_drift = rel;
        }
    }

    // Reasonable RK4 bound: << 1e-3 relative energy drift over 50s of
    // simulated time at dt=5ms. If you make dt bigger, this loosens.
    assert!(
        max_rel_drift < 1.0e-3,
        "energy drift {max_rel_drift} exceeded bound"
    );
    // Also confirm the drift is not literally zero — if it were, the test
    // would be trivially satisfied by a bug that constant-returns e0.
    // We deliberately require BOTH: bounded AND nonzero (integrator is not
    // doing exact-equality fakery).
    assert!(
        max_rel_drift > 0.0,
        "expected some numerical drift; got exactly zero — suspect fake conservation"
    );
}

#[test]
fn angular_momentum_conserved_in_world_frame() {
    let mut world = free_tumbler(Vec3::new(0.4, 1.2, 0.3));
    let l0 = world.bodies[0].angular_momentum_world();
    let l0_mag = l0.length();

    let mut max_rel_drift: f32 = 0.0;
    for _ in 0..10_000 {
        world.step();
        let l = world.bodies[0].angular_momentum_world();
        let rel = (l - l0).length() / l0_mag;
        if rel > max_rel_drift {
            max_rel_drift = rel;
        }
    }
    // Bound looser than energy because L involves both R and ω; both drift.
    assert!(
        max_rel_drift < 5.0e-3,
        "angular-momentum drift {max_rel_drift} exceeded bound"
    );
}

#[test]
fn dzhanibekov_intermediate_axis_flip_occurs() {
    // The intermediate-axis theorem: a body spun about the intermediate
    // principal axis (I₂, where I₁ < I₂ < I₃) is unstable and periodically
    // flips. Spins about the smallest (I₁) or largest (I₃) axis remain
    // bounded near the initial axis.
    //
    // Our inertia is diag(1, 2, 3): x is min, y is intermediate, z is max.
    // Seed a tiny perturbation on x and z so the instability can grow (a
    // perfect I₂ eigenstate is a mathematical fixed point).
    let mut world = free_tumbler(Vec3::new(1.0e-2, 5.0, 1.0e-2));
    let mut min_y_projection: f32 = f32::INFINITY;
    let mut max_y_projection: f32 = f32::NEG_INFINITY;
    for _ in 0..3_000 {
        world.step();
        let y = world.bodies[0].angular_velocity_body.y;
        if y < min_y_projection {
            min_y_projection = y;
        }
        if y > max_y_projection {
            max_y_projection = y;
        }
    }
    // A flip means the angular velocity's y-component reverses sign at some
    // point in the trajectory. Assert both a strongly positive AND strongly
    // negative extremum occur.
    assert!(
        max_y_projection > 4.0,
        "expected y-spin to remain near +5 before flip; got max {max_y_projection}"
    );
    assert!(
        min_y_projection < -4.0,
        "expected Dzhanibekov flip to negative y; got min {min_y_projection}"
    );
}

#[test]
fn major_axis_spin_stays_bounded() {
    // Same inertia, spin about z (the max axis). Tiny perturbations on x, y.
    // The intermediate-axis theorem says the spin stays bounded near +z.
    let mut world = free_tumbler(Vec3::new(1.0e-2, 1.0e-2, 5.0));
    let mut min_z_projection: f32 = f32::INFINITY;
    for _ in 0..3_000 {
        world.step();
        let z = world.bodies[0].angular_velocity_body.z;
        if z < min_z_projection {
            min_z_projection = z;
        }
    }
    // z should stay strictly positive and near 5. A drop below ~4.5 would
    // suggest instability where there should be none.
    assert!(
        min_z_projection > 4.5,
        "major-axis spin lost too much of its z-component: min {min_z_projection}"
    );
}

#[test]
fn minor_axis_spin_stays_bounded() {
    // Spin about x (the min axis). Should also be stable.
    let mut world = free_tumbler(Vec3::new(5.0, 1.0e-2, 1.0e-2));
    let mut min_x_projection: f32 = f32::INFINITY;
    for _ in 0..3_000 {
        world.step();
        let x = world.bodies[0].angular_velocity_body.x;
        if x < min_x_projection {
            min_x_projection = x;
        }
    }
    assert!(
        min_x_projection > 4.5,
        "minor-axis spin lost too much of its x-component: min {min_x_projection}"
    );
}
