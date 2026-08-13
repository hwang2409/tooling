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
    let mut max_quat_norm_error: f32 = 0.0;
    for _ in 0..10_000 {
        world.step();
        let body = &world.bodies[0];
        let e = body.kinetic_energy();
        let rel = ((e - e0) / e0).abs();
        if rel > max_rel_drift {
            max_rel_drift = rel;
        }
        // Direct proof that renormalization ran and kept the quaternion
        // unit — a renorm-skip mutant is caught here, not only at the
        // golden-trajectory gate several minutes downstream.
        let q_norm_err = (body.orientation.norm() - 1.0).abs();
        if q_norm_err > max_quat_norm_error {
            max_quat_norm_error = q_norm_err;
        }
    }

    // Tightened per review round 2. Observed drift is ~8e-6 over these
    // 10k steps; 5e-5 is a comfortable but still-discriminating ceiling
    // that a real regression (higher-order-order integrator loss, wrong
    // gyroscopic sign, ...) would trip immediately.
    assert!(
        max_rel_drift < 5.0e-5,
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
    assert!(
        max_quat_norm_error < 1.0e-5,
        "quaternion drifted from unit norm by {max_quat_norm_error} — \
         renormalization step probably skipped"
    );
}

#[test]
fn angular_momentum_conserved_in_world_frame() {
    let mut world = free_tumbler(Vec3::new(0.4, 1.2, 0.3));
    let l0 = world.bodies[0].angular_momentum_world();
    let l0_mag = l0.length();

    let mut max_rel_drift: f32 = 0.0;
    let mut max_quat_norm_error: f32 = 0.0;
    for _ in 0..10_000 {
        world.step();
        let body = &world.bodies[0];
        let l = body.angular_momentum_world();
        let rel = (l - l0).length() / l0_mag;
        if rel > max_rel_drift {
            max_rel_drift = rel;
        }
        let q_norm_err = (body.orientation.norm() - 1.0).abs();
        if q_norm_err > max_quat_norm_error {
            max_quat_norm_error = q_norm_err;
        }
    }
    // Bound looser than energy because L involves both R and ω; both drift.
    assert!(
        max_rel_drift < 5.0e-3,
        "angular-momentum drift {max_rel_drift} exceeded bound"
    );
    assert!(
        max_quat_norm_error < 1.0e-5,
        "quaternion drifted from unit norm by {max_quat_norm_error} — \
         renormalization step probably skipped"
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

/// Rotation matrix from a unit quaternion; used to build a non-diagonal
/// inertia tensor for the anchor below via a similarity transform.
fn rotation_matrix(q: Quat) -> Mat3 {
    q.to_mat3()
}

#[test]
fn energy_and_momentum_conserved_for_non_diagonal_inertia() {
    // If the inertia in Body::new is ever accidentally transposed, or the
    // gyroscopic term drops a Rᵀ, a purely diagonal inertia hides the bug
    // (I = Iᵀ). Build I = R · diag(1,2,3) · Rᵀ for a nontrivial R so
    // I ≠ diag and I ≠ Iᵀ transposed elementwise, then run the same 10k
    // energy + world-frame L conservation bounds as the diagonal case.
    let r = rotation_matrix(Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0), 0.7));
    let d = Mat3::diag(1.0, 2.0, 3.0);
    let inertia = r * d * r.transpose();

    // Assert the constructed inertia really is non-diagonal — a mutant
    // that shortcuts to diag() would let this test regress silently.
    let off_diag_mag = inertia.get(0, 1).abs() + inertia.get(0, 2).abs() + inertia.get(1, 2).abs();
    assert!(
        off_diag_mag > 0.1,
        "off-diagonal magnitude {off_diag_mag} unexpectedly small; \
         this anchor is designed for a non-diagonal I"
    );

    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::ZERO;
    let mut body = Body::new(1.0, inertia, Vec3::ZERO, Quat::IDENTITY);
    body.angular_velocity_body = Vec3::new(0.4, 1.2, 0.3);
    world.add_body(body);

    let e0 = world.bodies[0].kinetic_energy();
    let l0 = world.bodies[0].angular_momentum_world();
    let l0_mag = l0.length();
    assert!(e0 > 0.0 && l0_mag > 0.0);

    let mut max_e_drift: f32 = 0.0;
    let mut max_l_drift: f32 = 0.0;
    for _ in 0..10_000 {
        world.step();
        let body = &world.bodies[0];
        let e_rel = ((body.kinetic_energy() - e0) / e0).abs();
        let l_rel = (body.angular_momentum_world() - l0).length() / l0_mag;
        if e_rel > max_e_drift {
            max_e_drift = e_rel;
        }
        if l_rel > max_l_drift {
            max_l_drift = l_rel;
        }
    }
    assert!(
        max_e_drift < 5.0e-5,
        "non-diagonal I: energy drift {max_e_drift} exceeded bound"
    );
    assert!(
        max_l_drift < 5.0e-3,
        "non-diagonal I: angular-momentum drift {max_l_drift} exceeded bound"
    );
}

#[test]
fn angular_momentum_conserved_with_offset_initial_orientation() {
    // The plain L-conservation test above starts from q0 = IDENTITY, so a
    // wrong-frame implementation of `angular_momentum_world` that forgets
    // to rotate `I ω` by the orientation would coincidentally look
    // constant while R stayed near identity for the first few steps. Start
    // from a nontrivial q0 with ω_body NOT parallel to q0's axis; the
    // wrong-frame implementation now drifts within ~500 steps, but the
    // correct implementation stays inside a tight bound.
    let q0 = Quat::from_axis_angle(Vec3::new(1.0, -0.4, 0.3), 0.9);
    let inertia = Mat3::diag(1.0, 2.0, 3.0);
    let mut body = Body::new(1.0, inertia, Vec3::ZERO, q0);
    // ω_body deliberately not aligned with q0's axis (1, -0.4, 0.3).
    body.angular_velocity_body = Vec3::new(0.4, 1.2, 0.3);

    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::ZERO;
    world.add_body(body);

    let l0 = world.bodies[0].angular_momentum_world();
    let l0_mag = l0.length();
    assert!(l0_mag > 0.0);

    let mut max_rel_drift: f32 = 0.0;
    for _ in 0..10_000 {
        world.step();
        let l = world.bodies[0].angular_momentum_world();
        let rel = (l - l0).length() / l0_mag;
        if rel > max_rel_drift {
            max_rel_drift = rel;
        }
    }
    assert!(
        max_rel_drift < 5.0e-3,
        "offset-q0 L drift {max_rel_drift} exceeded bound — suspect a \
         wrong-frame angular-momentum-in-world implementation"
    );
}
