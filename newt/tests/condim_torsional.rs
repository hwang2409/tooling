//! condim 4 (torsional friction) anchor. A sphere resting on a plane,
//! spinning about the contact normal (world Z). With condim 4 and a
//! torsional-friction coefficient > 0, the spin decays monotonically
//! to rest. With condim 3 (mutant), sliding friction alone can't damp
//! this spin — the sphere-plane contact sits right at the axis of
//! rotation, so tangential velocity there is zero and the sliding
//! rows produce no restoring torque about z.
//!
//! Both scenarios asserted in the same file so removing the torsional
//! row from the solver surfaces as a paired failure.

use newt::body::Body;
use newt::geom::{Geom, SolRef};
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::world::World;

fn build_spinner(condim: u8) -> World {
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
    // Static ground.
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8);
    plane.solref = solref;
    plane.solimp = solimp;
    plane.condim = condim;
    plane.torsional_friction = 0.5;
    plane.rolling_friction = 0.0;
    w.add_geom(plane);
    // 1 kg solid sphere, r = 0.1. Contact-normal spin about Z: pure
    // torsion — sliding tangents contribute zero because the contact
    // point sits on the axis of rotation.
    let bi = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.0, 0.0, 0.5),
        Quat::IDENTITY,
    ));
    let mut sphere = Geom::sphere(bi, 0.1, Vec3::ZERO, 0.8);
    sphere.solref = solref;
    sphere.solimp = solimp;
    sphere.condim = condim;
    sphere.torsional_friction = 0.5;
    sphere.rolling_friction = 0.0;
    w.add_geom(sphere);
    // Initial spin about world Z (body Z, since orientation = identity).
    w.bodies[bi].angular_velocity_body = Vec3::new(0.0, 0.0, 8.0);
    w
}

fn spin_z(w: &World) -> f32 {
    // World-frame ω_z of the disc.
    w.bodies[0]
        .orientation
        .rotate(w.bodies[0].angular_velocity_body)
        .z
}

#[test]
fn condim_4_torsional_spin_down_monotonic() {
    let mut w = build_spinner(4);
    // Settle onto ground first.
    for _ in 0..100 {
        w.step();
    }
    let w0 = spin_z(&w).abs();
    // Sample spin over 800 steps (4 s). Torsional friction damps
    // monotonically toward zero.
    let mut last = w0;
    for chunk in 0..8 {
        for _ in 0..100 {
            w.step();
        }
        let now = spin_z(&w).abs();
        assert!(
            now <= last + 0.01,
            "condim 4 spin should decay monotonically; chunk {chunk} \
             now={now} > last={last}"
        );
        last = now;
    }
    let final_spin = spin_z(&w).abs();
    // Substantial decay from the initial 8 rad/s.
    assert!(
        final_spin < 0.5 * w0,
        "condim 4 should halve spin within 4 s: w0 = {w0}, final = {final_spin}"
    );
}

#[test]
fn condim_3_mutant_keeps_spinning() {
    let mut w = build_spinner(3);
    // Settle.
    for _ in 0..100 {
        w.step();
    }
    let w0 = spin_z(&w).abs();
    // Over the same 4 s window, condim 3 (sliding-only) barely damps
    // the pure spin — the tangent friction row's cone is about world +z
    // × the normal (also +z), which for a contact right at the axis of
    // spin gives zero torque. Any decay comes only from tangent
    // friction at off-axis contacts (edge effects), not from a
    // torsional row.
    for _ in 0..800 {
        w.step();
    }
    let w_end = spin_z(&w).abs();
    assert!(
        w_end > 0.5 * w0,
        "condim 3 mutant should keep spinning: w0 = {w0}, w_end = {w_end}"
    );
}
