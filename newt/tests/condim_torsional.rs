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

/// Build the spinning-sphere scene at rest (no spin yet — the caller
/// settles it onto the ground with `settle_and_inject` before measuring
/// decay). `mu_torsion` lets tests inject scaled coefficients (the
/// reviewer's 100× under-scale mutant uses 0.005 instead of 0.5).
fn build_spinner_with_mu(condim: u8, mu_torsion: f32) -> World {
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
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8);
    plane.solref = solref;
    plane.solimp = solimp;
    plane.condim = condim;
    plane.torsional_friction = mu_torsion;
    plane.rolling_friction = 0.0;
    w.add_geom(plane);
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
    sphere.torsional_friction = mu_torsion;
    sphere.rolling_friction = 0.0;
    w.add_geom(sphere);
    w
}

fn build_spinner(condim: u8) -> World {
    build_spinner_with_mu(condim, 0.5)
}

/// Settle the sphere on the plane, then inject 8 rad/s about world Z.
/// Splits the "reach steady contact" phase from the "torsion decay
/// under a known w0" measurement, so bounds can be picked from an
/// analytic per-step impulse (mu · m·g · dt) rather than the settle-
/// dependent equilibrium.
fn settle_and_inject(w: &mut World, initial_spin: f32) {
    for _ in 0..200 {
        w.step();
    }
    w.bodies[0].angular_velocity_body = Vec3::new(0.0, 0.0, initial_spin);
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
    settle_and_inject(&mut w, 8.0);
    let w0 = spin_z(&w).abs();
    // Sample spin over 8 chunks of 40 steps (200 ms each = 1.6 s
    // total). Torsional friction damps monotonically toward zero.
    // Analytic per-step impulse: mu·f_n·dt = 0.5·9.81·0.005 ≈ 0.024
    // N·m·s → per-step ω decay ≤ impulse/I_zz = 0.024/0.004 = 6.1
    // rad/s. Spin dies within ~2 steps for the correct coefficient.
    let mut last = w0;
    for chunk in 0..8 {
        for _ in 0..10 {
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
    // Tight bound: correct code kills the spin in ~2 steps (per-step
    // impulse mu·f_n·dt = 0.5·9.81·0.005 ≈ 0.024 N·m·s → ω decay
    // 6.1 rad/s per step, above the injected 8 rad/s). After 80
    // steps (0.4 s) spin is effectively zero. Reviewer's 100×
    // under-scale mutant (mu_torsion = 0.005) can only kill 0.061
    // rad/s per step → ~5 rad/s residual, ratio 0.6 — fails the
    // 0.05·w0 bound cleanly.
    assert!(
        final_spin < 0.05 * w0,
        "condim 4 should decay spin to under 5% of w0 within 0.4 s: \
         w0 = {w0}, final = {final_spin}"
    );
}

#[test]
fn condim_4_torsional_100x_underscale_mutant_fails_final_bound() {
    // Reviewer's under-scale mutant: torsional_friction is 100× too
    // small (0.005 instead of 0.5). Under a fixed 1.6 s window with
    // the same w0, decay is drastically slower — spin remains well
    // above the 0.05·w0 bound of `condim_4_torsional_spin_down_monotonic`.
    // Paired assertion so removing/under-scaling the torsional row
    // surfaces here.
    let mut w = build_spinner_with_mu(4, 0.005);
    settle_and_inject(&mut w, 8.0);
    let w0 = spin_z(&w).abs();
    for _ in 0..80 {
        w.step();
    }
    let final_spin = spin_z(&w).abs();
    // Analytic: per-step ω decay ≤ 0.005·9.81·0.005/0.004 = 0.061
    // rad/s (a 100th of the correct-mu figure). Over 80 steps only
    // 4.9 rad/s can be killed, leaving ~3 rad/s residual (ratio
    // ≈ 0.4). Well above the 0.05·w0 bound the correct-mu test uses.
    assert!(
        final_spin > 0.05 * w0,
        "100× under-scaled torsional_friction should leave residual \
         spin > 5% of w0 after 0.4 s: w0 = {w0}, final = {final_spin}"
    );
}

#[test]
fn condim_3_mutant_keeps_spinning() {
    let mut w = build_spinner(3);
    settle_and_inject(&mut w, 8.0);
    let w0 = spin_z(&w).abs();
    // Over the same window, condim 3 (sliding-only) barely damps
    // the pure spin — the tangent friction row's cone is about world +z
    // × the normal (also +z), which for a contact right at the axis of
    // spin gives zero torque. Any decay comes only from tangent
    // friction at off-axis contacts (edge effects), not from a
    // torsional row.
    for _ in 0..80 {
        w.step();
    }
    let w_end = spin_z(&w).abs();
    assert!(
        w_end > 0.5 * w0,
        "condim 3 mutant should keep spinning: w0 = {w0}, w_end = {w_end}"
    );
}
