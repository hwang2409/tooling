//! PGS solver smoke tests: resting box on plane, 3rd law, elliptic-cone
//! projection anchors, condim 1 sliding block.

use newt::body::Body;
use newt::geom::{Geom, SolRef};
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode, project_elliptic};
use newt::world::World;

fn approx(a: f32, b: f32, tol: f32, msg: &str) {
    assert!((a - b).abs() <= tol, "{msg}: {a} vs {b} (tol {tol})");
}

// ---------------------------------------------------------------------------
// Elliptic-cone projection anchors (hand-derived)
// ---------------------------------------------------------------------------

#[test]
fn elliptic_projection_inside_cone_unchanged() {
    // mu = 0.5, fn = 4 → cap = 2. (1, 1) has magnitude sqrt(2) ≈ 1.414 < 2.
    let (a, b) = project_elliptic(1.0, 1.0, 0.5, 4.0);
    approx(a, 1.0, 1e-6, "ft1 inside cone");
    approx(b, 1.0, 1e-6, "ft2 inside cone");
}

#[test]
fn elliptic_projection_outside_rescaled_to_cap() {
    // mu = 0.5, fn = 4 → cap = 2. (3, 4) has magnitude 5 > 2 → rescale to 2.
    // Projected vector direction is (3/5, 4/5); scaled: (1.2, 1.6).
    let (a, b) = project_elliptic(3.0, 4.0, 0.5, 4.0);
    approx(a, 1.2, 1e-6, "ft1 rescaled");
    approx(b, 1.6, 1e-6, "ft2 rescaled");
    // Post-projection magnitude equals cap.
    approx((a * a + b * b).sqrt(), 2.0, 1e-6, "magnitude at cap");
}

#[test]
fn elliptic_projection_on_edge_boundary_kept() {
    // Point exactly on the boundary — magnitude equals cap. Should stay.
    // mu = 1.0, fn = 5 → cap = 5. (3, 4) has magnitude 5, exactly on boundary.
    let (a, b) = project_elliptic(3.0, 4.0, 1.0, 5.0);
    approx(a, 3.0, 1e-6, "boundary ft1 unchanged");
    approx(b, 4.0, 1e-6, "boundary ft2 unchanged");
}

#[test]
fn elliptic_projection_zero_fn_returns_origin() {
    // No normal force → cone collapses to origin.
    let (a, b) = project_elliptic(3.0, 4.0, 0.5, 0.0);
    approx(a, 0.0, 1e-6, "zero fn ft1");
    approx(b, 0.0, 1e-6, "zero fn ft2");
}

// ---------------------------------------------------------------------------
// Resting box on plane (solver mode): with `d(pen)`-driven a_ref, the box
// should settle at a small positive penetration and hold there.
// ---------------------------------------------------------------------------

fn build_resting_box_on_plane() -> World {
    let mut w = World::new();
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };
    // Static ground plane.
    let plane_geom = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5);
    w.add_geom(plane_geom);
    // Unit box, starting slightly above the plane so it drops onto it.
    let bi = w.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::new(0.0, 0.0, 0.6),
        Quat::IDENTITY,
    ));
    let box_geom = Geom::r#box(bi, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY, 0.5);
    w.add_geom(box_geom);
    w
}

#[test]
fn solver_mode_box_settles_on_plane() {
    let mut w = build_resting_box_on_plane();
    // Integrate long enough to settle. dt = 5 ms, 400 steps = 2 s.
    for _ in 0..400 {
        w.step();
    }
    let z = w.bodies[0].position.z;
    let vz = w.bodies[0].linear_velocity.z;
    // Box half-height = 0.5, so a resting COM sits at z = 0.5. Solver
    // impedance allows a small positive penetration — SolImp::DEFAULT gives
    // dmax = 0.95 and width = 0.001, so the steady-state penetration is on
    // the order of a few mm (well under the box half-height).
    assert!(
        z < 0.5 + 0.01 && z > 0.5 - 0.02,
        "settled z should be near 0.5 (got {z})"
    );
    assert!(vz.abs() < 0.05, "settled vz should be near zero (got {vz})");
}

// ---------------------------------------------------------------------------
// Newton's third law: two spheres in contact — total linear momentum
// conserved to numerical precision.
// ---------------------------------------------------------------------------

#[test]
fn solver_two_body_contact_conserves_linear_momentum() {
    let mut w = World::new();
    w.gravity = Vec3::ZERO;
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };
    // Two overlapping spheres, initial velocity along +X on A, at rest on B.
    let a_idx = w.add_body(Body::solid_sphere(
        1.0,
        0.5,
        Vec3::new(-0.4, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    let b_idx = w.add_body(Body::solid_sphere(
        2.0,
        0.5,
        Vec3::new(0.4, 0.0, 0.0),
        Quat::IDENTITY,
    ));
    w.bodies[a_idx].linear_velocity = Vec3::new(1.0, 0.0, 0.0);
    // No gap: they overlap so a contact fires.
    w.add_geom(Geom::sphere(a_idx, 0.5, Vec3::ZERO, 0.5));
    w.add_geom(Geom::sphere(b_idx, 0.5, Vec3::ZERO, 0.5));
    let p0 = w.bodies[a_idx].linear_velocity * w.bodies[a_idx].mass
        + w.bodies[b_idx].linear_velocity * w.bodies[b_idx].mass;
    // Step; the pair should exchange momentum through the contact but the
    // total should be conserved (no external force, gravity zero).
    for _ in 0..20 {
        w.step();
    }
    let p1 = w.bodies[a_idx].linear_velocity * w.bodies[a_idx].mass
        + w.bodies[b_idx].linear_velocity * w.bodies[b_idx].mass;
    approx(p0.x, p1.x, 1e-3, "px conservation");
    approx(p0.y, p1.y, 1e-3, "py conservation");
    approx(p0.z, p1.z, 1e-3, "pz conservation");
}

// ---------------------------------------------------------------------------
// condim 1 (frictionless): a block on a tilted plane slides regardless of
// tangent angle. Even at a shallow angle where friction (condim=3) would
// hold it, condim=1 must let it slide.
// ---------------------------------------------------------------------------

#[test]
fn solver_condim_1_block_slides_regardless_of_angle() {
    let mut w = World::new();
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };
    // Plane tilted 10° about Y (very shallow — μ=1.0 friction would hold it).
    let tilt = 10.0f32.to_radians();
    let tilt_q = Quat::from_axis_angle(Vec3::Y, tilt);
    let normal = tilt_q.rotate(Vec3::Z);
    let mut plane_geom = Geom::static_plane(Vec3::ZERO, normal, 1.0);
    plane_geom.condim = 1;
    w.add_geom(plane_geom);
    let bi = w.add_body(Body::solid_box(
        1.0,
        Vec3::splat(0.5),
        Vec3::new(0.0, 0.0, 1.0),
        tilt_q,
    ));
    let mut box_geom = Geom::r#box(bi, Vec3::splat(0.5), Vec3::ZERO, Quat::IDENTITY, 1.0);
    box_geom.condim = 1;
    w.add_geom(box_geom);
    let x0 = w.bodies[0].position.x;
    for _ in 0..200 {
        w.step();
    }
    let x1 = w.bodies[0].position.x;
    // With condim 1 and tilt, the tangent component of gravity pushes the
    // box down-slope (roughly +X here). Confirm nontrivial displacement.
    let dx = x1 - x0;
    assert!(
        dx.abs() > 0.05,
        "condim-1 block should slide freely on tilt; got dx = {dx}"
    );
}

// Guard: the DEFAULT SolImp is still valid — quick sanity check so a
// downstream refactor that widens the range doesn't silently break
// existing tests that consume `SolImp::DEFAULT`.
#[test]
fn default_solimp_is_valid() {
    assert!(SolImp::DEFAULT.validate().is_ok());
    let _ = SolRef::DEFAULT;
}
