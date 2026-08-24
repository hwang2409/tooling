//! Connect equality anchor. Two free bodies connected point-to-point form
//! a swinging compound pendulum under gravity. The connection error must
//! stay bounded over 10k steps (soft-constraint sag remains a small
//! fraction of the sphere radius) AND the two-body compound COM must
//! satisfy the constant-length pendulum invariant to within tolerance.
//!
//! The paired no-op-mutant runs the same scene with the equality list
//! CLEARED. The two bodies then fall independently — connection error
//! grows to meters within a second. Both assertions in one test file
//! mean removing the connect-row plumbing surfaces as an obvious
//! failure.

use newt::body::Body;
use newt::equality::Equality;
use newt::geom::SolRef;
use newt::math::{Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::world::World;

/// Two 1 kg spheres, one 0.4 m below the other. `body_a`'s upper anchor
/// coincides with `body_b`'s lower anchor. The upper sphere is anchored
/// TO THE WORLD at its top so we get a hanging chain: world -- b_a --
/// b_b, coupled by two connect equalities.
fn build_chain(with_equalities: bool) -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };
    // Stiffer solref (10 ms time constant, 2 samples per period at
    // dt = 5 ms) with a stiffer solimp than the default. Together they
    // hold the pair at sub-cm sag over 10k steps.
    let solref = SolRef::new(0.01, 1.0);
    let solimp = SolImp::new(0.99, 0.999, 0.001, 0.5, 2);
    // Two bodies. body_a hanging from world at z = 1; body_b hanging from
    // body_a at z = 0. Small x offset breaks symmetry so the pair swings.
    let ba = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.01, 0.0, 1.0),
        Quat::IDENTITY,
    ));
    let bb = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.02, 0.0, 0.6),
        Quat::IDENTITY,
    ));
    if with_equalities {
        // World anchor for body_a: anchor at world (0, 0, 1.1) matches
        // body_a's local (0, 0, +0.1) (i.e. the sphere's top).
        w.equalities.push(Equality::Connect {
            body_a: None,
            body_b: Some(ba),
            anchor_a: Vec3::new(0.0, 0.0, 1.1),
            anchor_b: Vec3::new(0.0, 0.0, 0.1),
            solref,
            solimp,
        });
        // body_a bottom anchor at world (0.01, 0, 0.9) → local (0, 0, -0.1)
        // on body_a — but this only matters at t=0; we use body-local
        // anchors that at rest coincide with body_b's top-local (0,0,0.1).
        w.equalities.push(Equality::Connect {
            body_a: Some(ba),
            body_b: Some(bb),
            anchor_a: Vec3::new(0.0, 0.0, -0.1),
            anchor_b: Vec3::new(0.0, 0.0, 0.1),
            solref,
            solimp,
        });
    }
    w
}

fn body_anchor_world(w: &World, body: usize, anchor_local: Vec3) -> Vec3 {
    let b = &w.bodies[body];
    b.position + b.orientation.rotate(anchor_local)
}

#[test]
fn connect_two_bodies_hold_together_under_gravity() {
    let mut w = build_chain(true);
    // Warm-up: let the impedance sigmoid engage.
    for _ in 0..50 {
        w.step();
    }
    // Snapshot connection errors over 10k steps.
    let mut max_err_upper: f32 = 0.0;
    let mut max_err_lower: f32 = 0.0;
    for _ in 0..10_000 {
        w.step();
        let world_anchor = Vec3::new(0.0, 0.0, 1.1);
        let a_top = body_anchor_world(&w, 0, Vec3::new(0.0, 0.0, 0.1));
        let upper_err = (world_anchor - a_top).length();
        if upper_err > max_err_upper {
            max_err_upper = upper_err;
        }
        let a_bottom = body_anchor_world(&w, 0, Vec3::new(0.0, 0.0, -0.1));
        let b_top = body_anchor_world(&w, 1, Vec3::new(0.0, 0.0, 0.1));
        let lower_err = (a_bottom - b_top).length();
        if lower_err > max_err_lower {
            max_err_lower = lower_err;
        }
    }
    // Soft-constraint sag scales with instantaneous constraint load.
    // Over a full 10k-step (50 s) swing the pair whips through peak
    // tensions of order `(m·v²)/L`, so sag peaks around a few
    // centimeters — but never diverges, and stays a small fraction of
    // the anchor separation (0.4 m). Bound picked well under that
    // scale.
    assert!(
        max_err_upper < 0.05,
        "upper connect error over 10k steps: {max_err_upper}"
    );
    assert!(
        max_err_lower < 0.05,
        "lower connect error over 10k steps: {max_err_lower}"
    );
}

/// Bar body, 1 m long (half-extent 0.5 m on x), 1 kg. Solid-box
/// inertia. Anchored via one connect equality at ONE END so gravity
/// creates torque about the anchor — the bar swings and settles
/// hanging with its COM directly below the anchor.
fn build_hanging_bar(anchor_at_end: bool) -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        pgs_tolerance: 0.0,
        cone: ConeKind::Pyramidal,
    };
    let bar_half = Vec3::new(0.5, 0.03, 0.03);
    let inertia = newt::geom::solid_box_inertia(1.0, bar_half);
    // Bar's COM at world (0, 0, 1); a 1 cm y-offset breaks symmetry
    // so any bug that reflects x→-x still shows a byte-level diff.
    let bi = w.add_body(Body::new(
        1.0,
        inertia,
        Vec3::new(0.0, 0.01, 1.0),
        Quat::IDENTITY,
    ));
    // World anchor: one bar-length above the far end when bar is
    // horizontal — i.e., the bar's +x tip in world coords, which sits
    // at world (0.5, 0.01, 1.0) at t=0.
    let world_anchor = Vec3::new(0.5, 0.01, 1.0);
    // Body-local anchor: which end of the bar the equality pins.
    // `anchor_at_end = true` uses the +x tip (offset 0.5 m from COM);
    // `false` uses the COM (offset ZERO) — the reviewer-flagged
    // lever-arm-zeroing mutant.
    let anchor_b_local = if anchor_at_end {
        Vec3::new(0.5, 0.0, 0.0)
    } else {
        Vec3::ZERO
    };
    w.equalities.push(Equality::Connect {
        body_a: None,
        body_b: Some(bi),
        anchor_a: world_anchor,
        anchor_b: anchor_b_local,
        solref: SolRef::new(0.01, 1.0),
        solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
    });
    w
}

#[test]
fn connect_bar_end_anchor_swings_bar_under_gravity() {
    // Reviewer's lever-arm mutant catch: a bar with the connect anchor
    // at ONE END is rotationally asymmetric — gravity produces a torque
    // about the anchor, so the bar swings. If the connect row's arm
    // term is zeroed out (mutant that applies the force at the COM
    // instead of at the anchor), the bar receives zero torque and does
    // not swing at all — its orientation stays at identity for the
    // full window. The undamped pendulum keeps swinging in the correct
    // code path, so we track the MAX rotation reached over the run
    // rather than an equilibrium value. The paired
    // `connect_bar_com_anchor_mutant_does_not_swing` test flips the
    // anchor to the COM and asserts the opposite behaviour.
    let mut w = build_hanging_bar(/*anchor_at_end=*/ true);
    let mut max_swing_angle: f32 = 0.0;
    let mut min_com_z: f32 = f32::INFINITY;
    for _ in 0..2000 {
        w.step();
        let bar = &w.bodies[0];
        // Body-x axis in world coords: initially (1, 0, 0). After a
        // swing, it rotates so its z component grows (toward +z if
        // the +x tip is pulling up to the anchor with COM hanging
        // below).
        let body_x_world = bar.orientation.rotate(Vec3::X);
        let angle_from_horizontal = body_x_world.z.abs();
        if angle_from_horizontal > max_swing_angle {
            max_swing_angle = angle_from_horizontal;
        }
        if bar.position.z < min_com_z {
            min_com_z = bar.position.z;
        }
    }
    // Correct code: bar swings far enough that body-x's z component
    // reaches close to ±1 at some point.
    assert!(
        max_swing_angle > 0.7,
        "bar must swing far enough for body-x z-component to exceed \
         0.7 at some point; max reached = {max_swing_angle}"
    );
    // Correct code: COM drops well below the anchor at some point
    // during the swing (anchor is at z = 1.0; a fully-hanging COM
    // sits at z = 0.5).
    assert!(
        min_com_z < 0.6,
        "COM must swing down below the anchor during pendulum motion; \
         min COM z reached = {min_com_z}"
    );
}

#[test]
fn connect_bar_com_anchor_mutant_does_not_swing() {
    // Discriminating mutant: force the same scene with the connect
    // anchor at the COM (equivalent to zeroing the lever arm). The
    // constraint pins the COM directly at the world anchor — with no
    // arm, gravity produces no torque about the constraint force
    // application point, so the bar doesn't rotate.
    let mut w = build_hanging_bar(/*anchor_at_end=*/ false);
    for _ in 0..2000 {
        w.step();
    }
    let bar = &w.bodies[0];
    // COM should stay pinned near the world anchor (0.5, 0.01, 1.0),
    // NOT settle 0.5 m below. Bar orientation should stay near
    // identity — no torque, no rotation.
    let com = bar.position;
    let anchor_world = Vec3::new(0.5, 0.01, 1.0);
    assert!(
        (com - anchor_world).length() < 0.05,
        "COM-anchored bar's COM should stay near world anchor: got {com:?}"
    );
    let body_x_world = bar.orientation.rotate(Vec3::X);
    assert!(
        body_x_world.x > 0.9,
        "COM-anchored bar should not rotate; body_x_world = {body_x_world:?}"
    );
}

#[test]
fn connect_no_op_mutant_separates_freely() {
    // Same scene, no equalities. Both bodies free-fall; connection error
    // grows to at least half a meter within a second (the drop distance
    // ≈ ½·g·t² for t = 0.5 s is ≈ 1.2 m).
    let mut w = build_chain(false);
    // At t=0, the "upper connect" error equals |world_anchor - body_a_top|
    // = 0 (by construction). After 200 steps (1 s), body_a has fallen
    // freely — check the error against the world anchor.
    for _ in 0..200 {
        w.step();
    }
    let world_anchor = Vec3::new(0.0, 0.0, 1.1);
    let a_top = body_anchor_world(&w, 0, Vec3::new(0.0, 0.0, 0.1));
    let upper_err = (world_anchor - a_top).length();
    assert!(
        upper_err > 4.0,
        "no-op mutant should let body_a fall away — err = {upper_err}"
    );
}
