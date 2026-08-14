//! Solver-mode golden trajectories: stack + incline scenes, symmetry-
//! broken, macOS-guarded regen. Same pattern as `tests/golden.rs` — a
//! fixed scene stepped for a fixed number of iterations, `(position,
//! orientation, linear velocity, body angular velocity)` snapshotted at
//! fixed step counts, byte-compared against a tracked file.
//!
//! These live SEPARATE from the penalty-mode goldens so a change to the
//! solver (e.g. the R2 blocker fix that gated the penalty limit under
//! Pgs, or any future MuJoCo-parity work landing under NEWT-13) can
//! legitimately regenerate solver-mode bytes without touching penalty
//! ones.
//!
//! Symmetry-break: box scenes carry small (0.01 m) x-offsets so any
//! mutation that swaps an axis or drops a lever arm surfaces as a
//! byte-level diff (the NEWT-5 arc lesson).

use newt::body::Body;
use newt::geom::{Geom, SolRef};
use newt::math::{Quat, Vec3, cos, sin};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::world::World;

const F32_PER_BODY: usize = 3 + 4 + 3 + 3;

fn stack_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let solref = SolRef::new(0.05, 1.5);
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.8);
    plane.solref = solref;
    w.add_geom(plane);
    let half = Vec3::splat(0.25);
    for k in 0..3 {
        let off = Vec3::new(0.01 * k as f32, 0.0, 0.25 + 0.5 * k as f32);
        let idx = w.add_body(Body::solid_box(1.0, half, off, Quat::IDENTITY));
        let mut g = Geom::r#box(idx, half, Vec3::ZERO, Quat::IDENTITY, 0.8);
        g.solref = solref;
        w.add_geom(g);
    }
    w
}

fn incline_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    // Tilt gravity so tangent component is along +X. Same technique the
    // penalty-mode friction anchor uses (tests/contacts_friction.rs).
    let angle = 32.0 * std::f32::consts::PI / 180.0;
    let g = 9.81;
    w.gravity = Vec3::new(g * sin(angle), 0.0, -g * cos(angle));
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let solref = SolRef::new(0.05, 1.5);
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5);
    plane.solref = solref;
    w.add_geom(plane);
    let half = Vec3::new(0.2, 0.21, 0.19); // asymmetric — symmetry break
    let bi = w.add_body(Body::solid_box(
        1.0,
        half,
        Vec3::new(0.01, 0.0, 0.2), // 1 cm x-offset
        Quat::IDENTITY,
    ));
    let mut box_geom = Geom::r#box(bi, half, Vec3::ZERO, Quat::IDENTITY, 0.5);
    box_geom.solref = solref;
    w.add_geom(box_geom);
    w
}

fn snapshot(world: &World) -> Vec<u8> {
    let mut out = Vec::with_capacity(world.bodies.len() * F32_PER_BODY * 4);
    for body in &world.bodies {
        for value in [
            body.position.x,
            body.position.y,
            body.position.z,
            body.orientation.x,
            body.orientation.y,
            body.orientation.z,
            body.orientation.w,
            body.linear_velocity.x,
            body.linear_velocity.y,
            body.linear_velocity.z,
            body.angular_velocity_body.x,
            body.angular_velocity_body.y,
            body.angular_velocity_body.z,
        ] {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

/// Step `world` and snapshot at `[0, 50, 200, 1000]` steps.
fn record(world: &mut World) -> Vec<u8> {
    let mut out = snapshot(world);
    for _ in 0..50 {
        world.step();
    }
    out.extend_from_slice(&snapshot(world));
    for _ in 50..200 {
        world.step();
    }
    out.extend_from_slice(&snapshot(world));
    for _ in 200..1000 {
        world.step();
    }
    out.extend_from_slice(&snapshot(world));
    out
}

const STACK_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/solver_stack.bin"
);
const INCLINE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/solver_incline.bin"
);

fn assert_golden(path: &str, actual: &[u8]) {
    let expected = std::fs::read(path).unwrap_or_else(|e| {
        panic!(
            "golden file missing at {path} — run the ignored regen test on \
             macOS to produce it, then commit ({e})"
        )
    });
    assert_eq!(
        expected.len(),
        actual.len(),
        "golden byte length mismatch for {path}: expected {} got {}",
        expected.len(),
        actual.len()
    );
    if expected != actual.to_vec() {
        let first_diff = expected
            .iter()
            .zip(actual.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!("golden trajectory mismatch at {path}; first byte diff at offset {first_diff}");
    }
}

#[test]
fn solver_stack_golden_byte_identical() {
    let mut w = stack_scene();
    let bytes = record(&mut w);
    assert_golden(STACK_PATH, &bytes);
}

#[test]
fn solver_incline_golden_byte_identical() {
    let mut w = incline_scene();
    let bytes = record(&mut w);
    assert_golden(INCLINE_PATH, &bytes);
}

/// Regenerate BOTH solver-mode goldens. Ignored so it does not run in
/// CI. macOS-aarch64 only — see the panic in the standing
/// [`regenerate_golden`](tests/golden.rs) test for the rationale.
#[test]
#[ignore]
fn regenerate_solver_goldens() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_solver_goldens may only run on the reference host \
             (macOS aarch64); refusing to overwrite the tracked bytes on \
             {} / {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }
    for (path, bytes) in [
        (STACK_PATH, record(&mut stack_scene())),
        (INCLINE_PATH, record(&mut incline_scene())),
    ] {
        let dir = std::path::Path::new(path).parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(path, &bytes).unwrap();
        println!("wrote {} bytes to {path}", bytes.len());
    }
}
