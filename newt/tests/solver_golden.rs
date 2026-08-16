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
use newt::joint::JointKind;
use newt::math::{Quat, Vec3, cos, sin};
use newt::solver::{ConeKind, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
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

fn newton_stack_scene() -> World {
    let mut world = stack_scene();
    world.solver.mode = SolverMode::Newton;
    world
}

fn newton_incline_scene() -> World {
    let mut world = incline_scene();
    world.solver.mode = SolverMode::Newton;
    world
}

fn tree_contact_scene(mode: SolverMode) -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = SolverConfig {
        mode,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let mut plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.7);
    plane.solref = SolRef::new(0.05, 1.5);
    world.add_geom(plane);

    let mut tree = Tree::new();
    let root_orientation = Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.2).normalize(), 0.35);
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(0.13, -0.07, 0.78), root_orientation),
        (Vec3::ZERO, Quat::IDENTITY),
        1.2,
        newt::math::Mat3::diag(0.08, 0.09, 0.1),
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::new(0.0, 1.0, 0.0)),
        (Vec3::new(0.0, 0.0, -0.3), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.22), Quat::IDENTITY),
        0.75,
        newt::math::Mat3::diag(0.02, 0.025, 0.018),
    ));
    tree.set_hinge_angle(1, 0.37);
    tree.set_hinge_rate(1, -0.25);
    let tree_index = world.add_tree(tree);
    world.add_geom(Geom::box_on_link(
        tree_index,
        0,
        Vec3::new(0.22, 0.17, 0.19),
        Vec3::new(0.08, -0.06, 0.0),
        Quat::from_axis_angle(Vec3::Y, 0.2),
        0.7,
    ));
    world.add_geom(Geom::box_on_link(
        tree_index,
        1,
        Vec3::new(0.14, 0.12, 0.24),
        Vec3::new(0.0, 0.0, -0.18),
        Quat::from_axis_angle(Vec3::X, -0.15),
        0.7,
    ));
    world
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
const NEWTON_STACK_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/solver_newton_stack.bin"
);
const NEWTON_INCLINE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/solver_newton_incline.bin"
);
const TREE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/solver_tree.bin");
const NEWTON_TREE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/solver_newton_tree.bin"
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

#[test]
fn solver_newton_stack_golden_byte_identical() {
    let mut world = newton_stack_scene();
    assert_golden(NEWTON_STACK_PATH, &record(&mut world));
}

#[test]
fn solver_newton_incline_golden_byte_identical() {
    let mut world = newton_incline_scene();
    assert_golden(NEWTON_INCLINE_PATH, &record(&mut world));
}

#[test]
fn solver_tree_golden_byte_identical() {
    let mut world = tree_contact_scene(SolverMode::Pgs);
    assert_golden(TREE_PATH, &record_tree(&mut world));
}

#[test]
fn solver_newton_tree_golden_byte_identical() {
    let mut world = tree_contact_scene(SolverMode::Newton);
    assert_golden(NEWTON_TREE_PATH, &record_tree(&mut world));
}

fn snapshot_tree(world: &World) -> Vec<u8> {
    let mut out = Vec::new();
    for tree in &world.trees {
        for value in tree.q.iter().chain(&tree.qdot) {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    out
}

fn record_tree(world: &mut World) -> Vec<u8> {
    let mut out = snapshot_tree(world);
    for _ in 0..50 {
        world.step();
    }
    out.extend_from_slice(&snapshot_tree(world));
    for _ in 50..200 {
        world.step();
    }
    out.extend_from_slice(&snapshot_tree(world));
    for _ in 200..1000 {
        world.step();
    }
    out.extend_from_slice(&snapshot_tree(world));
    out
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
        (NEWTON_STACK_PATH, record(&mut newton_stack_scene())),
        (NEWTON_INCLINE_PATH, record(&mut newton_incline_scene())),
        (
            TREE_PATH,
            record_tree(&mut tree_contact_scene(SolverMode::Pgs)),
        ),
        (
            NEWTON_TREE_PATH,
            record_tree(&mut tree_contact_scene(SolverMode::Newton)),
        ),
    ] {
        let dir = std::path::Path::new(path).parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(path, &bytes).unwrap();
        println!("wrote {} bytes to {path}", bytes.len());
    }
}
