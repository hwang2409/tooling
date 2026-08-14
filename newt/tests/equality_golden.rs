//! Golden trajectories for the v1-tier-5 additions — one scene per
//! constraint family. Same pattern as [`crate::solver_golden`]: a fixed
//! scene stepped for a fixed number of iterations, `(position,
//! orientation, linear velocity, body angular velocity)` snapshotted at
//! fixed step counts, byte-compared against a tracked file.
//!
//! Symmetry-break: scenes carry small (0.01 m) asymmetries so a
//! mutation that swaps an axis or drops a lever arm surfaces as a
//! byte-level diff (the NEWT-5 arc lesson).

use newt::actuator::PdServo;
use newt::body::Body;
use newt::equality::Equality;
use newt::geom::{Geom, SolRef};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{ConeKind, SolImp, SolverConfig, SolverMode};
use newt::tree::{Link, Tree};
use newt::world::World;

const F32_PER_BODY: usize = 3 + 4 + 3 + 3;

fn connect_pair_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let solref = SolRef::new(0.01, 1.0);
    let solimp = SolImp::new(0.99, 0.999, 0.001, 0.5, 2);
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
    w.equalities.push(Equality::Connect {
        body_a: None,
        body_b: Some(ba),
        anchor_a: Vec3::new(0.0, 0.0, 1.1),
        anchor_b: Vec3::new(0.0, 0.0, 0.1),
        solref,
        solimp,
    });
    w.equalities.push(Equality::Connect {
        body_a: Some(ba),
        body_b: Some(bb),
        anchor_a: Vec3::new(0.0, 0.0, -0.1),
        anchor_b: Vec3::new(0.0, 0.0, 0.1),
        solref,
        solimp,
    });
    w
}

fn coupling_scene() -> World {
    let mut tree = Tree::new();
    let root = tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let link_a = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        // 1 cm y-offset for symmetry break.
        (Vec3::new(-0.3, 0.01, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    let link_b = tree.push_link(Link::new(
        Some(root),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.3, -0.01, 0.0), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(0.01, 0.01, 0.01),
    ));
    tree.add_actuator(PdServo::new(link_b, 5.0, 1.0, 100.0, 0.5));
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::ZERO;
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let ti = w.add_tree(tree);
    w.equalities.push(Equality::JointCoupling {
        tree: ti,
        link_a,
        link_b,
        polycoef: [0.0, 2.0, 0.0],
        solref: SolRef::new(0.01, 1.0),
        solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
    });
    w
}

fn weld_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::new(0.0, 0.0, -9.81);
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        cone: ConeKind::Pyramidal,
    };
    let solref = SolRef::new(0.01, 1.0);
    let solimp = SolImp::new(0.99, 0.999, 0.001, 0.5, 2);
    // Two spheres 0.5 m apart on x, small asymmetric y-offset for
    // symmetry break; body_b spun about +y so weld angular rows must
    // pick up the transfer.
    let ba = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(-0.25, 0.01, 1.0),
        Quat::IDENTITY,
    ));
    let bb = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.25, -0.01, 1.0),
        Quat::IDENTITY,
    ));
    w.bodies[bb].angular_velocity_body = Vec3::new(0.0, 1.5, 0.0);
    w.equalities.push(Equality::Weld {
        body_a: Some(ba),
        body_b: Some(bb),
        anchor_a: Vec3::new(0.25, 0.0, 0.0),
        anchor_b: Vec3::new(-0.25, 0.0, 0.0),
        relative_orientation: Quat::IDENTITY,
        solref,
        solimp,
    });
    w
}

fn distance_scene() -> World {
    let mut w = World::new();
    w.dt = 0.005;
    w.gravity = Vec3::ZERO;
    w.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 40,
        cone: ConeKind::Pyramidal,
    };
    // Orbiting-pair scene with asymmetric tangential velocities to
    // break any spin-symmetric bug.
    let ba = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(-0.5, 0.01, 0.0),
        Quat::IDENTITY,
    ));
    let bb = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.5, -0.01, 0.0),
        Quat::IDENTITY,
    ));
    w.bodies[ba].linear_velocity = Vec3::new(0.0, 0.3, 0.02);
    w.bodies[bb].linear_velocity = Vec3::new(0.0, -0.3, -0.02);
    w.equalities.push(Equality::Distance {
        body_a: Some(ba),
        body_b: Some(bb),
        anchor_a: Vec3::ZERO,
        anchor_b: Vec3::ZERO,
        distance: 1.0,
        solref: SolRef::new(0.01, 1.0),
        solimp: SolImp::new(0.99, 0.999, 0.001, 0.5, 2),
    });
    w
}

fn condim_rolling_scene() -> World {
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
    plane.condim = 6;
    plane.rolling_friction = 0.1;
    w.add_geom(plane);
    let radius = 0.1;
    // 1 cm y-offset for symmetry break.
    let bi = w.add_body(Body::solid_sphere(
        1.0,
        radius,
        Vec3::new(0.0, 0.01, radius),
        Quat::IDENTITY,
    ));
    // Rolling without slipping: v = ω × r → v_x = 1, ω_y = 10.
    w.bodies[bi].linear_velocity = Vec3::new(1.0, 0.0, 0.0);
    w.bodies[bi].angular_velocity_body = Vec3::new(0.0, 10.0, 0.0);
    let mut sphere = Geom::sphere(bi, radius, Vec3::ZERO, 0.9);
    sphere.solref = solref;
    sphere.solimp = solimp;
    sphere.condim = 6;
    sphere.rolling_friction = 0.1;
    w.add_geom(sphere);
    w
}

fn condim_torsional_scene() -> World {
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
    plane.condim = 4;
    plane.torsional_friction = 0.5;
    w.add_geom(plane);
    // Small x-offset for symmetry break.
    let bi = w.add_body(Body::solid_sphere(
        1.0,
        0.1,
        Vec3::new(0.01, 0.0, 0.5),
        Quat::IDENTITY,
    ));
    let mut sphere = Geom::sphere(bi, 0.1, Vec3::ZERO, 0.8);
    sphere.solref = solref;
    sphere.solimp = solimp;
    sphere.condim = 4;
    sphere.torsional_friction = 0.5;
    w.add_geom(sphere);
    w.bodies[bi].angular_velocity_body = Vec3::new(0.0, 0.0, 8.0);
    w
}

fn body_snapshot(bodies: &[Body]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bodies.len() * F32_PER_BODY * 4);
    for body in bodies {
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

fn tree_snapshot(w: &World) -> Vec<u8> {
    let mut out = Vec::new();
    for t in &w.trees {
        for v in &t.q {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for v in &t.qdot {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn record_bodies(world: &mut World) -> Vec<u8> {
    let mut out = body_snapshot(&world.bodies);
    for _ in 0..50 {
        world.step();
    }
    out.extend_from_slice(&body_snapshot(&world.bodies));
    for _ in 50..200 {
        world.step();
    }
    out.extend_from_slice(&body_snapshot(&world.bodies));
    for _ in 200..1000 {
        world.step();
    }
    out.extend_from_slice(&body_snapshot(&world.bodies));
    out
}

fn record_trees(world: &mut World) -> Vec<u8> {
    let mut out = tree_snapshot(world);
    for _ in 0..50 {
        world.step();
    }
    out.extend_from_slice(&tree_snapshot(world));
    for _ in 50..200 {
        world.step();
    }
    out.extend_from_slice(&tree_snapshot(world));
    for _ in 200..1000 {
        world.step();
    }
    out.extend_from_slice(&tree_snapshot(world));
    out
}

const CONNECT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/equality_connect.bin"
);
const COUPLING_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/equality_coupling.bin"
);
const TORSIONAL_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/condim_torsional.bin"
);
const WELD_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/equality_weld.bin"
);
const DISTANCE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/equality_distance.bin"
);
const ROLLING_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/condim_rolling.bin"
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
fn connect_pair_golden_byte_identical() {
    let mut w = connect_pair_scene();
    let bytes = record_bodies(&mut w);
    assert_golden(CONNECT_PATH, &bytes);
}

#[test]
fn coupling_golden_byte_identical() {
    let mut w = coupling_scene();
    let bytes = record_trees(&mut w);
    assert_golden(COUPLING_PATH, &bytes);
}

#[test]
fn condim_torsional_golden_byte_identical() {
    let mut w = condim_torsional_scene();
    let bytes = record_bodies(&mut w);
    assert_golden(TORSIONAL_PATH, &bytes);
}

#[test]
fn weld_golden_byte_identical() {
    let mut w = weld_scene();
    let bytes = record_bodies(&mut w);
    assert_golden(WELD_PATH, &bytes);
}

#[test]
fn distance_golden_byte_identical() {
    let mut w = distance_scene();
    let bytes = record_bodies(&mut w);
    assert_golden(DISTANCE_PATH, &bytes);
}

#[test]
fn condim_rolling_golden_byte_identical() {
    let mut w = condim_rolling_scene();
    let bytes = record_bodies(&mut w);
    assert_golden(ROLLING_PATH, &bytes);
}

/// Regenerate all six v1-tier-5 goldens. Ignored so it does not run
/// in CI. macOS-aarch64 only — see the panic in the standing
/// `regenerate_golden` test for the rationale.
#[test]
#[ignore]
fn regenerate_equality_goldens() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_equality_goldens may only run on the reference host \
             (macOS aarch64); refusing to overwrite the tracked bytes on \
             {} / {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }
    for (path, bytes) in [
        (CONNECT_PATH, record_bodies(&mut connect_pair_scene())),
        (COUPLING_PATH, record_trees(&mut coupling_scene())),
        (TORSIONAL_PATH, record_bodies(&mut condim_torsional_scene())),
        (WELD_PATH, record_bodies(&mut weld_scene())),
        (DISTANCE_PATH, record_bodies(&mut distance_scene())),
        (ROLLING_PATH, record_bodies(&mut condim_rolling_scene())),
    ] {
        let dir = std::path::Path::new(path).parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(path, &bytes).unwrap();
        println!("wrote {} bytes to {path}", bytes.len());
    }
}
