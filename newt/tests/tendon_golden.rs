//! Byte-identical golden trajectory for v2 tier 3 tendons: two coupled
//! pendulums with a fixed tendon (motor-driven) and a slide-hanging mass
//! on a spatial tendon (spring + limit). Snapshotted at steps 0, 300,
//! 700, 1500. Same macOS-aarch64 reference convention as the tier-3
//! golden.
//!
//! Symmetry break (per the tier-3 lesson):
//!   • pendulum arms differ (0.9 vs 1.1); masses (0.8, 1.2)
//!   • motor gear = 0.3 (not 1), asymmetric coef pair [1, -1]
//!   • spatial-tendon slide uses a hand-picked stiffness (137.3) so the
//!     equilibrium isn't a round number
//!   • tendon range [0.0, 1.03] — mass overshoots and settles ON the
//!     limit; the wrap discriminator activates
//!
//! Discriminating mutants (any of these would flip the trajectory):
//!   • fixed-tendon J^T applied without the coef (drops both drives)
//!   • fixed-tendon coef sign inverted on either joint
//!   • spatial-tendon site Jacobian mis-signed
//!   • solver-row escape-convention flip on the tendon limit
//!   • tendon-actuator torque going through the joint path (double-count)

use newt::actuator::Actuator;
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::solver::{SolverConfig, SolverMode};
use newt::tendon::{FixedTendonJoint, SpatialTendonSite, Tendon};
use newt::tree::{Link, Tree, forward_kinematics};
use newt::world::World;

fn snapshot_tree(tree: &Tree) -> Vec<u8> {
    let mut out = Vec::new();
    for v in tree
        .q
        .iter()
        .chain(tree.qdot.iter())
        .chain(tree.qfrc_applied.iter())
    {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for a in &tree.actuators {
        out.extend_from_slice(&a.act.to_le_bytes());
    }
    // Include current tendon lengths — pins tendon-Jacobian bytes so a
    // wrong-frame axis silently flipping length reads gets caught.
    let poses = forward_kinematics(tree);
    for tendon in &tree.tendons {
        let kin = newt::tendon::tendon_kinematics(tendon, tree, &poses);
        out.extend_from_slice(&kin.length.to_le_bytes());
        out.extend_from_slice(&kin.velocity.to_le_bytes());
    }
    out
}

fn build_scene() -> World {
    let mut world = World::new();
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.solver = SolverConfig {
        mode: SolverMode::Pgs,
        iterations: 30,
        ..SolverConfig::DEFAULT
    };
    // Tree 1: two coupled pendulums (fixed tendon + motor).
    let mut t1 = Tree::new();
    t1.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    t1.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 0.3,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(-0.4, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.9), Quat::IDENTITY),
        0.8,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    t1.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: Vec3::X,
            range: None,
            damping: 0.3,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(0.4, 0.0, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 1.1), Quat::IDENTITY),
        1.2,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    let mut coup = Tendon::fixed(vec![
        FixedTendonJoint { link: 1, coef: 1.0 },
        FixedTendonJoint {
            link: 2,
            coef: -1.0,
        },
    ]);
    coup.springlength = Some(0.0);
    coup.stiffness = 47.0;
    coup.damping = 8.0;
    t1.add_tendon(coup);
    let motor = Actuator::motor(0, 0.3, 0.0).on_tendon(0);
    let mid = t1.add_actuator(motor);
    t1.set_actuator_target(mid, 5.0);
    t1.set_hinge_angle(1, 0.25);
    t1.set_hinge_angle(2, -0.10);
    world.add_tree(t1);

    // Tree 2: slide-hanging mass with a spatial tendon (spring + limit).
    let mut t2 = Tree::new();
    t2.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    t2.push_link(Link::new(
        Some(0),
        JointKind::Slide {
            axis: Vec3::new(0.0, 0.0, -1.0),
            range: None,
            damping: 3.0,
            armature: 0.0,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.5,
        Mat3::diag(1e-3, 1e-3, 1e-3),
    ));
    let mut cable = Tendon::spatial(
        vec![
            SpatialTendonSite {
                link: Some(0),
                position_local: Vec3::ZERO,
            },
            SpatialTendonSite {
                link: Some(1),
                position_local: Vec3::ZERO,
            },
        ],
        vec![None],
    );
    cable.springlength = Some(0.6);
    cable.stiffness = 137.3; // hand-picked non-round for symmetry break
    cable.range = Some((0.0, 1.03));
    t2.add_tendon(cable);
    t2.set_slide_position(1, 0.5);
    world.add_tree(t2);
    world
}

fn produce_golden_bytes() -> Vec<u8> {
    let mut world = build_scene();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&snapshot_tree(&world.trees[0]));
    bytes.extend_from_slice(&snapshot_tree(&world.trees[1]));
    for _ in 0..300 {
        world.step();
    }
    bytes.extend_from_slice(&snapshot_tree(&world.trees[0]));
    bytes.extend_from_slice(&snapshot_tree(&world.trees[1]));
    // Change motor target mid-run.
    world.trees[0].set_actuator_target(0, -8.0);
    for _ in 0..400 {
        world.step();
    }
    bytes.extend_from_slice(&snapshot_tree(&world.trees[0]));
    bytes.extend_from_slice(&snapshot_tree(&world.trees[1]));
    for _ in 0..800 {
        world.step();
    }
    bytes.extend_from_slice(&snapshot_tree(&world.trees[0]));
    bytes.extend_from_slice(&snapshot_tree(&world.trees[1]));
    bytes
}

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/tendon_scene.bin"
);

#[test]
fn tendon_golden_trajectory_is_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect(
        "golden file missing — run the ignored `regenerate_tendon_golden` \
         test on macOS to produce it, then commit",
    );
    let actual = produce_golden_bytes();
    assert_eq!(
        expected.len(),
        actual.len(),
        "golden byte length mismatch: expected {} got {}",
        expected.len(),
        actual.len()
    );
    if expected != actual {
        let first_diff = expected
            .iter()
            .zip(actual.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "tendon golden mismatch; first byte diff at offset {first_diff}. \
             Layout: per-tree snapshot (q + qdot + qfrc + act + per-tendon (L, Ldot)) \
             × 2 trees × 4 samples."
        );
    }
}

/// Regenerate the golden. macOS-aarch64 only — same guard rationale as
/// the tier-3 golden (`tests/joints_golden.rs`).
#[test]
#[ignore]
fn regenerate_tendon_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_tendon_golden may only run on the reference host \
             (macOS aarch64); refusing to overwrite the tracked bytes on \
             {} / {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }
    let bytes = produce_golden_bytes();
    let dir = std::path::Path::new(GOLDEN_PATH).parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(GOLDEN_PATH, &bytes).unwrap();
    println!("wrote {} bytes to {GOLDEN_PATH}", bytes.len());
}
