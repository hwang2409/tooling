//! Symmetry-broken macOS reference golden for muscle activation and FLV state.

use newt::actuator::Actuator;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, rk4_step};

const GOLDEN_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/muscle.bin");

fn build_tree() -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::IDENTITY,
    ));
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.31), Quat::IDENTITY),
        0.8,
        Mat3::diag(0.04, 0.04, 0.001),
    ));
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::new(1.0, 0.15, 0.0).normalize()),
        (Vec3::new(0.0, 0.0, -0.31), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.23), Quat::IDENTITY),
        1.3,
        Mat3::diag(0.02, 0.03, 0.001),
    ));
    let first = [0.72, 1.08, 1.0, 42.0, 0.45, 1.55, 1.4, 1.2, 1.18];
    let second = [0.68, 1.12, 0.7, 31.0, 0.55, 1.7, 1.3, 1.4, 1.23];
    tree.add_actuator(Actuator::muscle(
        1,
        first,
        first,
        [-0.8, 0.9],
        0.8,
        -1.0,
        [0.013, 0.047, 0.08],
        Some((0.0, 1.0)),
        None,
    ));
    tree.add_actuator(Actuator::muscle(
        2,
        second,
        second,
        [-0.4, 0.7],
        1.2,
        0.7,
        [0.021, 0.061, 0.03],
        Some((0.0, 1.0)),
        None,
    ));
    tree.set_hinge_angle(1, 0.37);
    tree.set_hinge_angle(2, -0.22);
    tree
}

fn snapshot(tree: &Tree, bytes: &mut Vec<u8>) {
    for value in tree.q.iter().chain(&tree.qdot) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    for actuator in &tree.actuators {
        bytes.extend_from_slice(&actuator.act.to_le_bytes());
    }
}

fn produce() -> Vec<u8> {
    let mut tree = build_tree();
    let mut bytes = Vec::new();
    snapshot(&tree, &mut bytes);
    for actuator in &mut tree.actuators {
        actuator.ctrl = 0.83;
    }
    for _ in 0..300 {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -9.81), 0.003, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 3]
        });
    }
    snapshot(&tree, &mut bytes);
    tree.actuators[0].ctrl = 0.17;
    tree.actuators[1].ctrl = 0.61;
    for _ in 0..300 {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -9.81), 0.003, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 3]
        });
    }
    snapshot(&tree, &mut bytes);
    bytes
}

#[test]
fn muscle_golden_is_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect("muscle golden missing");
    assert_eq!(expected, produce(), "muscle golden changed");
}

#[test]
#[ignore]
fn regenerate_muscle_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!("muscle golden regeneration is macOS-aarch64 only");
    }
    std::fs::write(GOLDEN_PATH, produce()).expect("write muscle golden");
}
