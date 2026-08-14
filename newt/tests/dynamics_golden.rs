//! Byte-identical golden for the CRB mass matrix and RNE bias vector on a
//! fixed mixed-joint tree state. Same macOS-aarch64 reference convention as
//! `tests/joints_golden.rs` and `tests/joints_mixed_golden.rs`: the
//! `regenerate_*` test is `#[ignore]` and guards on host so a stray
//! `--ignored` run on Linux cannot silently swap the reference bytes.
//!
//! Scene: fixed root + hinge (oblique axis) + slide (oblique axis) + ball
//! joint with a non-COM anchor. Symmetry broken on every DOF so a
//! zero-lever-arm mutant flips the golden — see the tier-2 lesson (a
//! symmetric scene can pass byte-identically under a lever-arm bug).

use newt::dynamics::{bias_forces, mass_matrix};
use newt::joint::{JointKind, JointLimit};
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree};

fn box_inertia(mass: f32, hx: f32, hy: f32, hz: f32) -> Mat3 {
    let hx2 = hx * hx;
    let hy2 = hy * hy;
    let hz2 = hz * hz;
    Mat3::diag(
        (mass / 3.0) * (hy2 + hz2),
        (mass / 3.0) * (hx2 + hz2),
        (mass / 3.0) * (hx2 + hy2),
    )
}

fn scene() -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    let axis1 = Vec3::new(1.0, 0.3, -0.2).normalize();
    tree.push_link(Link::new(
        Some(0),
        JointKind::Hinge {
            axis: axis1,
            range: None,
            damping: 0.0,
            armature: 0.02,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(0.05, 0.0, -0.1), Quat::IDENTITY),
        (Vec3::new(-0.15, 0.0, 0.3), Quat::IDENTITY),
        0.9,
        box_inertia(0.9, 0.15, 0.1, 0.3),
    ));
    let axis2 = Vec3::new(0.2, 1.0, 0.4).normalize();
    tree.push_link(Link::new(
        Some(1),
        JointKind::Slide {
            axis: axis2,
            range: None,
            damping: 0.0,
            armature: 0.05,
            limit: JointLimit::DEFAULT,
        },
        (Vec3::new(0.0, 0.0, -0.3), Quat::IDENTITY),
        (Vec3::new(0.1, -0.2, 0.15), Quat::IDENTITY),
        0.7,
        box_inertia(0.7, 0.1, 0.2, 0.15),
    ));
    tree.push_link(Link::new(
        Some(2),
        JointKind::Ball {
            damping: 0.0,
            armature: 0.01,
        },
        (Vec3::new(0.1, -0.2, -0.15), Quat::IDENTITY),
        (Vec3::new(-0.05, 0.1, 0.2), Quat::IDENTITY),
        0.5,
        Mat3::diag(0.02, 0.03, 0.04),
    ));

    // Fixed state with symmetry broken on every DOF.
    tree.set_hinge_angle(1, 0.35);
    tree.set_hinge_rate(1, 0.6);
    tree.set_slide_position(2, 0.12);
    tree.set_slide_rate(2, -0.4);
    let ball_q = Quat::from_axis_angle(Vec3::new(0.4, 1.0, -0.3).normalize(), 0.55);
    tree.set_ball_orientation(3, ball_q);
    tree.set_ball_omega(3, Vec3::new(0.3, -0.5, 0.7));
    tree
}

fn produce_golden_bytes() -> Vec<u8> {
    let tree = scene();
    let gravity = Vec3::new(0.0, 0.0, -9.81);
    let m = mass_matrix(&tree);
    let bias = bias_forces(&tree, gravity);
    // Layout: M (nv * nv f32, row-major) then bias (nv f32), all little-endian.
    let mut bytes = Vec::with_capacity((m.len() + bias.len()) * 4);
    for v in m.iter().chain(bias.iter()) {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/dynamics_mass_bias.bin"
);

#[test]
fn dynamics_mass_and_bias_are_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect(
        "golden file missing — run the ignored `regenerate_dynamics_golden` \
         test on macOS to produce it, then commit",
    );
    let actual = produce_golden_bytes();
    assert_eq!(
        expected.len(),
        actual.len(),
        "golden length mismatch: expected {} got {}",
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
            "golden mismatch; first byte diff at offset {first_diff}. \
             Layout: 25 f32 M (nv=5 = hinge + slide + ball, row-major) + \
             5 f32 bias = 30 f32 = 120 bytes."
        );
    }
}

/// Regenerate the golden. Ignored — refuses to run on any host other than
/// the macOS-aarch64 reference host.
#[test]
#[ignore]
fn regenerate_dynamics_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_dynamics_golden may only run on the reference host \
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
