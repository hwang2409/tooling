//! Byte-identical golden trajectory for a tier-3 scene: one 3-link chain
//! (fixed root) and one floating-base link with a swinging arm. Snapshotted
//! at steps 0, 100, 1000. Same macOS-aarch64 reference convention as tier-1
//! (`tests/golden.rs`) and tier-2 (`tests/contacts_golden.rs`).
//!
//! Symmetry break (per tier-2 lesson): mixed masses/lengths, initial
//! nontrivial angles + rates, hinge axes NOT axis-aligned on the floating
//! arm. A zero-lever-arm mutant flips this golden at snapshot 2.

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, rk4_step};

/// One tree's serialized state (q then qdot, little-endian f32s).
fn snapshot(tree: &Tree) -> Vec<u8> {
    let mut out = Vec::with_capacity((tree.q.len() + tree.qdot.len()) * 4);
    for v in tree.q.iter().chain(tree.qdot.iter()) {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn scene_chain() -> Tree {
    // 3-link chain: fixed root + three uniform rods with mixed masses and
    // lengths, initial angles broken from symmetry.
    let masses = [1.1f32, 0.7, 1.3];
    let lengths = [0.6f32, 0.4, 0.5];
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    for i in 0..3 {
        let l = lengths[i];
        let m = masses[i];
        let i_perp = (1.0 / 12.0) * m * l * l;
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, 0.0, -lengths[i - 1] * 0.5)
        };
        tree.push_link(Link::new(
            Some(i),
            JointKind::hinge(Vec3::X),
            (parent_anchor, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
            m,
            Mat3::diag(i_perp, i_perp, 1e-6),
        ));
    }
    tree.set_hinge_angle(1, 0.4);
    tree.set_hinge_angle(2, -0.3);
    tree.set_hinge_angle(3, 0.2);
    tree.set_hinge_rate(2, 0.5);
    tree
}

fn scene_floating() -> Tree {
    // Floating base + swinging arm, non-axis-aligned hinge axis, offset COM
    // position from the origin. No gravity for this tree — momentum-
    // conservation-driven trajectory.
    let mut tree = Tree::new();
    let hx = 0.3;
    let hy = 0.2;
    let hz = 0.15;
    let m0 = 2.0f32;
    let box_i = {
        let hx2 = hx * hx;
        let hy2 = hy * hy;
        let hz2 = hz * hz;
        let ixx = (m0 / 3.0) * (hy2 + hz2);
        let iyy = (m0 / 3.0) * (hx2 + hz2);
        let izz = (m0 / 3.0) * (hx2 + hy2);
        Mat3::diag(ixx, iyy, izz)
    };
    let q0 = Quat::from_axis_angle(Vec3::new(1.0, 0.4, -0.3), 0.6);
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (Vec3::new(0.5, -1.0, 4.5), q0),
        (Vec3::ZERO, Quat::IDENTITY),
        m0,
        box_i,
    ));
    let l = 0.8f32;
    let m1 = 0.7f32;
    let i_perp = (1.0 / 12.0) * m1 * l * l;
    let axis = Vec3::new(1.0, 0.3, 0.0).normalize();
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(axis),
        (Vec3::new(0.0, 0.0, -hz), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
        m1,
        Mat3::diag(i_perp, i_perp, 1e-4),
    ));
    tree.set_hinge_rate(1, 4.0);
    tree
}

fn produce_golden_bytes() -> Vec<u8> {
    let mut chain = scene_chain();
    let mut floating = scene_floating();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&snapshot(&chain));
    bytes.extend_from_slice(&snapshot(&floating));
    let dt = 0.005f32;
    let gravity_chain = Vec3::new(0.0, 0.0, -9.81);
    let gravity_floating = Vec3::ZERO;
    for _ in 0..100 {
        rk4_step(&mut chain, gravity_chain, dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 4]
        });
        rk4_step(&mut floating, gravity_floating, dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 2]
        });
    }
    bytes.extend_from_slice(&snapshot(&chain));
    bytes.extend_from_slice(&snapshot(&floating));
    for _ in 100..1000 {
        rk4_step(&mut chain, gravity_chain, dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 4]
        });
        rk4_step(&mut floating, gravity_floating, dt, |_| {
            vec![(Vec3::ZERO, Vec3::ZERO); 2]
        });
    }
    bytes.extend_from_slice(&snapshot(&chain));
    bytes.extend_from_slice(&snapshot(&floating));
    bytes
}

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/joints_chain_and_floating.bin"
);

#[test]
fn joints_golden_trajectory_is_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect(
        "golden file missing — run the ignored `regenerate_joints_golden` \
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
            "golden trajectory mismatch; first byte diff at offset {first_diff}. \
             Layout: chain q(1) + chain qdot(3) + float q(9) + float qdot(8) \
             = 21 f32 per snapshot × 4 snapshots"
        );
    }
}

/// Regenerate the golden file. Ignored — only run on the macOS-aarch64
/// reference host (guard panics on any other host so an accidental
/// `--ignored` run cannot silently swap the reference).
#[test]
#[ignore]
fn regenerate_joints_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_joints_golden may only run on the reference host \
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
