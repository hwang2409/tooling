//! Mixed-joint golden trajectory: a single tree exercising every non-fixed
//! joint kind (Free root + Hinge + Slide + Ball) together. Serializes `(q,
//! qdot)` at fixed step counts and byte-compares against
//! `tests/goldens/joints_mixed.bin` — the deterministic cross-platform pin
//! for the v1 tier-1 joint set.
//!
//! # Symmetry breaking
//!
//! Following the tier-2/tier-3 lesson (symmetric scenes hide lever-arm
//! bugs), this scene deliberately breaks every symmetry:
//! - free root orientation off-axis (Rot((1, 0.3, -0.2), 0.4)) — no aligned
//!   principal axes with the world;
//! - hinge axis (0.7, 0.2, 0.15).normalize() — not a body axis;
//! - slide axis (0.3, 0.9, -0.1).normalize() — mixed direction;
//! - initial slide displacement and rate both nonzero;
//! - ball initial orientation and body-frame ω both nonzero and non-parallel;
//! - all masses distinct.
//!
//! Regen (macOS-aarch64 only, same guard as tiers 1-4):
//! ```text
//! cargo test --test joints_mixed_golden regenerate_mixed_joints_golden \
//!   -- --ignored --nocapture
//! ```

use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::spatial::SpatialMotion;
use newt::tree::{Link, Tree, rk4_step};

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/joints_mixed.bin"
);

fn zero_ext(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

fn build_mixed_tree() -> Tree {
    let mut tree = Tree::new();

    // 0. Free-root box. Off-axis initial pose + spin.
    tree.push_link(Link::new(
        None,
        JointKind::Free,
        (
            Vec3::new(0.1, -0.2, 1.4),
            Quat::from_axis_angle(Vec3::new(1.0, 0.3, -0.2).normalize(), 0.4),
        ),
        (Vec3::ZERO, Quat::IDENTITY),
        1.7,
        Mat3::diag(0.4, 0.3, 0.35),
    ));
    tree.set_free_root_velocity(SpatialMotion::new(
        Vec3::new(0.2, 0.15, -0.05), // ω_body
        Vec3::new(0.05, -0.1, 0.03), // v_body at COM
    ));

    // 1. Hinge child of the root: non-body-axis axis, non-trivial anchor.
    let hinge_axis = Vec3::new(0.7, 0.2, 0.15).normalize();
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(hinge_axis),
        (Vec3::new(0.2, 0.0, -0.15), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.3), Quat::IDENTITY),
        0.9,
        Mat3::diag(0.08, 0.08, 0.005),
    ));
    tree.set_hinge_angle(1, 0.4);
    tree.set_hinge_rate(1, -0.3);

    // 2. Slide child of the hinge: axis mixed (X-Y direction).
    let slide_axis = Vec3::new(0.3, 0.9, -0.1).normalize();
    tree.push_link(Link::new(
        Some(1),
        JointKind::slide(slide_axis),
        (Vec3::new(0.0, 0.0, -0.3), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        0.6,
        Mat3::diag(0.02, 0.02, 0.02),
    ));
    tree.set_slide_position(2, 0.15);
    tree.set_slide_rate(2, 0.25);

    // 3. Ball child of the slide: initial off-axis orientation + non-parallel ω.
    tree.push_link(Link::new(
        Some(2),
        JointKind::Ball {
            damping: 0.05,
            armature: 0.0,
        },
        (Vec3::new(0.05, 0.0, -0.1), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.25), Quat::IDENTITY),
        0.5,
        Mat3::diag(0.03, 0.03, 0.004),
    ));
    tree.set_ball_orientation(
        3,
        Quat::from_axis_angle(Vec3::X, 0.35) * Quat::from_axis_angle(Vec3::Z, 0.2),
    );
    tree.set_ball_omega(3, Vec3::new(0.3, 0.6, -0.4));

    tree
}

fn serialize_state(tree: &Tree) -> Vec<u8> {
    // Format: [nq (u32 LE)] [q...] [nv (u32 LE)] [qdot...] all little-endian
    // f32 for the arrays. Cross-platform byte-identical thanks to newt's
    // libm-free scalar policy.
    let mut out = Vec::with_capacity(4 + tree.q.len() * 4 + 4 + tree.qdot.len() * 4);
    out.extend_from_slice(&(tree.q.len() as u32).to_le_bytes());
    for &v in &tree.q {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(tree.qdot.len() as u32).to_le_bytes());
    for &v in &tree.qdot {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn produce_mixed_golden_bytes() -> Vec<u8> {
    let mut tree = build_mixed_tree();
    let dt = 0.005f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    let mut bytes = Vec::new();
    // Snapshot 0: initial state.
    bytes.extend_from_slice(&serialize_state(&tree));
    // Then snapshots after 100, 500, 1000 steps.
    for &(delta, _label) in &[(100usize, "100"), (400, "500"), (500, "1000")] {
        for _ in 0..delta {
            rk4_step(&mut tree, g, dt, zero_ext(4));
        }
        bytes.extend_from_slice(&serialize_state(&tree));
    }
    bytes
}

#[test]
fn joints_mixed_golden_is_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect(
        "mixed-joints golden missing — run the ignored \
         `regenerate_mixed_joints_golden` test on macOS to produce it, \
         then commit",
    );
    let actual = produce_mixed_golden_bytes();
    assert_eq!(
        expected.len(),
        actual.len(),
        "golden byte length: expected {}, got {}",
        expected.len(),
        actual.len()
    );
    if expected != actual {
        let first = expected
            .iter()
            .zip(actual.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "mixed-joints golden mismatch; first byte diff at offset {first}. \
             Layout per snapshot: [nq u32 LE][q f32 LE...][nv u32 LE][qdot f32 LE...]. \
             Free (nq=7,nv=6) + Hinge (nq=1,nv=1) + Slide (nq=1,nv=1) + Ball (nq=4,nv=3) \
             ⇒ nq = 13, nv = 11 per snapshot."
        );
    }
}

#[test]
#[ignore]
fn regenerate_mixed_joints_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_mixed_joints_golden may only run on the reference \
             host (macOS aarch64); refusing to overwrite on {} / {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }
    let bytes = produce_mixed_golden_bytes();
    let dir = std::path::Path::new(GOLDEN_PATH).parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(GOLDEN_PATH, &bytes).unwrap();
    println!("wrote {} bytes to {GOLDEN_PATH}", bytes.len());
}
