//! Byte-identical golden trajectory for the v2 tier 2 arm: three hinges,
//! three PD servos following a target sequence, a filter actuator sharing
//! the shoulder hinge (activation state integrated per step), a
//! constant motor-torque injection, and a persistent world-frame wrench —
//! every actuation channel active at once. Snapshotted at steps 0, 500,
//! 1000, and 1500. Same macOS-aarch64 reference convention as the tier-3
//! golden (`tests/joints_golden.rs`).
//!
//! Symmetry break (per the tier-3 lesson):
//!   • mixed masses `[1.1, 0.7, 0.9]` and lengths `[0.5, 0.4, 0.35]`
//!   • middle hinge axis `(1, 0.2, 0).normalize()` — not principal
//!   • target sequence is asymmetric across the two waypoints
//!   • persistent wrench is non-axis-aligned; direct torque non-zero
//!   • filter actuator ctrl steps mid-run — its `act` bytes discriminate
//!     an activation-integration mutation (wrong tau, wrong sign, skipped
//!     step-end update)
//!
//! A zero-lever-arm, wrong-axis, wrong-channel, or activation-integration
//! mutant flips this golden at snapshot 2 or later.

use newt::actuator::{Actuator, BiasType, DynType, GainType};
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::tree::{Link, Tree, rk4_step};

/// Serialize (q, qdot, qfrc_applied, [actuator.act ...]) as little-endian
/// f32 bytes. Layout:
///
///   q            nq f32   integrated joint state
///   qdot         nv f32   integrated joint rates
///   qfrc_applied nv f32   user forces (ZOH each step)
///   act          na f32   activation state for every actuator (0.0 for
///                         non-filter actuators; pins the layout so a
///                         change to activation storage cannot silently
///                         land)
///
/// Applied wrenches and actuator ctrl values are STATIC user inputs
/// across a snapshot window; the integrated state IS the discriminator.
fn snapshot(tree: &Tree) -> Vec<u8> {
    let n = tree.q.len() + tree.qdot.len() + tree.qfrc_applied.len() + tree.actuators.len();
    let mut out = Vec::with_capacity(n * 4);
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
    out
}

fn build_arm() -> Tree {
    let masses = [1.1f32, 0.7, 0.9];
    let lengths = [0.5f32, 0.4, 0.35];
    let axes = [Vec3::X, Vec3::new(1.0, 0.2, 0.0).normalize(), Vec3::X];
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
            JointKind::hinge(axes[i]),
            (parent_anchor, Quat::IDENTITY),
            (Vec3::new(0.0, 0.0, l * 0.5), Quat::IDENTITY),
            m,
            Mat3::diag(i_perp, i_perp, 1e-6),
        ));
    }
    tree
}

fn produce_golden_bytes() -> Vec<u8> {
    let mut tree = build_arm();

    // Servos: distinct kp/dampratio/clamp per joint so a mis-routed servo
    // (indexing bug) shifts a snapshot.
    let s1 = Actuator::position_from_dampratio(
        1, /*kp*/ 80.0, /*ζ*/ 0.8, /*I_ref*/ 0.3, /*clamp*/ 10.0,
    );
    let s2 = Actuator::position_from_dampratio(2, 60.0, 0.9, 0.15, 8.0);
    let s3 = Actuator::position_from_dampratio(3, 40.0, 1.0, 0.10, 6.0);
    let a1 = tree.add_actuator(s1);
    let a2 = tree.add_actuator(s2);
    let a3 = tree.add_actuator(s3);

    // Filter actuator sharing the shoulder hinge (link 1): a general
    // motor-shape with a first-order activation filter (tau=0.15 s) so
    // its `act` state evolves each step. Small gain so the extra torque
    // doesn't overwhelm the PD servo above — the point is to pin the
    // ACTIVATION LAYOUT byte-identically, not to change the arm's gross
    // motion. `ctrl` gets rewritten each waypoint (below) so the
    // activation transient is visible in every post-step-0 snapshot.
    let filter = Actuator::general(
        1,
        GainType::Fixed,
        [0.5, 0.0, 0.0],
        BiasType::None,
        [0.0, 0.0, 0.0],
        1.0,
        DynType::Filter,
        [0.15],
        None,
        None,
    );
    let af = tree.add_actuator(filter);

    // Persistent world-frame wrench on the tip link (link 3): small +y
    // force + small -x torque. Non-axis-aligned to break symmetry.
    tree.set_link_wrench(3, Vec3::new(0.0, 0.4, 0.0), Vec3::new(-0.05, 0.0, 0.02));
    // Persistent motor torque on link 1 via qfrc_applied (clamped to ±3):
    // this exercises the direct-torque channel alongside the servos.
    tree.set_joint_torque_clamped(1, 0.15, 3.0);

    let dt = 0.005f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    let mut bytes = Vec::new();

    // Snapshot 0: initial state.
    bytes.extend_from_slice(&snapshot(&tree));

    // Waypoint 1: reach positions (0.3, -0.4, 0.5). Filter ctrl steps
    // to 0.8 so its `act` state ramps in over ~5·tau = 0.75 s.
    tree.set_actuator_target(a1, 0.3);
    tree.set_actuator_target(a2, -0.4);
    tree.set_actuator_target(a3, 0.5);
    tree.set_actuator_target(af, 0.8);
    for _ in 0..500 {
        rk4_step(&mut tree, g, dt, |_| vec![(Vec3::ZERO, Vec3::ZERO); 4]);
    }
    // Snapshot 1: after 500 steps at waypoint 1.
    bytes.extend_from_slice(&snapshot(&tree));

    // Waypoint 2: switch targets. Filter ctrl steps to -0.4 so its `act`
    // has to cross zero — asymmetric transient a sign-flipped
    // integrator would blow.
    tree.set_actuator_target(a1, 0.1);
    tree.set_actuator_target(a2, 0.6);
    tree.set_actuator_target(a3, -0.3);
    tree.set_actuator_target(af, -0.4);
    for _ in 0..500 {
        rk4_step(&mut tree, g, dt, |_| vec![(Vec3::ZERO, Vec3::ZERO); 4]);
    }
    // Snapshot 2: after 1000 total steps.
    bytes.extend_from_slice(&snapshot(&tree));

    // Waypoint 3: settle to zero.
    tree.set_actuator_target(a1, 0.0);
    tree.set_actuator_target(a2, 0.0);
    tree.set_actuator_target(a3, 0.0);
    tree.set_actuator_target(af, 0.0);
    for _ in 0..500 {
        rk4_step(&mut tree, g, dt, |_| vec![(Vec3::ZERO, Vec3::ZERO); 4]);
    }
    // Snapshot 3: after 1500 total steps.
    bytes.extend_from_slice(&snapshot(&tree));

    bytes
}

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/actuators_arm_waypoints.bin"
);

#[test]
fn actuators_golden_trajectory_is_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect(
        "golden file missing — run the ignored `regenerate_actuators_golden` \
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
             Layout: (q(3) + qdot(3) + qfrc_applied(3) + act(4)) f32 per snapshot × 4 \
             snapshots = 13 f32 × 4 = 52 f32 = 208 bytes. The 4 actuators are \
             (shoulder PD, elbow PD, wrist PD, shoulder filter)."
        );
    }
}

/// Regenerate the golden. macOS-aarch64 only — see `tests/joints_golden.rs`
/// for the same guard rationale.
#[test]
#[ignore]
fn regenerate_actuators_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_actuators_golden may only run on the reference host \
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
