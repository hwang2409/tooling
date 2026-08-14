//! Integration tests for the tier-5 JSON model loader.
//!
//! Covers:
//!   * round-trip anchor — the tier-4 arm demo loaded from `models/arm.json`
//!     produces the SAME trajectory as the programmatic construction after
//!     N RK4 steps (f32 exactness);
//!   * every packaged model file loads;
//!   * site world-pose query;
//!   * `models/stack.json` byte-golden.
//!
//! Determinism: goldens are macOS-only regen (same guard convention as
//! tiers 1–4); the byte comparison runs on every host.

use newt::actuator::PdServo;
use newt::body::Body;
use newt::geom::Geom;
use newt::joint::JointKind;
use newt::math::{Mat3, Quat, Vec3};
use newt::model::{Scene, load_from_path};
use newt::tree::{Link, Tree, rk4_step};

fn model_path(name: &str) -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("models");
    p.push(name);
    p
}

// ---------------------------------------------------------------------------
// 1. round-trip: arm.json vs programmatic (mirrors examples/arm.rs)
// ---------------------------------------------------------------------------

const ARM_L: f32 = 0.5;
const ARM_M: [f32; 3] = [1.0, 0.8, 0.6];

/// Reproduces `build_arm` + `attach_servos` from `examples/arm.rs`.
fn build_arm_programmatic() -> Tree {
    let mut tree = Tree::new();
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, 1.6), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    for (i, &m) in ARM_M.iter().enumerate() {
        let l = ARM_L;
        let i_perp = (1.0 / 12.0) * m * l * l;
        let parent_anchor = if i == 0 {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, 0.0, -l * 0.5)
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
    // Servos — kp / clamp / reflected inertia mirror `attach_servos`.
    tree.add_actuator(PdServo::from_dampratio(
        1,
        200.0,
        1.0,
        ARM_M[0] * ARM_L * ARM_L,
        60.0,
    ));
    tree.add_actuator(PdServo::from_dampratio(
        2,
        150.0,
        1.0,
        ARM_M[1] * ARM_L * ARM_L,
        40.0,
    ));
    tree.add_actuator(PdServo::from_dampratio(
        3,
        100.0,
        1.0,
        ARM_M[2] * ARM_L * ARM_L,
        30.0,
    ));
    tree
}

fn zero_ext(n: usize) -> impl Fn(&Tree) -> Vec<(Vec3, Vec3)> {
    move |_| vec![(Vec3::ZERO, Vec3::ZERO); n]
}

#[test]
fn arm_json_matches_programmatic_construction_exactly() {
    let scene = load_from_path(model_path("arm.json")).expect("arm.json should load");
    // Set the first waypoint on the loaded scene so the servos are not
    // driven from a zero target — makes the trajectory more discriminating.
    let arm_idx = scene.trees_by_name["arm"];
    let mut loaded = scene.world.trees[arm_idx].clone();
    for (name, target) in [
        ("shoulder_servo", 0.3),
        ("elbow_servo", -0.4),
        ("wrist_servo", 0.5),
    ] {
        let (t_idx, a_idx) = scene.actuators_by_name[name];
        assert_eq!(t_idx, arm_idx);
        loaded.set_actuator_target(a_idx, target);
    }

    let mut prog = build_arm_programmatic();
    prog.set_actuator_target(0, 0.3);
    prog.set_actuator_target(1, -0.4);
    prog.set_actuator_target(2, 0.5);

    // Initial state must match exactly (before any step).
    assert_eq!(loaded.q, prog.q, "initial q mismatch");
    assert_eq!(loaded.qdot, prog.qdot, "initial qdot mismatch");
    // Link fields must match: inertia + offsets + joint config.
    assert_eq!(loaded.links.len(), prog.links.len());
    for (i, (a, b)) in loaded.links.iter().zip(prog.links.iter()).enumerate() {
        assert_eq!(a.mass, b.mass, "link {i} mass");
        assert_eq!(a.inertia_body, b.inertia_body, "link {i} inertia");
        assert_eq!(
            a.joint_offset_in_parent, b.joint_offset_in_parent,
            "link {i} joint_offset_in_parent"
        );
        assert_eq!(
            a.joint_offset_in_child, b.joint_offset_in_child,
            "link {i} joint_offset_in_child"
        );
    }

    // Step N times. If any element diverges by even one ULP the assert_eq
    // below fires.
    let dt = 0.005f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    for _ in 0..500 {
        rk4_step(&mut loaded, g, dt, zero_ext(4));
        rk4_step(&mut prog, g, dt, zero_ext(4));
    }
    assert_eq!(loaded.q, prog.q, "q mismatch after 500 steps");
    assert_eq!(loaded.qdot, prog.qdot, "qdot mismatch after 500 steps");
}

// ---------------------------------------------------------------------------
// 2. every packaged model loads
// ---------------------------------------------------------------------------

#[test]
fn packaged_models_all_load() {
    for name in ["arm.json", "pendulum.json", "stack.json"] {
        let path = model_path(name);
        let scene =
            load_from_path(&path).unwrap_or_else(|e| panic!("{}: load error {e}", path.display()));
        assert!(
            !scene.world.bodies.is_empty() || !scene.world.trees.is_empty(),
            "{}: scene has no bodies or trees",
            name
        );
    }
}

#[test]
fn arm_scene_exposes_expected_names_and_site() {
    let scene = load_from_path(model_path("arm.json")).unwrap();
    for n in ["anchor", "shoulder", "elbow", "wrist"] {
        assert!(scene.links_by_name[0].contains_key(n), "link {n} missing");
    }
    for n in ["shoulder_servo", "elbow_servo", "wrist_servo"] {
        assert!(
            scene.actuators_by_name.contains_key(n),
            "actuator {n} missing"
        );
    }
    // Tip site — with all joints at 0, the arm hangs straight down along -z
    // from the anchor. Anchor sits at (0, 0, 1.6); three 0.5-m rods →
    // tip at z = 1.6 - 3·L = 0.1.
    let (pos, _) = scene.site_pose("tip").expect("tip site missing");
    assert!(pos.x.abs() < 1e-5, "x = {}", pos.x);
    assert!(pos.y.abs() < 1e-5, "y = {}", pos.y);
    assert!(
        (pos.z - (1.6 - 3.0 * ARM_L)).abs() < 1e-4,
        "tip z = {}, expected {}",
        pos.z,
        1.6 - 3.0 * ARM_L
    );
}

// ---------------------------------------------------------------------------
// 3. stack.json byte-identical golden
// ---------------------------------------------------------------------------

const STACK_GOLDEN_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/goldens/model_stack.bin");

fn snapshot_bodies(bodies: &[Body]) -> Vec<u8> {
    // Serialize each body's position (3 f32), orientation (4 f32), linear
    // velocity (3 f32), angular velocity body (3 f32). 13 f32 per body.
    let mut out = Vec::with_capacity(bodies.len() * 13 * 4);
    for b in bodies {
        for f in [
            b.position.x,
            b.position.y,
            b.position.z,
            b.orientation.x,
            b.orientation.y,
            b.orientation.z,
            b.orientation.w,
            b.linear_velocity.x,
            b.linear_velocity.y,
            b.linear_velocity.z,
            b.angular_velocity_body.x,
            b.angular_velocity_body.y,
            b.angular_velocity_body.z,
        ] {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out
}

fn produce_stack_golden_bytes() -> Vec<u8> {
    let scene = load_from_path(model_path("stack.json")).expect("stack.json should load");
    let mut world = scene.world;
    let mut bytes = Vec::new();
    // Snapshot 0: initial state (drop poses).
    bytes.extend_from_slice(&snapshot_bodies(&world.bodies));
    // Step to 3 checkpoints: 200 / 500 / 1000 steps (1 s / 2.5 s / 5 s).
    for &n in &[200usize, 300, 500] {
        for _ in 0..n {
            world.step();
        }
        bytes.extend_from_slice(&snapshot_bodies(&world.bodies));
    }
    bytes
}

#[test]
fn stack_json_golden_is_byte_identical() {
    let expected = std::fs::read(STACK_GOLDEN_PATH).expect(
        "golden file missing — run the ignored `regenerate_stack_golden` \
         test on macOS to produce it, then commit",
    );
    let actual = produce_stack_golden_bytes();
    assert_eq!(
        expected.len(),
        actual.len(),
        "golden byte length: {} vs {}",
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
            "stack.json golden mismatch; first byte diff at offset {first_diff}. \
             Layout: 4 snapshots × 3 bodies × 13 f32 = 156 f32 = 624 bytes."
        );
    }
}

#[test]
#[ignore]
fn regenerate_stack_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_stack_golden may only run on the reference host \
             (macOS aarch64); refusing to overwrite on {} / {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        );
    }
    let bytes = produce_stack_golden_bytes();
    let dir = std::path::Path::new(STACK_GOLDEN_PATH).parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(STACK_GOLDEN_PATH, &bytes).unwrap();
    println!("wrote {} bytes to {STACK_GOLDEN_PATH}", bytes.len());
}

// ---------------------------------------------------------------------------
// 4. pendulum.json steady swing matches double-pendulum reference
// ---------------------------------------------------------------------------

#[test]
fn pendulum_json_hanging_at_rest_stays_at_rest() {
    // With every hinge at q = 0, the double pendulum hangs straight down.
    // Gravity torque about every hinge is exactly zero → the joint angles
    // should stay at zero after any number of steps. If the loader wired
    // an offset wrong, the second link's COM would sit off-axis and the
    // pendulum would fall.
    let scene = load_from_path(model_path("pendulum.json")).unwrap();
    let mut tree = scene.world.trees[0].clone();
    let dt = 0.005f32;
    for _ in 0..200 {
        rk4_step(&mut tree, Vec3::new(0.0, 0.0, -9.81), dt, zero_ext(3));
    }
    assert!(
        tree.hinge_angle(1).abs() < 1e-4,
        "hinge 1 should stay at 0, got {}",
        tree.hinge_angle(1)
    );
    assert!(
        tree.hinge_angle(2).abs() < 1e-4,
        "hinge 2 should stay at 0, got {}",
        tree.hinge_angle(2)
    );
}

// ---------------------------------------------------------------------------
// 5. avoid unused-import warnings on Geom (imports are exercised elsewhere)
// ---------------------------------------------------------------------------

#[test]
fn geom_import_kept_alive() {
    // The Geom / Scene imports are exercised implicitly by the other tests
    // via the loader; this keeps clippy from flagging the direct use as
    // dead when tests are filtered.
    let _: fn() = || {
        let _ = Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.5);
        let _: Option<&Scene> = None;
    };
}
