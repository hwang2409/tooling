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
use newt::world::World;

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
// 2b. round-trip: stack.json vs programmatic (mirrors examples/stack.rs)
// ---------------------------------------------------------------------------

/// Programmatic equivalent of `models/stack.json`. Same masses, same drop
/// positions, same +0.02 middle-box shift + top-box initial spin (the
/// tier-2 symmetry break carried across).
///
/// Kept side-by-side with `models/stack.json` so a divergence between the
/// tier-2 free-body path and the loader is caught by the byte-identical
/// assert below — this is the missing gate that let a broken stack
/// symmetry-break slip through the first pass on this ticket.
fn build_stack_programmatic() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.6));
    let half = Vec3::new(0.35, 0.35, 0.35);
    let drops: [(f32, Vec3, Vec3); 3] = [
        (1.2, Vec3::new(0.0, 0.0, 0.5), Vec3::ZERO),
        (0.9, Vec3::new(0.02, 0.0, 1.7), Vec3::ZERO),
        (1.5, Vec3::new(0.0, 0.0, 2.9), Vec3::new(0.0, 0.3, 0.0)),
    ];
    for (mass, pos, omega_body) in drops {
        let mut b = Body::solid_box(mass, half, pos, Quat::IDENTITY);
        b.angular_velocity_body = omega_body;
        let idx = world.add_body(b);
        world.add_geom(Geom::r#box(idx, half, Vec3::ZERO, Quat::IDENTITY, 0.6));
    }
    world
}

#[test]
fn stack_json_matches_programmatic_construction_exactly() {
    let scene = load_from_path(model_path("stack.json")).expect("stack.json should load");
    let mut loaded = scene.world;
    let mut prog = build_stack_programmatic();

    // Every mutable body field must match at step 0.
    assert_eq!(loaded.bodies.len(), prog.bodies.len());
    for (i, (a, b)) in loaded.bodies.iter().zip(prog.bodies.iter()).enumerate() {
        assert_eq!(a.mass, b.mass, "body {i} mass");
        assert_eq!(a.inertia_body, b.inertia_body, "body {i} inertia");
        assert_eq!(a.position, b.position, "body {i} position");
        assert_eq!(a.orientation, b.orientation, "body {i} orientation");
        assert_eq!(a.linear_velocity, b.linear_velocity, "body {i} v_lin");
        assert_eq!(
            a.angular_velocity_body, b.angular_velocity_body,
            "body {i} omega_body"
        );
    }
    // Every geom must match too (same shapes + attachments) — a mis-mapped
    // body index in the loader would blow this even before any step.
    assert_eq!(loaded.geoms.len(), prog.geoms.len());
    for (i, (a, b)) in loaded.geoms.iter().zip(prog.geoms.iter()).enumerate() {
        assert_eq!(a.attachment(), b.attachment(), "geom {i} attachment");
        assert_eq!(a.shape, b.shape, "geom {i} shape");
        assert_eq!(a.local_offset, b.local_offset, "geom {i} local_offset");
        assert_eq!(a.friction, b.friction, "geom {i} friction");
    }

    // Step both worlds 500 steps and require byte-identical body state at
    // every checkpoint. Box-box contacts DO participate here (the tier-2
    // reference in `stack.rs` settles the middle and top boxes on top of
    // the bottom one at z ≈ 1.05 / 1.75); if the loader dropped box-box
    // pairs while keeping box-plane, all three boxes would collapse onto
    // the plane at z ≈ 0.35 and this would blow.
    for step in 0..500 {
        loaded.step();
        prog.step();
        for (i, (a, b)) in loaded.bodies.iter().zip(prog.bodies.iter()).enumerate() {
            assert_eq!(
                a.position, b.position,
                "body {i} position mismatch at step {step}"
            );
            assert_eq!(
                a.linear_velocity, b.linear_velocity,
                "body {i} v_lin mismatch at step {step}"
            );
        }
    }
    // Sanity: the middle and top boxes should have landed ABOVE z = 0.5 —
    // this is the direct positive contradiction to the earlier "all three
    // fell through each other" failure mode.
    assert!(
        loaded.bodies[1].position.z > 0.6,
        "middle box collapsed to z={}, expected ≈ 1.05",
        loaded.bodies[1].position.z
    );
    assert!(
        loaded.bodies[2].position.z > 1.3,
        "top box collapsed to z={}, expected ≈ 1.75",
        loaded.bodies[2].position.z
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
// 4. pendulum.json round-trip: byte-identical to a programmatic twin
// ---------------------------------------------------------------------------

/// Programmatic equivalent of `models/pendulum.json`. Same masses / inertias
/// / anchor offsets — a hidden loader mis-wiring (swapped joint offset,
/// wrong hinge axis, dropped mass) would diverge from this reference within
/// a handful of RK4 steps once the pendulum starts swinging.
fn build_pendulum_programmatic() -> Tree {
    let mut tree = Tree::new();
    // Pivot: fixed root at (0, 0, 1.5), unit inertia (matches the json).
    tree.push_link(Link::new(
        None,
        JointKind::Fixed,
        (Vec3::new(0.0, 0.0, 1.5), Quat::IDENTITY),
        (Vec3::ZERO, Quat::IDENTITY),
        1.0,
        Mat3::diag(1.0, 1.0, 1.0),
    ));
    // Upper: 0.9 m rod, mass 1.3, hinge about x, joint_offset_in_child at
    // (0, 0, 0.45) so the COM sits 0.45 m below the pivot.
    tree.push_link(Link::new(
        Some(0),
        JointKind::hinge(Vec3::X),
        (Vec3::ZERO, Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.45), Quat::IDENTITY),
        1.3,
        Mat3::diag(0.087_75, 0.087_75, 1e-6),
    ));
    // Lower: 0.6 m rod, mass 0.7, joint_offset_in_parent at (0, 0, -0.45)
    // (bottom of upper rod), joint_offset_in_child at (0, 0, 0.3).
    tree.push_link(Link::new(
        Some(1),
        JointKind::hinge(Vec3::X),
        (Vec3::new(0.0, 0.0, -0.45), Quat::IDENTITY),
        (Vec3::new(0.0, 0.0, 0.3), Quat::IDENTITY),
        0.7,
        Mat3::diag(0.021, 0.021, 1e-6),
    ));
    tree
}

#[test]
fn pendulum_json_matches_programmatic_construction_exactly() {
    let scene = load_from_path(model_path("pendulum.json")).unwrap();
    let pendulum_idx = scene.trees_by_name["pendulum"];
    let mut loaded = scene.world.trees[pendulum_idx].clone();
    let mut prog = build_pendulum_programmatic();

    // Non-trivial initial angles + rates so the trajectory exercises both
    // hinges under chaos-adjacent dynamics — a mis-wired offset or axis
    // will diverge within a few steps.
    for tree in [&mut loaded, &mut prog] {
        tree.set_hinge_angle(1, 1.2);
        tree.set_hinge_angle(2, -0.6);
        tree.set_hinge_rate(1, 0.4);
        tree.set_hinge_rate(2, -0.2);
    }

    // Link-level match before stepping — catches inertia/offset/axis errors
    // even if they happen to produce the same trajectory in early steps.
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
        assert_eq!(a.joint, b.joint, "link {i} joint kind");
    }
    assert_eq!(loaded.q, prog.q, "initial q mismatch");
    assert_eq!(loaded.qdot, prog.qdot, "initial qdot mismatch");

    let dt = 0.005f32;
    let g = Vec3::new(0.0, 0.0, -9.81);
    for step in 0..500 {
        rk4_step(&mut loaded, g, dt, zero_ext(3));
        rk4_step(&mut prog, g, dt, zero_ext(3));
        assert_eq!(loaded.q, prog.q, "q mismatch at step {step}");
        assert_eq!(loaded.qdot, prog.qdot, "qdot mismatch at step {step}");
    }
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
