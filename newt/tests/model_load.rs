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

use newt::actuator::Actuator;
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
    tree.add_actuator(Actuator::position_from_dampratio(
        1,
        200.0,
        1.0,
        ARM_M[0] * ARM_L * ARM_L,
        60.0,
    ));
    tree.add_actuator(Actuator::position_from_dampratio(
        2,
        150.0,
        1.0,
        ARM_M[1] * ARM_L * ARM_L,
        40.0,
    ));
    tree.add_actuator(Actuator::position_from_dampratio(
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
    for name in ["arm.json", "pendulum.json", "stack.json", "pile.json"] {
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

// ---------------------------------------------------------------------------
// 6. round-trip: pile.json vs programmatic (mirrors examples/pile.rs)
// ---------------------------------------------------------------------------

/// Programmatic equivalent of `models/pile.json`. Same masses, positions,
/// orientations, inertias, geoms, and pair list as the demo scene in
/// `examples/pile.rs`. Kept side-by-side so a divergence between loader and
/// programmatic paths is caught cross-body by the strict byte-identical
/// assertion below.
fn build_pile_programmatic() -> World {
    use newt::geom::ConvexMesh;
    use newt::math::FRAC_PI_4;
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.7));

    // Mesh (must match pile.json's tetra: unit tetra scaled by 0.5).
    let mesh_id = world.add_mesh(ConvexMesh {
        vertices: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
        ],
        faces: vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
    });

    let cyl_r = 0.30f32;
    let cyl_h = 0.35f32;
    let ell_ax = Vec3::new(0.35, 0.25, 0.40);
    let box_h = Vec3::splat(0.25);
    let m = 1.0f32;

    // Cylinder
    let cyl = Body::new(
        m,
        newt::geom::solid_cylinder_inertia(m, cyl_r, cyl_h),
        Vec3::new(-0.9, -0.9, 1.4),
        Quat::from_axis_angle(Vec3::X, 0.15),
    );
    let ic = world.add_body(cyl);
    world.add_geom(Geom::cylinder(
        ic,
        cyl_r,
        cyl_h,
        Vec3::ZERO,
        Quat::IDENTITY,
        0.7,
    ));

    // Ellipsoid
    let ell = Body::new(
        m,
        newt::geom::solid_ellipsoid_inertia(m, ell_ax),
        Vec3::new(0.9, -0.9, 1.5),
        Quat::from_axis_angle(Vec3::Y, 0.35),
    );
    let ie = world.add_body(ell);
    world.add_geom(Geom::ellipsoid(ie, ell_ax, Vec3::ZERO, Quat::IDENTITY, 0.7));

    // Mesh tetra
    let tet = Body::new(
        m,
        Mat3::diag(0.02, 0.02, 0.02),
        Vec3::new(-0.9, 0.9, 1.5),
        Quat::IDENTITY,
    );
    let im = world.add_body(tet);
    world.add_geom(Geom::mesh(im, mesh_id, Vec3::ZERO, Quat::IDENTITY, 0.7));

    // Boxes
    let lower = Body::new(
        m,
        newt::geom::solid_box_inertia(m, box_h),
        Vec3::new(0.9, 0.9, 0.3),
        Quat::from_axis_angle(Vec3::Z, FRAC_PI_4),
    );
    let ibl = world.add_body(lower);
    world.add_geom(Geom::r#box(ibl, box_h, Vec3::ZERO, Quat::IDENTITY, 0.7));
    let upper = Body::new(
        m,
        newt::geom::solid_box_inertia(m, box_h),
        Vec3::new(0.92, 0.91, 1.0),
        Quat::IDENTITY,
    );
    let ibu = world.add_body(upper);
    world.add_geom(Geom::r#box(ibu, box_h, Vec3::ZERO, Quat::IDENTITY, 0.7));

    // Same restricted pair list as pile.json / examples/pile.rs.
    world.pair_list = Some(vec![(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (4, 5)]);
    world
}

#[test]
fn pile_json_matches_programmatic_construction_exactly() {
    let scene = load_from_path(model_path("pile.json")).expect("pile.json should load");
    let mut loaded = scene.world;
    let mut prog = build_pile_programmatic();

    assert_eq!(loaded.bodies.len(), prog.bodies.len());
    for (i, (a, b)) in loaded.bodies.iter().zip(prog.bodies.iter()).enumerate() {
        assert_eq!(a.mass, b.mass, "body {i} mass");
        assert_eq!(a.inertia_body, b.inertia_body, "body {i} inertia");
        assert_eq!(a.position, b.position, "body {i} position");
        assert_eq!(a.orientation, b.orientation, "body {i} orientation");
        assert_eq!(a.linear_velocity, b.linear_velocity, "body {i} v_lin");
    }
    assert_eq!(loaded.geoms.len(), prog.geoms.len());
    for (i, (a, b)) in loaded.geoms.iter().zip(prog.geoms.iter()).enumerate() {
        assert_eq!(a.shape, b.shape, "geom {i} shape");
        assert_eq!(a.attachment(), b.attachment(), "geom {i} attachment");
        assert_eq!(a.friction, b.friction, "geom {i} friction");
    }
    assert_eq!(loaded.meshes.len(), prog.meshes.len());
    for (i, (a, b)) in loaded.meshes.iter().zip(prog.meshes.iter()).enumerate() {
        assert_eq!(a.vertices, b.vertices, "mesh {i} vertices");
        assert_eq!(a.faces, b.faces, "mesh {i} faces");
    }
    assert_eq!(loaded.pair_list, prog.pair_list, "pair_list");

    // Step both worlds 200 steps and require byte-identical body state each
    // iteration. The v1-tier-2 primitives (cylinder-plane, ellipsoid-plane,
    // mesh-plane, sphere-none-here, box-box) all participate; a loader that
    // dropped one — or reordered pair enumeration — would blow this within
    // a few steps.
    for step in 0..200 {
        loaded.step();
        prog.step();
        for (i, (a, b)) in loaded.bodies.iter().zip(prog.bodies.iter()).enumerate() {
            assert_eq!(
                a.position, b.position,
                "body {i} position mismatch at step {step}"
            );
            assert_eq!(
                a.orientation, b.orientation,
                "body {i} orientation mismatch at step {step}"
            );
        }
    }
    // Sanity: upper box stayed above lower (yawed edge-edge stacking still
    // holds when driven from the loader).
    let lower_z = loaded.bodies[3].position.z;
    let upper_z = loaded.bodies[4].position.z;
    assert!(
        upper_z > lower_z + 0.3,
        "loaded pile yawed stack collapsed: lower {lower_z}, upper {upper_z}"
    );
}

// ---------------------------------------------------------------------------
// 7. loader coverage for v1-tier-2 geoms and margin/gap
// ---------------------------------------------------------------------------

#[test]
fn loader_parses_cylinder_and_ellipsoid_geoms_round_trip() {
    // Include a static ground plane so the auto pair list only enumerates
    // supported cylinder-plane / ellipsoid-plane combinations — the pure
    // cylinder-vs-ellipsoid pair is deferred and would trip the loader's
    // engine-level enforcement.
    let json = r#"{
        "version": "1",
        "bodies": [
            {"name":"a","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}},
            {"name":"b","mass":2,"inertia":{"kind":"solid","shape":{"kind":"cylinder","radius":0.5,"half_height":0.4}}},
            {"name":"c","mass":3,"inertia":{"kind":"solid","shape":{"kind":"ellipsoid","semi_axes":[0.3,0.2,0.4]}}}
        ],
        "geoms": [
            {"name":"ground","shape":{"kind":"plane"},"attach":{"kind":"static"}},
            {"name":"g1","shape":{"kind":"cylinder","radius":0.5,"half_height":0.4},"attach":{"kind":"body","body":"a"}},
            {"name":"g2","shape":{"kind":"ellipsoid","semi_axes":[0.3,0.2,0.4]},"attach":{"kind":"body","body":"b"}}
        ],
        "contact_pairs": {"explicit":[{"a":"ground","b":"g1"},{"a":"ground","b":"g2"}]}
    }"#;
    let scene = newt::model::load_str(json).expect("cylinder+ellipsoid should parse");
    assert_eq!(scene.world.bodies.len(), 3);
    // geoms[0] = ground plane; geoms[1] = cylinder; geoms[2] = ellipsoid.
    assert!(matches!(
        scene.world.geoms[1].shape,
        newt::geom::GeomShape::Cylinder {
            radius: 0.5,
            half_height: 0.4,
        }
    ));
    assert!(matches!(
        scene.world.geoms[2].shape,
        newt::geom::GeomShape::Ellipsoid { .. }
    ));
    // Inertia round-trip: body b's solid-cylinder inertia matches the helper.
    assert_eq!(
        scene.world.bodies[1].inertia_body,
        newt::geom::solid_cylinder_inertia(2.0, 0.5, 0.4)
    );
    assert_eq!(
        scene.world.bodies[2].inertia_body,
        newt::geom::solid_ellipsoid_inertia(3.0, Vec3::new(0.3, 0.2, 0.4))
    );
}

#[test]
fn loader_parses_mesh_geom_and_asset_round_trip() {
    let json = r#"{
        "version": "1",
        "meshes": [
            {"name":"tetra","vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],"faces":[[0,2,1],[0,1,3],[0,3,2],[1,2,3]]}
        ],
        "bodies": [
            {"name":"a","mass":1,"inertia":{"kind":"diag","values":[0.02,0.02,0.02]}}
        ],
        "geoms": [
            {"name":"g","shape":{"kind":"mesh","mesh":"tetra"},"attach":{"kind":"body","body":"a"}}
        ]
    }"#;
    let scene = newt::model::load_str(json).expect("mesh asset + geom should parse");
    assert_eq!(scene.world.meshes.len(), 1);
    assert_eq!(scene.world.meshes[0].vertices.len(), 4);
    assert_eq!(scene.world.meshes[0].faces.len(), 4);
    assert!(matches!(
        scene.world.geoms[0].shape,
        newt::geom::GeomShape::Mesh { mesh_id: 0 }
    ));
}

#[test]
fn loader_rejects_mesh_with_fewer_than_four_vertices() {
    let json = r#"{
        "version": "1",
        "meshes": [
            {"name":"bad","vertices":[[0,0,0],[1,0,0],[0,1,0]],"faces":[[0,1,2],[0,1,2],[0,1,2],[0,1,2]]}
        ]
    }"#;
    let err = newt::model::load_str(json).expect_err("mesh with < 4 vertices should fail");
    assert!(err.to_string().contains("4 vertices"), "err: {err}");
}

#[test]
fn loader_rejects_mesh_with_fewer_than_four_faces() {
    let json = r#"{
        "version": "1",
        "meshes": [
            {"name":"bad","vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],"faces":[[0,1,2]]}
        ]
    }"#;
    let err = newt::model::load_str(json).expect_err("mesh with < 4 faces should fail");
    assert!(err.to_string().contains("4 triangular faces"), "err: {err}");
}

#[test]
fn loader_rejects_mesh_face_index_out_of_range() {
    let json = r#"{
        "version": "1",
        "meshes": [
            {"name":"bad","vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,1]],"faces":[[0,1,2],[0,1,3],[0,3,2],[1,2,7]]}
        ]
    }"#;
    let err = newt::model::load_str(json).expect_err("mesh face index out of range should fail");
    assert!(err.to_string().contains("out of range"), "err: {err}");
}

#[test]
fn loader_rejects_non_finite_mesh_vertex() {
    let json = r#"{
        "version": "1",
        "meshes": [
            {"name":"bad","vertices":[[0,0,0],[1,0,0],[0,1,0],[0,0,null]],"faces":[[0,1,2],[0,1,3],[0,3,2],[1,2,3]]}
        ]
    }"#;
    let _ = newt::model::load_str(json).expect_err("null vertex coord should fail");
    // (The exact error path depends on the underlying JSON scalar parser
    // rejecting `null` as a number; either message is acceptable.)
}

#[test]
fn loader_rejects_cylinder_with_negative_radius() {
    let json = r#"{
        "version": "1",
        "bodies": [
            {"name":"a","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
        ],
        "geoms": [
            {"name":"g","shape":{"kind":"cylinder","radius":-0.5,"half_height":0.4},"attach":{"kind":"body","body":"a"}}
        ]
    }"#;
    let err = newt::model::load_str(json).expect_err("negative cylinder radius should fail");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("cylinder") && msg.contains("radius"),
        "err: {err}"
    );
}

#[test]
fn loader_rejects_ellipsoid_with_zero_semi_axis() {
    let json = r#"{
        "version": "1",
        "bodies": [
            {"name":"a","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
        ],
        "geoms": [
            {"name":"g","shape":{"kind":"ellipsoid","semi_axes":[0.3,0.0,0.4]},"attach":{"kind":"body","body":"a"}}
        ]
    }"#;
    let err = newt::model::load_str(json).expect_err("zero ellipsoid semi-axis should fail");
    let msg = err.to_string().to_lowercase();
    assert!(
        msg.contains("ellipsoid") && msg.contains("axes"),
        "err: {err}"
    );
}

#[test]
fn loader_parses_margin_and_gap_round_trip() {
    let json = r#"{
        "version": "1",
        "bodies": [
            {"name":"a","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
        ],
        "geoms": [
            {"name":"g","shape":{"kind":"sphere","radius":0.5},"attach":{"kind":"body","body":"a"},
             "margin":0.02,"gap":0.005}
        ]
    }"#;
    let scene = newt::model::load_str(json).expect("margin+gap should parse");
    assert_eq!(scene.world.geoms[0].margin, 0.02);
    assert_eq!(scene.world.geoms[0].gap, 0.005);
}

#[test]
fn loader_rejects_negative_margin() {
    let json = r#"{
        "version": "1",
        "bodies": [
            {"name":"a","mass":1,"inertia":{"kind":"diag","values":[1,1,1]}}
        ],
        "geoms": [
            {"name":"g","shape":{"kind":"sphere","radius":0.5},"attach":{"kind":"body","body":"a"},
             "margin":-0.01}
        ]
    }"#;
    let err = newt::model::load_str(json).expect_err("negative margin should fail");
    let msg = err.to_string();
    assert!(msg.contains("margin") && msg.contains("≥ 0"), "err: {err}");
}

#[test]
fn loader_axis_angle_orientation_matches_from_axis_angle_exactly() {
    // The `orientation_axis_angle` pose form is designed for byte-identity
    // with a programmatic `Quat::from_axis_angle` call — same sin/cos
    // polynomials on both sides. Round-trip against the equivalent
    // programmatic construction.
    let json = r#"{
        "version": "1",
        "bodies": [
            {"name":"a","mass":1,"inertia":{"kind":"diag","values":[1,1,1]},
             "pose":{"position":[0,0,0],"orientation_axis_angle":{"axis":[1,0,0],"angle":0.35}}}
        ]
    }"#;
    let scene = newt::model::load_str(json).unwrap();
    let expected = Quat::from_axis_angle(Vec3::X, 0.35);
    assert_eq!(scene.world.bodies[0].orientation, expected);
}
