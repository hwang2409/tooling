//! Golden trajectory anchor for the tier-2 contact model.
//!
//! Scene: three spheres dropped so they settle into a vertical stack on a
//! static plane, exercising sphere-plane AND sphere-sphere contacts, plus
//! penalty normal spring/damper, tangential friction clamp, and Newton's-third
//! reaction. Serialized `(position, orientation, linear_velocity,
//! angular_velocity_body)` for every body at steps 0, 100, and 1000 —
//! same layout and cadence as the tier-1 tumbling golden so the format stays
//! uniform.
//!
//! # Why spheres and not boxes
//!
//! Box-box collision is deferred to a later tier (see the PR body). Sphere-
//! sphere gives full stacking coverage today: it exercises the same normal
//! spring, the same friction pyramid, and the same wrench summation as any
//! future box-box will need.
//!
//! # Regen guard
//!
//! Follows the tier-1 pattern exactly: the ignored `regenerate_golden` test
//! refuses to run anywhere other than macOS aarch64, and the main test's
//! byte comparison locks the file in cross-platform CI.

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::world::World;

const F32_PER_BODY: usize = 3 + 4 + 3 + 3;

fn scene() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    let radius = 0.3;
    let mass = 1.0;
    let plane = world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.6));
    let _ = plane;

    // Stagger the drop heights so the spheres arrive at the pile in order.
    let heights = [0.5, 1.3, 2.1];
    for &h in &heights {
        let idx = world.add_body(Body::solid_sphere(
            mass,
            radius,
            Vec3::new(0.0, 0.0, h),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::sphere(idx, radius, Vec3::ZERO, 0.6));
    }
    world
}

fn snapshot(world: &World) -> Vec<u8> {
    let mut out = Vec::with_capacity(world.bodies.len() * F32_PER_BODY * 4);
    for body in &world.bodies {
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

fn produce_golden_bytes() -> Vec<u8> {
    let mut world = scene();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&snapshot(&world));
    for _ in 0..100 {
        world.step();
    }
    bytes.extend_from_slice(&snapshot(&world));
    for _ in 100..1000 {
        world.step();
    }
    bytes.extend_from_slice(&snapshot(&world));
    bytes
}

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/stacking_3_spheres.bin"
);

#[test]
fn contacts_golden_trajectory_is_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect(
        "contacts golden file missing — run the ignored `regenerate_contacts_golden` \
         test on macOS to produce it, then commit",
    );
    let actual = produce_golden_bytes();
    assert_eq!(
        expected.len(),
        actual.len(),
        "contacts golden byte length mismatch: expected {} got {}",
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
            "contacts golden trajectory mismatch; first byte diff at offset {first_diff} \
             (bodies * 13 f32 * 4 bytes per snapshot; snapshot index = offset / \
             {stride})",
            stride = 3 * F32_PER_BODY * 4
        );
    }
}

#[test]
#[ignore]
fn regenerate_contacts_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_contacts_golden may only run on the reference host \
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
