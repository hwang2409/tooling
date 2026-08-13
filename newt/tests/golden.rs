//! Golden trajectory anchor: serialize `(q, qdot)` for a fixed 3-body scene
//! at steps 0, 100, 1000. Byte-compared against a tracked file to prove
//! macOS/Linux produce bit-identical output.
//!
//! The golden was generated on macOS (aarch64) by running
//! `cargo test --test golden -- --nocapture regenerate_golden --ignored` on a
//! development machine and committing the resulting file. CI on Linux runs
//! the normal test and refuses to update the file.

use newt::body::Body;
use newt::math::{Quat, Vec3};
use newt::world::World;

/// One body's full state, laid out contiguously as little-endian f32s:
/// position(3) | orientation(x,y,z,w) | linear_velocity(3) | angular_velocity_body(3).
const F32_PER_BODY: usize = 3 + 4 + 3 + 3;

fn scene() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    let mut b0 = Body::solid_box(
        1.0,
        Vec3::new(0.4, 0.3, 0.2),
        Vec3::new(0.0, 0.0, 5.0),
        Quat::IDENTITY,
    );
    b0.linear_velocity = Vec3::new(1.0, 0.5, 3.0);
    b0.angular_velocity_body = Vec3::new(1.0, 2.0, 0.5);

    let mut b1 = Body::solid_box(
        2.0,
        Vec3::new(0.3, 0.3, 0.3),
        Vec3::new(1.5, -1.0, 4.0),
        Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.3),
    );
    b1.linear_velocity = Vec3::new(-0.5, 1.5, 2.0);
    b1.angular_velocity_body = Vec3::new(0.2, 3.5, 0.1);

    let mut b2 = Body::solid_box(
        0.5,
        Vec3::new(0.5, 0.2, 0.1),
        Vec3::new(-1.0, 2.0, 6.0),
        Quat::from_axis_angle(Vec3::new(0.0, 1.0, 1.0), 0.7),
    );
    b2.linear_velocity = Vec3::new(2.0, -1.0, 0.5);
    b2.angular_velocity_body = Vec3::new(0.05, 0.1, 4.0);

    world.add_body(b0);
    world.add_body(b1);
    world.add_body(b2);
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
    "/tests/goldens/tumbling_3_body.bin"
);

#[test]
fn golden_trajectory_is_byte_identical() {
    let expected = std::fs::read(GOLDEN_PATH).expect(
        "golden file missing — run the ignored `regenerate_golden` test on \
         macOS to produce it, then commit",
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
        // Find the first differing byte offset to make CI diagnosis fast.
        let first_diff = expected
            .iter()
            .zip(actual.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(0);
        panic!(
            "golden trajectory mismatch; first byte diff at offset {first_diff} \
             (bodies * 13 f32 * 4 bytes per snapshot; snapshot index = offset / \
             {stride})",
            stride = 3 * F32_PER_BODY * 4
        );
    }
}

/// Regenerate the golden file. Ignored so it does not run in CI. Only ever
/// run this on the reference machine (macOS aarch64) — the whole point of
/// the determinism doctrine is that Linux must reproduce these bytes
/// exactly, so a Linux regen would silently swap the reference and disable
/// the cross-platform check on future runs. The platform guard below
/// panics on any other host so an accidental `--ignored` invocation cannot
/// slip through.
#[test]
#[ignore]
fn regenerate_golden() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!(
            "regenerate_golden may only run on the reference host \
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
