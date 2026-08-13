//! Golden trajectory + orientation-stability anchor for the tier-2 contact
//! model.
//!
//! Scene: three axis-aligned boxes dropped so they settle into a vertical
//! stack on a static plane. This exercises box-plane (4 corner contacts on
//! the bottom box) AND box-box (vertex-vs-face on the upper interfaces),
//! covers the full penalty pipeline (normal spring/damper, pyramidal
//! friction, Newton's-third summation), and — unlike a sphere stack, which
//! collapses to single-point contacts along one axis — actually stresses
//! multi-contact torque balance. A wrong contact-point lever arm in the
//! wrench application would tilt the boxes over during settling; the
//! orientation anchor below pins that down.
//!
//! Same serialization cadence as the tier-1 tumbling golden — steps
//! 0/100/1000, `(position, orientation, linear_velocity, angular_velocity_body)`
//! per body in little-endian f32.
//!
//! # Regen guard
//!
//! Follows the tier-1 pattern exactly: the ignored `regenerate_contacts_golden`
//! test refuses to run anywhere other than macOS aarch64.

use newt::body::Body;
use newt::geom::Geom;
use newt::math::{Quat, Vec3};
use newt::world::World;

const F32_PER_BODY: usize = 3 + 4 + 3 + 3;
const HALF: Vec3 = Vec3::new(0.35, 0.35, 0.35);

fn scene() -> World {
    let mut world = World::new();
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);

    world.add_geom(Geom::static_plane(Vec3::ZERO, Vec3::Z, 0.6));

    // Perfectly aligned vertical drop; heights staggered so boxes arrive in
    // sequence rather than colliding mid-air.
    let heights = [0.5f32, 1.7, 2.9];
    for &h in &heights {
        let idx = world.add_body(Body::solid_box(
            1.0,
            HALF,
            Vec3::new(0.0, 0.0, h),
            Quat::IDENTITY,
        ));
        world.add_geom(Geom::r#box(idx, HALF, Vec3::ZERO, Quat::IDENTITY, 0.6));
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
    "/tests/goldens/stacking_3_boxes.bin"
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

/// A box stack that emerges perfectly upright is the acid test for lever-arm
/// correctness in the contact-force application. Wrong contact positions
/// (e.g. force applied at the box COM instead of at the corner), or a bug
/// where the `r × F` cross product uses the wrong arm, would tilt the stack
/// during settling and this anchor would catch it.
///
/// Bound: after 2000 steps (10 s of sim time) the deviation of each box's
/// orientation from identity, measured as `1 − |q.w|`, is under `1e-2` —
/// which corresponds to a rotation of well under 12 degrees. In practice
/// the current implementation settles to `< 5e-4` (< 1.7 degrees). A
/// mis-applied lever arm would rotate by tens of degrees within the first
/// second.
#[test]
fn stacked_boxes_stay_near_upright() {
    let mut world = scene();
    for _ in 0..2000 {
        world.step();
    }
    for (i, b) in world.bodies.iter().enumerate() {
        let tilt = 1.0 - newt::math::abs(b.orientation.w);
        assert!(
            tilt < 1.0e-2,
            "box {i} tilted (1 − |q.w| = {tilt}); orientation {:?}",
            b.orientation
        );
    }
    // Also assert the stack really did stack (not fall over sideways).
    for (i, b) in world.bodies.iter().enumerate() {
        assert!(
            newt::math::abs(b.position.x) < 5.0e-2,
            "box {i} drifted in x: {}",
            b.position.x
        );
        assert!(
            newt::math::abs(b.position.y) < 5.0e-2,
            "box {i} drifted in y: {}",
            b.position.y
        );
    }
    // And that the vertical order is preserved (bottom < middle < top).
    let zs: Vec<f32> = world.bodies.iter().map(|b| b.position.z).collect();
    assert!(zs[0] < zs[1] && zs[1] < zs[2], "stack collapsed: {zs:?}");
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
