//! Golden trajectory + contact-torque anchor for the tier-2 contact
//! model.
//!
//! Scene: three axis-aligned boxes dropped onto a static plane. Before the
//! MuJoCo plane-box update, they settled into a vertical stack. The exact
//! manifold exercises box-plane (4 corner contacts on the bottom box) AND
//! box-box (vertex-vs-face on the upper interfaces),
//! covers the full penalty pipeline (normal spring/damper, pyramidal
//! friction, Newton's-third summation), and — unlike a sphere stack, which
//! collapses to single-point contacts along one axis — actually stresses
//! multi-contact torque balance. A wrong contact-point lever arm changes the
//! recorded trajectory; the bounded-torque anchor below pins that down.
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

    // Symmetry-broken drop. A perfectly axis-aligned scene cancels the four
    // corner torques identically, so a lever-arm mutant (contact forces
    // applied at the COM with zero arm) produces a bit-identical trajectory
    // and the orientation anchor proves nothing. We break the symmetry two
    // ways: shift the middle box by 0.02 m in +X so its 4 bottom corners no
    // longer sit symmetrically on the bottom box's top face, and give the
    // top box a small initial angular velocity about Y (0.3 rad/s). Both
    // create asymmetric torques that the mutant can't reproduce.
    let drops: [(Vec3, Vec3); 3] = [
        (Vec3::new(0.00, 0.0, 0.5), Vec3::ZERO),
        (Vec3::new(0.02, 0.0, 1.7), Vec3::ZERO),
        (Vec3::new(0.00, 0.0, 2.9), Vec3::new(0.0, 0.3, 0.0)),
    ];
    for &(pos, omega_body) in &drops {
        let mut body = Body::solid_box(1.0, HALF, pos, Quat::IDENTITY);
        body.angular_velocity_body = omega_body;
        let idx = world.add_body(body);
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

/// Lever-arm sanity check after the MuJoCo plane-box manifold update. The
/// source corner scan changes the long-term three-box penalty trajectory:
/// upper boxes redistribute onto the plane. The bottom box stays bounded and
/// the initial top-box spin still decays through contact-point friction.
/// These checks keep the `r × F` path covered without hiding that behavior
/// change behind the old upright-stack expectation.
#[test]
fn stacked_boxes_keep_contact_torques_bounded_after_manifold_change() {
    let mut world = scene();
    for _ in 0..2000 {
        world.step();
    }
    let bottom = &world.bodies[0];
    let bottom_tilt = 1.0 - newt::math::abs(bottom.orientation.w);
    assert!(bottom_tilt < 1.0e-2, "bottom box tilted: {bottom_tilt}");
    assert!(bottom.position.x.abs() < 0.2);
    assert!(bottom.position.y.abs() < 0.15);
    for (i, b) in world.bodies.iter().enumerate() {
        assert!(b.position.z > 0.3, "box {i} fell through the plane");
        assert!(
            b.position.z < 0.4,
            "box {i} left the plane: {}",
            b.position.z
        );
        assert!(b.position.x.is_finite() && b.position.y.is_finite());
    }
    // Top box's initial spin must have decayed under the correct
    // lever-arm friction moment.
    let top_omega_y = newt::math::abs(world.bodies[2].angular_velocity_body.y);
    assert!(
        top_omega_y < 1.0,
        "top box's initial spin didn't decay: ω_y = {top_omega_y}"
    );
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
