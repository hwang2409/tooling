//! Small per-integrator trajectory goldens.
//!
//! The RK4 file is also a regression anchor for the pre-v3 default. The
//! ignored regeneration test is guarded to the macOS-aarch64 reference host.

use newt::body::Body;
use newt::math::{Mat3, Quat, Vec3};
use newt::world::{Integrator, World};

fn scene(integrator: Integrator) -> World {
    let mut world = World::new();
    world.integrator = integrator;
    world.dt = 0.005;
    world.gravity = Vec3::new(0.0, 0.0, -9.81);
    let mut body = Body::new(
        1.25,
        Mat3::diag(0.7, 1.3, 2.1),
        Vec3::new(0.4, -0.7, 2.0),
        Quat::from_axis_angle(Vec3::new(1.0, 0.2, -0.4), 0.6),
    );
    body.linear_velocity = Vec3::new(0.8, -0.4, 1.2);
    body.angular_velocity_body = Vec3::new(0.4, 1.1, -0.3);
    world.add_body(body);
    world
}

fn snapshot(world: &World) -> Vec<u8> {
    let mut bytes = Vec::new();
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
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn trajectory(integrator: Integrator) -> Vec<u8> {
    let mut world = scene(integrator);
    let mut bytes = snapshot(&world);
    for _ in 0..20 {
        world.step();
    }
    bytes.extend_from_slice(&snapshot(&world));
    for _ in 20..100 {
        world.step();
    }
    bytes.extend_from_slice(&snapshot(&world));
    bytes
}

const RK4_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/integrator_rk4.bin"
);
const EULER_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/integrator_euler.bin"
);
const IMPLICITFAST_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/goldens/integrator_implicitfast.bin"
);

fn assert_golden(path: &str, integrator: Integrator) {
    let expected = std::fs::read(path).expect("integrator golden missing");
    let actual = trajectory(integrator);
    assert_eq!(expected, actual, "integrator golden mismatch at {path}");
}

#[test]
fn rk4_integrator_golden_is_byte_identical() {
    assert_golden(RK4_GOLDEN, Integrator::Rk4);
}

#[test]
fn euler_integrator_golden_is_byte_identical() {
    assert_golden(EULER_GOLDEN, Integrator::Euler);
}

#[test]
fn implicitfast_integrator_golden_is_byte_identical() {
    assert_golden(IMPLICITFAST_GOLDEN, Integrator::ImplicitFast);
}

#[test]
#[ignore]
fn regenerate_integrator_goldens() {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        panic!("integrator goldens may only be regenerated on macOS aarch64");
    }
    std::fs::write(RK4_GOLDEN, trajectory(Integrator::Rk4)).unwrap();
    std::fs::write(EULER_GOLDEN, trajectory(Integrator::Euler)).unwrap();
    std::fs::write(IMPLICITFAST_GOLDEN, trajectory(Integrator::ImplicitFast)).unwrap();
}
