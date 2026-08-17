//! MuJoCo `mjc_PlaneBox` manifold anchors.
//!
//! Expected values come from the committed 3.11.0 capture in
//! `tests/references/box_plane_mujoco_anchors.json`.

#![allow(clippy::excessive_precision)]

use newt::contact::box_plane;
use newt::geom::{Geom, GeomPose, geom_world_pose};
use newt::math::{FRAC_PI_4, Quat, Vec3};

const HALF_EXTENTS: Vec3 = Vec3::new(0.5, 0.4, 0.3);
const TOLERANCE: f32 = 2.0e-6;

fn assert_anchor(
    name: &str,
    position: Vec3,
    orientation: Quat,
    margin: f32,
    expected: &[(Vec3, f32)],
) {
    let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
    let plane_pose = geom_world_pose(&plane, Vec3::ZERO, Quat::IDENTITY);
    let box_pose = GeomPose {
        position,
        orientation,
    };
    let contacts = box_plane(
        1,
        &box_pose,
        HALF_EXTENTS,
        1.0,
        margin,
        0.0,
        0,
        &plane,
        &plane_pose,
    );
    assert_eq!(contacts.len, expected.len(), "{name}: contact count");
    for (index, (contact, (position, dist))) in contacts.as_slice().iter().zip(expected).enumerate()
    {
        assert_eq!(contact.geom_a, 1, "{name} contact {index}: geom_a");
        assert_eq!(contact.geom_b, 0, "{name} contact {index}: geom_b");
        assert_vec_close(name, index, "position", contact.position_world, *position);
        assert_vec_close(name, index, "normal", contact.normal_world, Vec3::Z);
        assert_close(
            name,
            index,
            "penetration",
            contact.penetration,
            margin - dist,
        );
    }
}

fn assert_vec_close(name: &str, index: usize, field: &str, actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).length() <= TOLERANCE,
        "{name} contact {index} {field}: actual {actual:?}, expected {expected:?}"
    );
}

fn assert_close(name: &str, index: usize, field: &str, actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= TOLERANCE,
        "{name} contact {index} {field}: actual {actual}, expected {expected}"
    );
}

#[test]
fn box_plane_matches_mujoco_hand_posed_manifolds() {
    assert_anchor(
        "flat_rest",
        Vec3::new(0.0, 0.0, 0.2),
        Quat::IDENTITY,
        0.0,
        &[
            (Vec3::new(-0.5, -0.4, -0.05), -0.1),
            (Vec3::new(0.5, -0.4, -0.05), -0.1),
            (Vec3::new(-0.5, 0.4, -0.05), -0.1),
            (Vec3::new(0.5, 0.4, -0.05), -0.1),
        ],
    );
    assert_anchor(
        "edge_tilt",
        Vec3::new(0.0, 0.0, 0.4),
        Quat::from_axis_angle(Vec3::X, FRAC_PI_4),
        0.0,
        &[
            (Vec3::new(-0.5, -0.07071068, -0.047487374), -0.09497475),
            (Vec3::new(0.5, -0.07071068, -0.047487374), -0.09497475),
        ],
    );
    assert_anchor(
        "corner_tilt",
        Vec3::new(0.0, 0.0, 0.4),
        Quat::from_axis_angle(Vec3::new(1.0, 1.0, 0.0), 0.55),
        0.0,
        &[(Vec3::new(0.32275733, -0.22275733, -0.09419674), -0.18839347)],
    );
    assert_anchor(
        "deep_penetration",
        Vec3::new(0.12, -0.07, -0.1),
        Quat::from_axis_angle(Vec3::Z, 0.17),
        0.0,
        &[
            (Vec3::new(-0.30511944, -0.5488251, -0.2), -0.4),
            (Vec3::new(0.68046534, -0.37964273, -0.2), -0.4),
            (Vec3::new(-0.44046533, 0.23964273, -0.2), -0.4),
            (Vec3::new(0.54511946, 0.4088251, -0.2), -0.4),
        ],
    );
    assert_anchor(
        "shallow_margin",
        Vec3::new(0.12, -0.07, 0.3005),
        Quat::IDENTITY,
        0.001,
        &[
            (Vec3::new(-0.38, -0.47, 0.00025), 0.0005),
            (Vec3::new(0.62, -0.47, 0.00025), 0.0005),
            (Vec3::new(-0.38, 0.33, 0.00025), 0.0005),
            (Vec3::new(0.62, 0.33, 0.00025), 0.0005),
        ],
    );
    assert_anchor(
        "sliding_approach",
        Vec3::new(0.37, -0.21, 0.4),
        Quat::from_axis_angle(Vec3::Y, 0.22),
        0.0,
        &[
            (Vec3::new(0.79247986, -0.61, -0.00094202), -0.00188405),
            (Vec3::new(0.79247986, 0.19, -0.00094202), -0.00188405),
        ],
    );
}

#[test]
fn box_plane_order_is_byte_stable() {
    let plane = Geom::static_plane(Vec3::ZERO, Vec3::Z, 1.0);
    let plane_pose = geom_world_pose(&plane, Vec3::ZERO, Quat::IDENTITY);
    let box_pose = GeomPose {
        position: Vec3::new(0.12, -0.07, -0.1),
        orientation: Quat::from_axis_angle(Vec3::Z, 0.17),
    };
    let first = box_plane(
        1,
        &box_pose,
        HALF_EXTENTS,
        1.0,
        0.0,
        0.0,
        0,
        &plane,
        &plane_pose,
    );
    let second = box_plane(
        1,
        &box_pose,
        HALF_EXTENTS,
        1.0,
        0.0,
        0.0,
        0,
        &plane,
        &plane_pose,
    );
    assert_eq!(first.as_slice(), second.as_slice());
}
