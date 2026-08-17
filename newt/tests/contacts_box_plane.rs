//! MuJoCo `mjc_PlaneBox` manifold anchors.
//!
//! Expected values come from the committed 3.11.0 capture in
//! `tests/references/box_plane_mujoco_anchors.json`.

#![allow(clippy::excessive_precision)]

use newt::contact::box_plane;
use newt::geom::{Geom, GeomPose, geom_world_pose};
use newt::json::{self, Value};
use newt::math::{FRAC_PI_4, Quat, Vec3};
use std::fs;
use std::path::Path;

const HALF_EXTENTS: Vec3 = Vec3::new(0.5, 0.4, 0.3);
const TOLERANCE: f32 = 2.0e-6;

fn assert_anchor(
    name: &str,
    position: Vec3,
    orientation: Quat,
    margin: f32,
    expected: &[(Vec3, f32)],
) {
    assert_anchor_with_half_extents(name, HALF_EXTENTS, position, orientation, margin, expected);
}

fn assert_anchor_with_half_extents(
    name: &str,
    half_extents: Vec3,
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
        half_extents,
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

fn fixture_case(name: &str) -> (Vec3, Vec3, Quat, f32, usize, Vec<(Vec3, f32)>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source = fs::read_to_string(root.join("tests/references/box_plane_mujoco_anchors.json"))
        .expect("box-plane MuJoCo anchor fixture");
    let object = expect_object(json::parse(&source).expect("box-plane fixture is valid JSON"));
    assert_eq!(
        expect_string(object_value(&object, "mujoco_version")),
        "3.11.0"
    );
    let cases = match object_value(&object, "cases") {
        Value::Array(cases) => cases,
        other => panic!(
            "box-plane cases must be an array, got {}",
            other.type_name()
        ),
    };
    let case = cases
        .into_iter()
        .map(expect_object)
        .find(|case| expect_string(object_value(case, "name")) == name)
        .unwrap_or_else(|| panic!("box-plane fixture missing {name}"));
    let position = vec3(expect_f64_vec(object_value(&case, "position")));
    let half_extents = match optional_object_value(&case, "half_extents") {
        Some(Value::Array(values)) => vec3(values.into_iter().map(expect_f64).collect()),
        None => HALF_EXTENTS,
        Some(other) => panic!(
            "box-plane half_extents must be an array, got {}",
            other.type_name()
        ),
    };
    let quaternion = expect_f64_vec(object_value(&case, "quaternion_wxyz"));
    assert_eq!(quaternion.len(), 4);
    let orientation = Quat::new(
        quaternion[1] as f32,
        quaternion[2] as f32,
        quaternion[3] as f32,
        quaternion[0] as f32,
    );
    let contacts: Vec<(Vec3, f32)> = match object_value(&case, "contacts") {
        Value::Array(contacts) => contacts
            .into_iter()
            .map(|contact| {
                let contact = expect_object(contact);
                (
                    vec3(expect_f64_vec(object_value(&contact, "position"))),
                    expect_f64(object_value(&contact, "dist")) as f32,
                )
            })
            .collect(),
        other => panic!(
            "box-plane contacts must be an array, got {}",
            other.type_name()
        ),
    };
    let eligible_corners = optional_object_value(&case, "eligible_corners")
        .map(expect_usize)
        .unwrap_or(contacts.len());
    (
        position,
        half_extents,
        orientation,
        expect_f64(object_value(&case, "margin")) as f32,
        eligible_corners,
        contacts,
    )
}

fn vec3(values: Vec<f64>) -> Vec3 {
    assert_eq!(values.len(), 3);
    Vec3::new(values[0] as f32, values[1] as f32, values[2] as f32)
}

fn expect_string(value: Value) -> String {
    match value {
        Value::String(value) => value,
        other => panic!("expected string, got {}", other.type_name()),
    }
}

fn expect_f64(value: Value) -> f64 {
    match value {
        Value::Number(value) => value,
        other => panic!("expected number, got {}", other.type_name()),
    }
}

fn expect_f64_vec(value: Value) -> Vec<f64> {
    match value {
        Value::Array(values) => values.into_iter().map(expect_f64).collect(),
        other => panic!("expected array, got {}", other.type_name()),
    }
}

fn expect_usize(value: Value) -> usize {
    match value {
        Value::Number(value) => value as usize,
        other => panic!("expected number, got {}", other.type_name()),
    }
}

fn object_value(object: &[(String, Value)], key: &str) -> Value {
    object
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
        .unwrap_or_else(|| panic!("fixture object missing {key}"))
}

fn optional_object_value(object: &[(String, Value)], key: &str) -> Option<Value> {
    object
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value.clone())
}

fn expect_object(value: Value) -> Vec<(String, Value)> {
    match value {
        Value::Object(object) => object,
        other => panic!("expected object, got {}", other.type_name()),
    }
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

#[test]
fn box_plane_fixture_covers_cap_and_exact_margin() {
    let (position, half_extents, orientation, margin, eligible_corners, expected) =
        fixture_case("deep_over_four");
    assert!(eligible_corners > 4);
    assert_eq!(
        expected.len(),
        4,
        "fixture must exercise the four-contact cap"
    );
    assert_anchor_with_half_extents(
        "deep_over_four",
        half_extents,
        position,
        orientation,
        margin,
        &expected,
    );

    let (position, half_extents, orientation, margin, eligible_corners, expected) =
        fixture_case("exact_margin");
    assert_eq!(eligible_corners, 4);
    assert_eq!(expected.len(), 4);
    assert_anchor_with_half_extents(
        "exact_margin",
        half_extents,
        position,
        orientation,
        margin,
        &expected,
    );
    for (index, (_, dist)) in expected.iter().enumerate() {
        assert_close(
            "exact_margin",
            index,
            "zero penetration",
            margin - dist,
            0.0,
        );
    }
}
