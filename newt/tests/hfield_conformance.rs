use newt::contact::{Contact, box_hfield, capsule_hfield, sphere_hfield};
use newt::geom::{GeomPose, HeightField};
use newt::json::{self, Value};
use newt::math::{Quat, Vec3};

fn object<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Object(fields) = value else {
        panic!("{name} must be an object");
    };
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("missing {name}"))
}

fn number(value: &Value, name: &str) -> f32 {
    let Value::Number(value) = value else {
        panic!("{name} must be a number");
    };
    *value as f32
}

fn string<'a>(value: &'a Value, name: &str) -> &'a str {
    let Value::String(value) = value else {
        panic!("{name} must be a string");
    };
    value
}

fn numbers(value: &Value, name: &str) -> Vec<f32> {
    let Value::Array(values) = value else {
        panic!("{name} must be an array");
    };
    values.iter().map(|value| number(value, name)).collect()
}

fn optional_number(value: &Value, name: &str) -> Option<f32> {
    let Value::Object(fields) = value else {
        panic!("case must be an object");
    };
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| number(value, name))
}

fn pose(value: &Value) -> GeomPose {
    let position = numbers(object(value, "position"), "position");
    let orientation = if let Some(angle) = match value {
        Value::Object(fields) => fields
            .iter()
            .find(|(key, _)| key == "angle")
            .map(|(_, value)| number(value, "angle")),
        _ => None,
    } {
        let axis = numbers(object(value, "axis"), "axis");
        Quat::from_axis_angle(Vec3::new(axis[0], axis[1], axis[2]), angle)
    } else {
        Quat::IDENTITY
    };
    GeomPose {
        position: Vec3::new(position[0], position[1], position[2]),
        orientation,
    }
}

fn contact(value: &Value) -> Contact {
    let position = numbers(object(value, "position"), "position");
    let normal = numbers(object(value, "normal"), "normal");
    Contact {
        geom_a: 0,
        geom_b: 1,
        position_world: Vec3::new(position[0], position[1], position[2]),
        normal_world: Vec3::new(normal[0], normal[1], normal[2]),
        penetration: number(object(value, "penetration"), "penetration"),
        friction: 0.5,
        gap: 0.0,
    }
}

fn field(value: &Value) -> HeightField {
    let nrow = number(object(value, "nrow"), "nrow") as usize;
    let ncol = number(object(value, "ncol"), "ncol") as usize;
    let size = numbers(object(value, "size"), "size");
    HeightField {
        nrow,
        ncol,
        size: [size[0], size[1], size[2], size[3]],
        data: numbers(object(value, "data"), "data"),
    }
}

fn close_vec(actual: Vec3, expected: Vec3, tolerance: f32) {
    assert!(
        (actual - expected).length() <= tolerance,
        "{actual:?} != {expected:?}"
    );
}

#[test]
fn parsed_mujoco_heightfield_contacts_match_all_adversarial_poses() {
    let document = json::parse(include_str!("references/hfield_conformance.json"))
        .expect("conformance fixture must parse");
    let Value::Array(cases) = object(&document, "cases") else {
        panic!("cases must be an array");
    };
    for case in cases {
        let name = match object(case, "name") {
            Value::String(name) => name,
            _ => panic!("case name must be a string"),
        };
        let provenance = object(case, "provenance");
        let mode = string(object(provenance, "mode"), "mode");
        let nativeccd = string(object(provenance, "nativeccd"), "nativeccd");
        let source_xml = string(object(provenance, "source_xml"), "source_xml");
        let source_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/references")
            .join(source_xml);
        let source = std::fs::read_to_string(&source_path)
            .unwrap_or_else(|error| panic!("{name}: cannot read {source_xml}: {error}"));
        assert!(
            source.contains(&format!("case: {name}")),
            "{name}: source XML does not bind this case"
        );
        match nativeccd {
            "default" => {
                assert_eq!(mode, "default", "{name}: default mode mismatch");
                assert!(
                    !source.contains("nativeccd=\"disable\""),
                    "{name}: default source XML disables nativeccd"
                );
            }
            "disable" => {
                assert_eq!(mode, "legacy", "{name}: legacy mode mismatch");
                assert!(
                    source.contains("<flag nativeccd=\"disable\"/>"),
                    "{name}: legacy source XML lacks nativeccd=disable"
                );
            }
            _ => panic!("{name}: unsupported nativeccd state {nativeccd}"),
        }
        let shape = match object(case, "shape") {
            Value::String(shape) => shape,
            _ => panic!("case shape must be a string"),
        };
        let field = field(object(case, "field"));
        let field_pose = GeomPose {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        };
        let shape_pose = pose(object(case, "pose"));
        let actual = match shape.as_str() {
            "sphere" => sphere_hfield(
                0,
                &shape_pose,
                number(object(case, "radius"), "radius"),
                1,
                &field_pose,
                &field,
                0.5,
                0.0,
                0.0,
            ),
            "capsule" => capsule_hfield(
                0,
                &shape_pose,
                number(object(case, "radius"), "radius"),
                number(object(case, "half_height"), "half_height"),
                1,
                &field_pose,
                &field,
                0.5,
                0.0,
                0.0,
            ),
            "box" => {
                let half = numbers(object(case, "half_extents"), "half_extents");
                box_hfield(
                    0,
                    &shape_pose,
                    Vec3::new(half[0], half[1], half[2]),
                    1,
                    &field_pose,
                    &field,
                    0.5,
                    0.0,
                    0.0,
                )
            }
            _ => panic!("unsupported case shape {shape}"),
        };
        let Value::Array(expected_values) = object(case, "contacts") else {
            panic!("{name} contacts must be an array");
        };
        let mut expected: Vec<Contact> = expected_values.iter().map(contact).collect();
        let mut actual = actual.as_slice().to_vec();
        let position_tolerance = optional_number(case, "position_tolerance").unwrap_or(5.0e-3);
        let normal_bound = optional_number(case, "normal_bound").unwrap_or(5.0e-3);
        let order = |a: &Contact, b: &Contact| {
            b.penetration
                .total_cmp(&a.penetration)
                .then_with(|| a.position_world.x.total_cmp(&b.position_world.x))
                .then_with(|| a.position_world.y.total_cmp(&b.position_world.y))
        };
        expected.sort_by(order);
        actual.sort_by(order);
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (actual, expected) in actual.iter().zip(expected) {
            close_vec(
                actual.position_world,
                expected.position_world,
                position_tolerance,
            );
            close_vec(actual.normal_world, expected.normal_world, normal_bound);
            assert!(
                (actual.penetration - expected.penetration).abs() <= 5.0e-3,
                "{name}: {actual:?} != {expected:?}"
            );
        }
    }
}
