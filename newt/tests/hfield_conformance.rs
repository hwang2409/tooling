use newt::contact::{Contact, box_hfield, capsule_hfield, sphere_hfield};
use newt::geom::{GeomAttach, GeomPose, GeomShape, HeightField, geom_world_pose};
use newt::json::{self, Value};
use newt::math::{Quat, Vec3};
use newt::mjcf::load_mjcf_str;

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

fn optional_string<'a>(value: &'a Value, name: &str) -> Option<&'a str> {
    let Value::Object(fields) = value else {
        panic!("case must be an object");
    };
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| string(value, name))
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
        feature_id: (0, 0),
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

fn close_scalar(actual: f32, expected: f32, tolerance: f32, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: {actual} != {expected}"
    );
}

fn assert_source_matches_case(
    source: &newt::model::Scene,
    name: &str,
    case: &Value,
    shape: &str,
    expected_field: &HeightField,
    expected_pose: GeomPose,
) {
    assert!(
        !source.world.geoms.is_empty(),
        "{name}: source has no geoms"
    );
    assert!(
        !source.world.hfields.is_empty(),
        "{name}: source has no hfields"
    );
    let field_geom = source
        .geoms_by_name
        .get(&format!("field_geom_{name}"))
        .map(|&index| &source.world.geoms[index])
        .unwrap_or_else(|| panic!("{name}: source hfield geom is missing"));
    let hfield_id = match field_geom.shape {
        GeomShape::Hfield { hfield_id } => hfield_id,
        other => panic!("{name}: source field geom has shape {other:?}"),
    };
    let loaded_field = &source.world.hfields[hfield_id];
    assert_eq!(loaded_field.nrow, expected_field.nrow, "{name}: nrow");
    assert_eq!(loaded_field.ncol, expected_field.ncol, "{name}: ncol");
    for (index, (&actual, &expected)) in loaded_field
        .size
        .iter()
        .zip(expected_field.size.iter())
        .enumerate()
    {
        close_scalar(actual, expected, 1.0e-6, &format!("{name}: size[{index}]"));
    }
    assert_eq!(
        loaded_field.data.len(),
        expected_field.data.len(),
        "{name}: data length"
    );
    for (index, (&actual, &expected)) in loaded_field
        .data
        .iter()
        .zip(expected_field.data.iter())
        .enumerate()
    {
        close_scalar(actual, expected, 1.0e-6, &format!("{name}: data[{index}]"));
    }

    let shape_geom = source
        .geoms_by_name
        .get(&format!("shape_{name}"))
        .map(|&index| &source.world.geoms[index])
        .unwrap_or_else(|| panic!("{name}: source shape geom is missing"));
    match (shape, shape_geom.shape) {
        ("sphere", GeomShape::Sphere { radius }) => close_scalar(
            radius,
            number(object(case, "radius"), "radius"),
            1.0e-6,
            &format!("{name}: radius"),
        ),
        (
            "capsule",
            GeomShape::Capsule {
                radius,
                half_height,
            },
        ) => {
            close_scalar(
                radius,
                number(object(case, "radius"), "radius"),
                1.0e-6,
                &format!("{name}: radius"),
            );
            close_scalar(
                half_height,
                number(object(case, "half_height"), "half_height"),
                1.0e-6,
                &format!("{name}: half_height"),
            );
        }
        ("box", GeomShape::Box { half_extents }) => {
            let expected = numbers(object(case, "half_extents"), "half_extents");
            close_scalar(
                half_extents.x,
                expected[0],
                1.0e-6,
                &format!("{name}: half x"),
            );
            close_scalar(
                half_extents.y,
                expected[1],
                1.0e-6,
                &format!("{name}: half y"),
            );
            close_scalar(
                half_extents.z,
                expected[2],
                1.0e-6,
                &format!("{name}: half z"),
            );
        }
        _ => panic!("{name}: source shape does not match JSON shape {shape}"),
    }
    let (parent_position, parent_orientation) = match shape_geom.attachment() {
        GeomAttach::Static => (Vec3::ZERO, Quat::IDENTITY),
        GeomAttach::Body(index) => {
            let body = &source.world.bodies[index];
            (body.position, body.orientation)
        }
        GeomAttach::Link(_, _) => panic!("{name}: source shape is attached to a tree link"),
    };
    let loaded_pose = geom_world_pose(shape_geom, parent_position, parent_orientation);
    close_vec(loaded_pose.position, expected_pose.position, 1.0e-5);
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        close_vec(loaded_pose.rotate(axis), expected_pose.rotate(axis), 1.0e-5);
    }
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
        let source_scene = load_mjcf_str(&source)
            .unwrap_or_else(|error| panic!("{name}: source XML does not load: {error}"));
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
        assert_source_matches_case(&source_scene, name, case, shape, &field, shape_pose);
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
        let penetration_bound = optional_number(case, "penetration_bound").unwrap_or(5.0e-3);
        if optional_string(case, "order") != Some("source") {
            let order = |a: &Contact, b: &Contact| {
                b.penetration
                    .total_cmp(&a.penetration)
                    .then_with(|| a.position_world.x.total_cmp(&b.position_world.x))
                    .then_with(|| a.position_world.y.total_cmp(&b.position_world.y))
            };
            expected.sort_by(order);
            actual.sort_by(order);
        }
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (actual, expected) in actual.iter().zip(expected) {
            close_vec(
                actual.position_world,
                expected.position_world,
                position_tolerance,
            );
            close_vec(actual.normal_world, expected.normal_world, normal_bound);
            assert!(
                (actual.penetration - expected.penetration).abs() <= penetration_bound,
                "{name}: {actual:?} != {expected:?}"
            );
        }
    }
}
