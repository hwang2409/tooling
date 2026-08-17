//! Pointwise MuJoCo 3.11 muscle-function conformance.
//!
//! Regenerate the fixture with `tools/capture_muscle_functions.py` in the
//! MuJoCo oracle environment. The fixture records its provenance.

use newt::actuator::{muscle_bias, muscle_dynamics, muscle_gain};
use newt::json::{self, Value};

fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
    let Value::Object(fields) = value else {
        panic!("expected object while reading {name}");
    };
    fields
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("missing field {name}"))
}

fn number(value: &Value, name: &str) -> f32 {
    let Value::Number(number) = value else {
        panic!("{name} must be a number");
    };
    *number as f32
}

fn array<const N: usize>(value: &Value, name: &str) -> [f32; N] {
    let Value::Array(values) = value else {
        panic!("{name} must be an array");
    };
    assert_eq!(values.len(), N, "{name} length");
    std::array::from_fn(|index| number(&values[index], name))
}

fn close(actual: f32, expected: f32, label: &str) -> f32 {
    let error = (actual - expected).abs();
    assert!(
        error <= 2.0e-5,
        "{label}: actual={actual:?}, expected={expected:?}, error={error:?}"
    );
    error
}

#[test]
fn muscle_functions_match_mujoco_grid_at_float_tier() {
    let document = json::parse(include_str!("references/muscle_function_grid.json"))
        .expect("muscle grid fixture must parse");
    let length_range = array::<2>(field(&document, "lengthrange"), "lengthrange");
    let acc0 = number(field(&document, "acc0"), "acc0");
    let params = array::<9>(field(&document, "params"), "params");
    let mut max_error = 0.0f32;

    let Value::Array(gain_cases) = field(&document, "gain") else {
        panic!("gain must be an array");
    };
    for (index, case) in gain_cases.iter().enumerate() {
        let length = number(field(case, "length"), "length");
        let velocity = number(field(case, "velocity"), "velocity");
        let expected = number(field(case, "value"), "value");
        max_error = max_error.max(close(
            muscle_gain(length, velocity, length_range, acc0, params),
            expected,
            &format!("gain[{index}]"),
        ));
    }

    let Value::Array(bias_cases) = field(&document, "bias") else {
        panic!("bias must be an array");
    };
    for (index, case) in bias_cases.iter().enumerate() {
        let length = number(field(case, "length"), "length");
        let expected = number(field(case, "value"), "value");
        max_error = max_error.max(close(
            muscle_bias(length, length_range, acc0, params),
            expected,
            &format!("bias[{index}]"),
        ));
    }

    let dynprm = array::<3>(field(&document, "dynprm"), "dynprm");
    let Value::Array(dynamics_cases) = field(&document, "dynamics") else {
        panic!("dynamics must be an array");
    };
    for (index, case) in dynamics_cases.iter().enumerate() {
        let activation = number(field(case, "activation"), "activation");
        let control = number(field(case, "control"), "control");
        let expected = number(field(case, "value"), "value");
        max_error = max_error.max(close(
            muscle_dynamics(control, activation, dynprm),
            expected,
            &format!("dynamics[{index}]"),
        ));
    }
    eprintln!("muscle function grid: maximum float error {max_error:.6e}");
}
