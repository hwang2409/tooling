//! Typed attribute filter AST and evaluator.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::AttrValue;

/// A v0 attribute filter.
///
/// The `op` field is the lowercase operator name when serialized as JSON. An
/// equality, inequality, or membership comparison requires matching attribute
/// types. Numeric ordering accepts either `Int` or `Float` on both sides;
/// strings, booleans, and string lists do not support ordering. Any other type
/// mismatch evaluates to `false`, including `Ne` (it does not become `true`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Filter {
    /// Match an attribute exactly.
    Eq { field: String, value: AttrValue },
    /// Match an attribute that differs from the supplied value.
    Ne { field: String, value: AttrValue },
    /// Match an attribute equal to any supplied value.
    In {
        field: String,
        values: Vec<AttrValue>,
    },
    /// Numeric less-than comparison.
    Lt { field: String, value: AttrValue },
    /// Numeric less-than-or-equal comparison.
    Lte { field: String, value: AttrValue },
    /// Numeric greater-than comparison.
    Gt { field: String, value: AttrValue },
    /// Numeric greater-than-or-equal comparison.
    Gte { field: String, value: AttrValue },
    /// Require every nested filter to match.
    And { filters: Vec<Filter> },
    /// Require at least one nested filter to match.
    Or { filters: Vec<Filter> },
}

impl Filter {
    /// Evaluate this filter against a document's attributes.
    pub fn eval(&self, attributes: &BTreeMap<String, AttrValue>) -> bool {
        match self {
            Self::Eq { field, value } => attributes
                .get(field)
                .is_some_and(|candidate| same_typed_value(candidate, value)),
            Self::Ne { field, value } => attributes
                .get(field)
                .is_some_and(|candidate| same_type(candidate, value) && candidate != value),
            Self::In { field, values } => attributes.get(field).is_some_and(|candidate| {
                values
                    .iter()
                    .any(|value| same_typed_value(candidate, value))
            }),
            Self::Lt { field, value } => {
                compare_numeric(attributes, field, value, |order| order.is_lt())
            }
            Self::Lte { field, value } => {
                compare_numeric(attributes, field, value, |order| order.is_le())
            }
            Self::Gt { field, value } => {
                compare_numeric(attributes, field, value, |order| order.is_gt())
            }
            Self::Gte { field, value } => {
                compare_numeric(attributes, field, value, |order| order.is_ge())
            }
            Self::And { filters } => filters.iter().all(|filter| filter.eval(attributes)),
            Self::Or { filters } => filters.iter().any(|filter| filter.eval(attributes)),
        }
    }
}

fn same_typed_value(left: &AttrValue, right: &AttrValue) -> bool {
    match (left, right) {
        (AttrValue::String(left), AttrValue::String(right)) => left == right,
        (AttrValue::Int(left), AttrValue::Int(right)) => left == right,
        (AttrValue::Float(left), AttrValue::Float(right)) => left == right,
        (AttrValue::Bool(left), AttrValue::Bool(right)) => left == right,
        (AttrValue::StringList(left), AttrValue::StringList(right)) => left == right,
        _ => false,
    }
}

fn same_type(left: &AttrValue, right: &AttrValue) -> bool {
    matches!(
        (left, right),
        (AttrValue::String(_), AttrValue::String(_))
            | (AttrValue::Int(_), AttrValue::Int(_))
            | (AttrValue::Float(_), AttrValue::Float(_))
            | (AttrValue::Bool(_), AttrValue::Bool(_))
            | (AttrValue::StringList(_), AttrValue::StringList(_))
    )
}

fn compare_numeric(
    attributes: &BTreeMap<String, AttrValue>,
    field: &str,
    expected: &AttrValue,
    predicate: impl FnOnce(std::cmp::Ordering) -> bool,
) -> bool {
    let Some(actual) = attributes.get(field).and_then(Numeric::from_attr) else {
        return false;
    };
    let Some(expected) = Numeric::from_attr(expected) else {
        return false;
    };
    actual.compare(&expected).is_some_and(predicate)
}

#[derive(Debug, Clone, Copy)]
enum Numeric {
    Int(i64),
    Float(f64),
}

impl Numeric {
    fn from_attr(value: &AttrValue) -> Option<Self> {
        match value {
            AttrValue::Int(value) => Some(Self::Int(*value)),
            AttrValue::Float(value) => Some(Self::Float(*value)),
            _ => None,
        }
    }

    fn compare(self, other: &Self) -> Option<Ordering> {
        match (self, *other) {
            (Self::Int(left), Self::Int(right)) => Some(left.cmp(&right)),
            (Self::Float(left), Self::Float(right)) => left.partial_cmp(&right),
            (Self::Int(left), Self::Float(right)) => compare_int_float(left, right),
            (Self::Float(left), Self::Int(right)) => {
                compare_int_float(right, left).map(Ordering::reverse)
            }
        }
    }
}

/// Compare a finite or infinite f64 against an i64 without rounding the
/// integer through f64. This matters for integers just above 2^53, where f64
/// cannot represent every consecutive integer. NaN is unordered and makes a
/// numeric filter evaluate false.
fn compare_int_float(integer: i64, float: f64) -> Option<Ordering> {
    if float.is_nan() {
        return None;
    }
    if float == f64::INFINITY {
        return Some(Ordering::Less);
    }
    if float == f64::NEG_INFINITY {
        return Some(Ordering::Greater);
    }

    let float_magnitude = float.abs();
    if float_magnitude == 0.0 {
        return Some(integer.cmp(&0));
    }
    if integer == 0 {
        return Some(if float.is_sign_negative() {
            Ordering::Greater
        } else {
            Ordering::Less
        });
    }

    let integer_negative = integer.is_negative();
    let float_negative = float.is_sign_negative();
    if integer_negative != float_negative {
        return Some(if integer_negative {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }

    let magnitude = compare_unsigned_magnitude(integer.unsigned_abs(), float_magnitude);
    Some(if integer_negative {
        magnitude.reverse()
    } else {
        magnitude
    })
}

fn compare_unsigned_magnitude(integer: u64, float: f64) -> Ordering {
    debug_assert!(float.is_finite() && float > 0.0);

    let bits = float.to_bits();
    let raw_exponent = ((bits >> 52) & 0x7ff) as u16;
    let fraction = bits & ((1u64 << 52) - 1);
    if raw_exponent == 0 {
        // A positive subnormal is smaller than every positive integer.
        return Ordering::Greater;
    }

    let exponent = i32::from(raw_exponent) - 1023;
    let integer_most_significant_bit = 63 - integer.leading_zeros();
    if (integer_most_significant_bit as i32) > exponent {
        return Ordering::Greater;
    }
    if (integer_most_significant_bit as i32) < exponent {
        return Ordering::Less;
    }

    let significand = (1u64 << 52) | fraction;
    let shift = exponent - 52;
    if shift >= 0 {
        let float_integer = significand << shift as u32;
        integer.cmp(&float_integer)
    } else {
        let divisor = 1u64 << (-shift) as u32;
        let float_integer = significand / divisor;
        match integer.cmp(&float_integer) {
            Ordering::Equal if !significand.is_multiple_of(divisor) => Ordering::Less,
            order => order,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::Filter;
    use crate::AttrValue;

    fn attributes() -> BTreeMap<String, AttrValue> {
        BTreeMap::from([
            ("name".to_owned(), AttrValue::String("Ada".to_owned())),
            ("age".to_owned(), AttrValue::Int(37)),
            ("score".to_owned(), AttrValue::Float(9.5)),
            ("active".to_owned(), AttrValue::Bool(true)),
            (
                "tags".to_owned(),
                AttrValue::StringList(vec!["rust".to_owned(), "search".to_owned()]),
            ),
        ])
    }

    #[test]
    fn filter_evaluation_matrix_includes_type_mismatches() {
        let attrs = attributes();
        assert!(Filter::Eq {
            field: "name".to_owned(),
            value: AttrValue::String("Ada".to_owned()),
        }
        .eval(&attrs));
        assert!(!Filter::Eq {
            field: "name".to_owned(),
            value: AttrValue::Int(1),
        }
        .eval(&attrs));
        assert!(Filter::Ne {
            field: "age".to_owned(),
            value: AttrValue::Int(38),
        }
        .eval(&attrs));
        assert!(!Filter::Ne {
            field: "age".to_owned(),
            value: AttrValue::Float(37.0),
        }
        .eval(&attrs));
        assert!(Filter::In {
            field: "name".to_owned(),
            values: vec![
                AttrValue::String("Grace".to_owned()),
                AttrValue::String("Ada".to_owned()),
            ],
        }
        .eval(&attrs));
        assert!(Filter::Lt {
            field: "age".to_owned(),
            value: AttrValue::Float(37.5),
        }
        .eval(&attrs));
        assert!(Filter::Gte {
            field: "score".to_owned(),
            value: AttrValue::Int(9),
        }
        .eval(&attrs));
        assert!(!Filter::Gt {
            field: "name".to_owned(),
            value: AttrValue::String("A".to_owned()),
        }
        .eval(&attrs));
        assert!(Filter::And {
            filters: vec![
                Filter::Eq {
                    field: "active".to_owned(),
                    value: AttrValue::Bool(true),
                },
                Filter::Gte {
                    field: "age".to_owned(),
                    value: AttrValue::Int(18),
                },
            ],
        }
        .eval(&attrs));
        assert!(Filter::Or {
            filters: vec![
                Filter::Eq {
                    field: "name".to_owned(),
                    value: AttrValue::String("Nope".to_owned()),
                },
                Filter::Eq {
                    field: "name".to_owned(),
                    value: AttrValue::String("Ada".to_owned()),
                },
            ],
        }
        .eval(&attrs));
        assert!(!Filter::Eq {
            field: "missing".to_owned(),
            value: AttrValue::String("value".to_owned()),
        }
        .eval(&attrs));
    }

    #[test]
    fn filters_are_json_deserializable() {
        let filter: Filter = serde_json::from_value(json!({
            "op": "and",
            "filters": [
                {"op": "eq", "field": "active", "value": {"Bool": true}},
                {"op": "gte", "field": "age", "value": {"Int": 18}}
            ]
        }))
        .expect("filter JSON");
        assert!(filter.eval(&attributes()));
    }

    #[test]
    fn serialization_roundtrip_preserves_evaluation() {
        let filter = Filter::Or {
            filters: vec![Filter::Eq {
                field: "name".to_owned(),
                value: AttrValue::String("Ada".to_owned()),
            }],
        };
        let decoded: Filter =
            serde_json::from_slice(&serde_json::to_vec(&filter).expect("serialize filter"))
                .expect("deserialize filter");
        assert_eq!(decoded, filter);
        assert_eq!(decoded.eval(&attributes()), filter.eval(&attributes()));
    }

    #[test]
    fn mixed_numeric_comparison_is_exact_at_the_two_to_the_fifty_third_boundary() {
        let mut attrs = BTreeMap::new();
        attrs.insert(
            "large-int".to_owned(),
            AttrValue::Int(9_007_199_254_740_993),
        );
        attrs.insert(
            "large-float".to_owned(),
            AttrValue::Float(9_007_199_254_740_992.0),
        );

        assert!(Filter::Gt {
            field: "large-int".to_owned(),
            value: AttrValue::Float(9_007_199_254_740_992.0),
        }
        .eval(&attrs));
        assert!(!Filter::Lte {
            field: "large-int".to_owned(),
            value: AttrValue::Float(9_007_199_254_740_992.0),
        }
        .eval(&attrs));
        assert!(Filter::Lt {
            field: "large-float".to_owned(),
            value: AttrValue::Int(9_007_199_254_740_993),
        }
        .eval(&attrs));
        assert!(!Filter::Gte {
            field: "large-float".to_owned(),
            value: AttrValue::Int(9_007_199_254_740_993),
        }
        .eval(&attrs));
    }
}
