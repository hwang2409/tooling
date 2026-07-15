//! Shared data types used by the storage engine.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// A document stored in a namespace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Doc {
    /// The document's unique identifier within its namespace.
    pub id: String,
    /// The optional vector associated with the document.
    pub vector: Option<Vec<f32>>,
    /// Arbitrary typed document attributes.
    pub attributes: BTreeMap<String, AttrValue>,
}

/// A supported document attribute value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttrValue {
    /// A UTF-8 string.
    String(String),
    /// A signed 64-bit integer.
    Int(i64),
    /// A 64-bit floating-point number.
    Float(f64),
    /// A boolean.
    Bool(bool),
    /// A list of UTF-8 strings.
    StringList(Vec<String>),
}
