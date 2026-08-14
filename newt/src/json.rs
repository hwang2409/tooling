//! Hand-written JSON value parser.
//!
//! Zero dependencies (no `serde`, no `libm`). Accepts JSON values only; does
//! not serialize. Every error carries the byte offset at which parsing failed.
//! Nesting is capped at [`MAX_DEPTH`] so a hostile document cannot exhaust the
//! stack.
//!
//! The [`Value`] tree uses an insertion-ordered `Vec<(String, Value)>` for
//! objects. Duplicate keys are preserved (last-write-wins is the loader's
//! problem, not the parser's); the [`Value::field`] helper returns the FIRST
//! occurrence. Object iteration order is stable — the same document parses to
//! the same tree byte-for-byte.
//!
//! # Style match
//!
//! This module follows the same shape as `chimy2/src/json.rs` (byte-cursor
//! recursive descent with a strict number grammar and Unicode-escape
//! surrogate handling) but is a full copy — the engine crate imports no
//! dependencies, not even chimy2, per the spec.

use std::fmt::{Display, Formatter};

/// Maximum nesting depth (arrays + objects) accepted by [`parse`]. The
/// cap prevents runaway recursion on hostile input.
pub const MAX_DEPTH: usize = 128;

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    /// Insertion-ordered field list. Duplicate keys are preserved.
    Object(Vec<(String, Value)>),
}

impl Value {
    /// Type name for error messages: `"null" | "bool" | "number" | ...`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Array(_) => "array",
            Value::Object(_) => "object",
        }
    }

    /// Look up an object field by name. Returns `None` if the value is not an
    /// object or the field is absent. Returns the FIRST occurrence when
    /// duplicate keys are present.
    pub fn field(&self, name: &str) -> Option<&Value> {
        match self {
            Value::Object(fields) => fields.iter().find(|(k, _)| k == name).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// Parse error with the byte offset where parsing stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub offset: usize,
    pub message: String,
}

impl Error {
    fn new(offset: usize, message: impl Into<String>) -> Self {
        Self {
            offset,
            message: message.into(),
        }
    }
}

impl Display for Error {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for Error {}

/// Parse a UTF-8 string as a single JSON value. Trailing whitespace is
/// tolerated; any non-whitespace after the top-level value is an error.
pub fn parse(source: &str) -> Result<Value, Error> {
    let mut parser = Parser {
        bytes: source.as_bytes(),
        position: 0,
    };
    let value = parser.value(0)?;
    parser.whitespace();
    if parser.position != parser.bytes.len() {
        return Err(Error::new(parser.position, "trailing characters"));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Parser<'a> {
    fn value(&mut self, depth: usize) -> Result<Value, Error> {
        self.whitespace();
        if depth > MAX_DEPTH {
            return Err(Error::new(self.position, "maximum nesting depth exceeded"));
        }
        match self.peek() {
            Some(b'n') => self.literal(b"null", Value::Null),
            Some(b't') => self.literal(b"true", Value::Bool(true)),
            Some(b'f') => self.literal(b"false", Value::Bool(false)),
            Some(b'"') => self.string().map(Value::String),
            Some(b'[') => self.array(depth + 1),
            Some(b'{') => self.object(depth + 1),
            Some(b'-' | b'0'..=b'9') => self.number().map(Value::Number),
            Some(_) => Err(Error::new(self.position, "expected a JSON value")),
            None => Err(Error::new(self.position, "unexpected end of input")),
        }
    }

    fn literal(&mut self, expected: &[u8], value: Value) -> Result<Value, Error> {
        let start = self.position;
        if self.bytes.get(start..start + expected.len()) != Some(expected) {
            return Err(Error::new(start, "invalid literal"));
        }
        self.position += expected.len();
        Ok(value)
    }

    fn array(&mut self, depth: usize) -> Result<Value, Error> {
        self.position += 1;
        self.whitespace();
        let mut values = Vec::new();
        if self.consume(b']') {
            return Ok(Value::Array(values));
        }
        loop {
            values.push(self.value(depth)?);
            self.whitespace();
            if self.consume(b']') {
                return Ok(Value::Array(values));
            }
            if !self.consume(b',') {
                return Err(Error::new(self.position, "expected ',' or ']'"));
            }
            self.whitespace();
            if self.peek() == Some(b']') {
                return Err(Error::new(self.position, "trailing comma in array"));
            }
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, Error> {
        self.position += 1;
        self.whitespace();
        let mut fields = Vec::new();
        if self.consume(b'}') {
            return Ok(Value::Object(fields));
        }
        loop {
            self.whitespace();
            if self.peek() != Some(b'"') {
                return Err(Error::new(self.position, "object key must be a string"));
            }
            let key = self.string()?;
            self.whitespace();
            if !self.consume(b':') {
                return Err(Error::new(self.position, "expected ':' after object key"));
            }
            fields.push((key, self.value(depth)?));
            self.whitespace();
            if self.consume(b'}') {
                return Ok(Value::Object(fields));
            }
            if !self.consume(b',') {
                return Err(Error::new(self.position, "expected ',' or '}'"));
            }
            self.whitespace();
            if self.peek() == Some(b'}') {
                return Err(Error::new(self.position, "trailing comma in object"));
            }
        }
    }

    fn string(&mut self) -> Result<String, Error> {
        let start = self.position;
        if !self.consume(b'"') {
            return Err(Error::new(start, "expected string"));
        }
        let mut result = String::new();
        loop {
            let byte = self
                .next()
                .ok_or_else(|| Error::new(self.position, "unterminated string"))?;
            match byte {
                b'"' => return Ok(result),
                b'\\' => result.push(self.escape()?),
                0..=0x1f => return Err(Error::new(self.position - 1, "control byte in string")),
                byte if byte < 0x80 => result.push(byte as char),
                byte => {
                    let width = utf8_width(byte)
                        .ok_or_else(|| Error::new(self.position - 1, "invalid UTF-8 in string"))?;
                    let tail = self
                        .position
                        .checked_add(width - 1)
                        .ok_or_else(|| Error::new(self.position - 1, "string length overflow"))?;
                    if tail > self.bytes.len() {
                        return Err(Error::new(self.position - 1, "truncated UTF-8 in string"));
                    }
                    let slice = &self.bytes[self.position - 1..tail];
                    let text = std::str::from_utf8(slice)
                        .map_err(|_| Error::new(self.position - 1, "invalid UTF-8 in string"))?;
                    result.push_str(text);
                    self.position = tail;
                }
            }
        }
    }

    fn escape(&mut self) -> Result<char, Error> {
        let offset = self.position;
        let byte = self
            .next()
            .ok_or_else(|| Error::new(offset, "truncated escape"))?;
        match byte {
            b'"' => Ok('"'),
            b'\\' => Ok('\\'),
            b'/' => Ok('/'),
            b'b' => Ok('\u{8}'),
            b'f' => Ok('\u{c}'),
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'u' => {
                let first = self.hex_u16()?;
                if (0xd800..=0xdbff).contains(&first) {
                    if self.next() != Some(b'\\') || self.next() != Some(b'u') {
                        return Err(Error::new(self.position, "high surrogate needs a pair"));
                    }
                    let second = self.hex_u16()?;
                    if !(0xdc00..=0xdfff).contains(&second) {
                        return Err(Error::new(self.position - 4, "invalid low surrogate"));
                    }
                    let code = 0x1_0000
                        + ((u32::from(first) - 0xd800) << 10)
                        + (u32::from(second) - 0xdc00);
                    char::from_u32(code).ok_or_else(|| Error::new(offset, "invalid Unicode escape"))
                } else if (0xdc00..=0xdfff).contains(&first) {
                    Err(Error::new(offset, "unexpected low surrogate"))
                } else {
                    char::from_u32(u32::from(first))
                        .ok_or_else(|| Error::new(offset, "invalid Unicode escape"))
                }
            }
            _ => Err(Error::new(offset, "invalid escape")),
        }
    }

    fn hex_u16(&mut self) -> Result<u16, Error> {
        let start = self.position;
        let mut value = 0u16;
        for _ in 0..4 {
            let byte = self
                .next()
                .ok_or_else(|| Error::new(self.position, "truncated Unicode escape"))?;
            value = value
                .checked_mul(16)
                .and_then(|value| value.checked_add(hex_value(byte)?))
                .ok_or_else(|| Error::new(start, "Unicode escape overflow"))?;
        }
        Ok(value)
    }

    fn number(&mut self) -> Result<f64, Error> {
        let start = self.position;
        self.consume(b'-');
        match self.peek() {
            Some(b'0') => {
                self.position += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(Error::new(self.position, "leading zero in number"));
                }
            }
            Some(b'1'..=b'9') => {
                self.position += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.position += 1;
                }
            }
            _ => return Err(Error::new(self.position, "invalid number")),
        }
        if self.consume(b'.') {
            let fraction = self.position;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
            if self.position == fraction {
                return Err(Error::new(self.position, "fraction needs digits"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.position += 1;
            self.consume(b'+');
            self.consume(b'-');
            let exponent = self.position;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.position += 1;
            }
            if self.position == exponent {
                return Err(Error::new(self.position, "exponent needs digits"));
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.position])
            .map_err(|_| Error::new(start, "invalid number"))?;
        let number = text
            .parse::<f64>()
            .map_err(|_| Error::new(start, "number is out of range"))?;
        if !number.is_finite() {
            return Err(Error::new(start, "number is not finite"));
        }
        Ok(number)
    }

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn next(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.position += 1;
        Some(byte)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }
}

fn utf8_width(byte: u8) -> Option<usize> {
    match byte {
        0xc2..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf4 => Some(4),
        _ => None,
    }
}

fn hex_value(byte: u8) -> Option<u16> {
    match byte {
        b'0'..=b'9' => Some(u16::from(byte - b'0')),
        b'a'..=b'f' => Some(u16::from(byte - b'a' + 10)),
        b'A'..=b'F' => Some(u16::from(byte - b'A' + 10)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_values_and_escapes() {
        assert_eq!(
            parse(r#"{"a":[true,null,-1.25e2],"s":"a\n☺"}"#).unwrap(),
            Value::Object(vec![
                (
                    "a".into(),
                    Value::Array(vec![Value::Bool(true), Value::Null, Value::Number(-125.0)])
                ),
                ("s".into(), Value::String("a\n\u{263a}".into())),
            ])
        );
        assert_eq!(parse(r#""😀""#).unwrap(), Value::String("\u{1f600}".into()));
    }

    #[test]
    fn number_edge_cases() {
        assert_eq!(parse("0").unwrap(), Value::Number(0.0));
        assert_eq!(parse("-0").unwrap(), Value::Number(0.0));
        assert_eq!(parse("1.5e-3").unwrap(), Value::Number(0.0015));
        assert_eq!(parse("1E+3").unwrap(), Value::Number(1000.0));
        assert_eq!(parse("-2.5").unwrap(), Value::Number(-2.5));
    }

    #[test]
    fn nested_arrays_and_objects() {
        let value = parse(r#"[{"k":[1,2,{"m":[]}]}]"#).unwrap();
        // Structural spot-check: outer array has one object; the "k" field is
        // an array of length 3 whose last element is an object with one empty
        // array field.
        let Value::Array(outer) = &value else {
            panic!("expected array");
        };
        assert_eq!(outer.len(), 1);
        let Value::Object(fields) = &outer[0] else {
            panic!("expected object");
        };
        assert_eq!(fields[0].0, "k");
    }

    #[test]
    fn malformed_input_returns_offsets() {
        for source in [
            "",
            "[",
            "{\"x\":",
            "\"\\q\"",
            "1.",
            "01",
            "[1,]",
            "nope",
            "{,}",
            "{\"a\":1,}",
            "[1 2]",
            "1e",
            "-",
            ".5",
            "\"\\\"",
        ] {
            assert!(parse(source).is_err(), "{source} should be an error");
        }
        assert_eq!(parse("[}").unwrap_err().offset, 1);
        // Position pointer lands on the offending character, not the value
        // start.
        let err = parse("1.").unwrap_err();
        assert_eq!(err.offset, 2);
    }

    #[test]
    fn rejects_deep_nesting_and_huge_numbers() {
        let source = "[".repeat(MAX_DEPTH + 2) + &"]".repeat(MAX_DEPTH + 2);
        assert!(parse(&source).is_err());
        assert!(parse("1e9999").is_err(), "1e9999 should be non-finite");
    }

    #[test]
    fn rejects_unterminated_string() {
        let err = parse(r#""abc"#).unwrap_err();
        assert!(err.message.contains("unterminated"));
    }

    #[test]
    fn rejects_control_byte_in_string() {
        let source = "\"a\x01b\"";
        assert!(parse(source).is_err());
    }

    #[test]
    fn trailing_content_after_top_value_is_error() {
        let err = parse("42 garbage").unwrap_err();
        assert!(err.message.contains("trailing"));
    }

    #[test]
    fn field_returns_first_occurrence() {
        let v = parse(r#"{"a":1,"a":2,"b":3}"#).unwrap();
        // First-write-wins in the parser; last-write-wins would be a loader
        // policy decision (we reject duplicates in the loader anyway).
        assert_eq!(v.field("a"), Some(&Value::Number(1.0)));
        assert_eq!(v.field("b"), Some(&Value::Number(3.0)));
        assert_eq!(v.field("missing"), None);
    }

    #[test]
    fn type_name_is_stable() {
        assert_eq!(Value::Null.type_name(), "null");
        assert_eq!(Value::Bool(true).type_name(), "bool");
        assert_eq!(Value::Number(1.0).type_name(), "number");
        assert_eq!(Value::String("x".into()).type_name(), "string");
        assert_eq!(Value::Array(vec![]).type_name(), "array");
        assert_eq!(Value::Object(vec![]).type_name(), "object");
    }
}
