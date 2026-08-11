//! A small JSON value parser used by the glTF loader.
//!
//! The parser accepts JSON values only. It does not serialize values. Every
//! error contains the byte offset where parsing stopped. Nesting is limited to
//! 128 levels so hostile input cannot exhaust the stack.

use std::fmt::{Display, Formatter};

pub const MAX_DEPTH: usize = 128;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

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
        let mut values = Vec::new();
        if self.consume(b'}') {
            return Ok(Value::Object(values));
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
            values.push((key, self.value(depth)?));
            self.whitespace();
            if self.consume(b'}') {
                return Ok(Value::Object(values));
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
            parse(r#"{"a":[true,null,-1.25e2],"s":"a\n\u263a"}"#).unwrap(),
            Value::Object(vec![
                (
                    "a".into(),
                    Value::Array(vec![Value::Bool(true), Value::Null, Value::Number(-125.0)])
                ),
                ("s".into(), Value::String("a\n☺".into())),
            ])
        );
        assert_eq!(
            parse(r#""\uD83D\uDE00""#).unwrap(),
            Value::String("😀".into())
        );
    }

    #[test]
    fn malformed_input_returns_offsets() {
        for source in ["", "[", "{\"x\":", "\"\\q\"", "1.", "01", "[1,]"] {
            assert!(parse(source).is_err(), "{source}");
        }
        assert_eq!(parse("[}").unwrap_err().offset, 1);
    }

    #[test]
    fn rejects_deep_nesting_and_huge_numbers() {
        let source = "[".repeat(MAX_DEPTH + 2) + &"]".repeat(MAX_DEPTH + 2);
        assert!(parse(&source).is_err());
        assert!(parse("1e9999").is_err());
    }
}
