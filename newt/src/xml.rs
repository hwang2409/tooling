//! Hand-written XML parser.
//!
//! Zero dependencies. Accepts a subset of XML wide enough for MJCF fixtures:
//! elements, attributes (single- or double-quoted), text nodes, comments,
//! self-closing tags, entity escapes (`&lt; &gt; &amp; &quot; &apos;` and
//! numeric `&#N; / &#xH;`), the optional `<?xml ... ?>` declaration, and
//! processing instructions (skipped). Every error carries the byte offset at
//! which parsing failed.
//!
//! Not supported: DTDs (`<!DOCTYPE ...>` errors out), CDATA sections, XML
//! namespaces beyond raw string names, mixed-content interleaving semantics
//! (text nodes are preserved but the parser does not distinguish element-
//! only bodies from mixed bodies — the MJCF loader ignores stray whitespace
//! and rejects unexpected text).
//!
//! Nesting is capped at [`MAX_DEPTH`] so a hostile document cannot exhaust
//! the stack.
//!
//! # Style match
//!
//! Follows the same recursive-descent shape as [`crate::json`]: a byte-
//! cursor `Parser` with `peek` / `next` / `consume` primitives, no lookahead
//! beyond the current byte plus fixed-width lookaheads for delimiters like
//! `<!--` and `-->`.

use std::fmt::{Display, Formatter};

/// Maximum element nesting depth accepted by [`parse`].
pub const MAX_DEPTH: usize = 128;

/// A parsed XML node.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// An element: `<name attr="v">...</name>` or `<name/>`.
    Element(Element),
    /// Character data between elements. Entity escapes are already decoded.
    Text(String),
}

/// A parsed XML element. Attributes are kept in source order; the loader
/// walks the list linearly (matches the JSON loader's field iteration).
#[derive(Clone, Debug, PartialEq)]
pub struct Element {
    pub name: String,
    /// Attributes in source order. Duplicate attribute names on the same
    /// element are rejected by [`parse`] with an error at the second
    /// occurrence.
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

impl Element {
    /// Look up an attribute by name. Returns the first match (attribute
    /// duplicates were rejected at parse time, so first == only).
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    /// Iterator over child elements (skipping text nodes).
    pub fn child_elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|n| match n {
            Node::Element(e) => Some(e),
            Node::Text(_) => None,
        })
    }
}

/// Parse error with the byte offset at which parsing stopped.
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "byte {}: {}", self.offset, self.message)
    }
}

impl std::error::Error for Error {}

/// Parse a UTF-8 string as an XML document. Returns the single root element.
///
/// The optional `<?xml ... ?>` prolog is accepted and ignored. Comments and
/// processing instructions outside the root element are permitted and
/// skipped. Any content other than whitespace before the first root element
/// tag or after its closing tag is an error.
pub fn parse(source: &str) -> Result<Element, Error> {
    let mut parser = Parser {
        bytes: source.as_bytes(),
        position: 0,
    };
    parser.skip_prolog()?;
    let root = parser.element(0)?;
    parser.skip_epilog()?;
    if parser.position != parser.bytes.len() {
        return Err(Error::new(
            parser.position,
            "trailing content after root element",
        ));
    }
    Ok(root)
}

struct Parser<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Parser<'a> {
    // ---- top-level prolog / epilog handling -----------------------------

    /// Skip whitespace, an optional XML declaration `<?xml ... ?>`, and
    /// any leading comments or processing instructions. Errors on a
    /// `<!DOCTYPE ...>` (unsupported).
    fn skip_prolog(&mut self) -> Result<(), Error> {
        self.whitespace();
        // Optional `<?xml ... ?>` — only permitted at the very start.
        if self.starts_with(b"<?xml") {
            self.skip_until(b"?>", "unterminated XML declaration")?;
        }
        loop {
            self.whitespace();
            if self.starts_with(b"<!--") {
                self.skip_comment()?;
            } else if self.starts_with(b"<!DOCTYPE") {
                return Err(Error::new(
                    self.position,
                    "<!DOCTYPE ...> is not supported by this parser",
                ));
            } else if self.starts_with(b"<!") {
                return Err(Error::new(
                    self.position,
                    "unsupported '<!' construct (only comments are accepted here)",
                ));
            } else if self.starts_with(b"<?") {
                self.skip_until(b"?>", "unterminated processing instruction")?;
            } else {
                return Ok(());
            }
        }
    }

    /// Skip trailing whitespace, comments, and processing instructions after
    /// the root element.
    fn skip_epilog(&mut self) -> Result<(), Error> {
        loop {
            self.whitespace();
            if self.starts_with(b"<!--") {
                self.skip_comment()?;
            } else if self.starts_with(b"<?") {
                self.skip_until(b"?>", "unterminated processing instruction")?;
            } else {
                return Ok(());
            }
        }
    }

    // ---- element / children ---------------------------------------------

    fn element(&mut self, depth: usize) -> Result<Element, Error> {
        if depth > MAX_DEPTH {
            return Err(Error::new(
                self.position,
                "maximum XML nesting depth exceeded",
            ));
        }
        let tag_start = self.position;
        if !self.consume(b'<') {
            return Err(Error::new(tag_start, "expected '<' to start an element"));
        }
        if matches!(self.peek(), Some(b'/' | b'!' | b'?')) {
            return Err(Error::new(
                tag_start,
                "expected element name (found close/comment/PI)",
            ));
        }
        let name = self.parse_name()?;
        let mut attrs: Vec<(String, String)> = Vec::new();
        loop {
            let ws_before = self.skip_attr_whitespace();
            match self.peek() {
                Some(b'/') => {
                    self.position += 1;
                    if !self.consume(b'>') {
                        return Err(Error::new(
                            self.position,
                            "expected '>' after '/' in self-closing tag",
                        ));
                    }
                    return Ok(Element {
                        name,
                        attrs,
                        children: Vec::new(),
                    });
                }
                Some(b'>') => {
                    self.position += 1;
                    break;
                }
                Some(_) => {
                    if !ws_before {
                        return Err(Error::new(
                            self.position,
                            "expected whitespace before next attribute",
                        ));
                    }
                    let (key, value) = self.parse_attribute()?;
                    if attrs.iter().any(|(k, _)| k == &key) {
                        return Err(Error::new(
                            self.position,
                            format!("duplicate attribute \"{key}\" on <{name}>"),
                        ));
                    }
                    attrs.push((key, value));
                }
                None => {
                    return Err(Error::new(self.position, "unexpected end of input in tag"));
                }
            }
        }
        // Element body — parse children until we hit the matching close tag.
        let mut children: Vec<Node> = Vec::new();
        loop {
            if self.starts_with(b"<!--") {
                self.skip_comment()?;
                continue;
            }
            if self.starts_with(b"<?") {
                self.skip_until(b"?>", "unterminated processing instruction")?;
                continue;
            }
            if self.starts_with(b"</") {
                self.position += 2;
                let close = self.parse_name()?;
                if close != name {
                    return Err(Error::new(
                        self.position,
                        format!("closing tag </{close}> does not match <{name}>"),
                    ));
                }
                self.whitespace();
                if !self.consume(b'>') {
                    return Err(Error::new(
                        self.position,
                        format!("expected '>' after </{close}"),
                    ));
                }
                return Ok(Element {
                    name,
                    attrs,
                    children,
                });
            }
            if self.starts_with(b"<") {
                let child = self.element(depth + 1)?;
                children.push(Node::Element(child));
                continue;
            }
            // Text run up to next `<`, decoding entities.
            let text = self.parse_text()?;
            if !text.is_empty() {
                children.push(Node::Text(text));
            }
            if self.position == self.bytes.len() {
                return Err(Error::new(
                    self.position,
                    format!("unexpected end of input inside <{name}>"),
                ));
            }
        }
    }

    // ---- names, attributes, text ----------------------------------------

    fn parse_name(&mut self) -> Result<String, Error> {
        let start = self.position;
        match self.peek() {
            Some(b) if is_name_start(b) => self.position += 1,
            _ => return Err(Error::new(start, "expected an XML name")),
        }
        while matches!(self.peek(), Some(b) if is_name_char(b)) {
            self.position += 1;
        }
        let slice = &self.bytes[start..self.position];
        std::str::from_utf8(slice)
            .map(|s| s.to_string())
            .map_err(|_| Error::new(start, "invalid UTF-8 in name"))
    }

    fn parse_attribute(&mut self) -> Result<(String, String), Error> {
        let key = self.parse_name()?;
        self.whitespace();
        if !self.consume(b'=') {
            return Err(Error::new(
                self.position,
                format!("expected '=' after attribute name \"{key}\""),
            ));
        }
        self.whitespace();
        let quote = match self.next() {
            Some(b @ (b'"' | b'\'')) => b,
            Some(_) => {
                return Err(Error::new(
                    self.position - 1,
                    format!("attribute \"{key}\" value must be quoted with \" or '"),
                ));
            }
            None => {
                return Err(Error::new(
                    self.position,
                    format!("unexpected end of input in attribute \"{key}\""),
                ));
            }
        };
        let mut value = String::new();
        loop {
            let byte = self.next().ok_or_else(|| {
                Error::new(
                    self.position,
                    format!("unterminated attribute value for \"{key}\""),
                )
            })?;
            if byte == quote {
                return Ok((key, value));
            }
            if byte == b'<' {
                return Err(Error::new(
                    self.position - 1,
                    format!("'<' is not allowed inside attribute \"{key}\""),
                ));
            }
            if byte == b'&' {
                value.push(self.decode_entity()?);
            } else if byte < 0x80 {
                value.push(byte as char);
            } else {
                self.push_utf8_continuation(&mut value, byte)?;
            }
        }
    }

    fn parse_text(&mut self) -> Result<String, Error> {
        let mut out = String::new();
        loop {
            match self.peek() {
                None | Some(b'<') => return Ok(out),
                Some(b'&') => {
                    self.position += 1;
                    out.push(self.decode_entity()?);
                }
                Some(byte) => {
                    self.position += 1;
                    if byte < 0x80 {
                        out.push(byte as char);
                    } else {
                        self.push_utf8_continuation(&mut out, byte)?;
                    }
                }
            }
        }
    }

    /// Read the tail of a `&...;` entity reference (the `&` is already
    /// consumed) and append the decoded character.
    fn decode_entity(&mut self) -> Result<char, Error> {
        let start = self.position - 1;
        // Buffer the raw entity body up to and including the trailing `;`.
        let mut body = String::new();
        loop {
            let byte = self.next().ok_or_else(|| {
                Error::new(self.position, "entity reference is not terminated with ';'")
            })?;
            if byte == b';' {
                break;
            }
            if body.len() > 16 {
                return Err(Error::new(start, "entity reference is too long"));
            }
            body.push(byte as char);
        }
        match body.as_str() {
            "lt" => Ok('<'),
            "gt" => Ok('>'),
            "amp" => Ok('&'),
            "quot" => Ok('"'),
            "apos" => Ok('\''),
            other if other.starts_with("#x") || other.starts_with("#X") => {
                let hex = &other[2..];
                if hex.is_empty() {
                    return Err(Error::new(start, "empty hex character reference"));
                }
                let mut code: u32 = 0;
                for byte in hex.bytes() {
                    let digit = match byte {
                        b'0'..=b'9' => u32::from(byte - b'0'),
                        b'a'..=b'f' => u32::from(byte - b'a' + 10),
                        b'A'..=b'F' => u32::from(byte - b'A' + 10),
                        _ => {
                            return Err(Error::new(
                                start,
                                format!("invalid hex character reference \"&{other};\""),
                            ));
                        }
                    };
                    code = code
                        .checked_mul(16)
                        .and_then(|v| v.checked_add(digit))
                        .ok_or_else(|| Error::new(start, "hex character reference overflow"))?;
                }
                char::from_u32(code).ok_or_else(|| {
                    Error::new(start, format!("invalid Unicode scalar in \"&{other};\""))
                })
            }
            other if other.starts_with('#') => {
                let dec = &other[1..];
                if dec.is_empty() {
                    return Err(Error::new(start, "empty decimal character reference"));
                }
                let mut code: u32 = 0;
                for byte in dec.bytes() {
                    let digit = match byte {
                        b'0'..=b'9' => u32::from(byte - b'0'),
                        _ => {
                            return Err(Error::new(
                                start,
                                format!("invalid decimal character reference \"&{other};\""),
                            ));
                        }
                    };
                    code = code
                        .checked_mul(10)
                        .and_then(|v| v.checked_add(digit))
                        .ok_or_else(|| Error::new(start, "decimal character reference overflow"))?;
                }
                char::from_u32(code).ok_or_else(|| {
                    Error::new(start, format!("invalid Unicode scalar in \"&{other};\""))
                })
            }
            other => Err(Error::new(
                start,
                format!("unknown entity reference \"&{other};\""),
            )),
        }
    }

    /// Append `first` (a UTF-8 leading byte) plus its continuation bytes to
    /// `out`, validating the UTF-8 sequence.
    fn push_utf8_continuation(&mut self, out: &mut String, first: u8) -> Result<(), Error> {
        let width = utf8_width(first)
            .ok_or_else(|| Error::new(self.position - 1, "invalid UTF-8 leading byte"))?;
        let tail = self
            .position
            .checked_add(width - 1)
            .ok_or_else(|| Error::new(self.position - 1, "UTF-8 length overflow"))?;
        if tail > self.bytes.len() {
            return Err(Error::new(self.position - 1, "truncated UTF-8"));
        }
        let slice = &self.bytes[self.position - 1..tail];
        let text = std::str::from_utf8(slice)
            .map_err(|_| Error::new(self.position - 1, "invalid UTF-8 sequence"))?;
        out.push_str(text);
        self.position = tail;
        Ok(())
    }

    // ---- micro-helpers --------------------------------------------------

    fn whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.position += 1;
        }
    }

    /// Skip whitespace between attributes; return `true` if any bytes were
    /// consumed. The caller uses this to enforce that at least one
    /// whitespace byte separates attributes on the same tag.
    fn skip_attr_whitespace(&mut self) -> bool {
        let start = self.position;
        self.whitespace();
        self.position > start
    }

    fn skip_comment(&mut self) -> Result<(), Error> {
        // Caller has already peeked `<!--`.
        self.position += 4;
        loop {
            if self.starts_with(b"-->") {
                self.position += 3;
                return Ok(());
            }
            if self.starts_with(b"--") {
                return Err(Error::new(
                    self.position,
                    "'--' is not allowed inside an XML comment",
                ));
            }
            if self.next().is_none() {
                return Err(Error::new(self.position, "unterminated comment"));
            }
        }
    }

    fn skip_until(&mut self, marker: &[u8], msg: &'static str) -> Result<(), Error> {
        while !self.starts_with(marker) {
            if self.next().is_none() {
                return Err(Error::new(self.position, msg));
            }
        }
        self.position += marker.len();
        Ok(())
    }

    fn starts_with(&self, prefix: &[u8]) -> bool {
        self.bytes.get(self.position..self.position + prefix.len()) == Some(prefix)
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

fn is_name_start(b: u8) -> bool {
    matches!(b, b'A'..=b'Z' | b'a'..=b'z' | b'_' | b':') || b >= 0x80
}

fn is_name_char(b: u8) -> bool {
    is_name_start(b) || matches!(b, b'-' | b'.' | b'0'..=b'9')
}

fn utf8_width(byte: u8) -> Option<usize> {
    match byte {
        0xc2..=0xdf => Some(2),
        0xe0..=0xef => Some(3),
        0xf0..=0xf4 => Some(4),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(src: &str) -> Element {
        parse(src).unwrap_or_else(|e| panic!("expected element, got error: {e}"))
    }

    fn err(src: &str) -> Error {
        parse(src).expect_err("expected error")
    }

    #[test]
    fn empty_self_closing_root() {
        let r = root("<foo/>");
        assert_eq!(r.name, "foo");
        assert!(r.attrs.is_empty());
        assert!(r.children.is_empty());
    }

    #[test]
    fn attributes_single_and_double_quoted() {
        let r = root(r#"<foo a="1" b='two' />"#);
        assert_eq!(r.attr("a"), Some("1"));
        assert_eq!(r.attr("b"), Some("two"));
    }

    #[test]
    fn nested_elements_and_text() {
        let r = root("<a><b>hi</b><c/></a>");
        assert_eq!(r.name, "a");
        let bs: Vec<&Element> = r.child_elements().collect();
        assert_eq!(bs.len(), 2);
        assert_eq!(bs[0].name, "b");
        assert_eq!(bs[1].name, "c");
        // The text "hi" is the first child of <b>.
        match &bs[0].children[0] {
            Node::Text(t) => assert_eq!(t, "hi"),
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn comments_are_skipped_inside_and_outside_root() {
        let r = root("<!-- pre -->\n<a><!-- inline --><b/></a><!-- post -->");
        assert_eq!(r.child_elements().count(), 1);
    }

    #[test]
    fn xml_declaration_is_accepted_and_ignored() {
        let r = root(r#"<?xml version="1.0" encoding="UTF-8"?><root/>"#);
        assert_eq!(r.name, "root");
    }

    #[test]
    fn processing_instructions_skipped() {
        let r = root("<?xml-stylesheet href='x'?><a/>");
        assert_eq!(r.name, "a");
    }

    #[test]
    fn entities_decoded_in_attrs_and_text() {
        let r = root(r#"<a x="1 &lt; 2 &amp; 3">&lt;&gt;&quot;&apos;</a>"#);
        assert_eq!(r.attr("x"), Some("1 < 2 & 3"));
        match &r.children[0] {
            Node::Text(t) => assert_eq!(t, "<>\"'"),
            _ => panic!(),
        }
    }

    #[test]
    fn numeric_entities_decoded() {
        let r = root(r#"<a x="&#65;&#x2603;">&#10;</a>"#);
        assert_eq!(r.attr("x"), Some("A\u{2603}"));
        match &r.children[0] {
            Node::Text(t) => assert_eq!(t, "\n"),
            _ => panic!(),
        }
    }

    #[test]
    fn unclosed_tag_errors() {
        let e = err("<a><b></a>");
        assert!(
            e.message.contains("does not match") || e.message.contains("closing"),
            "message: {}",
            e.message
        );
    }

    #[test]
    fn bad_attribute_quote_errors() {
        let e = err("<a x=1/>");
        assert!(e.message.contains("quoted"), "{}", e.message);
    }

    #[test]
    fn unterminated_string_attribute_errors() {
        let e = err(r#"<a x="1 />"#);
        assert!(e.message.contains("unterminated"), "{}", e.message);
    }

    #[test]
    fn missing_root_errors() {
        let e = err("<!-- only a comment -->");
        assert!(e.message.contains("expected"), "{}", e.message);
    }

    #[test]
    fn unknown_entity_errors() {
        let e = err(r#"<a x="&bogus;"/>"#);
        assert!(e.message.contains("unknown entity"), "{}", e.message);
    }

    #[test]
    fn duplicate_attribute_errors() {
        let e = err(r#"<a x="1" x="2"/>"#);
        assert!(e.message.contains("duplicate"), "{}", e.message);
    }

    #[test]
    fn trailing_content_after_root_errors() {
        let e = err("<a/><b/>");
        assert!(e.message.contains("trailing"), "{}", e.message);
    }

    #[test]
    fn deep_nesting_rejected() {
        let src = "<a>".repeat(MAX_DEPTH + 2) + &"</a>".repeat(MAX_DEPTH + 2);
        assert!(parse(&src).is_err());
    }

    #[test]
    fn doctype_rejected() {
        let e = err(r#"<!DOCTYPE html><a/>"#);
        assert!(e.message.contains("DOCTYPE"), "{}", e.message);
    }

    #[test]
    fn self_closing_with_attrs() {
        let r = root(r#"<geom type="capsule" size="0.1" />"#);
        assert_eq!(r.attr("type"), Some("capsule"));
        assert_eq!(r.attr("size"), Some("0.1"));
        assert!(r.children.is_empty());
    }

    #[test]
    fn mixed_whitespace_and_children() {
        let r = root(
            r#"<worldbody>
                 <body name="a"/>
                 <body name="b">
                   <geom type="sphere"/>
                 </body>
               </worldbody>"#,
        );
        let bodies: Vec<&Element> = r.child_elements().collect();
        assert_eq!(bodies.len(), 2);
        assert_eq!(bodies[0].attr("name"), Some("a"));
        assert_eq!(bodies[1].attr("name"), Some("b"));
        assert_eq!(bodies[1].child_elements().count(), 1);
    }

    #[test]
    fn attribute_containing_lt_is_rejected() {
        let e = err(r#"<a x="1<2"/>"#);
        assert!(e.message.contains("'<'"), "{}", e.message);
    }

    #[test]
    fn utf8_in_attribute_and_text_preserved() {
        let r = root("<a x=\"日本語\">☃</a>");
        assert_eq!(r.attr("x"), Some("日本語"));
        match &r.children[0] {
            Node::Text(t) => assert_eq!(t, "☃"),
            _ => panic!(),
        }
    }

    #[test]
    fn cdata_marker_rejected() {
        // We do not support CDATA sections; the raw bytes must not be
        // smuggled through — parsing must fail.
        let e = err("<a><![CDATA[hi]]></a>");
        assert!(!e.message.is_empty());
    }

    #[test]
    fn double_dash_inside_comment_rejected() {
        let e = err("<!-- -- --><a/>");
        assert!(e.message.contains("'--'"), "{}", e.message);
    }

    #[test]
    fn attribute_missing_whitespace_rejected() {
        // <a b="1"c="2"/> — no whitespace before second attribute.
        let e = err(r#"<a b="1"c="2"/>"#);
        assert!(e.message.contains("whitespace"), "{}", e.message);
    }
}
