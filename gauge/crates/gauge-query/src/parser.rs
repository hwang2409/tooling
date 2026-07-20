use std::fmt;

use regex::Regex;

pub const MAX_PARSE_DEPTH: usize = 64;
pub const MAX_PARSE_NODES: usize = 11_000;

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Number(f64),
    Selector(Selector),
    Rate {
        selector: Selector,
        window_ms: i64,
    },
    Aggregate {
        op: AggregateOp,
        expr: Box<Expr>,
        by: Vec<String>,
    },
    Binary {
        left: Box<Expr>,
        op: BinaryOp,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Selector {
    pub name: String,
    pub matchers: Vec<LabelMatcher>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LabelMatcher {
    pub label: String,
    pub op: MatchOp,
    pub value: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchOp {
    Eq,
    NotEq,
    Regex,
    NotRegex,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AggregateOp {
    Sum,
    Avg,
    Min,
    Max,
    Count,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnaryOp {
    Plus,
    Minus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    pub position: usize,
    pub message: String,
}

impl ParseError {
    fn new(position: usize, message: impl Into<String>) -> Self {
        Self {
            position,
            message: message.into(),
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at byte {}: {}", self.position, self.message)
    }
}

impl std::error::Error for ParseError {}

pub fn parse(input: &str) -> Result<Expr, ParseError> {
    let mut parser = Parser {
        input,
        position: 0,
        nodes: 0,
    };
    parser.skip_whitespace();
    if parser.is_eof() {
        return Err(ParseError::new(0, "expected an expression"));
    }
    let expr = parser.parse_expression(0)?.expr;
    parser.skip_whitespace();
    if !parser.is_eof() {
        return Err(parser.error("unexpected trailing input"));
    }
    Ok(expr)
}

pub fn parse_selector(input: &str) -> Result<Selector, ParseError> {
    let mut parser = Parser {
        input,
        position: 0,
        nodes: 0,
    };
    parser.skip_whitespace();
    let selector = parser.parse_selector()?;
    parser.skip_whitespace();
    if !parser.is_eof() {
        return Err(parser.error("expected a selector to end here"));
    }
    Ok(selector)
}

struct Parser<'a> {
    input: &'a str,
    position: usize,
    nodes: usize,
}

struct ParsedExpr {
    expr: Expr,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn parse_expression(&mut self, depth: usize) -> Result<ParsedExpr, ParseError> {
        self.check_depth(depth)?;
        self.parse_add_sub(depth)
    }

    fn parse_add_sub(&mut self, depth: usize) -> Result<ParsedExpr, ParseError> {
        let mut expr = self.parse_mul_div(depth)?;
        loop {
            self.skip_whitespace();
            let op = match self.peek_char() {
                Some('+') => BinaryOp::Add,
                Some('-') => BinaryOp::Sub,
                _ => break,
            };
            self.advance_char();
            let right = self.parse_mul_div(depth)?;
            self.consume_node()?;
            let result_depth = expr.depth.max(right.depth).saturating_add(1);
            self.check_result_depth(result_depth)?;
            expr = ParsedExpr {
                expr: Expr::Binary {
                    left: Box::new(expr.expr),
                    op,
                    right: Box::new(right.expr),
                },
                depth: result_depth,
            };
        }
        Ok(expr)
    }

    fn parse_mul_div(&mut self, depth: usize) -> Result<ParsedExpr, ParseError> {
        let mut expr = self.parse_unary(depth)?;
        loop {
            self.skip_whitespace();
            let op = match self.peek_char() {
                Some('*') => BinaryOp::Mul,
                Some('/') => BinaryOp::Div,
                _ => break,
            };
            self.advance_char();
            let right = self.parse_unary(depth)?;
            self.consume_node()?;
            let result_depth = expr.depth.max(right.depth).saturating_add(1);
            self.check_result_depth(result_depth)?;
            expr = ParsedExpr {
                expr: Expr::Binary {
                    left: Box::new(expr.expr),
                    op,
                    right: Box::new(right.expr),
                },
                depth: result_depth,
            };
        }
        Ok(expr)
    }

    fn parse_unary(&mut self, depth: usize) -> Result<ParsedExpr, ParseError> {
        self.check_depth(depth)?;
        self.skip_whitespace();
        let op = match self.peek_char() {
            Some('+') => Some(UnaryOp::Plus),
            Some('-') => Some(UnaryOp::Minus),
            _ => None,
        };
        if let Some(op) = op {
            self.advance_char();
            self.consume_node()?;
            let expr = self.parse_unary(depth.saturating_add(1))?;
            let result_depth = expr.depth.saturating_add(1);
            self.check_result_depth(result_depth)?;
            return Ok(ParsedExpr {
                expr: Expr::Unary {
                    op,
                    expr: Box::new(expr.expr),
                },
                depth: result_depth,
            });
        }
        self.parse_primary(depth)
    }

    fn parse_primary(&mut self, depth: usize) -> Result<ParsedExpr, ParseError> {
        self.skip_whitespace();
        match self.peek_char() {
            Some('(') => {
                self.advance_char();
                let expr = self.parse_expression(depth.saturating_add(1))?;
                self.expect_char(')', "expected ')' to close the expression")?;
                Ok(expr)
            }
            Some(ch) if ch.is_ascii_digit() || ch == '.' => {
                let number = self.parse_number()?;
                self.consume_node()?;
                Ok(ParsedExpr {
                    expr: Expr::Number(number),
                    depth: 1,
                })
            }
            Some(ch) if is_identifier_start(ch) => {
                let name = self.parse_identifier("expected an identifier")?;
                self.skip_whitespace();
                if self.peek_char() == Some('(') {
                    match name.as_str() {
                        "rate" => self.parse_rate(),
                        "sum" => self.parse_aggregate(AggregateOp::Sum, depth),
                        "avg" => self.parse_aggregate(AggregateOp::Avg, depth),
                        "min" => self.parse_aggregate(AggregateOp::Min, depth),
                        "max" => self.parse_aggregate(AggregateOp::Max, depth),
                        "count" => self.parse_aggregate(AggregateOp::Count, depth),
                        _ => {
                            self.consume_node()?;
                            Ok(ParsedExpr {
                                expr: Expr::Selector(self.parse_selector_after_name(name)?),
                                depth: 1,
                            })
                        }
                    }
                } else {
                    self.consume_node()?;
                    Ok(ParsedExpr {
                        expr: Expr::Selector(self.parse_selector_after_name(name)?),
                        depth: 1,
                    })
                }
            }
            Some(_) => Err(self.error("expected a number, selector, or '('")),
            None => Err(self.error("expected an expression")),
        }
    }

    fn parse_rate(&mut self) -> Result<ParsedExpr, ParseError> {
        self.expect_char('(', "expected '(' after rate")?;
        self.skip_whitespace();
        let selector = self.parse_selector()?;
        self.skip_whitespace();
        self.expect_char('[', "expected '[' and a rate window")?;
        let window_ms = self.parse_duration()?;
        self.expect_char(']', "expected ']' after rate window")?;
        self.expect_char(')', "expected ')' after rate selector")?;
        self.consume_node()?;
        Ok(ParsedExpr {
            expr: Expr::Rate {
                selector,
                window_ms,
            },
            depth: 1,
        })
    }

    fn parse_aggregate(&mut self, op: AggregateOp, depth: usize) -> Result<ParsedExpr, ParseError> {
        self.expect_char('(', "expected '(' after aggregation")?;
        let expr = self.parse_expression(depth.saturating_add(1))?;
        self.expect_char(')', "expected ')' after aggregation expression")?;
        self.skip_whitespace();
        let by = self.parse_identifier("expected 'by' after aggregation")?;
        if by != "by" {
            return Err(self.error("expected 'by' after aggregation"));
        }
        self.skip_whitespace();
        self.expect_char('(', "expected '(' after 'by'")?;
        self.skip_whitespace();
        if self.peek_char() == Some(')') {
            return Err(self.error("aggregation requires at least one label"));
        }
        let mut labels = Vec::new();
        loop {
            self.skip_whitespace();
            let label = self.parse_identifier("expected a grouping label")?;
            if labels.contains(&label) {
                return Err(self.error("duplicate grouping label"));
            }
            labels.push(label);
            self.skip_whitespace();
            if self.peek_char() == Some(')') {
                self.advance_char();
                break;
            }
            self.expect_char(',', "expected ',' between grouping labels")?;
        }
        self.consume_node()?;
        let result_depth = expr.depth.saturating_add(1);
        self.check_result_depth(result_depth)?;
        Ok(ParsedExpr {
            expr: Expr::Aggregate {
                op,
                expr: Box::new(expr.expr),
                by: labels,
            },
            depth: result_depth,
        })
    }

    fn parse_selector(&mut self) -> Result<Selector, ParseError> {
        let name = self.parse_identifier("expected a metric name")?;
        self.parse_selector_after_name(name)
    }

    fn parse_selector_after_name(&mut self, name: String) -> Result<Selector, ParseError> {
        self.skip_whitespace();
        let mut matchers = Vec::new();
        if self.peek_char() != Some('{') {
            return Ok(Selector { name, matchers });
        }
        self.advance_char();
        self.skip_whitespace();
        if self.peek_char() == Some('}') {
            self.advance_char();
            return Ok(Selector { name, matchers });
        }
        loop {
            let label = self.parse_identifier("expected a label name")?;
            self.skip_whitespace();
            let op = self.parse_match_operator()?;
            self.skip_whitespace();
            let value_position = self.position;
            let value = self.parse_string()?;
            if matches!(op, MatchOp::Regex | MatchOp::NotRegex) {
                Regex::new(&format!("^(?:{value})$")).map_err(|error| {
                    ParseError::new(value_position, format!("invalid regex matcher: {error}"))
                })?;
            }
            self.consume_node()?;
            matchers.push(LabelMatcher { label, op, value });
            self.skip_whitespace();
            if self.peek_char() == Some('}') {
                self.advance_char();
                break;
            }
            self.expect_char(',', "expected ',' or '}' after matcher")?;
            self.skip_whitespace();
            if self.peek_char() == Some('}') {
                return Err(self.error("trailing comma in matcher list"));
            }
        }
        Ok(Selector { name, matchers })
    }

    fn parse_match_operator(&mut self) -> Result<MatchOp, ParseError> {
        for (text, op) in [
            ("!=", MatchOp::NotEq),
            ("=~", MatchOp::Regex),
            ("!~", MatchOp::NotRegex),
            ("=", MatchOp::Eq),
        ] {
            if self.input[self.position..].starts_with(text) {
                self.position += text.len();
                return Ok(op);
            }
        }
        Err(self.error("expected matcher operator '=', '!=', '=~', or '!~'"))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        let start = self.position;
        self.expect_char('"', "expected a quoted matcher value")?;
        let mut value = String::new();
        loop {
            let Some(ch) = self.peek_char() else {
                return Err(ParseError::new(start, "unterminated quoted string"));
            };
            self.advance_char();
            match ch {
                '"' => return Ok(value),
                '\\' => {
                    let Some(escaped) = self.peek_char() else {
                        return Err(ParseError::new(self.position, "unterminated escape"));
                    };
                    self.advance_char();
                    match escaped {
                        '"' => value.push('"'),
                        '\\' => value.push('\\'),
                        'n' => value.push('\n'),
                        'r' => value.push('\r'),
                        't' => value.push('\t'),
                        'u' => value.push_str(&self.parse_unicode_escape()?),
                        other => {
                            value.push('\\');
                            value.push(other);
                        }
                    }
                }
                other => value.push(other),
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<String, ParseError> {
        let start = self.position;
        let mut digits = String::new();
        for _ in 0..4 {
            let Some(ch) = self.peek_char() else {
                return Err(ParseError::new(start, "incomplete unicode escape"));
            };
            if !ch.is_ascii_hexdigit() {
                return Err(self.error("unicode escape must contain four hexadecimal digits"));
            }
            digits.push(ch);
            self.advance_char();
        }
        let codepoint = u32::from_str_radix(&digits, 16)
            .map_err(|_| ParseError::new(start, "invalid unicode escape"))?;
        let Some(ch) = char::from_u32(codepoint) else {
            return Err(ParseError::new(start, "invalid unicode codepoint"));
        };
        Ok(ch.to_string())
    }

    fn parse_number(&mut self) -> Result<f64, ParseError> {
        let start = self.position;
        let mut digits = 0;
        while self.peek_char().is_some_and(|ch| ch.is_ascii_digit()) {
            self.advance_char();
            digits += 1;
        }
        if self.peek_char() == Some('.') {
            self.advance_char();
            while self.peek_char().is_some_and(|ch| ch.is_ascii_digit()) {
                self.advance_char();
                digits += 1;
            }
        }
        if digits == 0 {
            return Err(ParseError::new(start, "expected digits in number"));
        }
        if self.peek_char().is_some_and(|ch| ch == 'e' || ch == 'E') {
            self.advance_char();
            if self.peek_char().is_some_and(|ch| ch == '+' || ch == '-') {
                self.advance_char();
            }
            let exponent_start = self.position;
            while self.peek_char().is_some_and(|ch| ch.is_ascii_digit()) {
                self.advance_char();
            }
            if exponent_start == self.position {
                return Err(self.error("expected digits in exponent"));
            }
        }
        self.input[start..self.position]
            .parse::<f64>()
            .map_err(|_| ParseError::new(start, "invalid number"))
    }

    fn parse_duration(&mut self) -> Result<i64, ParseError> {
        let start = self.position;
        let mut number = 0_i64;
        let mut digits = 0;
        while let Some(ch) = self.peek_char() {
            if !ch.is_ascii_digit() {
                break;
            }
            number = number
                .checked_mul(10)
                .and_then(|value| value.checked_add(i64::from(ch as u8 - b'0')))
                .ok_or_else(|| ParseError::new(start, "duration is too large"))?;
            self.advance_char();
            digits += 1;
        }
        if digits == 0 {
            return Err(ParseError::new(start, "expected a duration such as 5m"));
        }
        let unit_start = self.position;
        let unit = if self.input[unit_start..].starts_with("ms") {
            self.position += 2;
            1_i64
        } else {
            let Some(ch) = self.peek_char() else {
                return Err(ParseError::new(unit_start, "duration is missing a unit"));
            };
            self.advance_char();
            match ch {
                's' => 1_000,
                'm' => 60_000,
                'h' => 60 * 60_000,
                'd' => 24 * 60 * 60_000,
                _ => {
                    return Err(ParseError::new(
                        unit_start,
                        "duration unit must be ms, s, m, h, or d",
                    ));
                }
            }
        };
        let duration = number
            .checked_mul(unit)
            .ok_or_else(|| ParseError::new(start, "duration is too large"))?;
        if duration <= 0 {
            return Err(ParseError::new(start, "duration must be positive"));
        }
        Ok(duration)
    }

    fn parse_identifier(&mut self, message: &str) -> Result<String, ParseError> {
        self.skip_whitespace();
        let start = self.position;
        let Some(first) = self.peek_char() else {
            return Err(self.error(message));
        };
        if !is_identifier_start(first) {
            return Err(self.error(message));
        }
        self.advance_char();
        while self.peek_char().is_some_and(is_identifier_continue) {
            self.advance_char();
        }
        Ok(self.input[start..self.position].to_owned())
    }

    fn expect_char(&mut self, expected: char, message: &str) -> Result<(), ParseError> {
        self.skip_whitespace();
        if self.peek_char() == Some(expected) {
            self.advance_char();
            Ok(())
        } else {
            Err(self.error(message))
        }
    }

    fn skip_whitespace(&mut self) {
        while self.peek_char().is_some_and(char::is_whitespace) {
            self.advance_char();
        }
    }

    fn error(&self, message: impl Into<String>) -> ParseError {
        ParseError::new(self.position, message)
    }

    fn check_depth(&self, depth: usize) -> Result<(), ParseError> {
        if depth > MAX_PARSE_DEPTH {
            return Err(self.error(format!(
                "maximum expression nesting depth is {MAX_PARSE_DEPTH}"
            )));
        }
        Ok(())
    }

    fn check_result_depth(&self, depth: usize) -> Result<(), ParseError> {
        self.check_depth(depth)
    }

    fn consume_node(&mut self) -> Result<(), ParseError> {
        if self.nodes >= MAX_PARSE_NODES {
            return Err(self.error(format!(
                "expression exceeds maximum node budget of {MAX_PARSE_NODES}"
            )));
        }
        self.nodes += 1;
        Ok(())
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.position..].chars().next()
    }

    fn advance_char(&mut self) {
        if let Some(ch) = self.peek_char() {
            self.position += ch.len_utf8();
        }
    }

    fn is_eof(&self) -> bool {
        self.position >= self.input.len()
    }
}

fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_' || ch == ':'
}

fn is_identifier_continue(ch: char) -> bool {
    is_identifier_start(ch) || ch.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedence_and_whitespace_are_preserved_in_ast() {
        let expr = parse(" cpu * 100 + 2 / (1 - 1) ").unwrap();
        assert!(matches!(
            expr,
            Expr::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
        let Expr::Binary { left, .. } = expr else {
            unreachable!();
        };
        assert!(matches!(
            *left,
            Expr::Binary {
                op: BinaryOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn parses_all_matcher_operators_and_escapes() {
        let selector =
            parse_selector(r#"cpu{job="api",instance!="old",zone=~"us-\\d+",rack!~"bad"}"#)
                .unwrap();
        assert_eq!(selector.matchers.len(), 4);
        assert_eq!(selector.matchers[0].op, MatchOp::Eq);
        assert_eq!(selector.matchers[1].op, MatchOp::NotEq);
        assert_eq!(selector.matchers[2].op, MatchOp::Regex);
        assert_eq!(selector.matchers[2].value, r"us-\d+");
        assert_eq!(selector.matchers[3].op, MatchOp::NotRegex);
    }

    #[test]
    fn parses_rate_and_aggregation() {
        assert_eq!(
            parse("rate(http_requests_total{job=\"api\"}[5m])").unwrap(),
            Expr::Rate {
                selector: Selector {
                    name: "http_requests_total".to_owned(),
                    matchers: vec![LabelMatcher {
                        label: "job".to_owned(),
                        op: MatchOp::Eq,
                        value: "api".to_owned(),
                    }],
                },
                window_ms: 300_000,
            }
        );
        assert!(matches!(
            parse("sum (cpu) by (job, instance)").unwrap(),
            Expr::Aggregate {
                op: AggregateOp::Sum,
                ..
            }
        ));
    }

    #[test]
    fn malformed_input_reports_byte_position() {
        let error = parse("cpu{job=\"api\"").unwrap_err();
        assert_eq!(error.position, 13);
        let error = parse("rate(cpu[5])").unwrap_err();
        assert_eq!(error.position, 10);
        let error = parse("cpu *").unwrap_err();
        assert_eq!(error.position, 5);
    }

    #[test]
    fn deeply_nested_parentheses_are_rejected_without_stack_growth() {
        let depth = MAX_PARSE_DEPTH + 10;
        let input = format!("{}1{}", "(".repeat(depth), ")".repeat(depth));
        let error = parse(&input).unwrap_err();
        assert!(error.message.contains("maximum expression nesting depth"));
        assert!(error.position < input.len());
    }

    #[test]
    fn deeply_nested_unary_chains_are_rejected_without_stack_growth() {
        let input = format!("{}1", "-".repeat(MAX_PARSE_DEPTH + 10));
        let error = parse(&input).unwrap_err();
        assert!(error.message.contains("maximum expression nesting depth"));
        assert!(error.position < input.len());
    }

    #[test]
    fn invalid_regex_is_a_positioned_parse_error() {
        let input = r#"cpu{job=~"["}"#;
        let error = parse(input).unwrap_err();
        assert!(error.message.contains("invalid regex matcher"));
        assert!(error.position >= input.find('"').unwrap());
    }
}
