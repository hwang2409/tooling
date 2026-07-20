use std::collections::BTreeMap;

use gauge_store::Series;

#[derive(Clone, Debug, PartialEq)]
pub struct MetricSample {
    pub series: Series,
    pub value: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParseResult {
    pub samples: Vec<MetricSample>,
    pub malformed_lines: u64,
}

pub fn parse_exposition(input: &str) -> ParseResult {
    let mut result = ParseResult::default();
    for line in input.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if line.starts_with('#') {
            if !valid_comment(line) {
                result.malformed_lines += 1;
            }
            continue;
        }
        match parse_sample(line) {
            Ok(sample) => result.samples.push(sample),
            Err(()) => result.malformed_lines += 1,
        }
    }
    result
}

fn valid_comment(line: &str) -> bool {
    let Some(rest) = line.strip_prefix('#') else {
        return false;
    };
    if rest.is_empty() {
        return true;
    }
    let fields: Vec<_> = rest.split_whitespace().collect();
    match fields.first().copied() {
        Some("HELP") => {
            return fields.len() >= 3 && valid_name(fields[1]);
        }
        Some("TYPE") => {
            return fields.len() == 3 && valid_name(fields[1]) && valid_type_token(fields[2]);
        }
        _ => {}
    }
    true
}

fn parse_sample(line: &str) -> Result<MetricSample, ()> {
    let bytes = line.as_bytes();
    let mut index = 0;
    while index < bytes.len() && !bytes[index].is_ascii_whitespace() && bytes[index] != b'{' {
        index += 1;
    }
    let name = &line[..index];
    if !valid_name(name) {
        return Err(());
    }
    let mut labels = BTreeMap::new();
    if index < bytes.len() && bytes[index] == b'{' {
        let end = find_closing_brace(line, index)?;
        parse_labels(&line[index + 1..end], &mut labels)?;
        index = end + 1;
    }
    if index == bytes.len() || !bytes[index].is_ascii_whitespace() {
        return Err(());
    }
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    let value_start = index;
    while index < bytes.len() && !bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    let value = parse_value(&line[value_start..index])?;
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    if index != bytes.len() {
        // The optional Prometheus timestamp is accepted syntactically, but the
        // scraper deliberately timestamps samples at scrape time.
        let timestamp = &line[index..];
        if timestamp.is_empty() || timestamp.split_whitespace().count() != 1 {
            return Err(());
        }
        timestamp.parse::<i64>().map_err(|_| ())?;
    }
    Ok(MetricSample {
        series: Series::new(name, labels),
        value,
    })
}

fn find_closing_brace(line: &str, start: usize) -> Result<usize, ()> {
    let mut escaped = false;
    let mut quote = false;
    for (offset, byte) in line.as_bytes()[start + 1..].iter().enumerate() {
        if escaped {
            escaped = false;
        } else if *byte == b'\\' && quote {
            escaped = true;
        } else if *byte == b'"' {
            quote = !quote;
        } else if *byte == b'}' && !quote {
            return Ok(start + 1 + offset);
        }
    }
    Err(())
}

fn parse_labels(input: &str, labels: &mut BTreeMap<String, String>) -> Result<(), ()> {
    if input.is_empty() {
        return Ok(());
    }
    let mut index = 0;
    while index < input.len() {
        while input
            .as_bytes()
            .get(index)
            .is_some_and(u8::is_ascii_whitespace)
        {
            index += 1;
        }
        let name_start = index;
        while input
            .as_bytes()
            .get(index)
            .is_some_and(|byte| byte.is_ascii_alphanumeric() || *byte == b'_' || *byte == b':')
        {
            index += 1;
        }
        let name = &input[name_start..index];
        if !valid_name(name) || labels.contains_key(name) {
            return Err(());
        }
        while input
            .as_bytes()
            .get(index)
            .is_some_and(u8::is_ascii_whitespace)
        {
            index += 1;
        }
        if input.as_bytes().get(index) != Some(&b'=') {
            return Err(());
        }
        index += 1;
        while input
            .as_bytes()
            .get(index)
            .is_some_and(u8::is_ascii_whitespace)
        {
            index += 1;
        }
        if input.as_bytes().get(index) != Some(&b'"') {
            return Err(());
        }
        index += 1;
        let mut value = String::new();
        let mut closed = false;
        while index < input.len() {
            let character = input[index..].chars().next().ok_or(())?;
            index += character.len_utf8();
            match character {
                '"' => {
                    closed = true;
                    break;
                }
                '\\' => {
                    let escaped = input[index..].chars().next().ok_or(())?;
                    index += escaped.len_utf8();
                    value.push(match escaped {
                        '\\' => '\\',
                        '"' => '"',
                        'n' => '\n',
                        _ => return Err(()),
                    });
                }
                '\n' | '\r' => return Err(()),
                character => value.push(character),
            }
        }
        if !closed || labels.insert(name.to_owned(), value).is_some() {
            return Err(());
        }
        while input
            .as_bytes()
            .get(index)
            .is_some_and(u8::is_ascii_whitespace)
        {
            index += 1;
        }
        if index == input.len() {
            return Ok(());
        }
        if input.as_bytes().get(index) != Some(&b',') {
            return Err(());
        }
        index += 1;
        if index == input.len() {
            return Err(());
        }
    }
    Ok(())
}

fn valid_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(first) if first.is_ascii_alphabetic() || first == '_' || first == ':')
        && chars.all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == ':'
        })
}

fn valid_type_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn parse_value(value: &str) -> Result<f64, ()> {
    match value {
        "+Inf" | "Inf" => Ok(f64::INFINITY),
        "-Inf" => Ok(f64::NEG_INFINITY),
        "NaN" => Ok(f64::NAN),
        _ => value.parse::<f64>().map_err(|_| ()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_corpus_without_aborting_on_bad_lines() {
        let parsed = parse_exposition(
            "# HELP http_requests_total requests\n# TYPE http_requests_total counter\nhttp_requests_total{method=\"GET\",path=\"/a\\\"b\\\\c\"} 3\nnot valid\nvalue 4\n",
        );
        assert_eq!(parsed.malformed_lines, 1);
        assert_eq!(parsed.samples.len(), 2);
        assert_eq!(parsed.samples[0].series.labels["path"], "/a\"b\\c");
    }

    #[test]
    fn accepts_histogram_components_and_non_finite_values() {
        let parsed = parse_exposition(
            "# TYPE request_duration histogram\nrequest_duration_bucket{le=\"+Inf\"} +Inf\nrequest_duration_sum NaN\nrequest_duration_count -Inf\n",
        );
        assert_eq!(parsed.malformed_lines, 0);
        assert!(parsed.samples[0].value.is_infinite());
        assert!(parsed.samples[1].value.is_nan());
    }

    #[test]
    fn rejects_bad_label_structure() {
        let parsed = parse_exposition("metric{a=\"x\",} 1\nmetric{a=1} 2\n");
        assert_eq!(parsed.malformed_lines, 2);
    }

    #[test]
    fn preserves_utf8_label_values() {
        let parsed = parse_exposition("city{value=\"Zürich\"} 1\n");
        assert_eq!(parsed.malformed_lines, 0);
        assert_eq!(parsed.samples[0].series.labels["value"], "Zürich");
    }

    #[test]
    fn rejects_malformed_type_directives() {
        let parsed = parse_exposition(
            "# TYPE\n# TYPE missing\n# TYPE extra gauge trailing\n#\tTYPE\tmissing\n#  TYPE  extra  gauge  trailing\n# TYPE okay unknown_type\nokay 1\n",
        );
        assert_eq!(parsed.malformed_lines, 5);
        assert_eq!(parsed.samples.len(), 1);
    }
}
