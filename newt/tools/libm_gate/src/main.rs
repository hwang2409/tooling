use std::env;
use std::fs;
use std::io;
use std::path::Path;

const BANNED_METHODS: &[&str] = &[
    "sin", "cos", "tan", "asin", "acos", "atan2", "atan", "exp", "ln", "log", "powf", "powi",
];

#[derive(Debug, Eq, PartialEq)]
struct Match {
    line: usize,
    column: usize,
    method: &'static str,
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| "src".to_owned());
    let root = Path::new(&root);

    match scan_tree(root) {
        Ok(matches) => {
            for file in &matches {
                for found in &file.matches {
                    println!(
                        "{}:{}:{} — .{}",
                        file.path.display(),
                        found.line,
                        found.column,
                        found.method
                    );
                }
            }
            if matches.iter().any(|found| !found.matches.is_empty()) {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("libm-gate: {error}");
            std::process::exit(2);
        }
    }
}

struct FileMatches {
    path: std::path::PathBuf,
    matches: Vec<Match>,
}

fn scan_tree(root: &Path) -> io::Result<Vec<FileMatches>> {
    let mut paths = Vec::new();
    collect_rust_files(root, &mut paths)?;
    paths.sort();

    paths
        .into_iter()
        .map(|path| {
            let source = fs::read_to_string(&path)?;
            Ok(FileMatches {
                matches: scan_source(&source),
                path,
            })
        })
        .collect()
}

fn collect_rust_files(root: &Path, paths: &mut Vec<std::path::PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, paths)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            paths.push(path);
        }
    }
    Ok(())
}

fn scan_source(source: &str) -> Vec<Match> {
    let masked = mask_comments_and_literals(source);
    let bytes = masked.as_bytes();
    let mut matches = Vec::new();
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] != b'.' {
            index += 1;
            continue;
        }

        let mut method_start = index + 1;
        while method_start < bytes.len() && bytes[method_start].is_ascii_whitespace() {
            method_start += 1;
        }

        let Some(&method) = BANNED_METHODS
            .iter()
            .find(|method| bytes[method_start..].starts_with(method.as_bytes()))
        else {
            index += 1;
            continue;
        };
        let after_method = method_start + method.len();
        if after_method < bytes.len() && is_identifier_byte(bytes[after_method]) {
            index += 1;
            continue;
        }

        let mut opening_paren = after_method;
        while opening_paren < bytes.len() && bytes[opening_paren].is_ascii_whitespace() {
            opening_paren += 1;
        }
        if opening_paren >= bytes.len() || bytes[opening_paren] != b'(' {
            index += 1;
            continue;
        }

        let line = source[..index]
            .bytes()
            .filter(|&byte| byte == b'\n')
            .count()
            + 1;
        let line_start = source[..index].rfind('\n').map_or(0, |newline| newline + 1);
        let column = source[line_start..index].chars().count() + 1;
        matches.push(Match {
            line,
            column,
            method,
        });
        index = opening_paren + 1;
    }

    matches
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn mask_comments_and_literals(source: &str) -> String {
    let mut masked = source.as_bytes().to_vec();
    let bytes = source.as_bytes();
    let mut index = 0;

    while index < bytes.len() {
        let end = if bytes[index..].starts_with(b"//") {
            line_comment_end(bytes, index)
        } else if bytes[index..].starts_with(b"/*") {
            block_comment_end(bytes, index)
        } else if bytes[index] == b'b'
            && bytes.get(index + 1) == Some(&b'r')
            && is_raw_string_prefix(bytes, index + 2)
        {
            raw_string_end(bytes, index + 2)
        } else if bytes[index] == b'r' && is_raw_string_prefix(bytes, index + 1) {
            raw_string_end(bytes, index + 1)
        } else if bytes[index] == b'b' && matches!(bytes.get(index + 1), Some(b'"' | b'\'')) {
            let Some(end) = quoted_literal_end(bytes, index + 1) else {
                index += 1;
                continue;
            };
            Some(end)
        } else if bytes[index] == b'"' {
            quoted_literal_end(bytes, index)
        } else if bytes[index] == b'\'' {
            let Some(end) = quoted_literal_end(bytes, index) else {
                index += 1;
                continue;
            };
            Some(end)
        } else {
            index += 1;
            continue;
        };

        let end = end.unwrap_or(bytes.len());
        mask_range(&mut masked, index, end);
        index = end;
    }

    String::from_utf8(masked).expect("source was valid UTF-8")
}

fn line_comment_end(bytes: &[u8], start: usize) -> Option<usize> {
    bytes[start..]
        .iter()
        .position(|&byte| byte == b'\n')
        .map(|offset| start + offset)
        .or(Some(bytes.len()))
}

fn block_comment_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 1;
    let mut index = start + 2;
    while index + 1 < bytes.len() {
        if bytes[index..].starts_with(b"/*") {
            depth += 1;
            index += 2;
        } else if bytes[index..].starts_with(b"*/") {
            depth -= 1;
            index += 2;
            if depth == 0 {
                return Some(index);
            }
        } else {
            index += 1;
        }
    }
    None
}

fn quoted_literal_end(bytes: &[u8], quote: usize) -> Option<usize> {
    let delimiter = bytes[quote];
    let mut index = quote + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index += 2,
            byte if byte == delimiter => return Some(index + 1),
            b'\n' | b'\r' => return None,
            _ => index += 1,
        }
    }
    None
}

fn raw_string_end(bytes: &[u8], prefix_end: usize) -> Option<usize> {
    let mut index = prefix_end;
    while bytes.get(index) == Some(&b'#') {
        index += 1;
    }
    if bytes.get(index) != Some(&b'"') {
        return None;
    }

    let hash_count = index - prefix_end;
    index += 1;
    while index < bytes.len() {
        if bytes[index] == b'"' && has_hashes(bytes, index + 1, hash_count) {
            return Some(index + hash_count + 1);
        }
        index += 1;
    }
    None
}

fn is_raw_string_prefix(bytes: &[u8], prefix_end: usize) -> bool {
    let mut index = prefix_end;
    while bytes.get(index) == Some(&b'#') {
        index += 1;
    }
    bytes.get(index) == Some(&b'"')
}

fn has_hashes(bytes: &[u8], start: usize, count: usize) -> bool {
    bytes
        .get(start..start.saturating_add(count))
        .is_some_and(|suffix| suffix.iter().all(|&byte| byte == b'#'))
}

fn mask_range(masked: &mut [u8], start: usize, end: usize) {
    for byte in &mut masked[start..end] {
        if *byte != b'\n' {
            *byte = b' ';
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Match, scan_source};

    fn methods(source: &str) -> Vec<Match> {
        scan_source(source)
    }

    #[test]
    fn line_comment_hides_call() {
        assert!(methods("// .sin() here").is_empty());
    }

    #[test]
    fn block_comment_hides_call() {
        assert!(methods("/* .sin() */").is_empty());
    }

    #[test]
    fn nested_block_comment_hides_call() {
        assert!(methods("/* /* .sin() */ .cos() */").is_empty());
    }

    #[test]
    fn string_literal_hides_call() {
        assert!(methods("let s = \".sin()\";").is_empty());
    }

    #[test]
    fn escaped_quote_in_string_stays_in_string() {
        assert!(methods("let s = \"\\\".sin()\";").is_empty());
    }

    #[test]
    fn raw_string_hides_call() {
        assert!(methods(r#"let s = r"/* .sin() */";"#).is_empty());
    }

    #[test]
    fn raw_string_with_hashes_hides_call() {
        assert!(methods("let s = r##\"\".sin()\"##;").is_empty());
    }

    #[test]
    fn real_call_detected() {
        assert_eq!(
            methods("foo.sin()"),
            vec![Match {
                line: 1,
                column: 4,
                method: "sin"
            }]
        );
    }

    #[test]
    fn whitespace_between_receiver_and_call_detected() {
        assert_eq!(
            methods("foo\n    .sin()"),
            vec![Match {
                line: 2,
                column: 5,
                method: "sin"
            }]
        );
    }

    #[test]
    fn sqrt_allowed() {
        assert!(methods("x.sqrt()").is_empty());
    }

    #[test]
    fn powi_detected_but_powi_arg_not_call() {
        assert_eq!(
            methods("x.powi(2)"),
            vec![Match {
                line: 1,
                column: 2,
                method: "powi"
            }]
        );
    }

    #[test]
    fn byte_string_hides_call() {
        assert!(methods("let b = b\".sin()\";").is_empty());
    }
}
