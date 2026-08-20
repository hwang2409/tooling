#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

# Banned methods: .sin .cos .tan .asin .acos .atan .atan2 .exp .ln .log
# .powf and .powi. f32::sqrt is allowed because the design spec, line 65,
# permits + - * / sqrt as IEEE-exact operations.
find src -type f -name '*.rs' -print0 | xargs -0 awk '
BEGIN { in_block = 0 }
{
    code = $0
    if (in_block) {
        end = index(code, "*/")
        if (end == 0) next
        code = substr(code, end + 2)
        in_block = 0
    }
    while ((start = index(code, "/*")) != 0) {
        before = substr(code, 1, start - 1)
        rest = substr(code, start + 2)
        end = index(rest, "*/")
        if (end == 0) {
            code = before
            in_block = 1
            break
        }
        code = before substr(rest, end + 2)
    }
    sub("//.*$", "", code)
    if (code ~ /\.(sin|cos|tan|asin|acos|atan|atan2|exp|ln|log|powf|powi)[[:space:]]*\(/) {
        print FILENAME ":" FNR ":" code
        found = 1
    }
}
END { exit found }
'
