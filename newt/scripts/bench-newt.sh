#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
cargo bench --quiet --bench newt_perf -- "$@"
