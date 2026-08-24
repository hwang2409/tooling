#!/bin/sh
set -eu

cd "$(dirname "$0")/.."

cargo run --quiet --release --bin libm-gate -- src
