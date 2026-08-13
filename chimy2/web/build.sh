#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
crate_dir=$(cd -- "$script_dir/.." && pwd)

if ! command -v rustup >/dev/null 2>&1; then
  echo "rustup is required to install and check the wasm target" >&2
  exit 1
fi

if ! rustup target list --installed | grep -qx 'wasm32-unknown-unknown'; then
  echo "installing wasm32-unknown-unknown target"
  rustup target add wasm32-unknown-unknown
fi

cargo build --release --manifest-path "$crate_dir/Cargo.toml" --target wasm32-unknown-unknown --lib
cp "$crate_dir/target/wasm32-unknown-unknown/release/chimy2.wasm" "$script_dir/public/chimy2.wasm"
echo "wrote $script_dir/public/chimy2.wasm"
