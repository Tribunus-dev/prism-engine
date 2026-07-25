#!/usr/bin/env bash
# build-wasm.sh — build the WASM hydration bundle.
#
# The WASM is a cdylib output of `prism-docs-runtime` compiled
# for `wasm32-unknown-unknown` with the `hydrate` feature.
# `wasm-bindgen` then emits a JS shim that the SSG's
# `<script>` tag loads.
#
# Output: `docs/dist/pkg/{prism_docs_runtime.js, prism_docs_runtime_bg.wasm}`.
#
# Requirements:
#   - `rustup target add wasm32-unknown-unknown`
#   - `cargo install wasm-bindgen-cli` (matching the
#     wasm-bindgen version pinned in the runtime)
#
# Usage:
#   docs/scripts/build-wasm.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Use the rustup-managed toolchain. Homebrew's rustc is missing
# the wasm32 stdlib.
if command -v rustup >/dev/null 2>&1; then
  TOOLCHAIN_BIN="$(rustup which cargo 2>/dev/null | xargs dirname 2>/dev/null || true)"
  if [ -n "$TOOLCHAIN_BIN" ] && [ -d "$TOOLCHAIN_BIN" ]; then
    export PATH="$TOOLCHAIN_BIN:$PATH"
  fi
fi

echo "[build-wasm] compiling prism-docs-runtime for wasm32-unknown-unknown"
cargo build \
  -p prism-docs-runtime \
  --target wasm32-unknown-unknown \
  --release \
  --features hydrate

WASM_PATH="target/wasm32-unknown-unknown/release/prism_docs_runtime.wasm"
OUT_DIR="${ROOT}/docs/pkg"

if [ ! -f "$WASM_PATH" ]; then
  echo "[build-wasm] expected $WASM_PATH but it was not produced" >&2
  exit 1
fi

echo "[build-wasm] emitting JS shim to $OUT_DIR"
mkdir -p "$OUT_DIR"
wasm-bindgen "$WASM_PATH" --out-dir "$OUT_DIR" --target web

echo "[build-wasm] done"
ls -la "$OUT_DIR"
