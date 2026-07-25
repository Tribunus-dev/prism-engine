#!/usr/bin/env bash
# scripts/build-site.sh — Prism Observatory v1 build entry point.
#
# Per ADR-032 D2 (Strategy A: Pages builds, pinned toolchain).
# This script is the single source of build truth; the Cloudflare
# Pages dashboard "build command" field points here. The dashboard
# is not the source; this script is.
#
# Pipeline:
#   1. Install the pinned Rust toolchain (rust-toolchain.toml) and
#      WASM build tooling (wasm32-unknown-unknown, wasm-bindgen-cli,
#      wasm-opt) at the versions recorded in
#      crates/prism-docs-runtime/Cargo.toml and
#      docs/scripts/build-wasm.sh.
#   2. Build the WASM hydration bundle at docs/pkg/.
#   3. Run the SSG with --validate, which reads and validates the
#      data layer against the JSON Schemas before rendering.
#   4. Emit _redirects, _headers, and build.json at the site root
#      (per ADR-032 D3, D5, D6, and the §12 A16 build identity gate).
#
# Usage:
#   scripts/build-site.sh [--release|--debug] [--validate-only]
#
# The default build kind is --release. Pass --validate-only to
# run the validators and exit before emitting site output.

set -euo pipefail

# ---------- Configuration ----------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_KIND="release"
VALIDATE_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --release) BUILD_KIND="release" ;;
    --debug)   BUILD_KIND="debug" ;;
    --validate-only) VALIDATE_ONLY=1 ;;
    --help|-h)
      echo "Usage: $0 [--release|--debug] [--validate-only]"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Pinned toolchain version. Override with RUSTUP_TOOLCHAIN if set.
RUSTUP_TOOLCHAIN="${RUSTUP_TOOLCHAIN:-1.81.0}"

# Paths.
RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
DATA_ROOT="$REPO_ROOT/docs/data"
SCHEMA_DIR="$REPO_ROOT/schemas"
OUT_DIR="$REPO_ROOT/docs"
WASM_OUT_DIR="$OUT_DIR/pkg"

# ---------- 1. Install pinned toolchain (Strategy A) ----------

echo "build-site.sh: installing Rust toolchain (channel $RUSTUP_TOOLCHAIN)"
if [ ! -x "$RUSTUP_HOME/bin/rustup" ]; then
  echo "build-site.sh: rustup not found at $RUSTUP_HOME/bin/rustup" >&2
  echo "build-site.sh: install rustup or set RUSTUP_HOME" >&2
  exit 1
fi
"$RUSTUP_HOME/bin/rustup" toolchain install "$RUSTUP_TOOLCHAIN" --profile minimal --component rustfmt --component clippy
"$RUSTUP_HOME/bin/rustup" target add wasm32-unknown-unknown --toolchain "$RUSTUP_TOOLCHAIN"

# Pin the build's toolchain for this script's child commands.
export PATH="$RUSTUP_HOME/toolchains/$RUSTUP_TOOLCHAIN-x86_64-apple-darwin/bin:$RUSTUP_HOME/toolchains/$RUSTUP_TOOLCHAIN-aarch64-apple-darwin/bin:$PATH"
export RUSTUP_TOOLCHAIN
export RUSTC_VERSION="$("$RUSTUP_HOME/toolchains/$RUSTUP_TOOLCHAIN-$(uname -m | sed 's/x86_64/x86_64-apple-darwin/;s/aarch64/aarch64-apple-darwin/')/bin/rustc" --version 2>/dev/null || echo "rustc $RUSTUP_TOOLCHAIN")"

# wasm-bindgen-cli at the version pinned in the runtime crate.
WASM_BINDGEN_VERSION="$(
  grep -E '^wasm-bindgen[[:space:]]*=' "$REPO_ROOT/crates/prism-docs-runtime/Cargo.toml" \
    | head -1 \
    | sed -E 's/.*"([0-9.]+).*/\1/'
)"
if [ -z "$WASM_BINDGEN_VERSION" ]; then
  echo "build-site.sh: could not find wasm-bindgen version in Cargo.toml" >&2
  exit 1
fi
echo "build-site.sh: installing wasm-bindgen-cli $WASM_BINDGEN_VERSION"
cargo install --locked wasm-bindgen-cli --version "$WASM_BINDGEN_VERSION" --quiet

# ---------- 2. Build WASM hydration bundle ----------

echo "build-site.sh: building WASM hydration bundle"
bash "$REPO_ROOT/docs/scripts/build-wasm.sh"

# ---------- 3. Run the SSG with validation ----------

SSG_ARGS=(
  --content "$REPO_ROOT/docs/content"
  --styles  "$REPO_ROOT/docs/styles"
  --out     "$OUT_DIR"
  --data    "$DATA_ROOT"
  --schemas "$SCHEMA_DIR"
  --"$BUILD_KIND"
  --validate
)

if [ "$VALIDATE_ONLY" = "1" ]; then
  SSG_ARGS+=(--validate-only)
fi

echo "build-site.sh: running SSG"
cargo run -p prism-docs-ssg -- "${SSG_ARGS[@]}"

# ---------- 4. Generate _redirects, _headers, build.json ----------

# The SSG already writes these files when run normally. The
# --validate-only path skips the render and the file writes;
# the validators have already exercised the table sources.
if [ "$VALIDATE_ONLY" = "0" ]; then
  if [ ! -f "$OUT_DIR/_redirects" ]; then
    echo "build-site.sh: SSG did not emit _redirects" >&2
    exit 1
  fi
  if [ ! -f "$OUT_DIR/_headers" ]; then
    echo "build-site.sh: SSG did not emit _headers" >&2
    exit 1
  fi
  if [ ! -f "$OUT_DIR/build.json" ]; then
    echo "build-site.sh: SSG did not emit build.json" >&2
    exit 1
  fi
  echo "build-site.sh: deployable artifacts present:"
  echo "  $OUT_DIR/_redirects"
  echo "  $OUT_DIR/_headers"
  echo "  $OUT_DIR/build.json"
fi

echo "build-site.sh: done"
