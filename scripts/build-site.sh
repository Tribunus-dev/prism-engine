#!/usr/bin/env bash
# scripts/build-site.sh — Prism Observatory v1 build entry point.
#
# Per ADR-032 v2 (GitHub Pages publication layer, GitHub Actions
# build pipeline). This script is the single source of build truth;
# the GitHub Actions workflow runs it on push to `main` and on
# every pull request. The workflow file is not the source of
# build configuration; this script is.
#
# Pipeline:
#   1. Validate the data layer against the JSON Schemas. A
#      validation failure aborts the build with a non-zero exit.
#   2. Build the prism-transitions WebAssembly module. The
#      module holds the state-driven motion choreography
#      per §9 (Visual Direction). wasm-bindgen emits the
#      JS glue alongside the .wasm binary; both land in
#      docs/transitions/. If the wasm32 target is not
#      installed, the build continues with CSS-only
#      transitions (the no-JS fallback).
#   3. Run the SSG with the validated data layer, the manuscript,
#      and the styles. The SSG writes the rendered site to docs/,
#      copies the WASM artifacts to docs/transitions/, and emits
#      `build.json` (the build identity, per OBSERVATORY_V1_
#      SPEC.md §12 A16), `selection-controller.js` (the URL-
#      addressable selection reducer per §5.3 and §8), and
#      `theme.js` (the dark/light ThemeProvider per §9).
#
# The output directory is the `docs/` directory of the
# repository, served by GitHub Pages. The build is a normal
# static site: HTML, CSS, JavaScript, JSON. No `_redirects` or
# `_headers` files are emitted; the v1 site has no legacy URL
# surface and no custom response headers (per ADR-032 v2).
#
# Usage:
#   scripts/build-site.sh [--release|--debug] [--out <dir>]
#
# The default build kind is --release. The default output
# directory is `docs/`.

set -euo pipefail

# ---------- Configuration ----------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BUILD_KIND="release"
OUT_DIR="$REPO_ROOT/docs"
DATA_ROOT="$REPO_ROOT/docs/data"
SCHEMA_DIR="$REPO_ROOT/schemas"
MANUSCRIPT="$REPO_ROOT/OBSERVATORY_V1_MANUSCRIPT.md"
STYLES_DIR="$REPO_ROOT/docs/styles"

for arg in "$@"; do
  case "$arg" in
    --release) BUILD_KIND="release" ;;
    --debug)   BUILD_KIND="debug" ;;
    --out)     OUT_DIR="$2"; shift ;;
    --help|-h)
      echo "Usage: $0 [--release|--debug] [--out <dir>]"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ---------- 1. Validate the data layer ----------

echo "build-site.sh: validating data layer at $DATA_ROOT against $SCHEMA_DIR"
cargo run -p prism-docs-ssg --quiet -- \
  --data "$DATA_ROOT" \
  --schemas "$SCHEMA_DIR" \
  --manuscript "$MANUSCRIPT" \
  --styles "$STYLES_DIR" \
  --out "$OUT_DIR" \
  --validate-only

# ---------- 2. Build the prism-transitions WASM ----------
#
# The choreography module compiles to wasm32-unknown-unknown.
# wasm-bindgen emits an ES module (prism_transitions.js) and
# the .wasm binary (prism_transitions_bg.wasm). The
# orchestrator JS (transitions-orchestrator.js) loads the
# module on idle and dispatches state-driven motion.

WASM_OUT="$OUT_DIR/transitions"
if rustup target list --installed 2>/dev/null | grep -q wasm32-unknown-unknown; then
  echo "build-site.sh: building prism-transitions WASM"
  RUSTC="$(rustup which rustc 2>/dev/null || which rustc)"
  RUSTC="$RUSTC" cargo build \
    -p prism-transitions \
    --target wasm32-unknown-unknown \
    --release \
    --features wasm 2>&1 | tail -3
  WASM_BLOB="$(ls -t target/wasm32-unknown-unknown/release/prism_transitions.wasm | head -1)"
  if [ -n "$WASM_BLOB" ] && [ -f "$WASM_BLOB" ]; then
    mkdir -p "$WASM_OUT"
    "$(rustup which wasm-bindgen 2>/dev/null || which wasm-bindgen)" \
      "$WASM_BLOB" \
      --out-dir "$WASM_OUT" \
      --target web \
      --no-typescript >/dev/null
    echo "build-site.sh: WASM artifacts in $WASM_OUT"
  else
    echo "build-site.sh: WASM build produced no blob; CSS-only fallback" >&2
  fi
else
  echo "build-site.sh: wasm32 target not installed; CSS-only transitions"
fi

# ---------- 3. Run the SSG ----------

echo "build-site.sh: rendering site to $OUT_DIR"
cargo run -p prism-docs-ssg --quiet -- \
  --data "$DATA_ROOT" \
  --schemas "$SCHEMA_DIR" \
  --manuscript "$MANUSCRIPT" \
  --styles "$STYLES_DIR" \
  --out "$OUT_DIR"

# ---------- 4. Verify deployable artifacts ----------

for f in build.json selection-controller.js site.css index.html; do
  if [ ! -f "$OUT_DIR/$f" ]; then
    echo "build-site.sh: SSG did not emit $f" >&2
    exit 1
  fi
done

# The theme.js file is also expected.
if [ ! -f "$OUT_DIR/theme.js" ]; then
  echo "build-site.sh: SSG did not emit theme.js" >&2
  exit 1
fi

echo "build-site.sh: deployable artifacts present:"
echo "  $OUT_DIR/build.json"
echo "  $OUT_DIR/selection-controller.js"
echo "  $OUT_DIR/theme.js"
echo "  $OUT_DIR/site.css"
echo "  $OUT_DIR/index.html (+ 12 canonical pages + 404.html)"
if [ -d "$WASM_OUT" ]; then
  echo "  $WASM_OUT/ (WebAssembly transitions module)"
fi

# ---------- 5. Run the A-list axiom audit ----------

# The audit runner takes the built site and produces a
# 22-row pass/fail/skip table for OBSERVATORY_V1_SPEC.md
# §12. It is a CI gate: blocking failures exit non-zero.
# Warnings and skips are reported but do not stop the
# build; they go to the H-list review queue.

echo "build-site.sh: running A-list axiom audit"
AUDIT_REPORT="$OUT_DIR/audit-report.md"
cargo run -p prism-docs-audit --quiet -- \
  --site "$OUT_DIR" \
  --out "$AUDIT_REPORT" 2>&1 | tail -25
AUDIT_EXIT=$?
if [ $AUDIT_EXIT -ne 0 ]; then
  echo "build-site.sh: A-list audit failed (exit $AUDIT_EXIT)" >&2
  echo "build-site.sh: see $AUDIT_REPORT" >&2
  exit $AUDIT_EXIT
fi
echo "build-site.sh: A-list audit report at $AUDIT_REPORT"

echo "build-site.sh: done"
