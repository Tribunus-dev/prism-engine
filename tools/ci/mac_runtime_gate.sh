#!/usr/bin/env bash
# mac_runtime_gate.sh — the PRODUCTION runTIME gate (PRODUCTION_CONTRACT.md).
#
# Proves, on a clean macOS checkout with only Xcode CLT + Rust + this repo:
#   1. HERMETICITY — the production surface (`--features prism-backend`)
#      builds with the MLX research stack entirely absent from the dependency
#      graph: no mlx-rs, no mlx-sys, no cmake, no out-of-repo MLX checkout.
#   2. UNTAPPED DECODE SMOKE — a TapMode::Untapped orchestrator decodes and
#      refuses the taps API (needs TRIBUNUS_TEST_CIMAGE).
#   3. TAPPED PARITY SMOKE — explicit TappedAudit construction taps without
#      any TRIBUNUS_TAPS env, plus the Transport A/B parity gates
#      (needs TRIBUNUS_TEST_CIMAGE).
#
# Stages 2-3 are skipped (with a loud notice) when TRIBUNUS_TEST_CIMAGE is
# unset — CI machines without a model artifact still get the hermeticity
# proof, which is the part a Linux box can never provide.
#
# Usage:
#   tools/ci/mac_runtime_gate.sh                         # hermeticity only
#   TRIBUNUS_TEST_CIMAGE=/path/model.cimage tools/ci/mac_runtime_gate.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "[mac-gate] FAIL: this gate must run on macOS (got $(uname -s))" >&2
    exit 1
fi

echo "[mac-gate] 1/3 hermeticity: production surface must not know MLX"

# The dependency-graph assert runs FIRST: it fails fast and precisely if the
# feature graph regresses (e.g. someone re-adds mlx-backend to prism-backend),
# rather than letting a missing mlx-tribunus checkout produce an opaque cmake
# error later.
if cargo tree -p tribunus-compute-core --features prism-backend -e normal,build 2>/dev/null \
        | grep -qE "mlx-rs|mlx-sys"; then
    echo "[mac-gate] FAIL: mlx-rs/mlx-sys present in the prism-backend graph —" >&2
    echo "           the production surface has re-coupled to the research stack" >&2
    exit 1
fi
echo "[mac-gate]     dependency graph clean (no mlx-rs / mlx-sys)"

# Full production build. On a clean checkout this exercises exactly the
# contract: Xcode CLT (metal compiler) + Rust + repo, nothing else.
cargo build -p tribunus-compute-core --features prism-backend --lib
echo "[mac-gate]     cargo build --features prism-backend: OK"

# The production bins must build hermetically too.
cargo build -p tribunus-compute-core --features prism-backend \
    --bin prism-server --bin prism-bench-ab --bin gemma4-ingest
echo "[mac-gate]     production bins: OK"

if [[ -z "${TRIBUNUS_TEST_CIMAGE:-}" ]]; then
    echo "[mac-gate] 2/3 + 3/3 SKIPPED: TRIBUNUS_TEST_CIMAGE not set."
    echo "[mac-gate]     Hermeticity is proven; decode/parity smokes need a model."
    echo "[mac-gate] PASS (hermeticity only)"
    exit 0
fi

echo "[mac-gate] 2/3 untapped decode smoke + explicit-mode contract"
# tap_mode_explicit_construction_beats_env covers BOTH directions: explicit
# TappedAudit without env taps; explicit Untapped with env set refuses.
env -u TRIBUNUS_TAPS cargo test -p tribunus-compute-core \
    --features prism-backend --lib \
    tap_mode_explicit_construction_beats_env -- --test-threads=1 --nocapture

echo "[mac-gate] 3/3 tapped parity gates (Transport A oracle + Transport B ladder)"
TRIBUNUS_TAPS=1 cargo test -p tribunus-compute-core \
    --features prism-backend --lib \
    stage0_tap -- --test-threads=1
TRIBUNUS_TAPS=1 cargo test -p tribunus-compute-core \
    --features prism-backend --lib \
    transport_b -- --test-threads=1

echo "[mac-gate] PASS"
