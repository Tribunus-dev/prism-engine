#!/usr/bin/env bash
set -euo pipefail

test "$(uname -m)" = "x86_64" || {
  echo "MI300X validation requires an AMD64 host" >&2
  exit 2
}
command -v hipcc >/dev/null || { echo "hipcc is unavailable" >&2; exit 2; }
command -v rocminfo >/dev/null || { echo "rocminfo is unavailable" >&2; exit 2; }
rocminfo_output="$(rocminfo 2>&1)"
grep -q 'gfx942' <<<"$rocminfo_output" || {
  echo "MI300X gfx942 was not reported by rocminfo" >&2
  exit 2
}

export PRISM_MI300X_GPU=1
cargo test --manifest-path crates/prism-rocm-runtime/Cargo.toml --lib --quiet
cargo test --manifest-path crates/prism-amd-npu-runtime/Cargo.toml --quiet
cargo test -p prism-ecs-compile --lib --quiet

echo "Prism MI300X validation passed for gfx942"
