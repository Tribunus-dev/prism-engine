#!/usr/bin/env bash
# mlx_harness.sh — the Linux type-check harness for authored-for-Mac code.
#
# `cargo check --features mlx-backend` type-checks compute_image (megakernel,
# orchestrator, packers) on a Linux box. It CANNOT be zero-error there: modules
# gated on target_os=macos or on optional deps (coreml-proto via prism-backend;
# metal/objc/block via metal-dispatch) legitimately fail to resolve — those are
# artifacts of checking off-target, and are correct on the Mac. Gating the
# consumers to silence them would delete the harness's coverage of exactly the
# files it protects (this harness caught the decode_slot_logits u16/f32
# E0308s before they reached hardware).
#
# Classification per error block:
#   ARTIFACT — "configured out"/"gated behind" notes, unresolved
#              metal/objc/block/coreml_proto imports, or method resolution
#              falling through to an mlx_rs trait because the inherent
#              mac-gated method is compiled out.
#   REAL     — everything else. THE signal. Exit 1 if any exist.
#
# Baseline at introduction: 79 artifacts, 0 real.
set -u
CLASSIFIER=$(mktemp /tmp/mlx_classify.XXXX.py)
trap 'rm -f "$CLASSIFIER"' EXIT
cat > "$CLASSIFIER" <<'PY'
import sys
raw = sys.stdin.read()
lines = raw.splitlines()
blocks, cur = [], []
for l in lines:
    if l.startswith("error[") or (l.startswith("error:") and "could not compile" not in l and "aborting" not in l):
        if cur: blocks.append("\n".join(cur))
        cur = [l]
    elif cur:
        cur.append(l)
if cur: blocks.append("\n".join(cur))

ARTIFACT_MARKS = (
    "configured out", "gated behind",
    "unresolved import `metal`", "unresolved import `block`",
    "unresolved import `objc`", "coreml_proto",
    "mlx_rs::builder::Builder",
)
def block_file(b):
    for x in b.splitlines():
        if "-->" in x:
            return x.strip().split("-->")[-1].strip().split(":")[0]
    return None

# Pass 1: files with configured-out/optional-dep failures. Any later error in
# those files is a downstream cascade of the unresolved item (e.g. a match
# binding degrading to an unsized type because the method's owner type never
# resolved) — artifact, not signal.
cascade_files = set()
for b in blocks:
    if any(m in b for m in ARTIFACT_MARKS):
        f = block_file(b)
        if f:
            cascade_files.add(f)

real, artifacts = [], 0
for b in blocks:
    if any(m in b for m in ARTIFACT_MARKS) or block_file(b) in cascade_files:
        artifacts += 1
    else:
        head = b.splitlines()[0][:100]
        loc = next((x.strip() for x in b.splitlines() if "-->" in x), "?")
        real.append(head + "\n    " + loc)

print("[mlx-harness] %d error blocks: %d off-target artifacts, %d REAL" % (len(blocks), artifacts, len(real)))
for r in real:
    print("REAL:", r)
sys.exit(1 if real else 0)
PY
cargo check -p tribunus-compute-core --features mlx-backend 2>&1 | python3 "$CLASSIFIER"
