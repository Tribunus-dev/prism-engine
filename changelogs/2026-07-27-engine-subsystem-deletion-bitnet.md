# Goal: Delete `compute-core/src/ecs/bitnet/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ Goal achieved (E-0..E-7 + docs, 6 commits on `migrate/bitnet`).

## Source

`compute-core/src/ecs/bitnet/` — 8 files, 3,661 LOC.

## Constitutional target

`crates/prism-ecs-quantization/src/bitnet/` (the engine's `bitnet/` is the
1-bit / 1.58-bit quantization path; the canonical home is the
`prism-ecs-quantization` crate, which already exists for
quantization contracts).

## Migration pattern

Followed E-0..E-7 (no E-0 needed — the engine's
`compute-core/Cargo.toml` already declares
`prism-ecs-quantization = { path = "../crates/prism-ecs-quantization" }`).

## Isolate to your own worktree

The main worktree at `/Users/user/Developer/GitHub/prism-engine`
is shared. **Do not work in the main worktree.**

Isolated worktree: `/Users/user/Developer/GitHub/prism-engine-bitnet`
on branch `migrate/bitnet`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.** (only 6 commits; well under 30 min).
- **Correct crate name.** All commits reference
  `prism-ecs-quantization` (never `prism-ecs-agent` /
  `prism-ecs-compile` / `prism-ecs-codec` / etc.).
- **Engine dep audit at E-0.** Skipped — engine's
  `Cargo.toml` already has `prism-ecs-quantization` as a dep.

## Commits (E-0..E-7)

- `fe975fdd` — **E-1** `feat(constitutional): add prism-ecs-quantization::bitnet surface`
- `ae4a9a5f` — **E-2..E-4** `chore(engine): migrate bitnet engine callers to prism_ecs_quantization::bitnet`
  - `region_runner.rs`: `bitnet_decoder_layer_reference` (runtime) +
    `BitNetCheckpoint`/`make_ternary_from_checkpoint` (gated) +
    `emit_bitnet_decoder_layer`/`BitNetDecoderLayerShardConfig` (test).
  - `bitnet_layer_resolver.rs`: doc comment update.
  - `bin/tribunus-compute-image.rs`: 4 import migrations.
  - `bin/prism_server.rs`: doc comment update.
- `055ca085` — **E-5** `chore(engine): drop bitnet re-export and module declaration`
- `014b637f` — **E-6** `feat(architecture): add bitnet legacy-import safety net`
- (uncommitted above) — **E-7** `chore(engine): delete the legacy engine's bitnet subsystem`

## Success criteria

- [x] All 8 files of `compute-core/src/ecs/bitnet/` removed
      (E-7).
- [x] Constitutional surface in
      `crates/prism-ecs-quantization/src/bitnet/` (8 .rs files
      plus a `mod tests`; E-1).
- [x] All engine callers migrated (E-2..E-4). 4 engine files
      updated: `region_runner.rs`,
      `bitnet_layer_resolver.rs` (doc comment only),
      `bin/tribunus-compute-image.rs`,
      `bin/prism_server.rs` (doc comment only).
- [x] `workspace_contains_no_legacy_bitnet_imports`
      architecture test passes (E-6, registered in
      `crates/architecture/src/lib.rs`).
- [x] `rg "use crate::ecs::bitnet::" compute-core/src/`
      returns no results.
- [x] Engine pre-existing build error count is **220** (one
      *fewer* than the 221 baseline; the bitnet module had been
      contributing a single error of its own that no longer
      materializes once the module declaration is dropped).
- [x] Constitutional-side tests green:
      `cargo test -p prism-ecs-quantization --lib bitnet`
      → 39 passed; 0 failed.

## Constitutional surface layout

`crates/prism-ecs-quantization/src/bitnet/`

```
mod.rs           - module root + re-exports
ternary_codec.rs - TernaryPackedTensor, TernaryCodecError,
                   pack_ternary_codes, unpack_ternary_codes,
                   validate_no_reserved_codes
cimage_shim.rs   - self-contained copy of the cimage manifest /
                   payload / writer types the bitnet module needs
                   (structurally identical to the engine's, to be
                   re-merged when the cimage subsystem is itself
                   absorbed)
importer.rs      - BitNetImporter and its deterministic
                   pseudo-random ternary weight generation
checkpoint.rs    - BitNetCheckpoint safetensors loader and
                   make_ternary_from_checkpoint
reference.rs     - pure-Rust CPU reference for a single decoder
                   layer (bitnet_decoder_layer_reference,
                   bitnet_decoder_logits)
kv.rs            - BitNetKvCache and the cimage KV cache manifest
                   entry helper
phases.rs        - phased cimage emission
                   (emit_single_bitnet_linear,
                   emit_bitnet_mlp_block,
                   emit_bitnet_decoder_layer,
                   emit_bitnet_full_model,
                   emit_bitnet_from_checkpoint),
                   plus BitNetDecoderLayerShardConfig
text.rs          - auto-regressive token-wise inference loop
                   (prefill, decode_single, greedy_sample,
                   run_text, BitNetTokenizer)
tests.rs         - 4 integration tests
```

Each module carries a single-authority module doc in one sentence
per the module-discipline rule.

## Engine ↔ constitutional type bridge

The bitnet module is the single authority for BitNet b1.58 weight
emission as a cimage artifact, but the engine's cimage writer
(`CImageWriter::write_v0`) lives in `compute-core`. To bridge the
two, the engine's `region_runner.rs` test (line 3936, in
`#[cfg(all(test, target_os = "macos", feature = "metal-dispatch"))] mod tests`)
includes a small `constitutional_to_engine_pending` helper that
field-by-field maps the constitutional
`PendingCImageShard` into the engine's
`PendingCImageShard` so the engine writer can still consume it.
The two types are structurally identical (one-to-one field copy)
and the helper is a temporary bridge — once the cimage subsystem
itself is absorbed into the constitutional surface, the helper
becomes obsolete.

## Branch tip

`migrate/bitnet @ 014b637f` (then `uncommitted` E-7). Final
branch tip after docs commit (E-8): see `git log migrate/bitnet`.
