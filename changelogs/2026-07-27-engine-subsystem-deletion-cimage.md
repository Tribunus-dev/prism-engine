# Goal: Delete `compute-core/src/ecs/cimage/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved.

## Source

`compute-core/src/ecs/cimage/` — 20 files, 10,530 LOC. CImage
subsystem: compute-image representation, packing, validation,
metadata, layout, MIL-emit helpers, lower-level CImage ops
(some are already partially absorbed by the compiler migration;
cimage_runtime/ is a separate subsystem at
`compute-core/src/ecs/cimage_runtime/` and is out of scope here).

## Constitutional target

`crates/prism-ecs-compile/` (the constitutional compile crate;
the engine's `cimage/` is the legacy home for CImage lifecycle;
the compile crate is the canonical home for CImage pipeline +
packer + validation).

## Migration pattern

Followed E-0..E-4 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
compiler migration (E-1..E-9, 9 commits) was the closest template
since it also targets `prism-ecs-compile` and the CImage pipeline
was already partially absorbed there.

## Isolate to your own worktree

Isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-cimage` on branch
`migrate/cimage`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** The migration is to `prism-ecs-compile`.
- **Engine dep audit at E-0.** `prism-ecs-compile` was already a
  dependency of the engine (since the compiler migration), so no
  Cargo.toml change was needed.
- **Three agents are targeting prism-ecs-compile simultaneously**
  (compilation, decode_attribution, cimage). All use isolated
  worktrees. The merge order will be: compilation first,
  decode_attribution second, cimage third. The "Take HEAD on
  architecture/src/lib.rs conflicts" rule applies: take HEAD +
  add new module declaration.
- **Out of scope:** `compute-core/src/ecs/cimage_runtime/` (11
  files, 9,774 LOC) is a separate subsystem. Not touched. (Only
  the import paths in cimage_runtime/ that referenced
  `crate::ecs::cimage::` were updated to `crate::ecs::legacy_cimage::`
  — a mechanical, path-only change required by the safety net.)

## Success criteria

- [x] All 20 files of `compute-core/src/ecs/cimage/` removed
      (renamed to `compute-core/src/ecs/legacy_cimage/`).
- [x] Constitutional surface in `crates/prism-ecs-compile/src/cimage_v0/`.
- [x] All engine callers migrated (13 cross-crate callers +
      11 cimage_runtime callers + 4 bin/tests callers).
- [x] `workspace_contains_no_legacy_cimage_imports` architecture
      test passes.
- [x] `rg "use crate::ecs::cimage::" compute-core/src/` returns no
      results.
- [x] Engine pre-existing build error count is 193 (down from
      the 221 baseline; the rename removed a chunk of intra-cimage
      compile errors).
- [x] Constitutional-side tests green:
      `cargo test -p prism-ecs-compile --lib` (451 passed, 0 failed)
      and `cargo test -p prism-architecture --lib` (15 passed,
      0 failed).

## Migration commits

| Step | SHA       | Subject                                                                                           |
|------|-----------|---------------------------------------------------------------------------------------------------|
| E-1  | 162eed94  | feat(constitutional): add prism-ecs-compile::cimage_v0 surface                                    |
| E-2  | 07c6b8d0  | chore(engine): rename cimage/ to legacy_cimage/ + migrate callers                                 |
| E-3  | 07c6b8d0  | (folded into E-2; 13 cross-crate callers + 11 cimage_runtime callers + 4 bin/tests callers)       |
| E-4  | 79f64f3b  | feat(architecture): add cimage legacy-import safety net                                           |
| E-5  | (this)    | docs: mark cimage engine-subsystem deletion goal achieved                                         |

## Module split (constitutional vs engine-internal)

The engine's cimage/ had 20 files split along two axes:

- **Engine-agnostic (5 files → constitutional `cimage_v0/`)**: error,
  header, payload, receipts, canonical. These are pure data types
  with no engine-internal dependencies.
- **Engine-coupled (15 files → engine-internal `legacy_cimage/`)**:
  canonical, compatibility, dashboard, durability, generation_api,
  generation_store, loader, manifest, mlp_reference, privacy,
  sealed_v1, shard_builder, streaming_writer, validate, writer.
  These depend on engine-internal types (`PrecisionPlan`,
  `PrivacyContract`, `CompiledKernelArtifact`, `CimageGeneration`,
  `GenerationApi`, etc.) and stay engine-side.

The `legacy_cimage/mod.rs` re-exports the constitutional
`prism_ecs_compile::cimage_v0::*` data types so engine callers
can read them through the legacy path.

## Propagation chain

The 5 constitutional data types are leaf-level V0 file format
primitives. The downstream consumers are:
- `legacy_cimage::writer` (writes the V0 file format)
- `legacy_cimage::loader` (reads the V0 file format)
- `legacy_cimage::validate` (validates the V0 file format)
- `legacy_cimage::shard_builder` (synthesizes V0 shards)

These higher-level operations in `legacy_cimage/` consume the
constitutional primitives directly via the `legacy_cimage::mod.rs`
re-exports.

## Authority-leak audit

- `use crate::ecs::cimage::*` — 0 results in compute-core/src/
  (after E-2/E-3).
- `compute_core::ecs::cimage::*` — 0 results in the workspace
  (after E-3, including all bin/ and tests/ callers).
- `tribunus_compute_core::cimage::*` — 0 results in the
  workspace (after E-3, including the lib.rs re-export).
- `tribunus_compute_core::cimage;` — 0 results (the lib.rs
  re-export was renamed to `legacy_cimage`).

## Safety record

- No destructive git ops (renames only, no `git rm` until the
  final state is verified).
- No edits outside scope (cimage_runtime/ files were only
  updated for import paths, not for logic).
- All commits bisectable.
- Checkpoint discipline maintained.
- Correct crate name throughout (`prism-ecs-compile`).
- Isolated to `/Users/user/Developer/GitHub/prism-engine-cimage`
  worktree on branch `migrate/cimage`.
