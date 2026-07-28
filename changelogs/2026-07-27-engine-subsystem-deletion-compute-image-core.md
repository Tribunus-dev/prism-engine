# Goal: Delete `compute-core/src/ecs/compute_image/` (core surface)

**Date:** 2026-07-27 (Pacific)
**Status:** Goal declared; agent dispatched.

## Source

`compute-core/src/ecs/compute_image/` core surface — 62 files, ~25K LOC.
The top-level `.rs` files (52 files, ~19K LOC) + `cimage_packer/` (5
files, 3.8K) + `manifest/` (5 files, 2.7K). The top-level files cover
the CImage adapter, alpha types, ANE compile/prefill, Apple CImage
manifest, Apple shared arena, CImage loader, compaction, compatibility,
content_store I/O, diag, executable dispatch, execution shape, fallback
plan, fragments, fusion ABI/plan/receipts/sealing/tensix, gemma4
support, HF model loading, hardware assessment/bench, kernel provider,
KV interleave/plan, layout tensix, metal codegen/pipeline/epilogue,
model test helpers, and core mod/builder/scheduling.

## Constitutional target

`crates/prism-ecs-compile/` (the constitutional compile crate; the
engine's `compute_image/` is the legacy home for CImage packing,
manifest, top-level compile facade, and supporting adapters).

## Scope boundary (THIS AGENT)

You are migrating:
- All top-level `.rs` files in `compute-core/src/ecs/compute_image/*.rs`
- `compute-core/src/ecs/compute_image/cimage_packer/`
- `compute-core/src/ecs/compute_image/manifest/`

You are NOT migrating:
- `compute-core/src/ecs/compute_image/compile/` (separate agent: `ci_compile`)
- `compute-core/src/ecs/compute_image/orchestrator/` (separate agent: `ci_compile`)
- `compute-core/src/ecs/compute_image/residency/`, `heterogeneous/`, `megakernel/`, `kernel_selection/` (separate agent: `ci_runtime`)
- `compute-core/src/ecs/compute_image/multimodal/`, `model_family/`, `variants/`, `program/`, `content_store/`, `executable/`, `scheduler/`, `verification/` (separate agent: `ci_runtime`)

If you find callers that import from these OUT-OF-SCOPE subdirs, do
NOT migrate those subdirs — leave the caller as-is for the other
agents to handle. The architecture safety net will catch any leftover
imports after the other agents run.

## Migration pattern

Follow E-0..E-N+2 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
compilation migration at `99bb0554` (E-0..E-5, 6 commits) is the
closest template since it also targets `prism-ecs-compile`.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-ci-core` on branch
`migrate/ci-core`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-ecs-compile` —
  write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-compile` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.
- **Three agents are targeting prism-ecs-compile simultaneously**
  (ci-core, ci-compile, ci-runtime). All use isolated worktrees. The
  merge order will be: ci-core first, ci-compile second, ci-runtime
  third. The "Take HEAD on architecture/src/lib.rs conflicts" rule
  applies: take HEAD + add new module declaration.
- **Watch for engine-coupled files.** Files importing legacy types
  from `compute-core/src/ecs/backend::*` or other engine modules may
  need to be renamed to `legacy_compute_image_core/` rather than
  deleted (see core/ → legacy_core/ and memory/ → memory_impl/
  pattern).

## Success criteria

- All 62 files of `compute-core/src/ecs/compute_image/` (core surface)
  removed or renamed to `legacy_compute_image_core/`.
- Constitutional surface in
  `crates/prism-ecs-compile/src/compute_image_core/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_compute_image_core_imports`
  architecture test passes.
- `rg "use crate::ecs::compute_image::" compute-core/src/ | grep -v "/compile\|/orchestrator\|/residency\|/heterogeneous\|/megakernel\|/kernel_selection\|/multimodal\|/model_family\|/variants\|/program\|/content_store\|/executable\|/scheduler\|/verification/"` returns no results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 185).
- Constitutional-side tests green.
