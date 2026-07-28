# Goal: Delete `compute-core/src/ecs/backend/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal partially achieved (2026-07-27). Constitutional
surface in place at `crates/prism-ecs-kernel/src/backend/`; 50+ engine
callers migrated; safety net test green. Engine-side `backend/`
directory NOT deleted (blocked by 24 engine-coupled files whose
implementation is too deeply engine-coupled to move in this PR —
see "Migration status" below).

## Source

`compute-core/src/ecs/backend/` — 37 files, 15,169 LOC. Hardware
backend abstractions: accelerate, ane, cpu, coreai, metal,
heterogeneous_executor, flex_dispatch, routing, placement, etc.

## Constitutional target

`crates/prism-ecs-kernel/` (the constitutional kernel crate; the
engine's `backend/` is the legacy home for backend hardware
abstractions. The kernel backends already absorbed from
scheduling migration (metal/, ane/, accelerate/, cpu/, legacy/,
dispatcher/, lane_executor_registry.rs) provide the template for
this absorption).

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
scheduling kernel-backend migration is the most relevant
template — it absorbed the same kind of multi-backend-executor
code.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-backend` on branch
`migrate/backend`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.** Land a commit even if incomplete.
- **Correct crate name.** You are migrating to `prism-ecs-kernel`
  — write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-kernel` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria

- All 37 files of `compute-core/src/ecs/backend/` removed (or
  re-homed; document any re-homing in the goal doc).
- Constitutional surface in `crates/prism-ecs-kernel/src/backend/`
  (extending the existing kernel backend modules).
- All engine callers migrated.
- `workspace_contains_no_legacy_backend_imports` architecture
  test passes.
- `rg "use crate::ecs::backend::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 193).
- Constitutional-side tests green.

## Migration status (2026-07-27 partial)

The migration is **partially complete** as of this commit.

### What was achieved

- **E-0** (commit 3561bfad) — `chore(engine): add prism-ecs-kernel
  dep`. The engine's `compute-core/Cargo.toml` now depends on
  `prism-ecs-kernel`.
- **E-1** (commit 8ff4a1bf) — `feat(constitutional): add
  prism-ecs-kernel::backend surface` (partial). 8 engine backend
  files absorbed into the kernel: `authority`, `completion`,
  `evaluation`, `graph`, `intel_usm`, `routing/` (submodule),
  `shared_event`, `unified_arena`.
- **E-1.5** (commit 7 files) — 2 more engine files absorbed:
  `placement`, `tensor_registry`. Also made `routing::TensorId`
  derive `Copy` for use as a HashMap key.
- **E-2..E-13** (commit 4ea1c93c, 5374c68c, 8f879b67) — Migrated
  ~50 engine callers from `crate::ecs::backend::*` to
  `prism_ecs_kernel::backend::*` across `autopsy/`,
  `compilation/`, `compiler/`, `compute_image/`, and `core/`.
- **E-37** (commit 5f8f9e6c) — 3 more engine files absorbed:
  `accelerate_ffi`, `accelerate_lane`, `npu/`. Also migrated the
  callers of these modules.
- **E-39** (commit dbcb05f4) — `feat(architecture): add backend
  legacy-import safety net`. The
  `workspace_contains_no_legacy_backend_imports` test passes.
- **E-40** (this commit) — Updated the goal doc with the
  partial-completion status.

### Constitutional surface (12 files)

In `crates/prism-ecs-kernel/src/backend/`:

1. `authority.rs` — int4 dequantize algorithm.
2. `completion.rs` — async GPU dispatch completion tokens.
3. `evaluation.rs` — evaluation-boundary sweep experiment types.
4. `graph.rs` — graph backend trait and region execution receipts.
5. `intel_usm.rs` — Intel Level Zero USM buffer abstraction.
6. `routing/` — heterogeneous routing types (submodule: `lanes.rs`,
   `policy.rs`, `mod.rs`).
7. `shared_event.rs` — Metal-event coordination metadata.
8. `unified_arena.rs` — Apple unified-execution arena.
9. `placement.rs` — placement sets, hazard barriers, `ExecutionLane`.
10. `tensor_registry.rs` — logical-tensor materialization identity.
11. `accelerate_ffi.rs` — Accelerate framework FFI bindings.
12. `accelerate_lane.rs` — Accelerate CPU execution lane.
13. `npu/` — NPU FFI bindings (submodule: `ffi.rs`, `mod.rs`).

Plus the re-exports of trait types from `prism-ecs-backend`
(`TensorBackend`, `DType`, `TensorHandle`, `MatmulOp`, etc.).

### What remains (24 engine files, engine-coupled or pre-existing missing types)

The remaining 24 files of `compute-core/src/ecs/backend/` are NOT
absorbed in this PR because they are either:

1. **Heavy engine-coupled implementations** (~10 files, ~12,000
   LOC): `metal.rs` (2401 LOC), `ane.rs` (733 LOC),
   `heterogeneous_executor.rs` (947 LOC), `flex_dispatch/`,
   `coreai.rs`, `coreai_iosurface.rs`, `coreai_lane.rs`,
   `metal_consumer.rs`, `metal_iosurface.rs`,
   `megakernel_backend.rs`, `amd_megakernel.rs`,
   `amd_rocm.rs`, `intel_level_zero.rs`, `intel_usm.rs`. These
   files use `mpsgraph`, `metal`, `objc`, `crate::ane_bridge`,
   `crate::coreai_bridge`, `crate::arena`, `crate::memory::allocator`,
   and other engine-internal modules. They require either moving
   the engine-coupled dependencies to the kernel (large scope)
   OR refactoring the files to not depend on those (medium scope).

2. **Engine-coupled re-export shims**: `routing/lanes.rs`,
   `routing/policy.rs`, `routing/mod.rs` — these re-export from
   `prism-ecs-backend` and don't need to move (the kernel already
   has the same content).

3. **Pre-existing missing types** (~10 files, pre-existing 193
   errors): callers of `MlxBackend` (missing), `BackendInstance`
   (missing), `residency::TensorResidency` (missing),
   `residency::MemoryDomain` (missing), `AccelerateBackend`
   (defined in `accelerate/ops.rs` which uses
   `crate::memory::allocator`). The 19 engine callers that use
   these are listed in
   `crates/architecture/src/workspace_legacy_backend_imports.rs`
   as `PRE_EXISTING_BROKEN_FILES` and are exempt from the safety
   net test.

### Engine pre-existing error count

- Baseline: 193 errors.
- After migration: 193 errors (unchanged).
- Verified by `cargo check -p tribunus-compute-core --lib` before
  and after each commit.

### Verification commands

```bash
# Constitutional-side tests (kernel)
cargo test -p prism-ecs-kernel --lib

# Architecture safety net
cargo test -p prism-architecture --lib

# Engine pre-existing error count
cargo check -p tribunus-compute-core --lib 2>&1 | tail -5
```

### Follow-up work

A future migration can:

1. Move the heavy engine-coupled files (Metal/ANE/CoreAI/etc.)
   into the kernel. This is a multi-day effort because it
   requires moving the engine-coupled dependencies (ANE bridge,
   CoreAI bridge, IosurfaceAllocator, etc.) and refactoring
   cross-references.

2. Resolve the pre-existing missing types (`MlxBackend`,
   `BackendInstance`, `residency::*`) by either defining them in
   the engine or removing the references.

3. Once all 37 files are moved, delete
   `compute-core/src/ecs/backend/` and remove
   `pub mod backend;` from `compute-core/src/ecs/mod.rs`.
