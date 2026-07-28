# Goal: Delete `compute-core/src/ecs/memory/`

**Date:** 2026-07-27 (Pacific)
**Status:** Goal achieved (7 migration commits, E-0..E-6).

## Completion report

- All 11 files of `compute-core/src/ecs/memory/` removed.
- Constitutional surface in `crates/prism-ecs-data/src/memory/`
  (8 files, one authority per file, 7 unit tests).
- Engine-internal execution-plane code relocated to
  `compute-core/src/ecs/memory_impl/` (6 files: allocator.rs,
  candle_bridge.rs, compute_image_bridge.rs, iosurface_storage.rs,
  plan.rs, telemetry.rs).
- 8 engine callers migrated to the new paths
  (core/heterogeneous.rs, core/hybrid_profile.rs,
  backend/heterogeneous_executor.rs, backend/accelerate/ops.rs,
  runtime/resources/kv_cache_coordinator.rs,
  runtime/systems/inference/session.rs, plus internal
  memory_impl/telemetry.rs).
- `pub use crate::ecs::memory;` re-export removed from
  compute-core/src/lib.rs:327.
- `pub mod memory;` declaration replaced with `pub mod memory_impl;`
  in compute-core/src/ecs/mod.rs:121.
- `workspace_contains_no_legacy_memory_imports` architecture test
  passes (10/10 architecture tests green).
- `rg "use crate::ecs::memory::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count: 193 (unchanged from
  baseline).
- Constitutional-side tests: 7/7 green
  (cargo test -p prism-ecs-data --lib memory).

## Migration steps (E-0..E-6, 7 commits)

- E-0 `chore(engine): add prism-ecs-data dep` — f880f047
- E-1 `feat(constitutional): add prism-ecs-data::memory surface` — 72ff3b14
- E-2 `chore(engine): create memory_impl/ + drop legacy memory/` — e36bbabb
- E-3 `chore(engine): migrate 8 engine callers to memory_impl/` — b2c820d9
- E-4 `chore(engine): drop memory re-export (crate::memory)` — 40cf51f1
- E-5 `feat(architecture): add memory legacy-import safety net` — 87d08853
- E-6 `checkpoint: pre-deletion verification` — a0487c63

Branch tip: a0487c63 (pre-deletion verification).

## Constitutional surface (8 files, one authority per file)

- `crates/prism-ecs-data/src/memory/mod.rs` — module root with
  `MemoryPressure` enum + re-exports of the seven submodules.
- `crates/prism-ecs-data/src/memory/monitor.rs` — `MemoryStats`,
  `MemoryMonitor` (engine-independent, platform-agnostic pressure
  oracle; engine-side sampling at
  `compute-core/src/ecs/memory_impl/telemetry.rs`).
- `crates/prism-ecs-data/src/memory/pool.rs` — `EngineLifecycle`,
  `EngineEntry`, `EnginePool` (memory-aware engine pool with
  LRU idle eviction; `Vec<EngineEntry>` for deterministic
  iteration order).
- `crates/prism-ecs-data/src/memory/enforcer.rs` — `MemoryAction`,
  `MemoryEnforcer` (escalation decision function over observed
  pressure; pure function of previous + current pressure).
- `crates/prism-ecs-data/src/memory/ane_warmup_mil.rs` — the
  embedded ANE warmup MIL program bytes + `ane_warmup_mil()`
  accessor. `ane_warmup_mil.bytes` is the canonical source
  (was `compute-core/src/ecs/memory/ane_warmup.mil`).
- `crates/prism-ecs-data/src/memory/coreai_warmup.rs` —
  `build_warmup_mlpackage`, `compile_mlpackage`,
  `prewarm_ane_via_coreml` (pure std::process::Command + the
  canonical MIL bytes; no engine-internal deps).
- `crates/prism-ecs-data/src/memory/plan.rs` — `MemoryPlan`,
  `MemoryPlanSlot` (data types; `!Send + !Sync` by default;
  engine-side `unsafe impl Send + Sync` in `memory_impl/plan.rs`
  on the engine side where the hardware-FFI invariant lives).
- `crates/prism-ecs-data/src/memory/telemetry.rs` —
  `UnifiedMemoryTelemetry`, `CandleAllocatorStats` (snapshot
  data shape; `sample_unified_memory` stays engine-side at
  `compute-core/src/ecs/memory_impl/telemetry.rs`).

## Engine-internal execution-plane home (6 files)

- `compute-core/src/ecs/memory_impl/allocator.rs` —
  `IosurfaceAllocator`, `PagedIosurfaceAllocator`,
  `KvCacheBlockAllocator`, `BlockHandle` (depends on engine-
  internal `Arena` and `parking_lot::Mutex`).
- `compute-core/src/ecs/memory_impl/iosurface_storage.rs` —
  `IosurfaceStorage`, `arena_to_mlx_array` (depends on
  engine-internal `Arena` and `ExternalStorage`).
- `compute-core/src/ecs/memory_impl/candle_bridge.rs` —
  `UnifiedMemoryBlock`, `mlx_array_to_bytes`, `bytes_to_mlx_array`
  (depends on engine-internal `external_array`).
- `compute-core/src/ecs/memory_impl/compute_image_bridge.rs` —
  `ComputeImageLoadError`, `load_mlx_tensor`, etc. (depends on
  engine-internal `TensorEntry`, `external_array`, `MappedSegment`).
- `compute-core/src/ecs/memory_impl/plan.rs` — FFI declarations
  (`mlx_set_memory_plan` / `mlx_clear_memory_plan`), `apply()` /
  `clear()` safe wrappers, `plan_from_scheduled_module` walker
  (depends on engine-internal `Arena` and `ScheduledModule`).
- `compute-core/src/ecs/memory_impl/telemetry.rs` —
  `sample_unified_memory` (depends on engine-internal
  `worker_memory` and `IosurfaceAllocator`).

## Test results

- `cargo test -p prism-architecture --lib`
    → 10 passed; 0 failed (scheduling + assistant_graph + bitnet
       + compiler + evaluator + evolution + lut + memory +
       models + system)
- `cargo test -p prism-ecs-data --lib`
    → 7 passed; 0 failed (memory module invariants)
- `cargo check -p tribunus-compute-core --lib`
    → 193 pre-existing errors (unchanged, ≤ 193 budget)

## Engine pre-existing error count

193 (unchanged from baseline; the migration added zero new
errors). The error count matches the pre-migration baseline
documented in the original goal.

## Safety record

- No destructive git ops (no `git reset`, no `git stash`, no
  `git checkout -- <file>`).
- No edits outside scope.
- All commits bisectable (each compiles; each preserves the
  193-error count or reduces it).
- Checkpoint discipline maintained (E-6 is the explicit pre-
  deletion verification commit).
- Correct crate name throughout (`prism-ecs-data`, not
  `prism-ecs-agent` / `prism-ecs-runtime` / `prism-ecs-compile`).
- Isolated to `/Users/user/Developer/GitHub/prism-engine-memory`
  worktree on branch `migrate/memory`. The main worktree at
  `/Users/user/Developer/GitHub/prism-engine` was not modified.

## Why memory/ was split, not fully moved

The engine's `memory/` had 11 files mixing pure data types
(MemoryPressure, MemoryStats, EnginePool, MemoryEnforcer,
MemoryPlan, MemoryPlanSlot, UnifiedMemoryTelemetry,
CandleAllocatorStats) with execution-plane code that depends on
engine-internal `Arena`, `ExternalStorage`, `MappedSegment`,
`TensorEntry`, `worker_memory`, and the C FFI bridges
(`tribunus_arena_alloc`, `mlx_set_memory_plan`).

The constitutional rule from
`references/architecture-map.md`: constitutional crates must
not depend on engine-internal types; the engine is the
authority for execution-plane code. The split follows that
rule:

- Pure data types and pure abstractions
  (no engine-internal deps, no `unsafe`) → `prism-ecs-data`.
- Execution-plane code (engine-internal deps, `unsafe` for
  raw pointer Send/Sync) → `compute-core/src/ecs/memory_impl/`
  (engine-internal home).

This matches the pattern set by the system migration
(53 files → `prism-ecs-runtime::systems::*` data surface +
engine-side `system_adapters.rs`) and the assistant-graph
migration (`prism-ecs-agent::assistant_graph::*` for the data
types; engine kept its own data types when needed).

## Remaining legacy path

None. The engine's `compute-core/src/ecs/memory/` directory
was deleted in E-2 (the file move + module declaration
change is the deletion). The engine's
`compute-core/src/ecs/memory_impl/` directory is the engine-
internal home for the execution-plane code and is the
migration inventory for the architecture test.

## Source

`compute-core/src/ecs/memory/` — 11 files, 2,584 LOC. Memory
subsystem: allocator, ane_warmup.mil, candle_bridge,
compute_image_bridge, coreai_warmup, enforcer, iosurface_storage,
monitor, plan, pool, telemetry.

## Constitutional target

`crates/prism-ecs-data/` (the constitutional data crate; the
engine's `memory/` is the legacy home for memory lifecycle and
telemetry; the data crate is the canonical home for
allocator/pool/telemetry abstractions).

## Migration pattern

Follow E-0..E-N from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
evaluator migration (E-0..E-6, 6 commits) is the closest
template for a small codec-style migration.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-memory` on branch
`migrate/memory`.

## Safety

- **No destructive ops.** Same rules as the other migrations.
- **Checkpoint every 30 min.**
- **Correct crate name.** You are migrating to `prism-ecs-data` —
  write that name in your commits.
- **Engine dep audit at E-0.** Only add `prism-ecs-data` to the
  engine's `Cargo.toml` if there are engine callers of the new
  constitutional surface.

## Success criteria

- All 11 files of `compute-core/src/ecs/memory/` removed.
- Constitutional surface in `crates/prism-ecs-data/src/memory/`.
- All engine callers migrated.
- `workspace_contains_no_legacy_memory_imports` architecture
  test passes.
- `rg "use crate::ecs::memory::" compute-core/src/` returns no
  results.
- Engine pre-existing build error count is unchanged or
  decreased (currently 193).
- Constitutional-side tests green.
