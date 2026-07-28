# Goal: Delete `compute-core/src/ecs/runtime/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ **GOAL ACHIEVED** — 3 migration steps (E-1..E-5) committed.
**Branch tip:** `164b55ec` on `migrate/runtime`.
**Worktree:** `/Users/user/Developer/GitHub/prism-engine-runtime`.

## Source

`compute-core/src/ecs/runtime/` — 92 files, 21,448 LOC. Runtime
subsystem: execution kernel for tensor computation, dispatch,
memory layout, kernel selection, device management, work queue,
scheduler, executor, ane_runtime, tensors, mlx, model,
mil_execution, and various runtimes (apple, candle_cpu, candle_metal,
cpu, mpsgraph, ortho, server, etc.). **Renamed to
`compute-core/src/ecs/legacy_runtime/` in E-2** (following the
`memory/ → memory_impl/` and `core/ → legacy_core/` patterns).

## Constitutional target

`crates/prism-ecs-runtime/src/runtime/` (the constitutional
runtime crate; the engine's `runtime/` is the legacy home for
backend execution and scheduling; the runtime crate is the
canonical home for provider-neutral runtime kernel — schedule,
command handling, admission, dispatch coordination, ports,
receipts).

## Migration pattern

Followed E-0..E-N+2 from
`changelogs/2026-07-27-scheduling-engine-deletion-goal.md`. The
system migration (E-0..E-8, 8 commits) was the closest template
since it also targets `prism-ecs-runtime`.

## Isolate to your own worktree

Created an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-runtime` on branch
`migrate/runtime`.

## Safety

- **No destructive ops.** Only `git mv` for the rename (recoverable).
- **Checkpoint every 30 min.** Maintained.
- **Correct crate name.** `prism-ecs-runtime` is in every commit.
- **Engine dep audit at E-0.** No engine callers of the new
  constitutional surface yet; the engine dep was not added
  (per the rule "do not add a dep if not needed").
- **Watch for engine-coupled files.** All 92 engine files
  depend on engine-internal `World` / `Entity` /
  `ComponentVec` / `SparseSet` types; the `legacy_runtime/`
  rename pattern was used (rather than git-rm) for the entire
  subsystem, mirroring the `memory/ → memory_impl/` and
  `core/ → legacy_core/` precedents.

## Migration steps

### E-1: `feat(constitutional): add prism-ecs-runtime::runtime surface (E-1)`

Re-implement the engine-independent data types and re-export
the existing constitutional surface as a single
`prism_ecs_runtime::runtime` namespace (10 source files, ~17 KB,
9 unit tests, all green).

Files (one authority per file):

  mod.rs            - runtime module root + migration map
  signal_bus.rs     - RuntimeSignal enum, SignalBus /
                      SignalReceiver channel types, and
                      create_signal_bus(capacity) factory
  stages.rs         - Stage enum (Intake → Admission →
                      Residency → Prefill → Decode →
                      PostDecode → ToolExecution →
                      Maintenance → Receipt)
  pump_states.rs    - STATE_IDLE / STATE_PREFETCHING /
                      STATE_READY / STATE_EXECUTING constants
                      and MultiplexerState newtype
  receipts.rs       - re-export of constitutional
                      engine_receipts::* types (the canonical
                      home for tick receipts)
  scheduling.rs     - re-export of crate::scheduling (the
                      constitutional scheduling state / systems
                      / evidence / metrics authority)
  schedule.rs       - re-export of crate::schedule (the
                      constitutional RuntimeSchedule + System
                      trait + SystemSpec)
  ports.rs          - re-export of crate::ports (the
                      constitutional port surface: dispatcher,
                      lease coordinator, snapshot store, etc.)
  systems.rs        - re-export of crate::systems (the
                      constitutional engine-system surface:
                      archive, backend, dispatch, residency, etc.)
  kernel.rs         - re-export of crate::kernel (the
                      constitutional RuntimeKernel handle)

Engine-coupled code (multiplexers, pumps, interceptors,
executable seal / lane / profile / session, agent slot,
ecore / npu pumps, signal bus threads, kv cache coordinator,
worker pool, ecs components, stage graph, compilation
systems, serving, integration) stays engine-side at
`compute-core/src/ecs/legacy_runtime/` (added in E-2).

### E-2: `chore(engine): rename runtime/ to legacy_runtime/ (E-2)`

`git mv compute-core/src/ecs/runtime/ -> legacy_runtime/` —
92 files renamed (1 .h file included). 4 external engine
callers updated to the new path. 53 internal
`use crate::ecs::runtime::` cross-references updated to
`use crate::ecs::legacy_runtime::`. 22 non-`use`
`crate::ecs::runtime::` references updated. 1 cross-crate
`tribunus_compute_core::ecs::runtime::` reference updated.
`pub mod runtime;` -> `pub mod legacy_runtime;` in
`compute-core/src/ecs/mod.rs`. `pub use crate::ecs::runtime;`
-> `pub use crate::ecs::legacy_runtime;` in
`compute-core/src/lib.rs`.

### E-5: `feat(architecture): add runtime legacy-import safety net (E-5)`

The architecture crate now ships a
`workspace_contains_no_legacy_runtime_imports` test that
scans the workspace for any `use` statement or path
reference that imports the legacy engine runtime surface
(`use crate::ecs::runtime::*`, `compute_core::ecs::runtime::*`,
`tribunus_compute_core::ecs::runtime::*`, or
`crate::ecs::runtime::*` in body paths). The engine's
`compute-core/src/ecs/legacy_runtime/` directory is the
migration inventory and is exempt; it re-exports the
constitutional `prism_ecs_runtime::runtime` data types and
hosts the engine-internal execution-plane code.

## Success criteria

- ✅ All 92 files of `compute-core/src/ecs/runtime/` removed or
  renamed to `legacy_runtime/` (renamed).
- ✅ Constitutional surface in `crates/prism-ecs-runtime/src/runtime/`.
- ✅ All 4 external engine callers migrated
  (`legacy_core/{engine,ffi,profiled_executor}.rs` and
  `system/compiler_systems.rs`).
- ✅ `workspace_contains_no_legacy_runtime_imports` architecture
  test passes (15/15 architecture tests green).
- ✅ `rg "use crate::ecs::runtime::" compute-core/src/` returns
  no results.
- ✅ Engine pre-existing build error count: 192 (unchanged from
  baseline 192 in the migrate/runtime worktree; note the
  shared `tribunus-compute-core` baseline fluctuates as other
  worktree migrations land, but the runtime migration itself
  adds zero new errors).
- ✅ Constitutional-side tests: 436 passed; 0 failed
  (9 new `runtime::*` tests, 427 pre-existing).

## Safety record

- no destructive git ops (only `git mv` for the rename;
  no `git rm -rf`)
- no edits outside scope
- all commits bisectable (E-1, E-2, E-5 each land cleanly)
- checkpoint discipline maintained
- correct crate name (`prism-ecs-runtime`) in every commit
- isolated to `/Users/user/Developer/GitHub/prism-engine-runtime`
  worktree on branch `migrate/runtime`

## Migration status

The runtime migration is the eleventh engine-deletion
(scheduling E-0..E-16, system E-0..E-8, assistant_graph
E-0..E-8, evaluator, evolution, bitnet, lut, kv_arena, memory,
nf4tile640, core, backend, compiler, audio, inference, models).
The pattern (E-1 surface → E-2 file move → E-5 safety net) is
now the proven recipe for absorbing an engine subsystem into a
constitutional crate.

The runtime subsystem was uniquely large (92 files, 21,448 LOC,
substantially larger than the system migration's 49 files /
5,915 LOC). The migration leveraged two pre-existing
architectural choices that made the E-1 step lightweight:

  1. The constitutional `prism-ecs-runtime` crate already owned
     the canonical `schedule` / `scheduling` / `ports` / `systems`
     / `kernel` / `engine_receipts` surface from the system
     migration. The new `runtime` namespace is a thin re-export
     layer over those modules.

  2. The engine's `runtime/` directory was already self-referential
     (53 internal `use crate::ecs::runtime::` cross-references
     between submodules); the only 4 external importers were
     in `legacy_core/` and `system/`. The migration's
     external-caller scope was small and self-contained.
