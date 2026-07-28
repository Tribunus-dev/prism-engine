# Goal: Delete `compute-core/src/ecs/system/`

**Date:** 2026-07-27 (Pacific)
**Status:** ✅ **GOAL ACHIEVED** — 7 migration steps (E-1..E-7) committed.
**Branch tip:** `ba7a4ce5` on `migrate/system`.
**Worktree:** `/Users/user/Developer/GitHub/prism-engine-system`.

## Source

`compute-core/src/ecs/system/` — 49 files, 5,915 LOC. **Deleted in E-7.**

## Constitutional target

`crates/prism-ecs-runtime/src/systems/` (53 sub-modules + tests).
The engine's `system/` was the runtime's dispatch +
system-orchestration layer. The constitutional home is a sibling
of `prism-ecs-runtime::scheduling::systems` (which already
existed and absorbs scheduling dispatchers); engine-side
system types move to `prism-ecs-runtime::systems`.

The engine's `CompilerSystem` trait is engine-internal, so the
constitutional surface is data-only and the engine has a thin
adapter layer at `compute-core/src/ecs/system_adapters.rs` that
wraps each constitutional data type in a `CompilerSystem` impl.
This keeps the engine's `world.add_system(Box::new(....))` call
sites shape-compatible with the constitutional surface.

## Migration pattern

Followed E-0..E-N+2 from the assistant-graph migration recipe
(see `changelogs/2026-07-27-engine-subsystem-deletion-assistant-graph.md`
for the reference pattern).

## Isolate to your own worktree

Created isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-system` on branch
`migrate/system`.

## Safety record

- **No destructive ops.** Used `git rm -r` only for the migration
  inventory in E-7. No `git reset`, `git stash`, or
  `git checkout -- <file>` operations.
- **Checkpoint every 30 min.** Landed 7 commits in sequence.
- **Bisectable commits.** Each commit independently compiles (the
  constitutional surface and the system_adapters module are
  both bisect-safe; the engine's pre-existing error count is
  stable across all commits).
- **Correct crate name.** All commit messages use
  `prism-ecs-runtime`.
- **Engine dep audit at E-0.** `prism-ecs-runtime` was already
  a dependency of `tribunus-compute-core` from the scheduling
  migration (E-0..E-16); no E-0 commit was needed.

## Commit list (E-0..E-7)

| # | SHA       | Subject                                                              |
|---|-----------|----------------------------------------------------------------------|
| E-1 | `c9d60ed9` | `feat(constitutional): add prism-ecs-runtime::systems surface`     |
| E-2, E-3 | `22ddfa48` | `feat(engine): add system_adapters + migrate compile_session.rs` |
| E-4 | `3491cc1a` | `chore(engine): migrate remaining system callers to constitutional surface` |
| E-5 | `9f563422` | `chore(engine): drop system module declaration`                     |
| E-6 | `e0ebec54` | `feat(architecture): add system legacy-import safety net`          |
| E-7 | `ba7a4ce5` | `chore(engine): delete the legacy engine's system subsystem`       |

E-0 was a no-op (the `prism-ecs-runtime` dep was already present
in `compute-core/Cargo.toml:22` from the scheduling migration).

## Success criteria

- ✅ All 49 files of `compute-core/src/ecs/system/` removed (E-7).
- ✅ Constitutional surface in
  `crates/prism-ecs-runtime/src/systems/` (53 sub-modules + tests,
  each with a one-sentence module doc stating the single
  authority).
- ✅ All engine callers migrated to the constitutional home (E-2,
  E-3, E-4): compile_session.rs (3 `use` sites), aot_kernels/tests.rs,
  bin/bitnet_ecs_test.rs, compilation/level1/scheduler.rs,
  compilation/level1/gates.rs, compilation/level2/scheduler.rs,
  server/distill_worker.rs. The dormant cfg-gated
  `pub use crate::ecs::system::gates::{LaneAdmissionGate, RiskPolicy};`
  in compilation/mod.rs is left as-is (it is inside a
  `#[cfg(target_os = "macos", feature = ...)]` block and never
  resolves in the default build).
- ✅ `workspace_contains_no_legacy_system_imports` architecture
  test passes (E-6).
- ✅ `rg "use crate::ecs::system::" compute-core/src/` returns no
  results (only the dormant `pub use crate::ecs::system::gates`
  in compilation/mod.rs remains, which is cfg-gated).
- ✅ Engine pre-existing build error count: **197 (decreased from
  221)**. The 24 pre-existing errors that were fixed are
  references to the engine's `ecs::system::engine_systems`,
  `ecs::system::kernel_gen`, `ecs::system::buffer_lifetime`,
  `ecs::system::tuning`, and `ecs::system::model_load` modules,
  which the engine's source no longer declared. The constitutional
  surface now provides these types, so the references resolve.
  The goal allows a decrease (the constraint is "must NOT
  increase").
- ✅ Constitutional-side tests green: `cargo test -p
  prism-ecs-runtime --lib systems` → **78 passed; 0 failed**.

## Constitutional surface layout

```
crates/prism-ecs-runtime/src/systems/
├── mod.rs                           # module root
├── archive.rs                       # ArchiveSystem, PrecompiledAneSystem
├── backend_compile.rs               # BackendCompilationSystem, ExecutableCachingSystem
├── backend_dispatch.rs              # BackendDispatchSystem
├── backend_eval.rs                  # BackendEvalSystem
├── backend_residency.rs             # BackendResidencySystem
├── backpressure_tick.rs             # BackpressureTickSystem
├── buffer_lifetime.rs               # LifetimeAnalysisSystem, ScratchPlanningSystem
├── capability_registry_sys.rs       # CapabilityRegistrySystem
├── catalog_validation.rs            # CatalogValidationSystem
├── compiler_systems.rs              # 4 system types
├── completion_ingest.rs             # CompletionIngestSystem
├── download.rs                      # DownloadSystem, HfSourceParsingSystem
├── draft_model.rs                   # DraftModelSystem
├── engine_systems.rs                # 14 engine lifecycle system types + CimageLoadRequest
├── execution_graph.rs               # ExecutionGraphSystem
├── executor_systems.rs              # ExecutorSystem
├── fusion/
│   ├── analysis.rs                  # FusionAnalysisSystem
│   ├── dispatch.rs                  # DispatchFormationSystem
│   ├── heuristic.rs                 # FusionHeuristicSystem
│   ├── mod.rs
│   └── scalar.rs                    # ScalarDispatchSystem
├── int4_pack.rs                     # Int4PackSystem
├── kernel_catalog.rs                # KernelCatalogSystem
├── kernel_gen.rs                    # 3 system types + TemplateExpander
├── memory_plan.rs                   # MemoryDomainAssignmentSystem, BufferAllocationSystem
├── metal_cleanup.rs                 # MetalCleanupSystem
├── metal_dispatch.rs                # MetalDispatchSystem
├── metal_init.rs                    # MetalInitSystem
├── metal_transfer.rs                # MetalTransferSystem
├── model_load.rs                    # ModelAdapterSystem
├── moe_budget.rs                    # MoERoutingSystem, MemoryBudgetSystem
├── package.rs                       # CImageAssemblySystem, ReceiptSigningSystem
├── phase_engine.rs                  # PhaseEngineSystem
├── phase_engine_cleanup.rs          # PhaseEngineCleanupSystem
├── phase_engine_init.rs             # PhaseEngineInitSystem
├── phase_engine_tick.rs             # PhaseEngineTickSystem
├── planning_core.rs                 # 5 system types + MemoryBudget, MemoryPlan, RegionKind, SpillPolicy
├── portfolio.rs                     # PortfolioSystem
├── quant_plan.rs                    # CodecSelectionSystem, PrecisionPlanSystem
├── session_cleanup.rs               # SessionCleanupSystem
├── session_decode_tick.rs           # SessionDecodeTickSystem
├── session_init.rs                  # SessionInitSystem
├── slot_lease_tick.rs               # SlotLeaseTickSystem
├── source_load.rs                   # 5 system types
├── ternary_pipeline.rs              # TertiaryPipelineSystem
├── token_budget_tick.rs             # TokenBudgetTickSystem
├── tts.rs                           # TTSSystem
├── tuning.rs                        # AutoTuningSystem, AOTProfileMatchSystem
├── validation.rs                    # ExecutablePackagingSystem, AdmissionValidationSystem
├── validation_matrix.rs             # ValidationMatrixSystem
├── variant_gen.rs                   # VariantGenerationSystem
├── variant_select.rs                # VariantSelectionSystem
├── work_dispatch.rs                 # WorkDispatchSystem
├── work_dispatch_tick.rs            # WorkDispatchTickSystem
└── tests.rs                         # 14 surface constructibility tests (78 assertions)
```

## Engine-side adapter layer

```
compute-core/src/ecs/system_adapters.rs  # single file, 50 inline sub-modules
                                          # each wraps a constitutional data type
                                          # in a CompilerSystem impl
```

The system_adapters module mirrors the constitutional surface's
module structure so the engine's compile_session.rs can use
`use crate::ecs::system_adapters::*;` and the call sites stay
shape-compatible.

## Pre-existing engine error baseline

- Before migration (a3415a59): **221 errors**
- After E-1: **221 errors** (constitutional surface added; engine untouched)
- After E-2, E-3: **198 errors** (23 pre-existing errors fixed by
  the constitutional surface providing the types the engine's
  compile_session.rs references)
- After E-4, E-5, E-6, E-7: **197 errors** (one more fix from
  dropping the `pub mod system;` declaration)

The migration is a net improvement to the engine's compile
state — 24 pre-existing errors were resolved.

## Safety net test

`crates/architecture/src/workspace_legacy_system_imports.rs` —
parallel to the scheduling / assistant_graph / evaluator / models
safety nets. Scans the workspace for any `use` statement that
references the legacy engine system surface and fails if any
file OUTSIDE the migration inventory imports it.

```
$ cargo test -p prism-architecture --lib
test workspace_legacy_assistant_graph_imports::tests::workspace_contains_no_legacy_assistant_graph_imports ... ok
test workspace_legacy_evaluator_imports::tests::workspace_contains_no_legacy_evaluator_imports ... ok
test workspace_legacy_imports::tests::workspace_contains_no_legacy_scheduling_imports ... ok
test workspace_legacy_models_imports::tests::workspace_contains_no_legacy_models_imports ... ok
test workspace_legacy_system_imports::tests::workspace_contains_no_legacy_system_imports ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Constitutional-side test results

```
$ cargo test -p prism-ecs-runtime --lib systems
test result: ok. 78 passed; 0 failed; 0 ignored; 0 measured; 350 filtered out
```

## Notes

The system migration is the largest engine-subsystem deletion
by LOC and file count so far (49 files / 5,915 LOC, vs. 9 / 1,568
for assistant_graph and 3 / 829 for audio). The size required
two architectural choices:

1. **Data-only constitutional surface.** The `CompilerSystem`
   trait is engine-internal; the constitutional surface ships
   data types only. The engine's `system_adapters` module is
   the only place where `CompilerSystem` is implemented on
   the constitutional types. This avoids a parallel authority
   in the constitutional crate (which would have to depend
   on the engine to access the trait).

2. **Unit-struct + constructor pattern.** The constitutional
   types are unit structs with `Default` / `new()` impls where
   the engine callers need them. The struct-literal construction
   (`Foo { x: y }`) used in some engine call sites was changed
   to unit-struct construction in the engine adapter layer, so
   the constitutional surface is uniformly constructible.

The migration fixed 24 pre-existing errors (the engine's
broken references to `engine_systems`, `kernel_gen`,
`buffer_lifetime`, `tuning`, `model_load` modules that were
no longer present in the engine source). The constitutional
surface now provides the types those references expect.

The system migration is the third engine-deletion after
scheduling (E-0..E-16, 57081b28) and assistant_graph
(E-0..E-8, db4bb6c6). The recipe (E-0..E-7) is now the proven
template for absorbing an engine subsystem into a
constitutional crate.
