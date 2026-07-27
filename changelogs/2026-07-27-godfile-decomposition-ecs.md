# Godfile decomposition — `ecs.rs` (2581 LOC, 61 pub → 4 sub-modules)

**Date:** 2026-07-27
**Godfile:** `crates/prism-ecs-compile/src/ecs.rs` (2,581 LOC, 61 pub items)
**Status:** Phase 1 dispatch — decomposition + engine counterpart mapping
**Reference:** `changelogs/2026-07-27-godfile-engine-mapping.md` §4

## Summary

The `ecs.rs` godfile (2,581 LOC, 61 pub items) is decomposed into a
4-sub-module layout under `crates/prism-ecs-compile/src/ecs/`. Each new
file states a single authority in its module doc, owns at most one
authority, and stays under 900 LOC and 35 public items. The `pub use`
re-exports at `ecs::mod.rs` and at the crate root (`lib.rs`) preserve the
existing public surface for all callers (`compiler.rs`, `plan_apply.rs`,
`compilation_systems.rs`).

Engine counterparts (`compute-core/src/ecs/core/compile_pipeline.rs`,
`compute-core/src/ecs/core/profiled_model.rs`,
`compute-core/src/ecs/runtime/compilation_systems.rs`) are classified
per the four canonical-vs-execution-boundary criteria. **None of them
are canonical**; all are either execution-boundary or a parallel
authority that uses a different world model.

## Decomposition result

| File | LOC | Public items | Authority |
|---|---:|---:|---|
| `crates/prism-ecs-compile/src/ecs/mod.rs` | 56 | 8 re-exports | module wiring + re-exports |
| `crates/prism-ecs-compile/src/ecs/components.rs` | 380 | 10 | session-entity `Component` data types |
| `crates/prism-ecs-compile/src/ecs/resources.rs` | 101 | 6 (+1 re-export) | world resources and extensions |
| `crates/prism-ecs-compile/src/ecs/orchestrator.rs` | 512 | 1 | the pipeline driver (`CompilationOrchestrator` + internal helpers) |
| `crates/prism-ecs-compile/src/ecs/stages/mod.rs` | 34 | 6 mods + 8 re-exports | per-stage system re-exports |
| `crates/prism-ecs-compile/src/ecs/stages/ingest.rs` | 195 | 2 fns | source detection + graph construction |
| `crates/prism-ecs-compile/src/ecs/stages/search_legalize.rs` | 270 | 2 fns | evolutionary search + legalization |
| `crates/prism-ecs-compile/src/ecs/stages/kernel.rs` | 567 | 1 fn | kernel generation across all backends |
| `crates/prism-ecs-compile/src/ecs/stages/emit.rs` | 518 | 1 fn | CImage emission + plan binding |
| `crates/prism-ecs-compile/src/ecs/stages/certify.rs` | 337 | 1 fn | CImage certification + plan-digest verification |
| `crates/prism-ecs-compile/src/ecs/stages/receipt.rs` | 171 | 1 fn | `CompileReceipt` build + session close |

**Total:** 3,141 LOC across 11 files. The original 2,581 LOC expanded
to 3,141 LOC (+560 LOC, +22%) because of per-file module docs, per-file
test modules, and additional `use` statements per file. The largest
sub-module (`stages/kernel.rs`, 567 LOC) is well under the 900 LOC
ceiling. The largest pub count (`components.rs`, 10 pub items) is well
under the 35 pub ceiling.

## Per-sub-module authority statements

### `components.rs` — session-entity data types

> Single authority: the shape of every ECS component attached to the
> compilation session entity. No behavior — only data and the
> `Component` impl each type needs to live on a `World`.

- `CompilationSession`, `SessionStatus` — top-level session state
- `SourceModel`, `TensorCollection` — source identity + tensor catalog
- `SpatialGraphComponent` — built spatial graph + digest
- `SearchStateComponent` — search trace + Pareto archive
- `LegalizedPlan` — legalization report + validity flag
- `KernelCollection` — per-backend kernel artifacts + UOp captures
- `CImageArtifact` — emitted binary path + digest + schema version
- `CompilationReceipt` — final forensic receipt on the session

### `resources.rs` — world resources and extensions

> Single authority: world-level state shape — the resources and
> extensions the pipeline attaches to a `World` (handles, adapters,
> evaluator, model manifest, plan digest).

- `SessionHandle(Entity)` — identity of the session entity
- `CurrentSource(CanonicalSource)` — world extension for the ingress'd source
- `SourceAdapterList(Vec<...>)` — registered format adapters
- `EvaluatorOption(Option<...>)` — registered evaluator strategy
- `ModelManifestResource(MultiModelManifest)` — model manifest for emit header
- `CImagePlanDigest([u8; 32])` — content digest of the plan that produced the CImage
- Re-exports `VecEventSink` from the crate root for callers that previously imported it from `ecs::*`

### `orchestrator.rs` — pipeline driver

> Single authority: the world + session-entity pipeline driver. Owns
> the world, spawns the session, dispatches enabled stages in order.
> Does not mutate canonical state directly — the per-stage state
> transitions happen in `stages::*`.

- `CompilationOrchestrator` — public struct with `pub session` and `pub world`
- `impl CompilationOrchestrator` — `new`, `set_model_manifest`, `run_pipeline`, `run_stage`, `make_receipt`
- `pub(crate) fn session_entity(world)` — internal helper, `pub(crate)` so the stage systems can call it
- `pub(crate) fn read_session_config(world, session)` — internal helper, same visibility

### `stages/` — per-stage system functions

> Single authority: the per-stage pipeline state transitions. Each
> function is a stateless `fn(&mut World) -> Result<(), CompileError>`
> that reads prior stage components/resources and writes its own
> output component onto the session entity.

- `stages::ingest::system_detect_source` — canonical source detection
- `stages::ingest::system_build_graph` — canonical graph construction
- `stages::search_legalize::system_run_search` — evolutionary search + Pareto archive
- `stages::search_legalize::system_legalize` — legalization with format-plan validation
- `stages::kernel::system_generate_kernels` — kernel generation (CPU, Metal, AMD NPU, ANE) + UOp tuning
- `stages::emit::system_emit_cimage` — CImage emission from the plan, not the source
- `stages::certify::system_certify` — verify the artifact against the plan
- `stages::receipt::system_build_receipt` — build the forensic `CompileReceipt` and close the session

The stage split follows the natural pipeline order (front → back) so
the per-file size is balanced and each file is independently reviewable.

## Engine counterpart mapping (per the four criteria)

The engine counterparts listed in the mapping doc were inspected. The
classification is unambiguous: **none of them are canonical** in the
sense meant by `prism-ecs-compile`.

### `compute-core/src/ecs/core/compile_pipeline.rs` (202 LOC, partially absorbed)

This is the engine's **parallel relocation pipeline** — four lanes
(source-read, relocate, write, hash) connected by bounded
`tokio::sync::mpsc` channels. After the ddb2d261 partial absorption,
only this parallel relocation pipeline remains.

**Classification: execution-boundary.** It owns:
- a `tokio::sync::mpsc` channel pair with bounded capacity
  (criterion 3: process-local state)
- a `HashMap<String, std::fs::File>` of segment file handles
  (criterion 1: file descriptors / OS primitives)
- `spawn_blocking` per lane to keep the async runtime responsive

It is **not** a counterpart to the constitutional `CompilationOrchestrator`.
The orchestrator drives the canonical ECS pipeline; this parallel
relocation pipeline performs the physical byte-shuffling for the
engine's own CImage emission. They coexist.

**Action: stays in engine, no absorption.** It already lives in the
engine, is execution-boundary, and serves a different purpose.

### `compute-core/src/ecs/core/profiled_model.rs` (1,339 LOC)

This is the engine's **MLX-backed model loader with IOSurface arenas
and ANE DMA prefetch**.

**Classification: execution-boundary on all four criteria.**
- `mlx_rs::Array`, `Arena` (IOSurface), mmap-backed `MappedSegment`
  (criterion 1: hardware handles; criterion 4: raw FFI to a hardware surface)
- `unsafe { ... }` in `SegmentSlice::data_ptr`, `load_tensor_from_mapped_segment`,
  and `new_external_array` (criterion 2: `unsafe`)
- `thread_local! { static WEIGHT_ARENAS: RefCell<...> }` (criterion 3:
  process-local state)

**Action: stays in engine, no absorption.** The entire file is
gated behind `#[cfg(feature = "mlx-backend")]`, so the constitutional
crates never see it. There is nothing canonical to extract.

### `compute-core/src/ecs/runtime/compilation_systems.rs` (1,038 LOC)

This is the engine's **per-tensor admission pipeline with weight-space
NRMSE screening**. It uses a different set of components
(`SourceWeights(Vec<f32>)`, `CodesData`, `ReconstructedWeights`,
`TensorBinding`, `TensorShape`) and a different `World` model
(`crate::ecs::runtime::world::World`).

**Classification: canonical in isolation, but a parallel authority.**
The file itself does not touch hardware or use `unsafe` — it is a
pipeline over pure data. **But it operates on the engine's own `World`**
with engine-internal component types that the constitutional crates
do not know about. It is not directly absorbable into the
constitutional `stage_systems.rs` because:

1. The component types differ. The constitutional pipeline stores
   `SourceIdentity` + `TensorCatalog` on the world; the engine stores
   `SourceWeights(Vec<f32>)` (raw f32 weight bytes). The
   constitutional `KernelCollection` holds `Vec<KernelArtifact>`; the
   engine's `CodesData` holds the packed quantized bytes.
2. The world model differs. The constitutional pipeline uses
   `prism_ecs_core::world::World`; the engine uses
   `crate::ecs::runtime::world::World`. The two are not interchangeable.
3. The semantic gap is real. The constitutional `system_run_search`
   already runs an evolutionary search with progressive ternary
   admission (a strict superset of the engine's per-tensor NRMSE
   screening). Absorbing the engine's `admit_candidates` would
   duplicate authority.

**Action: stays in engine, no absorption.** The constitutional
`system_run_search` is the canonical successor; the engine file is
the legacy per-tensor weight-space admission pipeline and should be
deleted when the engine's own pipeline is fully cut over to the
constitutional path.

## Tests

Each new sub-module ships with its own `#[cfg(test)] mod tests`. Total
new tests: **17** (32 in the ecs module counting the original ones that
moved with the systems).

Test results:

- `cargo test -p prism-ecs-compile --lib ecs` — **32 passed, 0 failed**
- `cargo test -p prism-ecs-compile --lib` — **364 passed, 0 failed**
- `cargo check -p prism-ecs-compile` — **succeeds** (only the
  pre-existing constitutional-crate warnings)

### Test names (newly added in this decomposition)

The test names describe the invariant, not the function. Examples:

- `compilation_session_round_trips_through_world` — every session
  component survives a `insert_component` / `component_mut` cycle
- `emit_writes_plan_bytes_not_source_bytes` — the constitutional
  invariant that the CImage body is the plan, not the source
- `certify_rejects_plan_mutation_after_emit` — a mutated plan cannot
  pass certification because the plan digest is bound at emit time
- `orchestrator_stage_dispatch_rejects_unmet_prereqs` — out-of-order
  stage calls return errors and the session stays initialized

## Build verification

### `cargo check -p prism-ecs-compile`

```
    Finished `dev` profile [optimized + debuginfo] target(s) in 8.52s
```

No errors. Pre-existing warnings only (5 in `prism-ecs-constitutional`,
1 in `prism-ecs-kernel`).

### `cargo check -p tribunus-compute-core --lib --no-default-features 2>&1 | grep -E "(error|warning).*(ecs|compile_pipeline|profiled_model)"`

No new errors related to `compile_pipeline`, `profiled_model`, or
`compilation_systems`. The only error that surfaces is the pre-existing
`crates/prism-ecs-constitutional/src/world_txn.rs` / `world_txn/mod.rs`
ambiguity from an in-progress partial decomposition of `world_txn` by
a parallel agent — out of scope for this commit.

The 243 pre-existing engine errors are unrelated (they are about
`crate::ecs::compilation::tri_lane`, `crate::ecs::compute_image::*`,
`crate::ecs::CompEntity`, etc. — engine-internal references with
missing definitions).

## Hard rules compliance

- ✅ No `unsafe` in any new file (no new `unsafe` introduced)
- ✅ No `unwrap` / `expect` / `panic!` / `unreachable!` in production
  paths. The pre-existing `.expect("spawn session entity")` and
  `.expect("insert CompilationSession component")` calls in
  `orchestrator.rs` are preserved verbatim from the original godfile
  (line 1566, 1579). They are not new violations; they are existing
  ones that are out of scope for this decomposition.
- ✅ No `anyhow::Error` — every error path uses `CompileError` or
  `String` with explicit construction
- ✅ `BTreeMap` not needed (no canonical collections introduced in
  this decomposition)
- ✅ Newtypes preserved — `SessionHandle(Entity)`,
  `CImagePlanDigest([u8; 32])`, `CompilationReceipt(CompileReceipt)`
  all remain as newtype wrappers

## Out of scope (parallel work)

The following are NOT touched by this commit and remain as
pre-existing in-progress partial decompositions:

- `crates/prism-ecs-constitutional/src/compilation.rs` deleted +
  `compilation/` directory exists but the lib.rs `pub mod compilation;`
  was not updated to point at the directory. **Pre-existing.**
- `crates/prism-ecs-constitutional/src/world_txn.rs` deleted +
  `world_txn/` directory exists with `mod.rs`. **Pre-existing.**
- `crates/prism-ecs-compile/src/evaluator.rs` (1,784 LOC) +
  `evaluator/` directory exists. **Pre-existing.**
- `crates/prism-ecs-runtime/src/kernel.rs` (1,979 LOC) +
  `kernel/` directory exists. **Pre-existing.**
- `crates/prism-ecs-server/src/engine/bpe_tokenizer.rs` (2,256 LOC)
  + `bpe_tokenizer/` directory exists. **Pre-existing.**
- `crates/prism-ecs-server/src/runtime/server.rs` (2,284 LOC) +
  `server/` directory exists. **Pre-existing.**

These are tracked separately per the dispatch order in
`changelogs/2026-07-27-godfile-engine-mapping.md` §"Order of dispatch".

## Files changed

### New files (11)

- `crates/prism-ecs-compile/src/ecs/mod.rs`
- `crates/prism-ecs-compile/src/ecs/components.rs`
- `crates/prism-ecs-compile/src/ecs/resources.rs`
- `crates/prism-ecs-compile/src/ecs/orchestrator.rs`
- `crates/prism-ecs-compile/src/ecs/stages/mod.rs`
- `crates/prism-ecs-compile/src/ecs/stages/ingest.rs`
- `crates/prism-ecs-compile/src/ecs/stages/search_legalize.rs`
- `crates/prism-ecs-compile/src/ecs/stages/kernel.rs`
- `crates/prism-ecs-compile/src/ecs/stages/emit.rs`
- `crates/prism-ecs-compile/src/ecs/stages/certify.rs`
- `crates/prism-ecs-compile/src/ecs/stages/receipt.rs`

### Deleted (1)

- `crates/prism-ecs-compile/src/ecs.rs` (the original 2,581 LOC godfile)

### No changes to other files

- `crates/prism-ecs-compile/src/lib.rs` — unchanged. The existing
  `pub mod ecs;` and `pub use ecs::{...}` re-exports still work
  because the new `ecs/mod.rs` re-exports the same names.
- `crates/prism-ecs-compile/src/compilation_systems.rs` — unchanged.
  Still imports from `crate::ecs::{...}` which now resolves through
  the new `ecs/mod.rs` re-exports.
- `crates/prism-ecs-compile/src/compiler.rs` — unchanged.
- `crates/prism-ecs-compile/src/plan_apply.rs` — unchanged.
