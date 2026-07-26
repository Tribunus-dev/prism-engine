# compute-core.legacy → constitutional ECS — Integration Plan (2026-07-25)

**Status:** Plan. Phases 0, 1, 2, 3, 4-A, 4-B, 4-C are complete (or
in flight, see below). Phases 2.5 and 4-B / 4-C continuation are
running in parallel dispatches.

This file is the master integration plan for absorbing the
`compute-core.legacy/` engine (renamed to `compute-core/` in Phase 0)
into the constitutional ECS libraries. Per-file evidence lives in the
phase-specific changelogs referenced below; this file is the
single-page rollup.

## Phase Summary

| Phase | Scope | Status | Commit | Changelog |
|---|---|---|---|---|
| 0 | Rename `compute-core.legacy/` → `compute-core/`; remove 4 shim dirs from `compute-core/src/ecs/mod.rs`; restore 5-agent dispatch work | **DONE** | `ef826363` | (rolled into the changelog below) |
| 1 | Mechanical cleanup of the 4 absorbed shim directories (`constitutional/`, `quantization/`, `kv_cache/`, `inference_profile/`) | **DONE** (rolled into Phase 0 commit) | `ef826363` | (rolled into the changelog below) |
| 2 | `system/` absorption — 10 highest-leverage files (162 direct world mutations → WorldTxn) | **DONE** | `472d9754` | `changelogs/2026-07-25-compute-core-absorption-phase-2-system.md` |
| 2.5 | `system/` mutations in the remaining 42 files (writer & effect boundaries) | **IN PROGRESS** (parallel dispatch) | (TBD) | (TBD) |
| 3 | 10 remaining direct world mutations in `runtime/` (9 sites) + `core/` (1 site) → engine-local `WorldTxn` | **DONE** | `ebcaf2bc` | `changelogs/2026-07-25-compute-core-absorption-phase-3-runtime.md` |
| 3+4A | Part 2: read-only audit of 25+ partially-absorbed subsystems with per-subsystem plans | **DONE** | `c5ad9070` | `changelogs/2026-07-25-compute-core-absorption-phase-3-4a-runtime-audit.md` |
| 4B | `compute_image/` absorption (3 highest-leverage files: `compile/pipeline.rs`, `cimage_packer/pipeline.rs`, `compile/validation_matrix.rs`) | **DONE** (re-implementations committed; originals left in place for follow-up deletion) | `14e8edb1` | `changelogs/2026-07-25-compute-core-absorption-phase-4b-compute-image.md` |
| 4B continuation | More `compute_image/` files (per Phase 4B roadmap) | **IN PROGRESS** (parallel dispatch) | (TBD) | (TBD) |
| 4C | `core/` absorption (3 highest-leverage files: `engine_receipts.rs`, `executor.rs` `SinkState` pattern, `gguf.rs` manifest extraction) | **DONE** (re-implementations committed) | `b7d92c40` | `changelogs/2026-07-25-compute-core-absorption-phase-4c-core.md` |
| 4C continuation | More `core/` files (per Phase 4C roadmap) | **IN PROGRESS** (parallel dispatch) | (TBD) | (TBD) |
| 4C follow-up | `mil_builder` absorb — engine's 2,226-LOC MIL builder merged into `prism-ane` (engine file replaced with 68-LOC re-export shim) | **DONE** | `7cd96e16` | (see commit body) |
| 5 | Update `CAMPAIGN.md`, `AGENTS.md`, `project-absorption.md` to reflect the new state | **DONE** (this document + the doc-update changelog) | (this commit) | `changelogs/2026-07-25-compute-core-absorption-phase-5-docs.md` |

## Per-phase evidence pointers

- **Phase 0+1** — `ef826363` (commit). Removed `pub mod constitutional;`,
  `pub mod quantization;`, `pub mod kv_cache;`, `pub mod inference_profile;`
  from `compute-core/src/ecs/mod.rs`; engine re-exports now point at
  `prism_ecs_constitutional`, `prism_kv_cache`, `prism_ecs_quantization`.
  The engine's `lib.rs` was updated to re-export from the canonical
  crates. Stub crates were created at `mlx-rs-fork/mlx-rs/`,
  `mlx-rs-fork/mlx-sys/` so the engine's optional `mlx-rs`/`mlx-sys`
  deps resolve when the engine is built outside the workspace.
- **Phase 2** — 10 files re-implemented in 6 constitutional crates
  (`prism-ecs-runtime`, `prism-ecs-artifact`, `prism-ecs-kernel`,
  `prism-ecs-compile`, `prism-ecs-constitutional`); 162 direct world
  mutations eliminated; 140 new tests. See
  `changelogs/2026-07-25-compute-core-absorption-phase-2-system.md`.
- **Phase 3** — 10 remaining direct world mutations ported to a new
  engine-local `WorldTxn` (`compute-core/src/ecs/runtime/world_txn.rs`,
  459 LOC, 14 unit tests). See
  `changelogs/2026-07-25-compute-core-absorption-phase-3-runtime.md`.
- **Phase 4A** — Read-only audit of 25+ partially-absorbed subsystems
  with per-subsystem absorption plans. See
  `changelogs/2026-07-25-compute-core-absorption-phase-3-4a-runtime-audit.md`.
- **Phase 4B** — 3 of 162 `compute_image/` files re-implemented
  (`compile/pipeline.rs`, `cimage_packer/pipeline.rs`,
  `compile/validation_matrix.rs`); 50 new tests. Originals left in
  place for coordinated deletion in a follow-up phase. See
  `changelogs/2026-07-25-compute-core-absorption-phase-4b-compute-image.md`.
- **Phase 4C** — 3 of 121 `core/` files re-implemented
  (`engine_receipts.rs`, `executor.rs` `SinkState` pattern,
  `gguf.rs` manifest extraction); 42 new tests. Originals left in
  place for coordinated deletion. See
  `changelogs/2026-07-25-compute-core-absorption-phase-4c-core.md`.
- **Phase 4C follow-up (mil_builder)** — engine's 2,226-LOC
  `mil_builder.rs` (the *superset* of the canonical prism-ane one) was
  absorbed into `prism-ane`; engine file replaced with 68-LOC
  re-export shim. 28/28 prism-ane tests pass (+19 new). See commit
  `7cd96e16`.

## Out of scope (deferred)

- **`engine.rs`** (1,374 LOC, 1 direct world mutation) — full
  re-architecture to `prism-ecs-runtime::engine_orchestrator`. The
  single direct `world.spawn()` is now wrapped in the engine-local
  `WorldTxn` (see Phase 3 mutation 10), but the orchestrator pattern
  itself stays engine-side.
- **`compile/kernel_dispatch.rs`** (2,676 LOC, 19 Metal dispatcher
  structs) — execution-plane state; requires typed ports the
  constitutional libraries do not yet expose.
- **`orchestrator/runner.rs`** (2,436 LOC) — execution-plane state;
  same reason.
- **Remaining `system/`, `core/`, `compute_image/` files** (per the
  roadmaps in the Phase 2, 4B, 4C changelogs) — Phase 2.5 and the
  4B/4C continuations cover the highest-leverage items in parallel
  dispatches; the rest are sequenced behind them.

## Subsystem state after Phase 5

After Phase 5, the subsystem migration state in `CAMPAIGN.md` reflects
the new state:

- `Compilation & Model Production` advances to `Shadow` (the
  re-implementations in `prism-ecs-compile::cimage_pipeline`,
  `cimage_packer`, `cimage_validation` and
  `prism-ecs-compile::{compile_planning, hardware_tuning, fusion_analysis,
  fusion_scheduling, compile_pipeline}` are the constitutional path;
  the engine originals still exist for shadow comparison).
- A new `Reception & Sinks` (or `Attention Sinks`) subsystem entry
  appears for `prism-ecs-runtime::attention_sink`.
- A new `Engine Receipts` subsystem entry appears for
  `prism-ecs-runtime::engine_receipts`.
- The `system/` migration state moves from `Shadow` to a more nuanced
  `Canonical-with-engine-shadow` split.

See `CAMPAIGN.md` Subsystem Registry for the updated state.

## Completion report (rolled up)

### Phase 0+1 — DONE
- Engine renamed from `compute-core.legacy/` to `compute-core/`
  (the engine is now a workspace member of the Prism workspace).
- 4 absorbed shim directories removed from
  `compute-core/src/ecs/mod.rs`: `constitutional/`, `quantization/`,
  `kv_cache/`, `inference_profile/`. The engine's lib.rs re-exports
  now point at the canonical constitutional crates
  (`prism_ecs_constitutional`, `prism_ecs_quantization`,
  `prism_kv_cache`).
- Stub crates for `mlx-rs` / `mlx-sys` created so the engine
  builds out of the workspace.
- Commit: `ef826363`.
- Evidence: the `// <shim> was deleted in Phase 1` comments now in
  `compute-core/src/ecs/mod.rs` (lines 60-62, 104-106, 109, 157-159).

### Phase 2 — DONE
- 10 `system/` files re-implemented in 6 constitutional crates.
- 162 direct world mutations eliminated.
- 140 new tests passing across the 10 new modules.
- Commit: `472d9754`.
- Changelog: `changelogs/2026-07-25-compute-core-absorption-phase-2-system.md`.

### Phase 3 — DONE
- 10 remaining direct world mutations ported to the engine-local
  `WorldTxn` (mirroring the constitutional shape, scoped to the
  engine's runtime `World`).
- New module: `compute-core/src/ecs/runtime/world_txn.rs` (459 LOC,
  14 unit tests).
- Commit: `ebcaf2bc` (code), `c5ad9070` (changelog).
- Changelog: `changelogs/2026-07-25-compute-core-absorption-phase-3-runtime.md`.

### Phase 4A — DONE
- Read-only audit of 25+ partially-absorbed subsystems.
- Per-subsystem absorption plans with current state, Prism-domain
  target, re-implementation plan, effort estimate, and risk.
- Commit: `c5ad9070`.
- Changelog: `changelogs/2026-07-25-compute-core-absorption-phase-3-4a-runtime-audit.md`.

### Phase 4B (initial) — DONE
- 3 of 162 `compute_image/` files re-implemented in
  `prism-ecs-compile::cimage_pipeline/`,
  `prism-ecs-compile::cimage_packer/`, and
  `prism-ecs-compile::cimage_validation/`.
- 50 new tests; 233/233 tests pass in `prism-ecs-compile --lib`.
- Originals left in place for coordinated deletion.
- Commit: `14e8edb1`.
- Changelog: `changelogs/2026-07-25-compute-core-absorption-phase-4b-compute-image.md`.

### Phase 4C (initial) — DONE
- 3 of 121 `core/` files re-implemented:
  - `engine_receipts.rs` → `prism-ecs-runtime::engine_receipts`
  - `executor.rs` (`SinkState` pattern) → `prism-ecs-runtime::attention_sink`
  - `gguf.rs` (manifest extraction) → `prism-gguf::manifest`
- 42 new tests passing.
- Originals left in place for coordinated deletion.
- Commit: `b7d92c40`.
- Changelog: `changelogs/2026-07-25-compute-core-absorption-phase-4c-core.md`.

### Phase 4C follow-up (mil_builder) — DONE
- Engine's 2,226-LOC `mil_builder.rs` absorbed into
  `prism-ane::mil_builder` (the engine was the superset — unique
  methods like `topk`, `batch_size`, `silu`, `softmax`, `conv`,
  `reshape`, `transpose`, `const_i32`, `reserve_names` were merged
  into the prism-ane version; the 2-arg `gather` stub was replaced
  with the full 3-arg implementation).
- New `prism-ane::mil_layer_programs` module (278 LOC) for the
  high-level ANE program constructors.
- Engine file replaced with 68-LOC re-export shim at
  `compute-core/src/ecs/core/mil_builder.rs` so existing callers
  continue to work.
- 28/28 prism-ane tests passing (+19 new).
- Commit: `7cd96e16`.

### Phase 2.5 — IN PROGRESS
- `system/` mutations in the remaining 42 files (writer & effect
  boundaries). Running in a parallel dispatch; result commit and
  changelog TBD.

### Phase 4B continuation — IN PROGRESS
- More `compute_image/` files per the Phase 4B roadmap
  (`emit.rs`, `source.rs`, `quantize.rs`, `ternary.rs`,
  `kernel_dispatch.rs`, `ternary_pipeline.rs`, `tts_compile.rs`,
  `execution_graph.rs`, `gpu_pack.rs`, `int4_pack.rs`,
  `kernel_types.rs`, `kernel_registry.rs`, `capability_registry.rs`,
  `portfolio.rs`, `tensix.rs`, `hip_dispatch.rs`, etc.).
  Running in a parallel dispatch; result commit and changelog TBD.

### Phase 4C continuation — IN PROGRESS
- More `core/` files per the Phase 4C roadmap
  (`executor.rs` `run_prologue` / `run_layer` / `moe_forward` and
  mask helpers, `engine.rs` re-architecture, `worker_protocol.rs`,
  `arena.rs` and `arena_lifecycle.rs`, `ane_bridge.rs`,
  `ane_compile.rs`, `compute_ir.rs`, `compute_lane.rs`,
  `compute_service.rs`, etc.).
  Running in a parallel dispatch; result commit and changelog TBD.

### Phase 5 — DONE (this document)
- `CAMPAIGN.md` Subsystem Registry updated.
- `AGENTS.md` project-layout section updated to reflect the new
  engine path and the new constitutional files.
- `project-absorption.md` "Concrete violations" table updated with
  the absorbed file → re-implementation mappings.
- Doc-update changelog: `changelogs/2026-07-25-compute-core-absorption-phase-5-docs.md`.
- Commit: (this commit).

## Pre-existing build issues (out of scope for the absorption)

Two build issues pre-date the absorption and remain after Phase 5:

- **`prism-metal-runtime` not in workspace.** The crate at
  `crates/prism-metal-runtime/` declares `version.workspace = true` /
  `edition.workspace = true` (it expects to be a workspace member)
  but is not in the root `Cargo.toml` `[workspace] members` list.
  Building it standalone fails with "package believes it's in a
  workspace when it's not." This is pre-existing (the absorption did
  not add or remove this crate from the workspace). Tracked outside
  this plan.
- **`prism-metal-runtime` → `tribunus-compute-core` dependency.**
  `crates/prism-metal-runtime/Cargo.toml` declares
  `tribunus-compute-core = { path = "../../compute-core" }` to the
  engine. The engine builds with pre-existing errors
  (~219) including missing `ComputeRouteProfile`,
  `BoundaryExecutionReceipt`, `CompEntity`, etc. (Phase 3
  changelog baseline). The `prism-metal-runtime` → `compute-core`
  link is therefore broken even when the workspace membership is
  fixed. Tracked outside this plan.

## End state (when all phases complete)

- `compute-core.legacy/` is empty; `compute-core/` is a thin
  orchestrator that depends on the constitutional libraries for all
  authority-bearing state transitions.
- All 121 `core/` files are either absorbed, deleted, or explicitly
  kept engine-side with a documented Prism-domain justification.
- All 162 `compute_image/` files are either absorbed, deleted, or
  out-of-scope (Metal shader sources in `templates/`).
- All 49 remaining `system/` files have been ported to `WorldTxn` or
  removed.
- The constitutional libraries own the full CImage lifecycle
  (`prism-ecs-compile::cimage_pipeline`, `cimage_packer`,
  `cimage_validation`), the full engine receipts surface
  (`prism-ecs-runtime::engine_receipts`), the attention-sink
  pattern (`prism-ecs-runtime::attention_sink`), and the MIL
  builder (`prism-ane::mil_builder`).
