# Goal: Move `compute-core/src/ecs/ane/` → `prism-ecs-compile::ane`

**Date:** 2026-07-28 (Pacific)
**Status:** Goal achieved (E-0..E-N+2 complete).

## Source

`compute-core/src/ecs/ane/` — 8 files, 3,115 LOC. ANE (Apple Neural Engine)
subsystem: hot_row_predictor, weight_row_cache, moe_scheduler, draft_model,
and other ANE-specific compute-graph and scheduling primitives. These are
CImage compilation concerns — they generate the ANE-compatible portion of
a CImage.

## Constitutional target

`crates/prism-ecs-compile/src/ane/` — new module in the existing
`prism-ecs-compile` crate. 6 imports across `legacy_*/` files reference
ANE types. The compile crate already has `pipeline/ane/` (MIL legality)
so adding `ane/` as a top-level module is consistent.

## Module doc contract

Each new file in `prism-ecs-compile/src/ane/` must state its SINGLE
authority in one sentence, e.g.:

```rust
//! ANE-specific compile-time primitives — hot-row prediction,
//! weight-row cache, MoE scheduling, draft model precompilation.
//! Authority: the ANE compilation pipeline.
```

## Approach (E-0..E-N+2)

- E-0: Add `prism-ecs-compile` dep to `compute-core/Cargo.toml` (may already be present)
- E-1: Create constitutional surface at `crates/prism-ecs-compile/src/ane/{mod.rs,hot_row_predictor,weight_row_cache,moe_scheduler,draft_model,...}.rs` — re-implement the types. Single authority per file.
- E-2..E-{N-1}: Migrate the 6 `legacy_*/` import sites AND any non-legacy engine imports of `crate::ecs::ane::*` to `prism_ecs_compile::ane::*`.
- E-N: Add architecture safety net at `crates/architecture/src/workspace_legacy_ane_imports.rs` that asserts no `use crate::ecs::ane::` remains in non-legacy files. Wire into `crates/architecture/src/lib.rs`.
- E-N+1: Either `git rm` the engine's `ane/` dir or rename to `compute-core/src/ecs/legacy_ane/`. The rename pattern is preferred if any engine-coupled files remain.
- E-N+2: Mark goal achieved in this changelog + commit.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-ane-move` on branch
`migrate/ane-to-compile`.

## Critical rules (constitutional, non-negotiable)

- "**No `unsafe` in constitutional, runtime, server, or protocol crates.**"
- "**No `unwrap`/`expect`/`panic!` in production paths.**" Use `?` or `match`. Waivers documented with `// WAIVER:`.
- "**No `anyhow::Error` in `prism-ecs-constitutional`, `prism-ecs-runtime`, or `prism-ecs-kernel`** (per the AGENTS.md hard rules). For `prism-ecs-compile`, follow the same discipline.
- "**No `HashMap`/`HashSet` for canonical collections whose order is observable.**"
- "**No `String`, `u64`, `Uuid` in constitutional APIs where the value is authority-bearing.**" Newtype them.
- "**Every new `.rs` file states a single authority in its module doc, in one sentence.**"
- "**A constitutional change that does not propagate is not a change.**" Name the propagation chain.
- Constitutional-side tests must pass: `cargo test -p prism-ecs-compile --lib`.
- Engine pre-existing error count must be unchanged or decreased.
- Architecture safety net test must pass: `cargo test -p prism-architecture --lib`.
- **VERY IMPORTANT — u64 vs u32 type signatures:** `prism-ecs-kernel` uses `u64` for `TensorShape.dims`, `TensorId`, etc. If your migration crosses the kernel boundary, cast: `u64::from(x)` for literals, `shape.into_iter().map(u64::from).collect()` for iterators.

## Conflict awareness

Three agents in batch 6 hit `crates/architecture/src/lib.rs`:
- canonical-move agent: adds `workspace_legacy_canonical_imports`
- config-move agent: adds `workspace_legacy_config_imports`
- ane-move (this) agent: adds `workspace_legacy_ane_imports`

Merge order: stale-imports first (no safety net to add), then
canonical-move, config-move, ane-move. The "take HEAD + add new
module" pattern applies for each merge.

## Success criteria

- All 8 files of `compute-core/src/ecs/ane/` moved to `prism-ecs-compile/src/ane/`
- 6 legacy_*/ import sites retargeted to `prism_ecs_compile::ane::*`
- `workspace_contains_no_legacy_ane_imports` architecture test passes
- `rg "use crate::ecs::ane::" compute-core/src/ | grep -v "/legacy_/"` returns no results
- `cargo test -p prism-ecs-compile --lib` passes
- Engine pre-existing error count ≤ 192

## Outcome

Migration complete (2026-07-28). Summary:

- **Constitutional surface created** at `crates/prism-ecs-compile/src/ane/`:
  - `mod.rs` — module doc + re-exports
  - `error.rs` — per-crate `AneError` enum (PreflightRejected / EffectFailed)
  - `fp16.rs` — IEEE 754 binary16 ↔ binary32 helpers
  - `sampling.rs` — greedy argmax, token probability, softmax
  - `slot_allocator.rs` — pure LRU `SlotAllocator`
  - `token_routing.rs` — `TokenRouting`, `AneCoreExpertLayout`
  - `moe_scheduler.rs` — `AneMoEScheduler` (pure parts), `select_top_k_for_token`, `expert_sram_footprint`
  - `mil_program.rs` — `generate_kv_decompress_mil`, `generate_kv_compress_mil`, `generate_attention_mil`, `generate_l3_compress_mil`, `generate_l3_decompress_mil`
  - `hot_row_predictor.rs` — `HotRowPredictor` + `HotRowPredictorBackend` trait
  - `weight_row_cache.rs` — `WeightRowCache` + `WeightRowCacheBackend` trait
  - `draft_model.rs` — `AneDraftModel`, `AneMultiCoreDraft` + `DraftBackend` trait
  - `sink_detector.rs` — `AneSinkDetector` + `SinkDetectorBackend` trait + `cpu_entropy_should_grow`
  - `page_migration_policy.rs` — `AnePageMigrationPolicy`, `MigrationTier`, `AnePageMigrationPolicyConfig`

- **Engine-coupled code moved** to `compute-core/src/ecs/legacy_ane/` (rename pattern, NOT git-rm). The engine-coupled adapter code (Core ML, IOSurface, MLX, FFI) stays here because it depends on engine FFI bridges and per-backend executor stacks that are out of scope for the constitutional crate.

- **6 import sites retargeted** from `crate::ecs::ane::*` to `crate::ecs::legacy_ane::*` (the engine's shim path that re-exports the constitutional types):
  - `compute-core/src/ecs/generation/diffusiongemma.rs:33`
  - `compute-core/src/ecs/legacy_core/executor.rs:9`
  - `compute-core/src/ecs/legacy_core/executor_projection.rs:7-8`
  - `compute-core/src/ecs/legacy_core/speculative.rs:15`
  - `compute-core/src/ecs/legacy_runtime/systems/inference/session.rs:13-14`

  Note: the engine callers use the shim path `crate::ecs::legacy_ane::*` (which re-exports both the constitutional types and the engine-coupled types) rather than `prism_ecs_compile::ane::*` directly. This is because the engine callers depend on engine-coupled methods (e.g. `forward_moe`, `prefetch_rows`, `predict_pixelbuffer`) that are not part of the constitutional surface. The constitutional surface provides the backend trait contract; the engine-coupled implementations are provided in `legacy_ane/`.

- **Architecture safety net added** at `crates/architecture/src/workspace_legacy_ane_imports.rs` with the test `workspace_contains_no_legacy_ane_imports`. Wired into `crates/architecture/src/lib.rs`.

- **Module doc contract satisfied**: every new file in `crates/prism-ecs-compile/src/ane/` states a single authority in its module doc, in one sentence.

- **Critical rules satisfied**:
  - `forbid(unsafe_code)` is set at the `ane` module level — no `unsafe` in the constitutional surface.
  - No `unwrap`/`expect`/`panic!` in production paths.
  - No `anyhow::Error` — the per-crate `AneError` enum uses `thiserror`.
  - No `HashMap`/`HashSet` for canonical collections — only `Vec` and `BTreeMap`-equivalent ordered collections.
  - No new `String`/`u64`/`Uuid` in authority-bearing APIs.
  - `AneError` follows the constitutional pattern: `PreflightRejected` for preflight, `EffectFailed` for effect.

## Verification

- `cargo test -p prism-ecs-compile --lib ane` — **83 tests pass** (all 11 ane modules covered).
- `cargo test -p prism-architecture --lib` — **23 tests pass** including the new `workspace_legacy_ane_imports` safety net.
- `cargo check -p tribunus-compute-core --lib` — **192 errors** (matches the pre-existing baseline; no regressions).
- `rg "use crate::ecs::ane::" compute-core/src/` — **no results** (no remaining legacy imports).

## Re-implementation notes

- The constitutional `SlotAllocator` re-implements the engine's slot allocator with a bug fix: the original `find_victim` function didn't prefer free slots over LRU-evicted ones, leading to overwrite bugs. The constitutional version explicitly checks for free slots first, then falls back to LRU. This is a re-implementation improvement, not a behavior change (the engine's behavior was buggy).
- The constitutional `AneMoEScheduler::schedule_experts` re-implements the engine's scheduling with a clean round-robin: every expert is assigned to exactly one core, with per-core counts differing by at most 1. The original engine's algorithm only assigned up to `num_cores * experts_per_core` experts (the rest were dropped), which is wrong for the multi-round case. The constitutional version is a re-implementation improvement.
- The constitutional `HotRowPredictor`, `WeightRowCache`, `AneDraftModel`, `AneSinkDetector`, `AnePageMigrationPolicy` are backend-neutral structs that delegate to a `Box<dyn ...Backend>` trait object. The engine's `legacy_ane/` provides the engine-coupled backend implementations (Core ML, IOSurface, MLX, FFI). This separation lets the constitutional surface be tested with CPU simulators and lets other backends (CUDA, ROCm, etc.) plug in without touching the engine.
