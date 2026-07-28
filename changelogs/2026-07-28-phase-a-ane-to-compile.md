# Goal: Move `compute-core/src/ecs/ane/` → `prism-ecs-compile::ane`

**Date:** 2026-07-28 (Pacific)
**Status:** Goal declared; agent dispatched (batch 6, agent 4).

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
