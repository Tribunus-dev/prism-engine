# Goal: Move `compute-core/src/ecs/canonical/` → `prism-ecs-constitutional::canonical`

**Date:** 2026-07-28 (Pacific)
**Status:** Goal declared; agent dispatched (batch 6, agent 2).

## Source

`compute-core/src/ecs/canonical/` — 10 files, 1,334 LOC. The engine's
"canonical" types: identity (GenerationId, CandidateId, EngramArtifactId,
HardwareProfileId, ModelSourceId, CompilerIdentity, etc.), generation
(CimageGeneration, ReceiptId), kernel_abi (KernelAbi, CompiledKernelArtifact,
DispatchGeometryPolicy, KernelSemanticId), execution_graph
(ExecutionGraph, RegionId, ExecutionLane, etc.), provenance, representation,
compile_plan, model_ir, receipt_store.

## Constitutional target

`crates/prism-ecs-constitutional/src/canonical/` — new module in the
existing constitutional crate. This is the highest-leverage move because
28 imports across `legacy_*/` files reference these types, and after
they're constitutional, those 28 import sites can be retargeted to
`prism_ecs_constitutional::canonical::*`.

## Module doc contract

Each new file in `prism-ecs-constitutional/src/canonical/` must state
its SINGLE authority in one sentence, e.g.:

```rust
//! Canonical identity primitives — generation, candidate, engram,
//! hardware, model, compiler. Authority: the type system.
```

## Approach (E-0..E-N+2)

- E-0: Add `prism-ecs-constitutional` dep to `compute-core/Cargo.toml` (may already be present)
- E-1: Create constitutional surface at `crates/prism-ecs-constitutional/src/canonical/{mod.rs,identity,generation,kernel_abi,execution_graph,provenance,representation,compile_plan,model_ir,receipt_store}.rs` — re-implement the types. Single authority per file.
- E-2..E-{N-1}: Migrate the 28 `legacy_*/` import sites AND any non-legacy engine imports of `crate::ecs::canonical::*` to `prism_ecs_constitutional::canonical::*`.
- E-N: Add architecture safety net at `crates/architecture/src/workspace_legacy_canonical_imports.rs` that asserts no `use crate::ecs::canonical::` remains in non-legacy files. Wire into `crates/architecture/src/lib.rs`.
- E-N+1: Either `git rm` the engine's `canonical/` dir or rename to `compute-core/src/ecs/legacy_canonical/`. The rename pattern is preferred if any engine-coupled files remain.
- E-N+2: Mark goal achieved in this changelog + commit.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-canonical-move` on branch
`migrate/canonical-to-constitutional`.

## Critical rules (constitutional, non-negotiable)

- "**No `unsafe` in constitutional, runtime, server, or protocol crates.**"
- "**No `unwrap`/`expect`/`panic!` in production paths.**" Use `?` or `match`. Waivers documented with `// WAIVER:`.
- "**No `anyhow::Error` in `prism-ecs-constitutional`.**" Use per-crate error enums with `thiserror` derives.
- "**No `HashMap`/`HashSet` for canonical collections whose order is observable.**" Use `BTreeMap`, `IndexMap`, `BTreeSet`.
- "**No `String`, `u64`, `Uuid` in constitutional APIs where the value is authority-bearing.**" Newtype them.
- "**Every new `.rs` file states a single authority in its module doc, in one sentence.**"
- "**A constitutional change that does not propagate is not a change.**" Name the propagation chain: durable event → event store → replay applier → projection rebuild → read path → consumer.
- Constitutional-side tests must pass: `cargo test -p prism-ecs-constitutional --lib`.
- Engine pre-existing error count must be unchanged or decreased. Run `cargo check -p tribunus-compute-core --lib 2>&1 | tail -3` to verify.
- Architecture safety net test must pass: `cargo test -p prism-architecture --lib`.

## CRITICAL: prism-metal-runtime is a co-caller

`crates/prism-metal-runtime/src/pso_cache.rs` and `fusion_lowering.rs` import
`tribunus_compute_core::ecs::canonical::{kernel_abi, execution_graph}`. These
MUST be migrated simultaneously to `prism_ecs_constitutional::canonical::*` in
the same E-2..E-{N-1} commits. Otherwise the merge will break prism-metal-runtime.

## Success criteria

- All 10 files of `compute-core/src/ecs/canonical/` moved to `prism-ecs-constitutional/src/canonical/`
- 28 legacy_*/ import sites retargeted to `prism_ecs_constitutional::canonical::*`
- `crates/prism-metal-runtime/src/pso_cache.rs` and `fusion_lowering.rs` migrated
- `workspace_contains_no_legacy_canonical_imports` architecture test passes
- `rg "use crate::ecs::canonical::" compute-core/src/ | grep -v "/legacy_/"` returns no results
- `rg "use tribunus_compute_core::ecs::canonical::" crates/` returns no results (except in legacy_*/)
- `cargo test -p prism-ecs-constitutional --lib` passes
- Engine pre-existing error count ≤ 192
