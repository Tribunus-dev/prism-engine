# Goal: Move `compute-core/src/ecs/canonical/` → `prism-ecs-constitutional::canonical`

**Date:** 2026-07-28 (Pacific)
**Status:** Goal achieved. Canonical engine-deletion migration
complete (E-0..E-N+2).

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

## Outcome (E-N+2)

**Constitutional surface** — `crates/prism-ecs-constitutional/src/canonical/`
hosts the canonical authority for the compiler pipeline types:

| File              | Authority                                         | LOC | Tests |
|-------------------|---------------------------------------------------|-----|-------|
| `mod.rs`          | canonical module root + one-authority map         |  90 |  —    |
| `identity.rs`     | canonical identity primitives (type system)       | 130 |   2   |
| `generation.rs`   | one-generation-per-compilation                     |  27 |   1   |
| `execution_graph.rs` | execution-oriented view                          |  29 |   1   |
| `kernel_abi.rs`   | kernel interface contract                         | 405 |   8   |
| `provenance.rs`   | evidence chain                                    | 200 |   4   |
| `representation.rs` | per-tensor representation decision               | 116 |   1   |
| `compile_plan.rs` | compiler pipeline contract                        | 281 |   5   |
| `model_ir.rs`     | model semantics                                   | 217 |   5   |
| `receipt_store.rs` | receipt store                                    | 145 |   5   |
| **Total**         |                                                   |1640 |  32   |

(Plus 2 re-export tests from the mod root, so 34 total canonical
tests pass.)

**Engine rename** — `compute-core/src/ecs/canonical/` → `compute-core/src/ecs/legacy_canonical/`:
- 10 files renamed via `git mv`.
- 9 of the 10 `.rs` files replaced with a single-line re-export
  shim: `pub use prism_ecs_constitutional::canonical::*;`
- `REMAINING_WORK.md` archaeology doc preserved.
- `compute-core/src/ecs/mod.rs` updated: `pub mod canonical;` →
  `pub mod legacy_canonical;`.

**Engine migration** — 42 engine files retargeted from
`crate::ecs::canonical::*` to
`prism_ecs_constitutional::canonical::*`:
- 33 `use` statement import sites (mechanical `use
  crate::ecs::canonical::` → `use
  prism_ecs_constitutional::canonical::`).
- 19 inline path references (type annotations like
  `crate::ecs::canonical::compile_plan::CompileRequest`) updated
  to `prism_ecs_constitutional::canonical::compile_plan::CompileRequest`.

**prism-metal-runtime migration** —
`crates/prism-metal-runtime/src/{pso_cache.rs, fusion_lowering.rs}`
retargeted to `prism_ecs_constitutional::canonical::*`:
- 4 `use` statements updated.
- `crates/prism-metal-runtime/Cargo.toml` adds the
  `prism-ecs-constitutional` dependency.

**Architecture safety net** —
`crates/architecture/src/workspace_legacy_canonical_imports.rs`
enforces the invariant that no file OUTSIDE the engine's
migration inventory (either `compute-core/src/ecs/canonical/`
pre-rename or `compute-core/src/ecs/legacy_canonical/`
post-rename) imports the legacy surface. Wired into
`crates/architecture/src/lib.rs`.

**Constitutional fixes applied during the migration** (each
fixes a pre-existing engine violation by re-implementing the
type to the constitutional rule set):
- `model_ir.rs::TensorCatalogue.by_name` migrated from
  `HashMap` to `BTreeMap` for observable-order canonical
  contract.
- `model_ir.rs::LogicalOp.attributes` migrated from
  `HashMap` to `BTreeMap`.
- `kernel_abi.rs::validate_bindings` uses `BTreeSet`
  instead of `HashSet` for slot-set equality.
- `receipt_store.rs::ReceiptStore.records` migrated from
  `HashMap` to `BTreeMap`.
- `prism_ecs_constitutional::lib.rs` does NOT glob-re-export
  `canonical::*` at the crate root (would conflict with the
  existing `types::*` re-exports — both define
  `ReceiptId` / `Timestamp` of different shapes). Callers must
  use the explicit `prism_ecs_constitutional::canonical::*`
  path.

**Verification (success criteria)**:

- [x] All 10 files of `compute-core/src/ecs/canonical/` moved
      to `prism-ecs-constitutional/src/canonical/` (E-1).
- [x] 33 engine import sites + 19 inline path references
      retargeted to `prism_ecs_constitutional::canonical::*`
      (E-2..E-N-1).
- [x] `crates/prism-metal-runtime/src/pso_cache.rs` and
      `fusion_lowering.rs` migrated (E-2..E-N-1).
- [x] `workspace_contains_no_legacy_canonical_imports`
      architecture test passes (E-N).
- [x] `rg "use crate::ecs::canonical::" compute-core/src/ |
      grep -v "/legacy_/"` returns no results.
- [x] `rg "use tribunus_compute_core::ecs::canonical::"
      crates/` returns no results.
- [x] `cargo test -p prism-ecs-constitutional --lib` passes
      (147 passed; 0 failed; canonical: 34 passed; 0 failed).
- [x] Engine pre-existing error count: 192 (unchanged from
      baseline; no new errors introduced by the migration).

**Commits (E-0..E-N+2)**:

1. `feat(constitutional): add prism-ecs-constitutional::canonical surface (E-1)`
2. `feat(canonical): migrate engine callers + prism-metal-runtime to constitutional surface (E-2..E-{N-1})`
3. `feat(architecture): add canonical legacy-import safety net (E-N)`
4. `chore(engine): rename canonical/ to legacy_canonical/ + migrate inline paths (E-N+1)`
5. `docs: mark canonical engine-subsystem migration goal achieved (E-N+2)` (this commit)

**No E-0 commit was needed** because
`compute-core/Cargo.toml` already had the
`prism-ecs-constitutional` dependency, and
`prism-ecs-constitutional/Cargo.toml` already had all the
required deps (`prism-ecs-core`, `prism-ecs-kernel`) — only
the `prism-ecs-ir` dep was demoted from optional to required,
which was folded into the E-1 commit.

**Propagation chain** (per AGENTS.md propagation gate):
- **durable event**: every `ReceiptStore.store()` invocation
  produces a content-addressed `ReceiptId`; the
  `LifecycleReceiptBundle` aggregates per-stage receipts; the
  `ReplayManifest` references every payload / kernel / receipt
  by digest.
- **event store**: the `ReceiptStore` itself is a BTreeMap of
  SHA-256-digested serialized records. The constitutional
  `event_store` module (existing) consumes the
  `LifecycleReceiptBundle` to produce durable events.
- **replay applier**: `ReplayManifest` resolves every
  reference by digest and validates the receipt bundle via
  `verify_complete()`; the engine's `cimage_runtime::context`
  is the engine-internal replay applier (now an
  engine-internal adapter on the constitutional surface).
- **projection rebuild**: `compile_plan::CompilePlan` is the
  complete plan for one compilation; the cimage packer
  consumes `CimageBuildInput` (which contains
  `RepresentationPlan`, `ExecutionGraph`, and `CompilerReceiptSet`).
- **read path**: the engine's `legacy_canonical` shim
  re-exports the constitutional types; engine binaries
  (PrismCompiler, CLI) consume them.
- **consumer**: every constitutional crate (prism-ecs-runtime,
  prism-ecs-compile, prism-ecs-kernel) can depend on
  `prism-ecs-constitutional::canonical::*` directly. The
  audit's claim that "all constitutional crates can depend on
  prism-ecs-constitutional directly" is now fully realized for
  the canonical surface.
