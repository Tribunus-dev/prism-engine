# Goal: Move `compute-core/src/ecs/config/` → `prism-ecs-constitutional::config`

**Date:** 2026-07-28 (Pacific)
**Status:** Goal declared; agent dispatched (batch 6, agent 3).

## Source

`compute-core/src/ecs/config/` — 6 files, 2,583 LOC. The engine's
"config" types: TextArchitecture, VisionArchitecture, LayerPlan,
ModelExecutionPlan, CompileQuantMode, GenerationRegime, EpiloguePlan,
AttentionKind, HardwareTarget, operation_route, PackedLinearShapes,
ManifestModality, network, limits, hardware, parser.

These are PRODUCT-SHAPE types — they describe a model's structure, a
hardware target, an operation route. The constitutional home is
`prism-ecs-constitutional::config` because the constitutional surface
needs to know about these (e.g., dispatch needs HardwareTarget,
lifecycles need ModelExecutionPlan).

## Constitutional target

`crates/prism-ecs-constitutional/src/config/` — new module in the
existing constitutional crate. 25 imports across `legacy_*/` files
reference these types.

## Module doc contract

Each new file in `prism-ecs-constitutional/src/config/` must state
its SINGLE authority in one sentence, e.g.:

```rust
//! Product-shape configuration: architecture, layer plan, hardware
//! target, operation route. Authority: the configuration parser and
//! compiler-input shape.
```

## Approach (E-0..E-N+2)

- E-0: Add `prism-ecs-constitutional` dep to `compute-core/Cargo.toml` (may already be present)
- E-1: Create constitutional surface at `crates/prism-ecs-constitutional/src/config/{mod.rs,architecture,layer_plan,model_execution_plan,compile_quant_mode,hardware_target,operation_route,network,limits,parser}.rs` — re-implement the types. Single authority per file.
- E-2..E-{N-1}: Migrate the 25 `legacy_*/` import sites AND any non-legacy engine imports of `crate::ecs::config::*` to `prism_ecs_constitutional::config::*`.
- E-N: Add architecture safety net at `crates/architecture/src/workspace_legacy_config_imports.rs` that asserts no `use crate::ecs::config::` remains in non-legacy files. Wire into `crates/architecture/src/lib.rs`.
- E-N+1: Either `git rm` the engine's `config/` dir or rename to `compute-core/src/ecs/legacy_config/`. The rename pattern is preferred if any engine-coupled files remain.
- E-N+2: Mark goal achieved in this changelog + commit.

## Isolate to your own worktree

Create an isolated worktree at
`/Users/user/Developer/GitHub/prism-engine-config-move` on branch
`migrate/config-to-constitutional`.

## Critical rules (constitutional, non-negotiable)

- "**No `unsafe` in constitutional, runtime, server, or protocol crates.**"
- "**No `unwrap`/`expect`/`panic!` in production paths.**" Use `?` or `match`. Waivers documented with `// WAIVER:`.
- "**No `anyhow::Error` in `prism-ecs-constitutional`.**" Use per-crate error enums with `thiserror` derives.
- "**No `HashMap`/`HashSet` for canonical collections whose order is observable.**" Use `BTreeMap`, `IndexMap`, `BTreeSet`.
- "**No `String`, `u64`, `Uuid` in constitutional APIs where the value is authority-bearing.**" Newtype them.
- "**Every new `.rs` file states a single authority in its module doc, in one sentence.**"
- "**A constitutional change that does not propagate is not a change.**" Name the propagation chain.
- Constitutional-side tests must pass: `cargo test -p prism-ecs-constitutional --lib`.
- Engine pre-existing error count must be unchanged or decreased.
- Architecture safety net test must pass: `cargo test -p prism-architecture --lib`.

## Conflict awareness

The audit doc says: `config/`, `canonical/`, `ane/` all move in the same
batch to `prism-ecs-constitutional` / `prism-ecs-compile`. The
`crates/architecture/src/lib.rs` will have conflicts at merge time
(multiple agents adding new module declarations). The merge order at
cron time: this agent (config) merges third or fourth, after the others
have settled. The "take HEAD + add new module" pattern applies.

## Success criteria

- All 6 files of `compute-core/src/ecs/config/` moved to `prism-ecs-constitutional/src/config/`
- 25 legacy_*/ import sites retargeted to `prism_ecs_constitutional::config::*`
- `workspace_contains_no_legacy_config_imports` architecture test passes
- `rg "use crate::ecs::config::" compute-core/src/ | grep -v "/legacy_/"` returns no results
- `cargo test -p prism-ecs-constitutional --lib` passes
- Engine pre-existing error count ≤ 192
