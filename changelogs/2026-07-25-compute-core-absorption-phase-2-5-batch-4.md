# Compute-core.legacy Absorption — Phase 2.5 Batch 4: Already-Ported Verification

**Date:** 2026-07-26
**Agent:** Batch 4 verifier
**Status:** No-op. All 6 files in this batch were already ported in commit `e633567e`
**CAMPAIGN.md status:** `system/` subsystem at Canonical (post Phase 2.5 rest)

## TL;DR

The task brief asked for 12 direct mutations to be ported across 6 files
in `compute-core/src/ecs/system/`. **All 12 mutations were already ported
in commit `e633567e`** (Phase 2.5 — port 100 remaining system/ mutations to
WorldTxn, completed earlier today by a parallel batch agent). A fresh
`grep -nE 'world\.(spawn|add_component|remove_component|get_component_mut|insert)\s*\('`
over the 6 files returns **zero non-comment matches**. No code changes are
warranted; this changelog documents the verification and the reason the
port was done with `ConstitutionalWorldTxn` (not the engine-local
`WorldTxn`).

## Scope of the brief

| File | Mutations claimed | Mutations actually remaining |
|---|---:|---:|
| `phase_engine_init.rs` | 2 | 0 |
| `metal_init.rs` | 2 | 0 |
| `executor_systems.rs` | 2 | 0 |
| `execution_graph.rs` | 2 | 0 |
| `draft_model.rs` | 2 | 0 |
| `download.rs` | 2 | 0 |
| **Total** | **12** | **0** |

The brief's mutation counts (2 per file) appear to understate the
previous batch's per-file breakdown:

- `phase_engine_init.rs` — 1 spawn + 2 inserts (3 prod) — see
  `changelogs/2026-07-25-compute-core-absorption-phase-2.5-system-rest.md`
- `metal_init.rs` — 1 spawn + 2 inserts (3 prod)
- `executor_systems.rs` — 4 `ExecutorState` mutations + 1 `ExecutorStep`
  insert per Decode tick (7 prod)
- `execution_graph.rs` — 1 spawn-or-insert (3 prod)
- `draft_model.rs` — 1 spawn-or-insert (3 prod)
- `download.rs` — 1 `DownloadedSourceComp` insert + 1
  `HfDownloadComp` insert (2 prod)

The brief's "2 per file" count may be counting (1) the spawn-or-insert
path block + (1) the commit block, or (2) the entry + commit points. The
actual mutation count is higher. All mutations are routed through
`ConstitutionalWorldTxn`.

## Why `ConstitutionalWorldTxn` and not the engine-local `WorldTxn`

The brief asked to use the **engine-local** `WorldTxn` from
`compute-core/src/ecs/runtime/world_txn.rs` (added in commit `ebcaf2bc`).
That `WorldTxn` operates on the engine-local
`compute-core/src/ecs/runtime/world::World` — a simpler entity/component
store **without** `EntityKind` / name.

The 6 system files in this batch all receive `&mut prism_ecs_core::World`
through their `CompilerSystem::run(&self, world: &mut World)` signature
(re-exported as `crate::ecs::World`). This is the **constitutional**
`World` that carries `EntityKind` and an optional name, with an
`add_component` / `remove_component` API. It is a different type from
the engine-local one. The two `World`s are not interchangeable:

| Property | engine-local `World` | constitutional `World` |
|---|---|---|
| Source | `compute-core/src/ecs/runtime/world.rs` | `prism_ecs_core::World` |
| `EntityKind` | n/a | yes (Kernel, Tensor, Model, Session, Executable, …) |
| Named entities | n/a | yes (via `name(entity)`) |
| `add_component` API | `world.insert(entity, comp)` | `world.add_component(entity, comp)` |
| Components | `runtime::world::Component` | `prism_ecs_core::Component` |
| `WorldTxn` type | `WorldTxn` (in `world_txn.rs`) | `ConstitutionalWorldTxn` (in `constitutional_world_txn.rs`) |

The engine-local `WorldTxn::commit` takes `&mut World` of the engine-local
type. Calling it with the constitutional `&mut World` would be a type
mismatch. The constitutional `WorldTxn` in
`crates/prism-ecs-constitutional/src/world_txn.rs` only exposes
`put_durable` / `put_transient` publicly, both of which require
`DurableComponent` / `TransientComponent` trait impls. The system files'
components only implement `prism_ecs_core::Component` (the legacy
pattern that the engine's `CompilerSystem`s rely on).

**Resolution from commit `e633567e`:** add a parallel
`ConstitutionalWorldTxn` in
`compute-core/src/ecs/runtime/constitutional_world_txn.rs` (~470 LOC,
additive) that mirrors the engine-local `WorldTxn` pattern (staged
spawns, staged inserts on `PendingToken` or existing entity, staged
removes) but targets the constitutional `World` and accepts any
`prism_ecs_core::Component` (no classification required).

So the 6 files in this batch correctly use `ConstitutionalWorldTxn`. They
are the engine's own helper module, scoped to the engine's runtime
directory and added in commit `e633567e`. The brief's wording ("engine's
`world_txn.rs` is the engine's own copy") is satisfied by
`constitutional_world_txn.rs` rather than `world_txn.rs` because the
former is the engine's *adapter* for the constitutional `World` it
threads through `CompilerSystem::run`.

## Current call shape in each file

The staged-mutation pattern in each file is:

```rust
use crate::ecs::runtime::constitutional_world_txn::ConstitutionalWorldTxn;

let mut txn = ConstitutionalWorldTxn::new();
let token = txn.stage_spawn(EntityKind::X, Some("name".into()));
if let Err(e) = txn.stage_insert_on(token, ComponentA { ... }) { tracing::warn!(...); }
if let Err(e) = txn.stage_insert_on(token, ComponentB { ... }) { tracing::warn!(...); }
let _ = txn.commit(world).map_err(|e| {
    tracing::error!(error = %e, "...");
    anyhow::anyhow!("...")
})?;
```

or the spawn-or-insert shape used by `execution_graph.rs`, `draft_model.rs`:

```rust
let model_entities = world.entities_of_kind(EntityKind::Model);
let mut txn = ConstitutionalWorldTxn::new();
if let Some(entity) = model_entities.first() {
    if let Err(e) = txn.stage_insert(*entity, ComponentX(value)) { ... }
} else {
    let token = txn.stage_spawn(EntityKind::Model, Some("name".into()));
    if let Err(e) = txn.stage_insert_on(token, ComponentX(value)) { ... }
}
let _ = txn.commit(world).map_err(|e| { ... })?;
```

`executor_systems.rs` uses the extract-mutate-insert pattern for the
`ExecutorState` component (read via `get_component` — immutable borrow,
clone, mutate the local copy, stage as an insert):

```rust
let next_insert: Option<ExecutorState> =
    if let Some(state) = world.get_component::<ExecutorState>(*entity) {
        let mut projected = state.clone();
        projected.step_counter += 1;
        if projected.step_counter >= projected.max_steps {
            projected.stage = ExecutorStage::Draining;
        }
        Some(projected)
    } else { None };
if let Some(state) = next_insert {
    if let Err(e) = txn.stage_insert(*entity, state) { ... }
}
```

This is the only way to port `world.get_component_mut(...)` through the
constitutional `WorldTxn` without adding trait classifications to ~30
components. Cost: one `Clone` per mutation. Documented in the previous
batch's changelog.

`download.rs` performs filesystem / network side effects
(`download_hf_model`) **before** staging the result component insert.
The side effect is intentionally NOT routed through the `WorldTxn` — the
`WorldTxn` is the canonical authority seam for ECS state, not for
external resources. This matches the documented pattern from the
Phase 2.5 changelog (archive, tts, download, backend_compile all do
this).

## Build status

```text
cargo check -p tribunus-compute-core --lib --no-default-features
```

emits the 242 pre-existing engine errors (none introduced by this batch;
none in any of the 6 files). The 6 files in this batch compile without
errors in the pre-existing-error envelope. The workspace is unchanged
from `e633567e` for these 6 files.

## Action taken

**None.** All 12 mutations in the brief are already ported. The brief
appears to have been written from a snapshot predating commit
`e633567e` (which itself completed earlier today, 2026-07-26
09:57:20). No code changes are made in this batch.

A parent session re-evaluation is recommended: either
(a) drop batch 4 from the run list (work is already done), or
(b) redirect this batch to the ~2 system/ files that still have
non-comment direct mutations in production code paths (the previous
changelog noted 0 production mutations remain, but a fresh sweep may
uncover any the previous batch missed — `kernel_catalog.rs` and
`catalog_validation.rs` are the only files whose grep output
contained non-comment matches in the pre-batch sweep, and both are
`add_component` calls inside `#[cfg(test)]` legacy blocks; these are
intentionally left as opt-in legacy escape hatches).

## Cross-references

- Previous batch commit: `e633567e` (Phase 2.5 — port 100 remaining
  system/ mutations to WorldTxn, 44 files, 100/100 production
  mutations ported)
- Previous batch changelog:
  `changelogs/2026-07-25-compute-core-absorption-phase-2.5-system-rest.md`
- Engine-local `WorldTxn`: `compute-core/src/ecs/runtime/world_txn.rs`
  (commit `ebcaf2bc`)
- Constitutional `WorldTxn` helper: `compute-core/src/ecs/runtime/constitutional_world_txn.rs`
  (commit `e633567e`)
- CAMPAIGN.md: `system/` subsystem at Canonical (post Phase 2.5 rest)
